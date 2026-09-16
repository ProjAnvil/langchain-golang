// This file implements the LangSmith tracer (design t18 PR3, section 6): a
// callbacks.Handler that rebuilds a run tree from the flat callback event
// stream (start creates a Run and enqueues a create op; end/error fill
// outputs/error and enqueue an update op; stream events only count) and a
// background batch client that POSTs {"post":[creates],"patch":[updates]} to
// {endpoint}/runs/batch when a batch fills or a timer fires.
//
// Failure discipline: tracing must never fail the business call. HandleEvent
// always returns nil; POST failures (non-2xx or network) are retried once and
// then reported to OnError (default: silent slog.Debug). Flush waits, with a
// 5s cap, for the queue to drain; Close drains and stops the flusher and is
// idempotent. All methods are nil-safe, so callers can hold the result of
// NewLangChainTracer() without checking whether the environment enabled it.
//
// This file deliberately imports only core/callbacks (never core/runnables):
// the StreamEvents driver depends on this package for the env auto-attach
// hook, and that dependency must stay one-directional.
package tracers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
)

// Defaults mirroring the Python client: hosted endpoint, "default" project,
// 64-run batches, 500ms flush cadence, 30s per-request timeout, 100-op queue.
const (
	defaultLangSmithEndpoint = "https://api.smith.langchain.com"
	defaultLangSmithProject  = "default"
	defaultBatchSize         = 64
	defaultFlushEvery        = 500 * time.Millisecond
	defaultQueueCapacity     = 100
	langSmithRequestTimeout  = 30 * time.Second
	langSmithFlushTimeout    = 5 * time.Second
)

// LangSmithOptions configures a LangChainTracer. The zero value is usable
// after NewLangChainTracerWithOptions fills the defaults.
type LangSmithOptions struct {
	// Endpoint is the LangSmith API base URL; posted to as
	// {Endpoint}/runs/batch. Env: LANGSMITH_ENDPOINT / LANGCHAIN_ENDPOINT.
	Endpoint string
	// APIKey authenticates via the X-API-KEY header. Env:
	// LANGSMITH_API_KEY / LANGCHAIN_API_KEY. A non-empty key is required
	// for the env-based constructor to enable tracing.
	APIKey string
	// Project names the LangSmith project (session_name on each run). Env:
	// LANGSMITH_PROJECT / LANGCHAIN_PROJECT (legacy SESSION names).
	Project string
	// HTTPClient overrides the HTTP client used for batch POSTs. When nil a
	// client with a 30s timeout is created.
	HTTPClient *http.Client
	// BatchSize flushes a batch once this many operations are queued
	// (default 64).
	BatchSize int
	// FlushEvery is the periodic flush interval (default 500ms).
	FlushEvery time.Duration
	// OnError observes background failures after the single retry. When nil,
	// failures are logged at slog.Debug and dropped.
	OnError func(error)
}

// EnabledFromEnv reports whether LangSmith tracing should be active: a
// truthy LANGSMITH_TRACING_V2 / LANGCHAIN_TRACING_V2 / LANGSMITH_TRACING /
// LANGCHAIN_TRACING (checked in that order, mirroring the LANGSMITH-before-
// LANGCHAIN namespace precedence of langsmith.utils.get_env_var) plus a
// non-empty API key. Default is off — nothing is ever sent without it.
func EnabledFromEnv() bool {
	return tracingFlagFromEnv() && firstEnvValue("API_KEY") != ""
}

// tracingFlagFromEnv resolves the tracing flag with namespace precedence.
func tracingFlagFromEnv() bool {
	for _, name := range []string{
		"LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2",
		"LANGSMITH_TRACING", "LANGCHAIN_TRACING",
	} {
		if isTruthyEnv(os.Getenv(name)) {
			return true
		}
	}
	return false
}

// firstEnvValue returns the first non-empty value among the LANGSMITH_- and
// LANGCHAIN_-prefixed forms of suffix (LANGSMITH_ wins, as in Python).
func firstEnvValue(suffix string) string {
	for _, prefix := range []string{"LANGSMITH_", "LANGCHAIN_"} {
		if value := strings.TrimSpace(os.Getenv(prefix + suffix)); value != "" {
			return value
		}
	}
	return ""
}

// isTruthyEnv matches langsmith.utils.is_truish: "true" (case-insensitive)
// or "1".
func isTruthyEnv(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "true", "1":
		return true
	}
	return false
}

// langSmithOptionsFromEnv builds options from the environment using the same
// precedence rules as the Python client (LANGSMITH_ before LANGCHAIN_;
// project falls back to the legacy SESSION names; endpoint trailing slashes
// are trimmed).
func langSmithOptionsFromEnv() LangSmithOptions {
	endpoint := strings.TrimRight(firstEnvValue("ENDPOINT"), "/")
	project := firstEnvValue("PROJECT")
	if project == "" {
		project = firstEnvValue("SESSION")
	}
	return LangSmithOptions{
		Endpoint: endpoint,
		APIKey:   firstEnvValue("API_KEY"),
		Project:  project,
	}
}

// NewLangChainTracer returns a tracer configured from the environment, or
// nil when EnabledFromEnv() is false. Every method is nil-safe, so callers
// typically write `tracer := NewLangChainTracer()` and use it unconditionally.
func NewLangChainTracer() *LangChainTracer {
	if !EnabledFromEnv() {
		return nil
	}
	return NewLangChainTracerWithOptions(langSmithOptionsFromEnv())
}

// NewLangChainTracerWithOptions creates a tracer with explicit options
// (defaults filled for empty fields) and starts its background flusher. The
// caller owns the lifetime: Close when done, or Flush to wait for delivery.
func NewLangChainTracerWithOptions(options LangSmithOptions) *LangChainTracer {
	if options.Endpoint == "" {
		options.Endpoint = defaultLangSmithEndpoint
	} else {
		options.Endpoint = strings.TrimRight(options.Endpoint, "/")
	}
	if options.Project == "" {
		options.Project = defaultLangSmithProject
	}
	if options.BatchSize <= 0 {
		options.BatchSize = defaultBatchSize
	}
	if options.FlushEvery <= 0 {
		options.FlushEvery = defaultFlushEvery
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: langSmithRequestTimeout}
	}
	tracer := &LangChainTracer{
		opts:     options,
		runMap:   make(map[string]*Run),
		orderMap: make(map[string]runOrder),
		queue:    make(chan runOperation, defaultQueueCapacity),
		flushNow: make(chan chan struct{}),
		done:     make(chan struct{}),
	}
	go tracer.flushLoop()
	return tracer
}

// runOrder is the (trace_id, dotted_order) pair the Python tracer caches per
// run so late children can still chain onto an ended parent's order.
type runOrder struct {
	traceID     string
	dottedOrder string
}

// runOperation is one queued batch item: a create ("post") or update
// ("patch") carrying an exclusive snapshot of the Run, safe to serialize on
// the flusher goroutine while the tracer keeps mutating the live copy.
type runOperation struct {
	patch bool
	run   *Run
}

// LangChainTracer implements callbacks.Handler by rebuilding a run tree from
// flat events and batching it to the LangSmith ingest API.
type LangChainTracer struct {
	opts     LangSmithOptions
	queue    chan runOperation
	flushNow chan chan struct{}
	done     chan struct{}
	closing  atomic.Bool
	// dropped counts queue-overflow drops since the last aggregated report
	// (see reportDropped); an atomic, not mutex-guarded, because enqueue runs
	// on business goroutines.
	dropped atomic.Int64

	// mu guards runMap/orderMap and the runs inside them (outputs/error/end
	// fills and stream counting race with concurrent Each-style emitters).
	mu       sync.Mutex
	runMap   map[string]*Run
	orderMap map[string]runOrder
}

// startTrace indexes run and computes its trace id, dotted order, session
// fields, and local tree position — the verbatim port of core.py
// _start_trace. Callers must hold t.mu (or be single-threaded pre-enqueue).
func (t *LangChainTracer) startTrace(run *Run) {
	current := dottedOrderSuffix(run.StartTime, run.ID)
	if run.ParentRunID != "" {
		if parent, ok := t.orderMap[run.ParentRunID]; ok {
			run.TraceID = parent.traceID
			run.DottedOrder = parent.dottedOrder + "." + current
			if parentRun, ok := t.runMap[run.ParentRunID]; ok {
				parentRun.ChildRuns = append(parentRun.ChildRuns, run)
			}
		} else {
			// core.py:135-144 — unknown parent demotes the run to a root.
			run.ParentRunID = ""
			run.TraceID = run.ID
			run.DottedOrder = current
		}
	} else {
		run.TraceID = run.ID
		run.DottedOrder = current
	}
	run.SessionID = run.TraceID
	run.SessionName = t.opts.Project
	if len(run.Metadata) > 0 {
		metadata := make(map[string]any, len(run.Metadata))
		for key, value := range run.Metadata {
			metadata[key] = value
		}
		run.Extra = map[string]any{"metadata": metadata}
	}
	t.orderMap[run.ID] = runOrder{traceID: run.TraceID, dottedOrder: run.DottedOrder}
	t.runMap[run.ID] = run
}

// HandleEvent implements callbacks.Handler with the run-tree state machine.
// It always returns nil: tracing failures are silent by design.
func (t *LangChainTracer) HandleEvent(_ context.Context, event callbacks.Event) error {
	if t == nil || event.RunID == "" {
		return nil
	}
	switch kind := event.Kind; {
	case isStartKind(kind):
		t.handleStart(event)
	case isEndKind(kind), isErrorKind(kind):
		t.handleEnd(event)
	case isStreamKind(kind):
		t.handleStream(event)
	}
	return nil
}

func isStartKind(kind callbacks.EventKind) bool {
	switch kind {
	case callbacks.EventChainStart, callbacks.EventChatModelStart,
		callbacks.EventLLMStart, callbacks.EventToolStart, callbacks.EventRetrieverStart:
		return true
	}
	return false
}

func isEndKind(kind callbacks.EventKind) bool {
	switch kind {
	case callbacks.EventChainEnd, callbacks.EventChatModelEnd,
		callbacks.EventLLMEnd, callbacks.EventToolEnd, callbacks.EventRetrieverEnd:
		return true
	}
	return false
}

func isErrorKind(kind callbacks.EventKind) bool {
	switch kind {
	case callbacks.EventChainError, callbacks.EventChatModelError,
		callbacks.EventLLMError, callbacks.EventToolError, callbacks.EventRetrieverError:
		return true
	}
	return false
}

func isStreamKind(kind callbacks.EventKind) bool {
	switch kind {
	case callbacks.EventChainStream, callbacks.EventChatModelStream,
		callbacks.EventLLMStream, callbacks.EventChatModelProtocol:
		return true
	}
	return false
}

// runTypeFor maps a start-kind event to its RunType. Chat models keep the
// streaming-events "chat_model" type (the alignment target of the v2 event
// surface) instead of the legacy "llm".
func runTypeFor(kind callbacks.EventKind) RunType {
	switch kind {
	case callbacks.EventChatModelStart:
		return RunTypeChatModel
	case callbacks.EventLLMStart:
		return RunTypeLLM
	case callbacks.EventToolStart:
		return RunTypeTool
	case callbacks.EventRetrieverStart:
		return RunTypeRetriever
	default:
		return RunTypeChain
	}
}

// handleStart builds the Run for a start-kind event. Inputs is eagerly
// serialized to its wire bytes (eagerTraceJSON): the queued snapshot must own
// immutable data, because the caller may mutate its input map as soon as the
// traced call returns while the flusher might not serialize for up to
// FlushEvery — sharing the reference is a data race. The snapshot is taken
// inside the critical section so a concurrent handleEnd for the same run
// cannot mutate the Run struct mid-copy.
func (t *LangChainTracer) handleStart(event callbacks.Event) {
	run := &Run{
		ID:          event.RunID,
		Name:        event.Name,
		RunType:     runTypeFor(event.Kind),
		ParentRunID: event.ParentID,
		Tags:        event.Tags,
		Metadata:    event.Metadata,
		Inputs:      t.eagerTraceJSON(event.Input),
		StartTime:   event.Timestamp.UTC(),
	}
	t.mu.Lock()
	t.startTrace(run)
	snapshot := *run
	t.mu.Unlock()
	t.enqueue(runOperation{run: &snapshot})
}

// handleEnd fills outputs/error/end time on the live run and queues the patch
// snapshot. Outputs is eagerly serialized for the same ownership reason as
// handleStart's Inputs, and the snapshot copy happens under the lock so the
// run's fields cannot change mid-copy.
func (t *LangChainTracer) handleEnd(event callbacks.Event) {
	outputs := t.eagerTraceJSON(event.Output)
	t.mu.Lock()
	run, ok := t.runMap[event.RunID]
	var snapshot *Run
	if ok {
		if outputs != nil {
			run.Outputs = outputs
		}
		run.Error = event.Error
		endTime := event.Timestamp.UTC()
		run.EndTime = &endTime
		delete(t.runMap, event.RunID)
		copied := *run
		snapshot = &copied
	}
	t.mu.Unlock()
	if snapshot == nil {
		return // end for an unknown or already-ended run: nothing to update
	}
	t.enqueue(runOperation{patch: true, run: snapshot})
}

// eagerTraceJSON pre-serializes value into its exact wire form so the Run (and
// the snapshots queued for the flusher) owns immutable bytes instead of the
// caller's live maps. json.RawMessage embeds byte-identically during batch
// encoding — encoding/json treats it as literal JSON — so the /runs/batch
// payload is unchanged. A value that cannot be serialized at all (func,
// channel, cycle in a metadata-rich payload) is dropped with an OnError note:
// the alternative — keeping the reference — is a data race by construction.
func (t *LangChainTracer) eagerTraceJSON(value any) any {
	if value == nil {
		return nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		t.reportError(fmt.Errorf("langsmith: snapshot inputs/outputs not serializable; field dropped: %w", err))
		return nil
	}
	return json.RawMessage(raw)
}

func (t *LangChainTracer) handleStream(event callbacks.Event) {
	t.mu.Lock()
	if run, ok := t.runMap[event.RunID]; ok {
		run.streams++
	}
	t.mu.Unlock()
}

// enqueue offers one operation to the flusher without ever blocking the
// business call: a full (or closing) queue drops the op, counting it in
// t.dropped — the aggregated overflow is reported by the flusher's periodic
// reportDropped, not per op, so an overload episode surfaces as one bounded
// OnError wave instead of one callback per dropped operation (a per-op
// observer firing hundreds of times would itself become the outage). The
// recover guards the tiny race where Close closes the channel between the
// closing check and the send.
func (t *LangChainTracer) enqueue(op runOperation) {
	if t.closing.Load() {
		return
	}
	defer func() { _ = recover() }()
	select {
	case t.queue <- op:
	default:
		t.dropped.Add(1)
	}
}

// reportDropped folds the overflow drops observed since the previous report
// into a single OnError notification and resets the counter. Called by the
// flusher after each shipping point (batch fill, ticker, Flush barrier, final
// drain), so the number of notifications for one overload episode is bounded
// by the number of shipping points, not by the number of dropped operations.
func (t *LangChainTracer) reportDropped() {
	if n := t.dropped.Swap(0); n > 0 {
		t.reportError(fmt.Errorf("langsmith: tracing queue full; dropped %d run operations", n))
	}
}

// Flush drains the queue: it asks the flusher to ship everything currently
// queued and waits until that batch has been POSTed (or failed), capped at
// 5 seconds. Network failures are not returned here — they go to OnError;
// only the deadline is surfaced, as a context error.
func (t *LangChainTracer) Flush(ctx context.Context) error {
	if t == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, langSmithFlushTimeout)
	defer cancel()
	ack := make(chan struct{})
	select {
	case t.flushNow <- ack:
	case <-t.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case <-ack:
		return nil
	case <-t.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close stops accepting events, drains the queue one final time, waits (up
// to 5s) for the flusher to exit, and is idempotent. It always returns nil.
func (t *LangChainTracer) Close() error {
	if t == nil {
		return nil
	}
	if !t.closing.CompareAndSwap(false, true) {
		select {
		case <-t.done:
		case <-time.After(langSmithFlushTimeout):
		}
		return nil
	}
	// From here HandleEvent/enqueue drop new work; closing the queue makes
	// the flusher ship the remainder and exit.
	close(t.queue)
	select {
	case <-t.done:
	case <-time.After(langSmithFlushTimeout):
	}
	return nil
}

// flushLoop is the single flusher goroutine: it batches queue operations and
// ships them when the batch fills, the ticker fires, or a Flush caller asks
// for a barrier (the ack is sent only after the drained batch is POSTed).
func (t *LangChainTracer) flushLoop() {
	defer close(t.done)
	ticker := time.NewTicker(t.opts.FlushEvery)
	defer ticker.Stop()
	var batch []runOperation
	for {
		select {
		case op, ok := <-t.queue:
			if !ok {
				t.ship(batch)
				t.reportDropped()
				return
			}
			batch = append(batch, op)
			if len(batch) >= t.opts.BatchSize {
				batch = t.ship(batch)
				t.reportDropped()
			}
		case <-ticker.C:
			batch = t.ship(batch)
			t.reportDropped()
		case ack := <-t.flushNow:
			batch = t.drainQueue(batch)
			batch = t.ship(batch)
			t.reportDropped()
			close(ack)
		}
	}
}

// drainQueue pulls everything currently buffered on the queue into batch,
// shipping full batches as it goes, and returns when the queue is empty.
func (t *LangChainTracer) drainQueue(batch []runOperation) []runOperation {
	for {
		select {
		case op, ok := <-t.queue:
			if !ok {
				return batch
			}
			batch = append(batch, op)
			if len(batch) >= t.opts.BatchSize {
				batch = t.ship(batch)
			}
		default:
			return batch
		}
	}
}

// ship POSTs batch (if any) as one /runs/batch request with a single retry,
// then returns a nil batch. Failures after the retry go to OnError and the
// operations are dropped — they are never re-queued.
func (t *LangChainTracer) ship(batch []runOperation) []runOperation {
	if len(batch) == 0 {
		return nil
	}
	body := batchBody{Post: make([]*Run, 0, len(batch)), Patch: make([]*Run, 0, len(batch))}
	for _, op := range batch {
		if op.patch {
			body.Patch = append(body.Patch, op.run)
		} else {
			body.Post = append(body.Post, op.run)
		}
	}
	payload, err := json.Marshal(body)
	if err != nil {
		t.reportError(fmt.Errorf("langsmith: encode /runs/batch: %w", err))
		return nil
	}
	if err := t.post(payload); err != nil {
		t.reportError(err)
	}
	return nil
}

// batchBody is the /runs/batch request envelope. Both arrays are always
// present (possibly empty), matching the Python client.
type batchBody struct {
	Post  []*Run `json:"post"`
	Patch []*Run `json:"patch"`
}

// post sends one batch with at most one retry on any failure (network error
// or non-2xx status).
func (t *LangChainTracer) post(payload []byte) error {
	url := t.opts.Endpoint + "/runs/batch"
	var lastErr error
	for range 2 {
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return fmt.Errorf("langsmith: build /runs/batch request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if t.opts.APIKey != "" {
			req.Header.Set("X-API-KEY", t.opts.APIKey)
		}
		resp, err := t.opts.HTTPClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("langsmith: POST /runs/batch: %w", err)
			continue
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return nil
		}
		lastErr = fmt.Errorf("langsmith: POST /runs/batch returned %s", resp.Status)
	}
	return lastErr
}

// reportError routes a background failure to OnError, or to slog.Debug when
// no observer is installed. Recovering here keeps a panicking OnError from
// killing the flusher.
func (t *LangChainTracer) reportError(err error) {
	if err == nil {
		return
	}
	if t.opts.OnError != nil {
		defer func() { _ = recover() }()
		t.opts.OnError(err)
		return
	}
	slog.Debug(err.Error())
}
