// This file implements the Runnable-level StreamEvents driver (design t18
// PR2): the Go counterpart of Python's astream_events(v2). It wraps one
// Runnable invocation in a callback-collecting manager, projects the flat
// callbacks.Event stream onto the v2 StreamEvent shape (event name / run_id /
// root-first parent_ids / data), aggregates chat-model streams (both the
// legacy message-chunk callbacks and the v3 content-block protocol), filters
// at the projection output, and drives everything through a langgraph-style
// iterator whose break joins the producer goroutine.
//
// Layer boundaries: langgraph streams graph-state modes, langchain/agents
// projects its seven domain event kinds, and this driver surfaces the
// Runnable-tree v2 events. Pick one surface per run; they do not interlock.
//
// Root-level events: the driver does NOT synthesize its own chain run. It
// allocates the root run ID and installs the collector manager via
// WithRunID/WithCallbacks, which activates the root runnable's own chain
// instrumentation (chain_start{input} / chain_stream per chunk /
// chain_end|chain_error, design section 4). A non-instrumented leaf runnable
// therefore contributes only whatever leaf events it emits itself (e.g. a
// chat model's on_chat_model_*); the EventStreamer escape hatch for such
// leaves is deliberately deferred to P2.
package runnables

import (
	"context"
	"errors"
	"iter"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/streamevents"
)

// StreamEventOptions filters the events yielded by StreamEvents, the Go
// counterpart of Python's astream_events(include_*/exclude_*) filters.
//
// Semantics mirror _RootEventFilter.include_event (runnables/utils.py:696):
// each include list that is set must admit the event (any member matches —
// include OR), each exclude list must not match (exclude AND), and with no
// include lists set every event passes subject to the excludes. Filtering
// happens at the projection output only: collection, run-tree resolution,
// and chat-model aggregation always observe the full event stream, so a
// filtered-out run never breaks the parent_ids of its surviving descendants.
type StreamEventOptions struct {
	// IncludeNames / ExcludeNames match StreamEvent.Name.
	IncludeNames, ExcludeNames []string
	// IncludeTypes / ExcludeTypes match the run type derived from the event
	// name: chain|chat_model|llm|tool|retriever.
	IncludeTypes, ExcludeTypes []string
	// IncludeTags / ExcludeTags match inherited event tags: an include list
	// requires at least one shared tag, an exclude list rejects any overlap.
	IncludeTags, ExcludeTags []string
}

// Includes reports whether event passes the filter; see StreamEventOptions
// for the include-OR / exclude-AND semantics.
func (o StreamEventOptions) Includes(event StreamEvent) bool {
	if len(o.IncludeNames) > 0 && !containsString(o.IncludeNames, event.Name) {
		return false
	}
	if len(o.ExcludeNames) > 0 && containsString(o.ExcludeNames, event.Name) {
		return false
	}
	runType := eventRunType(event.Event)
	if len(o.IncludeTypes) > 0 && !containsString(o.IncludeTypes, runType) {
		return false
	}
	if len(o.ExcludeTypes) > 0 && containsString(o.ExcludeTypes, runType) {
		return false
	}
	if len(o.IncludeTags) > 0 && !overlapsTags(o.IncludeTags, event.Tags) {
		return false
	}
	if len(o.ExcludeTags) > 0 && overlapsTags(o.ExcludeTags, event.Tags) {
		return false
	}
	return true
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func overlapsTags(want []string, tags []string) bool {
	for _, tag := range tags {
		if containsString(want, tag) {
			return true
		}
	}
	return false
}

// StreamEvents runs r as a stream and yields v2-shaped StreamEvents for every
// callback the run emits (combinator chain events, chat-model lifecycle and
// stream events). The returned iterator is single-use; a run failure is
// yielded as the final (zero, err) pair after all events. Breaking out of the
// iteration cancels the run and joins the producer goroutine before the
// iterator returns, so no goroutine leaks. Concurrent child runs (e.g. Each)
// interleave freely; their events pair by RunID, not by global nesting order.
func StreamEvents[I, O any](
	ctx context.Context,
	r Runnable[I, O],
	input I,
	eo StreamEventOptions,
	opts ...Option,
) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		runCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		// The root run ID: a caller-supplied WithRunID wins, mirroring how
		// PR1 instrumentation honors it as the run's own identity.
		cfg := NewConfig(opts...)
		root := cfg.RunID
		if root == "" {
			root = NewRunID()
		}

		// The collector serializes handler callbacks (which may fire on
		// worker goroutines) into one buffered channel; the consumer loop is
		// the single reader, so projection state stays race-free.
		events := make(chan callbacks.Event, 64)
		collector := &eventCollector{ctx: runCtx, ch: events, closed: &atomic.Bool{}}

		manager := callbacks.NewManager(collector)
		if !cfg.Callbacks.Empty() {
			// Caller handlers stay in the fan-out (ahead of the collector) so
			// existing callback consumers observe the same events as before.
			// A Manager is itself a Handler, so nesting preserves its tag /
			// metadata / parent inheritance.
			manager = callbacks.NewManager(cfg.Callbacks, collector)
		}
		// TODO(t18 PR3): when tracers.EnabledFromEnv() and the caller has not
		// attached a LangSmith tracer, append one to this manager (before the
		// collector) so tracing works without a drain-gated channel hop. The
		// tracers package is intentionally unreferenced until PR3 lands.

		done := make(chan struct{})
		var runErr error
		runOpts := append(append([]Option(nil), opts...), WithRunID(root), WithCallbacks(manager))

		go func() {
			// LIFO teardown: mark the channel closed, close it, then signal
			// done. runErr is assigned before any of these run, and the
			// channel close synchronizes-with the consumer observing it, so
			// the consumer reads runErr race-free once the range ends
			// (same protocol as langgraph CompiledGraph.Stream).
			defer close(done)
			defer close(events)
			defer collector.closed.Store(true)
			stream, err := r.Stream(runCtx, input, runOpts...)
			if err != nil {
				runErr = err
				return
			}
			defer stream.Close()
			for {
				if _, ok, err := stream.Next(runCtx); err != nil {
					runErr = err
					return
				} else if !ok {
					return
				}
			}
		}()

		projector := &streamEventProjector{registry: &parentIDRegistry{}}
		for event := range events {
			projected := projector.project(event)
			if projected.Event == "" {
				continue // folded (message-start/finish) or suppressed (legacy chunk under v3)
			}
			if !eo.Includes(projected) {
				continue
			}
			if !yield(projected, nil) {
				// Early break: cancel the run and wait for the producer
				// goroutine to exit so nothing outlives the iterator.
				cancel()
				<-done
				return
			}
		}
		if runErr != nil {
			yield(StreamEvent{}, runErr)
		}
	}
}

// eventCollector is the callbacks.Handler that feeds the driver's channel.
// Sends select on the run context and are guarded by a closed flag plus
// recover, so a late emission racing the producer's teardown (e.g. a worker
// goroutine that outlived its runnable) is dropped instead of panicking on
// the closed channel. Handler errors are never surfaced: instrumentation
// must not fail the business call.
type eventCollector struct {
	ctx    context.Context
	ch     chan callbacks.Event
	closed *atomic.Bool
}

// HandleEvent implements callbacks.Handler.
func (c *eventCollector) HandleEvent(_ context.Context, event callbacks.Event) error {
	if c.closed.Load() {
		return nil
	}
	defer func() { _ = recover() }()
	select {
	case c.ch <- event:
	case <-c.ctx.Done():
	}
	return nil
}

// streamEventProjector carries the per-iteration projection state: the
// parent-id registry and one chat-model aggregator per model run. It lives on
// the consumer goroutine only, so its maps need no locking.
type streamEventProjector struct {
	registry *parentIDRegistry
	aggs     map[string]*chatModelAggregator
}

// project resolves parent_ids, feeds chat-model aggregation, and maps one
// callbacks.Event onto the v2 shape. A zero StreamEvent (empty Event name)
// means "no output event" — folded protocol events and legacy chunks
// suppressed under the v3 protocol.
func (p *streamEventProjector) project(e callbacks.Event) StreamEvent {
	var agg *chatModelAggregator
	if isChatModelKind(e.Kind) {
		agg = p.aggs[e.RunID]
		if agg == nil {
			agg = newChatModelAggregator()
			if p.aggs == nil {
				p.aggs = make(map[string]*chatModelAggregator)
			}
			p.aggs[e.RunID] = agg
		}
	}
	return StreamEventFromCallback(e, p.registry.resolve(e.RunID, e.ParentID), agg)
}

func isChatModelKind(kind callbacks.EventKind) bool {
	switch kind {
	case callbacks.EventChatModelStart,
		callbacks.EventChatModelStream,
		callbacks.EventChatModelProtocol,
		callbacks.EventChatModelEnd,
		callbacks.EventChatModelError:
		return true
	}
	return false
}

// parentIDRegistry caches each run's root-first ancestor chain, built by
// prefix append so nested runs share immutable prefixes. The root run (empty
// ParentID) resolves to nil. Mutex-protected because the exported
// StreamEventFromCallback surface may be driven from arbitrary goroutines.
type parentIDRegistry struct {
	mu      sync.Mutex
	parents map[string][]string
}

// resolve returns the ancestor chain of runID, root first, excluding runID
// itself; nil for a root run. Cached chains (and the chains handed to
// StreamEvent.ParentIDs) are shared, never mutated after storage — consumers
// must treat ParentIDs as read-only.
func (p *parentIDRegistry) resolve(runID string, parentID string) []string {
	if parentID == "" {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.parents == nil {
		p.parents = make(map[string][]string)
	}
	if cached, ok := p.parents[runID]; ok && runID != "" {
		return cached
	}
	prefix := p.parents[parentID]
	chain := make([]string, len(prefix)+1)
	copy(chain, prefix)
	chain[len(prefix)] = parentID
	if runID != "" {
		p.parents[runID] = chain
	}
	return chain
}

// chatModelAggregator assembles one chat-model run's final output for
// on_chat_model_end and implements the v3-over-legacy chunk preference. It
// reuses core/streamevents.ChatModelStream for the v3 content-block protocol
// and sums legacy message chunks directly (content concatenation — Python
// AIMessageChunk addition — rather than message-run merging, which joins
// with newlines).
type chatModelAggregator struct {
	inner       *streamevents.ChatModelStream
	legacy      []messages.Message
	sawProtocol bool
}

func newChatModelAggregator() *chatModelAggregator {
	return &chatModelAggregator{inner: streamevents.NewChatModelStream()}
}

// dispatchProtocol feeds one v3 protocol event into the aggregator and
// reports whether it surfaces as an on_chat_model_stream chunk (only
// content-block deltas do; message-start/finish and block boundaries fold
// into the surrounding start/end events).
func (a *chatModelAggregator) dispatchProtocol(event streamevents.Event) bool {
	if a == nil {
		return false
	}
	a.sawProtocol = true
	a.inner.Dispatch(event)
	return event.Event == streamevents.EventContentBlockDelta
}

// pushLegacy records one legacy message chunk for end-of-run aggregation.
func (a *chatModelAggregator) pushLegacy(chunk messages.Message) {
	if a == nil {
		return
	}
	a.legacy = append(a.legacy, chunk)
}

// suppressLegacy reports whether a legacy stream chunk should be dropped
// because the same run already emitted v3 protocol deltas (v3 delta
// preferred — providers emit both for one logical delta).
func (a *chatModelAggregator) suppressLegacy() bool {
	return a != nil && a.sawProtocol
}

// endOutput fills on_chat_model_end's Data.Output: the v3 aggregator's
// assembled message when message-finish was seen, else the event's own
// output (non-streaming runs and providers that attach the final message),
// else the summed legacy chunks, else nil.
func (a *chatModelAggregator) endOutput(eventOutput any) any {
	if a == nil {
		return eventOutput
	}
	if a.inner.Done() {
		if output, err := a.inner.Output(); err == nil {
			return output
		}
	}
	if eventOutput != nil {
		return eventOutput
	}
	if len(a.legacy) == 0 {
		return nil
	}
	summed := a.legacy[0]
	for _, chunk := range a.legacy[1:] {
		summed.Content += chunk.Content
		summed.ContentBlocks = append(summed.ContentBlocks, chunk.ContentBlocks...)
		summed.ToolCalls = append(summed.ToolCalls, chunk.ToolCalls...)
		summed.InvalidToolCalls = append(summed.InvalidToolCalls, chunk.InvalidToolCalls...)
		summed.UsageMetadata.InputTokens += chunk.UsageMetadata.InputTokens
		summed.UsageMetadata.OutputTokens += chunk.UsageMetadata.OutputTokens
		summed.UsageMetadata.TotalTokens += chunk.UsageMetadata.TotalTokens
		if summed.Name == "" {
			summed.Name = chunk.Name
		}
		if summed.ID == "" {
			summed.ID = chunk.ID
		}
		if summed.ResponseMetadata == nil {
			summed.ResponseMetadata = chunk.ResponseMetadata
		}
	}
	return summed
}

// StreamEventFromCallback projects one flat callbacks.Event onto the v2
// StreamEvent shape; it is the reusable projection surface for callers that
// drive their own callback collection (the StreamEvents driver uses it
// internally).
//
// parentIDs is the pre-resolved root-first ancestor chain (see
// parentIDRegistry); pass nil for a root run. agg carries the chat-model
// run's aggregation state: chat-model kinds require it for end-output
// assembly and the v3/legacy preference, and a nil agg degrades chat-model
// projection to direct mapping (protocol events then fold away entirely).
//
// A returned StreamEvent with an empty Event marks "no output event": v3
// protocol events other than content-block-delta fold into the surrounding
// start/end, and legacy chat-model chunks are suppressed once the same run
// emitted v3 deltas. Callers skip such events.
func StreamEventFromCallback(e callbacks.Event, parentIDs []string, agg *chatModelAggregator) StreamEvent {
	if e.Kind == callbacks.EventChatModelProtocol {
		protocol, ok := e.Chunk.(streamevents.Event)
		if !ok || !agg.dispatchProtocol(protocol) {
			return StreamEvent{}
		}
		// Only content-block deltas surface; the chunk payload is the v3
		// protocol event itself (documented divergence from v2's
		// AIMessageChunk — v3 delta preferred).
		return newStreamEvent("on_chat_model_stream", e, parentIDs, StreamEventData{Chunk: protocol})
	}
	name, ok := callbackEventNames[e.Kind]
	if !ok {
		return StreamEvent{}
	}
	data := StreamEventData{}
	switch {
	case strings.HasSuffix(name, "_start"):
		data.Input = e.Input
	case strings.HasSuffix(name, "_stream"):
		if e.Kind == callbacks.EventChatModelStream && agg.suppressLegacy() {
			return StreamEvent{}
		}
		if e.Kind == callbacks.EventChatModelStream {
			if chunk, isMessage := e.Chunk.(messages.Message); isMessage {
				agg.pushLegacy(chunk)
			}
		}
		data.Chunk = e.Chunk
	case strings.HasSuffix(name, "_end"):
		data.Output = e.Output
		if e.Kind == callbacks.EventChatModelEnd {
			data.Output = agg.endOutput(e.Output)
		}
	case strings.HasSuffix(name, "_error"):
		if e.Error != "" {
			data.Error = errors.New(e.Error)
		}
	}
	return newStreamEvent(name, e, parentIDs, data)
}

func newStreamEvent(event string, e callbacks.Event, parentIDs []string, data StreamEventData) StreamEvent {
	return StreamEvent{
		Event:     event,
		RunID:     e.RunID,
		Name:      e.Name,
		Tags:      e.Tags,
		Metadata:  e.Metadata,
		ParentIDs: parentIDs,
		Data:      data,
	}
}

// callbackEventNames maps callback kinds to the v2 event names (Python
// StandardStreamEvent). Kinds absent from the map project to no event.
// chat_model_protocol is absent by intent: protocol events project through
// the aggregator path in StreamEventFromCallback.
var callbackEventNames = map[callbacks.EventKind]string{
	callbacks.EventChainStart:      "on_chain_start",
	callbacks.EventChainStream:     "on_chain_stream",
	callbacks.EventChainEnd:        "on_chain_end",
	callbacks.EventChainError:      "on_chain_error",
	callbacks.EventChatModelStart:  "on_chat_model_start",
	callbacks.EventChatModelStream: "on_chat_model_stream",
	callbacks.EventChatModelEnd:    "on_chat_model_end",
	callbacks.EventChatModelError:  "on_chat_model_error",
	callbacks.EventLLMStart:        "on_llm_start",
	callbacks.EventLLMStream:       "on_llm_stream",
	callbacks.EventLLMEnd:          "on_llm_end",
	callbacks.EventLLMError:        "on_llm_error",
	callbacks.EventToolStart:       "on_tool_start",
	callbacks.EventToolEnd:         "on_tool_end",
	callbacks.EventToolError:       "on_tool_error",
	callbacks.EventRetrieverStart:  "on_retriever_start",
	callbacks.EventRetrieverEnd:    "on_retriever_end",
	callbacks.EventRetrieverError:  "on_retriever_error",
}

// eventRunType derives the v2 run type from an event name, the key that
// IncludeTypes/ExcludeTypes filter on.
func eventRunType(event string) string {
	switch {
	case strings.HasPrefix(event, "on_chat_model_"):
		return "chat_model"
	case strings.HasPrefix(event, "on_llm_"):
		return "llm"
	case strings.HasPrefix(event, "on_tool_"):
		return "tool"
	case strings.HasPrefix(event, "on_retriever_"):
		return "retriever"
	case strings.HasPrefix(event, "on_chain_"):
		return "chain"
	case event == "on_custom_event":
		return "custom"
	}
	return ""
}
