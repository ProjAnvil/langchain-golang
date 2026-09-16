// Tests for the LangSmith LangChainTracer (design t18 PR3, section 6): the
// /runs/batch payload contract against an httptest server (dotted_order,
// trace_id, parent_run_id, inputs/outputs/error, run_type), the env-var gate
// (four combinations, LANGSMITH_ alias first), silent failure with one retry
// on server errors, Flush draining, and Close idempotency.
package tracers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
)

// lsServer is a recording /runs/batch fake. mode "ok" answers 200; "500"
// always answers 500 (to exercise retry + silence); "hang" never answers.
type lsServer struct {
	srv      *httptest.Server
	mu       sync.Mutex
	bodies   []map[string]any
	rawBody  []string
	headers  []http.Header
	requests atomic.Int64
}

func newLSServer(mode string) *lsServer {
	s := &lsServer{}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.requests.Add(1)
		raw, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		s.rawBody = append(s.rawBody, string(raw))
		s.headers = append(s.headers, r.Header.Clone())
		s.mu.Unlock()
		var decoded map[string]any
		_ = json.Unmarshal(raw, &decoded)
		s.mu.Lock()
		s.bodies = append(s.bodies, decoded)
		s.mu.Unlock()
		switch mode {
		case "500":
			w.WriteHeader(http.StatusInternalServerError)
		case "hang":
			<-r.Context().Done()
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	return s
}

func (s *lsServer) URL() string { return s.srv.URL }

func (s *lsServer) Close() { s.srv.Close() }

func (s *lsServer) count() int64 { return s.requests.Load() }

func (s *lsServer) lastBody(t *testing.T) map[string]any {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.bodies) == 0 {
		t.Fatal("no /runs/batch request received")
	}
	return s.bodies[len(s.bodies)-1]
}

func (s *lsServer) lastHeader(t *testing.T) http.Header {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.headers) == 0 {
		t.Fatal("no /runs/batch request received")
	}
	return s.headers[len(s.headers)-1]
}

func lsTracer(t *testing.T, endpoint string, onError func(error)) *LangChainTracer {
	t.Helper()
	tracer := NewLangChainTracerWithOptions(LangSmithOptions{
		Endpoint: endpoint, APIKey: "test-key", Project: "proj",
		BatchSize: 64, FlushEvery: time.Hour, OnError: onError,
	})
	if tracer == nil {
		t.Fatal("NewLangChainTracerWithOptions returned nil")
	}
	return tracer
}

func lsRuns(t *testing.T, body map[string]any, key string) []map[string]any {
	t.Helper()
	value, ok := body[key].([]any)
	if !ok {
		t.Fatalf("body missing %q array: %#v", key, body)
	}
	out := make([]map[string]any, len(value))
	for i, item := range value {
		run, ok := item.(map[string]any)
		if !ok {
			t.Fatalf("%s[%d] is not an object: %#v", key, i, item)
		}
		out[i] = run
	}
	return out
}

// lsEvent builds a callback event with a deterministic timestamp.
func lsEvent(kind callbacks.EventKind, name, runID, parentID string, ts time.Time, mutators ...func(*callbacks.Event)) callbacks.Event {
	event := callbacks.Event{Kind: kind, Name: name, RunID: runID, ParentID: parentID, Timestamp: ts}
	for _, mutator := range mutators {
		mutator(&event)
	}
	return event
}

var lsTS = func(second int) time.Time {
	return time.Date(2026, 9, 12, 10, 20, second, 123456000, time.UTC)
}

func TestLangSmithBatchPayloadRootAndChild(t *testing.T) {
	server := newLSServer("ok")
	defer server.Close()
	tracer := lsTracer(t, server.URL(), nil)
	ctx := t.Context()

	rootID, childID := "aaaaaaaa-0000-0000-0000-000000000001", "aaaaaaaa-0000-0000-0000-000000000002"
	events := []callbacks.Event{
		lsEvent(callbacks.EventChainStart, "chain", rootID, "", lsTS(30), func(e *callbacks.Event) {
			e.Input = "go"
			e.Tags = []string{"team:x"}
			e.Metadata = map[string]any{"uid": 7}
		}),
		lsEvent(callbacks.EventChainStart, "step", childID, rootID, lsTS(31), func(e *callbacks.Event) { e.Input = "go" }),
		lsEvent(callbacks.EventChainEnd, "step", childID, rootID, lsTS(32), func(e *callbacks.Event) { e.Output = "gogo" }),
		lsEvent(callbacks.EventChainEnd, "chain", rootID, "", lsTS(33), func(e *callbacks.Event) { e.Output = "GOGO" }),
	}
	for _, event := range events {
		if err := tracer.HandleEvent(ctx, event); err != nil {
			t.Fatalf("HandleEvent returned %v, want always nil", err)
		}
	}
	if err := tracer.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := tracer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	body := server.lastBody(t)
	// Both arrays are always present (Python sends {"post": [...], "patch": [...]}).
	posts, patches := lsRuns(t, body, "post"), lsRuns(t, body, "patch")
	if len(posts) != 2 || len(patches) != 2 {
		t.Fatalf("post=%d patch=%d, want 2/2: %s", len(posts), len(patches), server.rawBody[len(server.rawBody)-1])
	}
	if got := server.lastHeader(t).Get("X-API-KEY"); got != "test-key" {
		t.Fatalf("x-api-key header %q", got)
	}

	rootPost, childPost := posts[0], posts[1]
	if rootPost["id"] != rootID || rootPost["name"] != "chain" || rootPost["run_type"] != "chain" {
		t.Fatalf("root post: %#v", rootPost)
	}
	if rootPost["trace_id"] != rootID {
		t.Fatalf("root trace_id %v", rootPost["trace_id"])
	}
	wantRootDO := "20260912T102030123456Z" + rootID
	if rootPost["dotted_order"] != wantRootDO {
		t.Fatalf("root dotted_order %v, want %s", rootPost["dotted_order"], wantRootDO)
	}
	if _, ok := rootPost["parent_run_id"]; ok {
		t.Fatalf("root post has parent_run_id: %#v", rootPost)
	}
	if rootPost["inputs"] != "go" {
		t.Fatalf("root inputs %#v", rootPost["inputs"])
	}
	if rootPost["session_id"] != rootID {
		t.Fatalf("session_id %v, want trace id", rootPost["session_id"])
	}
	if rootPost["session_name"] != "proj" {
		t.Fatalf("session_name %v", rootPost["session_name"])
	}
	if start, ok := rootPost["start_time"].(string); !ok || !strings.HasPrefix(start, "2026-09-12T10:20:30") {
		t.Fatalf("start_time %#v", rootPost["start_time"])
	}
	if _, ok := rootPost["end_time"]; ok {
		t.Fatalf("create posted end_time: %#v", rootPost)
	}
	// extra.metadata carries the start event metadata.
	extra, _ := rootPost["extra"].(map[string]any)
	metadata, _ := extra["metadata"].(map[string]any)
	if metadata["uid"] != float64(7) {
		t.Fatalf("extra.metadata %v", extra)
	}

	if childPost["parent_run_id"] != rootID || childPost["trace_id"] != rootID {
		t.Fatalf("child post: %#v", childPost)
	}
	wantChildDO := wantRootDO + ".20260912T102031123456Z" + childID
	if childPost["dotted_order"] != wantChildDO {
		t.Fatalf("child dotted_order %v, want %s", childPost["dotted_order"], wantChildDO)
	}

	childPatch, rootPatch := patches[0], patches[1]
	if childPatch["id"] != childID || childPatch["outputs"] != "gogo" {
		t.Fatalf("child patch: %#v", childPatch)
	}
	if rootPatch["id"] != rootID || rootPatch["outputs"] != "GOGO" {
		t.Fatalf("root patch: %#v", rootPatch)
	}
	if end, ok := rootPatch["end_time"].(string); !ok || !strings.HasPrefix(end, "2026-09-12T10:20:33") {
		t.Fatalf("patch end_time %#v", rootPatch["end_time"])
	}
	if rootPatch["dotted_order"] != wantRootDO {
		t.Fatalf("patch dotted_order %v", rootPatch["dotted_order"])
	}
}

func TestLangSmithErrorRunAndRunTypes(t *testing.T) {
	server := newLSServer("ok")
	defer server.Close()
	tracer := lsTracer(t, server.URL(), nil)
	ctx := t.Context()

	toolID := "bbbbbbbb-0000-0000-0000-000000000001"
	modelID := "bbbbbbbb-0000-0000-0000-000000000002"
	for _, event := range []callbacks.Event{
		lsEvent(callbacks.EventToolStart, "search", toolID, "", lsTS(30), func(e *callbacks.Event) { e.Input = "q" }),
		lsEvent(callbacks.EventToolError, "search", toolID, "", lsTS(31), func(e *callbacks.Event) { e.Error = "boom" }),
		lsEvent(callbacks.EventChatModelStart, "gpt", modelID, "", lsTS(32), func(e *callbacks.Event) { e.Input = "msgs" }),
		lsEvent(callbacks.EventChatModelEnd, "gpt", modelID, "", lsTS(33), func(e *callbacks.Event) { e.Output = "answer" }),
	} {
		if err := tracer.HandleEvent(ctx, event); err != nil {
			t.Fatalf("HandleEvent returned %v", err)
		}
	}
	_ = tracer.Flush(ctx)
	_ = tracer.Close()

	body := server.lastBody(t)
	posts, patches := lsRuns(t, body, "post"), lsRuns(t, body, "patch")
	if len(posts) != 2 || len(patches) != 2 {
		t.Fatalf("post=%d patch=%d", len(posts), len(patches))
	}
	if posts[0]["run_type"] != "tool" || posts[1]["run_type"] != "chat_model" {
		t.Fatalf("run types: %v %v", posts[0]["run_type"], posts[1]["run_type"])
	}
	if patches[0]["error"] != "boom" {
		t.Fatalf("tool error patch: %#v", patches[0])
	}
	if patches[1]["outputs"] != "answer" {
		t.Fatalf("model end patch: %#v", patches[1])
	}
}

func TestLangSmithStreamEventsCountOnly(t *testing.T) {
	server := newLSServer("ok")
	defer server.Close()
	tracer := lsTracer(t, server.URL(), nil)
	ctx := t.Context()
	runID := "cccccccc-0000-0000-0000-000000000001"
	_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainStart, "chain", runID, "", lsTS(30)))
	_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainStream, "chain", runID, "", lsTS(31), func(e *callbacks.Event) { e.Chunk = "tok" }))
	// A stream for an unknown run must neither panic nor create a run.
	_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainStream, "chain", "unknown", "", lsTS(31)))
	if server.count() != 0 {
		t.Fatalf("stream events must not enqueue batches, got %d requests", server.count())
	}
	_ = tracer.Flush(ctx)

	tracer.mu.Lock()
	run := tracer.runMap[runID]
	tracer.mu.Unlock()
	if run == nil || run.streams != 1 {
		t.Fatalf("stream counting: %#v", run)
	}
	if got := server.count(); got != 1 {
		t.Fatalf("flush shipped %d batches, want 1 (the create only)", got)
	}
	_ = tracer.Close()
}

func TestEnabledFromEnvCombinations(t *testing.T) {
	cases := []struct {
		name    string
		env     map[string]string
		enabled bool
	}{
		{"nothing set", nil, false},
		{"tracing without key", map[string]string{"LANGCHAIN_TRACING_V2": "true"}, false},
		{"v2 flag plus legacy key", map[string]string{
			"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": "k"}, true},
		{"langsmith alias", map[string]string{
			"LANGSMITH_TRACING": "1", "LANGSMITH_API_KEY": "k"}, true},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			for key := range testCase.env {
				t.Setenv(key, testCase.env[key])
			}
			if got := EnabledFromEnv(); got != testCase.enabled {
				t.Fatalf("EnabledFromEnv = %v, want %v", got, testCase.enabled)
			}
			tracer := NewLangChainTracer()
			if testCase.enabled {
				if tracer == nil {
					t.Fatal("NewLangChainTracer returned nil while enabled")
				}
				defer func() { _ = tracer.Close() }()
			} else if tracer != nil {
				t.Fatal("NewLangChainTracer returned a tracer while disabled")
			}
			// Nil-safety: calling any method on the disabled (nil) tracer is a no-op.
			if err := tracer.HandleEvent(t.Context(), callbacks.Event{Kind: callbacks.EventChainStart}); err != nil {
				t.Fatalf("nil HandleEvent: %v", err)
			}
			if err := tracer.Flush(t.Context()); err != nil {
				t.Fatalf("nil Flush: %v", err)
			}
			if err := tracer.Close(); err != nil {
				t.Fatalf("nil Close: %v", err)
			}
		})
	}
}

func TestNewLangChainTracerReadsOptionsFromEnv(t *testing.T) {
	t.Setenv("LANGSMITH_TRACING", "true")
	t.Setenv("LANGSMITH_API_KEY", "env-key")
	t.Setenv("LANGSMITH_ENDPOINT", "http://env-endpoint.test/")
	t.Setenv("LANGSMITH_PROJECT", "env-proj")
	tracer := NewLangChainTracer()
	if tracer == nil {
		t.Fatal("tracer nil")
	}
	defer func() { _ = tracer.Close() }()
	if tracer.opts.Endpoint != "http://env-endpoint.test" {
		t.Fatalf("endpoint %q (trailing slash must be trimmed)", tracer.opts.Endpoint)
	}
	if tracer.opts.APIKey != "env-key" || tracer.opts.Project != "env-proj" {
		t.Fatalf("opts: %#v", tracer.opts)
	}
	// LANGCHAIN_-prefixed fallbacks apply too.
	t.Setenv("LANGSMITH_API_KEY", "")
	t.Setenv("LANGSMITH_ENDPOINT", "")
	t.Setenv("LANGSMITH_PROJECT", "")
	t.Setenv("LANGSMITH_TRACING", "")
	t.Setenv("LANGCHAIN_TRACING_V2", "true")
	t.Setenv("LANGCHAIN_API_KEY", "legacy-key")
	t.Setenv("LANGCHAIN_ENDPOINT", "http://legacy-endpoint.test")
	t.Setenv("LANGCHAIN_PROJECT", "legacy-proj")
	tracer = NewLangChainTracer()
	if tracer == nil {
		t.Fatal("tracer nil for LANGCHAIN_ fallback env")
	}
	defer func() { _ = tracer.Close() }()
	if tracer.opts.APIKey != "legacy-key" || tracer.opts.Endpoint != "http://legacy-endpoint.test" || tracer.opts.Project != "legacy-proj" {
		t.Fatalf("legacy opts: %#v", tracer.opts)
	}
}

func TestLangSmithServer500BusinessNormalRetryOnceSilent(t *testing.T) {
	server := newLSServer("500")
	defer server.Close()
	var errsMu sync.Mutex
	var errs []error
	tracer := lsTracer(t, server.URL(), func(err error) {
		errsMu.Lock()
		defer errsMu.Unlock()
		errs = append(errs, err)
	})
	ctx := t.Context()
	runID := "dddddddd-0000-0000-0000-000000000001"

	start := lsEvent(callbacks.EventChainStart, "chain", runID, "", lsTS(30), func(e *callbacks.Event) { e.Input = "go" })
	end := lsEvent(callbacks.EventChainEnd, "chain", runID, "", lsTS(31), func(e *callbacks.Event) { e.Output = "GO" })
	if err := tracer.HandleEvent(ctx, start); err != nil {
		t.Fatalf("HandleEvent must never surface server errors: %v", err)
	}
	if err := tracer.HandleEvent(ctx, end); err != nil {
		t.Fatalf("HandleEvent must never surface server errors: %v", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = tracer.Flush(ctx) // must return (nil) rather than hang
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Flush hung on a failing server")
	}
	_ = tracer.Close()
	// Exactly two attempts for the one batch: initial + single retry.
	if got := server.count(); got != 2 {
		t.Fatalf("server saw %d requests, want 2 (one retry)", got)
	}
	errsMu.Lock()
	defer errsMu.Unlock()
	if len(errs) == 0 {
		t.Fatal("OnError was not invoked after retry exhaustion")
	}
	if !strings.Contains(errs[0].Error(), "500") {
		t.Fatalf("OnError error %v", errs[0])
	}
}

func TestLangSmithFlushDrainsAndIdleFlushIsQuiet(t *testing.T) {
	server := newLSServer("ok")
	defer server.Close()
	tracer := lsTracer(t, server.URL(), nil)
	ctx := t.Context()
	runID := "eeeeeeee-0000-0000-0000-000000000001"
	_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainStart, "chain", runID, "", lsTS(30)))
	if server.count() != 0 {
		t.Fatalf("premature flush: %d", server.count())
	}
	if err := tracer.Flush(ctx); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if got := server.count(); got != 1 {
		t.Fatalf("after flush: %d requests, want 1", got)
	}
	// A second flush with an empty queue must not POST again.
	if err := tracer.Flush(ctx); err != nil {
		t.Fatalf("second flush: %v", err)
	}
	if got := server.count(); got != 1 {
		t.Fatalf("idle flush posted again: %d", got)
	}
	_ = tracer.Close()
}

func TestLangSmithCloseIsIdempotentAndStopsAccepting(t *testing.T) {
	server := newLSServer("ok")
	defer server.Close()
	tracer := lsTracer(t, server.URL(), nil)
	ctx := t.Context()
	runID := "ffffffff-0000-0000-0000-000000000001"
	_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainStart, "chain", runID, "", lsTS(30)))
	if err := tracer.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if err := tracer.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	// Post-close events are dropped silently.
	_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainEnd, "chain", runID, "", lsTS(31)))
	if err := tracer.Flush(ctx); err != nil {
		t.Fatalf("flush after close: %v", err)
	}
	if got := server.count(); got != 1 {
		t.Fatalf("close flushed %d batches, want exactly 1 (the queued create)", got)
	}
}

func TestLangSmithBatchSizeFlushThreshold(t *testing.T) {
	server := newLSServer("ok")
	defer server.Close()
	tracer := NewLangChainTracerWithOptions(LangSmithOptions{
		Endpoint: server.URL(), APIKey: "k", Project: "p",
		BatchSize: 2, FlushEvery: time.Hour,
	})
	defer func() { _ = tracer.Close() }()
	ctx := t.Context()
	for i := range 2 {
		_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainStart, "chain",
			lsID(i), "", lsTS(30+i)))
	}
	deadline := time.Now().Add(2 * time.Second)
	for server.count() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if server.count() == 0 {
		t.Fatal("full batch (BatchSize=2) was not flushed without Flush()")
	}
	body := server.lastBody(t)
	if posts := lsRuns(t, body, "post"); len(posts) != 2 {
		t.Fatalf("threshold batch posts %d, want 2", len(posts))
	}
}

// lsID builds a deterministic uuid-shaped run id for index i.
func lsID(i int) string {
	digits := "0123456789abcdef"
	tail := strings.Repeat(string(digits[i%16]), 12)
	return "00000000-0000-0000-0000-" + tail
}

// TestLangSmithSnapshotOwnsCallerPayloads pins the eager-capture contract: a
// queued run snapshot must not hold references into the caller's input/output
// maps. The business caller legitimately reuses and mutates those maps as soon
// as Invoke returns, while the background flusher may not serialize for up to
// FlushEvery — sharing the reference is a data race (fatal under -race) and,
// sequenced, silently corrupts the trace with post-call values. Run under
// -race the pre-fix code reports the concurrent read/write; either way this
// test asserts the posted payloads carry the start-time / end-time values.
func TestLangSmithSnapshotOwnsCallerPayloads(t *testing.T) {
	server := newLSServer("ok")
	defer server.Close()
	tracer := lsTracer(t, server.URL(), nil)
	ctx := t.Context()
	runID := "12345678-0000-0000-0000-00000000000a"

	input := map[string]any{"q": "original"}
	_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainStart, "chain", runID, "", lsTS(30),
		func(e *callbacks.Event) { e.Input = input }))
	// Caller-side reuse immediately after the traced call returned.
	input["q"] = "mutated"
	input["late"] = true

	output := map[string]any{"a": "at-end"}
	_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainEnd, "chain", runID, "", lsTS(31),
		func(e *callbacks.Event) { e.Output = output }))
	output["a"] = "mutated"
	output["late"] = true

	if err := tracer.Flush(ctx); err != nil {
		t.Fatalf("flush: %v", err)
	}
	_ = tracer.Close()

	body := server.lastBody(t)
	posts := lsRuns(t, body, "post")
	inputs, ok := posts[0]["inputs"].(map[string]any)
	if !ok {
		t.Fatalf("post inputs: %#v", posts[0]["inputs"])
	}
	if inputs["q"] != "original" {
		t.Fatalf("inputs captured %v, want the start-time value %q", inputs["q"], "original")
	}
	if _, ok := inputs["late"]; ok {
		t.Fatal("inputs captured a mutation made after the traced call returned")
	}
	patches := lsRuns(t, body, "patch")
	outputs, ok := patches[0]["outputs"].(map[string]any)
	if !ok {
		t.Fatalf("patch outputs: %#v", patches[0]["outputs"])
	}
	if outputs["a"] != "at-end" || outputs["late"] != nil {
		t.Fatalf("outputs captured %v, want the end-time value", outputs)
	}
}

// TestLangSmithQueueOverflowAggregatesOnError pins the bounded-error contract
// of queue overflow: one overload episode must surface as a small number of
// aggregated OnError notifications (a per-op callback fired hundreds of times
// from an overloaded OnError observer would itself become the outage).
func TestLangSmithQueueOverflowAggregatesOnError(t *testing.T) {
	// A gated server pins the flusher inside its first POST so the 100-op
	// queue cannot drain while the test floods it.
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		<-release
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	var onErrorCount atomic.Int64
	tracer := lsTracer(t, srv.URL, func(error) { onErrorCount.Add(1) })
	ctx := t.Context()

	const flood = 600 // >> queue capacity (100) + one in-flight batch (64)
	for i := range flood {
		_ = tracer.HandleEvent(ctx, lsEvent(callbacks.EventChainStart, "chain",
			lsID(i%64)+fmt.Sprintf("-%04d", i), "", lsTS(30)))
	}
	close(release)
	if err := tracer.Flush(ctx); err != nil {
		t.Fatalf("flush: %v", err)
	}
	_ = tracer.Close()
	if n := onErrorCount.Load(); n < 1 {
		t.Fatal("queue overflow never reached OnError")
	} else if n > 2 {
		t.Fatalf("OnError fired %d times for one overload episode, want a bounded 1-2", n)
	}
}
