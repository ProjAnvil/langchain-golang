package mcp

// Elicitation tests: mid-call server input requests (modern multi
// round-trip) surface as LangGraph interrupts keyed by request key, resumed
// with accept/decline/cancel actions; outside a graph the call fails with a
// descriptive error instead of panicking.

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// elicitGraph builds a one-node graph invoking tool inside a checkpointed
// run, returning the value of the "out" state key.
func elicitGraph(t *testing.T, tool *Tool) (*graph.CompiledGraph, string) {
	t.Helper()
	saver := checkpoint.NewMemorySaver()
	g := graph.NewStateGraph()
	g.AddNode("call", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		res, err := tool.Invoke(ctx, map[string]any{})
		if err != nil {
			return nil, err
		}
		return map[string]any{"out": res.Content}, nil
	})
	g.AddEdge(types.START, "call")
	g.AddEdge("call", types.END)
	cg, err := g.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg, "t-" + t.Name()
}

func decodeRequests(t *testing.T, value any) []map[string]any {
	t.Helper()
	m, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("interrupt value = %T, want map", value)
	}
	requests, ok := m["requests"].([]any)
	if !ok || len(requests) == 0 {
		t.Fatalf("interrupt value requests = %#v", m["requests"])
	}
	out := make([]map[string]any, 0, len(requests))
	for _, r := range requests {
		rm, ok := r.(map[string]any)
		if !ok {
			t.Fatalf("request = %T, want map", r)
		}
		out = append(out, rm)
	}
	return out
}

// TestElicitationInterruptAccept: the first run pauses with an interrupt
// carrying the request under its key; resuming with an accept answer (plus
// content matching the requested schema) completes the call, and the server
// receives both the action and the echoed request state.
func TestElicitationInterruptAccept(t *testing.T) {
	cg, thread := elicitGraph(t, loadOne(t, "srv", "srv_elicit"))

	paused, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %#v, want 1", paused.Interrupts)
	}
	requests := decodeRequests(t, paused.Interrupts[0].Value)
	if requests[0]["key"] != "confirm" {
		t.Fatalf("request key = %v, want confirm", requests[0]["key"])
	}
	if requests[0]["message"] != "Please confirm the transfer" {
		t.Fatalf("request message = %v", requests[0]["message"])
	}
	schema, ok := requests[0]["requested_schema"].(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("requested_schema = %#v", requests[0]["requested_schema"])
	}

	// The wire-map payload form must be nested under the interrupt's NS:
	// a bare map[string]any Resume is NS/ID addressing in the graph.
	res, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{
		ThreadID: thread,
		Resume: map[string]any{paused.Interrupts[0].NS: map[string]any{"responses": map[string]any{
			"confirm": map[string]any{"action": "accept", "content": map[string]any{"amount": 42}},
		}}},
	})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	out, _ := res.Values["out"].(string)
	if !strings.Contains(out, "action=accept") {
		t.Fatalf("out = %q, want accept action delivered to the server", out)
	}
	if !strings.Contains(out, "42") {
		t.Fatalf("out = %q, want elicited content delivered to the server", out)
	}
	if !strings.Contains(out, "state=elicit-state-1") {
		t.Fatalf("out = %q, want request state echoed to the server", out)
	}
}

// TestElicitationInterruptDecline: declining completes the call with the
// decline action visible to the server.
func TestElicitationInterruptDecline(t *testing.T) {
	cg, thread := elicitGraph(t, loadOne(t, "srv", "srv_elicit"))

	paused, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %#v", paused.Interrupts)
	}
	res, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{
		ThreadID: thread,
		Resume: map[string]ElicitationAnswer{
			"confirm": {Action: ElicitationDecline},
		},
	})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	out, _ := res.Values["out"].(string)
	if !strings.Contains(out, "action=decline") {
		t.Fatalf("out = %q, want decline action delivered", out)
	}
}

// TestElicitationInterruptCancel: a cancel answer abandons the whole call:
// the tool invocation fails with *ElicitationCanceledError, which the graph
// surfaces as the node error.
func TestElicitationInterruptCancel(t *testing.T) {
	cg, thread := elicitGraph(t, loadOne(t, "srv", "srv_elicit"))

	paused, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %#v", paused.Interrupts)
	}
	_, err = cg.InvokeWithOptions(t.Context(), nil, graph.Options{
		ThreadID: thread,
		Resume: ElicitationResponses{Responses: map[string]ElicitationAnswer{
			"confirm": {Action: ElicitationCancel},
		}},
	})
	if err == nil {
		t.Fatal("cancel must abandon the call with an error")
	}
	if !errors.Is(err, ErrElicitationCanceled) {
		t.Fatalf("error = %v, want ErrElicitationCanceled", err)
	}
}

// TestElicitationTypedResume: the typed ElicitationResponses resume form
// works like the wire-map form.
func TestElicitationTypedResume(t *testing.T) {
	cg, thread := elicitGraph(t, loadOne(t, "srv", "srv_elicit"))

	paused, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %#v", paused.Interrupts)
	}
	res, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{
		ThreadID: thread,
		Resume: ElicitationResponses{Responses: map[string]ElicitationAnswer{
			"confirm": {Action: ElicitationAccept, Content: map[string]any{"amount": 7}},
		}},
	})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	out, _ := res.Values["out"].(string)
	if !strings.Contains(out, "action=accept") || !strings.Contains(out, "7") {
		t.Fatalf("out = %q", out)
	}
}

// TestElicitationOutsideGraphFails: invoking an eliciting tool outside a
// graph node (no interrupt machinery, no checkpointer) fails with a
// descriptive error rather than panicking.
func TestElicitationOutsideGraphFails(t *testing.T) {
	tool := loadOne(t, "srv", "srv_elicit")
	_, err := tool.Invoke(t.Context(), map[string]any{})
	if err == nil {
		t.Fatal("elicitation outside a graph must fail")
	}
	if !strings.Contains(err.Error(), "graph") {
		t.Fatalf("error = %v, want a message pointing at the graph/checkpointer requirement", err)
	}
}

// TestElicitationOverHTTPTransport: the interrupt bridge is
// transport-agnostic; over a streamable-HTTP connection the pause/resume
// round trip works identically.
func TestElicitationOverHTTPTransport(t *testing.T) {
	ts := newStreamableHTTPServer(t, newTestMCPServer("remote"))
	adapter, err := New(t.Context(), MCPConfig{
		Servers: map[string]Server{
			"remote": &HTTPServer{URL: ts.URL + "/mcp"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer adapter.Close()

	tools, err := adapter.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	var tool *Tool
	for _, tl := range tools {
		if tl.Name() == "remote_elicit" {
			tool = tl.(*Tool)
		}
	}
	if tool == nil {
		t.Fatal("remote_elicit not loaded")
	}

	cg, thread := elicitGraph(t, tool)
	paused, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %#v", paused.Interrupts)
	}
	res, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{
		ThreadID: thread,
		Resume: ElicitationResponses{Responses: map[string]ElicitationAnswer{
			"confirm": {Action: ElicitationAccept, Content: map[string]any{"amount": 1}},
		}},
	})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	out, _ := res.Values["out"].(string)
	if !strings.Contains(out, "action=accept") {
		t.Fatalf("out = %q", out)
	}
}

// TestDecodeElicitationResponsesForms: every accepted resume form decodes —
// the typed value/pointer/map forms Go callers pass directly and the wire-map
// spellings (snake_case and capitalized) JSON checkpoints leave behind — and
// the malformed shapes fail with descriptive errors.
func TestDecodeElicitationResponsesForms(t *testing.T) {
	typed := ElicitationResponses{Responses: map[string]ElicitationAnswer{
		"k": {Action: ElicitationAccept, Content: map[string]any{"n": 1.0}},
	}}
	accept := map[string]ElicitationAnswer{
		"k": {Action: ElicitationAccept, Content: map[string]any{"n": 1.0}},
	}

	cases := []struct {
		name    string
		value   any
		want    map[string]ElicitationAnswer
		wantErr string
	}{
		{name: "nil", value: nil, wantErr: "resume value is required"},
		{name: "typed value", value: typed, want: accept},
		{name: "typed pointer", value: &typed, want: accept},
		{name: "typed nil pointer", value: (*ElicitationResponses)(nil), wantErr: "resume value is required"},
		{
			name:  "answer map",
			value: map[string]ElicitationAnswer{"k": {Action: ElicitationDecline}},
			want:  map[string]ElicitationAnswer{"k": {Action: ElicitationDecline}},
		},
		{
			name: "wire map snake_case",
			value: map[string]any{"responses": map[string]any{
				"k": map[string]any{"action": "accept", "content": map[string]any{"n": 1.0}},
			}},
			want: accept,
		},
		{
			name: "wire map capitalized",
			value: map[string]any{"Responses": map[string]any{
				"k": map[string]any{"Action": "decline"},
			}},
			want: map[string]ElicitationAnswer{"k": {Action: ElicitationDecline}},
		},
		{
			name:    "wire map without responses key",
			value:   map[string]any{"answer": "yes"},
			wantErr: `without a "responses" key`,
		},
		{
			name:    "wire map with non-object responses",
			value:   map[string]any{"responses": []string{"accept"}},
			wantErr: `"responses" must be an object`,
		},
		{
			name: "wire map with bad inner answer",
			value: map[string]any{"responses": map[string]any{
				"k": map[string]any{"action": "maybe"},
			}},
			wantErr: `answer "k"`,
		},
		{name: "unsupported type", value: 42, wantErr: "cannot decode elicitation resume from int"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DecodeElicitationResponses(tc.value)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("DecodeElicitationResponses(%v) err = %v, want %q", tc.value, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("DecodeElicitationResponses(%v) error = %v", tc.value, err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("decoded = %#v, want %#v", got, tc.want)
			}
			for key, want := range tc.want {
				if !reflect.DeepEqual(got[key], want) {
					t.Fatalf("decoded[%q] = %#v, want %#v", key, got[key], want)
				}
			}
		})
	}
}

// TestDecodeElicitationAnswerForms: a single wire answer decodes its action
// from either spelling, carries the content when present, and rejects the
// malformed shapes.
func TestDecodeElicitationAnswerForms(t *testing.T) {
	cases := []struct {
		name    string
		value   any
		want    ElicitationAnswer
		wantErr string
	}{
		{name: "typed answer", value: ElicitationAnswer{Action: ElicitationCancel}, want: ElicitationAnswer{Action: ElicitationCancel}},
		{
			name:  "snake_case with content",
			value: map[string]any{"action": "accept", "content": map[string]any{"amount": 7.0}},
			want:  ElicitationAnswer{Action: ElicitationAccept, Content: map[string]any{"amount": 7.0}},
		},
		{
			name:  "capitalized without content",
			value: map[string]any{"Action": "decline"},
			want:  ElicitationAnswer{Action: ElicitationDecline},
		},
		{name: "missing action key", value: map[string]any{}, wantErr: `missing "action" key`},
		{
			name:  "capitalized action and content keys",
			value: map[string]any{"Action": "cancel", "Content": map[string]any{"why": "too late"}},
			want:  ElicitationAnswer{Action: ElicitationCancel, Content: map[string]any{"why": "too late"}},
		},
		{name: "non-string action", value: map[string]any{"action": 3}, wantErr: `"action" must be a string`},
		{name: "unknown action", value: map[string]any{"action": "maybe"}, wantErr: `unknown action "maybe"`},
		{name: "unsupported type", value: "accept", wantErr: "cannot decode from string"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := decodeElicitationAnswer(tc.value)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("decodeElicitationAnswer(%v) err = %v, want %q", tc.value, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("decodeElicitationAnswer(%v) error = %v", tc.value, err)
			}
			if got.Action != tc.want.Action || got.Content == nil != (tc.want.Content == nil) {
				t.Fatalf("decoded = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// elicitBridgeGraph builds a one-node graph that answers one elicitation
// request directly through the client bridge (the legacy server-initiated
// dispatch shape), recording the delivered action in the "action" state key.
func elicitBridgeGraph(t *testing.T, params mcp.ElicitationParams) (*graph.CompiledGraph, string) {
	t.Helper()
	saver := checkpoint.NewMemorySaver()
	g := graph.NewStateGraph()
	g.AddNode("elicit", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		result, err := elicitationBridge{}.Elicit(ctx, mcp.ElicitationRequest{Params: params})
		if err != nil {
			return nil, err
		}
		return map[string]any{"action": string(result.Action), "content": result.Content}, nil
	})
	g.AddEdge(types.START, "elicit")
	g.AddEdge("elicit", types.END)
	cg, err := g.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg, "t-" + t.Name()
}

// runBridgeInterrupt pauses the bridge graph once and returns the paused run
// plus its thread id.
func runBridgeInterrupt(t *testing.T, cg *graph.CompiledGraph, thread string) {
	t.Helper()
	paused, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %#v, want 1", paused.Interrupts)
	}
}

// TestElicitBridgeLegacyResumeAccept: a server-initiated elicitation request
// bridged on the node's own goroutine pauses as an interrupt keyed by the
// request id; an accept resume reaches the server as the elicitation result.
func TestElicitBridgeLegacyResumeAccept(t *testing.T) {
	params := mcp.ElicitationParams{
		Message:       "Approve the payment?",
		ElicitationID: "pay",
		Mode:          "form",
		URL:           "https://bank.example/form",
		RequestedSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"amount": map[string]any{"type": "number"}},
		},
	}
	cg, thread := elicitBridgeGraph(t, params)
	runBridgeInterrupt(t, cg, thread)

	// The interrupt payload carries the full request shape.
	cg2, thread2 := elicitBridgeGraph(t, params)
	paused, err := cg2.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread2})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	requests := decodeRequests(t, paused.Interrupts[0].Value)
	if requests[0]["key"] != "pay" || requests[0]["mode"] != "form" ||
		requests[0]["url"] != "https://bank.example/form" || requests[0]["message"] != "Approve the payment?" {
		t.Fatalf("interrupt request = %#v", requests[0])
	}

	res, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{
		ThreadID: thread,
		Resume: ElicitationResponses{Responses: map[string]ElicitationAnswer{
			"pay": {Action: ElicitationAccept, Content: map[string]any{"amount": 9.0}},
		}},
	})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if res.Values["action"] != "accept" {
		t.Fatalf("action = %v, want accept", res.Values["action"])
	}
}

// TestElicitBridgeLegacyDecodeFailure: a resume value that cannot decode as
// elicitation responses fails the node.
func TestElicitBridgeLegacyDecodeFailure(t *testing.T) {
	cg, thread := elicitBridgeGraph(t, mcp.ElicitationParams{ElicitationID: "pay"})
	runBridgeInterrupt(t, cg, thread)
	_, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread, Resume: 42})
	if err == nil || !strings.Contains(err.Error(), "cannot decode elicitation resume") {
		t.Fatalf("resume with undecodable value err = %v, want decode failure", err)
	}
}

// TestElicitBridgeLegacyMissingAnswer: a decoded resume that carries no
// answer under the request key fails the node.
func TestElicitBridgeLegacyMissingAnswer(t *testing.T) {
	cg, thread := elicitBridgeGraph(t, mcp.ElicitationParams{ElicitationID: "pay"})
	runBridgeInterrupt(t, cg, thread)
	_, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{
		ThreadID: thread,
		Resume:   ElicitationResponses{Responses: map[string]ElicitationAnswer{"other": {Action: ElicitationAccept}}},
	})
	if err == nil || !strings.Contains(err.Error(), `missing answer for key "pay"`) {
		t.Fatalf("resume without key err = %v, want missing-answer failure", err)
	}
}

// TestElicitBridgeLegacyCancel: a cancel answer abandons the request with
// ErrElicitationCanceled.
func TestElicitBridgeLegacyCancel(t *testing.T) {
	cg, thread := elicitBridgeGraph(t, mcp.ElicitationParams{ElicitationID: "pay"})
	runBridgeInterrupt(t, cg, thread)
	_, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{
		ThreadID: thread,
		Resume:   ElicitationResponses{Responses: map[string]ElicitationAnswer{"pay": {Action: ElicitationCancel}}},
	})
	if !errors.Is(err, ErrElicitationCanceled) {
		t.Fatalf("cancel resume err = %v, want ErrElicitationCanceled", err)
	}
}

// TestElicitBatchWaitDerivedKey: a request without an id is keyed by a stable
// per-batch counter, and its delivered answer reaches the waiting request.
func TestElicitBatchWaitDerivedKey(t *testing.T) {
	batch := newElicitBatch()
	waited := make(chan struct{})
	var result *mcp.ElicitationResult
	var waitErr error
	go func() {
		defer close(waited)
		result, waitErr = batch.wait(t.Context(), mcp.ElicitationParams{Message: "id-less"})
	}()
	<-batch.arrived

	pending := batch.snapshot()
	if len(pending) != 1 || pending[0].key != "request_1" {
		t.Fatalf("pending = %#v, want request_1", pending)
	}
	if batch.deliver(map[string]ElicitationAnswer{"request_1": {Action: ElicitationDecline}}) {
		t.Fatal("decline must not cancel the batch")
	}
	<-waited
	if waitErr != nil {
		t.Fatalf("wait() error = %v", waitErr)
	}
	if result == nil || result.Action != mcp.ElicitationResponseAction(ElicitationDecline) {
		t.Fatalf("wait() result = %#v, want decline", result)
	}
	if batch.snapshot() != nil && len(batch.snapshot()) != 0 {
		t.Fatalf("answered request must leave the pending set: %#v", batch.snapshot())
	}
}

// TestElicitBatchDuplicateKey: two concurrent requests under one key fail the
// second registrant; the first still receives its answer afterwards.
func TestElicitBatchDuplicateKey(t *testing.T) {
	batch := newElicitBatch()
	params := mcp.ElicitationParams{ElicitationID: "same"}
	first := make(chan error, 1)
	go func() {
		_, err := batch.wait(t.Context(), params)
		first <- err
	}()
	<-batch.arrived

	if _, err := batch.wait(t.Context(), params); err == nil || !strings.Contains(err.Error(), `duplicate elicitation key "same"`) {
		t.Fatalf("second wait() err = %v, want duplicate-key error", err)
	}
	batch.deliver(map[string]ElicitationAnswer{"same": {Action: ElicitationAccept}})
	if err := <-first; err != nil {
		t.Fatalf("first wait() error = %v", err)
	}
}

// TestElicitBatchWaitContextCanceled: wait returns the caller's context error
// when its context ends before an answer arrives.
func TestElicitBatchWaitContextCanceled(t *testing.T) {
	batch := newElicitBatch()
	ctx, cancel := context.WithCancel(t.Context())
	go func() {
		<-batch.arrived
		cancel()
	}()
	_, err := batch.wait(ctx, mcp.ElicitationParams{ElicitationID: "late"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("wait(canceled ctx) err = %v, want context.Canceled", err)
	}
}

// TestElicitBatchDeliverCancelAndAbort: a cancel answer marks the batch
// canceled and releases every waiter via the done channel; abort releases
// waiters without marking cancellation.
func TestElicitBatchDeliverCancelAndAbort(t *testing.T) {
	canceled := newElicitBatch()
	if !canceled.deliver(map[string]ElicitationAnswer{"k": {Action: ElicitationCancel}}) {
		t.Fatal("cancel answer must report the call canceled")
	}
	if !canceled.isCanceled() {
		t.Fatal("isCanceled() = false after a cancel answer")
	}
	// A second wait on the canceled batch returns the abandonment error.
	if _, err := canceled.wait(t.Context(), mcp.ElicitationParams{ElicitationID: "k"}); err == nil ||
		!strings.Contains(err.Error(), "abandoned") {
		t.Fatalf("wait() after cancel err = %v, want abandonment error", err)
	}

	aborted := newElicitBatch()
	aborted.abort()
	if aborted.isCanceled() {
		t.Fatal("abort() must not mark the batch canceled")
	}
	if _, err := aborted.wait(t.Context(), mcp.ElicitationParams{ElicitationID: "k"}); err == nil ||
		!strings.Contains(err.Error(), "abandoned") {
		t.Fatalf("wait() after abort err = %v, want abandonment error", err)
	}
}

// TestElicitInterruptValueFields: the interrupt payload renders every
// optional request field and names the tool and server when known.
func TestElicitInterruptValueFields(t *testing.T) {
	value := elicitInterruptValue(&Tool{name: "srv_confirm", serverKey: "srv"}, []pendingElicitation{{
		key: "confirm",
		params: mcp.ElicitationParams{
			Message:         "Proceed?",
			Mode:            "form",
			URL:             "https://example.test/form",
			RequestedSchema: map[string]any{"type": "object"},
		},
	}})
	if value["tool"] != "srv_confirm" || value["server"] != "srv" {
		t.Fatalf("tool/server = %v/%v", value["tool"], value["server"])
	}
	requests, _ := value["requests"].([]any)
	if len(requests) != 1 {
		t.Fatalf("requests = %#v", value["requests"])
	}
	request, _ := requests[0].(map[string]any)
	for _, key := range []string{"key", "message", "mode", "url", "requested_schema"} {
		if _, ok := request[key]; !ok {
			t.Fatalf("request missing %q: %#v", key, request)
		}
	}

	// Without a tool the payload stays request-only, and empty optional
	// fields are omitted.
	bare := elicitInterruptValue(nil, []pendingElicitation{{key: "k", params: mcp.ElicitationParams{Message: "m"}}})
	if _, ok := bare["tool"]; ok {
		t.Fatalf("tool leaked into bare payload: %#v", bare)
	}
	bareRequests, _ := bare["requests"].([]any)
	bareRequest, _ := bareRequests[0].(map[string]any)
	if _, ok := bareRequest["mode"]; ok {
		t.Fatalf("empty mode must be omitted: %#v", bareRequest)
	}
	if _, ok := bareRequest["requested_schema"]; ok {
		t.Fatalf("nil schema must be omitted: %#v", bareRequest)
	}
}

// TestElicitationResumeUndecodable: resuming an MRTR elicitation interrupt
// with a payload that cannot decode as elicitation responses fails the tool
// call (and the node) with the decode error.
func TestElicitationResumeUndecodable(t *testing.T) {
	cg, thread := elicitGraph(t, loadOne(t, "srv", "srv_elicit"))

	paused, err := cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread})
	if err != nil {
		t.Fatalf("run1 error = %v", err)
	}
	if len(paused.Interrupts) != 1 {
		t.Fatalf("run1 Interrupts = %#v, want 1", paused.Interrupts)
	}
	_, err = cg.InvokeWithOptions(t.Context(), nil, graph.Options{ThreadID: thread, Resume: 42})
	if err == nil || !strings.Contains(err.Error(), "cannot decode elicitation resume") {
		t.Fatalf("resume with undecodable payload err = %v, want decode failure", err)
	}
}
