package mcp

// Elicitation tests: mid-call server input requests (modern multi
// round-trip) surface as LangGraph interrupts keyed by request key, resumed
// with accept/decline/cancel actions; outside a graph the call fails with a
// descriptive error instead of panicking.

import (
	"errors"
	"strings"
	"testing"

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
