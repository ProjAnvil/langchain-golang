package agents

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/prompts"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// TestWithAgentNameStoresNameOnAgent verifies WithAgentName threads the name
// onto the returned Agent struct (Python's lc_agent_name equivalent).
func TestWithAgentNameStoresNameOnAgent(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("ok")}}

	agent, err := CreateAgent(model, nil, WithAgentName("support-bot"))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if agent.Name != "support-bot" {
		t.Fatalf("expected Agent.Name=%q, got %q", "support-bot", agent.Name)
	}
}

// nameReadingMiddleware records the run name it observes via NameFromContext
// inside BeforeModel, proving the Name is surfaced through the run context to
// middleware (since agentruntime/graph has no native run-metadata injection
// point, this context value is the surfaced channel).
type nameReadingMiddleware struct {
	got string
	ok  bool
}

func (m *nameReadingMiddleware) BeforeModel(ctx context.Context, _ map[string]any) (map[string]any, error) {
	m.got, m.ok = NameFromContext(ctx)
	return nil, nil
}

// TestWithAgentNameSurfacedViaContext verifies the Agent's Name reaches
// middleware through NameFromContext on each Invoke.
func TestWithAgentNameSurfacedViaContext(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("ok")}}
	reader := &nameReadingMiddleware{}

	agent, err := CreateAgent(model, nil,
		WithAgentName("tracer-agent"),
		WithAgentMiddleware(reader),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !reader.ok || reader.got != "tracer-agent" {
		t.Fatalf("expected NameFromContext to return %q, got (%q, %v)", "tracer-agent", reader.got, reader.ok)
	}
}

// TestWithAgentNameEmptyHasNoContextTag verifies that an unnamed agent does
// not pretend to carry a run-name tag.
func TestWithAgentNameEmptyHasNoContextTag(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("ok")}}
	reader := &nameReadingMiddleware{}

	agent, err := CreateAgent(model, nil, WithAgentMiddleware(reader))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if reader.ok {
		t.Fatalf("expected NameFromContext to be absent for an unnamed agent, got %q", reader.got)
	}
}

// TestWithAgentDebugEmitsVerboseLogs verifies WithAgentDebug(true) causes
// verbose slog output on the graph execution path (model node entry / model
// call / response), captured by swapping the slog default handler.
func TestWithAgentDebugEmitsVerboseLogs(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	defer slog.SetDefault(prev)
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))

	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}

	agent, err := CreateAgent(model, nil, WithAgentDebug(true))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	logs := buf.String()
	for _, want := range []string{"agents: invoke start", "agents: model node entry", "agents: model call", "agents: model response", "agents: invoke done"} {
		if !strings.Contains(logs, want) {
			t.Fatalf("debug log missing %q; got:\n%s", want, logs)
		}
	}
}

// TestWithAgentDebugOffIsSilent verifies the default (debug off) emits no
// agent debug logs.
func TestWithAgentDebugOffIsSilent(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	defer slog.SetDefault(prev)
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))

	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil) // debug off by default
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if strings.Contains(buf.String(), "agents:") {
		t.Fatalf("expected no agent debug logs when debug is off, got:\n%s", buf.String())
	}
}

// TestWithAgentDebugEmitsToolDispatchLogs verifies WithAgentDebug(true) logs a
// per-tool-dispatch line for each tool call.
func TestWithAgentDebugEmitsToolDispatchLogs(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	defer slog.SetDefault(prev)
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))

	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "echo", Args: map[string]any{"tool_input": "hi"}},
			},
		},
		messages.AI("done"),
	}}

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)}, WithAgentDebug(true))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	logs := buf.String()
	if !strings.Contains(logs, "agents: tools node entry") || !strings.Contains(logs, "agents: tool dispatch") {
		t.Fatalf("expected tools-node and tool-dispatch debug logs, got:\n%s", logs)
	}
}

// TestWithAgentSystemPromptTemplateRendered verifies a templated system prompt
// is rendered (with build-time variables) and prepended to the model call,
// mirroring Python's system_prompt: SystemMessage overload.
func TestWithAgentSystemPromptTemplateRendered(t *testing.T) {
	tmpl, err := prompts.NewPromptTemplate("sys", "You are a {{.role}} assistant.")
	if err != nil {
		t.Fatalf("new prompt template: %v", err)
	}
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}

	agent, err := CreateAgent(model, nil,
		WithAgentSystemPromptTemplate(&tmpl, map[string]any{"role": "helpful"}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(model.invocations) != 1 {
		t.Fatalf("expected 1 invocation, got %d", len(model.invocations))
	}
	invoked := model.invocations[0]
	if len(invoked) != 2 || invoked[0].Role != messages.RoleSystem || invoked[0].Content != "You are a helpful assistant." {
		t.Fatalf("expected rendered leading system message, got %#v", invoked)
	}
}

// TestWithAgentSystemPromptTemplatePerInvokeVars verifies per-Invoke template
// variables (via InvokeWithStateAndVars) override build-time variables.
func TestWithAgentSystemPromptTemplatePerInvokeVars(t *testing.T) {
	tmpl, err := prompts.NewPromptTemplate("sys", "You are a {{.role}} assistant.")
	if err != nil {
		t.Fatalf("new prompt template: %v", err)
	}
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}

	agent, err := CreateAgent(model, nil,
		WithAgentSystemPromptTemplate(&tmpl, map[string]any{"role": "default"}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.InvokeWithStateAndVars(
		context.Background(),
		[]messages.Message{messages.Human("hi")},
		map[string]any{"role": "per-invoke-override"},
	); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	invoked := model.invocations[0]
	if invoked[0].Content != "You are a per-invoke-override assistant." {
		t.Fatalf("expected per-Invoke variable to override build-time, got %q", invoked[0].Content)
	}
}

// TestWithAgentSystemPromptTemplateWinsOverLiteral verifies that when both a
// literal SystemPrompt and a SystemPromptTemplate are set, the template wins
// (documented behavior).
func TestWithAgentSystemPromptTemplateWinsOverLiteral(t *testing.T) {
	tmpl, err := prompts.NewPromptTemplate("sys", "template: {{.x}}")
	if err != nil {
		t.Fatalf("new prompt template: %v", err)
	}
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}

	agent, err := CreateAgent(model, nil,
		WithAgentSystemPrompt("literal-prompt"),
		WithAgentSystemPromptTemplate(&tmpl, map[string]any{"x": "rendered"}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	invoked := model.invocations[0]
	if invoked[0].Content != "template: rendered" {
		t.Fatalf("expected template to win over literal, got %q", invoked[0].Content)
	}
}

// TestWithAgentSystemPromptBackwardCompatible verifies the existing literal
// string path still works unchanged (backward compatibility is required).
func TestWithAgentSystemPromptBackwardCompatible(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}
	agent, err := CreateAgent(model, nil, WithAgentSystemPrompt("be nice"))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	invoked := model.invocations[0]
	if invoked[0].Role != messages.RoleSystem || invoked[0].Content != "be nice" {
		t.Fatalf("expected literal system prompt, got %#v", invoked)
	}
}

// TestWithAgentSystemPromptTemplateNilClears verifies passing a nil template
// to WithAgentSystemPromptTemplate clears a previously configured template,
// leaving the literal path (here empty).
func TestWithAgentSystemPromptTemplateNilClears(t *testing.T) {
	tmpl, err := prompts.NewPromptTemplate("sys", "template: {{.x}}")
	if err != nil {
		t.Fatalf("new prompt template: %v", err)
	}
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}

	agent, err := CreateAgent(model, nil,
		WithAgentSystemPromptTemplate(&tmpl, map[string]any{"x": "v"}),
		WithAgentSystemPromptTemplate(nil, nil),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if agent.systemPromptTemplate != nil {
		t.Fatalf("expected nil template after clearing, got %#v", agent.systemPromptTemplate)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	invoked := model.invocations[0]
	if len(invoked) != 1 || invoked[0].Role != messages.RoleHuman {
		t.Fatalf("expected no system message after clearing template, got %#v", invoked)
	}
}
