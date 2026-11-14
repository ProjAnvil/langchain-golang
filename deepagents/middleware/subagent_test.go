package middleware

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// staticModel replies with a fixed message on every call — the minimum
// language.ChatModel needed to compile and run a subagent (same shape as
// loopToolModel in langchain/agents/create_agent_parity_test.go).
type staticModel struct{ reply messages.Message }

func (m *staticModel) Invoke(context.Context, []messages.Message, ...runnables.Option) (messages.Message, error) {
	return m.reply, nil
}

func (m *staticModel) Batch(context.Context, [][]messages.Message, ...runnables.Option) ([]messages.Message, error) {
	return nil, errors.New("staticModel: batch not supported")
}

func (m *staticModel) Stream(context.Context, []messages.Message, ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return nil, errors.New("staticModel: streaming not supported")
}

func (m *staticModel) InputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *staticModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *staticModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) { return m, nil }

func (m *staticModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{}
}

// TestValidateSubAgents covers name validation over a mixed
// SubAgentSpec/*CompiledSubAgent list: unique non-empty names pass, duplicates
// and unknown types fail (UP _validate_unique_subagent_names,
// subagents.py:370-381).
func TestValidateSubAgents(t *testing.T) {
	tests := []struct {
		name    string
		specs   []any
		wantErr string // empty means want nil error
	}{
		{
			name: "unique mixed specs and compiled agents pass",
			specs: []any{
				SubAgentSpec{Name: "researcher", Description: "researches things"},
				&CompiledSubAgent{Name: "writer", Description: "writes things"},
				SubAgentSpec{Name: "reviewer", Description: "reviews things"},
			},
		},
		{
			name: "duplicate names across kinds rejected",
			specs: []any{
				SubAgentSpec{Name: "dup", Description: "first"},
				&CompiledSubAgent{Name: "dup", Description: "second"},
			},
			wantErr: "Duplicate subagent name 'dup'; each subagent must have a unique name.",
		},
		{
			name:    "duplicate names within specs rejected",
			specs:   []any{SubAgentSpec{Name: "a"}, SubAgentSpec{Name: "a"}},
			wantErr: "Duplicate subagent name 'a'; each subagent must have a unique name.",
		},
		{
			name:    "empty name rejected",
			specs:   []any{SubAgentSpec{Name: "", Description: "anonymous"}},
			wantErr: "must not be empty",
		},
		{
			name:    "unknown type rejected",
			specs:   []any{SubAgentSpec{Name: "a"}, 42},
			wantErr: "int",
		},
		{
			name:    "nil compiled subagent rejected",
			specs:   []any{(*CompiledSubAgent)(nil)},
			wantErr: "nil",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateSubAgents(tt.specs)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidateSubAgents() unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("ValidateSubAgents() = nil, want error containing %q", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ValidateSubAgents() error = %q, want substring %q", err.Error(), tt.wantErr)
			}
		})
	}
}

// TestCompileSubAgent checks the compile path: an agent is built with the
// spec's name/description/prompt/model, a missing model is an error, and the
// fallback-model entry point threads in a parent model only when the spec
// leaves Model unset (UP create_sub_agent, subagents.py:619-673).
func TestCompileSubAgent(t *testing.T) {
	model := &staticModel{reply: messages.AI("done")}

	compiled, err := CompileSubAgent(SubAgentSpec{
		Name:        "researcher",
		Description: "researches things",
		Prompt:      "You are a researcher.",
		Model:       model,
	})
	if err != nil {
		t.Fatalf("CompileSubAgent() error: %v", err)
	}
	if compiled.Name != "researcher" {
		t.Errorf("Name = %q, want %q", compiled.Name, "researcher")
	}
	if compiled.Description != "researches things" {
		t.Errorf("Description = %q, want %q", compiled.Description, "researches things")
	}
	if compiled.Agent == nil {
		t.Fatal("Agent = nil, want compiled agent")
	}
	// The compiled agent runs end to end: a tool-less final answer terminates
	// the loop and surfaces the model's reply.
	out, err := compiled.Agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("compiled agent Invoke() error: %v", err)
	}
	if len(out) == 0 || out[len(out)-1].Content != "done" {
		t.Errorf("final message = %v, want content %q", out, "done")
	}
}

func TestCompileSubAgentRequiresModel(t *testing.T) {
	_, err := CompileSubAgent(SubAgentSpec{Name: "orphan", Description: "no model"})
	if err == nil {
		t.Fatal("CompileSubAgent() with nil Model = nil error, want error")
	}
	if !strings.Contains(err.Error(), "must specify") {
		t.Errorf("error = %q, want it to mention the missing model", err.Error())
	}

	_, err = CompileSubAgentWithModel(SubAgentSpec{Name: "orphan", Description: "no model"}, nil)
	if err == nil {
		t.Fatal("CompileSubAgentWithModel() with nil Model and nil fallback = nil error, want error")
	}
}

// TestCompileSubAgentWithModelFallback verifies the fallback entry point used
// by SubAgentMiddleware when a spec inherits the parent's model: a set
// spec.Model wins over the fallback.
func TestCompileSubAgentWithModelFallback(t *testing.T) {
	fallback := &staticModel{reply: messages.AI("from fallback")}
	own := &staticModel{reply: messages.AI("own model")}

	compiled, err := CompileSubAgentWithModel(SubAgentSpec{Name: "child", Description: "inherits"}, fallback)
	if err != nil {
		t.Fatalf("CompileSubAgentWithModel() error: %v", err)
	}
	out, err := compiled.Agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("compiled agent Invoke() error: %v", err)
	}
	if got := out[len(out)-1].Content; got != "from fallback" {
		t.Errorf("fallback not used: final content = %q, want %q", got, "from fallback")
	}

	compiled, err = CompileSubAgentWithModel(SubAgentSpec{Name: "child", Description: "own model", Model: own}, fallback)
	if err != nil {
		t.Fatalf("CompileSubAgentWithModel() error: %v", err)
	}
	out, err = compiled.Agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("compiled agent Invoke() error: %v", err)
	}
	if got := out[len(out)-1].Content; got != "own model" {
		t.Errorf("spec model not used: final content = %q, want %q", got, "own model")
	}
}

// TestCompileSubAgentWithBackend verifies a spec carrying its own backend
// compiles: the FilesystemMiddleware built on it contributes tools without
// breaking agent construction.
func TestCompileSubAgentWithBackend(t *testing.T) {
	compiled, err := CompileSubAgent(SubAgentSpec{
		Name:        "fs-child",
		Description: "has a backend",
		Prompt:      "Use the filesystem.",
		Model:       &staticModel{reply: messages.AI("done")},
		Backend:     backends.NewStateBackend(),
	})
	if err != nil {
		t.Fatalf("CompileSubAgent() with backend error: %v", err)
	}
	if compiled.Agent == nil {
		t.Fatal("Agent = nil, want compiled agent")
	}
}

// TestCompileSubAgentInstallsModelRefusal asserts the default subagent stack
// installs ModelRefusalMiddleware. *agents.Agent does not expose its
// middleware list, so presence is asserted behaviorally: a model response
// carrying a refusal content block must surface *lcerrors.ModelRefusalError
// from a compiled subagent's Invoke (UP appends ModelRefusalMiddleware to
// every subagent stack unconditionally, graph.py:771 and :871).
func TestCompileSubAgentInstallsModelRefusal(t *testing.T) {
	refusal := messages.AI("")
	refusal.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "Cannot answer."}}
	compiled, err := CompileSubAgent(SubAgentSpec{
		Name:        "refusing",
		Description: "model refuses",
		Model:       &staticModel{reply: refusal},
	})
	if err != nil {
		t.Fatalf("CompileSubAgent() error: %v", err)
	}
	_, err = compiled.Agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if _, ok := errors.AsType[*lcerrors.ModelRefusalError](err); !ok {
		t.Fatalf("Invoke() error = %v, want *lcerrors.ModelRefusalError", err)
	}
}

// TestGeneralPurposeSubAgent pins the general-purpose spec's name, model
// passthrough, and the upstream text constants, which must match
// deepagents/middleware/subagents.py VERBATIM: DEFAULT_GENERAL_PURPOSE_DESCRIPTION
// (:564) and DEFAULT_SUBAGENT_PROMPT (:441-444), including the mid-sentence
// line break in the prompt's final sentence.
func TestGeneralPurposeSubAgent(t *testing.T) {
	model := &staticModel{reply: messages.AI("ok")}
	spec := GeneralPurposeSubAgent(model)

	if spec.Name != "general-purpose" {
		t.Errorf("Name = %q, want %q", spec.Name, "general-purpose")
	}
	if spec.Model != language.ChatModel(model) {
		t.Error("Model not passed through to spec")
	}

	desc := spec.Description
	if !strings.HasPrefix(desc, "General-purpose agent for researching complex questions") {
		t.Errorf("Description prefix = %q", desc[:min(50, len(desc))])
	}
	if !strings.HasSuffix(desc, "This agent has access to all tools as the main agent.") {
		t.Errorf("Description suffix = %q, want the upstream closing sentence", desc[max(0, len(desc)-60):])
	}
	if strings.Contains(desc, "\n") {
		t.Error("Description must be a single line, as upstream")
	}

	prompt := spec.Prompt
	if !strings.HasPrefix(prompt, "In order to complete the objective that the user asks of you, you have access to a number of standard tools.") {
		t.Errorf("Prompt first line = %q", strings.SplitN(prompt, "\n", 2)[0])
	}
	if !strings.Contains(prompt, "\n\n") {
		t.Error("Prompt must keep the upstream blank line between its two paragraphs")
	}
	wantTail := "The calling agent only sees your final assistant message, not your intermediate work, tool results, or status tracking. Ensure your final\nresponse contains the complete answer."
	if !strings.HasSuffix(prompt, wantTail) {
		t.Errorf("Prompt tail mismatch:\n got: %q\nwant: %q", prompt[max(0, len(prompt)-len(wantTail)-10):], wantTail)
	}
}

// TestBuildTaskInput covers the isolated-subagent state handoff (UP
// subagents.py:933-935): every parent key outside ExcludedStateKeys ∪
// PrivateStateKeys is shallow-copied, messages is rebuilt from the task
// description, files passes through, and the parent map is not mutated.
func TestBuildTaskInput(t *testing.T) {
	files := map[string]any{"a.txt": "alpha"}
	oldMessages := []messages.Message{messages.Human("parent history")}
	parent := map[string]any{
		"messages":                       oldMessages,
		"todos":                          []string{"parent todo"},
		"structured_response":            map[string]any{"x": 1},
		"skills_metadata":                map[string]any{},
		"pinned_skills":                  []string{},
		"_skill_tools_disclosed":         true,
		"_deepagents_forked_context":     true,
		"rubric":                         "grade me",
		agentsmw.EditedToolCallsStateKey: map[string]any{"pending": true},
		"files":                          files,
		"custom_key":                     42,
	}

	out := BuildTaskInput(parent, "do the thing")

	msgs, ok := out["messages"].([]messages.Message)
	if !ok || len(msgs) != 1 {
		t.Fatalf("out[\"messages\"] = %#v, want exactly one message", out["messages"])
	}
	if msgs[0].Role != messages.RoleHuman || msgs[0].Content != "do the thing" {
		t.Errorf("messages[0] = %#v, want human message %q", msgs[0], "do the thing")
	}

	for _, key := range slices.Concat(ExcludedStateKeys, PrivateStateKeys) {
		// "messages" is excluded from the copy but rebuilt from the task
		// description; its rebuilt value is asserted above.
		if key == "messages" {
			continue
		}
		if _, present := out[key]; present {
			t.Errorf("excluded key %q leaked into subagent input", key)
		}
	}
	if _, present := out[agentsmw.EditedToolCallsStateKey]; present {
		t.Errorf("private key %q leaked into subagent input", agentsmw.EditedToolCallsStateKey)
	}

	// files is public state and passes through by shallow reference (E2).
	gotFiles, ok := out["files"].(map[string]any)
	if !ok {
		t.Fatalf("out[\"files\"] = %#v, want the parent files map", out["files"])
	}
	if reflect.ValueOf(gotFiles).Pointer() != reflect.ValueOf(files).Pointer() {
		t.Error("files was deep-copied; want shallow pass-through of the same map")
	}
	if got, ok := out["custom_key"].(int); !ok || got != 42 {
		t.Errorf("out[\"custom_key\"] = %#v, want 42", out["custom_key"])
	}

	// The parent state is left untouched.
	if len(parent["messages"].([]messages.Message)) != len(oldMessages) {
		t.Error("BuildTaskInput mutated parent messages")
	}
	if len(parent) != 11 {
		t.Errorf("parent state key count changed: %d", len(parent))
	}
}

// TestBuildTaskInputNilState verifies a nil parent state is safe: the input is
// just the rebuilt messages list.
func TestBuildTaskInputNilState(t *testing.T) {
	out := BuildTaskInput(nil, "hello")
	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1 (messages only)", len(out))
	}
	msgs, ok := out["messages"].([]messages.Message)
	if !ok || len(msgs) != 1 || msgs[0].Content != "hello" {
		t.Fatalf("out[\"messages\"] = %#v, want single human message", out["messages"])
	}
}
