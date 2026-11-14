package middleware

// Task 8: the `task` tool and SubAgentMiddleware. The e2e tests drive a stub
// subagent through CreateAgent with a scripted model (same double as
// permissions_test.go) plus two Artifact-Command tools — one writing a custom
// state key, one writing a file through a StateBackend view — and pin the
// brief's behaviors (a)-(g): final-text ToolMessage, parent-state writeback,
// writeback filtering (value equality), file pass-through, unknown-agent error
// text, ThreadID derivation, and the files-channel fold semantics.

import (
	"context"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	agents "github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// --- fixtures -------------------------------------------------------------

// taskToolCall builds the parent model's `task` tool call for the scripts.
func taskToolCall(id, subagentType, description string) messages.ToolCall {
	return messages.ToolCall{ID: id, Name: TaskToolName, Args: map[string]any{
		"subagent_type": subagentType,
		"description":   description,
	}}
}

// newCustomKeyTool builds the stub subagent's custom-state-key tool: it
// returns a Command carrying a "custom_key" update in its artifact, the same
// convention write_todos uses (see NewWriteTodosTool).
func newCustomKeyTool(t *testing.T) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewFunc(
		"set_flag",
		"Sets the custom_key state field.",
		schema.Object(map[string]schema.Schema{
			"value": schema.String("the new value"),
		}, "value"),
		func(_ context.Context, input map[string]any) (coretools.Result, error) {
			value, _ := input["value"].(string)
			return coretools.Result{
				Content: "flag set",
				Artifact: &types.Command{
					Update: map[string]any{"custom_key": value},
				},
			}, nil
		},
	)
	if err != nil {
		t.Fatalf("NewFunc(set_flag): %v", err)
	}
	return tool
}

// newHelperSubAgentMiddleware builds a SubAgentMiddleware around one "helper"
// spec whose model is its own scripted double and whose only tool is the
// custom-key tool.
func newHelperSubAgentMiddleware(t *testing.T, parentModel, subModel *scriptedChatModel) *SubAgentMiddleware {
	t.Helper()
	mw, err := NewSubAgentMiddleware(parentModel, []any{SubAgentSpec{
		Name:        "helper",
		Description: "Helper subagent.",
		Model:       subModel,
		Tools:       []coretools.Tool{newCustomKeyTool(t)},
	}})
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}
	return mw
}

// taskResultMessage finds the ToolMessage answering the given tool call.
func taskResultMessage(t *testing.T, values map[string]any, toolCallID string) messages.Message {
	t.Helper()
	msgs, _ := values["messages"].([]messages.Message)
	for _, msg := range msgs {
		if msg.Role == messages.RoleTool && msg.ToolCallID == toolCallID {
			return msg
		}
	}
	t.Fatalf("no ToolMessage for tool call %q in final messages: %#v", toolCallID, msgs)
	return messages.Message{}
}

// --- NewSubAgentMiddleware construction ------------------------------------

// TestNewSubAgentMiddlewareCompilesAndAutoAddsGeneralPurpose pins the
// constructor: specs compile (nil-Model specs inherit the parent model,
// explicit models win), *CompiledSubAgent entries pass through, and the
// general-purpose subagent is auto-added FIRST when no spec claims the name
// (UP graph.py:854, 918 insert the default spec at index 0) — and is not
// duplicated when the caller supplies one.
func TestNewSubAgentMiddlewareCompilesAndAutoAddsGeneralPurpose(t *testing.T) {
	parentModel := &staticModel{reply: messages.AI("parent")}
	ownModel := &staticModel{reply: messages.AI("own")}

	mw, err := NewSubAgentMiddleware(parentModel, []any{
		SubAgentSpec{Name: "inherits", Description: "inherits the parent model"},
		SubAgentSpec{Name: "explicit", Description: "own model", Model: ownModel},
	})
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}
	if len(mw.SubAgents) != 3 {
		t.Fatalf("len(SubAgents) = %d, want 3 (helper specs + auto-added general-purpose): %#v", len(mw.SubAgents), mw.SubAgents)
	}
	// Auto-added general-purpose is first, mirroring insert(0, ...).
	if mw.SubAgents[0].Name != "general-purpose" {
		t.Errorf("SubAgents[0].Name = %q, want general-purpose", mw.SubAgents[0].Name)
	}
	if mw.SubAgents[1].Name != "inherits" || mw.SubAgents[2].Name != "explicit" {
		t.Errorf("SubAgents order = [%s %s], want [inherits explicit]", mw.SubAgents[1].Name, mw.SubAgents[2].Name)
	}
	for i, s := range mw.SubAgents {
		if s.Agent == nil {
			t.Errorf("SubAgents[%d].Agent is nil", i)
		}
	}

	// A caller-supplied general-purpose spec is not duplicated.
	mw2, err := NewSubAgentMiddleware(parentModel, []any{
		SubAgentSpec{Name: "general-purpose", Description: "mine", Model: ownModel},
	})
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware(custom gp): %v", err)
	}
	if len(mw2.SubAgents) != 1 || mw2.SubAgents[0].Description != "mine" {
		t.Fatalf("custom general-purpose must not be duplicated or replaced: %#v", mw2.SubAgents)
	}

	// *CompiledSubAgent entries pass through untouched.
	prebuilt, err := CompileSubAgent(GeneralPurposeSubAgent(ownModel))
	if err != nil {
		t.Fatalf("CompileSubAgent: %v", err)
	}
	mw3, err := NewSubAgentMiddleware(parentModel, []any{prebuilt})
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware(compiled): %v", err)
	}
	if len(mw3.SubAgents) != 1 || mw3.SubAgents[0].Agent != prebuilt.Agent {
		t.Fatalf("compiled subagent must pass through by identity: %#v", mw3.SubAgents)
	}
}

// TestNewSubAgentMiddlewareErrors: invalid spec lists reject (delegated to
// ValidateSubAgents), and a nil parent model cannot compile a nil-Model spec
// (nor the auto-added general-purpose subagent).
func TestNewSubAgentMiddlewareErrors(t *testing.T) {
	if _, err := NewSubAgentMiddleware(&staticModel{reply: messages.AI("x")}, []any{
		SubAgentSpec{Name: "a", Description: "a"},
		SubAgentSpec{Name: "a", Description: "dup"},
	}); err == nil || !strings.Contains(err.Error(), "Duplicate subagent name 'a'") {
		t.Fatalf("duplicate names must error, got %v", err)
	}
	if _, err := NewSubAgentMiddleware(nil, []any{SubAgentSpec{Name: "a", Description: "a"}}); err == nil ||
		!strings.Contains(err.Error(), "SubAgent 'a' must specify 'model'") {
		t.Fatalf("nil model + nil-Model spec must error with the missing-model text, got %v", err)
	}
}

// --- (a)+(b) e2e: task returns the subagent's final text, state writeback ---

// TestTaskToolReturnsSubagentFinalTextAndWritesState drives the full loop:
// the parent delegates via `task`, the stub subagent runs its custom-key tool
// and finishes with a text answer, and the parent receives (a) that final text
// as the task ToolMessage content and (b) the custom_key state writeback.
func TestTaskToolReturnsSubagentFinalTextAndWritesState(t *testing.T) {
	parentModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			taskToolCall("call_1", "helper", "set the flag to hello"),
		}},
		messages.AI("parent done"),
	}}
	subModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			{ID: "sub_1", Name: "set_flag", Args: map[string]any{"value": "hello"}},
		}},
		messages.AI("subagent final answer"),
	}}
	mw := newHelperSubAgentMiddleware(t, parentModel, subModel)

	agent, err := agents.CreateAgent(parentModel, nil, agents.WithAgentMiddleware(mw))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("delegate")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}

	// (a) the ToolMessage content is the subagent's final assistant text.
	if got := taskResultMessage(t, values, "call_1").Content; got != "subagent final answer" {
		t.Fatalf("task ToolMessage content = %q, want %q", got, "subagent final answer")
	}

	// (b) the parent state gains the subagent's custom key.
	if got, _ := values["custom_key"].(string); got != "hello" {
		t.Fatalf("values[custom_key] = %#v, want %q (full values: %#v)", values["custom_key"], "hello", values)
	}
}

// --- (c) writeback filter ---------------------------------------------------

// TestFilteredStateUpdate pins the Command Update filter: excluded and private
// keys never pass, unchanged keys are dropped by VALUE equality (a fresh,
// deeply-equal value is still unchanged — never a reference comparison), and
// genuinely changed keys (including files) survive.
func TestFilteredStateUpdate(t *testing.T) {
	input := map[string]any{
		"messages": []messages.Message{messages.Human("do it")},
		"files":    map[string]backends.FileMeta{"a.md": {Content: "old"}},
		"todos":    []string{"parent"},
		"same":     map[string]int{"k": 1},
		"custom":   "inherited",
	}
	result := map[string]any{
		"messages":                       []messages.Message{messages.Human("do it"), messages.AI("done")},
		"files":                          map[string]backends.FileMeta{"a.md": {Content: "new"}},
		"todos":                          []string{"child"},
		"same":                           map[string]int{"k": 1}, // equal value, different reference
		"custom":                         "rewritten",
		"brand_new":                      42,
		"_skill_tools_disclosed":         true,
		agentsmw.EditedToolCallsStateKey: map[string]any{"pending": true},
	}

	update := filteredStateUpdate(result, input)

	if _, ok := update["messages"]; ok {
		t.Error("update must never carry a messages key")
	}
	for _, key := range ExcludedStateKeys {
		if _, ok := update[key]; ok {
			t.Errorf("excluded key %q leaked into the update", key)
		}
	}
	for _, key := range PrivateStateKeys {
		if _, ok := update[key]; ok {
			t.Errorf("private key %q leaked into the update", key)
		}
	}
	if _, ok := update["same"]; ok {
		t.Error("unchanged-value key must be filtered (value equality, not reference)")
	}
	wantKeys := []string{"files", "custom", "brand_new"}
	if len(update) != len(wantKeys) {
		t.Fatalf("update keys = %v, want exactly %v", keysOf(update), wantKeys)
	}
	for _, key := range wantKeys {
		if _, ok := update[key]; !ok {
			t.Errorf("update is missing changed key %q (have %v)", key, keysOf(update))
		}
	}
	if files, _ := update["files"].(map[string]backends.FileMeta); files["a.md"].Content != "new" {
		t.Errorf("files writeback = %#v, want the subagent snapshot", update["files"])
	}
}

// keysOf is a stable helper for failure messages.
func keysOf(m map[string]any) []string {
	return slices.Sorted(maps.Keys(m))
}

// --- (d) e2e: subagent file writeback lands in the parent files ------------

// TestTaskToolWritesSubagentFilesBackToParent proves the files pass-through
// (E2): a subagent with its own StateBackend writes a file through its
// filesystem view; the writeback filter passes the files key (public state),
// and the parent's files channel commits it via MapMergeReducer.
func TestTaskToolWritesSubagentFilesBackToParent(t *testing.T) {
	parentModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			taskToolCall("call_1", "writer", "write /notes/a.md with hello"),
		}},
		messages.AI("parent done"),
	}}
	subModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			writeFileToolCall("sub_1", "/notes/a.md", "hello"),
		}},
		messages.AI("file written"),
	}}
	mw, err := NewSubAgentMiddleware(parentModel, []any{SubAgentSpec{
		Name:        "writer",
		Description: "Writes files.",
		Model:       subModel,
		Backend:     backends.NewStateBackend(),
	}})
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}
	fsMw := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
	agent, err := agents.CreateAgent(parentModel, nil, agents.WithAgentMiddleware(fsMw, mw))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("delegate")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}

	if got := fileContentFromState(values, "/notes/a.md"); got != "hello" {
		t.Fatalf("parent files[/notes/a.md] = %q, want %q (files: %#v)", got, "hello", values[backends.FilesStateKey])
	}
	if got := taskResultMessage(t, values, "call_1").Content; got != "file written" {
		t.Fatalf("task ToolMessage content = %q, want %q", got, "file written")
	}
}

// --- (e) unknown subagent + unknown argument errors --------------------------

// TestTaskToolUnknownSubagentErrorText pins the UP-shaped error text returned
// as tool content when the model names a subagent that does not exist, with
// the allowed types listed backtick-quoted in declaration order.
func TestTaskToolUnknownSubagentErrorText(t *testing.T) {
	parentModel := &staticModel{reply: messages.AI("x")}
	mw, err := NewSubAgentMiddleware(parentModel, []any{SubAgentSpec{
		Name: "helper", Description: "Helper subagent.", Model: parentModel,
	}})
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}

	res, err := mw.Tools[0].Invoke(t.Context(), map[string]any{
		"subagent_type": "ghost",
		"description":   "boo",
	})
	if err != nil {
		t.Fatalf("unknown subagent must be tool content, not an error: %v", err)
	}
	want := "We cannot invoke subagent ghost because it does not exist, the only allowed types are `general-purpose`, `helper`"
	if res.Content != want {
		t.Fatalf("content = %q, want %q", res.Content, want)
	}
}

// TestTaskToolRejectsUnknownArguments pins the unknown-key rejection: any
// argument outside {description, subagent_type} fails the call with the UP
// per-key message naming the key.
func TestTaskToolRejectsUnknownArguments(t *testing.T) {
	parentModel := &staticModel{reply: messages.AI("x")}
	mw, err := NewSubAgentMiddleware(parentModel, nil)
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}
	_, err = mw.Tools[0].Invoke(t.Context(), map[string]any{
		"subagent_type": "general-purpose",
		"description":   "do things",
		"prompt":        "real instructions",
	})
	if err == nil {
		t.Fatal("unknown argument must error")
	}
	if !strings.Contains(err.Error(), "Unexpected argument 'prompt'; put all instructions for the subagent in `description`.") {
		t.Fatalf("error text = %q, want the UP unknown-argument message", err.Error())
	}
}

// --- (f) ThreadID derivation ------------------------------------------------

// TestTaskThreadID pins the subagent thread derivation: "task:" + tool call
// ID (E10).
func TestTaskThreadID(t *testing.T) {
	if got := taskThreadID("call_abc"); got != "task:call_abc" {
		t.Fatalf("taskThreadID(call_abc) = %q, want task:call_abc", got)
	}
	if got := taskThreadID(""); got != "task:" {
		t.Fatalf("taskThreadID(%q) = %q", "", got)
	}
}

// --- (g) fold semantics + parallel-mutation guard ----------------------------

// TestFilesChannelFoldsSameBatchUpdates pins Review Focus #1's fold semantics:
// a files channel built as channels.NewBinaryOperator(MapMergeReducer) folding
// one superstep's writes from BOTH a task tool and a write_file tool (each
// carrying a full files snapshot) merges last-write-wins per key with no
// error — untouched keys survive.
func TestFilesChannelFoldsSameBatchUpdates(t *testing.T) {
	ch := channels.NewBinaryOperator(backends.MapMergeReducer)

	taskFiles := map[string]backends.FileMeta{
		"a.md": {Content: "from task"},
		"b.md": {Content: "task b"},
	}
	writeFileFiles := map[string]backends.FileMeta{
		"a.md": {Content: "from write_file"},
	}
	changed, err := ch.Update([]any{taskFiles, writeFileFiles})
	if err != nil {
		t.Fatalf("same-batch task + write_file files updates must fold without error: %v", err)
	}
	if !changed {
		t.Fatal("fold must report a change")
	}
	got, err := ch.Get()
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	files, ok := got.(map[string]backends.FileMeta)
	if !ok {
		t.Fatalf("channel value type = %T, want map[string]backends.FileMeta", got)
	}
	if files["a.md"].Content != "from write_file" {
		t.Errorf("a.md = %q, want last write (write_file) to win", files["a.md"].Content)
	}
	if files["b.md"].Content != "task b" {
		t.Errorf("b.md = %q, want the task-only key to survive the fold", files["b.md"].Content)
	}
}

// TestParallelFileMutationError pins the ported _parallel_file_mutation_error
// guard (filesystem.py:204-228): a mutating tool call whose same batch carries
// an EARLIER mutating call on the same path is rejected with the ported
// literal; different paths, the call itself, read-only tools, and malformed
// paths all pass.
func TestParallelFileMutationError(t *testing.T) {
	batch := []messages.Message{
		messages.Human("edit the file"),
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			{ID: "call_1", Name: "write_file", Args: map[string]any{"file_path": "/a.md", "content": "one"}},
			{ID: "call_2", Name: "edit_file", Args: map[string]any{"file_path": "/a.md", "old_string": "one", "new_string": "two"}},
			{ID: "call_3", Name: "write_file", Args: map[string]any{"file_path": "/b.md", "content": "other"}},
			{ID: "call_4", Name: "read_file", Args: map[string]any{"file_path": "/a.md"}},
		}},
	}

	tests := []struct {
		name    string
		request agentsmw.ToolCallRequest
		wantErr bool
	}{
		{
			name: "later same-path edit is rejected",
			request: agentsmw.ToolCallRequest{
				ToolCall: agentsmw.ToolCall{ID: "call_2", Name: "edit_file", Args: map[string]any{"file_path": "/a.md"}},
				State:    map[string]any{"messages": batch},
			},
			wantErr: true,
		},
		{
			name: "different path passes",
			request: agentsmw.ToolCallRequest{
				ToolCall: agentsmw.ToolCall{ID: "call_3", Name: "write_file", Args: map[string]any{"file_path": "/b.md"}},
				State:    map[string]any{"messages": batch},
			},
		},
		{
			name: "read-only tool passes",
			request: agentsmw.ToolCallRequest{
				ToolCall: agentsmw.ToolCall{ID: "call_4", Name: "read_file", Args: map[string]any{"file_path": "/a.md"}},
				State:    map[string]any{"messages": batch},
			},
		},
		{
			name: "non-string path passes",
			request: agentsmw.ToolCallRequest{
				ToolCall: agentsmw.ToolCall{ID: "call_2", Name: "edit_file", Args: map[string]any{"file_path": 42}},
				State:    map[string]any{"messages": batch},
			},
		},
		{
			name: "no messages in state passes",
			request: agentsmw.ToolCallRequest{
				ToolCall: agentsmw.ToolCall{ID: "call_2", Name: "edit_file", Args: map[string]any{"file_path": "/a.md"}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, blocked := ParallelFileMutationError(tt.request)
			if blocked != tt.wantErr {
				t.Fatalf("blocked = %v, want %v (msg: %#v)", blocked, tt.wantErr, msg)
			}
			if blocked {
				if msg.Role != messages.RoleTool {
					t.Errorf("guard must return a ToolMessage, got role %q", msg.Role)
				}
				if msg.ToolCallID != "call_2" || msg.Name != "edit_file" {
					t.Errorf("guard must address the current call, got id=%q name=%q", msg.ToolCallID, msg.Name)
				}
				if msg.Content != "Error: parallel file mutations to the same path are not allowed." {
					t.Errorf("error literal = %q", msg.Content)
				}
			}
		})
	}
}

// --- refusal notices ---------------------------------------------------------

// TestRefusalNotice pins the notice text assembled from a refusal block's
// category and explanation (UP _refusal_notice, subagents.py:95-103).
func TestRefusalNotice(t *testing.T) {
	full := refusalNotice(messages.RefusalBlock{
		Refusal: "cannot comply",
		Extras:  map[string]any{"category": "safety"},
	})
	want := `The subagent stopped because a model refused the request. Category: safety. Explanation: "cannot comply" Retrying the same request is unlikely to succeed; rephrase it or continue without it.`
	if full != want {
		t.Fatalf("refusalNotice = %q, want %q", full, want)
	}

	bare := refusalNotice(messages.RefusalBlock{})
	wantBare := "The subagent stopped because a model refused the request. Retrying the same request is unlikely to succeed; rephrase it or continue without it."
	if bare != wantBare {
		t.Fatalf("bare refusalNotice = %q, want %q", bare, wantBare)
	}
}

// TestTaskToolRefusalBecomesNotice: when the subagent's model refuses, the
// refusal error raised inside the subagent surfaces as the task result notice
// instead of failing the parent run.
func TestTaskToolRefusalBecomesNotice(t *testing.T) {
	parentModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			taskToolCall("call_1", "helper", "do the thing"),
		}},
		messages.AI("understood"),
	}}
	subModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ContentBlocks: []messages.ContentBlock{
			messages.RefusalBlock{Refusal: "cannot comply", Extras: map[string]any{"category": "safety"}},
		}},
	}}
	mw := newHelperSubAgentMiddleware(t, parentModel, subModel)

	agent, err := agents.CreateAgent(parentModel, nil, agents.WithAgentMiddleware(mw))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("delegate")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}
	got := taskResultMessage(t, values, "call_1").Content
	if !strings.HasPrefix(got, "The subagent stopped because a model refused the request.") {
		t.Fatalf("task result = %q, want the refusal notice", got)
	}
	if !strings.Contains(got, `Explanation: "cannot comply"`) {
		t.Fatalf("task result = %q, want the refusal explanation", got)
	}
}

// TestTaskToolInterruptedSubagentReturnsPinnedText pins D10 (review F1):
// when the subagent run pauses with pending interrupts, the task tool answers
// with the pinned text as tool CONTENT and a nil error — the parent run
// survives to completion. The stub subagent pauses via the HITL middleware
// interrupting its one tool call (a *CompiledSubAgent passthrough, so no
// parent checkpointer ever threads into it).
func TestTaskToolInterruptedSubagentReturnsPinnedText(t *testing.T) {
	parentModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			taskToolCall("call_1", "helper", "do the thing"),
		}},
		messages.AI("parent continues"),
	}}
	subModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			{ID: "sub_1", Name: "set_flag", Args: map[string]any{"value": "hello"}},
		}},
	}}
	hitl := agentsmw.NewInterruptHumanInTheLoopMiddleware(map[string]agentsmw.InterruptConfig{
		"set_flag": {AllowedDecisions: []agentsmw.DecisionType{
			agentsmw.DecisionApprove, agentsmw.DecisionEdit,
			agentsmw.DecisionReject, agentsmw.DecisionRespond,
		}},
	})
	interrupted, err := agents.CreateAgent(subModel, []coretools.Tool{newCustomKeyTool(t)},
		agents.WithAgentMiddleware(hitl))
	if err != nil {
		t.Fatalf("CreateAgent(subagent): %v", err)
	}
	mw, err := NewSubAgentMiddleware(parentModel, []any{&CompiledSubAgent{
		Name: "helper", Description: "Helper subagent.", Agent: interrupted,
	}})
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}

	agent, err := agents.CreateAgent(parentModel, nil, agents.WithAgentMiddleware(mw))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("delegate")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}

	const want = "subagent interrupted (not resumable in this port)"
	if got := taskResultMessage(t, values, "call_1").Content; got != want {
		t.Fatalf("task ToolMessage content = %q, want %q", got, want)
	}
	// The parent run survives: the model produced its final answer after the
	// interrupted delegation.
	msgs, _ := values["messages"].([]messages.Message)
	if last := msgs[len(msgs)-1]; last.Role != messages.RoleAI || last.Content != "parent continues" {
		t.Fatalf("parent must finish normally after an interrupted subagent, last message = %#v", last)
	}
}

// --- final-text walk ----------------------------------------------------------

// TestTaskResultContentLastNonEmptyAIText pins the content walk: trailing
// empty AI messages are skipped, the last non-empty AI text wins, refusal in
// the final message wins over the walk, and an empty run yields "".
func TestTaskResultContentLastNonEmptyAIText(t *testing.T) {
	msgs := []messages.Message{
		messages.Human("task"),
		messages.AI("first answer"),
		messages.Tool("t1", "result"),
		messages.AI("final answer"),
		messages.AI(""), // trailing empty end_turn message
	}
	if got := taskResultContent(msgs); got != "final answer" {
		t.Fatalf("content = %q, want final answer", got)
	}
	if got := taskResultContent([]messages.Message{messages.Human("task")}); got != "" {
		t.Fatalf("content = %q, want empty", got)
	}
	refusing := []messages.Message{
		messages.Human("task"),
		{Role: messages.RoleAI, Content: "partial", ContentBlocks: []messages.ContentBlock{
			messages.RefusalBlock{Refusal: "no"},
		}},
	}
	if got := taskResultContent(refusing); !strings.HasPrefix(got, "The subagent stopped because a model refused the request.") {
		t.Fatalf("content = %q, want refusal notice", got)
	}
}
