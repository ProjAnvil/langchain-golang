package deepagents

// Task 9: the CreateDeepAgent assembler. The e2e tests drive a real compiled
// agent with a scripted fake model (same double as deepagents/middleware's
// tests, extended to record the tool names each BindTools call binds) and pin
// the brief's behaviors (a)-(d) plus the three Task 8 review handoffs:
//
//	R1: same-batch [task, write_file] Commands carrying files updates fold
//	    last-write-wins at the AGENT level (a real run, not just a channel).
//	R2: same-batch write_file+edit_file on one path yields the ported
//	    parallel-mutation error on the second call and files stay consistent.
//	R3: the auto-added general-purpose subagent carries the parent's tools
//	    and middleware.

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	deepmw "github.com/projanvil/langchain-golang/deepagents/middleware"
	agents "github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// --- fixtures -------------------------------------------------------------

// scriptedChatModel mirrors the sequenceModel double of
// deepagents/middleware/permissions_test.go (the repo's scripted fake
// ChatModel convention), extended with a per-bind record of the bound tool
// names so tests can assert what the agent actually binds to the model.
type scriptedChatModel struct {
	mu         sync.Mutex
	responses  []messages.Message
	idx        int
	boundTools [][]string
}

func (m *scriptedChatModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("scriptedChatModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

func (m *scriptedChatModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		out[i], err = m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *scriptedChatModel) Stream(_ context.Context, _ []messages.Message, _ ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return nil, fmt.Errorf("scriptedChatModel: streaming not supported")
}

func (m *scriptedChatModel) InputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *scriptedChatModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *scriptedChatModel) BindTools(tools []coretools.Tool) (language.ChatModel, error) {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name())
	}
	m.mu.Lock()
	m.boundTools = append(m.boundTools, names)
	m.mu.Unlock()
	return m, nil
}

func (m *scriptedChatModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// binds returns the recorded BindTools tool-name sets in call order.
func (m *scriptedChatModel) binds() [][]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.boundTools)
}

// newPingTool builds the caller-tool fixture: a trivial tool the assembler
// must bind alongside the middleware-provided filesystem and task tools. The
// second return value reports whether the tool fn ever ran, so tests can
// prove a subagent executed it (subagent messages never reach the parent).
func newPingTool(t *testing.T) (coretools.Tool, func() bool) {
	t.Helper()
	invoked := false
	tool, err := coretools.NewSimple("ping", "replies with pong",
		func(context.Context, string) (coretools.Result, error) {
			invoked = true
			return coretools.Result{Content: "pong"}, nil
		})
	if err != nil {
		t.Fatalf("NewSimple(ping): %v", err)
	}
	return tool, func() bool { return invoked }
}

func writeFileToolCall(id, path, content string) messages.ToolCall {
	return messages.ToolCall{ID: id, Name: "write_file", Args: map[string]any{
		"file_path": path, "content": content,
	}}
}

func readFileToolCall(id, path string) messages.ToolCall {
	return messages.ToolCall{ID: id, Name: "read_file", Args: map[string]any{"file_path": path}}
}

func editFileToolCall(id, path, oldStr, newStr string) messages.ToolCall {
	return messages.ToolCall{ID: id, Name: "edit_file", Args: map[string]any{
		"file_path": path, "old_string": oldStr, "new_string": newStr,
	}}
}

func deleteFileToolCall(id, path string) messages.ToolCall {
	return messages.ToolCall{ID: id, Name: "delete", Args: map[string]any{"file_path": path}}
}

func taskToolCall(id, subagentType, description string) messages.ToolCall {
	return messages.ToolCall{ID: id, Name: deepmw.TaskToolName, Args: map[string]any{
		"subagent_type": subagentType,
		"description":   description,
	}}
}

// toolMessageFor finds the ToolMessage answering the given tool call.
func toolMessageFor(t *testing.T, values map[string]any, toolCallID string) messages.Message {
	t.Helper()
	msgs, _ := values["messages"].([]messages.Message)
	for _, msg := range msgs {
		if msg.Role == messages.RoleTool && msg.ToolCallID == toolCallID {
			return msg
		}
	}
	t.Fatalf("no ToolMessage for tool call %q in final state", toolCallID)
	return messages.Message{}
}

// fileContentFromState reads a file's content out of an agent-run values map.
func fileContentFromState(t *testing.T, values map[string]any, path string) string {
	t.Helper()
	files, _ := values[backends.FilesStateKey].(map[string]backends.FileMeta)
	return files[path].Content
}

// interruptPrivateWrites gates every write under /private/** on human
// approval (same rule set as deepagents/middleware/permissions_test.go).
func interruptPrivateWrites() backends.PermissionRule {
	return backends.PermissionRule{
		Operations: []string{"write"},
		Paths:      []string{"/private/**"},
		Mode:       backends.PermissionInterrupt,
	}
}

// --- (a) bound tools -------------------------------------------------------

// TestCreateDeepAgentBindsFileSystemToolsAndTask pins the assembled tool
// surface: the caller's tools plus the seven filesystem tools (delete
// included — StateBackend implements Deleter) plus the task tool, all bound
// to the model on its first call.
func TestCreateDeepAgentBindsFileSystemToolsAndTask(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{messages.AI("hi")}}
	ping, _ := newPingTool(t)
	agent, err := CreateDeepAgent(model,
		WithSystemPrompt("You are a test agent."),
		WithTools(ping),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}
	if _, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("hello")}); err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}

	binds := model.binds()
	if len(binds) == 0 {
		t.Fatal("model was never bound tools")
	}
	want := []string{
		"ping", "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
		deepmw.TaskToolName,
	}
	for _, name := range want {
		if !slices.Contains(binds[0], name) {
			t.Errorf("first bind %v is missing tool %q", binds[0], name)
		}
	}
}

// --- (b) write→read roundtrip over the default StateBackend -----------------

// TestCreateDeepAgentWriteReadRoundtrip drives write_file then read_file
// through a default-backend agent: the write commits into the files state
// (the run's values carry it), and the read returns the written content.
func TestCreateDeepAgentWriteReadRoundtrip(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			writeFileToolCall("call_1", "/notes/a.md", "hello"),
		}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			readFileToolCall("call_2", "/notes/a.md"),
		}},
		messages.AI("done"),
	}}
	agent, err := CreateDeepAgent(model)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("write then read")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}

	if got := toolMessageFor(t, values, "call_1").Content; got != "Updated file /notes/a.md" {
		t.Errorf("write result = %q, want the upstream Updated-file literal", got)
	}
	if got := toolMessageFor(t, values, "call_2").Content; !strings.Contains(got, "hello") {
		t.Errorf("read result = %q, want it to contain the written content", got)
	}
	if got := fileContentFromState(t, values, "/notes/a.md"); got != "hello" {
		t.Errorf("files[/notes/a.md] = %q, want %q", got, "hello")
	}
}

// --- (c) permission interrupt pause/resume ----------------------------------

// TestCreateDeepAgentPermissionInterruptPauseResume proves the assembler
// installs the HITL middleware from interrupt-mode permission rules: the
// write pauses before the tool runs, and an approved resume completes it.
func TestCreateDeepAgentPermissionInterruptPauseResume(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			writeFileToolCall("call_1", "/private/x.md", "approved"),
		}},
		messages.AI("done"),
	}}
	agent, err := CreateDeepAgent(model,
		WithPermissions(interruptPrivateWrites()),
		WithCheckpointer(checkpoint.NewMemorySaver()),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}

	values, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("write /private/x.md")}, graphpkg.Options{ThreadID: "approve"})
	if err != nil {
		t.Fatalf("first invoke: %v", err)
	}
	if len(interrupts) != 1 {
		t.Fatalf("expected exactly one pending interrupt, got %+v", interrupts)
	}
	request, err := agentsmw.HITLRequestFromInterrupt(interrupts[0])
	if err != nil {
		t.Fatalf("decode interrupt: %v", err)
	}
	if len(request.ActionRequests) != 1 || request.ActionRequests[0].Name != "write_file" {
		t.Fatalf("action requests = %#v, want one write_file", request.ActionRequests)
	}
	if got := fileContentFromState(t, values, "/private/x.md"); got != "" {
		t.Fatalf("file must not exist before approval, got %q", got)
	}

	values, interrupts, err = agent.Resume(t.Context(), graphpkg.Options{
		ThreadID: "approve",
		Resume:   agentsmw.HITLResponse{Decisions: []agentsmw.Decision{{Type: agentsmw.DecisionApprove}}},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(interrupts) != 0 {
		t.Fatalf("expected completion after approval, got %+v", interrupts)
	}
	if got := fileContentFromState(t, values, "/private/x.md"); got != "approved" {
		t.Fatalf("approved write must land in files state, got %q", got)
	}
}

// TestCreateDeepAgentUserInterruptOnGatesToolCall pins the opts.InterruptOn
// passthrough: a user-supplied entry (no permission rules at all) pauses the
// covered tool, and the union rule — user entries win per tool — is exercised
// by mergedInterruptConfigs below.
func TestCreateDeepAgentUserInterruptOnGatesToolCall(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			readFileToolCall("call_1", "/notes/missing.md"),
		}},
		messages.AI("done"),
	}}
	agent, err := CreateDeepAgent(model,
		WithInterruptOn(map[string]agentsmw.InterruptConfig{
			"read_file": {AllowedDecisions: []agentsmw.DecisionType{agentsmw.DecisionApprove}},
		}),
		WithCheckpointer(checkpoint.NewMemorySaver()),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}

	_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("read the file")}, graphpkg.Options{ThreadID: "gate"})
	if err != nil || len(interrupts) != 1 {
		t.Fatalf("first invoke: %v interrupts=%+v, want one interrupt", err, interrupts)
	}
	values, interrupts, err := agent.Resume(t.Context(), graphpkg.Options{
		ThreadID: "gate",
		Resume:   agentsmw.HITLResponse{Decisions: []agentsmw.Decision{{Type: agentsmw.DecisionApprove}}},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(interrupts) != 0 {
		t.Fatalf("expected completion after approval, got %+v", interrupts)
	}
	toolMessageFor(t, values, "call_1") // the read executed and answered
}

// --- (d) delegation writes files back through the shared backend ------------

// TestCreateDeepAgentDelegationWritesFilesBackToParent proves E2 end-to-end
// through the assembler: the parent delegates via task, the subagent writes
// through its own FilesystemMiddleware over the SHARED StateBackend, and the
// writeback lands in the parent's files state.
func TestCreateDeepAgentDelegationWritesFilesBackToParent(t *testing.T) {
	backend := backends.NewStateBackend()
	subModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			writeFileToolCall("sub_1", "/notes/a.md", "hello"),
		}},
		messages.AI("file written"),
	}}
	parentModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			taskToolCall("call_1", "writer", "write /notes/a.md with hello"),
		}},
		messages.AI("parent done"),
	}}
	agent, err := CreateDeepAgent(parentModel,
		WithBackend(backend),
		WithSubAgents([]any{deepmw.SubAgentSpec{
			Name:        "writer",
			Description: "Writes files.",
			Model:       subModel,
			Backend:     backend,
		}}),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("delegate")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}

	if got := fileContentFromState(t, values, "/notes/a.md"); got != "hello" {
		t.Fatalf("parent files[/notes/a.md] = %q, want %q (files: %#v)", got, "hello", values[backends.FilesStateKey])
	}
	if got := toolMessageFor(t, values, "call_1").Content; got != "file written" {
		t.Fatalf("task ToolMessage = %q, want the subagent final text", got)
	}
}

// --- R1: same-batch task + write_file fold at the agent level ---------------

// TestCreateDeepAgentSameBatchTaskAndWriteFoldLastWriteWins pins D13 at the
// AGENT level (Review handoff R1): one AI message carries BOTH a task call
// (whose subagent writes /notes/a.md) and a write_file call on the same path.
// Both tools run, both return files Commands, and the agent-level fold
// (applyToolNodeCommands overwrites the files value with each later Command
// in tool order, then the channel commits the folded Overwrite-wrapped
// snapshot as a replacement — D15) reflects the LATER Command — write_file,
// second in the batch — with no error and the run surviving to the parent's
// final message.
func TestCreateDeepAgentSameBatchTaskAndWriteFoldLastWriteWins(t *testing.T) {
	backend := backends.NewStateBackend()
	subModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			writeFileToolCall("sub_1", "/notes/a.md", "from task"),
		}},
		messages.AI("task done"),
	}}
	parentModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			taskToolCall("call_task", "writer", "write /notes/a.md with from task"),
			writeFileToolCall("call_write", "/notes/a.md", "from write_file"),
		}},
		messages.AI("parent done"),
	}}
	agent, err := CreateDeepAgent(parentModel,
		WithBackend(backend),
		WithSubAgents([]any{deepmw.SubAgentSpec{
			Name:        "writer",
			Description: "Writes files.",
			Model:       subModel,
			Backend:     backend,
		}}),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("write it twice")})
	if err != nil {
		t.Fatalf("same-batch task + write_file must not error the run: %v", err)
	}

	// The later Command (write_file, second in the batch) wins the key.
	if got := fileContentFromState(t, values, "/notes/a.md"); got != "from write_file" {
		t.Fatalf("files[/notes/a.md] = %q, want the later Command's %q", got, "from write_file")
	}
	// Both tools answered: the delegation result and the write result.
	if got := toolMessageFor(t, values, "call_task").Content; got != "task done" {
		t.Errorf("task ToolMessage = %q, want the subagent final text", got)
	}
	if got := toolMessageFor(t, values, "call_write").Content; got != "Updated file /notes/a.md" {
		t.Errorf("write_file ToolMessage = %q, want the Updated-file literal", got)
	}
	// The run survived to the parent's final answer.
	msgs, _ := values["messages"].([]messages.Message)
	if last := msgs[len(msgs)-1]; last.Role != messages.RoleAI || last.Content != "parent done" {
		t.Errorf("parent must finish normally, last message = %#v", last)
	}
}

// --- R2: same-batch write+edit parallel-mutation guard -----------------------

// TestCreateDeepAgentParallelMutationGuardErrorsSecondCall pins the wired
// ParallelFileMutationError (Review handoff R2): a model message carrying
// write_file + edit_file on the SAME path lets both calls run, the first
// write lands, and the second call is answered with the ported upstream
// error literal — the run survives and the files state stays consistent
// (the edit never applied).
func TestCreateDeepAgentParallelMutationGuardErrorsSecondCall(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			writeFileToolCall("call_1", "/notes/a.md", "one"),
			editFileToolCall("call_2", "/notes/a.md", "one", "two"),
		}},
		messages.AI("done"),
	}}
	agent, err := CreateDeepAgent(model)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("write and edit")})
	if err != nil {
		t.Fatalf("guarded batch must not error the run: %v", err)
	}

	if got := toolMessageFor(t, values, "call_1").Content; got != "Updated file /notes/a.md" {
		t.Errorf("first write result = %q, want the Updated-file literal", got)
	}
	const wantErr = "Error: parallel file mutations to the same path are not allowed."
	if got := toolMessageFor(t, values, "call_2").Content; got != wantErr {
		t.Errorf("second call result = %q, want %q", got, wantErr)
	}
	if got := fileContentFromState(t, values, "/notes/a.md"); got != "one" {
		t.Errorf("files[/notes/a.md] = %q, want the first write (edit blocked)", got)
	}
}

// --- C-1 regression: deletion persists across supersteps --------------------

// TestCreateDeepAgentDeletePersistsAcrossSupersteps pins the final-review C-1
// fix end-to-end on the default CreateDeepAgent backend: the files channel
// commits each mutating tool's snapshot as a REPLACEMENT (channels.Overwrite
// around filesCommand's artifact), so a key deleted in one superstep cannot
// resurrect from the stored map in the next. The read_file AFTER the delete
// must return the upstream not-found text, not the deleted content.
func TestCreateDeepAgentDeletePersistsAcrossSupersteps(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			writeFileToolCall("call_w", "/notes/a.md", "hello"),
		}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			deleteFileToolCall("call_d", "/notes/a.md"),
		}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			readFileToolCall("call_r", "/notes/a.md"),
		}},
		messages.AI("done"),
	}}
	agent, err := CreateDeepAgent(model)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("write, delete, then read")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}

	if got := toolMessageFor(t, values, "call_r").Content; got != "Error: File '/notes/a.md' not found" {
		t.Fatalf("read-after-delete = %q, want the UP not-found text (deletion must not resurrect)", got)
	}
	files, _ := values[backends.FilesStateKey].(map[string]backends.FileMeta)
	if _, alive := files["/notes/a.md"]; alive {
		t.Fatalf("final files still carry the deleted /notes/a.md: %#v", files)
	}
}

// TestCreateDeepAgentSubagentDeletePersistsInParentState is the subagent-scope
// half of C-1: a task-invoked subagent deleting a file over the SHARED
// StateBackend must leave the PARENT's files state without the key. The task
// writeback wraps its files value in channels.Overwrite (invokeTask), so the
// parent's channel replaces its stored map instead of overlaying it.
func TestCreateDeepAgentSubagentDeletePersistsInParentState(t *testing.T) {
	backend := backends.NewStateBackend()
	subModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			deleteFileToolCall("sub_1", "/notes/a.md"),
		}},
		messages.AI("deleted"),
	}}
	parentModel := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			writeFileToolCall("call_w", "/notes/a.md", "hello"),
		}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			taskToolCall("call_t", "deleter", "delete /notes/a.md"),
		}},
		messages.AI("parent done"),
	}}
	agent, err := CreateDeepAgent(parentModel,
		WithBackend(backend),
		WithSubAgents([]any{deepmw.SubAgentSpec{
			Name:        "deleter",
			Description: "Deletes files.",
			Model:       subModel,
			Backend:     backend,
		}}),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("write then delegate the delete")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}

	if got := toolMessageFor(t, values, "call_t").Content; got != "deleted" {
		t.Fatalf("task ToolMessage = %q, want the subagent final text", got)
	}
	files, _ := values[backends.FilesStateKey].(map[string]backends.FileMeta)
	if _, alive := files["/notes/a.md"]; alive {
		t.Fatalf("parent files still carry the subagent-deleted /notes/a.md: %#v", files)
	}
}

// --- R3: general-purpose auto-add carries the parent's tools/middleware ------

// TestCreateDeepAgentGeneralPurposeCarriesParentTools pins Review handoff R3
// two ways. White-box: assembleSubAgents prepends a general-purpose spec
// carrying opts.Tools and opts.Middleware when no caller spec claims the
// name, and leaves the list untouched when one does. End-to-end: a
// delegation to general-purpose binds exactly the parent's caller tools
// inside the subagent (UP graph.py:918 — the default subagent gets the main
// agent's tools).
func TestCreateDeepAgentGeneralPurposeCarriesParentTools(t *testing.T) {
	marker := agentsmw.NewSummarizationMiddleware(func(string, []messages.Message) (string, error) {
		return "", nil
	})
	ping, pingRan := newPingTool(t)

	// White-box: the auto-added spec threads tools and middleware.
	opts := DeepAgentOptions{
		Tools:      []coretools.Tool{ping},
		Middleware: []any{marker},
		SubAgents:  []any{deepmw.SubAgentSpec{Name: "helper", Description: "Helper."}},
	}
	list := assembleSubAgents(&scriptedChatModel{}, &opts)
	if len(list) != 2 {
		t.Fatalf("assembled list = %#v, want [general-purpose, helper]", list)
	}
	gp, ok := list[0].(deepmw.SubAgentSpec)
	if !ok {
		t.Fatalf("list[0] type = %T, want SubAgentSpec", list[0])
	}
	if gp.Name != "general-purpose" {
		t.Fatalf("list[0].Name = %q, want general-purpose", gp.Name)
	}
	if len(gp.Tools) != 1 || gp.Tools[0].Name() != "ping" {
		t.Fatalf("general-purpose tools = %#v, want the parent's [ping]", gp.Tools)
	}
	if len(gp.Middleware) != 1 || gp.Middleware[0] != any(marker) {
		t.Fatalf("general-purpose middleware = %#v, want the parent's [marker]", gp.Middleware)
	}

	// A caller-supplied general-purpose is left untouched.
	opts.SubAgents = []any{deepmw.SubAgentSpec{Name: "general-purpose", Description: "mine"}}
	if got := assembleSubAgents(&scriptedChatModel{}, &opts); len(got) != 1 {
		t.Fatalf("caller general-purpose must not be duplicated: %#v", got)
	}

	// End-to-end: delegation to general-purpose binds and runs the parent's
	// tools inside the subagent (the parent itself never calls ping).
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			taskToolCall("call_1", "general-purpose", "use the ping tool once"),
		}},
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			{ID: "call_gp", Name: "ping", Args: map[string]any{"text": "hi"}},
		}},
		messages.AI("sub done"),
		messages.AI("parent done"),
	}}
	agent, err := CreateDeepAgent(model, WithTools(ping))
	if err != nil {
		t.Fatalf("CreateDeepAgent: %v", err)
	}
	values, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("delegate")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}
	if got := toolMessageFor(t, values, "call_1").Content; got != "sub done" {
		t.Fatalf("task ToolMessage = %q, want sub done", got)
	}
	if !pingRan() {
		t.Fatal("the general-purpose subagent must carry and execute the parent's ping tool")
	}
	var gpBind []string
	for _, bind := range model.binds() {
		if slices.Equal(bind, []string{"ping"}) {
			gpBind = bind
		}
	}
	if gpBind == nil {
		t.Fatalf("no subagent bind carried exactly the parent tools; binds = %v", model.binds())
	}
}

// --- assembler unit coverage ------------------------------------------------

// TestDeepAgentOptionSetters pins every functional option against the options
// struct.
func TestDeepAgentOptionSetters(t *testing.T) {
	summarizer := agentsmw.SummarizerFunc(func(string, []messages.Message) (string, error) { return "s", nil })
	saver := checkpoint.NewMemorySaver()
	backend := backends.NewStateBackend()
	ping, _ := newPingTool(t)
	marker := agentsmw.NewModelRefusalMiddleware()
	perms := []backends.PermissionRule{{Operations: []string{"write"}, Paths: []string{"/**"}, Mode: backends.PermissionDeny}}
	interruptOn := map[string]agentsmw.InterruptConfig{
		"read_file": {AllowedDecisions: []agentsmw.DecisionType{agentsmw.DecisionApprove}},
	}
	spec := deepmw.SubAgentSpec{Name: "helper", Description: "Helper."}
	responseFormat := agents.NewToolStrategy(answerSchema())
	stateField := agents.StateField{Name: "visits", Reducer: sumVisitsReducer}
	contextField := agents.ContextField{Name: "user_id", Type: "string"}

	var opts DeepAgentOptions
	for _, opt := range []DeepAgentOption{
		WithSystemPrompt("prompt"),
		WithTools(ping),
		WithMiddleware(marker),
		WithSubAgents([]any{spec}),
		WithBackend(backend),
		WithPermissions(perms...),
		WithInterruptOn(interruptOn),
		WithCheckpointer(saver),
		WithStore(store.NewInMemoryStore()),
		WithSummarizer(summarizer),
		WithResponseFormat(responseFormat),
		WithStateFields(stateField),
		WithContextSchema(contextField),
	} {
		opt(&opts)
	}

	if opts.SystemPrompt != "prompt" {
		t.Errorf("SystemPrompt = %q", opts.SystemPrompt)
	}
	if len(opts.Tools) != 1 || opts.Tools[0].Name() != "ping" {
		t.Errorf("Tools = %#v", opts.Tools)
	}
	if len(opts.Middleware) != 1 || opts.Middleware[0] != any(marker) {
		t.Errorf("Middleware = %#v", opts.Middleware)
	}
	if len(opts.SubAgents) != 1 || opts.SubAgents[0].(deepmw.SubAgentSpec).Name != "helper" {
		t.Errorf("SubAgents = %#v", opts.SubAgents)
	}
	if opts.Backend != backend {
		t.Error("Backend not set")
	}
	if len(opts.Permissions) != 1 || opts.Permissions[0].Mode != backends.PermissionDeny {
		t.Errorf("Permissions = %#v", opts.Permissions)
	}
	if _, ok := opts.InterruptOn["read_file"]; !ok {
		t.Errorf("InterruptOn = %#v", opts.InterruptOn)
	}
	if opts.Checkpointer != saver {
		t.Error("Checkpointer not set")
	}
	if opts.Store == nil {
		t.Error("Store not set")
	}
	if opts.Summarizer == nil {
		t.Error("Summarizer not set")
	}
	if !reflect.DeepEqual(opts.ResponseFormat, any(responseFormat)) {
		t.Errorf("ResponseFormat = %#v", opts.ResponseFormat)
	}
	if len(opts.StateFields) != 1 || opts.StateFields[0].Name != stateField.Name || opts.StateFields[0].Reducer == nil {
		t.Errorf("StateFields = %#v", opts.StateFields)
	}
	if len(opts.ContextSchema) != 1 || opts.ContextSchema[0] != contextField {
		t.Errorf("ContextSchema = %#v", opts.ContextSchema)
	}
}

// TestCreateDeepAgentRequiresModel: a nil model cannot compile even the
// auto-added general-purpose subagent, so CreateDeepAgent fails loudly.
func TestCreateDeepAgentRequiresModel(t *testing.T) {
	if _, err := CreateDeepAgent(nil); err == nil || !strings.Contains(err.Error(), "must specify 'model'") {
		t.Fatalf("nil model must error with the missing-model text, got %v", err)
	}
}

// TestCreateDeepAgentInvalidPermissionRule: an invalid permission rule fails
// at the FilesystemMiddleware constructor and the error surfaces unwrapped.
func TestCreateDeepAgentInvalidPermissionRule(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{messages.AI("hi")}}
	if _, err := CreateDeepAgent(model, WithPermissions(backends.PermissionRule{
		Operations: []string{"read"}, Paths: []string{"no-slash/**"}, Mode: backends.PermissionDeny,
	})); err == nil || !strings.Contains(err.Error(), "Permission path must start with '/'") {
		t.Fatalf("invalid rule must fail assembly, got %v", err)
	}
}

// TestCreateDeepAgentStorePassthrough: WithStore threads through to
// CreateAgent (the assembled agent still builds and runs with a store).
func TestCreateDeepAgentStorePassthrough(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{messages.AI("hi")}}
	agent, err := CreateDeepAgent(model, WithStore(store.NewInMemoryStore()))
	if err != nil {
		t.Fatalf("CreateDeepAgent(store): %v", err)
	}
	if _, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("hello")}); err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}
}

// TestAssembleSubAgentsNilCompiledEntry: a nil *CompiledSubAgent does not
// claim the general-purpose name, so the default is still prepended.
func TestAssembleSubAgentsNilCompiledEntry(t *testing.T) {
	opts := DeepAgentOptions{SubAgents: []any{(*deepmw.CompiledSubAgent)(nil)}}
	list := assembleSubAgents(&scriptedChatModel{}, &opts)
	if len(list) != 2 {
		t.Fatalf("assembled list len = %d, want 2 (default + nil entry): %#v", len(list), list)
	}
	if gp, ok := list[0].(deepmw.SubAgentSpec); !ok || gp.Name != "general-purpose" {
		t.Fatalf("list[0] = %#v, want the prepended general-purpose spec", list[0])
	}
}

// TestMergedInterruptConfigs pins the interrupt map union: permission-derived
// entries per covered tool, user entries winning per tool, both empty → empty.
func TestMergedInterruptConfigs(t *testing.T) {
	derived := mergedInterruptConfigs([]backends.PermissionRule{interruptPrivateWrites()}, nil)
	for _, tool := range []string{"write_file", "edit_file", "delete"} {
		if _, ok := derived[tool]; !ok {
			t.Errorf("derived map missing %q: %#v", tool, derived)
		}
	}
	if len(derived) != 3 {
		t.Fatalf("derived map = %#v, want exactly the three write tools", derived)
	}

	userOnly := agentsmw.InterruptConfig{AllowedDecisions: []agentsmw.DecisionType{agentsmw.DecisionApprove}}
	merged := mergedInterruptConfigs([]backends.PermissionRule{interruptPrivateWrites()},
		map[string]agentsmw.InterruptConfig{"write_file": userOnly, "custom_tool": userOnly})
	if fmt.Sprint(merged["write_file"].AllowedDecisions) != fmt.Sprint(userOnly.AllowedDecisions) {
		t.Errorf("user entry must win per tool, got %#v", merged["write_file"])
	}
	if _, ok := merged["custom_tool"]; !ok {
		t.Error("user-only entries must be kept")
	}
	if len(merged) != 4 {
		t.Fatalf("merged map = %#v, want write_file+edit_file+delete+custom_tool", merged)
	}

	if got := mergedInterruptConfigs(nil, nil); len(got) != 0 {
		t.Fatalf("empty inputs must yield an empty map, got %#v", got)
	}

	// User-only entries (no interrupt rules): the derived map is empty, so
	// the merge seeds from the user map alone.
	userOnlyMerged := mergedInterruptConfigs(nil, map[string]agentsmw.InterruptConfig{"custom_tool": userOnly})
	if len(userOnlyMerged) != 1 {
		t.Fatalf("user-only merge = %#v, want the single custom entry", userOnlyMerged)
	}
}

// TestDefaultSummarizer pins the model-backed default: one summarize call
// over the prompt and transcript, returning the model's text; a model error
// propagates.
func TestDefaultSummarizer(t *testing.T) {
	ok := &scriptedChatModel{responses: []messages.Message{messages.AI("summary text")}}
	summarize := defaultSummarizer(ok)
	got, err := summarize("Summarize:", []messages.Message{messages.Human("hello"), messages.AI("world")})
	if err != nil {
		t.Fatalf("defaultSummarizer: %v", err)
	}
	if got != "summary text" {
		t.Fatalf("summary = %q, want the model reply", got)
	}

	failing := &scriptedChatModel{}
	if _, err := defaultSummarizer(failing)("p", []messages.Message{messages.Human("x")}); err == nil {
		t.Fatal("model failure must propagate")
	}
}

// TestCreateDeepAgentCustomSummarizerAccepted: WithSummarizer overrides the
// default and the agent still assembles and runs.
func TestCreateDeepAgentCustomSummarizerAccepted(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{messages.AI("hi")}}
	agent, err := CreateDeepAgent(model,
		WithSummarizer(agentsmw.SummarizerFunc(func(string, []messages.Message) (string, error) {
			return "condensed", nil
		})),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent(custom summarizer): %v", err)
	}
	if _, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("hello")}); err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}
}

// --- assembler parameter passthrough (response_format / state_schema /
// context_schema; port-completeness audit C1-C3, UP graph.py:308-310) ------

// answerSchema is the structured-output fixture (title "Answer" so the
// strategy's structured tool is named "Answer", as in langchain/agents'
// own structured-output tests).
func answerSchema() schema.Schema {
	s := schema.Object(map[string]schema.Schema{
		"text": schema.String("the answer text"),
	}, "text")
	s["title"] = "Answer"
	return s
}

// TestCreateDeepAgentResponseFormatPassthrough: WithResponseFormat reaches
// agents.WithAgentResponseFormat — a ToolStrategy's structured tool is bound
// to the model, the model's matching structured call ends the run, and the
// parsed value surfaces in the final state's "structured_response" key (UP
// graph.py:308).
func TestCreateDeepAgentResponseFormatPassthrough(t *testing.T) {
	strategy := agents.NewToolStrategy(answerSchema())
	toolName := strategy.SchemaSpecs[0].Name
	model := &scriptedChatModel{responses: []messages.Message{{
		Role: messages.RoleAI,
		ToolCalls: []messages.ToolCall{
			{ID: "call_1", Name: toolName, Args: map[string]any{"text": "42"}},
		},
	}}}
	agent, err := CreateDeepAgent(model, WithResponseFormat(strategy))
	if err != nil {
		t.Fatalf("CreateDeepAgent(response format): %v", err)
	}
	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("answer me")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}
	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["text"] != "42" {
		t.Fatalf("structured_response = %#v, want parsed text=42 (option must reach agents.WithAgentResponseFormat)", state["structured_response"])
	}
}

// TestCreateDeepAgentResponseFormatRejectsUnknown: an unsupported
// response_format value fails at CreateAgent's resolveResponseFormat,
// proving the passthrough (the deepagents layer adds no validation of its
// own — the platform's error surfaces verbatim).
func TestCreateDeepAgentResponseFormatRejectsUnknown(t *testing.T) {
	model := &scriptedChatModel{}
	if _, err := CreateDeepAgent(model, WithResponseFormat("not-a-strategy")); err == nil {
		t.Fatal("expected CreateDeepAgent to reject an unsupported response format")
	}
}

// newBumpTool builds a tool whose artifact Command writes 1 to the "visits"
// state key, so a run exercising it twice proves the custom field's reducer
// merges across supersteps (UP graph.py:309 state_schema).
func newBumpTool(t *testing.T) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewFunc(
		"bump",
		"adds one visit",
		schema.Object(map[string]schema.Schema{}),
		func(context.Context, map[string]any) (coretools.Result, error) {
			return coretools.Result{
				Content:  "bumped",
				Artifact: &types.Command{Update: map[string]any{"visits": 1}},
			}, nil
		},
	)
	if err != nil {
		t.Fatalf("NewFunc(bump): %v", err)
	}
	return tool
}

// sumVisitsReducer is a merge reducer: successive writes to "visits" add up
// (LastValue semantics would keep only the latest write).
func sumVisitsReducer(existing, update any) (any, error) {
	if existing == nil {
		return update, nil
	}
	base, _ := existing.(int)
	delta, _ := update.(int)
	return base + delta, nil
}

// TestCreateDeepAgentStateFieldsMergeReducerFlowsThroughRun: WithStateFields
// registers the custom "visits" field with a merge reducer on the main graph;
// two supersteps of bump tool calls each committing visits=1 must fold to 2
// under the reducer (UP graph.py:309; without the registration the key would
// be LastValue and the final value would stay 1).
func TestCreateDeepAgentStateFieldsMergeReducerFlowsThroughRun(t *testing.T) {
	bumpCall := func(id string) messages.Message {
		return messages.Message{
			Role:      messages.RoleAI,
			ToolCalls: []messages.ToolCall{{ID: id, Name: "bump", Args: map[string]any{}}},
		}
	}
	model := &scriptedChatModel{responses: []messages.Message{
		bumpCall("call_1"),
		bumpCall("call_2"),
		messages.AI("done"),
	}}
	agent, err := CreateDeepAgent(model,
		WithTools(newBumpTool(t)),
		WithStateFields(agents.StateField{Name: "visits", Reducer: sumVisitsReducer}),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent(state fields): %v", err)
	}
	state, err := agent.InvokeWithState(t.Context(), []messages.Message{messages.Human("bump twice")})
	if err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}
	if got := state["visits"]; got != 2 {
		t.Fatalf("state[visits] = %#v, want 2 (merge reducer must fold both supersteps' writes)", got)
	}
}

// TestCreateDeepAgentContextSchemaAccepted: WithContextSchema threads
// through to agents.WithAgentContextSchema (UP graph.py:310; spec §7's
// promised Context passthrough). The platform layer is purely declarative —
// the schema declares fields but does not gate the context values — so this
// is the introspection-level check: the agent assembles with the declared
// schema and the corresponding context values still flow to tools.
func TestCreateDeepAgentContextSchemaAccepted(t *testing.T) {
	var seenUser any
	tool, err := coretools.NewFunc(
		"whoami",
		"reports the caller",
		schema.Object(map[string]schema.Schema{}),
		func(ctx context.Context, _ map[string]any) (coretools.Result, error) {
			seenUser, _ = agents.ContextValue(ctx, "user_id")
			return coretools.Result{Content: "ok"}, nil
		},
	)
	if err != nil {
		t.Fatalf("NewFunc(whoami): %v", err)
	}
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{ID: "c1", Name: "whoami", Args: map[string]any{}}}},
		messages.AI("done"),
	}}
	agent, err := CreateDeepAgent(model,
		WithTools(tool),
		WithContextSchema(agents.ContextField{Name: "user_id", Type: "string"}),
	)
	if err != nil {
		t.Fatalf("CreateDeepAgent(context schema): %v", err)
	}
	ctx := agents.WithContextValues(t.Context(), map[string]any{"user_id": "u-7"})
	if _, err := agent.InvokeWithState(ctx, []messages.Message{messages.Human("who am I")}); err != nil {
		t.Fatalf("InvokeWithState: %v", err)
	}
	if seenUser != "u-7" {
		t.Fatalf("context user_id seen by tool = %#v, want u-7", seenUser)
	}
}
