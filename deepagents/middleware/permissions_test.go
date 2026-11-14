package middleware

// Task 6: permission-derived HITL interrupt wiring. InterruptConfigsFrom-
// Permissions turns interrupt-mode permission rules into the interrupt_on map
// NewInterruptHumanInTheLoopMiddleware consumes (UP _fs_interrupt.py
// _build_interrupt_on_from_permissions), and the e2e test proves the wiring
// through CreateAgent: a write_file call covered by an interrupt rule pauses
// the run, an approved resume writes the file, and a rejected resume answers
// with the HITL denial ToolMessage and leaves the file unwritten.

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	agents "github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
)

// --- InterruptConfigsFromPermissions mapping ------------------------------

func TestInterruptConfigsFromPermissions(t *testing.T) {
	readInterrupt := backends.PermissionRule{
		Operations: []string{"read"}, Paths: []string{"/docs/**"}, Mode: backends.PermissionInterrupt,
	}
	writeInterrupt := backends.PermissionRule{
		Operations: []string{"write"}, Paths: []string{"/private/**"}, Mode: backends.PermissionInterrupt,
	}
	denyLockedWrites := backends.PermissionRule{
		Operations: []string{"write"}, Paths: []string{"/locked/**"}, Mode: backends.PermissionDeny,
	}
	allowPublicReads := backends.PermissionRule{
		Operations: []string{"read"}, Paths: []string{"/public/**"}, Mode: backends.PermissionAllow,
	}

	readTools := []string{"ls", "read_file", "glob", "grep"}
	writeTools := []string{"write_file", "edit_file", "delete"}
	allTools := append(append([]string(nil), readTools...), writeTools...)
	sort.Strings(allTools)

	tests := []struct {
		name  string
		perms []backends.PermissionRule
		want  []string // sorted tool names that must carry an entry (and only those)
	}{
		{"nil permissions", nil, nil},
		{"no rules", []backends.PermissionRule{}, nil},
		{"deny and allow rules never create entries", []backends.PermissionRule{denyLockedWrites, allowPublicReads}, nil},
		{"write-op interrupt rule covers the write tools", []backends.PermissionRule{writeInterrupt}, writeTools},
		{"read-op interrupt rule covers the read tools", []backends.PermissionRule{readInterrupt}, readTools},
		{
			"both op classes cover all seven tools",
			[]backends.PermissionRule{readInterrupt, writeInterrupt},
			allTools,
		},
		{
			"deny alongside interrupt still enters only the interrupt op class",
			[]backends.PermissionRule{denyLockedWrites, readInterrupt},
			readTools,
		},
		{
			"first interrupt rule per tool wins with duplicate op classes",
			[]backends.PermissionRule{writeInterrupt, {Operations: []string{"write"}, Paths: []string{"/other/**"}, Mode: backends.PermissionInterrupt}},
			writeTools,
		},
		{
			"non-filesystem operation in an interrupt rule creates nothing",
			[]backends.PermissionRule{{Operations: []string{"exec"}, Paths: []string{"/**"}, Mode: backends.PermissionInterrupt}},
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := InterruptConfigsFromPermissions(tt.perms)
			var gotNames []string
			for name, config := range got {
				gotNames = append(gotNames, name)
				// Every entry offers the full decision set (UP offers
				// approve/edit/reject/respond for every derived tool) and no
				// when predicate (D11: path scoping not ported).
				wantDecisions := []agentsmw.DecisionType{
					agentsmw.DecisionApprove, agentsmw.DecisionEdit,
					agentsmw.DecisionReject, agentsmw.DecisionRespond,
				}
				if fmt.Sprint(config.AllowedDecisions) != fmt.Sprint(wantDecisions) {
					t.Fatalf("tool %q AllowedDecisions = %v, want %v", name, config.AllowedDecisions, wantDecisions)
				}
				if config.When != nil {
					t.Fatalf("tool %q must not carry a when predicate (D11)", name)
				}
			}
			sort.Strings(gotNames)
			sort.Strings(tt.want)
			if fmt.Sprint(gotNames) != fmt.Sprint(tt.want) {
				t.Fatalf("entries = %v, want %v", gotNames, tt.want)
			}
		})
	}
}

// --- End-to-end wiring through CreateAgent ---------------------------------

// scriptedChatModel mirrors the sequenceModel test double in
// langchain/agents/create_agent_test.go (the repo's scripted fake ChatModel
// convention): it returns Responses in order and — unlike
// language.FakeChatModel — returns itself from BindTools, so the cursor
// advances across CreateAgent's repeated bind+invoke cycles.
type scriptedChatModel struct {
	mu          sync.Mutex
	responses   []messages.Message
	idx         int
	invocations int
}

func (m *scriptedChatModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.invocations++
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

func (m *scriptedChatModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}

func (m *scriptedChatModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// interruptPrivateWrites is the Task 6 e2e rule set: every write under
// /private/** must be approved by a human before the tool runs.
var interruptPrivateWrites = backends.PermissionRule{
	Operations: []string{"write"},
	Paths:      []string{"/private/**"},
	Mode:       backends.PermissionInterrupt,
}

// newPermissionHitlAgent assembles the Task 9 shape manually: a
// FilesystemMiddleware over a StateBackend with an interrupt rule, the HITL
// middleware built from InterruptConfigsFromPermissions, and a memory
// checkpointer so the pause/resume mechanics run (mirroring the assembly
// idioms of langchain/agents/create_agent_hitl_test.go).
func newPermissionHitlAgent(t *testing.T, model *scriptedChatModel, rules ...backends.PermissionRule) *agents.Agent {
	t.Helper()
	fsMw := mustNewFilesystemMiddleware(t, backends.NewStateBackend(), rules...)
	hitl := agentsmw.NewInterruptHumanInTheLoopMiddleware(InterruptConfigsFromPermissions(fsMw.Permissions))
	agent, err := agents.CreateAgent(model, nil,
		agents.WithAgentMiddleware(fsMw, hitl),
		agents.WithAgentCheckpointer(checkpoint.NewMemorySaver()),
	)
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	return agent
}

func writeFileToolCall(id, path, content string) messages.ToolCall {
	return messages.ToolCall{ID: id, Name: "write_file", Args: map[string]any{
		"file_path": path, "content": content,
	}}
}

// fileContentFromState reads a file's content out of an agent-run values map.
// The files channel stores exactly one form — map[string]backends.FileMeta,
// since Changed() returns the full post-mutation snapshot in the stored form —
// so no other shape is tolerated. Empty when the file is absent.
func fileContentFromState(values map[string]any, path string) string {
	files, _ := values[backends.FilesStateKey].(map[string]backends.FileMeta)
	return files[path].Content
}

// TestPermissionHitlApprovedResumeWritesFile: the write_file call pauses the
// run (Interrupts non-empty, tool not yet executed), and a resume with an
// approve decision lets the tool run — the file lands in the agent's files
// state, proving the HITL middleware gates the call BEFORE the filesystem
// tool's own allow-through interrupt placeholder.
func TestPermissionHitlApprovedResumeWritesFile(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{writeFileToolCall("call_1", "/private/x.md", "approved")}},
		messages.AI("done"),
	}}
	agent := newPermissionHitlAgent(t, model, interruptPrivateWrites)

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
		t.Fatalf("decode interrupt value: %v", err)
	}
	if len(request.ActionRequests) != 1 || request.ActionRequests[0].Name != "write_file" {
		t.Fatalf("action requests mismatch: %#v", request.ActionRequests)
	}
	// The pause happens before the tool runs: no file committed yet.
	if got := fileContentFromState(values, "/private/x.md"); got != "" {
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
	files, ok := values[backends.FilesStateKey].(map[string]backends.FileMeta)
	if !ok {
		t.Fatalf("files state must be strict map[string]backends.FileMeta, got %T", values[backends.FilesStateKey])
	}
	if got := files["/private/x.md"].Content; got != "approved" {
		t.Fatalf("approved write must land in files state, got %q (files: %#v)", got, files)
	}
	out, _ := values["messages"].([]messages.Message)
	found := false
	for _, msg := range out {
		if msg.Role == messages.RoleTool && msg.Content == "Updated file /private/x.md" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected the write_file tool result in final messages: %#v", out)
	}
}

// TestPermissionHitlRejectedResumeDeniesWrite: resuming the pause with a
// reject decision answers the call with the HITL denial ToolMessage and the
// write never executes — no file lands in the state and the model sees the
// rejection.
func TestPermissionHitlRejectedResumeDeniesWrite(t *testing.T) {
	model := &scriptedChatModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{writeFileToolCall("call_1", "/private/y.md", "denied")}},
		messages.AI("understood"),
	}}
	agent := newPermissionHitlAgent(t, model, interruptPrivateWrites)

	_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("write /private/y.md")}, graphpkg.Options{ThreadID: "reject"})
	if err != nil || len(interrupts) != 1 {
		t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
	}

	values, interrupts, err := agent.Resume(t.Context(), graphpkg.Options{
		ThreadID: "reject",
		Resume: agentsmw.HITLResponse{Decisions: []agentsmw.Decision{{
			Type: agentsmw.DecisionReject, Message: "not allowed",
		}}},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(interrupts) != 0 {
		t.Fatalf("expected completion after rejection, got %+v", interrupts)
	}
	if got := fileContentFromState(values, "/private/y.md"); got != "" {
		t.Fatalf("rejected write must not execute, got %q (files: %#v)", got, values[backends.FilesStateKey])
	}
	out, _ := values["messages"].([]messages.Message)
	found := false
	for _, msg := range out {
		if msg.Role == messages.RoleTool && strings.Contains(msg.Content, "User rejected the tool call for `write_file`") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected the HITL denial ToolMessage in final messages: %#v", out)
	}
}
