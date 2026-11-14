package middleware

// Task 9 release-gate coverage mop-up: focused tests for the error and edge
// branches of Tasks 4-8 left uncovered by the task suites, so the
// deepagents/... packages clear the 95% Makefile gate.

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	"github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// failingBackend is a Backend whose every operation errors, standing in for
// an unusable filesystem so the tools' backend-error branches render as tool
// text instead of failing the run.
type failingBackend struct{}

func (failingBackend) Ls(context.Context, string) (backends.LsResult, error) {
	return backends.LsResult{}, errors.New("boom-ls")
}
func (failingBackend) Read(context.Context, string, int, int) (backends.ReadResult, error) {
	return backends.ReadResult{}, errors.New("boom-read")
}
func (failingBackend) Write(context.Context, string, string) (backends.WriteResult, error) {
	return backends.WriteResult{}, errors.New("boom-write")
}
func (failingBackend) Edit(context.Context, string, string, string, bool) (backends.EditResult, error) {
	return backends.EditResult{}, errors.New("boom-edit")
}
func (failingBackend) Glob(context.Context, string, string) (backends.GlobResult, error) {
	return backends.GlobResult{}, errors.New("boom-glob")
}
func (failingBackend) Grep(context.Context, string, string, string, int) (backends.GrepResult, error) {
	return backends.GrepResult{}, errors.New("boom-grep")
}

func TestNewFilesystemMiddlewareRejectsInvalidRule(t *testing.T) {
	if _, err := NewFilesystemMiddleware(backends.NewStateBackend(), backends.PermissionRule{
		Operations: []string{"read"}, Paths: []string{"secrets/**"}, Mode: backends.PermissionDeny,
	}); err == nil || !strings.Contains(err.Error(), "Permission path must start with '/'") {
		t.Fatalf("invalid rule must fail construction, got %v", err)
	}
}

func TestDeleteDenyGateSkipsDuplicatePatterns(t *testing.T) {
	m := mustNewFilesystemMiddleware(t, backends.NewStateBackend(),
		backends.PermissionRule{Operations: []string{"write"}, Paths: []string{"/a/**"}, Mode: backends.PermissionDeny},
		backends.PermissionRule{Operations: []string{"write"}, Paths: []string{"/a/**"}, Mode: backends.PermissionDeny},
	)
	if _, denied := m.deleteDenyGate("/a/b.md"); !denied {
		t.Fatal("duplicate-pattern deny rules must still deny the delete")
	}
}

func TestPathHelpersEdgeBranches(t *testing.T) {
	if got := globAnchor("/a*"); got != "/" {
		t.Fatalf(`globAnchor("/a*") = %q, want "/"`, got)
	}
	if !pathWithin("/anything/deep", "/") {
		t.Fatal(`pathWithin(p, "/") must be true`)
	}
	cases := []struct {
		pattern, anchor, target string
		want                    bool
	}{
		// A single non-** suffix that cannot match the target's ancestors
		// does not block (deny /work/*.log vs deleting /work/notes.txt).
		{"/work/*.log", "/work", "/work/notes.txt", false},
		// A wildcard ancestor match blocks: deleting /work/app/child
		// mutates the denied /work/app.
		{"/work/*", "/work", "/work/app/child", true},
		// A target outside the deny anchor is untouched.
		{"/elsewhere/**", "/elsewhere", "/work/x", false},
	}
	for _, tc := range cases {
		if got := wildcardDeleteOverlap(tc.pattern, tc.anchor, tc.target); got != tc.want {
			t.Errorf("wildcardDeleteOverlap(%q, %q, %q) = %v, want %v", tc.pattern, tc.anchor, tc.target, got, tc.want)
		}
	}
}

func TestFormatReadAndGrepEdgeBranches(t *testing.T) {
	// A window with content but TotalLines 0 renders the bare span (no
	// "of N").
	got := formatReadResult(backends.ReadResult{Content: "x", StartLine: 1, EndLine: 1}, 0, 100)
	if !strings.Contains(got, "lines 1-1") || strings.Contains(got, "of ") {
		t.Fatalf("formatReadResult without TotalLines = %q, want a bare span", got)
	}
	// Grep matches render sorted by path.
	got = formatReadResult(backends.ReadResult{Content: "x", StartLine: 1, EndLine: 1, TotalLines: 1}, 0, 100)
	if !strings.Contains(got, "lines 1-1 of 1") {
		t.Fatalf("formatReadResult with TotalLines = %q, want the of-<total> span", got)
	}
	grep := formatGrepResult(backends.GrepResult{Matches: []backends.GrepFileMatches{
		{Path: "/b.md", Lines: []backends.GrepLine{{LineNumber: 1, Text: "x"}}},
		{Path: "/a.md", Lines: []backends.GrepLine{{LineNumber: 2, Text: "y"}}},
	}}, "files_with_matches")
	if grep != "/a.md\n/b.md" {
		t.Fatalf("formatGrepResult must sort by path, got %q", grep)
	}
}

func TestArgIntShapes(t *testing.T) {
	if got := argInt(map[string]any{"k": int64(3)}, "k", 0); got != 3 {
		t.Fatalf("argInt(int64) = %d", got)
	}
	if got := argInt(map[string]any{"k": 2.9}, "k", 7); got != 2 {
		t.Fatalf("argInt(float64) = %d", got)
	}
	if got := argInt(map[string]any{"k": "x"}, "k", 7); got != 7 {
		t.Fatalf("argInt(string) = %d, want default", got)
	}
	if got, ok := argIntPresent(map[string]any{"k": int64(4)}, "k"); !ok || got != 4 {
		t.Fatalf("argIntPresent(int64) = (%d, %v)", got, ok)
	}
	if _, ok := argIntPresent(map[string]any{"k": "x"}, "k"); ok {
		t.Fatal("argIntPresent(string) must not be present")
	}
}

func TestLsEmptyListingIsNoFilesFound(t *testing.T) {
	m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
	res := invokeFileTool(t, stateCtx(nil), m, "ls", map[string]any{"path": "/"})
	if res.Content != fsNoFilesFound {
		t.Fatalf("ls on an empty filesystem = %q, want %q", res.Content, fsNoFilesFound)
	}
}

func TestToolPathNormalizationErrorsRenderAsToolText(t *testing.T) {
	m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
	cases := []struct {
		tool string
		args map[string]any
	}{
		{"read_file", map[string]any{"file_path": "a/../../etc/passwd"}},
		{"edit_file", map[string]any{"file_path": "a/../../x", "old_string": "a", "new_string": "b"}},
		{"delete", map[string]any{"file_path": "a/../../x"}},
		{"glob", map[string]any{"path": "a/../../x", "pattern": "*"}},
		{"grep", map[string]any{"path": "a/../../x", "pattern": "needle"}},
	}
	for _, tc := range cases {
		res := invokeFileTool(t, stateCtx(nil), m, tc.tool, tc.args)
		if !strings.HasPrefix(res.Content, "Error:") {
			t.Errorf("%s with escaping path = %q, want an Error: text", tc.tool, res.Content)
		}
	}
}

func TestWriteFileBackendErrorIsToolText(t *testing.T) {
	m := mustNewFilesystemMiddleware(t, failingBackend{})
	res := invokeFileTool(t, context.Background(), m, "write_file",
		map[string]any{"file_path": "/a.md", "content": "x"})
	if !strings.Contains(res.Content, "boom-write") {
		t.Fatalf("write_file over a failing backend = %q, want the backend error as text", res.Content)
	}
}

func TestGrepFiltersReadDeniedMatches(t *testing.T) {
	m := mustNewFilesystemMiddleware(t, backends.NewStateBackend(),
		backends.PermissionRule{Operations: []string{"read"}, Paths: []string{"/secret/**"}, Mode: backends.PermissionDeny},
	)
	ctx := stateCtx(map[string]backends.FileMeta{
		"/secret/a.md": meta("needle"),
		"/open/b.md":   meta("needle"),
	})
	res := invokeFileTool(t, ctx, m, "grep", map[string]any{"path": "/", "pattern": "needle"})
	if res.Content != "/open/b.md" {
		t.Fatalf("grep with a denied subtree = %q, want only /open/b.md", res.Content)
	}
}

func TestCompileSubAgentDuplicateMiddlewareErrors(t *testing.T) {
	_, err := CompileSubAgent(SubAgentSpec{
		Name:        "dup",
		Description: "d",
		Model:       &staticModel{reply: messages.AI("x")},
		Middleware:  []any{agentsmw.NewModelRefusalMiddleware()},
	})
	if err == nil || !strings.Contains(err.Error(), "SubAgent 'dup': create agent:") {
		t.Fatalf("duplicate middleware in a subagent must error, got %v", err)
	}
}

func TestNewSubAgentMiddlewareNilModelWithoutSpecs(t *testing.T) {
	if _, err := NewSubAgentMiddleware(nil, nil); err == nil ||
		!strings.Contains(err.Error(), "SubAgent 'general-purpose' must specify 'model'") {
		t.Fatalf("nil model with no specs must fail compiling the default subagent, got %v", err)
	}
}

// taskCtx builds the context the WrapToolCall hook injects around a real
// task tool call (parent state + tool call ID).
func taskCtx(parent map[string]any, toolCallID string) context.Context {
	ctx := context.WithValue(context.Background(), taskParentStateCtxKey{}, parent)
	return context.WithValue(ctx, taskToolCallIDCtxKey{}, toolCallID)
}

func TestTaskToolRequiresToolCallID(t *testing.T) {
	mw, err := NewSubAgentMiddleware(&staticModel{reply: messages.AI("x")}, nil)
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}
	_, err = mw.Tools[0].Invoke(context.Background(), map[string]any{
		"subagent_type": "general-purpose", "description": "d",
	})
	if err == nil || !strings.Contains(err.Error(), "Tool call ID is required") {
		t.Fatalf("direct invoke without the wrapper must error, got %v", err)
	}
}

func TestTaskToolContentOnlyWhenNoStateChange(t *testing.T) {
	mw, err := NewSubAgentMiddleware(&staticModel{reply: messages.AI("nothing to do")}, nil)
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}
	res, err := mw.Tools[0].Invoke(taskCtx(map[string]any{}, "t1"), map[string]any{
		"subagent_type": "general-purpose", "description": "d",
	})
	if err != nil {
		t.Fatalf("task invoke: %v", err)
	}
	if res.Content != "nothing to do" {
		t.Fatalf("task content = %q, want the subagent final text", res.Content)
	}
	if res.Artifact != nil {
		t.Fatalf("a subagent that changed no state must return no Command, got %#v", res.Artifact)
	}
}

func TestTaskToolSubagentErrorPropagates(t *testing.T) {
	// A subagent model that errors on its first call fails the subagent run;
	// the error is not a refusal notice, so it propagates out of the tool.
	mw, err := NewSubAgentMiddleware(&staticModel{reply: messages.AI("x")}, []any{SubAgentSpec{
		Name:        "failer",
		Description: "fails",
		Model:       &scriptedChatModel{}, // no responses: first call errors
	}})
	if err != nil {
		t.Fatalf("NewSubAgentMiddleware: %v", err)
	}
	_, err = mw.Tools[0].Invoke(taskCtx(map[string]any{}, "t1"), map[string]any{
		"subagent_type": "failer", "description": "d",
	})
	if err == nil || !strings.Contains(err.Error(), "no more responses") {
		t.Fatalf("subagent failure must propagate, got %v", err)
	}
}

func TestRefusalNoticeForStructuredOutputError(t *testing.T) {
	refusing := messages.Message{Role: messages.RoleAI, ContentBlocks: []messages.ContentBlock{
		messages.RefusalBlock{Refusal: "no"},
	}}
	notice, ok := refusalNoticeForError(&agents.StructuredOutputValidationError{AIMessage: refusing})
	if !ok || !strings.HasPrefix(notice, "The subagent stopped because a model refused the request.") {
		t.Fatalf("structured refusal = (%q, %v), want the refusal notice", notice, ok)
	}
	if _, ok := refusalNoticeForError(&agents.StructuredOutputValidationError{
		AIMessage: messages.AI("plain"),
	}); ok {
		t.Fatal("a structured error without a refusal block must not notice")
	}
}

func TestParallelFileMutationErrorBatchEdgeShapes(t *testing.T) {
	batch := []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			{ID: "call_1", Name: "write_file", Args: map[string]any{"file_path": 42, "content": "x"}},
			{ID: "call_2", Name: "edit_file", Args: map[string]any{"file_path": "a/../../x", "old_string": "a", "new_string": "b"}},
		}},
	}
	// Malformed earlier sibling paths never collide.
	msg, blocked := ParallelFileMutationError(agentsmw.ToolCallRequest{
		ToolCall: agentsmw.ToolCall{ID: "call_3", Name: "write_file", Args: map[string]any{"file_path": "/a.md"}},
		State:    map[string]any{"messages": batch},
	})
	if blocked {
		t.Fatalf("malformed sibling paths must pass, got %#v", msg)
	}
	// A current call absent from the batch's AI message passes.
	_, blocked = ParallelFileMutationError(agentsmw.ToolCallRequest{
		ToolCall: agentsmw.ToolCall{ID: "ghost", Name: "write_file", Args: map[string]any{"file_path": "/a.md"}},
		State:    map[string]any{"messages": batch},
	})
	if blocked {
		t.Fatal("a call not present in the batch must pass")
	}
}
