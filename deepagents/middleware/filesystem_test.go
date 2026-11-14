package middleware

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// fakeTool is a minimal coretools.Tool for asserting ProvidedTools passes the
// Tools field through untouched.
type fakeTool struct{ name string }

func (f fakeTool) Name() string              { return f.name }
func (f fakeTool) Description() string       { return "" }
func (f fakeTool) ArgsSchema() schema.Schema { return schema.Object(nil) }
func (f fakeTool) Invoke(context.Context, map[string]any) (coretools.Result, error) {
	return coretools.Result{}, nil
}

func TestNewFilesystemMiddleware(t *testing.T) {
	backend := backends.NewStateBackend()
	rules := []backends.PermissionRule{{
		Operations: []string{"read"},
		Paths:      []string{"/secrets/**"},
		Mode:       backends.PermissionDeny,
	}}
	m, err := NewFilesystemMiddleware(backend, rules...)
	if err != nil {
		t.Fatalf("NewFilesystemMiddleware: %v", err)
	}
	if m.Backend != backend {
		t.Fatal("Backend should be the backend passed to the constructor")
	}
	if !reflect.DeepEqual(m.Permissions, rules) {
		t.Fatalf("Permissions = %v, want %v", m.Permissions, rules)
	}
	if m.Tools != nil {
		t.Fatalf("Tools = %v, want nil until Task 5 fills it", m.Tools)
	}
	if got := m.ProvidedTools(); got != nil {
		t.Fatalf("ProvidedTools() = %v, want nil while Tools is nil", got)
	}
}

func TestProvidedToolsReturnsToolsFieldAsIs(t *testing.T) {
	m, err := NewFilesystemMiddleware(backends.NewStateBackend())
	if err != nil {
		t.Fatalf("NewFilesystemMiddleware: %v", err)
	}
	tools := []coretools.Tool{fakeTool{"ls"}, fakeTool{"read_file"}}
	m.Tools = tools
	if got := m.ProvidedTools(); !sameSlice(got, tools) {
		t.Fatalf("ProvidedTools() = %v, want the Tools field unchanged", got)
	}
}

// sameSlice reports whether got and want are the same slice (shared backing
// array), so the field is returned as-is rather than copied.
func sameSlice[T any](got, want []T) bool {
	if len(got) != len(want) {
		return false
	}
	return len(got) == 0 || &got[0] == &want[0]
}

func TestFilesystemMiddlewareCheck(t *testing.T) {
	denySecrets := backends.PermissionRule{
		Operations: []string{"read"},
		Paths:      []string{"/secrets/**"},
		Mode:       backends.PermissionDeny,
	}
	denyWritesToLocked := backends.PermissionRule{
		Operations: []string{"write"},
		Paths:      []string{"/locked/**"},
		Mode:       backends.PermissionDeny,
	}
	interruptReviews := backends.PermissionRule{
		Operations: []string{"read", "write"},
		Paths:      []string{"/review/**", "/notes/**"},
		Mode:       backends.PermissionInterrupt,
	}
	allowPublicFirst := []backends.PermissionRule{
		{Operations: []string{"read", "write"}, Paths: []string{"/public/**"}, Mode: backends.PermissionAllow},
		{Operations: []string{"read", "write"}, Paths: []string{"/public/secret.txt"}, Mode: backends.PermissionDeny},
	}

	tests := []struct {
		name      string
		rules     []backends.PermissionRule
		tool      string
		args      map[string]any
		wantMode  backends.PermissionMode
		wantMatch bool
	}{
		// Default allow: no rules configured.
		{"no rules allows reads", nil, "read_file",
			map[string]any{"file_path": "/secrets/key.pem"}, backends.PermissionAllow, false},
		{"no rules allows writes", nil, "write_file",
			map[string]any{"file_path": "/locked/a.txt"}, backends.PermissionAllow, false},

		// Read-op deny: every read-class tool is denied under /secrets/**.
		{"read_file denied under secrets", []backends.PermissionRule{denySecrets}, "read_file",
			map[string]any{"file_path": "/secrets/key.pem"}, backends.PermissionDeny, true},
		{"ls denied under secrets", []backends.PermissionRule{denySecrets}, "ls",
			map[string]any{"path": "/secrets"}, backends.PermissionDeny, true},
		{"glob denied under secrets", []backends.PermissionRule{denySecrets}, "glob",
			map[string]any{"path": "/secrets", "pattern": "**"}, backends.PermissionDeny, true},
		{"grep denied under secrets", []backends.PermissionRule{denySecrets}, "grep",
			map[string]any{"path": "/secrets", "pattern": "token"}, backends.PermissionDeny, true},
		{"read rule does not deny writes", []backends.PermissionRule{denySecrets}, "write_file",
			map[string]any{"file_path": "/secrets/key.pem"}, backends.PermissionAllow, false},

		// Write-op deny: every write-class tool is denied under /locked/**,
		// reads there are unaffected.
		{"write_file denied under locked", []backends.PermissionRule{denyWritesToLocked}, "write_file",
			map[string]any{"file_path": "/locked/a.txt"}, backends.PermissionDeny, true},
		{"edit_file denied under locked", []backends.PermissionRule{denyWritesToLocked}, "edit_file",
			map[string]any{"file_path": "/locked/a.txt", "old_string": "a", "new_string": "b"}, backends.PermissionDeny, true},
		{"delete denied under locked", []backends.PermissionRule{denyWritesToLocked}, "delete",
			map[string]any{"path": "/locked/a.txt"}, backends.PermissionDeny, true},
		{"write rule does not deny reads", []backends.PermissionRule{denyWritesToLocked}, "read_file",
			map[string]any{"file_path": "/locked/a.txt"}, backends.PermissionAllow, false},

		// Path filtering: a matching operation on a non-matching path falls
		// through to the default.
		{"non-matching path allowed", []backends.PermissionRule{denyWritesToLocked}, "write_file",
			map[string]any{"file_path": "/other/a.txt"}, backends.PermissionAllow, false},

		// First-match-wins: an earlier allow shadows a later deny for the
		// same call.
		{"first matching rule wins", allowPublicFirst, "read_file",
			map[string]any{"file_path": "/public/secret.txt"}, backends.PermissionAllow, true},
		{"later rule still applies elsewhere", allowPublicFirst, "read_file",
			map[string]any{"file_path": "/private/secret.txt"}, backends.PermissionAllow, false},

		// Interrupt mode surfaces as its own disposition.
		{"interrupt on write", []backends.PermissionRule{interruptReviews}, "write_file",
			map[string]any{"file_path": "/review/draft.md"}, backends.PermissionInterrupt, true},
		// A rule's Paths is an any-match list.
		{"any pattern in rule paths matches", []backends.PermissionRule{interruptReviews}, "read_file",
			map[string]any{"file_path": "/notes/a.txt"}, backends.PermissionInterrupt, true},

		// Unknown tools and unusable path arguments never match.
		{"unknown tool never matches", []backends.PermissionRule{denySecrets}, "execute",
			map[string]any{"command": "ls"}, backends.PermissionAllow, false},
		{"missing path arg never matches", []backends.PermissionRule{denySecrets}, "read_file",
			map[string]any{}, backends.PermissionAllow, false},
		{"non-string path arg never matches", []backends.PermissionRule{denySecrets}, "read_file",
			map[string]any{"file_path": 42}, backends.PermissionAllow, false},
		{"nil args never match", []backends.PermissionRule{denySecrets}, "ls",
			nil, backends.PermissionAllow, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewFilesystemMiddleware(backends.NewStateBackend(), tt.rules...)
			if err != nil {
				t.Fatalf("NewFilesystemMiddleware: %v", err)
			}
			mode, matched := m.Check(tt.tool, tt.args)
			if mode != tt.wantMode || matched != tt.wantMatch {
				t.Fatalf("Check(%q, %v) = (%q, %v), want (%q, %v)",
					tt.tool, tt.args, mode, matched, tt.wantMode, tt.wantMatch)
			}
		})
	}
}

func TestFilesystemMiddlewareWrapToolCallInjectsStateView(t *testing.T) {
	seed := map[string]backends.FileMeta{
		"/a.txt":   {Content: "alpha", Encoding: "utf-8"},
		"/b/c.txt": {Content: "beta", Encoding: "utf-8"},
	}
	state := map[string]any{backends.FilesStateKey: seed}
	request := agentsmw.ToolCallRequest{
		ToolCall: agentsmw.ToolCall{Name: "read_file", Args: map[string]any{"file_path": "/a.txt"}},
		State:    state,
	}
	m, err := NewFilesystemMiddleware(backends.NewStateBackend())
	if err != nil {
		t.Fatalf("NewFilesystemMiddleware: %v", err)
	}

	var gotFiles map[string]backends.FileMeta
	var gotState map[string]any
	msg, err := m.WrapToolCall(t.Context(), request, func(ctx context.Context, req agentsmw.ToolCallRequest) (messages.Message, error) {
		view, ok := backends.StateViewFromContext(ctx)
		if !ok {
			t.Fatal("handler context should carry a StateView for a StateBackend")
		}
		gotFiles = view.Files()
		gotState = req.State
		return messages.Message{Role: messages.RoleTool, Content: "ok"}, nil
	})
	if err != nil {
		t.Fatalf("WrapToolCall: %v", err)
	}
	if msg.Content != "ok" {
		t.Fatalf("handler message should pass through, got %+v", msg)
	}
	if !reflect.DeepEqual(gotFiles, seed) {
		t.Fatalf("view files = %v, want the seeded files %v", gotFiles, seed)
	}
	if !reflect.DeepEqual(gotState, state) {
		t.Fatalf("handler state = %v, want the request state passed through unchanged", gotState)
	}
}

func TestFilesystemMiddlewareWrapToolCallEmptyViewWithoutFiles(t *testing.T) {
	m, err := NewFilesystemMiddleware(backends.NewStateBackend())
	if err != nil {
		t.Fatalf("NewFilesystemMiddleware: %v", err)
	}

	cases := []struct {
		name  string
		state map[string]any
	}{
		{"state without files key", map[string]any{"todos": "x"}},
		{"nil state", nil},
		{"files key of wrong type", map[string]any{backends.FilesStateKey: "not a map"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := agentsmw.ToolCallRequest{
				ToolCall: agentsmw.ToolCall{Name: "ls", Args: map[string]any{"path": "/"}},
				State:    tc.state,
			}
			_, err := m.WrapToolCall(t.Context(), request, func(ctx context.Context, _ agentsmw.ToolCallRequest) (messages.Message, error) {
				view, ok := backends.StateViewFromContext(ctx)
				if !ok {
					t.Fatal("handler context should carry a StateView even with no files")
				}
				if files := view.Files(); len(files) != 0 {
					t.Fatalf("view files = %v, want empty", files)
				}
				return messages.Message{}, nil
			})
			if err != nil {
				t.Fatalf("WrapToolCall: %v", err)
			}
		})
	}
}

func TestFilesystemMiddlewareWrapToolCallFreshViewPerCall(t *testing.T) {
	m, err := NewFilesystemMiddleware(backends.NewStateBackend())
	if err != nil {
		t.Fatalf("NewFilesystemMiddleware: %v", err)
	}
	state := map[string]any{
		backends.FilesStateKey: map[string]backends.FileMeta{"/keep.txt": {Content: "kept"}},
	}
	request := agentsmw.ToolCallRequest{
		ToolCall: agentsmw.ToolCall{Name: "write_file", Args: map[string]any{"file_path": "/x.txt", "content": "x"}},
		State:    state,
	}

	// First call mutates its view.
	_, err = m.WrapToolCall(t.Context(), request, func(ctx context.Context, _ agentsmw.ToolCallRequest) (messages.Message, error) {
		view, _ := backends.StateViewFromContext(ctx)
		view.Set("/x.txt", backends.FileMeta{Content: "x"})
		return messages.Message{}, nil
	})
	if err != nil {
		t.Fatalf("WrapToolCall: %v", err)
	}

	// Second call must see a fresh view: only the seeded file.
	_, err = m.WrapToolCall(t.Context(), request, func(ctx context.Context, _ agentsmw.ToolCallRequest) (messages.Message, error) {
		view, _ := backends.StateViewFromContext(ctx)
		if _, leaked := view.Files()["/x.txt"]; leaked {
			t.Fatal("second call's view leaked the first call's mutation; views must be fresh per call")
		}
		if len(view.Files()) != 1 {
			t.Fatalf("view files = %v, want only the seeded key", view.Files())
		}
		if view.Changed() != nil {
			t.Fatalf("fresh view should have no changes, got %v", view.Changed())
		}
		return messages.Message{}, nil
	})
	if err != nil {
		t.Fatalf("WrapToolCall: %v", err)
	}
}

func TestFilesystemMiddlewareWrapToolCallPassthroughForNonStateBackend(t *testing.T) {
	backend, err := backends.NewFilesystemBackend(t.TempDir())
	if err != nil {
		t.Fatalf("NewFilesystemBackend: %v", err)
	}
	m, err := NewFilesystemMiddleware(backend)
	if err != nil {
		t.Fatalf("NewFilesystemMiddleware: %v", err)
	}
	request := agentsmw.ToolCallRequest{
		ToolCall: agentsmw.ToolCall{Name: "ls", Args: map[string]any{"path": "/"}},
		State:    map[string]any{backends.FilesStateKey: map[string]backends.FileMeta{}},
	}

	called := false
	_, err = m.WrapToolCall(t.Context(), request, func(ctx context.Context, req agentsmw.ToolCallRequest) (messages.Message, error) {
		called = true
		if _, ok := backends.StateViewFromContext(ctx); ok {
			t.Fatal("non-StateBackend backend must not get a StateView injected")
		}
		if req.State == nil {
			t.Fatal("request should pass through unchanged")
		}
		return messages.Message{}, nil
	})
	if err != nil {
		t.Fatalf("WrapToolCall: %v", err)
	}
	if !called {
		t.Fatal("handler must still be invoked for non-StateBackend backends")
	}
}

func TestFilesCommand(t *testing.T) {
	t.Run("nil view yields nil command", func(t *testing.T) {
		if cmd := filesCommand(nil); cmd != nil {
			t.Fatalf("filesCommand(nil) = %+v, want nil", cmd)
		}
	})

	t.Run("view without changes yields nil command", func(t *testing.T) {
		view := backends.NewStateView(map[string]backends.FileMeta{"/a.txt": {Content: "a"}})
		if cmd := filesCommand(view); cmd != nil {
			t.Fatalf("filesCommand on unchanged view = %+v, want nil", cmd)
		}
	})

	t.Run("update carries only the files key", func(t *testing.T) {
		view := backends.NewStateView(nil)
		view.Set("/a.txt", backends.FileMeta{Content: "alpha"})
		view.Set("/b.txt", backends.FileMeta{Content: "beta"})
		view.Delete("/gone.txt")

		cmd := filesCommand(view)
		if cmd == nil {
			t.Fatal("changed view should yield a command")
		}
		if cmd.Goto != nil || cmd.Resume != nil || cmd.Graph != "" {
			t.Fatalf("command should be update-only, got %+v", cmd)
		}
		keys := make([]string, 0, len(cmd.Update))
		for k := range cmd.Update {
			keys = append(keys, k)
		}
		if len(keys) != 1 || keys[0] != backends.FilesStateKey {
			t.Fatalf("update keys = %v, want exactly [%s]", keys, backends.FilesStateKey)
		}
		if _, hasMessages := cmd.Update["messages"]; hasMessages {
			t.Fatal("filesCommand must never emit a messages key")
		}
		want := map[string]*backends.FileMeta{
			"/a.txt":    {Content: "alpha"},
			"/b.txt":    {Content: "beta"},
			"/gone.txt": nil,
		}
		got, ok := cmd.Update[backends.FilesStateKey].(map[string]*backends.FileMeta)
		if !ok {
			t.Fatalf("update[files] has type %T, want map[string]*backends.FileMeta", cmd.Update[backends.FilesStateKey])
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("update[files] = %v, want %v", got, want)
		}
	})
}

func TestFilesystemMiddlewareStateSchema(t *testing.T) {
	t.Run("state backend registers files field with merge reducer", func(t *testing.T) {
		m, err := NewFilesystemMiddleware(backends.NewStateBackend())
		if err != nil {
			t.Fatalf("NewFilesystemMiddleware: %v", err)
		}
		fields := m.StateSchema()
		if len(fields) != 1 {
			t.Fatalf("StateSchema() = %v, want exactly one field", fields)
		}
		if fields[0].Name != backends.FilesStateKey {
			t.Fatalf("field name = %q, want %q", fields[0].Name, backends.FilesStateKey)
		}
		if fields[0].Reducer == nil {
			t.Fatal("field reducer must be backends.MapMergeReducer, got nil")
		}
		if reflect.ValueOf(fields[0].Reducer).Pointer() != reflect.ValueOf(backends.MapMergeReducer).Pointer() {
			t.Fatal("field reducer must be backends.MapMergeReducer")
		}
	})

	t.Run("non-state backend contributes no state", func(t *testing.T) {
		backend, err := backends.NewFilesystemBackend(t.TempDir())
		if err != nil {
			t.Fatalf("NewFilesystemBackend: %v", err)
		}
		m, err := NewFilesystemMiddleware(backend)
		if err != nil {
			t.Fatalf("NewFilesystemMiddleware: %v", err)
		}
		if fields := m.StateSchema(); len(fields) != 0 {
			t.Fatalf("StateSchema() = %v, want nil for non-StateBackend", fields)
		}
	})
}
