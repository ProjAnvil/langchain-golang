package middleware

import (
	"context"
	"slices"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	"github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Compile-time interface assertions: the middleware plugs into CreateAgent as
// a tool-call wrapper (agents.WrapToolCallHook), a tool provider
// (agentsmw.ToolProvider), and a state-schema contributor
// (agentsmw.StateSchemaContributor).
var (
	_ agents.WrapToolCallHook         = (*FilesystemMiddleware)(nil)
	_ agentsmw.ToolProvider           = (*FilesystemMiddleware)(nil)
	_ agentsmw.StateSchemaContributor = (*FilesystemMiddleware)(nil)
)

// Filesystem operation classes, mirroring FilesystemOperation and
// _DEFAULT_FS_TOOL_OPS in the Python reference, deepagents/middleware/
// filesystem.py:170-182: ls, read_file, glob, and grep are read-only;
// write_file, edit_file, and delete mutate.
const (
	fsOperationRead  = "read"
	fsOperationWrite = "write"
)

// fsToolOperations maps each filesystem tool to its operation class,
// mirroring _DEFAULT_FS_TOOL_OPS in the Python reference,
// deepagents/middleware/filesystem.py:173-182.
var fsToolOperations = map[string]string{
	"ls":         fsOperationRead,
	"read_file":  fsOperationRead,
	"glob":       fsOperationRead,
	"grep":       fsOperationRead,
	"write_file": fsOperationWrite,
	"edit_file":  fsOperationWrite,
	"delete":     fsOperationWrite,
}

// fsToolPathArgs maps each filesystem tool to the name of the argument
// carrying its target path — every filesystem tool carries exactly one path
// argument. The names are pinned by the port plan's tool schemas (E3, from
// the Python reference's tool input models, deepagents/middleware/
// filesystem.py:1447-1546): ls/glob/grep take `path`, read_file/write_file/
// edit_file take `file_path`, and delete takes `path`.
//
// Divergence from the reference, intentional: the reference's DeleteSchema
// names the argument `file_path`, but the port plan pins the Go delete tool's
// argument to `path`, so the evaluator reads `path` to stay in lockstep with
// the Task 5 tools.
var fsToolPathArgs = map[string]string{
	"ls":         "path",
	"read_file":  "file_path",
	"write_file": "file_path",
	"edit_file":  "file_path",
	"delete":     "path",
	"glob":       "path",
	"grep":       "path",
}

// FilesystemMiddleware binds the filesystem tools to a Backend, evaluates
// permission rules for filesystem tool calls, and — for a
// *backends.StateBackend — injects the files state into each tool call's
// context. It mirrors FilesystemMiddleware in the Python reference,
// deepagents/middleware/filesystem.py:1903ff, with the reference's ToolRuntime
// state injection replaced by WrapToolCall context injection (the Go
// replacement documented on backends.WithStateView).
type FilesystemMiddleware struct {
	// Backend is the virtual filesystem the tools operate on.
	Backend backends.Backend
	// Permissions are the permission rules consulted by Check, in order;
	// the first matching rule wins.
	Permissions []backends.PermissionRule
	// Tools holds the filesystem tools bound to this middleware instance.
	// NewFilesystemMiddleware leaves it nil; Task 5's constructor update
	// fills it, and ProvidedTools returns it as-is.
	Tools []coretools.Tool
}

// NewFilesystemMiddleware returns a FilesystemMiddleware over backend with
// the given permission rules (which the caller may also set or replace on the
// struct directly). Tools is left nil in this task; Task 5 fills it by
// binding the seven filesystem tools, which is why the constructor already
// returns an error.
func NewFilesystemMiddleware(backend backends.Backend, permissions ...backends.PermissionRule) (*FilesystemMiddleware, error) {
	return &FilesystemMiddleware{
		Backend:     backend,
		Permissions: permissions,
	}, nil
}

// ProvidedTools implements agentsmw.ToolProvider: the filesystem tools are
// registered with the agent's ToolNode automatically (factory.py:1005
// collects AgentMiddleware.tools; the Go method name differs because the
// public Tools field must stay and Go forbids a field/method name
// collision). The slice is returned as-is and may be empty until Task 5
// fills the Tools field.
func (m *FilesystemMiddleware) ProvidedTools() []coretools.Tool {
	return m.Tools
}

// StateSchema implements agentsmw.StateSchemaContributor. A
// *backends.StateBackend keeps its file map in agent state, so the `files`
// key is declared with the MapMergeReducer merge strategy (the Go form of
// the reference's FilesystemState.files annotated with _file_data_reducer,
// deepagents/middleware/filesystem.py:1324-1377). Any other backend touches
// no agent state and contributes no fields.
func (m *FilesystemMiddleware) StateSchema() []agentsmw.StateField {
	if _, ok := m.Backend.(*backends.StateBackend); !ok {
		return nil
	}
	return []agentsmw.StateField{{
		Name:    backends.FilesStateKey,
		Reducer: backends.MapMergeReducer,
	}}
}

// Check evaluates the middleware's permission rules for a filesystem tool
// call, mirroring _check_fs_permission in the Python reference,
// deepagents/middleware/filesystem.py:417-427. Rules are consulted in order;
// the first rule whose Operations contain the tool's operation class and one
// of whose Paths glob-matches the tool's path argument (backends.MatchGlob)
// wins, returning the rule's Mode and matched=true. When no rule matches,
// the call is allowed by default (matched=false). Calls that are not
// filesystem tool calls, or whose path argument is missing, empty, or not a
// string, never match.
//
// The path is evaluated exactly as passed; the Task 5 tools normalize their
// path arguments before the deny gate consults Check, mirroring the
// reference's validate_path-then-check ordering.
func (m *FilesystemMiddleware) Check(tool string, args map[string]any) (backends.PermissionMode, bool) {
	operation, ok := fsToolOperations[tool]
	if !ok {
		return backends.PermissionAllow, false
	}
	pathArg, ok := fsToolPathArgs[tool]
	if !ok {
		return backends.PermissionAllow, false
	}
	path, _ := args[pathArg].(string)
	if path == "" {
		return backends.PermissionAllow, false
	}
	for _, rule := range m.Permissions {
		if !slices.Contains(rule.Operations, operation) {
			continue
		}
		for _, pattern := range rule.Paths {
			if backends.MatchGlob(pattern, path) {
				return rule.Mode, true
			}
		}
	}
	return backends.PermissionAllow, false
}

// WrapToolCall implements agents.WrapToolCallHook. For a
// *backends.StateBackend it snapshots the files map out of the request state
// (request.State[backends.FilesStateKey]; a missing, nil, or wrongly typed
// entry seeds an empty view) into a fresh StateView and injects the view
// into the context via backends.WithStateView before invoking the handler,
// so backend calls made under the wrapped context read and write through the
// view. This stands in for the ToolRuntime state injection the Python
// reference's filesystem tools receive. Each call gets its own view: the
// reference's tools read the state passed to that one call, so mutations
// never leak across calls. Any other backend passes the request through
// unchanged.
//
// The middleware only injects. Committing the view's delta back to state is
// the tools' job: each filesystem tool returns a filesCommand in its result
// artifact (Task 5), mirroring the reference's tools sending through their
// ToolRuntime.
func (m *FilesystemMiddleware) WrapToolCall(ctx context.Context, request agentsmw.ToolCallRequest, handler agentsmw.ToolHandler) (messages.Message, error) {
	if _, ok := m.Backend.(*backends.StateBackend); !ok {
		return handler(ctx, request)
	}
	files, _ := request.State[backends.FilesStateKey].(map[string]backends.FileMeta)
	view := backends.NewStateView(files)
	return handler(backends.WithStateView(ctx, view), request)
}

// filesCommand builds the state-update Command a filesystem tool returns in
// its result artifact: Update carries only the files key holding the view's
// change delta (nil entries deleting keys), which the graph commits through
// backends.MapMergeReducer on the FilesStateKey channel. It never emits a
// messages key — the ToolNode derives the tool message from the result
// content (the port's documented convention, langchain/tools/tool_node.go).
// A nil view, or a view with no recorded changes, yields a nil Command:
// there is nothing to commit, and sending an empty delta would be a no-op.
func filesCommand(v *backends.StateView) *types.Command {
	if v == nil {
		return nil
	}
	changed := v.Changed()
	if changed == nil {
		return nil
	}
	return &types.Command{
		Update: map[string]any{backends.FilesStateKey: changed},
	}
}
