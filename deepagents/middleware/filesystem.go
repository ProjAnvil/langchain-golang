package middleware

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	"github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/channels"
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
// argument. The names come from the Python reference's tool input models and
// its _FS_TOOL_PATH_ARGS table (deepagents/middleware/filesystem.py:1447-1546
// and deepagents/middleware/_fs_interrupt.py:39-48): ls/glob/grep take
// `path`, read_file/write_file/edit_file/delete take `file_path`.
var fsToolPathArgs = map[string]string{
	"ls":         "path",
	"read_file":  "file_path",
	"write_file": "file_path",
	"edit_file":  "file_path",
	"delete":     "file_path",
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
	// NewFilesystemMiddleware fills it (six tools, plus delete when the
	// backend implements backends.Deleter), and ProvidedTools returns it
	// as-is.
	Tools []coretools.Tool
}

// NewFilesystemMiddleware returns a FilesystemMiddleware over backend with
// the given permission rules (which the caller may also set or replace on the
// struct directly, at their own risk — the constructor-time validation below
// then does not re-run). Each rule's paths are validated like the reference's
// FilesystemPermission.__post_init__, deepagents/middleware/filesystem.py:
// 402-414: a violation fails construction. On success the seven filesystem
// tools (filetools.go) are bound into Tools; delete is included only when
// the backend implements backends.Deleter.
func NewFilesystemMiddleware(backend backends.Backend, permissions ...backends.PermissionRule) (*FilesystemMiddleware, error) {
	for _, rule := range permissions {
		if err := validatePermissionRule(rule); err != nil {
			return nil, err
		}
	}
	m := &FilesystemMiddleware{
		Backend:     backend,
		Permissions: permissions,
	}
	tools, err := m.filesystemTools()
	if err != nil {
		return nil, err
	}
	m.Tools = tools
	return m, nil
}

// validatePermissionRule checks one rule's path patterns, mirroring
// FilesystemPermission.__post_init__ in the Python reference,
// deepagents/middleware/filesystem.py:402-414: every path must start with
// "/", contain no ".." component (after rewriting backslashes to slashes),
// and contain no "~" component, checked in that order. The error wordings
// are the reference's ValueError literals, with {path!r} rendered as a
// single-quoted string.
func validatePermissionRule(rule backends.PermissionRule) error {
	for _, p := range rule.Paths {
		if !strings.HasPrefix(p, "/") {
			return fmt.Errorf("Permission path must start with '/': '%s'", p) //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
		}
		parts := strings.Split(strings.ReplaceAll(p, "\\", "/"), "/")
		if slices.Contains(parts, "..") {
			return fmt.Errorf("Permission path must not contain '..': '%s'", p) //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
		}
		if slices.Contains(parts, "~") {
			return fmt.Errorf("Permission path must not contain '~': '%s'", p) //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
		}
	}
	return nil
}

// ProvidedTools implements agentsmw.ToolProvider: the filesystem tools are
// registered with the agent's ToolNode automatically (factory.py:1005
// collects AgentMiddleware.tools; the Go method name differs because the
// public Tools field must stay and Go forbids a field/method name
// collision). The slice is returned as-is; NewFilesystemMiddleware fills it
// with the bound filesystem tools.
func (m *FilesystemMiddleware) ProvidedTools() []coretools.Tool {
	return m.Tools
}

// StateSchema implements agentsmw.StateSchemaContributor. A
// *backends.StateBackend keeps its file map in agent state, so the `files`
// key is declared with the MapMergeReducer merge strategy (the Go form of
// the reference's FilesystemState.files annotated with _file_data_reducer,
// deepagents/middleware/filesystem.py:1324-1377). The reducer is defense for
// UNMARKED writes only — it overlays the update key-by-key onto the stored
// map; the port's own tools commit Overwrite-wrapped snapshots through
// filesCommand, and the channel replaces its whole value for those (see
// filesCommand). Any other backend touches no agent state and contributes no
// fields.
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
	return m.checkPath(operation, path)
}

// checkPath evaluates the permission rules for one operation on one
// (normalized) path — the core of _check_fs_permission,
// deepagents/middleware/filesystem.py:417-427, shared by the tools' deny
// gates and the search post-filters. Rules are consulted in order; the first
// rule whose Operations contain op and one of whose Paths glob-matches the
// path wins, returning (Mode, true); with no match the call is allowed by
// default.
//
// Note the matcher's conservative direction: backends.MatchGlob treats a
// "**" segment as matching zero or more segments, so a rule pattern like
// "/dir/**" also matches "/dir" itself, denying the directory along with its
// subtree.
func (m *FilesystemMiddleware) checkPath(operation, path string) (backends.PermissionMode, bool) {
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

// WrapToolCall implements agents.WrapToolCallHook. First, the same-batch
// parallel-mutation guard runs (Task 9's wiring of the tested helper, the
// port of the reference's wrap_tool_call guard at filesystem.py:3960, 3990):
// ParallelFileMutationError inspects the batch's AI message in the request
// state and short-circuits a mutating call whose EARLIER batch sibling
// mutates the same normalized path, answering with the upstream error
// ToolMessage instead of invoking the handler. The middleware sees its own
// calls sequentially, but the batch sibling list is already in the state the
// tools node hands each wrapped call (the AI message carrying the whole
// batch), so this is faithful same-batch detection, not a per-superstep
// approximation; what it cannot see is ordering inside a concurrently
// executing batch beyond the AI message's declared call order, which is also
// the order the reference compares against.
//
// Then, for a *backends.StateBackend it snapshots the files map out of the
// request state (request.State[backends.FilesStateKey]; a missing, nil, or
// wrongly typed entry seeds an empty view) into a fresh StateView and
// injects the view into the context via backends.WithStateView before
// invoking the handler, so backend calls made under the wrapped context read
// and write through the view. This stands in for the ToolRuntime state
// injection the Python reference's filesystem tools receive. Each call gets
// its own view: the reference's tools read the state passed to that one
// call, so mutations never leak across calls. Any other backend passes the
// request through unchanged (the guard above still applies).
//
// The middleware only injects. Committing the view's delta back to state is
// the tools' job: each filesystem tool returns a filesCommand in its result
// artifact (Task 5), mirroring the reference's tools sending through their
// ToolRuntime.
func (m *FilesystemMiddleware) WrapToolCall(ctx context.Context, request agentsmw.ToolCallRequest, handler agentsmw.ToolHandler) (messages.Message, error) {
	if msg, blocked := ParallelFileMutationError(request); blocked {
		return msg, nil
	}
	if _, ok := m.Backend.(*backends.StateBackend); !ok {
		return handler(ctx, request)
	}
	files, _ := request.State[backends.FilesStateKey].(map[string]backends.FileMeta)
	view := backends.NewStateView(files)
	return handler(backends.WithStateView(ctx, view), request)
}

// filesCommand builds the state-update Command a filesystem tool returns in
// its result artifact: Update carries only the files key holding the view's
// full post-mutation snapshot — Changed() returns the entire files map after
// the tool's mutations, not a delta, so a view seeded from a raw state still
// commits a complete map — wrapped in a channels.Overwrite marker, so the
// FilesStateKey channel REPLACES its stored map with the snapshot instead of
// overlaying it: a deleted key is absent from the snapshot and therefore
// stays deleted (Spec §4; without the marker, backends.MapMergeReducer would
// overlay key-by-key and the deleted key would resurrect from the stored
// map). MapMergeReducer remains the channel's reducer purely as defense for
// UNMARKED writes (a hand-built Command Update without the marker still
// overlays); the port's own tools always send Overwrite-wrapped snapshots.
// On the channel's first-ever write the binaryOperator seed path unwraps the
// marker, so the stored form is the snapshot map either way
// (langgraph/channels/binop.go, TestBinaryOperatorSeedsOverwriteInnerValue).
// It never emits a messages key — the ToolNode derives the tool message from
// the result content (the port's documented convention,
// langchain/tools/tool_node.go). A nil view, or a view with no recorded
// changes, yields a nil Command: there is nothing to commit, and re-sending
// an unchanged snapshot would be a no-op.
func filesCommand(v *backends.StateView) *types.Command {
	if v == nil {
		return nil
	}
	changed := v.Changed()
	if changed == nil {
		return nil
	}
	return &types.Command{
		Update: map[string]any{backends.FilesStateKey: channels.NewOverwrite(changed)},
	}
}
