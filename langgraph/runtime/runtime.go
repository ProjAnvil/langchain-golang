// Package runtime defines the first-class Runtime value that the langgraph
// executor injects into every node function and conditional edge router,
// mirroring Python's `langgraph.runtime.Runtime[ContextT]` (added in v0.6.0).
//
// Runtime bundles run-scoped context and other run utilities: the static
// Context (user_id, db_conn, ...), a cross-thread Store, a StreamWriter, a
// Heartbeat callback, the functional-API Previous value, read-only
// ExecutionInfo, optional ServerInfo, and a RunControl surface for cooperative
// draining.
//
// Runtime satisfies the standard context.Context interface: Deadline, Done,
// Err, and Value all delegate to an unexported ctx field captured at
// construction (see NewRuntime). The Context field — the run-scoped static
// context carried by context_schema — is a SEPARATE concern from the
// context.Context the executor threads; the two deliberately do not collide
// (the field is named `Context` and is an `any`, while the backing
// context.Context is unexported). This is why Runtime does NOT embed
// context.Context as a named field.
//
// This is a Python->Go port: the field set, Merge/Override/PatchExecutionInfo
// semantics, and the no-op defaults mirror `langgraph/runtime.py` exactly.
// See https://github.com/langchain-ai/langgraph/blob/main/langgraph/runtime.py
package runtime

import (
	"context"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/store"
)

// Store is the cross-thread store interface Runtime exposes to nodes,
// mirroring Python's `langgraph.store.base.BaseStore`. It is an alias for the
// real store.Store interface defined in langgraph/store (M1.2); the alias
// keeps runtime.Store and WithRuntimeStore as the canonical names consumers
// already use, while delegating the contract to the store package. The store
// package must NOT import runtime (it does not) to avoid an import cycle.
type Store = store.Store

// StreamWriter writes a custom payload into an active stream, mirroring
// Python's `langgraph.types.StreamWriter` callable. Writes after the run has
// ended are silently dropped (never panic). The zero/nil value is treated as
// a no-op by NewRuntime and Merge.
type StreamWriter func(payload any)

// NoOpStreamWriter is a no-op StreamWriter consumers may install explicitly,
// mirroring Python's `_no_op_stream_writer`. NewRuntime does NOT install it:
// it leaves StreamWriter nil so Merge can detect "unset" via a nil check
// (Go function values can only be compared to nil, not to each other, so the
// Python identity-comparison sentinel cannot be reproduced). Consumers that
// want an always-callable writer can install NoOpStreamWriter themselves, or
// nil-check before calling (the Go-idiomatic pattern, matching
// graph.StreamWriterFromContext).
var NoOpStreamWriter StreamWriter = func(_ any) {}

// NoOpHeartbeat is the no-op analogue of NoOpStreamWriter for Heartbeat.
var NoOpHeartbeat func() = func() {}

// streamWriterSet reports whether w is a non-default writer. Because Go
// cannot compare function values for identity (only against nil), the
// "default/unset" sentinel is nil itself — NewRuntime leaves StreamWriter
// nil, and Merge keeps the receiver's writer when other's is nil. This is a
// documented divergence from Python (which uses the _no_op_stream_writer
// sentinel).
func streamWriterSet(w StreamWriter) bool { return w != nil }

// heartbeatSet is the Heartbeat analogue of streamWriterSet.
func heartbeatSet(h func()) bool { return h != nil }

// ExecutionInfo is read-only execution metadata for the current
// thread/run/node, mirroring Python's `ExecutionInfo` (runtime.py:26). It is
// a value type: use Patch to derive a copy with overridden fields.
type ExecutionInfo struct {
	// CheckpointID is the checkpoint ID for the current execution.
	CheckpointID string
	// CheckpointNS is the checkpoint namespace for the current execution
	// (empty for the top-level graph; non-empty inside a subgraph).
	CheckpointNS string
	// TaskID is the task ID for the current execution.
	TaskID string
	// ThreadID is the thread ID for the current execution. Empty when
	// running without a checkpointer (i.e. no persistence).
	ThreadID string
	// RunID is the run ID for the current execution. Empty when not
	// provided in the runnable config.
	RunID string
	// NodeAttempt is the current node execution attempt number (1-indexed).
	NodeAttempt int
	// NodeFirstAttemptTime is the wall-clock time the first attempt
	// started, or nil when unset.
	NodeFirstAttemptTime *time.Time
}

// ExecutionInfoOverride configures ExecutionInfo.Patch (and, via
// Runtime.PatchExecutionInfo, runtime patching). Each override mutates the
// in-progress copy; pass one per field you want to replace.
type ExecutionInfoOverride func(*ExecutionInfo)

// WithCheckpointID overrides ExecutionInfo.CheckpointID.
func WithCheckpointID(v string) ExecutionInfoOverride {
	return func(e *ExecutionInfo) { e.CheckpointID = v }
}

// WithCheckpointNS overrides ExecutionInfo.CheckpointNS.
func WithCheckpointNS(v string) ExecutionInfoOverride {
	return func(e *ExecutionInfo) { e.CheckpointNS = v }
}

// WithTaskID overrides ExecutionInfo.TaskID.
func WithTaskID(v string) ExecutionInfoOverride {
	return func(e *ExecutionInfo) { e.TaskID = v }
}

// WithThreadID overrides ExecutionInfo.ThreadID.
func WithThreadID(v string) ExecutionInfoOverride {
	return func(e *ExecutionInfo) { e.ThreadID = v }
}

// WithRunID overrides ExecutionInfo.RunID.
func WithRunID(v string) ExecutionInfoOverride {
	return func(e *ExecutionInfo) { e.RunID = v }
}

// WithNodeAttempt overrides ExecutionInfo.NodeAttempt.
func WithNodeAttempt(v int) ExecutionInfoOverride {
	return func(e *ExecutionInfo) { e.NodeAttempt = v }
}

// WithNodeFirstAttemptTime overrides ExecutionInfo.NodeFirstAttemptTime.
func WithNodeFirstAttemptTime(t *time.Time) ExecutionInfoOverride {
	return func(e *ExecutionInfo) { e.NodeFirstAttemptTime = t }
}

// Patch returns a copy of info with the given overrides applied, mirroring
// Python's `ExecutionInfo.patch(**overrides)` (which uses
// `dataclasses.replace`). Each override replaces the corresponding field
// unconditionally (matching Python's replace, which has no "only if non-zero"
// rule at the field level).
func (info ExecutionInfo) Patch(opts ...ExecutionInfoOverride) ExecutionInfo {
	out := info
	for _, opt := range opts {
		opt(&out)
	}
	return out
}

// ServerInfo is metadata injected by LangGraph Server, mirroring Python's
// `ServerInfo` (runtime.py:61). It is nil when running open-source LangGraph
// without a LangSmith deployment.
type ServerInfo struct {
	// AssistantID is the assistant ID for the current execution.
	AssistantID string
	// GraphID is the graph ID for the current execution.
	GraphID string
	// User is the authenticated user, if any. Typed as any (rather than a
	// concrete user type) to keep this package free of the server/auth
	// dependency; consumers type-assert as needed.
	User any
}

// RunControl is the run-scoped control surface for cooperative draining,
// mirroring Python's `RunControl` (runtime.py:79). Create a fresh RunControl
// per run; reusing one after RequestDrain leaves it drained. Safe to call
// from any goroutine: the drain request is a single atomic-ish attribute
// write, so no lock is needed for this signal (matching Python).
type RunControl struct {
	drainReason *string
}

// NewRunControl returns a fresh, un-drained RunControl.
func NewRunControl() *RunControl { return &RunControl{} }

// RequestDrain requests cooperative draining of the current run, mirroring
// Python's `RunControl.request_drain(reason="shutdown")`. After this call,
// DrainRequested reports true and DrainReason returns reason. A no-op when
// already drained (the reason is NOT overwritten — first writer wins — so a
// high-priority "shutdown" reason is not clobbered by a later "timeout").
func (c *RunControl) RequestDrain(reason string) {
	if c == nil || c.drainReason != nil {
		return
	}
	if reason == "" {
		reason = "shutdown"
	}
	c.drainReason = &reason
}

// DrainRequested reports whether a drain has been requested on c, mirroring
// Python's `RunControl.drain_requested` property. Safe on a nil receiver
// (returns false).
func (c *RunControl) DrainRequested() bool {
	return c != nil && c.drainReason != nil
}

// DrainReason returns the drain reason set by RequestDrain, or "" when no
// drain has been requested, mirroring Python's `RunControl.drain_reason`
// property. Safe on a nil receiver (returns "").
func (c *RunControl) DrainReason() string {
	if c == nil || c.drainReason == nil {
		return ""
	}
	return *c.drainReason
}

// Runtime bundles run-scoped context and other runtime utilities, mirroring
// Python's `Runtime[ContextT]` (runtime.py:125). It is injected into graph
// nodes and middleware.
//
// Runtime satisfies context.Context by delegating Deadline/Done/Err/Value to
// an unexported ctx field (see NewRuntime), so existing context.Context
// idioms survive: pass a Runtime wherever a context.Context is expected. The
// Context field below is a SEPARATE concern (the run-scoped static
// context_schema values), which is why context.Context is NOT an embedded
// named field.
type Runtime struct {
	ctx context.Context // backs Deadline/Done/Err/Value; never nil after NewRuntime

	// Context is the run-scoped static context (user_id, db_conn, ...),
	// populated from context_schema values at construction time. Mirrors
	// Python's `Runtime.context`. Default nil.
	Context any
	// Store is the cross-thread BaseStore, enabling persistence and memory
	// shared across threads, mirroring Python's `Runtime.store`. Populated
	// from a compile-option store (see graph.WithStore /
	// fn.EntrypointOpts.Store); nil when no store is configured, in which
	// case consumers must nil-check before use.
	Store Store
	// StreamWriter writes a custom payload into an active stream. Mirrors
	// Python's `Runtime.stream_writer`. Default nil; consumers must nil-check
	// before calling (the Go-idiomatic pattern, matching
	// graph.StreamWriterFromContext). Install NoOpStreamWriter explicitly to
	// get an always-callable no-op writer.
	StreamWriter StreamWriter
	// Heartbeat refreshes the current node's idle_timeout. Mirrors Python's
	// `Runtime.heartbeat`. Default nil; consumers must nil-check before
	// calling. Install NoOpHeartbeat explicitly to get an always-callable
	// no-op.
	Heartbeat func()
	// Previous is the previous return value for the thread, used by the
	// functional API when a checkpointer is provided. Mirrors Python's
	// `Runtime.previous`. Default nil.
	Previous any
	// ExecutionInfo is read-only execution metadata. Nil before task
	// preparation populates it. Mirrors Python's `Runtime.execution_info`.
	ExecutionInfo *ExecutionInfo
	// ServerInfo is metadata injected by LangGraph Server. Nil when running
	// outside LangGraph Server. Mirrors Python's `Runtime.server_info`.
	ServerInfo *ServerInfo
	// Control is the run-scoped cooperative-draining control plane.
	// Populated automatically during graph runs; nil outside an active
	// graph runtime. Mirrors Python's `Runtime.control`.
	Control *RunControl
}

// Compile-time assertion that Runtime satisfies context.Context. The four
// methods below implement the interface by delegating to the unexported ctx
// field.
var _ context.Context = Runtime{}

// Deadline returns the time and ok flag reported by the backing
// context.Context (see NewRuntime). Required by context.Context.
func (r Runtime) Deadline() (time.Time, bool) {
	if r.ctx == nil {
		return time.Time{}, false
	}
	return r.ctx.Deadline()
}

// Done returns the channel closed when the backing context.Context is
// cancelled. Required by context.Context.
func (r Runtime) Done() <-chan struct{} {
	if r.ctx == nil {
		return nil
	}
	return r.ctx.Done()
}

// Err returns the error reported by the backing context.Context after Done
// is closed. Required by context.Context.
func (r Runtime) Err() error {
	if r.ctx == nil {
		return nil
	}
	return r.ctx.Err()
}

// Value returns the value associated with key on the backing context.Context.
// Required by context.Context. Interrupt state, stream writer, event sink,
// and context_schema values all live as context values on the backing ctx, so
// rt.Value(key) reaches every value a plain ctx.Value(key) would.
func (r Runtime) Value(key any) any {
	if r.ctx == nil {
		return nil
	}
	return r.ctx.Value(key)
}

// NewRuntime constructs a Runtime backed by ctx with nil StreamWriter and
// Heartbeat (consumers nil-check before calling) and zero values for the
// remaining fields. ctx is captured for context.Context delegation:
// Deadline/Done/Err/Value on the returned Runtime reach ctx. A nil ctx is
// normalized to context.Background() so the methods are always safe to call.
//
// Documented divergence from Python: Python's default stream_writer/heartbeat
// are no-op callables so rt.stream_writer(...) is always safe; Go function
// values can only be compared to nil, so NewRuntime uses nil as the "unset"
// sentinel and consumers nil-check before calling. Install NoOpStreamWriter
// or NoOpHeartbeat explicitly to recover an always-callable no-op.
func NewRuntime(ctx context.Context) Runtime {
	if ctx == nil {
		ctx = context.Background()
	}
	return Runtime{ctx: ctx}
}

// Merge returns a new Runtime combining r and other, mirroring Python's
// `Runtime.merge`. For each field, other's value wins when it is "set",
// otherwise r's value is kept. "Set" follows Python's per-field rules:
//
//   - Context: other wins when non-nil.
//   - Store: other wins when non-nil.
//   - StreamWriter: other wins when non-nil (documented divergence — Python
//     checks "is not _no_op_stream_writer"; Go uses nil since function values
//     are not comparable).
//   - Heartbeat: other wins when non-nil (same divergence as StreamWriter).
//   - Previous: r wins when other's is nil; otherwise other wins.
//   - ExecutionInfo / ServerInfo / Control: other wins when non-nil.
//
// The backing ctx is other's when non-nil, else r's, so the merged Runtime's
// context.Context delegation tracks the most-derived run (typically other).
func (r Runtime) Merge(other Runtime) Runtime {
	out := r
	if other.ctx != nil {
		out.ctx = other.ctx
	}
	if other.Context != nil {
		out.Context = other.Context
	}
	if other.Store != nil {
		out.Store = other.Store
	}
	if streamWriterSet(other.StreamWriter) {
		out.StreamWriter = other.StreamWriter
	}
	if heartbeatSet(other.Heartbeat) {
		out.Heartbeat = other.Heartbeat
	}
	if other.Previous != nil {
		out.Previous = other.Previous
	}
	if other.ExecutionInfo != nil {
		out.ExecutionInfo = other.ExecutionInfo
	}
	if other.ServerInfo != nil {
		out.ServerInfo = other.ServerInfo
	}
	if other.Control != nil {
		out.Control = other.Control
	}
	return out
}

// OverrideOption configures Runtime.Override (one per overridable field).
type OverrideOption func(*Runtime)

// WithRuntimeCtx replaces the backing context.Context (the one Deadline/Done/
// Err/Value delegate to). Exported so packages that derive a child context
// (e.g. fn/task's goroutine, which wraps the task context in a
// call-path value and a timeout) can preserve the parent Runtime's fields
// while swapping the backing ctx to the derived one. A nil ctx is ignored.
func WithRuntimeCtx(ctx context.Context) OverrideOption {
	return func(r *Runtime) {
		if ctx != nil {
			r.ctx = ctx
		}
	}
}

// WithContext sets Runtime.Context.
func WithRuntimeContext(v any) OverrideOption {
	return func(r *Runtime) { r.Context = v }
}

// WithRuntimeStore sets Runtime.Store.
func WithRuntimeStore(s Store) OverrideOption {
	return func(r *Runtime) { r.Store = s }
}

// WithRuntimeStreamWriter sets Runtime.StreamWriter.
func WithRuntimeStreamWriter(w StreamWriter) OverrideOption {
	return func(r *Runtime) { r.StreamWriter = w }
}

// WithRuntimeHeartbeat sets Runtime.Heartbeat.
func WithRuntimeHeartbeat(h func()) OverrideOption {
	return func(r *Runtime) { r.Heartbeat = h }
}

// WithRuntimePrevious sets Runtime.Previous.
func WithRuntimePrevious(v any) OverrideOption {
	return func(r *Runtime) { r.Previous = v }
}

// WithRuntimeExecutionInfo sets Runtime.ExecutionInfo.
func WithRuntimeExecutionInfo(ei *ExecutionInfo) OverrideOption {
	return func(r *Runtime) { r.ExecutionInfo = ei }
}

// WithRuntimeServerInfo sets Runtime.ServerInfo.
func WithRuntimeServerInfo(si *ServerInfo) OverrideOption {
	return func(r *Runtime) { r.ServerInfo = si }
}

// WithRuntimeControl sets Runtime.Control.
func WithRuntimeControl(c *RunControl) OverrideOption {
	return func(r *Runtime) { r.Control = c }
}

// Override returns a copy of r with the given overrides applied, mirroring
// Python's `Runtime.override(**overrides)`. Each option replaces the
// corresponding field unconditionally (matching Python's
// `dataclasses.replace`).
func (r Runtime) Override(opts ...OverrideOption) Runtime {
	out := r
	for _, opt := range opts {
		opt(&out)
	}
	return out
}

// PatchExecutionInfo returns a copy of r whose ExecutionInfo has the given
// overrides applied, mirroring Python's `Runtime.patch_execution_info`. It
// returns an error when r.ExecutionInfo is nil — Python raises RuntimeError
// for the same condition.
func (r Runtime) PatchExecutionInfo(opts ...ExecutionInfoOverride) (Runtime, error) {
	if r.ExecutionInfo == nil {
		return Runtime{}, fmt.Errorf("runtime: cannot patch ExecutionInfo before it has been set")
	}
	patched := r.ExecutionInfo.Patch(opts...)
	out := r
	out.ExecutionInfo = &patched
	return out, nil
}

// DrainRequested reports whether a cooperative drain has been requested on
// r.Control, mirroring Python's `Runtime.drain_requested` property. Safe on a
// Runtime whose Control is nil (returns false).
func (r Runtime) DrainRequested() bool {
	return r.Control != nil && r.Control.DrainRequested()
}

// DrainReason returns the drain reason set on r.Control, or "" when no drain
// has been requested (including when Control is nil), mirroring Python's
// `Runtime.drain_reason` property.
func (r Runtime) DrainReason() string {
	if r.Control == nil {
		return ""
	}
	return r.Control.DrainReason()
}

// contextSchemaKey is the context.Context key under which the run-scoped
// context_schema values map is stashed, mirroring the role of Python's
// context_schema data flow. The key type is shared between the langchain/agents
// package (which attaches values at invoke time) and the langgraph/graph
// package (which copies them onto Runtime.Context at build time); defining it
// here avoids an import cycle (graph cannot import agents).
type contextSchemaKey struct{}

// ContextWithValues attaches a bag of named context_schema values to ctx,
// mirroring Python's `invoke(input, context=...)`. This is the shared key the
// executor's buildRuntime reads to populate Runtime.Context. The agents
// package's WithContextValues delegates here.
func ContextWithValues(ctx context.Context, values map[string]any) context.Context {
	return context.WithValue(ctx, contextSchemaKey{}, values)
}

// ValuesFromContext returns the context_schema values map attached to ctx via
// ContextWithValues, or nil when no map was attached. The agents package's
// ContextValue/Ruby helpers delegate here.
func ValuesFromContext(ctx context.Context) map[string]any {
	if ctx == nil {
		return nil
	}
	m, _ := ctx.Value(contextSchemaKey{}).(map[string]any)
	return m
}

// ValueFromRuntime reads one named context_schema field from rt.Context,
// returning the value and true when the key is present; otherwise (nil, false).
// Mirrors the agents.ContextValue contract but operates on a Runtime (the
// value a node function receives post-M1.1) rather than a bare
// context.Context.
func ValueFromRuntime(rt Runtime, key string) (any, bool) {
	m, _ := rt.Context.(map[string]any)
	if m == nil {
		return nil, false
	}
	v, ok := m[key]
	return v, ok
}
