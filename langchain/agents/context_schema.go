package agents

// This file implements the context_schema half of Step 3c's state_schema +
// context_schema support for CreateAgent (see
// migration_plan/state-schema-design.md).
//
// Design (per spec): the Go equivalent of Python's `context_schema` is a
// read-only bag of named runtime values carried on the standard
// context.Context that already threads through every node and middleware.
// This keeps per-request, cross-cutting data (caller identity, request IDs,
// tenant, etc.) out of the mutable graph state, semantically cleaner than a
// reserved state key and matching Go idiom.
//
// M1.1 convergence: the context_schema values now also surface on the
// first-class runtime.Runtime's Context field (see langgraph/runtime). The
// shared key + helpers live in langgraph/runtime so both the executor's
// buildRuntime (which populates Runtime.Context) and the agents helpers below
// (which attach and read the values) share one key without an import cycle
// (graph cannot import agents). WithContextValues/ContextValue delegate to
// runtime.ContextWithValues/runtime.ValuesFromContext; nodes built post-M1.1
// should prefer RuntimeValue(rt, key) / runtime.ValueFromRuntime(rt, key) over
// the deprecated ContextValue.
//
// The declarative ContextField/WithAgentContextSchema layer is purely
// documentation + reserved room for future validation today; it does not gate
// WithContextValues/ContextValue/RuntimeValue, which work whether or not a
// schema was declared. (Python's context_schema is itself just a typed
// declaration; the data flow there is identical — context is passed in at
// invoke time and read inside the graph.)

import (
	"context"

	"github.com/projanvil/langchain-golang/langgraph/runtime"
)

// ContextField declares one named runtime-context field (the Go equivalent
// of one field of Python's context_schema TypedDict). Purely declarative at
// present: it documents the expected fields and reserves room for future
// validation. Type is optional and not yet validated at runtime (YAGNI per
// the spec); pass nil to leave it unset.
type ContextField struct {
	Name string
	Type any // optional, for future validation
}

// WithAgentContextSchema declares the agent's runtime-context schema,
// mirroring Python's `create_agent(context_schema=...)`. Purely declarative
// for now: it records the expected fields on AgentOptions.ContextSchema for
// documentation and inspection but does not enable or restrict
// WithContextValues/ContextValue/RuntimeValue (those work without a schema
// declared). The last call wins, replacing any previously declared schema.
func WithAgentContextSchema(fields ...ContextField) AgentOption {
	return func(o *AgentOptions) { o.ContextSchema = fields }
}

// WithContextValues attaches a bag of named runtime-context values to ctx
// (the Go equivalent of passing context= into Python's
// `agent.invoke({"messages": ...}, context=...)`). Call this on the context
// passed to Agent.Invoke / InvokeWithState / InvokeWithStateAndVars; the
// values then reach every node and middleware via the runtime.Runtime they
// receive (populated on Runtime.Context by the executor's buildRuntime) and
// are read with RuntimeValue or the deprecated ContextValue. Values are
// read-only from a node's perspective. Passing a nil values map is harmless:
// every reader returns (nil, false) for every key.
func WithContextValues(ctx context.Context, values map[string]any) context.Context {
	return runtime.ContextWithValues(ctx, values)
}

// ContextValue reads one named runtime-context field inside a node or
// middleware. It returns the value and true when the key is present in an
// attached values map; otherwise (nil, false) — including when no values map
// was attached to ctx at all.
//
// Deprecated: M1.1 introduces a first-class runtime.Runtime that carries the
// context_schema values on its Context field. New nodes should receive a
// runtime.Runtime and read values via RuntimeValue (or runtime.ValueFromRuntime)
// instead. ContextValue remains functional because the values are still
// attached to the backing context.Context (and Runtime.Value delegates to it),
// but it will not be extended to read Runtime-only fields (Store,
// StreamWriter, ExecutionInfo, ...).
func ContextValue(ctx context.Context, key string) (any, bool) {
	m := runtime.ValuesFromContext(ctx)
	if m == nil {
		return nil, false
	}
	v, ok := m[key]
	return v, ok
}

// RuntimeValue reads one named runtime-context field from rt.Context, the
// context_schema values bag the executor populates on a runtime.Runtime. It
// returns the value and true when the key is present; otherwise (nil, false).
// This is the M1.1 replacement for ContextValue: nodes that receive a
// runtime.Runtime should read context_schema values via RuntimeValue.
func RuntimeValue(rt runtime.Runtime, key string) (any, bool) {
	return runtime.ValueFromRuntime(rt, key)
}
