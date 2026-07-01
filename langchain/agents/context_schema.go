package agents

// This file implements the context_schema half of Step 3c's state_schema +
// context_schema support for CreateAgent (see
// migration_plan/state-schema-design.md).
//
// Design (per spec): the Go equivalent of Python's `context_schema` is a
// read-only bag of named runtime values carried on the standard
// context.Context that already threads through every node and middleware
// (NodeFunc receives ctx; middleware hooks receive ctx). This keeps
// per-request, cross-cutting data (caller identity, request IDs, tenant,
// etc.) out of the mutable graph state, semantically cleaner than a reserved
// state key and matching Go idiom.
//
// The declarative ContextField/WithAgentContextSchema layer is purely
// documentation + reserved room for future validation today; it does not gate
// WithContextValues/ContextValue, which work whether or not a schema was
// declared. (Python's context_schema is itself just a typed declaration; the
// data flow there is identical — context is passed in at invoke time and read
// inside the graph.)

import "context"

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
// WithContextValues/ContextValue (those work without a schema declared). The
// last call wins, replacing any previously declared schema.
func WithAgentContextSchema(fields ...ContextField) AgentOption {
	return func(o *AgentOptions) { o.ContextSchema = fields }
}

// ctxValuesKey is the unexported context.Context key under which a map of
// runtime-context values is stashed by WithContextValues and read by
// ContextValue. A distinct unexported struct{}-shaped key type avoids
// collisions with any other context value (per the context.Context doc
// comment's guidance against using built-in string keys).
type ctxValuesKey struct{}

// WithContextValues attaches a bag of named runtime-context values to ctx
// (the Go equivalent of passing context= into Python's
// `agent.invoke({"messages": ...}, context=...)`). Call this on the context
// passed to Agent.Invoke / InvokeWithState / InvokeWithStateAndVars; the
// values then reach every node and middleware via the ctx they receive, where
// they are read with ContextValue. Values are read-only from a node's
// perspective. Passing a nil values map is harmless: ContextValue will return
// (nil, false) for every key.
func WithContextValues(ctx context.Context, values map[string]any) context.Context {
	return context.WithValue(ctx, ctxValuesKey{}, values)
}

// ContextValue reads one named runtime-context field inside a node or
// middleware (counterpart to WithContextValues). It returns the value and
// true when the key is present in an attached values map; otherwise
// (nil, false) — including when no values map was attached to ctx at all.
func ContextValue(ctx context.Context, key string) (any, bool) {
	m, ok := ctx.Value(ctxValuesKey{}).(map[string]any)
	if !ok {
		return nil, false
	}
	v, ok := m[key]
	return v, ok
}
