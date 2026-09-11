package agents

// This file implements the schema-registry half of Step 3c's state_schema
// support for CreateAgent (see migration_plan/state-schema-design.md). The
// context_schema half lives alongside it in context_schema.go.
//
// Design (per spec, confirmed 2026-07-02): a schema *registry* on top of the
// existing langgraph/graph map[string]any state. Custom state fields are
// declared at agent-build time via WithAgentStateFields; each carries the
// channels.Reducer that merges successive writes to its key. Nodes still
// access state the same way they always have (type assertion on the state
// map), so no existing node or middleware needs to change.
//
// Explicit non-goals (deferred per the spec):
//   - No field-visibility metadata (Python's EphemeralValue / PrivateStateAttr
//     / OmitFromInput). Every Go node sees every state key today.
//   - No typed-accessor helper (GetStateField[T]). Plain type assertion is
//     adequate for Phase 1.
//   - No Initial seed value on StateField. The graph exposes no seed hook for
//     custom fields; nodes tolerate an absent key on first read (Go idiom:
//     `v, ok := state[name]`).
//
// Middleware state-schema contributions ARE supported (see
// middleware.StateSchemaContributor in the middleware package and
// CreateAgent's registration loop): middleware-declared fields register ahead
// of the caller's StateFields so the caller's declaration wins a name
// conflict, mirroring factory.py:1150-1156.

import (
	"github.com/projanvil/langchain-golang/langgraph/channels"
)

// StateField describes one custom graph-state field. It mirrors one field of
// Python's AgentState TypedDict together with the reducer that field is
// annotated with (e.g. `Annotated[list[AnyMessage], add_messages]`).
//
// A StateField whose Name collides with a default AgentState key
// ("messages"/"jump_to"/"structured_response") overrides that key's reducer;
// this is the supported way to swap, for example, the default
// channels.MessagesReducer for a custom merge strategy on "messages".
//
// Reducer may be nil: CreateAgent defaults it to
// channels.LastValueReducer (Python's LastValue "last write wins" channel),
// which is also the implicit reducer langgraph/graph applies to any
// unregistered key, so passing nil and omitting the field are equivalent.
//
// There is deliberately no Initial seed value (see the file doc comment).
type StateField struct {
	Name    string            // state key, e.g. "documents"
	Reducer channels.Reducer  // merge strategy; nil → LastValue (replace)
}

// WithAgentStateFields registers custom state fields, mirroring Python's
// `create_agent(state_schema=...)`. Fields are appended to whatever
// AgentOptions.StateFields already holds (so the option is composable /
// repeatable). The default AgentState already provides "messages"
// (MessagesReducer), "jump_to", and "structured_response" (both LastValue);
// these fields sit alongside them, and a Name collision overrides a default
// key's reducer.
func WithAgentStateFields(fields ...StateField) AgentOption {
	return func(o *AgentOptions) {
		o.StateFields = append(o.StateFields, fields...)
	}
}
