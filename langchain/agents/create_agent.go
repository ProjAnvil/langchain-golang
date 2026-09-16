package agents

// CreateAgent builds a scoped Go equivalent of Python's
// `langchain.agents.create_agent(...)`: a model node <-> tools node loop
// wired on top of `langgraph/graph`, with middleware hooks composed around
// the model call and each tool call.
//
// Scope note (see migration_plan/core-v1-migration-todo.md P5
// `langchain/agents` for the authoritative list): this deliberately does not
// port subagent transformer behavior (this also requires a middleware-facing
// streaming layer that doesn't exist in this port yet — see
// migration_plan/core-v1-migration-todo.md). Command returned from tools IS
// wired: the tools node consumes the *types.Command a tool places in its
// Result.Artifact (see newToolsNode; Send remains out of scope — see
// langchain/tools/tool_node.go). Interrupts ARE wired through
// CreateAgent: every model-loop hook (BeforeModelHook/BeforeModelCommandHook/
// AfterModelHook/WrapModelCallHook/WrapToolCallHook), not just
// BeforeAgentHook/AfterAgentHook, receives a context.Context a middleware can
// pass to graphpkg.Interrupt to pause the run (see
// create_agent_test.go's TestCreateAgent_InterruptBeforeNode for a
// round-trip example using WithAgentCheckpointer + Agent.Graph.
// InvokeWithOptions to resume). Structured output (`ToolStrategy`/
// `ProviderStrategy`) IS wired via WithAgentResponseFormat — see its doc
// comment for exact scope. Every model call (streaming and non-streaming)
// re-normalizes the request's response_format into an effective strategy
// (factory.py:1323-1344): middleware WithResponseFormat overrides — including
// ToolStrategy↔ProviderStrategy switches — apply per call, an AutoStrategy
// re-resolves against the model of the current call (so a DynamicModel swap
// re-checks it), and the effective strategy drives the bind (final tools +
// tool_choice + model kwargs) and the post-call structured detection. A
// ToolStrategy's HandleErrors retry IS wired:
// a multiple-structured-outputs error or a parse failure injects error
// ToolMessages and loops back to the model when the policy elects retry
// (mirroring factory.py:1204-1270), and raises otherwise. return_direct tools
// ARE honored: when every executed client-side tool call targets a
// return-direct tool (core/tools.ReturnDirecter / Func.WithReturnDirect), the
// loop ends after the tools node instead of returning to the model
// (_make_tools_to_model_edge parity). The compiled graph defaults to
// recursion_limit 9999 (factory.py:1780), and a per-call DynamicModel resolver
// is available (a superset mirroring langgraph.prebuilt's callable-model
// overload). `BeforeAgent`/`AfterAgent` hooks ARE wired:
// when at least one middleware implements BeforeAgentHook/AfterAgentHook/
// AfterAgentUpdateHook, CreateAgent adds dedicated "before_agent"/
// "after_agent" nodes around the model<->tools loop (mirroring Python's
// `before_agent`/`after_agent` running once per run, not once per model
// call); every "end" routing decision (normal completion, a jump_to "end",
// or a structured-output match) is redirected through "after_agent" first
// when present. Middleware-contributed tools and state fields ARE
// auto-collected: middleware implementing middleware.ToolProvider/
// StateSchemaContributor register their tools and state keys with the agent
// without any manual wiring (factory.py:1005, 1054-1055, 1150-1156), and
// duplicate middleware names are rejected at build time (factory.py:
// 1080-1082). The agent's Name, when set, is stamped onto every AI message
// the model produces (factory.py:1418-1419).
//
// Middleware hook discovery: unlike Python's `AgentMiddleware` base class
// (which defines every hook as a no-op an implementation can selectively
// override), Go has no shared base with overridable defaults. CreateAgent
// instead uses type assertions against the *Hook interfaces below, so a
// middleware value need only implement the hooks it cares about.
//
// "jump_to" convention: a BeforeModel/AfterModel/BeforeAgent hook (or an
// AfterAgentUpdateHook) can short-circuit normal routing by setting
// update["jump_to"] to "model", "tools", or "end" (mirroring Python's
// `AgentState.jump_to` field, routed by the middleware conditional edges —
// factory.py:1694-1713 before_agent, :1753-1776 after_agent). CreateAgent
// consumes this key out of the update before merging it into graph state (it
// is never itself persisted). For before_agent, "model" enters the
// model<->tools loop (the default) and "end" exits through "after_agent"
// when present; for after_agent, "model" re-enters the loop and "end"
// (or no jump) completes the run.
//
// BeforeModel "messages" scope note: a BeforeModelHook's returned
// update["messages"], if present, reshapes only the *local* view of the
// conversation used to build this model call (e.g. `SummarizationMiddleware`
// collapsing older messages into a summary); it is intentionally NOT
// persisted into the graph's committed state (MessagesReducer now supports
// RemoveMessage, including the empty-ID remove-all sentinel, but a
// BeforeModelHook "messages" update is deliberately kept local-only). Every
// OTHER key a BeforeModelHook returns IS persisted, mirroring Python where
// before_model hooks run as dedicated graph nodes whose state updates commit.
// AfterModel "messages" updates are additive (new tool/AI messages) and are
// persisted normally.

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/projanvil/langchain-golang/core/caches"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/prompts"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/streamevents"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
	agenttools "github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/types"
	"github.com/projanvil/langchain-golang/modelprofiles"

	// Blank-import partners/openai so "openai:..." resolves out-of-the-box via
	// its init() self-registration with chatmodels. Importing partners/openai
	// here is the Go equivalent of Python's "import langchain_openai" making
	// `model="openai:..."` just work. There is no import cycle: neither
	// partners/openai nor langchain/chatmodels imports langchain/agents
	// (verified). Callers wanting a different provider must blank-import that
	// provider's partner package themselves.
	_ "github.com/projanvil/langchain-golang/partners/openai"
)

// Node names used by the compiled graph, mirroring Python's "model"/"tools"
// node names in `create_agent`. BeforeAgentNodeName/AfterAgentNodeName are
// only added to the graph when at least one middleware implements the
// corresponding hook (see WithAgentMiddleware). HITLNodeName is added when an
// interrupt-mode HumanInTheLoopMiddleware is configured (see
// middleware.NewInterruptHumanInTheLoopMiddleware): the review runs there,
// after the model node's update commits, so pausing never replays the model.
const (
	ModelNodeName       = "model"
	ToolsNodeName       = "tools"
	BeforeAgentNodeName = "before_agent"
	AfterAgentNodeName  = "after_agent"
	HITLNodeName        = "hitl"
)

// BeforeModelHook lets middleware inspect/modify state before the model is
// called, mirroring Python's `AgentMiddleware.before_model`. Returning a
// non-nil update with update["jump_to"] set short-circuits normal routing
// (see the package doc comment). It receives a context.Context (like
// BeforeAgentHook/AfterAgentHook), so it can call graphpkg.Interrupt to pause
// the run for external input (see the package doc comment's Interrupts note).
type BeforeModelHook interface {
	BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error)
}

// BeforeModelCommandHook is an alternative BeforeModel shape for middleware
// that always wants full control over routing (e.g. ModelCallLimitMiddleware
// ending the run outright), returning a middleware.Command directly instead
// of a plain map.
type BeforeModelCommandHook interface {
	BeforeModel(ctx context.Context, state map[string]any) (*middleware.Command, error)
}

// AfterModelHook lets middleware inspect/modify state after the model call,
// mirroring Python's `AgentMiddleware.after_model`. It receives a
// context.Context for the same reason as BeforeModelHook.
type AfterModelHook interface {
	AfterModel(ctx context.Context, state map[string]any) (map[string]any, error)
}

// AfterModelNodeHook is the dedicated-node form of AfterModelHook, mirroring
// Python's per-middleware after_model graph nodes (factory.py:1569): instead
// of running inline inside the model node (where an Interrupt would pause
// BEFORE the model node's update commits, replaying the model call on
// resume), it runs as the dedicated "hitl" node wired after the model node,
// so a pause/resume only ever re-runs this hook. CreateAgent wires the node
// for middleware that ALSO implements middleware.HitlInterrupter with
// HitlInterruptEnabled() true (an interrupt-mode
// HumanInTheLoopMiddleware); at most one such middleware is allowed per
// agent. ctx is the node's runtime (langgraph Runtime implements
// context.Context), so the hook can call graphpkg.Interrupt to pause.
type AfterModelNodeHook interface {
	AfterModelNode(ctx context.Context, state map[string]any) (map[string]any, error)
}

// WrapModelCallHook lets middleware intercept the model call itself,
// mirroring Python's `AgentMiddleware.wrap_model_call`. It receives a
// context.Context for the same reason as BeforeModelHook.
type WrapModelCallHook interface {
	WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error)
}

// WrapModelCallResultHook is the ModelCallResult-returning sibling of
// WrapModelCallHook, mirroring Python middleware whose wrap_model_call
// returns the `ModelCallResult` union (`ModelResponse | AIMessage |
// ExtendedModelResponse`, middleware/types.py:313). A bare messages.Message
// with role ai (the AIMessage short form) is normalized into a ModelResponse
// by middleware.NormalizeModelCallResult at the composition boundary
// (factory._normalize_to_model_response, factory.py:177). When a middleware
// implements both WrapModelCallHook and WrapModelCallResultHook, the Result
// variant takes precedence.
//
// An ExtendedModelResponse may carry a Command: its Update is applied on top
// of the model node's default messages/structured_response state update
// (middleware keys win conflicts, mirroring factory._build_commands'
// `commands.extend` ordering); a Command with Goto/Resume/Graph fails the run
// with an error (middleware.Command.ValidateForWrapModelCall).
type WrapModelCallResultHook interface {
	WrapModelCallResult(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error)
}

// WrapToolCallHook lets middleware intercept a single tool call, mirroring
// Python's `AgentMiddleware.wrap_tool_call`. It receives a context.Context
// for the same reason as BeforeModelHook.
type WrapToolCallHook interface {
	WrapToolCall(ctx context.Context, request middleware.ToolCallRequest, handler middleware.ToolHandler) (messages.Message, error)
}

// BeforeAgentHook lets middleware run once, before the model<->tools loop
// starts, mirroring Python's `AgentMiddleware.before_agent`. It receives a
// context.Context (unlike BeforeModelHook/AfterModelHook) since it runs as
// its own dedicated graph node rather than inline within the model node.
// Returning an update with update["jump_to"] set to "model"/"tools"/"end"
// short-circuits the node's default edge into the model<->tools loop (see
// the package doc comment's jump_to note).
type BeforeAgentHook interface {
	BeforeAgent(ctx context.Context, state map[string]any) (map[string]any, error)
}

// AfterAgentHook lets middleware run once, after the model<->tools loop ends
// (however it ends: normal completion, a jump_to "end", or a matched
// structured-output response), mirroring Python's
// `AgentMiddleware.after_agent`. It is typically used for cleanup (e.g.
// `ShellToolMiddleware` closing a persistent shell session) and, matching
// that existing implementation, does not itself produce a state update.
type AfterAgentHook interface {
	AfterAgent(ctx context.Context, state map[string]any) error
}

// AfterAgentUpdateHook is the update-returning sibling of AfterAgentHook for
// middleware whose after_agent hook produces state (Python's after_agent may
// return a dict). Mirroring Python's after_agent edge wiring
// (factory.py:1753-1776, model_destination=loop entry), an update carrying
// update["jump_to"] = "model" re-enters the model<->tools loop ("end", or no
// jump, finishes the run). A middleware implements AfterAgentHook OR
// AfterAgentUpdateHook — Go forbids two AfterAgent methods on one type — and
// the after_agent node checks the update variant first.
type AfterAgentUpdateHook interface {
	AfterAgent(ctx context.Context, state map[string]any) (map[string]any, error)
}

// MiddlewareNamer is implemented by middleware that carry an explicit
// instance name, mirroring Python's AgentMiddleware.name property
// (middleware/types.py:410-417; the default there is the class name, the Go
// analog of which is the type name — see middlewareName). The name drives
// the duplicate-middleware validation (see CreateAgent).
type MiddlewareNamer interface {
	Name() string
}

// AgentOptions configures CreateAgent.
type AgentOptions struct {
	SystemPrompt string
	// SystemPromptTemplate, when non-nil, renders the system prompt on every
	// model call via core/prompts, mirroring Python's
	// `create_agent(system_prompt: SystemMessage)` templated form. It takes
	// precedence over SystemPrompt when both are set (the two are mutually
	// exclusive in practice; see WithAgentSystemPromptTemplate). Variables are
	// merged from SystemPromptVariables (build-time) and any per-Invoke
	// variables supplied via InvokeWithStateAndVars.
	SystemPromptTemplate  *prompts.PromptTemplate
	SystemPromptVariables map[string]any
	Middleware            []any
	Checkpointer          checkpoint.Saver
	RecursionLimit        int
	// InterruptBefore / InterruptAfter name graph nodes the compiled graph
	// should pause before/after running, mirroring Python's
	// `create_agent(interrupt_before=..., interrupt_after=...)`. The relevant
	// node names are ModelNodeName ("model"), ToolsNodeName ("tools"),
	// BeforeAgentNodeName, and AfterAgentNodeName. A checkpointer
	// (WithAgentCheckpointer) is required for the pauses to be resumable; the
	// run is resumed via Agent.Graph.InvokeWithOptions with the same ThreadID
	// and a nil Resume (mirroring Python's `invoke(None, config)`).
	InterruptBefore []string
	InterruptAfter  []string
	// Name is the agent's run name / tracing tag, mirroring Python's
	// `create_agent(name=...)` (the `lc_agent_name` equivalent). It is stored
	// on the Agent (see Agent.Name) and surfaced through the run-name context
	// on each Invoke, since langgraph/graph has no native run-metadata
	// injection point.
	Name string
	// Debug toggles verbose structured logging of the graph execution path
	// (each superstep, node entry, tool dispatch, model call), mirroring
	// Python's `create_agent(debug=True)`. Off by default.
	Debug bool
	// ResponseFormat configures structured output, mirroring Python's
	// `create_agent(response_format=...)`. Accepted values are ToolStrategy
	// (or *ToolStrategy), ProviderStrategy (or *ProviderStrategy), AutoStrategy
	// (or *AutoStrategy), and a raw JSON-schema map[string]any / schema.Schema
	// (Python's `response_format: dict` overload, resolved as AutoStrategy);
	// see WithAgentResponseFormat and the package doc comment for scope.
	ResponseFormat any
	// StateFields registers custom graph-state fields, mirroring Python's
	// `create_agent(state_schema=...)`. See WithAgentStateFields and
	// state_schema.go. A field whose Name collides with a default AgentState
	// key overrides that key's reducer.
	StateFields []StateField
	// ContextSchema declares the agent's runtime-context schema, mirroring
	// Python's `create_agent(context_schema=...)`. See WithAgentContextSchema
	// and context_schema.go. Purely declarative at present: it documents the
	// expected fields and reserves room for future validation; it does not
	// gate WithContextValues/ContextValue.
	ContextSchema []ContextField
	// Store is the agent's cross-thread semantic store (Python's
	// `create_agent(store=BaseStore)` / langgraph BaseStore). When non-nil, it
	// is installed on the compiled graph (surfaced on Runtime.Store) and
	// injected into every ToolCallRequest.Store (see middleware.ToolCallRequest).
	// It is distinct from core/stores.BaseStore[V] (the generic typed KV).
	Store store.Store
	// Cache caches model responses, mirroring Python's `create_agent(cache=...)`
	// parameter (but with a deliberate behavioral divergence — see
	// WithAgentCache). When non-nil, the model node consults it before the
	// model-call middleware chain: a hit short-circuits the chain, so
	// WrapModelCall middleware is NOT invoked for a cached call. This is an
	// intentional divergence from Python, where BaseCache sits inside the
	// chat-model layer and wrap_model_call middleware DOES observe
	// cache-served calls; skipping the whole chain here means
	// summarization/PII/retry middleware does not re-run on a cached answer,
	// which is the intended behavior. Misses are written back. Only terminal
	// text responses are cached; tool-call responses are skipped (a cached
	// tool call would be rebuilt lossily as text). The cache is scoped to the
	// non-streaming Invoke path; StreamEvents bypasses it entirely so
	// model_delta/model_end events always fire. Keyed by (promptString,
	// llmString); see cacheKey in create_agent.go.
	Cache caches.Cache
	// ModelString, when non-empty, is a "provider:model" string (e.g.
	// "openai:gpt-4o") resolved at CreateAgent time via
	// chatmodels.ParseModelString + chatmodels.Resolve into the ChatModel the
	// agent uses. See WithAgentModel for the full precedence rule.
	ModelString string
	// ToolSpecs are provider-native dict tool specs (Python's
	// `tools: Sequence[... | dict]` form), converted into core/tools tools and
	// appended after the positional tool list at CreateAgent time. See
	// WithAgentToolSpecs for the accepted dict shape and semantics.
	ToolSpecs []map[string]any
	// DynamicModel, when non-nil, resolves the agent's ChatModel per model
	// call from the current state, mirroring langgraph.prebuilt
	// create_react_agent's `model: Callable[[AgentState, Runtime],
	// LanguageModel]` overload (chat_agent_executor.py). Note that
	// langchain.agents.create_agent itself accepts only a static model — this
	// option is a superset kept at the agents layer so the prebuilt entry can
	// expose it without a parallel implementation. The resolver runs inside
	// the model node, after BeforeModel hooks (so hooks can rewrite the state
	// the resolver sees) and before middleware WrapModelCall composition. A
	// nil return falls back to the static model; a nil resolver always uses
	// the static model. An AutoStrategy ResponseFormat is re-resolved against
	// the model of each call (see buildModelNode /
	// resolveEffectiveResponseFormat), so swapping models via the resolver
	// re-checks the strategy; the build-time graph wiring (structured-output
	// bindings, routing) still comes from the static model's eager
	// resolution.
	DynamicModel func(state map[string]any, rt runtime.Runtime) language.ChatModel
}

// WithAgentDynamicModel installs a per-call model resolver, mirroring
// langgraph.prebuilt create_react_agent's callable-model overload (see
// AgentOptions.DynamicModel). It takes precedence over the positional model /
// WithAgentModel for each model call where it returns a non-nil ChatModel.
// The positional model arg may be nil when only a resolver is supplied
// (CreateAgent(nil, tools, WithAgentDynamicModel(...)) is valid); the run then
// fails only if the resolver itself returns nil.
func WithAgentDynamicModel(resolver func(state map[string]any, rt runtime.Runtime) language.ChatModel) AgentOption {
	return func(o *AgentOptions) { o.DynamicModel = resolver }
}

// AgentOption applies a functional option to AgentOptions.
type AgentOption func(*AgentOptions)

// WithAgentSystemPrompt sets the agent's system prompt, mirroring Python's
// `create_agent(system_prompt="...")` literal-string form. Backward
// compatible: this remains the common case. To pass a prompt with template
// variables (Python's `system_prompt: SystemMessage` form) use
// WithAgentSystemPromptTemplate instead.
func WithAgentSystemPrompt(prompt string) AgentOption {
	return func(o *AgentOptions) { o.SystemPrompt = prompt }
}

// WithAgentSystemPromptTemplate sets a templated system prompt, mirroring
// Python's `create_agent(system_prompt=SystemMessage(...))` form. The template
// is rendered via core/prompts (Go text/template syntax, e.g.
// `"You are {{.role}}."`) on every model call, so partial variables can change
// between runs. variables are the build-time template variables; per-Invoke
// variables can additionally be supplied via Agent.InvokeWithStateAndVars.
//
// Design choice (per Step 3b of the completeness plan): rather than overloading
// WithAgentSystemPrompt to also accept an interface{}, a dedicated option keeps
// the existing string path's signature stable and lets callers pass an explicit
// *prompts.PromptTemplate constructed via prompts.NewPromptTemplate, reusing
// core/prompts verbatim with no new abstraction. When both SystemPrompt and
// SystemPromptTemplate are set, the template wins.
//
// Passing a nil template clears any previously configured template.
func WithAgentSystemPromptTemplate(template *prompts.PromptTemplate, variables map[string]any) AgentOption {
	return func(o *AgentOptions) {
		o.SystemPromptTemplate = template
		if variables != nil {
			o.SystemPromptVariables = cloneStringAnyMap(variables)
		} else {
			o.SystemPromptVariables = nil
		}
	}
}

// WithAgentName sets the agent's run name / tracing tag, mirroring Python's
// `create_agent(name=...)` (the `lc_agent_name` equivalent). The name is stored
// on the Agent (see Agent.Name) and surfaced as a run-name tag through the
// context on each Invoke, since langgraph/graph exposes no native
// run-metadata injection point.
func WithAgentName(name string) AgentOption {
	return func(o *AgentOptions) { o.Name = name }
}

// WithAgentDebug toggles verbose structured logging of the graph execution
// path (each superstep, node entry, tool dispatch, model call), mirroring
// Python's `create_agent(debug=True)`. Off by default. Uses log/slog (no new
// dependency is introduced).
func WithAgentDebug(enabled bool) AgentOption {
	return func(o *AgentOptions) { o.Debug = enabled }
}

// WithAgentMiddleware appends middleware to the agent's middleware chain, in
// the order middleware runs for BeforeModel/outermost-WrapModelCall,
// mirroring Python's `create_agent(middleware=[...])`.
func WithAgentMiddleware(mw ...any) AgentOption {
	return func(o *AgentOptions) { o.Middleware = append(o.Middleware, mw...) }
}

// WithAgentCheckpointer installs a checkpoint.Saver, enabling interrupt/
// resume support on the compiled graph.
func WithAgentCheckpointer(saver checkpoint.Saver) AgentOption {
	return func(o *AgentOptions) { o.Checkpointer = saver }
}

// WithAgentStore installs a cross-thread semantic store (the langgraph
// BaseStore), mirroring Python's `create_agent(store=BaseStore)`. The store is
// surfaced on Runtime.Store for every node and injected into each tool call
// via middleware.ToolCallRequest.Store (Go has no Python-style InjectedStore
// annotation; tools read it explicitly).
//
// BREAKING (M1.2): the parameter type changed from core/stores.BaseStore[any]
// (the generic typed KV) to store.Store (the langgraph semantic store), so
// call sites must now pass a store.Store such as store.NewInMemoryStore().
// This realigns the Go port with Python's create_agent(store=BaseStore). The
// port is a v0.3.x preview whose README declares the API may change.
func WithAgentStore(s store.Store) AgentOption {
	return func(o *AgentOptions) { o.Store = s }
}

// WithAgentCache installs a model-response cache, mirroring Python's
// `create_agent(cache=...)` parameter (but with a deliberate behavioral
// divergence — see below). When non-nil, the model node looks up the cache
// before entering the model-call middleware chain: a hit returns the cached
// response without invoking the model OR any WrapModelCall middleware, and a
// miss invokes the model as usual then writes the result back.
//
// Divergence from Python: in Python, BaseCache sits inside the chat-model
// layer, so wrap_model_call middleware DOES observe cache-served calls. The
// Go port intentionally short-circuits the whole model call (including
// WrapModelCall middleware) on a cache hit, so summarization/PII/retry
// middleware does not re-run on a cached answer. This is a beneficial
// divergence, not an attempt to match Python's semantics.
//
// Scope: only terminal text responses are cached — tool-call responses are
// skipped, since messages.Text would drop ToolCalls/ToolCallID and a cached
// tool call would be rebuilt lossily as text on lookup. The cache is also
// scoped to the non-streaming Invoke path; StreamEvents bypasses it entirely
// so model_delta/model_end events always fire (a cache hit would otherwise
// short-circuit the handler and emit no events). The cache key is
// (promptString, llmString) as derived by cacheKey (the rendered, role-tagged
// message text, and the model type plus a stable hash of the request's tools,
// ModelSettings — which the model node seeds with the per-call effective
// strategy's kwargs — and ToolChoice), mirroring Python's llm_string, which
// serializes the bound model INCLUDING its bind kwargs (langchain_core
// caches.py:49-84) so different response formats / tool choices never
// collide. See cacheKey for the runtime-override caveat this implies.
func WithAgentCache(cache caches.Cache) AgentOption {
	return func(o *AgentOptions) { o.Cache = cache }
}

// WithAgentRecursionLimit overrides the compiled graph's superstep limit.
func WithAgentRecursionLimit(limit int) AgentOption {
	return func(o *AgentOptions) { o.RecursionLimit = limit }
}

// WithAgentInterruptBefore registers graph nodes the compiled agent pauses
// before running, mirroring Python's `create_agent(interrupt_before=[...])`.
// nodes are typically ModelNodeName ("model"), ToolsNodeName ("tools"),
// BeforeAgentNodeName, or AfterAgentNodeName. Requires a checkpointer
// (WithAgentCheckpointer) for the pauses to be resumable; resume via
// Agent.Graph.InvokeWithOptions with the same ThreadID and a nil Resume.
func WithAgentInterruptBefore(nodes ...string) AgentOption {
	return func(o *AgentOptions) { o.InterruptBefore = append(o.InterruptBefore, nodes...) }
}

// WithAgentInterruptAfter registers graph nodes the compiled agent pauses
// after running (and after their state update is merged), mirroring Python's
// `create_agent(interrupt_after=[...])`. See WithAgentInterruptBefore for the
// node names and resume semantics.
func WithAgentInterruptAfter(nodes ...string) AgentOption {
	return func(o *AgentOptions) { o.InterruptAfter = append(o.InterruptAfter, nodes...) }
}

// WithAgentResponseFormat configures structured output, mirroring Python's
// `create_agent(response_format=...)`. format must be a ToolStrategy,
// *ToolStrategy, ProviderStrategy, *ProviderStrategy, AutoStrategy,
// *AutoStrategy, or a raw JSON schema (a plain map[string]any or a
// schema.Schema — Python's `response_format: dict` overload); CreateAgent
// returns an error for any other type.
//
// A raw schema is interpreted exactly as Python treats a raw dict: it is
// wrapped in an AutoStrategy and resolved eagerly at CreateAgent time against
// the agent's bound model into a ToolStrategy or ProviderStrategy (ToolStrategy
// when the model declares ToolCalling, else ProviderStrategy when it declares
// StructuredOutput), so the schema is auto-detected rather than pinned to one
// strategy. This is how `type[ResponseT]` (Pydantic model class) is covered in
// the Go port: Go has no runtime type→JSON-schema reflection comparable to
// Pydantic, so callers pass the equivalent JSON schema directly.
//
// ToolStrategy is fully wired into the model loop: each of its SchemaSpecs is
// bound to the model as an extra callable tool, and a matching tool call in
// the model's response is intercepted (never reaching the tools node),
// parsed, and surfaces via the final state's "structured_response" key (see
// Agent.InvokeWithState), ending the run. Multiple structured tool calls in
// one response raise MultipleStructuredOutputsError — unless the strategy's
// HandleErrors policy elects retry, in which case error ToolMessages are
// injected and the loop returns to the model (mirroring Python's
// _handle_structured_output_error); see handleStructuredOutputError for the
// accepted HandleErrors forms.
//
// ProviderStrategy requests provider-native structured output: its kwargs
// (to_model_kwargs, e.g. OpenAI's response_format json_schema dict) reach the
// model on EVERY call — streaming and non-streaming — through, in order of
// capability: the ModelSettingsBinder interface (models that accept per-call
// bind kwargs), or language.StructuredCaller's InvokeStructured on the
// non-streaming path. Models with neither capability degrade to a best-effort
// post-hoc JSON-decode of the model's final text response (the structured
// response plumbing is identical in all three cases). See ModelSettingsBinder
// for why shipped partner models currently take the StructuredCaller /
// post-hoc paths.
//
// Middleware may replace the per-call strategy via
// ModelRequest.Override(WithResponseFormat(...)) — including switching
// between ToolStrategy and ProviderStrategy (factory.py:1323-1344); a
// ToolStrategy override may only narrow to structured tools declared in the
// original response format (factory.py:1375-1385).
//
// AutoStrategy is resolved eagerly at CreateAgent time against the agent's
// bound model into a concrete ToolStrategy or ProviderStrategy via its
// Resolve method (ToolStrategy when Capabilities().ToolCalling, else
// ProviderStrategy when Capabilities().StructuredOutput, else a
// *StructuredOutputUnsupportedError) — this fixes the graph wiring
// (structured-output bindings, routing). On every model call the effective
// strategy is then re-derived (resolveEffectiveResponseFormat): a
// ProviderStrategy is chosen when the CURRENT model supports it
// (SupportsProviderStrategy over the model's profile/name when exposed,
// else its Capabilities().StructuredOutput flag), so a DynamicModel swap
// onto a structured-output model switches to the provider path
// mid-conversation, and middleware overrides apply per call.
func WithAgentResponseFormat(format any) AgentOption {
	return func(o *AgentOptions) { o.ResponseFormat = format }
}

// WithAgentModel configures the agent's ChatModel from a "provider:model"
// string (e.g. "openai:gpt-4o"), mirroring Python's
// `create_agent(model="openai:gpt-4o")` bare-string overload. When set,
// CreateAgent resolves the string via chatmodels.ParseModelString (strict
// "provider:model" parser) + chatmodels.Resolve (registered ProviderFactory
// lookup) into a language.ChatModel and uses it as the agent's model.
//
// Precedence: when both the positional `model` arg and WithAgentModel are
// supplied, the ModelString wins (the positional arg is dropped). This
// matches Python's `create_agent(model=...)` taking a single model value and
// keeps the positional path (which remains the common case for callers with
// an already-constructed ChatModel) backward compatible: WithAgentModel is
// only set when the caller explicitly opts into the bare-string form.
//
// The positional `model` arg may be nil when WithAgentModel is set, so
// `CreateAgent(nil, nil, WithAgentModel("openai:gpt-4o"))` works end-to-end.
// When neither is supplied, CreateAgent returns its existing "model is
// required" error. A malformed string or unknown provider surfaces the
// underlying chatmodels parse/resolve error verbatim.
//
// Provider availability: a Go ProviderFactory must be registered for the
// provider half via chatmodels.RegisterProvider. The blank-import of
// partners/openai in this package makes "openai:..." resolve out-of-the-box;
// other providers require the caller to blank-import the relevant partner
// package.
func WithAgentModel(spec string) AgentOption {
	return func(o *AgentOptions) { o.ModelString = spec }
}

// WithAgentToolSpecs appends provider-native dict tool specs to the agent's
// tool list, mirroring Python's `create_agent(tools=[..., {...}])` dict form
// (`tools: Sequence[BaseTool | Callable | dict]`). Each spec is a
// map[string]any carrying a non-empty string "name", an optional string
// "description", and an optional JSON-schema object under "parameters" (also
// accepted under "input_schema" or "args_schema", the names used by other
// tool representations); when no schema key is present the args schema
// defaults to an empty object schema. Each spec is converted into a
// core/tools.Func via tools.NewFunc at CreateAgent time and appended after the
// positional tool list, so the dict tools are bound to the model alongside
// regular tools.
//
// Semantics note: a dict spec declares a provider-native tool (a schema + name
// + description), not a Go callable. Its converted Func therefore has no
// executable implementation — it is bound to the model so the model can emit
// tool calls for it, and any such call that reaches the tools node produces an
// explanatory tool error rather than running a real function. This is the Go
// equivalent of Python keeping dict (built-in) tools out of the ToolNode and
// handing them to the provider's bind_tools for server-side execution.
func WithAgentToolSpecs(specs ...map[string]any) AgentOption {
	return func(o *AgentOptions) { o.ToolSpecs = append(o.ToolSpecs, specs...) }
}

// Agent wraps a compiled model<->tools graph, mirroring Python's
// `CompiledStateGraph` returned by `create_agent(...)`.
type Agent struct {
	Graph *graphpkg.CompiledGraph
	// Name is the agent's run name / tracing tag (see WithAgentName). It is the
	// Go equivalent of Python's `lc_agent_name`. Exposed so callers and
	// observability tooling can read it without re-deriving it from options.
	Name string
	// debug toggles verbose graph-execution logging (see WithAgentDebug).
	debug bool
	// systemPromptTemplate/systemPromptVariables back the templated
	// system-prompt path (see WithAgentSystemPromptTemplate).
	systemPromptTemplate  *prompts.PromptTemplate
	systemPromptVariables map[string]any
}

// runNameCtxKey carries the agent's Name through a run as a run-name /
// tracing tag, mirroring Python's `lc_agent_name`. langgraph/graph has no
// native run-metadata injection point, so this is the lowest-friction place to
// surface the name to middleware/nodes that want to read it.
type runNameCtxKey struct{}

// promptVarsCtxKey carries per-Invoke template variables for the system prompt
// (see Agent.InvokeWithStateAndVars), merged over the build-time
// SystemPromptVariables.
type promptVarsCtxKey struct{}

// NameFromContext returns the run name carried in ctx, if any (set by
// Agent.Invoke/InvokeWithState/InvokeWithStateAndVars from the Agent's Name).
// Middleware or nodes that want to tag traces/logs with the agent name can read
// it here, mirroring how Python's middleware reads `lc_agent_name` off the run.
func NameFromContext(ctx context.Context) (string, bool) {
	name, ok := ctx.Value(runNameCtxKey{}).(string)
	return name, ok && name != ""
}

// PromptVarsFromContext returns per-Invoke system-prompt template variables
// carried in ctx (set by Agent.InvokeWithStateAndVars), merged over the
// build-time SystemPromptVariables. The returned bool reports whether any
// per-Invoke variables were supplied.
func PromptVarsFromContext(ctx context.Context) (map[string]any, bool) {
	vars, ok := ctx.Value(promptVarsCtxKey{}).(map[string]any)
	return vars, ok
}

// CreateAgent builds a create_agent-equivalent Agent around model and
// toolList. See the package doc comment for scope.
func CreateAgent(model language.ChatModel, toolList []coretools.Tool, opts ...AgentOption) (*Agent, error) {
	// Apply options BEFORE the nil-model check so WithAgentModel can supply
	// the model: when ModelString is set, it resolves a ChatModel via the
	// chatmodels registry (ParseModelString + Resolve) and overrides the
	// positional `model` arg (see WithAgentModel's doc comment). When neither
	// ModelString nor a positional model is supplied, the existing "model is
	// required" error fires.
	options := AgentOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	if options.ModelString != "" {
		spec, err := chatmodels.ParseModelString(options.ModelString)
		if err != nil {
			return nil, err
		}
		resolved, err := chatmodels.Resolve(spec)
		if err != nil {
			return nil, err
		}
		model = resolved
	}

	// A nil model is allowed only when a DynamicModel resolver (or a
	// ModelString, already handled above) supplies the model; the model node
	// fails the run if the resolver itself yields nil at call time.
	if model == nil && options.DynamicModel == nil {
		return nil, fmt.Errorf("agents: model is required")
	}

	if err := validateDeclaredJumpTargets(options.Middleware); err != nil {
		return nil, err
	}

	// Duplicate middleware validation (factory.py:1080-1082): two middleware
	// sharing a name — an explicit Name() or the same Go type — are rejected,
	// matching Python's default class-name identity.
	if err := validateMiddlewareNames(options.Middleware); err != nil {
		return nil, err
	}

	// Convert any dict tool specs (Python's `tools: [... | dict]` form) into
	// core/tools tools and append them after the positional tool list, so they
	// are bound to the model and routed through the tools node exactly like
	// positional tools. See WithAgentToolSpecs for the accepted dict shape.
	if len(options.ToolSpecs) > 0 {
		specTools, err := toolsFromToolSpecs(options.ToolSpecs)
		if err != nil {
			return nil, err
		}
		toolList = append(append([]coretools.Tool(nil), toolList...), specTools...)
	}

	// Middleware tool auto-collection (factory.py:1005: `middleware_tools =
	// [t for m in middleware for t in getattr(m, "tools", [])]`; merged ahead
	// of the caller's tools at factory.py:1054-1055). Middleware implementing
	// middleware.ToolProvider contribute their tools to both the ToolNode and
	// the model's default bound tools, with no manual append at the call site.
	toolList = mergeMiddlewareTools(options.Middleware, toolList)

	toolStrategy, providerStrategy, err := resolveResponseFormat(options.ResponseFormat, model)
	if err != nil {
		return nil, err
	}

	modelTools := toolList
	var structuredBindings map[string]OutputToolBinding
	if toolStrategy != nil {
		bindings, extraTools, err := buildStructuredOutputTools(toolStrategy)
		if err != nil {
			return nil, err
		}
		structuredBindings = bindings
		modelTools = append(append([]coretools.Tool(nil), toolList...), extraTools...)
	}

	// finalNode is where every "run is over" routing decision (normal
	// completion, a jump_to "end", or a matched structured-output response)
	// goes: types.END directly, or through a dedicated "after_agent" node
	// first when at least one AfterAgentHook/AfterAgentUpdateHook is
	// configured (see the package doc comment).
	finalNode := types.END
	hasAfterAgent := hasHook[AfterAgentHook](options.Middleware) || hasHook[AfterAgentUpdateHook](options.Middleware)
	if hasAfterAgent {
		finalNode = AfterAgentNodeName
	}

	// Interrupt-mode HITL middleware get a dedicated "hitl" graph node (see
	// AfterModelNodeHook). The count check runs unconditionally: a second
	// interrupt-mode HITL is rejected at build time even before tool wiring
	// decides whether the node is actually added (without tools there is
	// nothing reviewable, and the middleware stays inert).
	if err := validateSingleInterruptHITL(options.Middleware); err != nil {
		return nil, err
	}
	hitlNodePresent := len(options.Middleware) > 0 && hasInterruptHITL(options.Middleware) && len(toolList) > 0

	logger := debugLogger(options.Debug)

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	// Register middleware-contributed state fields (factory.py:1150-1156:
	// `state_schemas = [*(m.state_schema for m in middleware), base_state]`,
	// merged in order with later declarations winning field conflicts).
	// Middleware schemas run FIRST so the caller's explicit StateFields below
	// override any middleware contribution on a name conflict — the Go
	// AddReducer map is last-call-wins, which reproduces Python's
	// base_state-wins merge. A nil reducer defaults to
	// channels.LastValueReducer (replace semantics).
	for _, mw := range options.Middleware {
		contributor, ok := mw.(middleware.StateSchemaContributor)
		if !ok {
			continue
		}
		for _, f := range contributor.StateSchema() {
			r := f.Reducer
			if r == nil {
				r = channels.LastValueReducer
			}
			g.AddReducer(f.Name, r)
		}
	}
	// Register user-supplied state fields (Python state_schema). A nil reducer
	// defaults to channels.LastValueReducer (replace semantics), which is also
	// the implicit reducer for any unregistered key, so a nil-reducer field is
	// equivalent to omitting it. A field whose Name collides with a default
	// key ("messages"/"jump_to"/"structured_response") overrides that key's
	// reducer (see WithAgentStateFields / state_schema.go).
	for _, f := range options.StateFields {
		r := f.Reducer
		if r == nil {
			r = channels.LastValueReducer
		}
		g.AddReducer(f.Name, r)
	}
	// resolveModel picks the model per call: the DynamicModel resolver when
	// configured (nil return falls through), else the static model (positional
	// arg or ModelString). See AgentOptions.DynamicModel.
	resolveModel := func(rt runtime.Runtime, state map[string]any) language.ChatModel {
		if options.DynamicModel != nil {
			if m := options.DynamicModel(state, rt); m != nil {
				return m
			}
		}
		return model
	}
	g.AddNode(ModelNodeName, buildModelNode(resolveModel, modelTools, systemPromptResolver(options), logger, options.Middleware, structuredBindings, toolStrategy, providerStrategy, normalizeInitialResponseFormat(options.ResponseFormat), finalNode, options.Cache, options.Name, hitlNodePresent))

	entryNode := ModelNodeName
	if hasHook[BeforeAgentHook](options.Middleware) {
		entryNode = BeforeAgentNodeName
		g.AddNode(BeforeAgentNodeName, buildBeforeAgentNode(options.Middleware, logger, finalNode))
		g.AddEdge(BeforeAgentNodeName, ModelNodeName)
	}
	g.AddEdge(types.START, entryNode)

	if hasAfterAgent {
		g.AddNode(AfterAgentNodeName, buildAfterAgentNode(options.Middleware, logger))
		g.AddEdge(AfterAgentNodeName, types.END)
	}

	if len(toolList) > 0 {
		toolNode, err := newToolsNode(toolList, options.Middleware, logger, options.Store)
		if err != nil {
			return nil, err
		}
		g.AddNode(ToolsNodeName, toolNode)
		// tools -> (model | exit): exit when every executed client-side tool
		// call targeted a return-direct tool, mirroring Python's
		// `_make_tools_to_model_edge` (factory.py:1623-1649, 1921-1947).
		g.AddConditionalEdges(ToolsNodeName, buildRouteAfterTools(toolsByNameFromList(toolList), finalNode))
		if hitlNodePresent {
			// model -> hitl -> (tools | model | end): with an interrupt-mode
			// HITL middleware, the model's "tools" destinations are remapped
			// through the dedicated hitl node (mirroring Python's per-
			// middleware after_model nodes, factory.py:1569/:1738-1748), so the
			// review pauses AFTER the model node's update commits and a resume
			// re-runs only the hitl node. The hitl node itself carries the
			// ORIGINAL routing, judging the post-decision state: revised calls
			// continue to tools, fully-answered ones (reject/respond) loop back
			// to the model.
			hitlNode, err := buildHITLNode(options.Middleware, logger, finalNode)
			if err != nil {
				return nil, err
			}
			g.AddNode(HITLNodeName, hitlNode)
			g.AddConditionalEdges(ModelNodeName, routeAfterModelWithHITL(buildRouteAfterModel(structuredBindings, finalNode), HITLNodeName))
			g.AddConditionalEdges(HITLNodeName, buildRouteAfterModel(structuredBindings, finalNode))
		} else {
			g.AddConditionalEdges(ModelNodeName, buildRouteAfterModel(structuredBindings, finalNode))
		}
	} else if len(structuredBindings) > 0 {
		// No client-side tools but structured-output tools are bound: loop the
		// model back to itself until a structured response lands, mirroring
		// Python's `_make_model_to_model_edge` branch (factory.py:1666-1677).
		g.AddConditionalEdges(ModelNodeName, buildRouteStructuredOnly(finalNode))
	} else {
		g.AddEdge(ModelNodeName, finalNode)
	}

	compileOpts := make([]graphpkg.CompileOption, 0, 4)
	if options.Checkpointer != nil {
		compileOpts = append(compileOpts, graphpkg.WithCheckpointer(options.Checkpointer))
	}
	// Python's create_agent compiles with recursion_limit=9999 (factory.py:1780,
	// raised in langchain#7313) so long tool loops are gated by the caller
	// rather than langgraph's much lower platform default. Mirror that here:
	// an unset RecursionLimit becomes 9999 instead of the graph package's
	// defaultRecursionLimit. Per-invoke overrides via
	// CompiledGraph.InvokeWithOptions still take precedence.
	recursionLimit := options.RecursionLimit
	if recursionLimit <= 0 {
		recursionLimit = 9999
	}
	compileOpts = append(compileOpts, graphpkg.WithRecursionLimit(recursionLimit))
	if len(options.InterruptBefore) > 0 {
		compileOpts = append(compileOpts, graphpkg.WithInterruptBefore(options.InterruptBefore...))
	}
	if len(options.InterruptAfter) > 0 {
		compileOpts = append(compileOpts, graphpkg.WithInterruptAfter(options.InterruptAfter...))
	}

	compiled, err := g.Compile(compileOpts...)
	if err != nil {
		return nil, err
	}
	return &Agent{
		Graph:                 compiled,
		Name:                  options.Name,
		debug:                 options.Debug,
		systemPromptTemplate:  options.SystemPromptTemplate,
		systemPromptVariables: options.SystemPromptVariables,
	}, nil
}

// systemPromptResolver returns a closure the model node calls on every model
// invocation to resolve the current system-prompt string. When a
// SystemPromptTemplate is configured it renders the template, merging build-time
// SystemPromptVariables with any per-Invoke variables carried in the context
// (see Agent.InvokeWithStateAndVars / PromptVarsFromContext); otherwise it
// returns the literal SystemPrompt. Mirrors Python's `system_prompt: str |
// SystemMessage` overload from a single code path.
func systemPromptResolver(options AgentOptions) func(ctx context.Context) string {
	template := options.SystemPromptTemplate
	buildVars := options.SystemPromptVariables
	literal := options.SystemPrompt
	if template == nil {
		return func(_ context.Context) string { return literal }
	}
	return func(ctx context.Context) string {
		vars := cloneStringAnyMap(buildVars)
		if perInvoke, ok := PromptVarsFromContext(ctx); ok {
			for k, v := range perInvoke {
				vars[k] = v
			}
		}
		rendered, err := template.Format(vars)
		if err != nil {
			// Fall back to the literal so a template-render failure can never
			// silently turn a configured prompt into empty; the underlying
			// core/prompts error is logged for diagnosis.
			slog.Warn("agents: system prompt template render failed; using literal fallback",
				slog.String("error", err.Error()))
			return literal
		}
		return rendered
	}
}

// debugLogger returns a *slog.Logger used for verbose graph-execution logging
// when WithAgentDebug(true) is set; nil otherwise (callers must nil-check).
func debugLogger(enabled bool) *slog.Logger {
	if !enabled {
		return nil
	}
	return slog.Default()
}

// cloneStringAnyMap returns a shallow copy of m (nil-safe).
func cloneStringAnyMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// Invoke runs the agent over msgs and returns the full resulting message
// history, mirroring a simplified `agent.invoke({"messages": [...]})`. Use
// InvokeWithState instead to also access a "structured_response" produced by
// a configured ResponseFormat.
func (a *Agent) Invoke(ctx context.Context, msgs []messages.Message) ([]messages.Message, error) {
	state, err := a.InvokeWithState(ctx, msgs)
	if err != nil {
		return nil, err
	}
	out, _ := state["messages"].([]messages.Message)
	return out, nil
}

// InvokeWithState runs the agent over msgs and returns the full final state
// map, mirroring a simplified `agent.invoke({"messages": [...]})` before
// Python narrows the result to its typed AgentState. In addition to
// "messages", this includes a "structured_response" key (parsed structured
// output data) whenever WithAgentResponseFormat produced one, plus any other
// state keys middleware wrote (e.g. tool/model call counters).
//
// The Agent's Name (see WithAgentName) is surfaced through the returned
// context-free run via NameFromContext for any middleware/node that wants to
// tag traces with it.
func (a *Agent) InvokeWithState(ctx context.Context, msgs []messages.Message) (map[string]any, error) {
	return a.InvokeWithStateAndVars(ctx, msgs, nil)
}

// InvokeWithStateAndVars is like InvokeWithState but additionally supplies
// per-Invoke template variables for a configured SystemPromptTemplate (see
// WithAgentSystemPromptTemplate). variables are merged over the build-time
// SystemPromptVariables for this run only. Passing nil variables is equivalent
// to InvokeWithState.
func (a *Agent) InvokeWithStateAndVars(ctx context.Context, msgs []messages.Message, variables map[string]any) (map[string]any, error) {
	runCtx := a.withRunTags(ctx)
	// Mark this run non-streaming: InvokeWithState must never emit streaming
	// events, even when ctx inherited an event sink from a streaming ancestor
	// (e.g. a nested invoke called from a tool inside a StreamEvents parent).
	// sinkFromContext honors suppressStreamSinkCtxKey and returns nil, so the
	// model/tools nodes take the non-streaming path and the cache gate stays on.
	runCtx = context.WithValue(runCtx, suppressStreamSinkCtxKey{}, true)
	if variables != nil {
		runCtx = context.WithValue(runCtx, promptVarsCtxKey{}, cloneStringAnyMap(variables))
	}
	if a.debug {
		slog.Info("agents: invoke start",
			slog.String("agent_name", a.Name),
			slog.Int("input_messages", len(msgs)))
	}
	result, err := a.Graph.Invoke(runCtx, map[string]any{"messages": msgs})
	if err != nil {
		return nil, err
	}
	if len(result.Interrupts) > 0 {
		return nil, fmt.Errorf("agents: run interrupted (%d pending interrupt(s)); use Agent.Graph directly with a checkpointer to resume", len(result.Interrupts))
	}
	if a.debug {
		outMsgs, _ := result.Values["messages"].([]messages.Message)
		slog.Info("agents: invoke done",
			slog.String("agent_name", a.Name),
			slog.Int("output_messages", len(outMsgs)))
	}
	return result.Values, nil
}

// InvokeWithStateOptions runs the agent like InvokeWithState but with full
// graph Options (ThreadID/CheckpointID/Resume/Graph/RecursionLimit...), and
// surfaces pauses instead of erroring on them: a run interrupted by an
// in-node Interrupt (e.g. an interrupt-mode HumanInTheLoopMiddleware's hitl
// node) returns the paused state together with the pending interrupts and a
// nil error. Resume the run with Agent.Resume (or Agent.Graph.InvokeWithOptions
// directly for full control), passing the answer as Options.Resume — for a
// HITL pause, a middleware.HITLResponse (or its map wire form, see
// middleware.DecodeHITLResponse).
//
// Like InvokeWithState, this is a non-streaming entry point: it never emits
// streaming events even under a streaming ancestor. Use a ThreadID together
// with WithAgentCheckpointer for interrupts to be resumable at all.
func (a *Agent) InvokeWithStateOptions(ctx context.Context, msgs []messages.Message, opts graphpkg.Options) (map[string]any, []types.Interrupt, error) {
	runCtx := a.withRunTags(ctx)
	runCtx = context.WithValue(runCtx, suppressStreamSinkCtxKey{}, true)
	if a.debug {
		slog.Info("agents: invoke start",
			slog.String("agent_name", a.Name),
			slog.Int("input_messages", len(msgs)))
	}
	result, err := a.Graph.InvokeWithOptions(runCtx, map[string]any{"messages": msgs}, opts)
	if err != nil {
		return nil, nil, err
	}
	if len(result.Interrupts) > 0 {
		return result.Values, result.Interrupts, nil
	}
	if a.debug {
		outMsgs, _ := result.Values["messages"].([]messages.Message)
		slog.Info("agents: invoke done",
			slog.String("agent_name", a.Name),
			slog.Int("output_messages", len(outMsgs)))
	}
	return result.Values, nil, nil
}

// Resume continues a previously interrupted run from its thread's latest
// checkpoint: it is the resume half of InvokeWithStateOptions, invoking the
// graph with a nil input (fresh input would start a new turn instead) and
// the caller's Options. opts.Resume supplies the answer(s) to the pending
// interrupt(s): a scalar (e.g. a middleware.HITLResponse) feeds a single
// pending interrupt, a map addresses several by interrupt NS or ID, and a
// nil Resume re-pauses in-node interrupts (boundary interrupts —
// WithAgentInterruptBefore/After — resume with nil by design). A run that
// pauses again returns its state plus the new pending interrupts and a nil
// error. opts must carry the ThreadID the paused run used.
func (a *Agent) Resume(ctx context.Context, opts graphpkg.Options) (map[string]any, []types.Interrupt, error) {
	runCtx := a.withRunTags(ctx)
	runCtx = context.WithValue(runCtx, suppressStreamSinkCtxKey{}, true)
	result, err := a.Graph.InvokeWithOptions(runCtx, nil, opts)
	if err != nil {
		return nil, nil, err
	}
	if len(result.Interrupts) > 0 {
		return result.Values, result.Interrupts, nil
	}
	return result.Values, nil, nil
}

// withRunTags returns ctx annotated with this Agent's run-name tag, so
// middleware/nodes can read it via NameFromContext (the lc_agent_name
// equivalent). langgraph/graph has no native run-metadata injection point,
// so this context value is the surfaced channel.
func (a *Agent) withRunTags(ctx context.Context) context.Context {
	if a.Name == "" {
		return ctx
	}
	return context.WithValue(ctx, runNameCtxKey{}, a.Name)
}

// buildRouteAfterModel is the model node's exit routing, mirroring Python's
// `_make_model_to_tools_edge` (factory.py:1840-1897) step by step against the
// committed state after the node's update lands:
//
//  1. (Python step 1, jump_to, is handled upstream: the model node returns a
//     Command Goto for a jump, which bypasses this edge entirely.)
//  2. no last AIMessage (e.g. messages were cleared) -> end
//  3. the AI message has no tool calls -> end (the classic agent-loop exit)
//  4. unanswered, non-structured-output tool calls -> tools
//  5. a structured_response already in state -> end
//  6. the AI message has tool calls but all are answered by tool messages
//     (artificially injected ones, e.g. a structured-output retry's error
//     ToolMessages) -> back to model
func buildRouteAfterModel(structuredBindings map[string]OutputToolBinding, finalNode string) graphpkg.ConditionalEdge {
	return func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		lastAI, toolMsgs := fetchLastAIAndToolMessages(msgs)
		if lastAI == nil {
			return graphpkg.To(finalNode), nil
		}
		if len(lastAI.ToolCalls) == 0 {
			return graphpkg.To(finalNode), nil
		}
		answered := make(map[string]bool, len(toolMsgs))
		for _, m := range toolMsgs {
			answered[m.ToolCallID] = true
		}
		pending := false
		for _, call := range lastAI.ToolCalls {
			if answered[call.ID] {
				continue
			}
			if _, structured := structuredBindings[call.Name]; structured {
				continue
			}
			pending = true
			break
		}
		if pending {
			return graphpkg.To(ToolsNodeName), nil
		}
		if _, ok := state["structured_response"]; ok {
			return graphpkg.To(finalNode), nil
		}
		return graphpkg.To(ModelNodeName), nil
	}
}

// buildRouteAfterTools is the tools node's exit routing, mirroring Python's
// `_make_tools_to_model_edge` (factory.py:1921-1947): the loop ends when every
// executed client-side tool call targeted a return-direct tool (Python's
// BaseTool.return_direct, surfaced in Go via core/tools.ReturnDirecter);
// otherwise it continues to the model. Python's third condition — exit when a
// structured-output tool was executed — has no Go counterpart here because
// structured tool calls are intercepted in the model node and never reach the
// tools node.
func buildRouteAfterTools(toolsByName map[string]coretools.Tool, finalNode string) graphpkg.ConditionalEdge {
	return func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		lastAI, _ := fetchLastAIAndToolMessages(msgs)
		if lastAI == nil {
			return graphpkg.To(ModelNodeName), nil
		}
		clientCalls := 0
		allReturnDirect := true
		for _, call := range lastAI.ToolCalls {
			tool, ok := toolsByName[call.Name]
			if !ok {
				continue
			}
			clientCalls++
			if !coretools.IsReturnDirect(tool) {
				allReturnDirect = false
			}
		}
		if clientCalls > 0 && allReturnDirect {
			return graphpkg.To(finalNode), nil
		}
		return graphpkg.To(ModelNodeName), nil
	}
}

// buildRouteStructuredOnly mirrors Python's `_make_model_to_model_edge`
// (factory.py:1899-1916), wired when the agent has structured-output tools but
// no client-side tools: the model loops back to itself until a structured
// response lands in state (the match itself exits via its terminal Command,
// which bypasses this edge), mirroring Python's retry-until-structured loop.
func buildRouteStructuredOnly(finalNode string) graphpkg.ConditionalEdge {
	return func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		if _, ok := state["structured_response"]; ok {
			return graphpkg.To(finalNode), nil
		}
		return graphpkg.To(ModelNodeName), nil
	}
}

// fetchLastAIAndToolMessages returns the most recent AI message (searching
// from the end) together with any ToolMessages after it, mirroring Python's
// `_fetch_last_ai_and_tool_messages` (factory.py:1819-1837). A nil first
// return means no AIMessage exists in msgs.
func fetchLastAIAndToolMessages(msgs []messages.Message) (*messages.Message, []messages.Message) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != messages.RoleAI {
			continue
		}
		after := msgs[i+1:]
		toolMsgs := make([]messages.Message, 0, len(after))
		for _, m := range after {
			if m.Role == messages.RoleTool {
				toolMsgs = append(toolMsgs, m)
			}
		}
		ai := msgs[i]
		return &ai, toolMsgs
	}
	return nil, nil
}

// toolsByNameFromList indexes toolList by name for the tools node's
// return-direct routing (see buildRouteAfterTools).
func toolsByNameFromList(toolList []coretools.Tool) map[string]coretools.Tool {
	byName := make(map[string]coretools.Tool, len(toolList))
	for _, t := range toolList {
		byName[t.Name()] = t
	}
	return byName
}

func resolveJumpTarget(jumpTo string, finalNode string) string {
	switch jumpTo {
	case "end", "":
		return finalNode
	case "model":
		return ModelNodeName
	case "tools":
		return ToolsNodeName
	default:
		return jumpTo
	}
}

func popJumpTo(update map[string]any) (string, bool) {
	if update == nil {
		return "", false
	}
	value, ok := update["jump_to"]
	if !ok {
		return "", false
	}
	delete(update, "jump_to")
	jumpTo, _ := value.(string)
	return jumpTo, jumpTo != ""
}

// validateDeclaredJumpTargets enforces Python's JumpTo literal ("tools" |
// "model" | "end") on statically declared hook can_jump_to values
// (HookConfig/CanJumpToHook) at agent build time.
func validateDeclaredJumpTargets(mws []any) error {
	for _, mw := range mws {
		hook, ok := mw.(CanJumpToHook)
		if !ok {
			continue
		}
		for _, hookName := range []string{"before_model", "after_model"} {
			for _, target := range hook.CanJumpTo(hookName) {
				switch target {
				case "model", "tools", "end":
				default:
					return fmt.Errorf("agents: invalid can_jump_to target %q declared for %s (valid: model, tools, end)", target, hookName)
				}
			}
		}
	}
	return nil
}

func cloneMapState(state map[string]any) map[string]any {
	out := make(map[string]any, len(state))
	for k, v := range state {
		out[k] = v
	}
	return out
}

func toolsToAny(toolList []coretools.Tool) []any {
	out := make([]any, len(toolList))
	for i, t := range toolList {
		out[i] = t
	}
	return out
}

func toolsFromAny(list []any) ([]coretools.Tool, error) {
	out := make([]coretools.Tool, 0, len(list))
	for _, item := range list {
		t, ok := item.(coretools.Tool)
		if !ok {
			return nil, fmt.Errorf("agents: expected core/tools.Tool in ModelRequest.Tools, got %T", item)
		}
		out = append(out, t)
	}
	return out, nil
}

// toolsFromToolSpecs converts provider-native dict tool specs (Python's
// `tools: Sequence[... | dict]` form) into core/tools tools via toolFromSpec.
func toolsFromToolSpecs(specs []map[string]any) ([]coretools.Tool, error) {
	out := make([]coretools.Tool, 0, len(specs))
	for i, spec := range specs {
		tool, err := toolFromSpec(spec)
		if err != nil {
			return nil, fmt.Errorf("agents: tool spec %d: %w", i, err)
		}
		out = append(out, tool)
	}
	return out, nil
}

// toolFromSpec converts a single dict tool spec into a core/tools.Tool (a
// tools.Func) with its name, description, and args JSON schema taken from the
// dict. See WithAgentToolSpecs for the accepted dict shape.
func toolFromSpec(spec map[string]any) (coretools.Tool, error) {
	name, _ := spec["name"].(string)
	if name == "" {
		return nil, fmt.Errorf("tool spec requires a non-empty string \"name\"")
	}
	description, _ := spec["description"].(string)

	argsSchema, err := toolSpecArgsSchema(spec, name)
	if err != nil {
		return nil, err
	}

	return coretools.NewFunc(name, description, argsSchema,
		func(context.Context, map[string]any) (coretools.Result, error) {
			return coretools.Result{}, fmt.Errorf("agents: tool %q was declared as a dict spec and has no executable implementation", name)
		})
}

// toolSpecArgsSchema extracts the args JSON schema from a dict tool spec,
// checking the "parameters", "input_schema", and "args_schema" keys (in that
// order of precedence). A missing key defaults to an empty object schema,
// mirroring a provider tool with no parameters.
func toolSpecArgsSchema(spec map[string]any, name string) (schema.Schema, error) {
	for _, key := range []string{"parameters", "input_schema", "args_schema"} {
		raw, ok := spec[key]
		if !ok {
			continue
		}
		if s, ok := asSchema(raw); ok {
			return s, nil
		}
		return nil, fmt.Errorf("tool spec %q: %s must be a JSON-schema object, got %T", name, key, raw)
	}
	return schema.Object(map[string]schema.Schema{}), nil
}

// asSchema coerces a raw schema value into a schema.Schema. It accepts both a
// plain map[string]any and the schema.Schema spelling, since a JSON-decoded
// dict is the former while schema package constructors return the latter.
func asSchema(v any) (schema.Schema, bool) {
	switch s := v.(type) {
	case schema.Schema:
		return s, true
	case map[string]any:
		return schema.Schema(s), true
	default:
		return nil, false
	}
}

// hasHook reports whether any middleware in mws implements hookType (used as
// a type parameter, e.g. hasHook[BeforeAgentHook](mws)).
func hasHook[T any](mws []any) bool {
	for _, mw := range mws {
		if _, ok := mw.(T); ok {
			return true
		}
	}
	return false
}

// mergeMiddlewareTools returns the effective tool list for an agent: every
// middleware's ProvidedTools() (in middleware registration order) followed by
// the caller's tools, mirroring Python's
// `available_tools = middleware_tools + regular_tools` (factory.py:1054-1055)
// and the ToolNode's dict registration (langgraph prebuilt tool_node.py:784),
// where a later tool with the same name silently replaces an earlier one
// while keeping the first registration's position — so on a name conflict the
// caller's tool wins over the middleware's. When no middleware provides
// tools, userTools is returned unchanged (same slice, no copy).
func mergeMiddlewareTools(mws []any, userTools []coretools.Tool) []coretools.Tool {
	var middlewareTools []coretools.Tool
	for _, mw := range mws {
		if provider, ok := mw.(middleware.ToolProvider); ok {
			middlewareTools = append(middlewareTools, provider.ProvidedTools()...)
		}
	}
	if len(middlewareTools) == 0 {
		return userTools
	}
	out := make([]coretools.Tool, 0, len(middlewareTools)+len(userTools))
	indexByName := make(map[string]int, len(middlewareTools)+len(userTools))
	add := func(t coretools.Tool) {
		if t == nil {
			return
		}
		if i, ok := indexByName[t.Name()]; ok {
			out[i] = t // later registration replaces, first position kept
			return
		}
		indexByName[t.Name()] = len(out)
		out = append(out, t)
	}
	for _, t := range middlewareTools {
		add(t)
	}
	for _, t := range userTools {
		add(t)
	}
	return out
}

// buildBeforeAgentNode returns the "before_agent" graph node, running every
// BeforeAgentHook middleware once (in order) before the model<->tools loop
// starts. A hook update carrying "jump_to" ("model"/"tools"/"end"
// — factory.py:1694-1713: the before_agent conditional edge routes jump_to
// "model" to the loop entry and "end" to the exit node) makes the node return
// a *types.Command whose Goto bypasses the default before_agent→model edge;
// the key is consumed (never persisted). logger, when non-nil, emits a debug
// log at node entry.
func buildBeforeAgentNode(mws []any, logger *slog.Logger, finalNode string) graphpkg.NodeFunc {
	return func(rt runtime.Runtime, rawState map[string]any) (any, error) {
		if logger != nil {
			logger.Info("agents: before_agent node entry")
		}
		state := cloneMapState(rawState)
		update := map[string]any{}
		for _, mw := range mws {
			hook, ok := mw.(BeforeAgentHook)
			if !ok {
				continue
			}
			hookUpdate, err := hook.BeforeAgent(rt, state)
			if err != nil {
				return nil, err
			}
			for k, v := range hookUpdate {
				state[k] = v
				update[k] = v
			}
		}
		if jumpTo, ok := popJumpTo(update); ok {
			return &types.Command{Update: update, Goto: graphpkg.To(resolveJumpTarget(jumpTo, finalNode))}, nil
		}
		return update, nil
	}
}

// buildAfterAgentNode returns the "after_agent" graph node, running every
// AfterAgentHook/AfterAgentUpdateHook middleware once (in order) after the
// model<->tools loop ends. Update-returning hooks' keys commit to state; an
// update carrying "jump_to" = "model" re-enters the model<->tools loop
// (factory.py:1753-1776: the after_agent→END edge declares
// model_destination=loop entry), consumed like the model-node jump_to so it
// never persists. "end" (or no jump) completes the run through the node's
// END edge. logger, when non-nil, emits a debug log at node entry.
func buildAfterAgentNode(mws []any, logger *slog.Logger) graphpkg.NodeFunc {
	return func(rt runtime.Runtime, rawState map[string]any) (any, error) {
		if logger != nil {
			logger.Info("agents: after_agent node entry")
		}
		state := cloneMapState(rawState)
		update := map[string]any{}
		for _, mw := range mws {
			if hook, ok := mw.(AfterAgentUpdateHook); ok {
				hookUpdate, err := hook.AfterAgent(rt, state)
				if err != nil {
					return nil, err
				}
				for k, v := range hookUpdate {
					state[k] = v
					update[k] = v
				}
				continue
			}
			hook, ok := mw.(AfterAgentHook)
			if !ok {
				continue
			}
			if err := hook.AfterAgent(rt, state); err != nil {
				return nil, err
			}
		}
		if len(update) == 0 {
			return nil, nil
		}
		if jumpTo, ok := popJumpTo(update); ok {
			// "end" maps to END directly: finalNode here IS the after_agent
			// node itself, so routing to it would re-enter the node forever.
			return &types.Command{Update: update, Goto: graphpkg.To(resolveJumpTarget(jumpTo, types.END))}, nil
		}
		return update, nil
	}
}

// hasInterruptHITL reports whether any middleware in mws is an interrupt-mode
// HITL: it implements both AfterModelNodeHook and middleware.HitlInterrupter
// with HitlInterruptEnabled() true (a Decide-mode HumanInTheLoopMiddleware
// runs inline via AfterModel and is excluded).
func hasInterruptHITL(mws []any) bool {
	for _, mw := range mws {
		interrupter, ok := mw.(middleware.HitlInterrupter)
		if !ok || !interrupter.HitlInterruptEnabled() {
			continue
		}
		if _, ok := mw.(AfterModelNodeHook); ok {
			return true
		}
	}
	return false
}

// validateSingleInterruptHITL rejects a second interrupt-mode HITL
// middleware: both would share the single fixed-name hitl node, and their
// Interrupt calls would interleave into one resume queue, so the second one
// is a build-time error (mirroring the fixed-node-name constraint rather
// than a Python limit — Python derives one node per middleware instance).
func validateSingleInterruptHITL(mws []any) error {
	count := 0
	for _, mw := range mws {
		interrupter, ok := mw.(middleware.HitlInterrupter)
		if !ok || !interrupter.HitlInterruptEnabled() {
			continue
		}
		if _, ok := mw.(AfterModelNodeHook); ok {
			count++
		}
	}
	if count > 1 {
		return fmt.Errorf("agents: at most one interrupt-mode human-in-the-loop middleware is supported (the dedicated %q node is shared); found %d", HITLNodeName, count)
	}
	return nil
}

// buildHITLNode returns the dedicated "hitl" graph node running every
// interrupt-mode HITL middleware's AfterModelNodeHook (see the interface's
// doc comment for why the review lives in its own node). Multiple such
// middleware are rejected (see validateSingleInterruptHITL). Hook updates
// merge like buildAfterAgentNode's: "messages" concatenates (add_messages
// semantics — the revised AIMessage replaces the committed one by ID via
// MessagesReducer), other keys last-write-wins, and update["jump_to"] routes
// through resolveJumpTarget ("tools"/"end" bypass the node's default routing;
// note "tools" skips the HITL review by construction, mirroring Python's
// jump_to which addresses the tools node directly). logger, when non-nil,
// emits a debug log at node entry.
func buildHITLNode(mws []any, logger *slog.Logger, finalNode string) (graphpkg.NodeFunc, error) {
	if err := validateSingleInterruptHITL(mws); err != nil {
		return nil, err
	}
	return func(rt runtime.Runtime, rawState map[string]any) (any, error) {
		if logger != nil {
			logger.Info("agents: hitl node entry")
		}
		state := cloneMapState(rawState)
		update := map[string]any{}
		for _, mw := range mws {
			interrupter, ok := mw.(middleware.HitlInterrupter)
			if !ok || !interrupter.HitlInterruptEnabled() {
				continue
			}
			hook, ok := mw.(AfterModelNodeHook)
			if !ok {
				continue
			}
			// rt IS the node's context (langgraph Runtime implements
			// context.Context), so graphpkg.Interrupt inside the hook reaches
			// the task's interrupt state.
			hookUpdate, err := hook.AfterModelNode(rt, state)
			if err != nil {
				return nil, err
			}
			if hookUpdate == nil {
				continue
			}
			if extra, ok := hookUpdate["messages"].([]messages.Message); ok {
				delete(hookUpdate, "messages")
				base, _ := update["messages"].([]messages.Message)
				merged := make([]messages.Message, 0, len(base)+len(extra))
				merged = append(merged, base...)
				merged = append(merged, extra...)
				update["messages"] = merged
				if stateMsgs, ok := state["messages"].([]messages.Message); ok {
					state["messages"] = append(append([]messages.Message(nil), stateMsgs...), extra...)
				}
			}
			for k, v := range hookUpdate {
				state[k] = v
				update[k] = v
			}
		}
		if len(update) == 0 {
			return nil, nil
		}
		if jumpTo, ok := popJumpTo(update); ok {
			return &types.Command{Update: update, Goto: graphpkg.To(resolveJumpTarget(jumpTo, finalNode))}, nil
		}
		return update, nil
	}, nil
}

// routeAfterModelWithHITL wraps a model-exit ConditionalEdge so every
// "tools" destination is remapped to the hitl node: the review runs between
// the model node and the tools node (Python's after_model node ordering,
// factory.py:1738-1748). Everything else passes through unchanged.
func routeAfterModelWithHITL(inner graphpkg.ConditionalEdge, hitlNode string) graphpkg.ConditionalEdge {
	return func(rt runtime.Runtime, state map[string]any) ([]any, error) {
		dests, err := inner(rt, state)
		if err != nil {
			return nil, err
		}
		out := make([]any, len(dests))
		for i, d := range dests {
			if name, ok := d.(string); ok && name == ToolsNodeName {
				out[i] = hitlNode
				continue
			}
			out[i] = d
		}
		return out, nil
	}
}

// middlewareName returns a middleware's identity for the duplicate check,
// mirroring Python's AgentMiddleware.name (types.py:410-417): an explicit
// Name() when the middleware implements MiddlewareNamer, else the Go type
// name (the analog of Python's class-name default). Functional adapters
// (FuncBeforeModel and friends) share one Go type, so their name folds in
// the wrapped function's address — two adapters around different functions
// are distinct middleware (Python derives distinct class names from the
// function names), while the same function lifted twice collides exactly as
// two instances of one Python function middleware do.
func middlewareName(mw any) string {
	if namer, ok := mw.(MiddlewareNamer); ok {
		if name := namer.Name(); name != "" {
			return name
		}
	}
	return fmt.Sprintf("%T", mw)
}

// validateMiddlewareNames rejects duplicate middleware names at build time,
// mirroring factory.py:1080-1082 (`len({m.name for m in middleware}) !=
// len(middleware)` → AssertionError "Please remove duplicate middleware
// instances.").
func validateMiddlewareNames(mws []any) error {
	seen := make(map[string]struct{}, len(mws))
	for _, mw := range mws {
		name := middlewareName(mw)
		if _, dup := seen[name]; dup {
			return fmt.Errorf("agents: duplicate middleware instances (name %q); please remove duplicate middleware instances", name)
		}
		seen[name] = struct{}{}
	}
	return nil
}

// buildModelNode returns the graph node function driving one model call:
// BeforeModel hooks, the (middleware-wrapped) model invocation, then
// AfterModel hooks.
//
// resolveModel picks the ChatModel for this call: the agent's static model,
// or the DynamicModel resolver's answer when one is configured (resolved
// after BeforeModel hooks so their state updates are visible to it).
//
// systemPrompt resolves the system-prompt string for this call: a literal when
// WithAgentSystemPrompt is used, or a core/prompts render (with build-time +
// per-Invoke variables) when WithAgentSystemPromptTemplate is used.
//
// initialResponseFormat is the agent's configured ResponseFormat preserved in
// pointer form (nil / *ToolStrategy / *ProviderStrategy / *AutoStrategy — raw
// schemas are wrapped in an *AutoStrategy). Unlike the eagerly resolved
// toolStrategy/providerStrategy (which freeze the graph wiring: structured
// bindings, routing), it stays with the request so the strategy can be
// re-normalized on EVERY model call against the request's actual model and
// any middleware override (see resolveEffectiveResponseFormat; Python
// factory.py:1323-1344 re-derives effective_response_format per call, so an
// AutoStrategy is re-checked after a DynamicModel swap and a middleware
// WithResponseFormat override — including ToolStrategy↔ProviderStrategy
// switches — replaces the build-time strategy for that call).
//
// agentName, when non-empty, is stamped onto the model's output AIMessage
// (factory.py:1418-1419: `if name: output.name = name` inside
// _execute_model_sync — the create_agent name= value, applied before
// wrap_model_call middleware observe the result).
//
// logger, when non-nil, emits verbose debug logs (see WithAgentDebug) for node
// entry, the model call, and structured-output detection.
//
// hitlNodePresent wires the AI-message ID minting: when the graph carries the
// dedicated hitl node, every ID-less AI message this node produces gets a
// collision-free ID (see mintAIMessageIDs), because the HITL decision revises
// the committed AIMessage in place and MessagesReducer replaces messages by
// ID — an ID-less revision would append a duplicate instead.
func buildModelNode(
	resolveModel func(rt runtime.Runtime, state map[string]any) language.ChatModel,
	toolList []coretools.Tool,
	systemPrompt func(ctx context.Context) string,
	logger *slog.Logger,
	mws []any,
	structuredBindings map[string]OutputToolBinding,
	toolStrategy *ToolStrategy,
	providerStrategy *ProviderStrategy,
	initialResponseFormat any,
	finalNode string,
	cache caches.Cache,
	agentName string,
	hitlNodePresent bool,
) graphpkg.NodeFunc {
	toolsAny := toolsToAny(toolList)

	return func(rt runtime.Runtime, rawState map[string]any) (any, error) {
		state := cloneMapState(rawState)

		localMessages, _ := state["messages"].([]messages.Message)
		if logger != nil {
			logger.Info("agents: model node entry",
				slog.Int("messages", len(localMessages)),
				slog.Int("tools", len(toolsAny)))
		}
		// baseUpdate carries BeforeModel non-"messages" keys into this node's
		// returned update so they persist, mirroring Python where before_model
		// hooks are dedicated graph nodes whose state updates commit ("messages"
		// stays local-only by design; see the package doc comment).
		baseUpdate := map[string]any{}
		for _, mw := range mws {
			if hook, ok := mw.(BeforeModelCommandHook); ok {
				cmd, err := hook.BeforeModel(rt, state)
				if err != nil {
					return nil, err
				}
				if cmd != nil {
					return &types.Command{
						Update: cmd.Update,
						Goto:   graphpkg.To(resolveJumpTarget(cmd.Goto, finalNode)),
					}, nil
				}
				continue
			}
			hook, ok := mw.(BeforeModelHook)
			if !ok {
				continue
			}
			update, err := hook.BeforeModel(rt, state)
			if err != nil {
				return nil, err
			}
			if update == nil {
				continue
			}
			if jumpTo, ok := popJumpTo(update); ok {
				return &types.Command{Update: update, Goto: graphpkg.To(resolveJumpTarget(jumpTo, finalNode))}, nil
			}
			// See the package doc comment: "messages" updates from
			// BeforeModel hooks only reshape the local model-call view, they
			// are not persisted into committed graph state.
			if msgs, ok := update["messages"].([]messages.Message); ok {
				localMessages = msgs
				delete(update, "messages")
			}
			for k, v := range update {
				state[k] = v
				baseUpdate[k] = v
			}
		}

		resolvedPrompt := systemPrompt(rt)
		resolvedModel := resolveModel(rt, state)
		if resolvedModel == nil {
			return nil, fmt.Errorf("agents: model resolver returned nil for this model call")
		}
		// Node-level strategy normalization (mirrors the request-construction
		// half of Python's _get_bound_model): against the model resolved for
		// THIS call, so a DynamicModel swap re-checks an AutoStrategy even
		// before middleware runs. The derived ProviderStrategy kwargs seed the
		// request's ModelSettings so WrapModelCall middleware can observe the
		// intent and the cache key distinguishes the effective strategy (see
		// cacheKey); the inner handler re-normalizes after middleware overrides.
		nodeEffective, nodeBindings, err := resolveEffectiveResponseFormat(initialResponseFormat, resolvedModel, toolsAny, initialResponseFormat, toolStrategy, providerStrategy, structuredBindings)
		if err != nil {
			return nil, err
		}
		initialSettings := providerStrategyModelSettings(nodeEffective.provider)
		// Seeded from the PER-CALL effective strategy (not the build-time
		// toolStrategy), so the request WrapModelCall middleware observe is
		// self-consistent with initialSettings: an AutoStrategy whose
		// node-level re-resolution swapped Tool↔Provider (e.g. a DynamicModel
		// swap) seeds "any"/"" matching what the bind will actually thread.
		initialToolChoice := structuredOutputToolChoice(nodeEffective.tool)
		req, err := middleware.NewModelRequest(middleware.ModelRequest{
			Model:    resolvedModel,
			Messages: localMessages,
			Tools:    toolsAny,
			// initialResponseFormat rides on the request so WrapModelCall
			// middleware observe the configured strategy and may replace it
			// via ModelRequest.Override(middleware.WithResponseFormat(...))
			// (Python's ModelRequest.response_format, factory.py:1438).
			ResponseFormat: initialResponseFormat,
			SystemPrompt:   resolvedPrompt,
			State:          state,
			Runtime:        rt,
			ModelSettings:  initialSettings,
			// Mirrors factory.py:1388 (`tool_choice = "any" if
			// structured_output_tools else request.tool_choice`): a
			// ToolStrategy binds its schema tools and forces the model to
			// answer through one of them, so the request carries tool_choice
			// "any" for WrapModelCall middleware to observe and for the bind
			// to thread into the model when it implements
			// language.ToolBinder. Middleware may still override it via
			// ModelRequest.Override(middleware.WithToolChoice(...)).
			ToolChoice: initialToolChoice,
		})
		if err != nil {
			return nil, err
		}

		// effectiveTool/effectiveProvider/effectiveBindings hold the strategy
		// the handler's model call actually ran under. They start at the
		// node-level normalization (so a cache hit — which skips the handler —
		// still detects structured output under the current call's strategy,
		// like Python whose cache sits inside the chat-model layer and whose
		// _handle_model_output runs on the freshly re-derived effective
		// format) and are overwritten by each handler run with the
		// middleware-overridden normalization.
		effectiveTool, effectiveProvider, effectiveBindings := nodeEffective.tool, nodeEffective.provider, nodeBindings

		handler := func(c context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
			if logger != nil {
				logger.Info("agents: model call",
					slog.Int("messages", len(r.Messages)),
					slog.Bool("has_system_prompt", r.SystemMessage != nil))
			}
			model, ok := r.Model.(language.ChatModel)
			if !ok || model == nil {
				return middleware.ModelResponse{}, fmt.Errorf("agents: ModelRequest.Model must be a language.ChatModel, got %T", r.Model)
			}
			// Per-call re-normalization after middleware overrides
			// (factory.py:1323-1344): the request's response_format wins over
			// the build-time strategy, an AutoStrategy re-resolves against the
			// request's actual model, and the resulting effective strategy
			// drives both the bind below and the post-call structured-output
			// detection.
			eff, effBindings, err := resolveEffectiveResponseFormat(r.ResponseFormat, model, r.Tools, initialResponseFormat, toolStrategy, providerStrategy, structuredBindings)
			if err != nil {
				return middleware.ModelResponse{}, err
			}
			prepared, err := prepareModelCall(r, model, eff, effBindings, structuredBindings, initialToolChoice, initialSettings)
			if err != nil {
				return middleware.ModelResponse{}, err
			}
			effectiveTool, effectiveProvider, effectiveBindings = eff.tool, eff.provider, effBindings
			// Streaming path: when an event sink is active (i.e. the run was
			// started via Agent.StreamEvents / graph.InvokeStream), drive the
			// model through Stream, emit a model_delta per chunk, and assemble
			// the final message via core/streamevents.ChatModelStream before
			// emitting model_end. When no sink is active, the non-streaming
			// Invoke path is used with zero added overhead (see invokeModel).
			if sink := sinkFromContext(c); sink != nil {
				return invokeModelStreaming(c, r, prepared, sink, mws, agentName)
			}
			return invokeModel(c, r, prepared, agentName)
		}
		// mwCommands accumulates the update-only Commands returned by
		// WrapModelCallResult middleware (each layer's
		// ExtendedModelResponse.command, surfaced by
		// middleware.NormalizeModelCallResult), inner-first then outer —
		// mirroring factory._chain_model_call_handlers' command accumulation
		// (factory.py:258-275). They are validated and applied by
		// applyModelNodeCommands once the model call returns. A middleware
		// that successfully invokes its inner handler more than once would
		// accumulate the inner command per pass; no shipped middleware does
		// (retry middleware only re-invokes on error, and inner commands only
		// materialize on success), whereas Python additionally clears its
		// per-pair accumulator on each inner call (factory.py:311).
		var mwCommands []*middleware.Command
		for i := len(mws) - 1; i >= 0; i-- {
			if hook, ok := mws[i].(WrapModelCallResultHook); ok {
				next := handler
				handler = func(c context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
					result, err := hook.WrapModelCallResult(c, r, next)
					if err != nil {
						return middleware.ModelResponse{}, err
					}
					// Normalize the ModelCallResult union (AIMessage short form,
					// ExtendedModelResponse unwrap) at the composition boundary,
					// mirroring factory._normalize_to_model_response, and capture
					// any ExtendedModelResponse command for the node to apply.
					resp, cmd, err := middleware.NormalizeModelCallResult(result)
					if err != nil {
						return middleware.ModelResponse{}, err
					}
					if cmd != nil {
						mwCommands = append(mwCommands, cmd)
					}
					return resp, nil
				}
				continue
			}
			hook, ok := mws[i].(WrapModelCallHook)
			if !ok {
				continue
			}
			next := handler
			handler = func(c context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
				return hook.WrapModelCall(c, r, next)
			}
		}

		// The cache is scoped to the non-streaming Invoke path: when a stream
		// sink is active, the streaming path (invokeModelStreaming) owns this
		// call and must emit its own model_delta/model_end events — a cache hit
		// here would short-circuit the handler and the consumer would see an
		// abrupt, event-less completion. We also skip writing tool-call
		// responses to the cache: messages.Text drops a message's
		// ToolCalls/ToolCallID, so a cached tool call would be rebuilt as a
		// text-only AI message on lookup, breaking a tool-calling agent on a
		// second identical Invoke (the agent would get plain text instead of
		// routing through the tool). Only terminal text responses are cached.
		//
		// On a cache hit the entire model call (including WrapModelCall
		// middleware) is skipped. This is a deliberate divergence from Python:
		// there BaseCache sits inside the chat-model layer and wrap_model_call
		// middleware DOES observe cache-served calls. Skipping the whole chain
		// here means summarization/PII/retry middleware does not re-run on a
		// cached answer, which is the intended behavior. See WithAgentCache's
		// doc comment for the full rationale.
		var resp middleware.ModelResponse
		cacheHit := false
		cacheEnabled := cache != nil && sinkFromContext(rt) == nil
		if cacheEnabled {
			promptString, llmString := cacheKey(req)
			gens, ok, lerr := cache.Lookup(rt, promptString, llmString)
			if lerr != nil {
				// A failing cache must not masquerade as a miss (Python's
				// BaseCache.lookup propagates too): that would silently
				// disable caching for the run and make a permanently broken
				// cache indistinguishable from a cold one.
				return nil, fmt.Errorf("agents: model cache lookup: %w", lerr)
			}
			if ok && len(gens) > 0 {
				cached := make([]messages.Message, 0, len(gens))
				for _, g := range gens {
					cached = append(cached, messages.AI(g.Text))
				}
				// Cache-served outputs get the agent name too: Python's
				// `output.name = name` runs after model_.invoke returns,
				// which is where a cache hit surfaces (factory.py:1418-1419).
				applyAgentName(cached, agentName)
				resp = middleware.ModelResponse{Result: cached}
				cacheHit = true
				if logger != nil {
					logger.Info("agents: model cache hit",
						slog.Int("generations", len(gens)))
				}
			}
		}
		if !cacheHit {
			resp, err = handler(rt, req)
			if err != nil {
				return nil, err
			}
			// Write the fresh model response back to the cache before
			// structured-output detection / AfterModel hooks run, so the cached
			// value is the raw model output (not post-hook additions). Errors
			// from Update are non-fatal (cache is best-effort); logging them
			// would be noisy, so they are silently ignored. A response that
			// carries tool calls is never written (see the cacheEnabled note
			// above): only terminal text responses are cacheable. The streaming
			// path is also skipped (cacheEnabled is false when a sink is
			// active) so streamed runs neither read from nor pollute the cache.
			if cacheEnabled && len(resp.Result) > 0 && !anyResultHasToolCalls(resp.Result) {
				promptString, llmString := cacheKey(req)
				generations := make([]caches.Generation, 0, len(resp.Result))
				for _, m := range resp.Result {
					generations = append(generations, caches.Generation{Text: messages.Text(m)})
				}
				_ = cache.Update(rt, promptString, llmString, generations)
			}
		}
		newMessages := append([]messages.Message(nil), resp.Result...)
		if hitlNodePresent {
			promptString, _ := cacheKey(req)
			mintAIMessageIDs(newMessages, promptString)
		}
		if logger != nil {
			logger.Info("agents: model response",
				slog.Int("new_messages", len(newMessages)))
		}

		// Structured output detection happens before AfterModel hooks (see
		// WithAgentResponseFormat's doc comment): a matched structured tool
		// call, or a ProviderStrategy JSON parse, ends the run immediately
		// without executing any tools or running AfterModel hooks. A
		// HandleErrors retry appends error ToolMessages after the model output
		// (the calls count as answered, so routing loops back to the model);
		// AfterModel hooks still run on the retry path, matching Python where
		// after_model nodes sit between the model node and its routing edge.
		// The detection runs under the EFFECTIVE strategy of the model call
		// (middleware override / AutoStrategy re-resolution included — Python's
		// _handle_model_output receives effective_response_format from
		// _get_bound_model, factory.py:1408-1413).
		decision, cmd, retryToolMsgs, err := detectStructuredOutput(newMessages, effectiveBindings, effectiveTool, effectiveProvider, finalNode)
		if err != nil {
			return nil, err
		}
		switch decision {
		case structuredDone:
			// Persist any BeforeModel non-"messages" keys even on the
			// structured-match exit: Python commits before_model updates in
			// their own node before the model node's Command is produced.
			for k, v := range baseUpdate {
				if k != "messages" && k != "structured_response" {
					cmd.Update[k] = v
				}
			}
			// wrap_model_call middleware commands apply on this exit path too
			// (Python's model node returns them alongside the structured
			// response command).
			if err := applyModelNodeCommands(cmd.Update, mwCommands); err != nil {
				return nil, err
			}
			return cmd, nil
		case structuredRetry:
			newMessages = append(newMessages, retryToolMsgs...)
		}

		update := map[string]any{"messages": newMessages}
		for k, v := range baseUpdate {
			update[k] = v
		}
		if err := applyModelNodeCommands(update, mwCommands); err != nil {
			return nil, err
		}
		// Keep newMessages in sync with any middleware-appended messages so
		// AfterModel hooks (and the afterState they observe) see the merged
		// view — Python's after_model nodes run after the model node's
		// commands, middleware commands included, are committed.
		if merged, ok := update["messages"].([]messages.Message); ok {
			newMessages = merged
		}

		afterState := cloneMapState(state)
		afterState["messages"] = append(append([]messages.Message(nil), localMessages...), newMessages...)

		gotoOverride := ""
		for _, mw := range mws {
			hook, ok := mw.(AfterModelHook)
			if !ok {
				continue
			}
			hookUpdate, err := hook.AfterModel(rt, afterState)
			if err != nil {
				return nil, err
			}
			if hookUpdate == nil {
				continue
			}
			if jumpTo, ok := popJumpTo(hookUpdate); ok {
				gotoOverride = resolveJumpTarget(jumpTo, finalNode)
			}
			if extra, ok := hookUpdate["messages"].([]messages.Message); ok {
				delete(hookUpdate, "messages")
				newMessages = append(newMessages, extra...)
				update["messages"] = newMessages
				afterState["messages"] = append(afterState["messages"].([]messages.Message), extra...)
			}
			for k, v := range hookUpdate {
				update[k] = v
				afterState[k] = v
			}
		}

		if gotoOverride != "" {
			return &types.Command{Update: update, Goto: graphpkg.To(gotoOverride)}, nil
		}
		return update, nil
	}
}

// aiMessageIDSeq is the monotonic factor folded into every ID minted by
// mintAIMessageIDs (one bump per minting call). The graph step is not
// observable inside a graph NodeFunc (its signature is
// func(runtime, state) — no step counter), so the factor is process-level.
// That is safe for the one consumer of these IDs — HITL revision replacing a
// committed AIMessage by ID: the model node is never replayed (the hitl node
// gates execution between the model and the tools, so a resume re-runs only
// the hitl node), a single process therefore mints strictly increasing
// factors, and a resumed process never re-mints IDs for already-committed
// messages.
var aiMessageIDSeq atomic.Uint64

// mintAIMessageIDs assigns an ID to every ID-less AI message in msgs, derived
// from the model call's prompt string, the message content, the message index
// within the response, and a per-call monotonic counter
// (sha256(promptString\x1fcontent\x1fseq\x1findex)[:8] hex). It runs only when
// the graph wires the dedicated hitl node: the revised AIMessage a HITL
// decision returns must replace the committed one through MessagesReducer's
// ID match, which requires the committed message to carry an ID (models and
// messages.AI leave ID empty). The prompt string alone does NOT guarantee
// uniqueness: with keep-last-K history trimming two turns' local prompts can
// be byte-identical, and a deterministic or cached model can then return
// identical content — the hash would collide and MessagesReducer would
// silently REPLACE the earlier committed message instead of appending. The
// monotonic counter (and the per-message index, which keeps two
// identical-content messages in one response distinct) rules that out.
func mintAIMessageIDs(msgs []messages.Message, promptString string) {
	seq := aiMessageIDSeq.Add(1)
	for i := range msgs {
		if msgs[i].Role != messages.RoleAI || msgs[i].ID != "" {
			continue
		}
		preimage := strings.Join([]string{
			promptString,
			msgs[i].Content,
			strconv.FormatUint(seq, 10),
			strconv.Itoa(i),
		}, "\x1f")
		sum := sha256.Sum256([]byte(preimage))
		msgs[i].ID = hex.EncodeToString(sum[:])[:8]
	}
}

// unansweredToolCalls returns the last AI message's tool calls minus those
// already answered by a ToolMessage after it, mirroring the pending filter of
// Python's `_make_model_to_tools_edge` (factory.py:1870-1875): Python routes
// only pending calls into the tools node (one Send per call), so a call a
// HITL reject/respond decision already answered artificially is never
// re-executed even when a sibling call is still pending. Without the filter,
// a mixed approve/reject batch would run the rejected tool a second time and
// duplicate its ToolMessage.
func unansweredToolCalls(msgs []messages.Message) []messages.ToolCall {
	lastAI, toolMsgs := fetchLastAIAndToolMessages(msgs)
	if lastAI == nil {
		return nil
	}
	answered := make(map[string]bool, len(toolMsgs))
	for _, m := range toolMsgs {
		answered[m.ToolCallID] = true
	}
	pending := make([]messages.ToolCall, 0, len(lastAI.ToolCalls))
	for _, call := range lastAI.ToolCalls {
		if answered[call.ID] {
			continue
		}
		pending = append(pending, call)
	}
	return pending
}

// applyModelNodeCommands applies the update-only Commands returned by
// wrap_model_call middleware (each layer's ExtendedModelResponse.command,
// accumulated inner-first at the composition boundary) to the model node's
// pending update, mirroring factory._build_commands (factory.py:193-232):
// the node's default state update commits first and the middleware Commands
// are appended after it as additional Commands.
//
// Go graph nodes commit a single update per execution (unlike Python's
// list-of-Commands node return), so the additional Commands are folded into
// that one update map while preserving their sequential semantics:
//   - a Command carrying Goto/Resume/Graph is a hard error
//     (middleware.Command.ValidateForWrapModelCall — Python raises
//     NotImplementedError there);
//   - "messages" (an append-reducer channel) is concatenated after the
//     default messages, matching the result of Python's two sequential
//     add_messages writes;
//   - every other key is overridden by the middleware value, matching the
//     last-write-wins application of Python's sequential Commands (so on a
//     conflicting key the later — outer — middleware Command wins).
//
// AfterModel hook updates are applied after this and still win conflicts,
// matching Python where after_model nodes run after the model node's
// commands are committed.
func applyModelNodeCommands(update map[string]any, mwCommands []*middleware.Command) error {
	for _, cmd := range mwCommands {
		if err := cmd.ValidateForWrapModelCall(); err != nil {
			return err
		}
		if len(cmd.Update) == 0 {
			continue
		}
		for k, v := range cmd.Update {
			if k == "messages" {
				if extra, ok := v.([]messages.Message); ok {
					base, _ := update["messages"].([]messages.Message)
					merged := make([]messages.Message, 0, len(base)+len(extra))
					merged = append(merged, base...)
					merged = append(merged, extra...)
					update["messages"] = merged
					continue
				}
			}
			update[k] = v
		}
	}
	return nil
}

// resolveResponseFormat validates and unpacks an AgentOptions.ResponseFormat
// value into (at most) one of a ToolStrategy or ProviderStrategy. See
// WithAgentResponseFormat's doc comment for accepted types and behavior.
//
// An AutoStrategy (or *AutoStrategy) is resolved eagerly at CreateAgent time
// against model into a concrete ToolStrategy/ProviderStrategy via its Resolve
// method, then re-dispatched; this keeps AutoStrategy a build-time selection
// over the agent's bound model rather than a per-Invoke branch, matching how
// the other strategies are unpacked once up front. model is the agent's bound
// ChatModel (the positional first arg to CreateAgent); it is required only for
// the AutoStrategy path and is otherwise ignored.
func resolveResponseFormat(format any, model language.ChatModel) (*ToolStrategy, *ProviderStrategy, error) {
	switch v := format.(type) {
	case nil:
		return nil, nil, nil
	case ToolStrategy:
		return &v, nil, nil
	case *ToolStrategy:
		return v, nil, nil
	case ProviderStrategy:
		return nil, &v, nil
	case *ProviderStrategy:
		return nil, v, nil
	case AutoStrategy:
		resolved, err := v.Resolve(model)
		if err != nil {
			return nil, nil, err
		}
		return resolveResponseFormat(resolved, model)
	case *AutoStrategy:
		if v == nil {
			return nil, nil, nil
		}
		resolved, err := v.Resolve(model)
		if err != nil {
			return nil, nil, err
		}
		return resolveResponseFormat(resolved, model)
	// Raw JSON schema (Python's `response_format: dict` overload): wrap in an
	// AutoStrategy to preserve auto-detection intent, matching Python, then
	// resolve against the bound model. Both the plain map spelling and the
	// schema.Schema spelling (the type the strategy constructors take) are
	// accepted.
	case map[string]any:
		return resolveResponseFormat(NewAutoStrategy(schema.Schema(v)), model)
	case schema.Schema:
		return resolveResponseFormat(NewAutoStrategy(v), model)
	default:
		return nil, nil, fmt.Errorf("agents: unsupported ResponseFormat type %T (expected ToolStrategy, ProviderStrategy, AutoStrategy, or a raw JSON schema)", format)
	}
}

// buildStructuredOutputTools converts a ToolStrategy's SchemaSpecs into
// callable tools the model can be bound to, keyed by tool name for lookup by
// detectStructuredOutput once the model responds.
func buildStructuredOutputTools(strategy *ToolStrategy) (map[string]OutputToolBinding, []coretools.Tool, error) {
	bindings := make(map[string]OutputToolBinding, len(strategy.SchemaSpecs))
	extraTools := make([]coretools.Tool, 0, len(strategy.SchemaSpecs))
	for _, spec := range strategy.SchemaSpecs {
		binding, err := OutputToolBindingFromSchemaSpec(spec)
		if err != nil {
			return nil, nil, err
		}
		bindings[spec.Name] = binding
		extraTools = append(extraTools, binding.Tool)
	}
	return bindings, extraTools, nil
}

// providerStrategyModelSettings surfaces a ProviderStrategy's model kwargs
// (e.g. a provider's native `response_format`) via ModelRequest.ModelSettings
// so WrapModelCall middleware can observe the caller's intent and the model
// node's cache key distinguishes the strategy (see cacheKey). The kwargs also
// reach the actual bind: see mergedBindSettings/prepareModelCall.
func providerStrategyModelSettings(providerStrategy *ProviderStrategy) map[string]any {
	if providerStrategy == nil {
		return nil
	}
	return providerStrategy.ToModelKwargs()
}

// structuredOutputToolChoice returns the tool_choice the model-node request
// should carry: "any" when a ToolStrategy's structured-output tools are in
// play (forcing the model to answer through one of them, factory.py:1388),
// or "" (no constraint) otherwise.
func structuredOutputToolChoice(toolStrategy *ToolStrategy) any {
	if toolStrategy == nil {
		return ""
	}
	return string(language.ToolChoiceAny)
}

// ModelSettingsBinder is an optional language.ChatModel capability for models
// that accept per-call model kwargs — the Go stand-in for Python's
// `model.bind_tools(tools, **model_settings)` / `model.bind(**model_settings)`
// kwargs passthrough (factory.py:1360-1404). Implementations return a copy of
// the model configured with settings (e.g. a provider-native
// "response_format" dict, a "temperature", or any other provider kwarg); the
// agents layer calls it on EVERY model call (streaming and non-streaming
// alike) with the merged settings (effective-strategy kwargs overlaid by
// middleware ModelSettings overrides).
//
// Capability detection + degradation (this port does not modify partner
// packages): no shipped partner model implements this interface yet —
// partners/openai exposes WithResponseFormat and partners/anthropic
// WithThinking/WithToolChoice, but each returns the partner's own concrete
// type, which cannot be duck-typed generically from the agents layer. When a
// model does not implement ModelSettingsBinder:
//   - a ProviderStrategy's response_format still reaches the model through
//     language.StructuredCaller on the non-streaming path (see invokeModel);
//   - on the streaming path the settings degrade to the retained post-hoc
//     detectStructuredOutput JSON parse (see invokeModelStreaming);
//   - other settings (temperature, ...) are dropped, mirroring how
//     bindModelTools drops tool_choice for models without language.ToolBinder.
type ModelSettingsBinder interface {
	// BindModelSettings returns a copy of the model carrying settings as
	// per-call bind kwargs. settings must not be retained or mutated by the
	// implementation.
	BindModelSettings(settings map[string]any) (language.ChatModel, error)
}

// modelProfileProvider and modelNameProvider are the duck-typed surfaces
// AutoStrategy re-resolution reads off a per-call model (Python's
// `_supports_provider_strategy` reads `model.model_name`/`model.model`/
// `model.model_id` and `model.profile`, factory.py:528-570). language's
// FakeChatModel implements ModelProfile; partner chat models currently expose
// neither (see ModelSettingsBinder's degradation note).
type modelProfileProvider interface {
	ModelProfile() modelprofiles.Profile
}

type modelNameProvider interface {
	ModelName() string
}

// modelSupportsProviderStrategy reports whether the model of THIS call should
// take the ProviderStrategy branch of AutoStrategy resolution, mirroring
// `_supports_provider_strategy(request.model, tools=request.tools)`
// (factory.py:1333). Models exposing a profile and/or a model name are
// checked through the shared SupportsProviderStrategy (profile
// structured_output flag first, then the fallback name regexes — including
// the Gemini<3-with-tools exception). Models exposing neither (today's
// partner chat models) degrade to the Capabilities().StructuredOutput flag,
// the nearest existing signal; no partner package is modified for this.
func modelSupportsProviderStrategy(model language.ChatModel, tools []any) bool {
	if model == nil {
		return false
	}
	info := ModelInfo{}
	found := false
	if namer, ok := model.(modelNameProvider); ok {
		name := namer.ModelName()
		info.ModelName, info.Model, info.ModelID = name, name, name
		found = true
	}
	if profiler, ok := model.(modelProfileProvider); ok {
		info.Profile = profiler.ModelProfile()
		found = true
	}
	if found {
		return SupportsProviderStrategy(info, tools)
	}
	return model.Capabilities().StructuredOutput
}

// normalizeInitialResponseFormat canonicalizes a configured ResponseFormat
// into pointer form (nil / *ToolStrategy / *ProviderStrategy / *AutoStrategy)
// so the request can carry it without Go interface-value comparison pitfalls
// and so sameStrategyInstance can compare identity. A raw JSON schema
// (Python's `response_format: dict` overload) is wrapped in an
// *AutoStrategy, exactly like factory.py:1327-1331 normalizes raw schemas.
func normalizeInitialResponseFormat(format any) any {
	switch v := format.(type) {
	case nil:
		return nil
	case ToolStrategy:
		return &v
	case *ToolStrategy:
		if v == nil {
			return nil
		}
		return v
	case ProviderStrategy:
		return &v
	case *ProviderStrategy:
		if v == nil {
			return nil
		}
		return v
	case AutoStrategy:
		return &v
	case *AutoStrategy:
		if v == nil {
			return nil
		}
		return v
	case map[string]any:
		auto := NewAutoStrategy(schema.Schema(v))
		return &auto
	case schema.Schema:
		auto := NewAutoStrategy(v)
		return &auto
	default:
		return format
	}
}

// sameStrategyInstance reports whether a and b are the same strategy instance
// (pointer identity per strategy type). Used to distinguish "the request
// still carries the agent's initial response format" from "middleware
// replaced it" (Python compares `response_format is initial_response_format`,
// factory.py:1336).
func sameStrategyInstance(a, b any) bool {
	switch av := a.(type) {
	case *ToolStrategy:
		bv, ok := b.(*ToolStrategy)
		return ok && av == bv
	case *ProviderStrategy:
		bv, ok := b.(*ProviderStrategy)
		return ok && av == bv
	case *AutoStrategy:
		bv, ok := b.(*AutoStrategy)
		return ok && av == bv
	}
	return false
}

// effectiveFormat is the per-call normalized response format: at most one of
// tool/provider is non-nil (nil/nil = no structured output). It is the Go
// analog of Python's `effective_response_format` returned by
// _get_bound_model (factory.py:1323-1344).
type effectiveFormat struct {
	tool     *ToolStrategy
	provider *ProviderStrategy
}

// resolveEffectiveResponseFormat re-derives the effective strategy for one
// model call from the (possibly middleware-overridden) requested format,
// mirroring factory.py:1323-1344:
//
//   - nil → the agent's build-time strategies (toolStrategy/providerStrategy);
//   - a raw JSON schema → wrapped in an AutoStrategy;
//   - an AutoStrategy → ProviderStrategy when the CURRENT model supports it
//     (modelSupportsProviderStrategy), else the build-time ToolStrategy when
//     the request still carries the initial format (preserving tool names and
//     HandleErrors config, factory.py:1336-1339), else a fresh ToolStrategy;
//   - an explicit ToolStrategy → used as-is, validated as a subset of the
//     structured tools declared at build time (factory.py:1375-1385:
//     middleware may narrow but not introduce structured tools), with the
//     bindings narrowed to the override's own specs;
//   - an explicit ProviderStrategy → used as-is.
//
// setupBindings are the build-time structured-output bindings (empty when the
// build-time resolution was a ProviderStrategy or no format). The returned
// bindings map is what structured-output detection should match against for
// this call: the build-time map, narrowed to the override's own specs when
// middleware supplies a narrower ToolStrategy, and extended with fresh
// bindings when an AutoStrategy re-resolution synthesizes a ToolStrategy the
// build-time never declared (a DynamicModel swap onto a tool-calling model
// after a provider-only build-time resolution — Python's setup always
// declares the structured tools for an AutoStrategy, so its effective
// ToolStrategy can always reuse them; the Go port reconstructs them here
// instead).
func resolveEffectiveResponseFormat(
	requested any,
	model language.ChatModel,
	tools []any,
	initialFormat any,
	setupTool *ToolStrategy,
	setupProvider *ProviderStrategy,
	setupBindings map[string]OutputToolBinding,
) (effectiveFormat, map[string]OutputToolBinding, error) {
	rf := normalizeInitialResponseFormat(requested)
	if rf == nil {
		return effectiveFormat{tool: setupTool, provider: setupProvider}, setupBindings, nil
	}
	switch v := rf.(type) {
	case *AutoStrategy:
		if modelSupportsProviderStrategy(model, tools) {
			ps := NewProviderStrategy(v.Schema)
			return effectiveFormat{provider: &ps}, setupBindings, nil
		}
		if sameStrategyInstance(rf, initialFormat) && setupTool != nil {
			// Reuse the setup strategy to preserve tool names (factory.py:1336-1339).
			return effectiveFormat{tool: setupTool}, setupBindings, nil
		}
		ts := NewToolStrategy(v.Schema)
		if !sameStrategyInstance(rf, initialFormat) && len(setupBindings) > 0 {
			// Middleware replaced the AutoStrategy: the derived ToolStrategy
			// may only narrow to the declared structured tools.
			if err := validateToolStrategySpecsDeclared(&ts, setupBindings); err != nil {
				return effectiveFormat{}, nil, err
			}
			return effectiveFormat{tool: &ts}, setupBindings, nil
		}
		// Initial AutoStrategy whose build-time resolution declared no
		// structured tools (eager ProviderStrategy resolution): synthesize the
		// bindings the ToolStrategy needs for this call.
		bindings, _, err := buildStructuredOutputTools(&ts)
		if err != nil {
			return effectiveFormat{}, nil, err
		}
		return effectiveFormat{tool: &ts}, bindings, nil
	case *ToolStrategy:
		if !sameStrategyInstance(rf, initialFormat) {
			if err := validateToolStrategySpecsDeclared(v, setupBindings); err != nil {
				return effectiveFormat{}, nil, err
			}
			// Middleware narrowed the strategy: the bind and structured-output
			// detection run against the OVERRIDE's own specs, not the
			// build-time superset (factory.py:1375-1385 binds the effective
			// strategy's structured_output_tools).
			return effectiveFormat{tool: v}, filterBindingsToSpecs(v, setupBindings), nil
		}
		return effectiveFormat{tool: v}, setupBindings, nil
	case *ProviderStrategy:
		return effectiveFormat{provider: v}, setupBindings, nil
	default:
		return effectiveFormat{}, nil, fmt.Errorf("agents: unsupported ResponseFormat type %T (expected ToolStrategy, ProviderStrategy, AutoStrategy, or a raw JSON schema)", requested)
	}
}

// validateToolStrategySpecsDeclared enforces factory.py:1375-1385: every
// schema spec of a middleware-supplied ToolStrategy must name a structured
// tool declared in the agent's original response_format at build time.
func validateToolStrategySpecsDeclared(strategy *ToolStrategy, setupBindings map[string]OutputToolBinding) error {
	for _, spec := range strategy.SchemaSpecs {
		if _, ok := setupBindings[spec.Name]; !ok {
			return fmt.Errorf("agents: ToolStrategy specifies tool %q which wasn't declared in the original response format when creating the agent", spec.Name)
		}
	}
	return nil
}

// filterBindingsToSpecs narrows setupBindings to the specs a middleware
// ToolStrategy override declared (already validated by
// validateToolStrategySpecsDeclared), mirroring factory.py:1375-1385: the
// effective strategy binds and matches structured output against its own
// narrow set, never the build-time superset.
func filterBindingsToSpecs(strategy *ToolStrategy, setupBindings map[string]OutputToolBinding) map[string]OutputToolBinding {
	filtered := make(map[string]OutputToolBinding, len(strategy.SchemaSpecs))
	for _, spec := range strategy.SchemaSpecs {
		if binding, ok := setupBindings[spec.Name]; ok {
			filtered[spec.Name] = binding
		}
	}
	return filtered
}

// toolChoiceKey flattens a ModelRequest.ToolChoice value to a comparable
// string for "was it middleware-overridden?" comparisons.
func toolChoiceKey(choice any) string {
	switch v := choice.(type) {
	case nil:
		return ""
	case string:
		return v
	case language.ToolChoice:
		return string(v)
	default:
		return fmt.Sprint(v)
	}
}

// mergedBindSettings computes the per-call bind kwargs, mirroring
// factory.py:1361 (`bind_kwargs = {**kwargs, **request.model_settings}` — the
// request's settings win conflicts) and :1392/:1397 (tool-strategy /
// no-strategy binds pass request.model_settings alone):
//   - effective ProviderStrategy kwargs form the base (or none, when the
//     effective strategy is Tool/none);
//   - the request's ModelSettings overlay them ONLY when middleware actually
//     replaced them — a request still carrying the node-seeded initial value
//     (the node-level ProviderStrategy kwargs, kept for middleware
//     observability and cache-key separation) was not overridden, and the
//     stale build-time kwargs must not leak into a call whose effective
//     strategy changed (e.g. a Tool override of a ProviderStrategy setup).
func mergedBindSettings(eff effectiveFormat, requestSettings, initialSettings map[string]any) map[string]any {
	merged := map[string]any{}
	if eff.provider != nil {
		for k, v := range eff.provider.ToModelKwargs() {
			merged[k] = v
		}
	}
	if !reflect.DeepEqual(requestSettings, initialSettings) {
		for k, v := range requestSettings {
			merged[k] = v
		}
	}
	return merged
}

// effectiveBindToolChoice computes the tool_choice for the bind, mirroring
// factory.py:1366-1371 (ProviderStrategy: no tool_choice at all, even an
// overridden one), :1388 (`tool_choice = "any" if structured_output_tools
// else request.tool_choice`), and :1397 (no strategy: request.tool_choice).
// initialChoice is the value the model node seeded the request with; a
// request still carrying it was not middleware-overridden, and the seeded
// "any" default mirrors Python's None at bind time (it exists on the request
// only for WrapModelCall middleware observability).
func effectiveBindToolChoice(requestChoice, initialChoice any, eff effectiveFormat, bindings map[string]OutputToolBinding) any {
	overridden := toolChoiceKey(requestChoice) != toolChoiceKey(initialChoice)
	if eff.provider != nil {
		return nil
	}
	if eff.tool != nil && len(bindings) > 0 {
		return string(language.ToolChoiceAny)
	}
	if overridden {
		return requestChoice
	}
	return nil
}

// preparedModelCall carries the per-call bind result shared by the streaming
// and non-streaming model invocations: the bound model, the merged settings
// (and whether the model actually accepted them), and the effective strategy.
type preparedModelCall struct {
	model         language.ChatModel
	eff           effectiveFormat
	settings      map[string]any
	settingsBound bool
}

// prepareModelCall binds a (possibly middleware-overridden) ModelRequest for
// invocation — the Go equivalent of Python's `_get_bound_model`
// (factory.py:1350-1404), split into the capability calls the Go ChatModel
// interface actually exposes:
//
//   - ModelSettingsBinder.BindModelSettings receives the merged kwargs
//     (effective ProviderStrategy response_format overlaid by middleware
//     model_settings) on every call, streaming included. Models without the
//     capability degrade (see ModelSettingsBinder's doc comment).
//   - final tools: the request's tools minus the structured-output tools when
//     the effective strategy is not the ToolStrategy (factory.py:1351-1355),
//     plus any bindings the effective ToolStrategy synthesized at resolution
//     time; the setup structured tools that ride on the request (the Go
//     stand-in for Python extending final_tools with them — Python's
//     request.tools carries no structured tools) are narrowed to the
//     effective ToolStrategy's own specs, so a middleware override that
//     narrows a union ToolStrategy restricts which structured tools the model
//     may answer through (upstream factory.py #39259); tool_choice per
//     effectiveBindToolChoice threads through language.ToolBinder
//     (factory.py:1366-1404).
//
// setupBindings are the agent's build-time structured-output bindings (empty
// when the build-time resolution declared none) — the superset a narrowed
// effective ToolStrategy is filtered against.
func prepareModelCall(
	req middleware.ModelRequest,
	model language.ChatModel,
	eff effectiveFormat,
	bindings map[string]OutputToolBinding,
	setupBindings map[string]OutputToolBinding,
	initialToolChoice any,
	initialSettings map[string]any,
) (*preparedModelCall, error) {
	settings := mergedBindSettings(eff, req.ModelSettings, initialSettings)
	prepared := &preparedModelCall{model: model, eff: eff, settings: settings}

	// Settings bind first: Python threads the kwargs through the single
	// bind_tools call; Go splits it, and partner-style copy-semantics models
	// carry the settings field through the subsequent tool bind.
	if len(settings) > 0 {
		if binder, ok := model.(ModelSettingsBinder); ok {
			bound, err := binder.BindModelSettings(settings)
			if err != nil {
				return nil, fmt.Errorf("agents: bind model settings: %w", err)
			}
			if bound != nil {
				prepared.model = bound
				prepared.settingsBound = true
			}
		}
		// No ModelSettingsBinder: capability detection + degrade — see
		// ModelSettingsBinder's doc comment (StructuredCaller / post-hoc parse
		// still cover the ProviderStrategy response_format).
	}

	finalTools, err := toolsFromAny(req.Tools)
	if err != nil {
		return nil, err
	}
	if eff.tool != nil {
		present := make(map[string]bool, len(finalTools))
		for _, t := range finalTools {
			present[t.Name()] = true
		}
		// Narrow the setup structured tools that ride on the request down to
		// the effective strategy's own specs: middleware may narrow a union
		// ToolStrategy to a subset, and the bind must restrict the model to
		// that subset (upstream factory.py #39259 binds only the tools present
		// in the possibly middleware-narrowed response format).
		if len(setupBindings) > 0 {
			filtered := make([]coretools.Tool, 0, len(finalTools))
			for _, t := range finalTools {
				if _, setup := setupBindings[t.Name()]; setup {
					if _, keep := bindings[t.Name()]; keep {
						filtered = append(filtered, t)
					}
					continue
				}
				filtered = append(filtered, t)
			}
			finalTools = filtered
		}
		// Ensure the effective strategy's structured tools are present (they
		// are pre-baked from the setup except when resolution synthesized
		// fresh bindings for a dynamic-swap ToolStrategy).
		for name, binding := range bindings {
			if !present[name] {
				finalTools = append(finalTools, binding.Tool)
			}
		}
	} else if len(bindings) > 0 {
		// Not in tool mode: strip the structured-output tools from the bind
		// (factory.py:1351-1355 adds them only for a ToolStrategy).
		structured := make(map[string]bool, len(bindings))
		for name := range bindings {
			structured[name] = true
		}
		filtered := make([]coretools.Tool, 0, len(finalTools))
		for _, t := range finalTools {
			if !structured[t.Name()] {
				filtered = append(filtered, t)
			}
		}
		finalTools = filtered
	}

	if len(finalTools) > 0 {
		choice := effectiveBindToolChoice(req.ToolChoice, initialToolChoice, eff, bindings)
		bound, err := bindModelTools(prepared.model, finalTools, choice, settings)
		if err != nil {
			return nil, err
		}
		prepared.model = bound
	}
	return prepared, nil
}

// bindModelTools binds tools on model, threading toolChoice through when the
// model implements the optional language.ToolBinder capability. Models that
// only implement the base BindTools keep today's behavior: the choice cannot
// be enforced and is silently dropped, mirroring Python where bind_tools
// kwargs are honored per provider. toolChoice values that are not a string
// (or language.ToolChoice) are likewise dropped rather than fatal — before
// this capability existed ModelRequest.ToolChoice was never applied at all.
//
// settings is the per-call merged kwargs (see prepareModelCall); the
// "parallel_tool_calls" entry maps onto language.BindToolsOptions for
// ToolBinder models, the one bind_tools kwarg the Go options struct carries
// today.
func bindModelTools(model language.ChatModel, boundTools []coretools.Tool, toolChoice any, settings map[string]any) (language.ChatModel, error) {
	binder, ok := model.(language.ToolBinder)
	if !ok {
		return model.BindTools(boundTools)
	}
	opts := language.BindToolsOptions{}
	switch choice := toolChoice.(type) {
	case nil:
	case string:
		opts.ToolChoice = language.ToolChoice(choice)
	case language.ToolChoice:
		opts.ToolChoice = choice
	default:
		// Non-string tool_choice shapes (e.g. provider-native dicts) are not
		// expressible in language.BindToolsOptions yet; leave unconstrained.
	}
	if parallel, ok := settings["parallel_tool_calls"].(bool); ok {
		opts.ParallelToolCalls = &parallel
	}
	return binder.BindToolsWithOptions(boundTools, opts)
}

// structuredDecision is detectStructuredOutput's outcome for the model node.
type structuredDecision int

const (
	// structuredPass: no structured-output involvement; the model node
	// continues to AfterModel hooks and normal routing.
	structuredPass structuredDecision = iota
	// structuredDone: a structured response matched; cmd is a terminal
	// Command carrying the parsed value under "structured_response".
	structuredDone
	// structuredRetry: a structured-output failure the strategy elected to
	// retry; retryToolMsgs are error ToolMessages the model node appends after
	// the model's output so the loop routes back to the model (which sees its
	// own call plus the error and regenerates).
	structuredRetry
)

// detectStructuredOutput inspects the model's newMessages for a match against
// the EFFECTIVE strategy of the model call (middleware override / AutoStrategy
// re-resolution included — the model node passes the per-call strategies and
// bindings from resolveEffectiveResponseFormat, mirroring how Python's
// _handle_model_output receives _get_bound_model's effective_response_format,
// factory.py:1408-1413): a tool call into structuredBindings (ToolStrategy),
// or — absent any tool calls — a ProviderStrategy JSON-decodable text
// response. A match returns a terminal *types.Command (structuredDone)
// carrying the parsed value under state key "structured_response", ending the
// run without visiting the tools node or running AfterModel hooks.
//
// Failure retry: mirroring Python's model node (factory.py:1204-1270), a
// multiple-structured-outputs error or a single call's parse failure consults
// the ToolStrategy's HandleErrors policy (see handleStructuredOutputError).
// When the policy elects retry, detectStructuredOutput returns structuredRetry
// with one error ToolMessage per structured call; the model node appends them
// to the update and the graph routes back to the model (the calls count as
// answered, so buildRouteAfterModel's step 6 loops instead of dispatching
// tools or exiting). When the policy elects raise, the error is returned
// verbatim. ProviderStrategy parse failures always raise (Python does not
// retry them either).
func detectStructuredOutput(
	newMessages []messages.Message,
	structuredBindings map[string]OutputToolBinding,
	toolStrategy *ToolStrategy,
	providerStrategy *ProviderStrategy,
	finalNode string,
) (structuredDecision, *types.Command, []messages.Message, error) {
	if len(newMessages) == 0 {
		return structuredPass, nil, nil, nil
	}
	last := newMessages[len(newMessages)-1]
	if last.Role != messages.RoleAI {
		return structuredPass, nil, nil, nil
	}

	if toolStrategy != nil && len(structuredBindings) > 0 {
		matched := make([]messages.ToolCall, 0, 1)
		for _, call := range last.ToolCalls {
			if _, ok := structuredBindings[call.Name]; ok {
				matched = append(matched, call)
			}
		}
		if len(matched) > 1 {
			names := make([]string, len(matched))
			for i, call := range matched {
				names[i] = call.Name
			}
			exc := NewMultipleStructuredOutputsError(names, last)
			if shouldRetry, message := handleStructuredOutputError(toolStrategy, exc); shouldRetry {
				retry := make([]messages.Message, 0, len(matched))
				for _, call := range matched {
					m := messages.Tool(call.ID, message)
					m.Name = call.Name
					retry = append(retry, m)
				}
				return structuredRetry, nil, retry, nil
			}
			return structuredPass, nil, nil, exc
		}
		if len(matched) == 1 {
			call := matched[0]
			binding := structuredBindings[call.Name]
			parsed, err := binding.Parse(call.Args)
			if err != nil {
				exc := NewStructuredOutputValidationError(call.Name, err, last)
				if shouldRetry, message := handleStructuredOutputError(toolStrategy, exc); shouldRetry {
					m := messages.Tool(call.ID, message)
					m.Name = call.Name
					return structuredRetry, nil, []messages.Message{m}, nil
				}
				return structuredPass, nil, nil, exc
			}
			content := ""
			if toolStrategy != nil {
				content = toolStrategy.ToolMessageContent
			}
			content = cmp.Or(content, fmt.Sprintf("Returned structured response via %s.", call.Name))
			toolMsg := messages.Tool(call.ID, content)
			toolMsg.Name = call.Name
			updatedMessages := append(append([]messages.Message(nil), newMessages...), toolMsg)
			return structuredDone, &types.Command{
				Update: map[string]any{
					"messages":            updatedMessages,
					"structured_response": parsed,
				},
				Goto: graphpkg.To(finalNode),
			}, nil, nil
		}
	}

	if providerStrategy != nil && len(last.ToolCalls) == 0 {
		binding := ProviderStrategyBindingFromSchemaSpec(providerStrategy.SchemaSpec)
		parsed, err := binding.Parse(last)
		if err != nil {
			return structuredPass, nil, nil, err
		}
		return structuredDone, &types.Command{
			Update: map[string]any{
				"messages":            newMessages,
				"structured_response": parsed,
			},
			Goto: graphpkg.To(finalNode),
		}, nil, nil
	}

	return structuredPass, nil, nil, nil
}

// providerStrategySchema returns the JSON schema the model should enforce
// natively for a ProviderStrategy, or nil when none is in play. Kept for
// callers holding a bare *ProviderStrategy; the model node passes the
// effective strategy through preparedModelCall instead.
func providerStrategySchema(providerStrategy *ProviderStrategy) schema.Schema {
	if providerStrategy == nil {
		return nil
	}
	return providerStrategy.Schema
}

// applyAgentName stamps agentName onto every AI-role message in msgs,
// mirroring `output.name = name` in factory.py:1418-1419 (the create_agent
// name= value set on the model's output inside _execute_model_sync, before
// wrap_model_call middleware observe the result). A empty agentName is a
// no-op; non-AI messages are left untouched.
func applyAgentName(msgs []messages.Message, agentName string) {
	if agentName == "" {
		return
	}
	for i := range msgs {
		if msgs[i].Role == messages.RoleAI {
			msgs[i].Name = agentName
		}
	}
}

// invokeModel runs the actual chat model call for a (possibly
// middleware-overridden) ModelRequest, using the bind prepared by
// prepareModelCall (tools + tool_choice + model settings already applied; see
// preparedModelCall).
//
// When the effective strategy is a ProviderStrategy whose response_format
// kwargs were NOT already bound via ModelSettingsBinder AND the (possibly
// tool-bound) model implements language.StructuredCaller, the call routes
// through language.InvokeStructured — the provider-native structured-output
// path (e.g. OpenAI's response_format). The native path produces JSON text
// that detectStructuredOutput's post-hoc parse still handles, so the
// structured_response plumbing is unchanged. Models that do NOT implement
// StructuredCaller fall through to plain model.Invoke, preserving the existing
// post-hoc JSON-decode behavior. (When the kwargs WERE bound via
// ModelSettingsBinder the bound model already carries the provider-native
// response_format, so plain Invoke is used — Python parity, where the kwargs
// travel through bind_tools and model_.invoke is called directly.)
//
// Tool binding (if any) runs BEFORE the StructuredCaller check, so a model
// whose bound form implements StructuredCaller still takes the native path —
// the bound value (not the original) is what gets passed to InvokeStructured.
//
// agentName, when non-empty, is stamped onto the resulting AI message(s) (see
// applyAgentName).
//
// The streaming sibling (invokeModelStreaming) shares the same prepared bind,
// so a ProviderStrategy's response_format reaches the model on both paths
// when it implements ModelSettingsBinder; streaming additionally keeps the
// post-hoc detectStructuredOutput parse as the fallback for models that only
// implement StructuredCaller (or neither).
func invokeModel(ctx context.Context, req middleware.ModelRequest, prepared *preparedModelCall, agentName string) (middleware.ModelResponse, error) {
	model := prepared.model

	invokeMessages := req.Messages
	if req.SystemMessage != nil {
		invokeMessages = append([]messages.Message{*req.SystemMessage}, req.Messages...)
	}

	// ProviderStrategy native path: when the caller supplied a schema and
	// the (possibly bound) model implements StructuredCaller, route the call
	// through language.InvokeStructured, which prefers the native path. The
	// up-front interface check is intentional: language.InvokeStructured
	// would otherwise fall back to its own Invoke + JSON-validate path for
	// non-StructuredCaller models, which would surface schema-violation
	// errors here rather than letting the existing detectStructuredOutput
	// post-hoc parse own that behavior (preserving backward compatibility).
	if structuredSchema := providerStrategySchema(prepared.eff.provider); len(structuredSchema) > 0 && !prepared.settingsBound {
		if _, ok := model.(language.StructuredCaller); ok {
			result, err := language.InvokeStructured(ctx, model, invokeMessages, structuredSchema)
			if err != nil {
				return middleware.ModelResponse{}, err
			}
			resultMsgs := []messages.Message{result}
			applyAgentName(resultMsgs, agentName)
			return middleware.ModelResponse{Result: resultMsgs}, nil
		}
	}

	result, err := model.Invoke(ctx, invokeMessages)
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	resultMsgs := []messages.Message{result}
	applyAgentName(resultMsgs, agentName)
	return middleware.ModelResponse{Result: resultMsgs}, nil
}

// anyResultHasToolCalls reports whether any of msgs carries tool calls (i.e.
// the model is requesting tool execution rather than producing a terminal text
// response). Such responses must not be written to the cache: messages.Text
// drops a message's ToolCalls/ToolCallID, so a cached tool call would be
// rebuilt as a lossy text-only AI message on lookup, and a second identical
// Invoke would return plain text instead of routing through the tool.
func anyResultHasToolCalls(msgs []messages.Message) bool {
	for _, m := range msgs {
		if len(m.ToolCalls) > 0 {
			return true
		}
	}
	return false
}

// cacheKey derives (promptString, llmString) for a model request, mirroring
// langchain_core's BaseCache contract (caches.py:49-84: lookup/update key on
// `(prompt, llm_string)`, where llm_string serializes the model INCLUDING its
// bound kwargs — so in Python two calls whose bind differs, e.g. a different
// response_format or tool_choice, can never collide). promptString is the
// role-tagged, rendered message text (each message's role + messages.Text of
// its content); llmString identifies the model and its configuration as the
// model type plus a stable hash of the bound tool names, ModelSettings, and
// ToolChoice (so two requests that differ only in tools, settings, or tool
// choice do not collide).
//
// The request's ModelSettings carries the node-level effective strategy's
// kwargs (see buildModelNode), so two calls resolving to different effective
// response formats — e.g. an AutoStrategy re-checked after a DynamicModel
// swap — produce different keys, matching Python's bound-model llm_string.
// Known limitation, inherent to this port's intentional cache divergence (a
// hit skips the whole WrapModelCall chain — see WithAgentCache): middleware
// runtime overrides apply INSIDE the chain, after this key is derived, so a
// state-dependent override that flips response_format between two identical
// prompts cannot be distinguished by the key; the first cached entry is then
// served. Deterministic middleware overrides stay consistent (the same
// override produced the cached entry).
func cacheKey(req middleware.ModelRequest) (string, string) {
	var sb strings.Builder
	for _, m := range req.Messages {
		sb.WriteString(string(m.Role))
		sb.WriteByte(':')
		sb.WriteString(messages.Text(m))
		sb.WriteByte('\n')
	}
	if req.SystemMessage != nil {
		sb.WriteString(string(req.SystemMessage.Role))
		sb.WriteByte(':')
		sb.WriteString(messages.Text(*req.SystemMessage))
		sb.WriteByte('\n')
	}
	promptString := sb.String()

	modelID := fmt.Sprintf("%T", req.Model)
	llmString := modelID + "|" + hashToolsAndSettings(req.Tools, req.ModelSettings, req.ToolChoice)
	return promptString, llmString
}

// hashToolsAndSettings returns a stable, 16-char hex digest of the bound tool
// names, model settings, and tool choice. Tool names are collected in order;
// the names slice, the settings map, and the flattened tool choice are
// JSON-encoded together (encoding/json sorts map keys, so the encoding is
// deterministic) and sha256-hashed. An encode failure (which would only occur
// for non-JSON-representable settings values) yields an empty string, which
// collapses all such requests onto the same key rather than panicking — the
// cache is best-effort.
func hashToolsAndSettings(tools []any, settings map[string]any, toolChoice any) string {
	toolNames := make([]string, 0, len(tools))
	for _, t := range tools {
		if tool, ok := t.(coretools.Tool); ok {
			toolNames = append(toolNames, tool.Name())
		}
	}
	payload := map[string]any{
		"tools":       toolNames,
		"settings":    settings,
		"tool_choice": toolChoiceKey(toolChoice),
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])[:16]
}

// invokeModelStreaming is the streaming counterpart of invokeModel, used when
// an event sink is active (see buildModelNode's handler). It uses the same
// prepared bind (prepareModelCall): tools + tool_choice + model settings —
// including a ProviderStrategy's response_format kwargs when the model
// implements ModelSettingsBinder — are applied before model.Stream, so the
// effective per-call strategy reaches the streaming path exactly like the
// non-streaming one (Python binds once and the bound model serves both invoke
// and stream). For models without ModelSettingsBinder the ProviderStrategy
// degrades to the retained post-hoc detectStructuredOutput parse of the
// assembled message (language.StructuredCaller has no streaming form).
//
// The assembled message is returned in a ModelResponse exactly like
// invokeModel, so the rest of the model node (structured-output detection,
// AfterModel hooks, state update) is identical between the streaming and
// non-streaming paths — state semantics are unchanged (see the design spec's
// Step 2).
//
// mws is the agent's full middleware list. Any middleware implementing
// middleware.WrapModelStreamHook is composed (in WrapModelCall order,
// outermost-first) into a DeltaTransform that rewrites each text delta before
// it is emitted as a model_delta, and rewrites the assembled model_end text so
// the two stay consistent. When no middleware implements the hook, transform
// stays identity and this path is identical to the no-middleware behavior.
func invokeModelStreaming(ctx context.Context, req middleware.ModelRequest, prepared *preparedModelCall, sink *eventSink, mws []any, agentName string) (middleware.ModelResponse, error) {
	model := prepared.model

	invokeMessages := req.Messages
	if req.SystemMessage != nil {
		invokeMessages = append([]messages.Message{*req.SystemMessage}, req.Messages...)
	}

	stream, err := model.Stream(ctx, invokeMessages)
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	defer stream.Close()

	// Compose any WrapModelStreamHook middleware into a single DeltaTransform
	// applied to each text delta and to the assembled model_end text. Start
	// from identity so the no-middleware path is a no-op (behavior unchanged).
	// Composition runs at dispatch-construction time, before any delta flows,
	// so each returned transform closes over a stable inner reference (the
	// loop reassigns `transform` but each hook.TransformModelStream received
	// the prior value as its argument, capturing it by value inside the
	// returned closure).
	//
	// The loop iterates BACKWARD (i := len-1 .. 0) so mws[0] ends up wrapping
	// all the others — i.e. mws[0] is outermost at execution. This matches
	// WrapModelCall / WrapToolCall composition (see the model node's
	// WrapModelCallHook loop above and composeToolCallWrapper), so the same
	// middleware list orders identically under streaming and non-streaming
	// model-call wrapping.
	transform := func(text string) string { return text }
	for i := len(mws) - 1; i >= 0; i-- {
		if hook, ok := mws[i].(middleware.WrapModelStreamHook); ok {
			transform = hook.TransformModelStream(transform)
		}
	}

	projection := streamevents.NewChatModelStream()
	// Wrapper that both projects (into ChatModelStream) and emits a model_delta
	// for every v3 event we observe, whether produced natively by the model
	// (provider protocol events surfaced via callbacks — see
	// language.StreamEvents) or bridged from legacy message chunks below. The
	// projection receives the untransformed event (so the assembled message
	// text is the raw model output); the transform is applied only to the
	// delta text the consumer sees, and separately to the assembled model_end
	// text (see projection.Output() below).
	dispatch := func(ev streamevents.Event) {
		projection.Dispatch(ev)
		if ev.Delta != nil {
			deltaMap := messages.BlockToMap(ev.Delta)
			if deltaMap["type"] == "text-delta" {
				if text, ok := deltaMap["text"].(string); ok && text != "" {
					deltaMap["text"] = transform(text)
					ev.Delta = messages.ParseContentBlock(deltaMap)
				}
			}
		}
		// The legacy-bridge finish() also emits a content-block-finish event
		// carrying the fully-assembled text block in ev.Content["text"] (see
		// streamChunkBridge.finish). emitModelDelta passes the raw event
		// through as StreamEvent.Delta, so without redacting it here a consumer
		// reading StreamEvent.Delta.Content["text"] would see un-redacted text
		// — a leak for the PII use case (Task 3.2). The transform is idempotent
		// for redaction, so re-running it on the assembled text is safe. The
		// projection already saw the raw event above, so its assembled message
		// is unaffected (applyDeltaTransform handles model_end separately).
		if ev.Content != nil {
			contentMap := messages.BlockToMap(ev.Content)
			if contentMap["type"] == "text" {
				if text, ok := contentMap["text"].(string); ok && text != "" {
					contentMap["text"] = transform(text)
					ev.Content = messages.ParseContentBlock(contentMap)
				}
			}
		}
		sink.emitModelDelta(ev)
	}

	// Bridge legacy message-chunk streams (e.g. FakeChatModel, or any partner
	// model that hasn't adopted the v3 protocol callbacks) into v3 events,
	// emitting + projecting each. This mirrors language.chunkProtocolBridge
	// but emits through `dispatch` so each delta is surfaced live.
	bridge := &streamChunkBridge{dispatch: dispatch}
	for {
		chunk, ok, err := stream.Next(ctx)
		if err != nil {
			return middleware.ModelResponse{}, err
		}
		if !ok {
			break
		}
		bridge.push(chunk)
	}
	bridge.finish()

	// applyDeltaTransform rewrites the text carried on the assembled message
	// (both Content and any "text" content blocks) via the composed delta
	// transform, so the assembled model_end message — and the ModelResponse
	// returned downstream into state / AfterModel hooks — stays consistent
	// with the streamed (transformed) deltas. Non-text blocks (tool calls,
	// reasoning) are left untouched. Re-running the per-delta transform on the
	// assembled text is intentional: for the redaction use case it is
	// idempotent, and it keeps model_end == concat(deltas) for consumers.
	applyDeltaTransform := func(msg messages.Message) messages.Message {
		if msg.Content != "" {
			msg.Content = transform(msg.Content)
		}
		for i, block := range msg.ContentBlocks {
			if block == nil {
				continue
			}
			m := messages.BlockToMap(block)
			if m["type"] != "text" {
				continue
			}
			if text, ok := m["text"].(string); ok && text != "" {
				m["text"] = transform(text)
				msg.ContentBlocks[i] = messages.ParseContentBlock(m)
			}
		}
		return msg
	}

	if projection.Done() {
		out, err := projection.Output()
		if err != nil {
			return middleware.ModelResponse{}, err
		}
		out = applyDeltaTransform(out)
		// Same agent-name stamping as invokeModel (factory.py:1418-1419),
		// applied to the assembled message after the delta transform so
		// model_end and the returned ModelResponse stay consistent.
		resultMsgs := []messages.Message{out}
		applyAgentName(resultMsgs, agentName)
		sink.emitModelEnd(resultMsgs[0])
		return middleware.ModelResponse{Result: resultMsgs}, nil
	}
	// Stream ended without an explicit message-finish (provider quirk): fall
	// back to a non-streaming Invoke so the caller still gets a well-formed
	// message. This keeps state semantics identical to the Invoke path. The
	// transform is applied for parity with the streaming path so model_end /
	// state stay consistent with any deltas already emitted.
	result, err := model.Invoke(ctx, invokeMessages)
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	result = applyDeltaTransform(result)
	resultMsgs := []messages.Message{result}
	applyAgentName(resultMsgs, agentName)
	sink.emitModelEnd(resultMsgs[0])
	return middleware.ModelResponse{Result: resultMsgs}, nil
}

// newToolsNode builds the "tools" graph node backed by a
// langchain/tools.ToolNode, with any WrapToolCallHook middleware composed
// into its ToolCallWrapper. logger, when non-nil, emits a debug log per tool
// dispatch (mirroring Python's debug output around the tools node). s, when
// non-nil, is installed on the ToolNode so each ToolCallRequest.Store is
// populated for tools/wrappers that need it (mirroring Python's
// `create_agent(store=...)`).
//
// Commands a tool returns via Result.Artifact are consumed here (T12b):
// Python's ToolNode returns tool Commands straight through to langgraph
// (langgraph/prebuilt/tool_node.py:864-912, `_combine_tool_outputs`: a
// Command output stays a Command; non-command outputs become the node's
// `{"messages": [...]}` update), and langgraph applies each Command's Update
// and honors its goto — goto is neither an error nor ignored under
// create_agent's fixed routing. A Go graph node commits a single value, so
// the Commands are folded into that one return (mirroring
// applyModelNodeCommands): updates merge into the node's update in call order
// ("messages" concatenates, other keys last-write-wins) and every Goto
// destination is concatenated onto the returned *types.Command's Goto (empty
// when no tool jumped, falling back to the tools node's normal conditional
// edges). Command Resume and non-empty Graph are rejected: the Go tools node
// cannot forward them (Python's langgraph supports both; documented
// divergence).
func newToolsNode(toolList []coretools.Tool, mws []any, logger *slog.Logger, s store.Store) (graphpkg.NodeFunc, error) {
	nodeOpts := make([]agenttools.ToolNodeOption, 0, 2)
	if wrap := composeToolCallWrapper(mws, logger); wrap != nil {
		nodeOpts = append(nodeOpts, agenttools.WithToolCallWrapper(wrap))
	}
	if s != nil {
		nodeOpts = append(nodeOpts, agenttools.WithToolNodeStore(s))
	}
	toolNode, err := agenttools.NewToolNode(toolList, nodeOpts...)
	if err != nil {
		return nil, err
	}

	return func(rt runtime.Runtime, state map[string]any) (any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		if logger != nil {
			pending := 0
			for _, m := range msgs {
				if m.Role == messages.RoleAI && len(m.ToolCalls) > 0 {
					pending += len(m.ToolCalls)
				}
			}
			logger.Info("agents: tools node entry", slog.Int("pending_tool_calls", pending))
		}
		outcomes, err := toolNode.InvokeToolCallsFull(rt, unansweredToolCalls(msgs), state)
		if err != nil {
			return nil, err
		}
		results := make([]messages.Message, len(outcomes))
		var commands []*types.Command
		for i, outcome := range outcomes {
			results[i] = outcome.Message
			if outcome.Command != nil {
				commands = append(commands, outcome.Command)
			}
		}
		if len(commands) == 0 {
			return map[string]any{"messages": results}, nil
		}
		update := map[string]any{"messages": results}
		gotoDests, err := applyToolNodeCommands(update, commands)
		if err != nil {
			return nil, err
		}
		if len(gotoDests) > 0 {
			return &types.Command{Update: update, Goto: gotoDests}, nil
		}
		return update, nil
	}, nil
}

// applyToolNodeCommands merges the Commands returned by tools (via
// Result.Artifact, surfaced by langchain/tools.ToolNode.InvokeToolCallsFull)
// into the tools node's pending update, mirroring the sequential application
// of Python's ToolNode list-of-Commands return (langgraph/prebuilt/
// tool_node.py:864-912: the node's `{"messages": [...]}` update plus each
// Command, applied in order). The fold rules match applyModelNodeCommands:
//   - "messages" (an append-reducer channel) concatenates after the tool
//     messages, matching two sequential add_messages writes;
//   - every other key is overridden by the later Command's value;
//   - every Goto destination is collected, in order, onto the returned slice
//     (langgraph fans out to the union of the Commands' gotos);
//   - Command Resume and non-empty Graph are hard errors (unsupported in the
//     Go create_agent tools node).
func applyToolNodeCommands(update map[string]any, commands []*types.Command) ([]any, error) {
	var gotoDests []any
	for _, cmd := range commands {
		if cmd.Resume != nil {
			return nil, fmt.Errorf("agents: tool Command resume is not supported by create_agent's tools node")
		}
		if cmd.Graph != "" {
			return nil, fmt.Errorf("agents: tool Command graph %q is not supported by create_agent's tools node", cmd.Graph)
		}
		for k, v := range cmd.Update {
			if k == "messages" {
				if extra, ok := v.([]messages.Message); ok {
					base, _ := update["messages"].([]messages.Message)
					merged := make([]messages.Message, 0, len(base)+len(extra))
					merged = append(merged, base...)
					merged = append(merged, extra...)
					update["messages"] = merged
					continue
				}
			}
			update[k] = v
		}
		gotoDests = append(gotoDests, cmd.Goto...)
	}
	return gotoDests, nil
}

func composeToolCallWrapper(mws []any, logger *slog.Logger) agenttools.ToolCallWrapper {
	hooks := make([]WrapToolCallHook, 0)
	for _, mw := range mws {
		if hook, ok := mw.(WrapToolCallHook); ok {
			hooks = append(hooks, hook)
		}
	}
	// Always install a wrapper so that, when an event sink is active (i.e. the
	// run was started via Agent.StreamEvents / graph.InvokeStream), tool_start
	// / tool_end events can be emitted around each tool dispatch. The wrapper's
	// cost on the non-streaming path is a single context.Value lookup that
	// returns nil (sinkFromContext short-circuits to a no-op when no sink is
	// installed); when there is also no logger and no WrapToolCallHook
	// middleware, dispatch falls straight through to `next` with no further
	// work. This is the single chokepoint every tool call flows through, so
	// emitting here covers both the direct execute path and any
	// WrapToolCallHook middleware.

	return func(ctx context.Context, req agenttools.ToolCallRequest, next agenttools.ToolHandler) (messages.Message, error) {
		if logger != nil {
			logger.Info("agents: tool dispatch",
				slog.String("tool", req.ToolCall.Name),
				slog.String("call_id", req.ToolCall.ID))
		}
		// Streaming: when an event sink is active, emit tool_start/tool_end
		// around the (middleware-wrapped) tool dispatch. sinkFromContext returns
		// nil on the non-streaming path, so the emit calls are skipped with no
		// per-tool overhead beyond the lookup.
		sink := sinkFromContext(ctx)
		if sink != nil {
			sink.emitToolStart(req.ToolCall)
		}
		result, err := dispatchThroughMiddleware(ctx, req, next, hooks)
		if sink != nil {
			resultMap := toolResultMap(result)
			sink.emitToolEnd(req.ToolCall, resultMap)
		}
		return result, err
	}
}

// dispatchThroughMiddleware runs the tool call through the composed
// WrapToolCallHook middleware chain (outermost hook first, matching
// buildModelNode's WrapModelCall composition order), falling back to `next`
// when no middleware wraps tool calls.
func dispatchThroughMiddleware(
	ctx context.Context,
	req agenttools.ToolCallRequest,
	next agenttools.ToolHandler,
	hooks []WrapToolCallHook,
) (messages.Message, error) {
	handler := func(c context.Context, r middleware.ToolCallRequest) (messages.Message, error) {
		return next(c, agenttools.ToolCallRequest{
			ToolCall: fromMiddlewareToolCall(r.ToolCall),
			Tool:     r.Tool,
			State:    r.State,
			Store:    r.Store,
		})
	}
	for i := len(hooks) - 1; i >= 0; i-- {
		hook := hooks[i]
		inner := handler
		handler = func(c context.Context, r middleware.ToolCallRequest) (messages.Message, error) {
			return hook.WrapToolCall(c, r, inner)
		}
	}
	return handler(ctx, middleware.ToolCallRequest{
		ToolCall: toMiddlewareToolCall(req.ToolCall),
		Tool:     req.Tool,
		State:    req.State,
		Store:    req.Store,
	})
}

// toolResultMap derives a structured result map for a tool_end event from the
// returned ToolMessage. The ToolMessage carries the tool's textual content; we
// surface it under "content" so SSE-style callers can read it without parsing
// the message. Returns nil if msg is empty.
func toolResultMap(msg messages.Message) map[string]any {
	if msg.Content == "" && len(msg.ResponseMetadata) == 0 {
		return nil
	}
	out := map[string]any{}
	if msg.Content != "" {
		out["content"] = msg.Content
	}
	if status, ok := msg.ResponseMetadata["status"]; ok {
		out["status"] = status
	}
	return out
}

func toMiddlewareToolCall(tc messages.ToolCall) middleware.ToolCall {
	return middleware.ToolCall{Name: tc.Name, Args: tc.Args, ID: tc.ID}
}

func fromMiddlewareToolCall(tc middleware.ToolCall) messages.ToolCall {
	return messages.ToolCall{Name: tc.Name, Args: tc.Args, ID: tc.ID}
}
