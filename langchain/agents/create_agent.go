package agents

// CreateAgent builds a scoped Go equivalent of Python's
// `langchain.agents.create_agent(...)`: a model node <-> tools node loop
// wired on top of `agentruntime/graph`, with middleware hooks composed around
// the model call and each tool call.
//
// Scope note (see migration_plan/core-v1-migration-todo.md P5
// `langchain/agents` for the authoritative list): this deliberately does not
// port subagent transformer behavior (this also requires a middleware-facing
// streaming layer that doesn't exist in this port yet — see
// migration_plan/core-v1-migration-todo.md), or Command/Send returned
// directly from tools (langgraph's ToolNode `Command` support is out of
// scope; see langchain/tools/tool_node.go). Interrupts ARE wired through
// CreateAgent: every model-loop hook (BeforeModelHook/BeforeModelCommandHook/
// AfterModelHook/WrapModelCallHook/WrapToolCallHook), not just
// BeforeAgentHook/AfterAgentHook, receives a context.Context a middleware can
// pass to graphpkg.Interrupt to pause the run (see
// create_agent_test.go's TestCreateAgent_InterruptThroughModelHook for a
// round-trip example using WithAgentCheckpointer + Agent.Graph.
// InvokeWithOptions to resume). Structured output (`ToolStrategy`/
// `ProviderStrategy`) IS wired via WithAgentResponseFormat — see its doc
// comment for exact scope (ProviderStrategy's provider-native model-kwargs
// binding is best-effort only). `BeforeAgent`/`AfterAgent` hooks ARE wired:
// when at least one middleware implements BeforeAgentHook/AfterAgentHook,
// CreateAgent adds dedicated "before_agent"/"after_agent" nodes around the
// model<->tools loop (mirroring Python's `before_agent`/`after_agent` running
// once per run, not once per model call); every "end" routing decision
// (normal completion, a jump_to "end", or a structured-output match) is
// redirected through "after_agent" first when present.
//
// Middleware hook discovery: unlike Python's `AgentMiddleware` base class
// (which defines every hook as a no-op an implementation can selectively
// override), Go has no shared base with overridable defaults. CreateAgent
// instead uses type assertions against the *Hook interfaces below, so a
// middleware value need only implement the hooks it cares about.
//
// "jump_to" convention: a BeforeModel/AfterModel hook can short-circuit
// normal routing by setting update["jump_to"] to "model", "tools", or "end"
// (mirroring Python's `AgentState.jump_to` field). CreateAgent consumes this
// key out of the update before merging it into graph state (it is never
// itself persisted).
//
// BeforeModel "messages" scope note: a BeforeModelHook's returned
// update["messages"], if present, reshapes only the *local* view of the
// conversation used to build this model call (e.g. `SummarizationMiddleware`
// collapsing older messages into a summary); it is intentionally NOT
// persisted into the graph's committed state, since agentruntime/channels'
// MessagesReducer (unlike Python's `add_messages`) has no
// RemoveMessage/REMOVE_ALL_MESSAGES support to express "replace history."
// AfterModel "messages" updates are additive (new tool/AI messages) and are
// persisted normally.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/projanvil/langchain-golang/core/caches"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/prompts"
	"github.com/projanvil/langchain-golang/core/stores"
	"github.com/projanvil/langchain-golang/core/streamevents"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/channels"
	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langchain/internal/agentruntime/graph"
	agenttools "github.com/projanvil/langchain-golang/langchain/tools"
)

// Node names used by the compiled graph, mirroring Python's "model"/"tools"
// node names in `create_agent`. BeforeAgentNodeName/AfterAgentNodeName are
// only added to the graph when at least one middleware implements the
// corresponding hook (see WithAgentMiddleware).
const (
	ModelNodeName       = "model"
	ToolsNodeName       = "tools"
	BeforeAgentNodeName = "before_agent"
	AfterAgentNodeName  = "after_agent"
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

// WrapModelCallHook lets middleware intercept the model call itself,
// mirroring Python's `AgentMiddleware.wrap_model_call`. It receives a
// context.Context for the same reason as BeforeModelHook.
type WrapModelCallHook interface {
	WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error)
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
	// on each Invoke, since agentruntime/graph has no native run-metadata
	// injection point.
	Name string
	// Debug toggles verbose structured logging of the graph execution path
	// (each superstep, node entry, tool dispatch, model call), mirroring
	// Python's `create_agent(debug=True)`. Off by default.
	Debug bool
	// ResponseFormat configures structured output, mirroring Python's
	// `create_agent(response_format=...)`. Accepted values are ToolStrategy
	// (or *ToolStrategy) and ProviderStrategy (or *ProviderStrategy); see
	// WithAgentResponseFormat and the package doc comment for scope.
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
	// Store is the agent's cross-thread KV store, mirroring Python's
	// `create_agent(store=...)`. When non-nil, it is injected into every
	// ToolCallRequest.Store (see middleware.ToolCallRequest).
	Store stores.BaseStore[any]
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
// context on each Invoke, since agentruntime/graph exposes no native
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

// WithAgentStore installs a cross-thread KV store, mirroring Python's
// `create_agent(store=...)`. The store is injected into each tool call via
// middleware.ToolCallRequest.Store (Go has no Python-style InjectedStore
// annotation; tools read it explicitly).
func WithAgentStore(store stores.BaseStore[any]) AgentOption {
	return func(o *AgentOptions) { o.Store = store }
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
// message text, and the model type plus a stable hash of the bound tools and
// model settings).
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
// *ToolStrategy, ProviderStrategy, or *ProviderStrategy; CreateAgent returns
// an error for any other type.
//
// ToolStrategy is fully wired into the model loop: each of its SchemaSpecs is
// bound to the model as an extra callable tool, and a matching tool call in
// the model's response is intercepted (never reaching the tools node),
// parsed, and surfaces via the final state's "structured_response" key (see
// Agent.InvokeWithState), ending the run. Multiple structured tool calls in
// one response is an error (MultipleStructuredOutputsError), mirroring
// Python.
//
// ProviderStrategy is parsed on a best-effort basis: CreateAgent attempts to
// JSON-decode the model's final text response against the schema whenever no
// tool calls are present. Unlike Python, it does NOT pass model kwargs (e.g.
// a provider's native `response_format`) to the underlying language.ChatModel
// to request schema-constrained output, since that call shape isn't exposed
// generically by the language.ChatModel interface yet; the model must be
// separately configured (or reliably prompted) to emit matching JSON.
func WithAgentResponseFormat(format any) AgentOption {
	return func(o *AgentOptions) { o.ResponseFormat = format }
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
// tracing tag, mirroring Python's `lc_agent_name`. agentruntime/graph has no
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
	if model == nil {
		return nil, fmt.Errorf("agents: model is required")
	}

	options := AgentOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	toolStrategy, providerStrategy, err := resolveResponseFormat(options.ResponseFormat)
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
	// goes: agentruntime.END directly, or through a dedicated "after_agent" node
	// first when at least one AfterAgentHook is configured (see the package
	// doc comment).
	finalNode := agentruntime.END
	hasAfterAgent := hasHook[AfterAgentHook](options.Middleware)
	if hasAfterAgent {
		finalNode = AfterAgentNodeName
	}

	logger := debugLogger(options.Debug)

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
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
	g.AddNode(ModelNodeName, buildModelNode(model, modelTools, systemPromptResolver(options), logger, options.Middleware, structuredBindings, toolStrategy, providerStrategy, finalNode, options.Cache))

	entryNode := ModelNodeName
	if hasHook[BeforeAgentHook](options.Middleware) {
		entryNode = BeforeAgentNodeName
		g.AddNode(BeforeAgentNodeName, buildBeforeAgentNode(options.Middleware, logger))
		g.AddEdge(BeforeAgentNodeName, ModelNodeName)
	}
	g.AddEdge(agentruntime.START, entryNode)

	if hasAfterAgent {
		g.AddNode(AfterAgentNodeName, buildAfterAgentNode(options.Middleware, logger))
		g.AddEdge(AfterAgentNodeName, agentruntime.END)
	}

	if len(toolList) > 0 {
		toolNode, err := newToolsNode(toolList, options.Middleware, logger, options.Store)
		if err != nil {
			return nil, err
		}
		g.AddNode(ToolsNodeName, toolNode)
		g.AddEdge(ToolsNodeName, ModelNodeName)
		g.AddConditionalEdges(ModelNodeName, buildRouteAfterModel(finalNode))
	} else {
		g.AddEdge(ModelNodeName, finalNode)
	}

	compileOpts := make([]graphpkg.CompileOption, 0, 4)
	if options.Checkpointer != nil {
		compileOpts = append(compileOpts, graphpkg.WithCheckpointer(options.Checkpointer))
	}
	if options.RecursionLimit > 0 {
		compileOpts = append(compileOpts, graphpkg.WithRecursionLimit(options.RecursionLimit))
	}
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

// withRunTags returns ctx annotated with this Agent's run-name tag, so
// middleware/nodes can read it via NameFromContext (the lc_agent_name
// equivalent). agentruntime/graph has no native run-metadata injection point,
// so this context value is the surfaced channel.
func (a *Agent) withRunTags(ctx context.Context) context.Context {
	if a.Name == "" {
		return ctx
	}
	return context.WithValue(ctx, runNameCtxKey{}, a.Name)
}

func buildRouteAfterModel(finalNode string) graphpkg.ConditionalEdge {
	return func(_ context.Context, state map[string]any) ([]any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		if agenttools.HasPendingToolCalls(msgs) {
			return graphpkg.To(ToolsNodeName), nil
		}
		return graphpkg.To(finalNode), nil
	}
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

// buildBeforeAgentNode returns the "before_agent" graph node, running every
// BeforeAgentHook middleware once (in order) before the model<->tools loop
// starts. logger, when non-nil, emits a debug log at node entry.
func buildBeforeAgentNode(mws []any, logger *slog.Logger) graphpkg.NodeFunc {
	return func(ctx context.Context, rawState map[string]any) (any, error) {
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
			hookUpdate, err := hook.BeforeAgent(ctx, state)
			if err != nil {
				return nil, err
			}
			for k, v := range hookUpdate {
				state[k] = v
				update[k] = v
			}
		}
		return update, nil
	}
}

// buildAfterAgentNode returns the "after_agent" graph node, running every
// AfterAgentHook middleware once (in order) after the model<->tools loop
// ends. Matching AfterAgentHook's signature, it does not produce a state
// update. logger, when non-nil, emits a debug log at node entry.
func buildAfterAgentNode(mws []any, logger *slog.Logger) graphpkg.NodeFunc {
	return func(ctx context.Context, state map[string]any) (any, error) {
		if logger != nil {
			logger.Info("agents: after_agent node entry")
		}
		for _, mw := range mws {
			hook, ok := mw.(AfterAgentHook)
			if !ok {
				continue
			}
			if err := hook.AfterAgent(ctx, state); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}
}

// buildModelNode returns the graph node function driving one model call:
// BeforeModel hooks, the (middleware-wrapped) model invocation, then
// AfterModel hooks.
//
// systemPrompt resolves the system-prompt string for this call: a literal when
// WithAgentSystemPrompt is used, or a core/prompts render (with build-time +
// per-Invoke variables) when WithAgentSystemPromptTemplate is used.
//
// logger, when non-nil, emits verbose debug logs (see WithAgentDebug) for node
// entry, the model call, and structured-output detection.
func buildModelNode(
	model language.ChatModel,
	toolList []coretools.Tool,
	systemPrompt func(ctx context.Context) string,
	logger *slog.Logger,
	mws []any,
	structuredBindings map[string]OutputToolBinding,
	toolStrategy *ToolStrategy,
	providerStrategy *ProviderStrategy,
	finalNode string,
	cache caches.Cache,
) graphpkg.NodeFunc {
	toolsAny := toolsToAny(toolList)

	return func(ctx context.Context, rawState map[string]any) (any, error) {
		state := cloneMapState(rawState)

		localMessages, _ := state["messages"].([]messages.Message)
		if logger != nil {
			logger.Info("agents: model node entry",
				slog.Int("messages", len(localMessages)),
				slog.Int("tools", len(toolsAny)))
		}
		for _, mw := range mws {
			if hook, ok := mw.(BeforeModelCommandHook); ok {
				cmd, err := hook.BeforeModel(ctx, state)
				if err != nil {
					return nil, err
				}
				if cmd != nil {
					return &agentruntime.Command{
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
			update, err := hook.BeforeModel(ctx, state)
			if err != nil {
				return nil, err
			}
			if update == nil {
				continue
			}
			if jumpTo, ok := popJumpTo(update); ok {
				return &agentruntime.Command{Update: update, Goto: graphpkg.To(resolveJumpTarget(jumpTo, finalNode))}, nil
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
			}
		}

		resolvedPrompt := systemPrompt(ctx)
		req, err := middleware.NewModelRequest(middleware.ModelRequest{
			Model:         model,
			Messages:      localMessages,
			Tools:         toolsAny,
			SystemPrompt:  resolvedPrompt,
			State:         state,
			ModelSettings: providerStrategyModelSettings(providerStrategy),
		})
		if err != nil {
			return nil, err
		}

		handler := func(c context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
			if logger != nil {
				logger.Info("agents: model call",
					slog.Int("messages", len(r.Messages)),
					slog.Bool("has_system_prompt", r.SystemMessage != nil))
			}
			// Streaming path: when an event sink is active (i.e. the run was
			// started via Agent.StreamEvents / graph.InvokeStream), drive the
			// model through Stream, emit a model_delta per chunk, and assemble
			// the final message via core/streamevents.ChatModelStream before
			// emitting model_end. When no sink is active, the non-streaming
			// Invoke path is used with zero added overhead (see invokeModel).
			if sink := sinkFromContext(c); sink != nil {
				return invokeModelStreaming(c, r, sink, mws)
			}
			return invokeModel(c, r)
		}
		for i := len(mws) - 1; i >= 0; i-- {
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
		cacheEnabled := cache != nil && sinkFromContext(ctx) == nil
		if cacheEnabled {
			promptString, llmString := cacheKey(req)
			if gens, ok, lerr := cache.Lookup(ctx, promptString, llmString); lerr == nil && ok && len(gens) > 0 {
				cached := make([]messages.Message, 0, len(gens))
				for _, g := range gens {
					cached = append(cached, messages.AI(g.Text))
				}
				resp = middleware.ModelResponse{Result: cached}
				cacheHit = true
				if logger != nil {
					logger.Info("agents: model cache hit",
						slog.Int("generations", len(gens)))
				}
			}
		}
		if !cacheHit {
			resp, err = handler(ctx, req)
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
				_ = cache.Update(ctx, promptString, llmString, generations)
			}
		}
		newMessages := append([]messages.Message(nil), resp.Result...)
		if logger != nil {
			logger.Info("agents: model response",
				slog.Int("new_messages", len(newMessages)))
		}

		// Structured output detection happens before AfterModel hooks (see
		// WithAgentResponseFormat's doc comment): a matched structured tool
		// call, or a ProviderStrategy JSON parse, ends the run immediately
		// without executing any tools or running AfterModel hooks.
		if cmd, handled, err := detectStructuredOutput(newMessages, structuredBindings, toolStrategy, providerStrategy, finalNode); err != nil {
			return nil, err
		} else if handled {
			return cmd, nil
		}

		update := map[string]any{"messages": newMessages}

		afterState := cloneMapState(state)
		afterState["messages"] = append(append([]messages.Message(nil), localMessages...), newMessages...)

		gotoOverride := ""
		for _, mw := range mws {
			hook, ok := mw.(AfterModelHook)
			if !ok {
				continue
			}
			hookUpdate, err := hook.AfterModel(ctx, afterState)
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
			return &agentruntime.Command{Update: update, Goto: graphpkg.To(gotoOverride)}, nil
		}
		return update, nil
	}
}

// resolveResponseFormat validates and unpacks an AgentOptions.ResponseFormat
// value into (at most) one of a ToolStrategy or ProviderStrategy. See
// WithAgentResponseFormat's doc comment for accepted types and behavior.
func resolveResponseFormat(format any) (*ToolStrategy, *ProviderStrategy, error) {
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
	default:
		return nil, nil, fmt.Errorf("agents: unsupported ResponseFormat type %T (expected ToolStrategy or ProviderStrategy)", format)
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
// so WrapModelCall middleware or a provider-aware language.ChatModel can
// observe the caller's intent, even though invokeModel itself does not act
// on them (see WithAgentResponseFormat's doc comment).
func providerStrategyModelSettings(providerStrategy *ProviderStrategy) map[string]any {
	if providerStrategy == nil {
		return nil
	}
	return providerStrategy.ToModelKwargs()
}

// detectStructuredOutput inspects the model's newMessages for a
// ResponseFormat match: a tool call into structuredBindings (ToolStrategy),
// or — absent any tool calls — a ProviderStrategy JSON-decodable text
// response. A match returns a terminal *agentruntime.Command (handled=true)
// carrying the parsed value under state key "structured_response", ending
// the run without visiting the tools node or running AfterModel hooks.
func detectStructuredOutput(
	newMessages []messages.Message,
	structuredBindings map[string]OutputToolBinding,
	toolStrategy *ToolStrategy,
	providerStrategy *ProviderStrategy,
	finalNode string,
) (*agentruntime.Command, bool, error) {
	if len(newMessages) == 0 {
		return nil, false, nil
	}
	last := newMessages[len(newMessages)-1]
	if last.Role != messages.RoleAI {
		return nil, false, nil
	}

	if len(structuredBindings) > 0 {
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
			return nil, false, NewMultipleStructuredOutputsError(names, last)
		}
		if len(matched) == 1 {
			call := matched[0]
			binding := structuredBindings[call.Name]
			parsed, err := binding.Parse(call.Args)
			if err != nil {
				return nil, false, NewStructuredOutputValidationError(call.Name, err, last)
			}
			content := ""
			if toolStrategy != nil {
				content = toolStrategy.ToolMessageContent
			}
			if content == "" {
				content = fmt.Sprintf("Returned structured response via %s.", call.Name)
			}
			toolMsg := messages.Tool(call.ID, content)
			toolMsg.Name = call.Name
			updatedMessages := append(append([]messages.Message(nil), newMessages...), toolMsg)
			return &agentruntime.Command{
				Update: map[string]any{
					"messages":            updatedMessages,
					"structured_response": parsed,
				},
				Goto: graphpkg.To(finalNode),
			}, true, nil
		}
	}

	if providerStrategy != nil && len(last.ToolCalls) == 0 {
		binding := ProviderStrategyBindingFromSchemaSpec(providerStrategy.SchemaSpec)
		parsed, err := binding.Parse(last)
		if err != nil {
			return nil, false, err
		}
		return &agentruntime.Command{
			Update: map[string]any{
				"messages":            newMessages,
				"structured_response": parsed,
			},
			Goto: graphpkg.To(finalNode),
		}, true, nil
	}

	return nil, false, nil
}

// invokeModel runs the actual chat model call for a (possibly
// middleware-overridden) ModelRequest, binding req.Tools if present.
func invokeModel(ctx context.Context, req middleware.ModelRequest) (middleware.ModelResponse, error) {
	model, ok := req.Model.(language.ChatModel)
	if !ok || model == nil {
		return middleware.ModelResponse{}, fmt.Errorf("agents: ModelRequest.Model must be a language.ChatModel, got %T", req.Model)
	}
	if len(req.Tools) > 0 {
		boundTools, err := toolsFromAny(req.Tools)
		if err != nil {
			return middleware.ModelResponse{}, err
		}
		bound, err := model.BindTools(boundTools)
		if err != nil {
			return middleware.ModelResponse{}, err
		}
		model = bound
	}

	invokeMessages := req.Messages
	if req.SystemMessage != nil {
		invokeMessages = append([]messages.Message{*req.SystemMessage}, req.Messages...)
	}
	result, err := model.Invoke(ctx, invokeMessages)
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	return middleware.ModelResponse{Result: []messages.Message{result}}, nil
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
// langchain_core's BaseCache contract. promptString is the role-tagged,
// rendered message text (each message's role + messages.Text of its content);
// llmString identifies the model and its configuration as the model type plus a
// stable hash of the bound tool names and ModelSettings (so two requests that
// differ only in tools or settings do not collide).
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
	llmString := modelID + "|" + hashToolsAndSettings(req.Tools, req.ModelSettings)
	return promptString, llmString
}

// hashToolsAndSettings returns a stable, 16-char hex digest of the bound tool
// names and model settings. Tool names are collected in order; the names slice
// and the settings map are JSON-encoded together (encoding/json sorts map keys,
// so the encoding is deterministic) and sha256-hashed. An encode failure (which
// would only occur for non-JSON-representable settings values) yields an empty
// string, which collapses all such requests onto the same key rather than
// panicking — the cache is best-effort.
func hashToolsAndSettings(tools []any, settings map[string]any) string {
	toolNames := make([]string, 0, len(tools))
	for _, t := range tools {
		if tool, ok := t.(coretools.Tool); ok {
			toolNames = append(toolNames, tool.Name())
		}
	}
	payload := map[string]any{
		"tools":    toolNames,
		"settings": settings,
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])[:16]
}

// invokeModelStreaming is the streaming counterpart of invokeModel, used when
// an event sink is active (see buildModelNode's handler). It binds tools the
// same way, then drives model.Stream and projects the chunks through
// core/streamevents.ChatModelStream — emitting one model_delta event per v3
// protocol event and a single model_end with the assembled message at the end.
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
func invokeModelStreaming(ctx context.Context, req middleware.ModelRequest, sink *eventSink, mws []any) (middleware.ModelResponse, error) {
	model, ok := req.Model.(language.ChatModel)
	if !ok || model == nil {
		return middleware.ModelResponse{}, fmt.Errorf("agents: ModelRequest.Model must be a language.ChatModel, got %T", req.Model)
	}
	if len(req.Tools) > 0 {
		boundTools, err := toolsFromAny(req.Tools)
		if err != nil {
			return middleware.ModelResponse{}, err
		}
		bound, err := model.BindTools(boundTools)
		if err != nil {
			return middleware.ModelResponse{}, err
		}
		model = bound
	}

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
		if ev.Delta != nil && ev.Delta["type"] == "text-delta" {
			if text, ok := ev.Delta["text"].(string); ok && text != "" {
				ev.Delta["text"] = transform(text)
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
		if ev.Content != nil && ev.Content["type"] == "text" {
			if text, ok := ev.Content["text"].(string); ok && text != "" {
				ev.Content["text"] = transform(text)
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
			if block == nil || block["type"] != "text" {
				continue
			}
			if text, ok := block["text"].(string); ok && text != "" {
				msg.ContentBlocks[i]["text"] = transform(text)
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
		sink.emitModelEnd(out)
		return middleware.ModelResponse{Result: []messages.Message{out}}, nil
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
	sink.emitModelEnd(result)
	return middleware.ModelResponse{Result: []messages.Message{result}}, nil
}

// newToolsNode builds the "tools" graph node backed by a
// langchain/tools.ToolNode, with any WrapToolCallHook middleware composed
// into its ToolCallWrapper. logger, when non-nil, emits a debug log per tool
// dispatch (mirroring Python's debug output around the tools node). store,
// when non-nil, is installed on the ToolNode so each ToolCallRequest.Store is
// populated for tools/wrappers that need it (mirroring Python's
// `create_agent(store=...)`).
func newToolsNode(toolList []coretools.Tool, mws []any, logger *slog.Logger, store stores.BaseStore[any]) (graphpkg.NodeFunc, error) {
	nodeOpts := make([]agenttools.ToolNodeOption, 0, 2)
	if wrap := composeToolCallWrapper(mws, logger); wrap != nil {
		nodeOpts = append(nodeOpts, agenttools.WithToolCallWrapper(wrap))
	}
	if store != nil {
		nodeOpts = append(nodeOpts, agenttools.WithToolNodeStore(store))
	}
	toolNode, err := agenttools.NewToolNode(toolList, nodeOpts...)
	if err != nil {
		return nil, err
	}

	return func(ctx context.Context, state map[string]any) (any, error) {
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
		results, err := toolNode.Invoke(ctx, msgs, state)
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": results}, nil
	}, nil
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
