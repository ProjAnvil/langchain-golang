package deepagents

import (
	"context"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	deepmw "github.com/projanvil/langchain-golang/deepagents/middleware"
	"github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/store"
)

// deepAgentRecursionLimit is the assembler's compiled-graph superstep budget,
// matching upstream's default deepagent recursion budget (plan-audit E13 /
// M13; UP graph.py). It is passed explicitly even though CreateAgent already
// defaults to 9999, so the assembler pins the value it was audited against.
const deepAgentRecursionLimit = 9999

// DeepAgentOptions configures CreateDeepAgent, the Go counterpart of the
// Python deepagents() assembler's parameters (UP graph.py, Tier-4 reduced
// set — spec §7). Zero values select the defaults documented per field; the
// functional-option constructors below are the conventional way to set them.
type DeepAgentOptions struct {
	// SystemPrompt is the main agent's whole system prompt. There is NO
	// built-in base prompt: upstream's BASE_AGENT_PROMPT is deprecated and
	// v0.7.38 injects none (audit I5), so an empty value means no system
	// prompt at all.
	SystemPrompt string
	// Tools are the main agent's own tools, bound alongside the filesystem
	// tools (FilesystemMiddleware) and the task tool (SubAgentMiddleware),
	// which the middleware stack provides. They are also threaded onto the
	// auto-added general-purpose subagent (UP graph.py:918).
	Tools []coretools.Tool
	// Middleware is appended AFTER the assembler's default stack (see
	// CreateDeepAgent for the order). It is also threaded onto the
	// auto-added general-purpose subagent.
	Middleware []any
	// SubAgents is the declarative subagent list: deepmw.SubAgentSpec values
	// (compiled by the assembler's SubAgentMiddleware) mixed with
	// *deepmw.CompiledSubAgent values (passed through). When no entry is
	// named "general-purpose", a default is auto-added at the front carrying
	// Tools and Middleware (UP graph.py:854+918).
	SubAgents []any
	// Backend is the virtual filesystem the filesystem tools operate on.
	// Nil defaults to backends.NewStateBackend() — files live in agent state.
	Backend backends.Backend
	// Permissions are the filesystem permission rules handed to
	// FilesystemMiddleware: deny rules gate the tools directly, interrupt
	// rules additionally derive HITL entries (see InterruptOn).
	Permissions []backends.PermissionRule
	// InterruptOn is merged over the permission-derived interrupt map per
	// tool — a user entry always wins for its tool (UP _merge_fs_interrupt_
	// on semantics). Non-empty union installs the HITL middleware.
	InterruptOn map[string]agentsmw.InterruptConfig
	// Checkpointer is passed through to agents.WithAgentCheckpointer,
	// enabling interrupt/resume (required for permission interrupts to be
	// resumable).
	Checkpointer checkpoint.Saver
	// Store is passed through to agents.WithAgentStore (the langgraph
	// semantic store surfaced to tools).
	Store store.Store
	// Summarizer overrides the default model-backed SummarizerFunc installed
	// by the SummarizationMiddleware. Nil selects the default (see
	// defaultSummarizer).
	Summarizer agentsmw.SummarizerFunc
	// ResponseFormat configures the main agent's structured output, passed
	// through to agents.WithAgentResponseFormat (UP graph.py:308
	// response_format). Accepted values are the platform's ToolStrategy/
	// ProviderStrategy/AutoStrategy (or pointers) and a raw JSON-schema
	// map[string]any / schema.Schema (see agents.WithAgentResponseFormat);
	// nil disables structured output. Note upstream also allows per-subagent
	// response formats — that half is recorded as skipped in the port spec
	// (§6) and is NOT covered by this option.
	ResponseFormat any
	// StateFields declares custom graph-state fields with their reducers,
	// passed through to agents.WithAgentStateFields (UP graph.py:309
	// state_schema). Without a declaration a custom key still flows through
	// Go's dynamic map state (LastValue semantics); registering a field is
	// how a custom key gets a merge reducer for conflict-safe parallel
	// writes. Upstream additionally forwards the schema to its subagent
	// middleware (graph.py:939); this port wires it to the main agent only
	// (see the "Subagent stack trims" entry in DIVERGENCES.md).
	StateFields []agents.StateField
	// ContextSchema declares the agent's runtime-context schema, passed
	// through to agents.WithAgentContextSchema (UP graph.py:310
	// context_schema; the spec §7 "Context passthrough" promise). Purely
	// declarative platform-side: it documents the expected fields but does
	// not gate agents.WithContextValues/ContextValue, which work without a
	// declared schema.
	ContextSchema []agents.ContextField
}

// DeepAgentOption applies one functional option to DeepAgentOptions.
type DeepAgentOption func(*DeepAgentOptions)

// WithSystemPrompt sets the main agent's system prompt (the whole prompt —
// no base prompt is prepended; audit I5).
func WithSystemPrompt(prompt string) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.SystemPrompt = prompt }
}

// WithTools sets the main agent's own tools (bound after the assembler's
// default stack; also threaded onto the auto-added general-purpose
// subagent).
func WithTools(tools ...coretools.Tool) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.Tools = append(o.Tools, tools...) }
}

// WithMiddleware appends caller middleware after the assembler's default
// stack.
func WithMiddleware(mw ...any) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.Middleware = append(o.Middleware, mw...) }
}

// WithSubAgents sets the declarative subagent list (SubAgentSpec and
// *CompiledSubAgent entries; see DeepAgentOptions.SubAgents).
func WithSubAgents(subagents []any) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.SubAgents = subagents }
}

// WithBackend sets the virtual filesystem backend (nil default:
// backends.NewStateBackend()).
func WithBackend(backend backends.Backend) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.Backend = backend }
}

// WithPermissions sets the filesystem permission rules.
func WithPermissions(rules ...backends.PermissionRule) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.Permissions = append(o.Permissions, rules...) }
}

// WithInterruptOn sets the user interrupt-on map merged over (and winning
// per tool over) the permission-derived entries.
func WithInterruptOn(interruptOn map[string]agentsmw.InterruptConfig) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.InterruptOn = interruptOn }
}

// WithCheckpointer installs a checkpoint.Saver, enabling resumable
// interrupts.
func WithCheckpointer(saver checkpoint.Saver) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.Checkpointer = saver }
}

// WithStore installs the cross-thread semantic store.
func WithStore(s store.Store) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.Store = s }
}

// WithSummarizer overrides the default model-backed summarizer.
func WithSummarizer(summarizer agentsmw.SummarizerFunc) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.Summarizer = summarizer }
}

// WithResponseFormat sets the main agent's structured output, passed through
// to agents.WithAgentResponseFormat (UP graph.py:308 response_format). See
// DeepAgentOptions.ResponseFormat for the accepted values and the
// per-subagent skip.
func WithResponseFormat(format any) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.ResponseFormat = format }
}

// WithStateFields appends custom graph-state fields (with their reducers) to
// the main agent, passed through to agents.WithAgentStateFields (UP
// graph.py:309 state_schema). See DeepAgentOptions.StateFields.
func WithStateFields(fields ...agents.StateField) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.StateFields = append(o.StateFields, fields...) }
}

// WithContextSchema declares the agent's runtime-context schema, passed
// through to agents.WithAgentContextSchema (UP graph.py:310 context_schema;
// spec §7's promised Context passthrough). See DeepAgentOptions.ContextSchema.
func WithContextSchema(fields ...agents.ContextField) DeepAgentOption {
	return func(o *DeepAgentOptions) { o.ContextSchema = fields }
}

// CreateDeepAgent assembles a Deep Agent over langchain/agents.CreateAgent,
// the Go counterpart of the Python deepagents() assembler (UP graph.py,
// Tier-4 reduced per spec §7). Every capability is middleware on the
// existing hook surface; the returned *agents.Agent is an ordinary agent.
//
// The default middleware stack, in order (UP assembly order, corrected):
//
//  1. FilesystemMiddleware over opts.Backend (default NewStateBackend()) with
//     opts.Permissions — provides the seven filesystem tools, the files
//     state schema (StateBackend), permission evaluation, the state-view
//     injection, and the same-batch parallel-mutation guard.
//  2. SubAgentMiddleware over model and the assembled subagent list —
//     provides the task tool; auto-adds the general-purpose subagent
//     (carrying opts.Tools and opts.Middleware, UP graph.py:918) when no
//     caller entry claims the name.
//  3. SummarizationMiddleware with opts.Summarizer, or the model-backed
//     default (one summarize call; see defaultSummarizer).
//  4. ModelRefusalMiddleware.
//  5. The interrupt HITL middleware — only when the union of
//     permission-derived entries and opts.InterruptOn is non-empty (user
//     entries win per tool).
//  6. opts.Middleware.
//
// CreateAgent then receives opts.Tools as the caller tools (the filesystem
// and task tools arrive via their middleware's ProvidedTools), the system
// prompt only when non-empty (no default prompt — audit I5), recursion limit
// 9999 (E13), and the checkpointer/store, response_format, state_schema,
// and context_schema passthroughs (UP graph.py:308-310). No execute tool is
// registered (divergence D5).
func CreateDeepAgent(model language.ChatModel, opts ...DeepAgentOption) (*agents.Agent, error) {
	options := DeepAgentOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	backend := options.Backend
	if backend == nil {
		backend = backends.NewStateBackend()
	}
	fsMw, err := deepmw.NewFilesystemMiddleware(backend, options.Permissions...)
	if err != nil {
		return nil, fmt.Errorf("deepagents: filesystem middleware: %w", err)
	}

	subMw, err := deepmw.NewSubAgentMiddleware(model, assembleSubAgents(model, &options))
	if err != nil {
		return nil, fmt.Errorf("deepagents: subagent middleware: %w", err)
	}

	summarizer := options.Summarizer
	if summarizer == nil {
		summarizer = defaultSummarizer(model)
	}

	stack := []any{
		fsMw,
		subMw,
		agentsmw.NewSummarizationMiddleware(summarizer),
		agentsmw.NewModelRefusalMiddleware(),
	}
	if interruptOn := mergedInterruptConfigs(options.Permissions, options.InterruptOn); len(interruptOn) > 0 {
		stack = append(stack, agentsmw.NewInterruptHumanInTheLoopMiddleware(interruptOn))
	}
	stack = append(stack, options.Middleware...)

	agentOpts := []agents.AgentOption{
		agents.WithAgentMiddleware(stack...),
		agents.WithAgentRecursionLimit(deepAgentRecursionLimit),
	}
	if options.SystemPrompt != "" {
		agentOpts = append(agentOpts, agents.WithAgentSystemPrompt(options.SystemPrompt))
	}
	if options.Checkpointer != nil {
		agentOpts = append(agentOpts, agents.WithAgentCheckpointer(options.Checkpointer))
	}
	if options.Store != nil {
		agentOpts = append(agentOpts, agents.WithAgentStore(options.Store))
	}
	if options.ResponseFormat != nil {
		agentOpts = append(agentOpts, agents.WithAgentResponseFormat(options.ResponseFormat))
	}
	if len(options.StateFields) > 0 {
		agentOpts = append(agentOpts, agents.WithAgentStateFields(options.StateFields...))
	}
	if len(options.ContextSchema) > 0 {
		agentOpts = append(agentOpts, agents.WithAgentContextSchema(options.ContextSchema...))
	}

	return agents.CreateAgent(model, options.Tools, agentOpts...)
}

// assembleSubAgents prepends the default general-purpose subagent — carrying
// the parent's tools and middleware — when no caller entry claims the name,
// mirroring UP graph.py:854+918 (the default spec is inserted at index 0
// with tools=the main agent's tools and the inheritable middleware). A
// caller-supplied general-purpose (either entry kind) is left untouched: an
// explicit spec is how callers override the default.
//
// Scope note (divergence, DIVERGENCES.md): upstream's default subagent also
// carries its own FilesystemMiddleware over the shared backend with the
// parent's permissions, summarization, and PatchToolCalls; this port's
// compiled default carries ONLY the parent's caller tools/middleware over
// the standard subagent stack (ModelRefusalMiddleware) — the Tier-3 subagent
// stack trim.
func assembleSubAgents(model language.ChatModel, opts *DeepAgentOptions) []any {
	for _, spec := range opts.SubAgents {
		switch s := spec.(type) {
		case deepmw.SubAgentSpec:
			if s.Name == deepmw.GeneralPurposeName {
				return opts.SubAgents
			}
		case *deepmw.CompiledSubAgent:
			if s != nil && s.Name == deepmw.GeneralPurposeName {
				return opts.SubAgents
			}
		}
	}
	gp := deepmw.GeneralPurposeSubAgent(model)
	gp.Tools = opts.Tools
	gp.Middleware = opts.Middleware
	return append([]any{gp}, opts.SubAgents...)
}

// mergedInterruptConfigs unions the permission-derived interrupt map with the
// user-supplied entries, the user's entry winning for its tool (UP
// _merge_fs_interrupt_on: interrupt_on entries override the derived ones).
// Both inputs may be empty; an empty result installs no HITL middleware.
func mergedInterruptConfigs(perms []backends.PermissionRule, user map[string]agentsmw.InterruptConfig) map[string]agentsmw.InterruptConfig {
	merged := deepmw.InterruptConfigsFromPermissions(perms)
	if len(user) == 0 {
		return merged
	}
	if merged == nil {
		merged = make(map[string]agentsmw.InterruptConfig, len(user))
	}
	for tool, config := range user {
		merged[tool] = config
	}
	return merged
}

// defaultSummarizer builds the SummarizerFunc installed when
// DeepAgentOptions.Summarizer is nil (plan-audit I7 / E9): it issues ONE
// model call — a human message carrying the summary prompt followed by the
// to-summarize transcript as "role: text" lines — and returns the model's
// reply text, mirroring upstream's create_summarization_middleware(model,
// backend) single-call default (without the backend offload superset,
// divergence D1). A model error propagates.
func defaultSummarizer(model language.ChatModel) agentsmw.SummarizerFunc {
	return func(prompt string, msgs []messages.Message) (string, error) {
		var transcript strings.Builder
		for _, msg := range msgs {
			if text := messages.Text(msg); text != "" {
				fmt.Fprintf(&transcript, "\n%s: %s", msg.Role, text)
			}
		}
		out, err := model.Invoke(context.Background(), []messages.Message{
			messages.Human(prompt + transcript.String()),
		})
		if err != nil {
			return "", fmt.Errorf("deepagents: summarize: %w", err)
		}
		return messages.Text(out), nil
	}
}
