package middleware

import (
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	"github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// SubAgentSpec is a declarative subagent specification, the Go counterpart of
// the SubAgent TypedDict in the Python reference (deepagents/middleware/
// subagents.py:121-235). Name and Description are required and must be unique
// across the agent's subagent list (see ValidateSubAgents); the model uses
// optional-pointer semantics instead of UP's NotRequired keys: a nil Model
// means "inherit", resolved either at compile time via CompileSubAgentWithModel
// or by erroring in CompileSubAgent.
type SubAgentSpec struct {
	// Name is the unique identifier the main agent uses to select this
	// subagent in the task tool (UP SubAgent.name, subagents.py:126-127).
	Name string
	// Description tells the main agent what this subagent does and when to
	// delegate to it (UP SubAgent.description, subagents.py:129-132).
	Description string
	// Prompt is the subagent's system prompt. A nil-equivalent empty prompt is
	// allowed, as upstream defaults system_prompt to "" (UP SubAgent.system_prompt,
	// subagents.py:216-224; create_sub_agent, subagents.py:667).
	Prompt string
	// Model overrides the main agent's model for this subagent (UP
	// SubAgent.model, subagents.py:142-145). Nil means inherit the parent's
	// model; CompileSubAgent treats nil as an error while
	// CompileSubAgentWithModel threads in a fallback.
	Model language.ChatModel
	// Tools are the subagent's own tools; unlike upstream there is no
	// default_tools inheritance in this port's compile step (UP
	// SubAgent.tools, subagents.py:134-137).
	Tools []coretools.Tool
	// Middleware is appended after the default subagent middleware stack (UP
	// SubAgent.middleware, subagents.py:147-150).
	Middleware []any
	// Backend, when non-nil, gives the subagent its own filesystem tool set
	// via FilesystemMiddleware with no permission rules (this port's analogue
	// of the per-subagent FilesystemMiddleware in graph.py:752-756).
	Backend backends.Backend
}

// CompiledSubAgent pairs a subagent's identity with an already-built agent,
// the Go counterpart of UP's CompiledSubAgent TypedDict (name, description,
// runnable — deepagents/middleware/subagents.py:269). Callers that
// already hold an *agents.Agent wrap it here instead of declaring a
// SubAgentSpec; SubAgentMiddleware (Task 8) dispatches both kinds.
type CompiledSubAgent struct {
	Name        string
	Description string
	Agent       *agents.Agent
}

// DefaultGeneralPurposeDescription is the general-purpose subagent's
// description, copied VERBATIM from DEFAULT_GENERAL_PURPOSE_DESCRIPTION in
// the Python reference (deepagents/middleware/subagents.py:564). Do not edit;
// parity is pinned by TestGeneralPurposeSubAgent.
const DefaultGeneralPurposeDescription = `General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent.`

// DefaultSubagentPrompt is the general-purpose subagent's system prompt,
// copied VERBATIM — including the blank line between paragraphs and the
// mid-sentence line break in the final sentence — from DEFAULT_SUBAGENT_PROMPT
// in the Python reference (deepagents/middleware/subagents.py:441-444). Do
// not edit; parity is pinned by TestGeneralPurposeSubAgent.
const DefaultSubagentPrompt = `In order to complete the objective that the user asks of you, you have access to a number of standard tools.

The calling agent only sees your final assistant message, not your intermediate work, tool results, or status tracking. Ensure your final
response contains the complete answer.`

// GeneralPurposeSubAgent returns the base general-purpose subagent spec for
// the given model, mirroring GENERAL_PURPOSE_SUBAGENT in the Python reference
// (deepagents/middleware/subagents.py:566-570): name "general-purpose" with
// the verbatim default description and prompt. The caller adds tools and any
// further middleware, as upstream documents ("caller adds model, tools,
// middleware" — here the model is the explicit parameter).
func GeneralPurposeSubAgent(model language.ChatModel) SubAgentSpec {
	return SubAgentSpec{
		Name:        "general-purpose",
		Description: DefaultGeneralPurposeDescription,
		Prompt:      DefaultSubagentPrompt,
		Model:       model,
	}
}

// CompileSubAgent builds a runnable agent from a SubAgentSpec. The spec's
// Model is REQUIRED here — a nil Model is an error, mirroring upstream's
// "SubAgent '<name>' must specify 'model'" (create_sub_agent,
// subagents.py:643-645). A spec meant to inherit the parent's model is
// compiled instead through CompileSubAgentWithModel, which SubAgentMiddleware
// (Task 8) uses to thread the parent's model in when spec.Model is nil.
//
// The compiled agent carries the spec's Prompt as its system prompt, its
// Tools, and a default middleware stack — the Tier-3 subset of upstream's
// per-subagent stack (graph.py:748-790): ModelRefusalMiddleware is installed
// unconditionally (graph.py:771; the general-purpose stack gets it too,
// graph.py:871), a FilesystemMiddleware on spec.Backend (no permission rules)
// is installed when Backend is non-nil, and spec.Middleware is appended after
// the defaults. Upstream's remaining stack pieces (summarization,
// PatchToolCallsMiddleware, profile/skills middleware, custom-middleware
// name-splicing rules) are out of scope for this tier and documented as a
// divergence. In particular, a FilesystemMiddleware the caller also lists in
// spec.Middleware next to a non-nil Backend makes CreateAgent fail loudly on
// the duplicate middleware, where upstream name-splices the custom entry over
// the matching default slot instead (_apply_custom_middleware,
// graph.py:792-796) — a deliberate divergence.
func CompileSubAgent(spec SubAgentSpec) (*CompiledSubAgent, error) {
	return compileSubAgent(spec, nil)
}

// CompileSubAgentWithModel is CompileSubAgent with a fallback model used when
// spec.Model is nil: SubAgentMiddleware calls it with the parent agent's
// model so an inherit-style spec compiles. The spec's own Model, when set,
// wins over the fallback; when both are nil the call errors with the
// upstream-style missing-model message.
func CompileSubAgentWithModel(spec SubAgentSpec, fallbackModel language.ChatModel) (*CompiledSubAgent, error) {
	return compileSubAgent(spec, fallbackModel)
}

func compileSubAgent(spec SubAgentSpec, fallbackModel language.ChatModel) (*CompiledSubAgent, error) {
	model := spec.Model
	if model == nil {
		model = fallbackModel
	}
	if model == nil {
		return nil, fmt.Errorf("SubAgent '%s' must specify 'model'", spec.Name)
	}

	// Default stack first, then the spec's custom middleware (UP order:
	// default subagent stack, with custom middleware appended after —
	// graph.py:748-790).
	mw := make([]any, 0, 2+len(spec.Middleware))
	if spec.Backend != nil {
		fs, err := NewFilesystemMiddleware(spec.Backend)
		if err != nil {
			return nil, fmt.Errorf("SubAgent '%s': filesystem middleware: %w", spec.Name, err)
		}
		mw = append(mw, fs)
	}
	mw = append(mw, agentsmw.NewModelRefusalMiddleware())
	mw = append(mw, spec.Middleware...)

	agent, err := agents.CreateAgent(
		model,
		spec.Tools,
		agents.WithAgentSystemPrompt(spec.Prompt),
		agents.WithAgentName(spec.Name),
		agents.WithAgentMiddleware(mw...),
	)
	if err != nil {
		return nil, fmt.Errorf("SubAgent '%s': create agent: %w", spec.Name, err)
	}
	return &CompiledSubAgent{Name: spec.Name, Description: spec.Description, Agent: agent}, nil
}

// ValidateSubAgents rejects a mixed SubAgentSpec/*CompiledSubAgent list whose
// names are empty or duplicated, mirroring _validate_unique_subagent_names
// (deepagents/middleware/subagents.py:370-381): the model selects a subagent
// purely by name, so a collision has no coherent meaning. The duplicate-name
// message is upstream's verbatim wording.
//
// Wording of the other errors is this port's (upstream has no literals here —
// its typing rejects foreign shapes statically, and an absent name would
// surface as a KeyError on spec["name"]): empty names and unrecognized /
// nil-valued entries report plain Go-style messages.
func ValidateSubAgents(specs []any) error {
	seen := make(map[string]struct{}, len(specs))
	for _, spec := range specs {
		var name string
		switch s := spec.(type) {
		case SubAgentSpec:
			name = s.Name
		case *CompiledSubAgent:
			if s == nil {
				return fmt.Errorf("invalid subagent spec: nil *CompiledSubAgent")
			}
			name = s.Name
		default:
			return fmt.Errorf("invalid subagent spec type %T: want SubAgentSpec or *CompiledSubAgent", spec)
		}
		if name == "" {
			return fmt.Errorf("subagent name must not be empty")
		}
		if _, dup := seen[name]; dup {
			return fmt.Errorf("Duplicate subagent name '%s'; each subagent must have a unique name.", name) //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
		}
		seen[name] = struct{}{}
	}
	return nil
}

// BuildTaskInput constructs the initial state handed to an isolated subagent
// for a delegated task, mirroring the isolated branch of upstream's state
// handoff (deepagents/middleware/subagents.py:929-930): every parent key
// outside ExcludedStateKeys ∪ PrivateStateKeys is shallow-copied as-is —
// notably "files", which is public state and passes through — and "messages"
// is excluded from the copy and rebuilt as a single human message carrying
// the task description. The parent map is never mutated.
func BuildTaskInput(parentState map[string]any, description string) map[string]any {
	excluded := make(map[string]struct{}, len(ExcludedStateKeys)+len(PrivateStateKeys))
	for _, key := range ExcludedStateKeys {
		excluded[key] = struct{}{}
	}
	for _, key := range PrivateStateKeys {
		excluded[key] = struct{}{}
	}
	input := make(map[string]any, len(parentState)+1)
	for key, value := range parentState {
		if _, drop := excluded[key]; drop {
			continue
		}
		input[key] = value
	}
	input["messages"] = []messages.Message{messages.Human(description)}
	return input
}
