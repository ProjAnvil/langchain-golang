package middleware

// Task 8: the `task` tool and SubAgentMiddleware, the port of
// deepagents/middleware/subagents.py's SubAgentMiddleware core (schema
// subagents.py:498-557, task/ataisk flow subagents.py:898-1000, state
// writeback subagents.py:836-876). The main agent delegates a task by calling
// `task` with a subagent name and description; the middleware selects the
// compiled subagent, invokes it as an isolated graph on a per-call thread,
// returns the subagent's final assistant text as the tool result, and commits
// the subagent's changed state back to the parent through the result
// Command's Update.

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Compile-time interface assertions: the middleware plugs into CreateAgent as
// a tool provider (the `task` tool) and a tool-call wrapper (stashing the
// parent state + tool call ID the tool fn needs; same injection pattern as
// FilesystemMiddleware.WrapToolCall).
var (
	_ agentsmw.ToolProvider   = (*SubAgentMiddleware)(nil)
	_ agents.WrapToolCallHook = (*SubAgentMiddleware)(nil)
)

// TaskToolName is the delegation tool's name.
const TaskToolName = "task"

// generalPurposeSubAgentName mirrors GENERAL_PURPOSE_SUBAGENT["name"]
// (subagents.py:566-570; the same literal GeneralPurposeSubAgent builds).
const generalPurposeSubAgentName = "general-purpose"

// TaskToolDescriptionTemplate is the task tool's description, copied VERBATIM
// from TASK_TOOL_DESCRIPTION (subagents.py:559-572) with its
// {available_agents} placeholder left for renderTaskToolDescription. The
// fork-mode note appended per subagent upstream (_FORKED_SUBAGENT_TOOL_NOTE)
// is not ported: fork-mode subagents are out of scope for this tier.
const TaskToolDescriptionTemplate = `Launch an ephemeral subagent to handle a complex, multi-step task.

Available agent types and the tools they have access to:
{available_agents}

Specify subagent_type to select the agent. Usage notes:
- Launch multiple agents concurrently when their tasks are independent, using a single message with multiple tool calls.
- Each invocation is stateless by default: the agent sees only the prompt you give it and returns a single final report. Put full detail in the prompt and state exactly what it should return — unless an agent type below says it inherits your conversation instead.
- The agent's report is not shown to the user; relay a summary yourself.
- Tell the agent whether to create content, analyze, or only research, since it can't necessarily see the user's intent unless it inherits your conversation, as noted per agent type below.
- If an agent's description says to use it proactively, do so without waiting to be asked.
- When only general-purpose is available, use it for any complex, context-heavy task; it has the same capabilities as the main agent.`

// SubAgentMiddleware equips the main agent with the `task` tool and the
// compiled subagents it dispatches to, the Go counterpart of UP's
// SubAgentMiddleware (subagents.py:583-830). Construct with
// NewSubAgentMiddleware.
type SubAgentMiddleware struct {
	// SubAgents holds the compiled subagents in listing order: the
	// auto-added general-purpose subagent first (UP inserts the default spec
	// at index 0, graph.py:854+918), then the caller's entries in order.
	SubAgents []*CompiledSubAgent
	// Tools holds exactly the `task` tool (see ProvidedTools).
	Tools []coretools.Tool
}

// NewSubAgentMiddleware validates and compiles the subagent list and builds
// the `task` tool over it. subagents mixes SubAgentSpec values (compiled
// here: a spec with its own Model compiles as-is, a nil-Model spec inherits
// the parent model) and *CompiledSubAgent values (passed through untouched),
// exactly the list ValidateSubAgents accepts. When no entry is named
// "general-purpose", GeneralPurposeSubAgent(model) is auto-added at the front
// of the list (UP graph.py:854-918).
func NewSubAgentMiddleware(model language.ChatModel, subagents []any) (*SubAgentMiddleware, error) {
	if err := ValidateSubAgents(subagents); err != nil {
		return nil, err
	}

	compiled := make([]*CompiledSubAgent, 0, len(subagents)+1)
	hasGeneralPurpose := false
	for _, spec := range subagents {
		switch s := spec.(type) {
		case SubAgentSpec:
			var (
				c   *CompiledSubAgent
				err error
			)
			if s.Model != nil {
				c, err = CompileSubAgent(s)
			} else {
				c, err = CompileSubAgentWithModel(s, model)
			}
			if err != nil {
				return nil, err
			}
			if s.Name == generalPurposeSubAgentName {
				hasGeneralPurpose = true
			}
			compiled = append(compiled, c)
		case *CompiledSubAgent:
			// ValidateSubAgents already rejected nil entries.
			if s.Name == generalPurposeSubAgentName {
				hasGeneralPurpose = true
			}
			compiled = append(compiled, s)
		}
	}
	if !hasGeneralPurpose {
		gp, err := CompileSubAgent(GeneralPurposeSubAgent(model))
		if err != nil {
			return nil, err
		}
		compiled = append([]*CompiledSubAgent{gp}, compiled...)
	}

	m := &SubAgentMiddleware{SubAgents: compiled}
	tool, err := coretools.NewFunc(
		TaskToolName,
		m.renderTaskToolDescription(),
		taskToolSchema(),
		m.invokeTask,
	)
	if err != nil {
		return nil, fmt.Errorf("SubAgentMiddleware: task tool: %w", err)
	}
	m.Tools = []coretools.Tool{tool}
	return m, nil
}

// ProvidedTools implements agentsmw.ToolProvider: the `task` tool is
// registered with the agent's ToolNode automatically.
func (m *SubAgentMiddleware) ProvidedTools() []coretools.Tool {
	return m.Tools
}

// taskParentStateCtxKey and taskToolCallIDCtxKey carry the two pieces of the
// wrapped tool-call request the task tool fn cannot get from its arguments:
// the parent graph state (BuildTaskInput's source) and the tool call ID (the
// subagent thread's derivation). WrapToolCall stashes them; a direct
// tool.Invoke without the wrapper sees zero values (nil state, empty ID), the
// same degrade FilesystemMiddleware's view injection tolerates.
type (
	taskParentStateCtxKey struct{}
	taskToolCallIDCtxKey  struct{}
)

// WrapToolCall implements agents.WrapToolCallHook: it injects the parent
// state and the tool call ID into the context and delegates, mirroring
// FilesystemMiddleware.WrapToolCall's WithStateView injection (the tool fn
// reads them back via parentStateFromContext / taskToolCallIDFromContext).
func (m *SubAgentMiddleware) WrapToolCall(ctx context.Context, request agentsmw.ToolCallRequest, handler agentsmw.ToolHandler) (messages.Message, error) {
	ctx = context.WithValue(ctx, taskParentStateCtxKey{}, request.State)
	ctx = context.WithValue(ctx, taskToolCallIDCtxKey{}, request.ToolCall.ID)
	return handler(ctx, request)
}

func parentStateFromContext(ctx context.Context) map[string]any {
	state, _ := ctx.Value(taskParentStateCtxKey{}).(map[string]any)
	return state
}

func taskToolCallIDFromContext(ctx context.Context) string {
	id, _ := ctx.Value(taskToolCallIDCtxKey{}).(string)
	return id
}

// taskToolSchema is the task tool's input schema: description and
// subagent_type, both required, with UP's field descriptions verbatim
// (TaskToolSchema, subagents.py:507-516). Unknown keys are rejected in the
// tool fn (UP's model_validator, subagents.py:518-542) rather than by the
// schema because the tool executor does not enforce extra="forbid".
func taskToolSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"description": schema.String(
			"A detailed description of the task for the subagent to perform autonomously. " +
				"Include all necessary context and specify the expected output format."),
		"subagent_type": schema.String(
			"The type of subagent to use. Must be one of the available agent types listed in the tool description."),
	}, "description", "subagent_type")
}

// renderTaskToolDescription fills TaskToolDescriptionTemplate's
// {available_agents} placeholder with one "- name: description" line per
// compiled subagent (UP _describe_subagent_for_tool, subagents.py:592-595,
// without the fork suffix this tier does not port).
func (m *SubAgentMiddleware) renderTaskToolDescription() string {
	lines := make([]string, 0, len(m.SubAgents))
	for _, s := range m.SubAgents {
		lines = append(lines, fmt.Sprintf("- %s: %s", s.Name, s.Description))
	}
	return strings.Replace(TaskToolDescriptionTemplate, "{available_agents}", strings.Join(lines, "\n"), 1)
}

// taskThreadID derives the subagent invocation's thread ID from the task tool
// call's ID (E10): each delegation gets its own thread, so parallel `task`
// calls never share checkpoint history. Without a parent checkpointer these
// threads persist nothing — the ID still namespaces the invocation for
// tracing and future checkpointer wiring.
func taskThreadID(toolCallID string) string {
	return "task:" + toolCallID
}

// invokeTask is the task tool body, mirroring task() (subagents.py:898-931):
// reject unknown arguments, resolve the subagent (unknown name → UP's
// KeyError-style message as tool content), build the isolated input state,
// invoke the subagent graph on its per-call thread, and shape the result.
func (m *SubAgentMiddleware) invokeTask(ctx context.Context, input map[string]any) (coretools.Result, error) {
	// Unknown-key rejection (UP _reject_unknown_keys, subagents.py:518-542):
	// a model that hides the real instructions under an invented key must get
	// an error naming the key, not a silently-dropped argument. One line per
	// unknown key, UP's message verbatim (Python repr quotes → '%s').
	var unknown []string
	for key := range input {
		if key != "description" && key != "subagent_type" {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) > 0 {
		slices.Sort(unknown)
		msgs := make([]string, 0, len(unknown))
		for _, key := range unknown {
			msgs = append(msgs, fmt.Sprintf("Unexpected argument '%s'; put all instructions for the subagent in `description`.", key))
		}
		return coretools.Result{}, errors.New(strings.Join(msgs, "\n"))
	}

	subagentType, _ := input["subagent_type"].(string)
	description, _ := input["description"].(string)

	subagent := m.subagentByName(subagentType)
	if subagent == nil {
		// UP task(), subagents.py:903-904: the message lists the allowed
		// types backtick-quoted in declaration order.
		allowed := make([]string, 0, len(m.SubAgents))
		for _, s := range m.SubAgents {
			allowed = append(allowed, fmt.Sprintf("`%s`", s.Name))
		}
		return coretools.Result{
			Content: fmt.Sprintf(
				"We cannot invoke subagent %s because it does not exist, the only allowed types are %s",
				subagentType, strings.Join(allowed, ", ")),
		}, nil
	}

	toolCallID := taskToolCallIDFromContext(ctx)
	if toolCallID == "" {
		return coretools.Result{}, errors.New("Tool call ID is required for subagent invocation")
	}

	// Isolated handoff (UP _validate_and_prepare_state's else branch,
	// subagents.py:927-930): everything public passes through, messages are
	// rebuilt from the task description.
	taskInput := BuildTaskInput(parentStateFromContext(ctx), description)

	result, err := subagent.Agent.Graph.InvokeWithOptions(ctx, taskInput, graphpkg.Options{
		ThreadID: taskThreadID(toolCallID),
	})
	if err != nil {
		if notice, ok := refusalNoticeForError(err); ok {
			return coretools.Result{Content: notice}, nil
		}
		return coretools.Result{}, err
	}
	if len(result.Interrupts) > 0 {
		// D10: the port does not resume paused subagents — the parent model
		// sees the failure and can re-delegate with different instructions.
		return coretools.Result{}, errors.New("subagent interrupted (not resumable in this port)")
	}

	msgs, _ := result.Values["messages"].([]messages.Message)
	update := filteredStateUpdate(result.Values, taskInput)
	if len(update) == 0 {
		return coretools.Result{Content: taskResultContent(msgs)}, nil
	}
	return coretools.Result{
		Content:  taskResultContent(msgs),
		Artifact: &types.Command{Update: update},
	}, nil
}

func (m *SubAgentMiddleware) subagentByName(name string) *CompiledSubAgent {
	for _, s := range m.SubAgents {
		if s.Name == name {
			return s
		}
	}
	return nil
}

// filteredStateUpdate builds the state writeback from a subagent result,
// mirroring _return_command_with_state_update's filter (subagents.py:849-851):
// a key passes only when it is outside ExcludedStateKeys ∪ PrivateStateKeys
// and the subagent did not return it exactly as received. "messages" is an
// excluded key, so the Update NEVER carries it (Review Focus #3) — the parent
// ToolNode derives the ToolMessage from the tool result content.
func filteredStateUpdate(result, input map[string]any) map[string]any {
	drop := make(map[string]struct{}, len(ExcludedStateKeys)+len(PrivateStateKeys))
	for _, key := range ExcludedStateKeys {
		drop[key] = struct{}{}
	}
	for _, key := range PrivateStateKeys {
		drop[key] = struct{}{}
	}
	update := make(map[string]any, len(result))
	for key, value := range result {
		if _, bad := drop[key]; bad {
			continue
		}
		if inValue, ok := input[key]; ok && reflect.DeepEqual(value, inValue) {
			// UP _is_unchanged (subagents.py:820-834): a returned value equal
			// to the inherited one is dropped — by VALUE, never by map
			// identity, so an untouched-but-copied `files` snapshot still
			// reads as unchanged. Go has no ambiguous `==` fallback; DeepEqual
			// is the whole rule.
			continue
		}
		update[key] = value
	}
	return update
}

// taskResultContent extracts the task result text from a subagent's final
// messages, mirroring _return_command_with_state_update's content branch
// (subagents.py:853-875): a refusal block in the FINAL message becomes the
// refusal notice; otherwise the walk finds the last AIMessage with non-empty
// text (rstrip'd — a trailing empty end_turn message must not blank the
// result); otherwise "". UP's structured_response branch is not ported
// (Task 7 compiles no response_format into subagents).
func taskResultContent(msgs []messages.Message) string {
	if len(msgs) > 0 {
		if block := refusalIn(msgs[len(msgs)-1]); block != nil {
			return refusalNotice(*block)
		}
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != messages.RoleAI {
			continue
		}
		if text := strings.TrimRight(messages.Text(msgs[i]), " \t\n\r\v\f"); text != "" {
			return text
		}
	}
	return ""
}

// refusalIn returns the refusal block in message's content blocks, if any
// (UP _refusal_in, subagents.py:110-112).
func refusalIn(msg messages.Message) *messages.RefusalBlock {
	for i := range msg.ContentBlocks {
		if block, ok := msg.ContentBlocks[i].(messages.RefusalBlock); ok {
			return &block
		}
	}
	return nil
}

// refusalNotice renders the task result for a refused subagent, VERBATIM from
// UP _refusal_notice (subagents.py:95-103): the fixed prefix, the provider
// refusal category and explanation when present, and the fixed suffix.
func refusalNotice(block messages.RefusalBlock) string {
	notice := "The subagent stopped because a model refused the request."
	if category, ok := block.Extras["category"].(string); ok && category != "" {
		notice += " Category: " + category + "."
	}
	if block.Refusal != "" {
		notice += " Explanation: \"" + block.Refusal + "\""
	}
	return notice + " Retrying the same request is unlikely to succeed; rephrase it or continue without it."
}

// refusalNoticeForError converts an error raised inside a subagent into the
// refusal notice when a refusal caused it, mirroring _refusal_notice_for
// (subagents.py:105-119): a *lcerrors.ModelRefusalError (raised by the
// ModelRefusalMiddleware Task 7 installs in every subagent) always notices,
// falling back to the error's own text when the raw message is missing; a
// StructuredOutputValidationError notices only when its AI message carries a
// refusal block; anything else returns ok=false and re-raises.
func refusalNoticeForError(err error) (string, bool) {
	var refusalErr *lcerrors.ModelRefusalError
	if errors.As(err, &refusalErr) {
		if refusalErr.RawMessage != nil {
			if block := refusalIn(*refusalErr.RawMessage); block != nil {
				return refusalNotice(*block), true
			}
		}
		return refusalNotice(messages.RefusalBlock{Refusal: refusalErr.Error()}), true
	}
	var structuredErr *agents.StructuredOutputValidationError
	if errors.As(err, &structuredErr) {
		if block := refusalIn(structuredErr.AIMessage); block != nil {
			return refusalNotice(*block), true
		}
	}
	return "", false
}

// fileMutationTools are the filesystem tools that mutate state, mirroring
// _FILE_MUTATION_TOOLS (filesystem.py:193).
var fileMutationTools = map[string]struct{}{
	"write_file": {},
	"edit_file":  {},
	"delete":     {},
}

// ParallelFileMutationError rejects later same-path file mutations inside one
// model response, the ported _parallel_file_mutation_error guard
// (filesystem.py:204-228): when the tool call being wrapped is a mutating
// filesystem call and an EARLIER call in the same batch (the last AI
// message's tool calls, up to the current call's own entry) already mutates
// the same normalized path, the guard returns the error ToolMessage (UP's
// ported literal) and blocked=true; the caller short-circuits the handler.
// Read-only tools, malformed or non-string paths, and batches without an AI
// message all pass. Paths are compared through normalizeToolPath (the port's
// validate_path), so "/a.md" and "/./a.md" collide as upstream intends.
//
// Wiring: upstream calls this from FilesystemMiddleware.wrap_tool_call
// (filesystem.py:3960, 3990); wiring it into this port's
// FilesystemMiddleware.WrapToolCall is Task 9's assembly decision — the
// helper ships tested here either way.
func ParallelFileMutationError(request agentsmw.ToolCallRequest) (messages.Message, bool) {
	toolCall := request.ToolCall
	if _, mutates := fileMutationTools[toolCall.Name]; !mutates {
		return messages.Message{}, false
	}
	path, ok := toolCall.Args["file_path"].(string)
	if !ok {
		return messages.Message{}, false
	}
	filePath, err := normalizeToolPath(path)
	if err != nil {
		return messages.Message{}, false
	}
	msgs, _ := request.State["messages"].([]messages.Message)
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != messages.RoleAI {
			continue
		}
		for _, call := range msgs[i].ToolCalls {
			if call.ID == toolCall.ID {
				// Reached the current call: everything before it in the
				// batch has been checked.
				return messages.Message{}, false
			}
			if _, mutates := fileMutationTools[call.Name]; !mutates {
				continue
			}
			otherPath, ok := call.Args["file_path"].(string)
			if !ok {
				continue
			}
			normalized, err := normalizeToolPath(otherPath)
			if err != nil {
				continue
			}
			if normalized == filePath {
				msg := messages.Tool(toolCall.ID, "Error: parallel file mutations to the same path are not allowed.")
				msg.Name = toolCall.Name
				msg.ResponseMetadata = map[string]any{"status": "error"}
				return msg, true
			}
		}
		return messages.Message{}, false
	}
	return messages.Message{}, false
}
