package middleware

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
)

type DecisionType string

const (
	DecisionApprove DecisionType = "approve"
	DecisionEdit    DecisionType = "edit"
	DecisionReject  DecisionType = "reject"
	DecisionRespond DecisionType = "respond"
)

type ActionRequest struct {
	Name        string
	Args        map[string]any
	Description string
}

type ReviewConfig struct {
	ActionName       string
	AllowedDecisions []DecisionType
	ArgsSchema       map[string]any
}

type HITLRequest struct {
	ActionRequests []ActionRequest
	ReviewConfigs  []ReviewConfig
}

type Decision struct {
	Type         DecisionType
	EditedAction *ToolCall
	Message      string
}

type InterruptConfig struct {
	AllowedDecisions []DecisionType
	Description      string
	// DescriptionFunc, when set, generates the description dynamically from
	// the pending tool call/state/runtime, taking precedence over
	// Description. Mirrors Python's `InterruptOnConfig.description` accepting
	// a `Callable[[ToolCall, AgentState, Runtime], str]` in addition to a
	// plain string; State and Runtime are surfaced via the same
	// ToolCallRequest shape used by When/WrapToolCallHook.
	DescriptionFunc func(ToolCallRequest) string
	ArgsSchema      map[string]any
	When            func(ToolCallRequest) bool
}

type HumanDecisionFunc func(HITLRequest) ([]Decision, error)

type HumanInTheLoopMiddleware struct {
	InterruptOn       map[string]InterruptConfig
	DescriptionPrefix string
	Decide            HumanDecisionFunc
}

func NewHumanInTheLoopMiddleware(interruptOn map[string]InterruptConfig, decide HumanDecisionFunc) *HumanInTheLoopMiddleware {
	resolved := map[string]InterruptConfig{}
	for name, config := range interruptOn {
		if len(config.AllowedDecisions) > 0 {
			resolved[name] = config
		}
	}
	return &HumanInTheLoopMiddleware{
		InterruptOn:       resolved,
		DescriptionPrefix: "Tool execution requires approval",
		Decide:            decide,
	}
}

func (m *HumanInTheLoopMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	msgs, ok := messagesFromState(state)
	if !ok || len(msgs) == 0 {
		return nil, nil
	}
	lastIdx := -1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleAI {
			lastIdx = i
			break
		}
	}
	if lastIdx < 0 || len(msgs[lastIdx].ToolCalls) == 0 {
		return nil, nil
	}
	lastAI := msgs[lastIdx]

	actionRequests := []ActionRequest{}
	reviewConfigs := []ReviewConfig{}
	interruptIndices := []int{}
	for idx, call := range lastAI.ToolCalls {
		config, ok := m.InterruptOn[call.Name]
		if !ok {
			continue
		}
		req := ToolCallRequest{ToolCall: ToolCall{Name: call.Name, Args: call.Args, ID: call.ID}, State: state}
		if config.When != nil && !config.When(req) {
			continue
		}
		description := config.Description
		if config.DescriptionFunc != nil {
			description = config.DescriptionFunc(req)
		}
		if description == "" {
			prefix := m.DescriptionPrefix
			if prefix == "" {
				prefix = "Tool execution requires approval"
			}
			description = fmt.Sprintf("%s\n\nTool: %s\nArgs: %v", prefix, call.Name, call.Args)
		}
		actionRequests = append(actionRequests, ActionRequest{Name: call.Name, Args: cloneAnyMap(call.Args), Description: description})
		reviewConfigs = append(reviewConfigs, ReviewConfig{ActionName: call.Name, AllowedDecisions: config.AllowedDecisions, ArgsSchema: config.ArgsSchema})
		interruptIndices = append(interruptIndices, idx)
	}
	if len(actionRequests) == 0 {
		return nil, nil
	}
	if m.Decide == nil {
		return nil, fmt.Errorf("human-in-the-loop decision function is required")
	}
	decisions, err := m.Decide(HITLRequest{ActionRequests: actionRequests, ReviewConfigs: reviewConfigs})
	if err != nil {
		return nil, err
	}
	if len(decisions) != len(interruptIndices) {
		return nil, fmt.Errorf("number of human decisions (%d) does not match number of hanging tool calls (%d)", len(decisions), len(interruptIndices))
	}

	revised := []messages.ToolCall{}
	artificial := []messages.Message{}
	decisionIdx := 0
	interruptSet := map[int]bool{}
	for _, idx := range interruptIndices {
		interruptSet[idx] = true
	}
	for idx, call := range lastAI.ToolCalls {
		if !interruptSet[idx] {
			revised = append(revised, call)
			continue
		}
		config := m.InterruptOn[call.Name]
		decision := decisions[decisionIdx]
		decisionIdx++
		nextCall, toolMessage, err := processHumanDecision(decision, call, config)
		if err != nil {
			return nil, err
		}
		if nextCall != nil {
			revised = append(revised, *nextCall)
		}
		if toolMessage != nil {
			artificial = append(artificial, *toolMessage)
		}
	}
	lastAI.ToolCalls = revised
	out := []messages.Message{lastAI}
	out = append(out, artificial...)
	return map[string]any{"messages": out}, nil
}

func processHumanDecision(decision Decision, call messages.ToolCall, config InterruptConfig) (*messages.ToolCall, *messages.Message, error) {
	if !decisionAllowed(decision.Type, config.AllowedDecisions) {
		return nil, nil, fmt.Errorf("unexpected human decision: %s is not allowed for tool %q", decision.Type, call.Name)
	}
	switch decision.Type {
	case DecisionApprove:
		return &call, nil, nil
	case DecisionEdit:
		if decision.EditedAction == nil {
			return nil, nil, fmt.Errorf("edit decision requires edited action")
		}
		return &messages.ToolCall{ID: call.ID, Name: decision.EditedAction.Name, Args: cloneAnyMap(decision.EditedAction.Args)}, nil, nil
	case DecisionReject:
		content := decision.Message
		if content == "" {
			content = fmt.Sprintf("User rejected the tool call for `%s` with id %s. The tool was not executed. Do not retry this tool call unless the user explicitly requests it.", call.Name, call.ID)
		}
		msg := errorToolMessage(call.ID, call.Name, content)
		return &call, &msg, nil
	case DecisionRespond:
		msg := messages.Tool(call.ID, decision.Message)
		msg.Name = call.Name
		msg.ResponseMetadata = map[string]any{"status": "success"}
		return &call, &msg, nil
	default:
		return nil, nil, fmt.Errorf("unexpected human decision: %s", decision.Type)
	}
}

func decisionAllowed(decision DecisionType, allowed []DecisionType) bool {
	for _, value := range allowed {
		if value == decision {
			return true
		}
	}
	return false
}
