package middleware

import (
	"context"
	"fmt"
	"slices"

	"github.com/projanvil/langchain-golang/core/messages"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/types"
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

// HITLResponse is the resume payload answering a HITLRequest interrupt,
// mirroring Python's HITLResponse TypedDict (human_in_the_loop.py:129-133).
// A caller resumes a paused HITL run by passing it (or its map wire form, see
// DecodeHITLResponse) as graph Options.Resume.
type HITLResponse struct {
	Decisions []Decision
}

// HitlInterrupter marks middleware whose human review pauses the run through
// a langgraph interrupt instead of a synchronous Decide callback.
// *HumanInTheLoopMiddleware implements it exactly when its Decide is nil
// (the two modes are mutually exclusive), letting agents wiring detect the
// interrupt mode without depending on the concrete type.
type HitlInterrupter interface {
	HitlInterruptEnabled() bool
}

// HitlInterruptEnabled reports whether the middleware reviews through
// interrupts: true when and only if Decide is nil (interrupt mode).
func (m *HumanInTheLoopMiddleware) HitlInterruptEnabled() bool { return m.Decide == nil }

// HITLOption configures NewInterruptHumanInTheLoopMiddleware.
type HITLOption func(*HumanInTheLoopMiddleware)

// WithDescriptionPrefix overrides the default description prefix used when an
// InterruptConfig carries no description (NewHumanInTheLoopMiddleware's fixed
// "Tool execution requires approval").
func WithDescriptionPrefix(prefix string) HITLOption {
	return func(m *HumanInTheLoopMiddleware) { m.DescriptionPrefix = prefix }
}

// NewInterruptHumanInTheLoopMiddleware builds a HumanInTheLoopMiddleware in
// interrupt mode: instead of a synchronous Decide callback, the review pauses
// the graph run via graph.Interrupt (surfaced as Result.Interrupts with the
// HITLRequest) and resumes when the caller supplies a HITLResponse through
// graph Options.Resume. This mirrors Python's HumanInTheLoopMiddleware, which
// is interrupt-based only; the Go Decide callback remains the synchronous
// alternative (NewHumanInTheLoopMiddleware) with identical decision
// semantics.
func NewInterruptHumanInTheLoopMiddleware(interruptOn map[string]InterruptConfig, opts ...HITLOption) *HumanInTheLoopMiddleware {
	m := NewHumanInTheLoopMiddleware(interruptOn, nil)
	for _, opt := range opts {
		if opt != nil {
			opt(m)
		}
	}
	return m
}

// DecodeHITLResponse decodes a HITL resume value into its decisions,
// accepting every shape a resume can arrive in:
//
//   - HITLResponse (or *HITLResponse), the typed form a Go caller passes
//     directly to graph Options.Resume;
//   - map[string]any{"decisions": [...]} — the JSON round-trip form (a
//     checkpoint serialized through a JSON saver, or a Python-side producer),
//     each decision a map with the wire keys "type", "edited_action"
//     ({"name", "args"}), and "message". The capitalized Go-JSON spelling
//     ("Decisions", "Type", ...) is accepted too since the structs carry no
//     JSON tags;
//   - a bare decision slice: []Decision, or []any of decision maps.
//
// Anything else is an error, as is a decision map without a string "type".
func DecodeHITLResponse(v any) ([]Decision, error) {
	switch resp := v.(type) {
	case HITLResponse:
		return resp.Decisions, nil
	case *HITLResponse:
		if resp == nil {
			return nil, fmt.Errorf("cannot decode HITL response from nil *HITLResponse")
		}
		return resp.Decisions, nil
	case []Decision:
		return resp, nil
	case map[string]any:
		raw, ok := resp["decisions"]
		if !ok {
			raw, ok = resp["Decisions"]
		}
		if !ok {
			return nil, fmt.Errorf("cannot decode HITL response from map without a \"decisions\" key")
		}
		return decodeDecisionList(raw)
	case []any:
		return decodeDecisionList(resp)
	default:
		return nil, fmt.Errorf("cannot decode HITL response from %T", v)
	}
}

func decodeDecisionList(raw any) ([]Decision, error) {
	switch list := raw.(type) {
	case []Decision:
		return list, nil
	case []any:
		decisions := make([]Decision, 0, len(list))
		for i, item := range list {
			decision, err := decodeDecision(item)
			if err != nil {
				return nil, fmt.Errorf("decision %d: %w", i, err)
			}
			decisions = append(decisions, decision)
		}
		return decisions, nil
	default:
		return nil, fmt.Errorf("decisions must be a list, got %T", raw)
	}
}

func decodeDecision(v any) (Decision, error) {
	switch decision := v.(type) {
	case Decision:
		return decision, nil
	case map[string]any:
		var out Decision
		raw, ok := decision["type"]
		if !ok {
			raw, ok = decision["Type"]
		}
		if !ok {
			return out, fmt.Errorf("missing \"type\" key")
		}
		typeName, ok := raw.(string)
		if !ok {
			return out, fmt.Errorf("\"type\" must be a string, got %T", raw)
		}
		out.Type = DecisionType(typeName)
		if edited, ok := decision["edited_action"]; ok && edited != nil {
			editedMap, ok := edited.(map[string]any)
			if !ok {
				return out, fmt.Errorf("\"edited_action\" must be an object, got %T", edited)
			}
			call := &ToolCall{}
			call.Name, _ = editedMap["name"].(string)
			if args, ok := editedMap["args"].(map[string]any); ok {
				call.Args = cloneAnyMap(args)
			}
			if id, ok := editedMap["id"].(string); ok {
				call.ID = id
			}
			out.EditedAction = call
		}
		if message, ok := decision["message"].(string); ok {
			out.Message = message
		} else if message, ok := decision["Message"].(string); ok {
			out.Message = message
		}
		return out, nil
	default:
		return Decision{}, fmt.Errorf("cannot decode decision from %T", v)
	}
}

// HITLRequestFromInterrupt restores the typed HITLRequest from an interrupt's
// Value, accepting both the struct form a live run produces and the map wire
// form a JSON round-trip leaves behind (keys "action_requests" /
// "review_configs", each entry carrying "name"/"args"/"description" and
// "action_name"/"allowed_decisions"/"args_schema" respectively; the
// capitalized Go-JSON spellings are accepted too). It is the companion of
// DecodeHITLResponse for reading Result.Interrupts entries.
func HITLRequestFromInterrupt(intr types.Interrupt) (HITLRequest, error) {
	switch value := intr.Value.(type) {
	case HITLRequest:
		return value, nil
	case *HITLRequest:
		if value == nil {
			return HITLRequest{}, fmt.Errorf("cannot decode HITL request from nil *HITLRequest")
		}
		return *value, nil
	case map[string]any:
		var out HITLRequest
		rawRequests, ok := firstMapValue(value, "action_requests", "ActionRequests")
		if !ok {
			return out, fmt.Errorf("cannot decode HITL request from map without an \"action_requests\" key")
		}
		requests, ok := rawRequests.([]any)
		if !ok {
			return out, fmt.Errorf("\"action_requests\" must be a list, got %T", rawRequests)
		}
		for i, item := range requests {
			entry, ok := item.(map[string]any)
			if !ok {
				return out, fmt.Errorf("action request %d must be an object, got %T", i, item)
			}
			request := ActionRequest{}
			request.Name, _ = entry["name"].(string)
			if args, ok := entry["args"].(map[string]any); ok {
				request.Args = cloneAnyMap(args)
			}
			request.Description, _ = entry["description"].(string)
			out.ActionRequests = append(out.ActionRequests, request)
		}
		rawConfigs, _ := firstMapValue(value, "review_configs", "ReviewConfigs")
		configs, _ := rawConfigs.([]any)
		for i, item := range configs {
			entry, ok := item.(map[string]any)
			if !ok {
				return out, fmt.Errorf("review config %d must be an object, got %T", i, item)
			}
			config := ReviewConfig{}
			config.ActionName, _ = entry["action_name"].(string)
			if allowed, ok := entry["allowed_decisions"].([]any); ok {
				for _, d := range allowed {
					if name, ok := d.(string); ok {
						config.AllowedDecisions = append(config.AllowedDecisions, DecisionType(name))
					}
				}
			}
			if schemaMap, ok := entry["args_schema"].(map[string]any); ok {
				config.ArgsSchema = cloneAnyMap(schemaMap)
			}
			out.ReviewConfigs = append(out.ReviewConfigs, config)
		}
		return out, nil
	default:
		return HITLRequest{}, fmt.Errorf("cannot decode HITL request from %T", intr.Value)
	}
}

// firstMapValue returns the first of keys present in m (nil, false when none
// is present), tolerating both wire spellings of a HITL payload key.
func firstMapValue(m map[string]any, keys ...string) (any, bool) {
	for _, key := range keys {
		if v, ok := m[key]; ok {
			return v, true
		}
	}
	return nil, false
}

// hitlRequestValue renders a HITLRequest as the JSON-native interrupt payload
// (snake_case keys, mirroring Python's TypedDict wire form). The interrupt
// value is deliberately NOT the bare struct: the checkpoint serde registry
// rejects unregistered concrete types, so a struct value would fail to
// persist under the JSON-based savers; a map round-trips losslessly and
// HITLRequestFromInterrupt restores either form.
func hitlRequestValue(req HITLRequest) map[string]any {
	requests := make([]any, 0, len(req.ActionRequests))
	for _, action := range req.ActionRequests {
		requests = append(requests, map[string]any{
			"name":        action.Name,
			"args":        cloneAnyMap(action.Args),
			"description": action.Description,
		})
	}
	configs := make([]any, 0, len(req.ReviewConfigs))
	for _, config := range req.ReviewConfigs {
		entry := map[string]any{
			"action_name": config.ActionName,
		}
		if len(config.AllowedDecisions) > 0 {
			allowed := make([]any, 0, len(config.AllowedDecisions))
			for _, d := range config.AllowedDecisions {
				allowed = append(allowed, string(d))
			}
			entry["allowed_decisions"] = allowed
		}
		if config.ArgsSchema != nil {
			entry["args_schema"] = cloneAnyMap(config.ArgsSchema)
		}
		configs = append(configs, entry)
	}
	return map[string]any{
		"action_requests": requests,
		"review_configs":  configs,
	}
}

// AfterModel implements the synchronous Decide-mode review inline in the
// model node. When the middleware was built for interrupt mode (a nil Decide,
// i.e. NewInterruptHumanInTheLoopMiddleware), it is a deliberate no-op: the
// review flow runs in the dedicated HITL graph node via AfterModelNode, and
// pausing inline (before the model node's update commits) would replay the
// whole model node — re-invoking the model — on resume.
func (m *HumanInTheLoopMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	if m.Decide == nil {
		return nil, nil
	}
	lastAI, actionRequests, reviewConfigs, interruptIndices, ok := m.reviewableCalls(state)
	if !ok {
		return nil, nil
	}
	decisions, err := m.Decide(HITLRequest{ActionRequests: actionRequests, ReviewConfigs: reviewConfigs})
	if err != nil {
		return nil, err
	}
	return m.applyHITLDecisions(lastAI, interruptIndices, decisions)
}

// AfterModelNode implements the interrupt-mode review as a dedicated graph
// node hook (see agents.AfterModelNodeHook / agents.HITLNodeName, mirroring
// Python where every after_model middleware is its own graph node). It runs
// AFTER the model node's update has committed, so an Interrupt here pauses
// the run with the pending AI message already checkpointed and a Resume
// re-runs only this node — the model is never re-invoked, and the decisions
// act on the committed, stable AIMessage.
//
// The flow mirrors Python's after_model (human_in_the_loop.py:384-471):
// scan the last AI message's tool calls, filter via When, build the
// HITLRequest, interrupt, then apply the decoded HITLResponse decisions
// through the same four-branch logic as Decide mode (processHumanDecision).
// The returned update replaces the AI message in place (MessagesReducer
// matches by ID) and appends any artificial ToolMessages a reject/respond
// decision produced.
func (m *HumanInTheLoopMiddleware) AfterModelNode(ctx context.Context, state map[string]any) (map[string]any, error) {
	lastAI, actionRequests, reviewConfigs, interruptIndices, ok := m.reviewableCalls(state)
	if !ok {
		return nil, nil
	}
	response := graphpkg.Interrupt(ctx, hitlRequestValue(HITLRequest{
		ActionRequests: actionRequests,
		ReviewConfigs:  reviewConfigs,
	}))
	decisions, err := DecodeHITLResponse(response)
	if err != nil {
		return nil, err
	}
	return m.applyHITLDecisions(lastAI, interruptIndices, decisions)
}

// reviewableCalls scans state's last AI message for tool calls requiring
// human review and builds the matching ActionRequests/ReviewConfigs plus the
// indices of the reviewed calls within that message's ToolCalls. ok is false
// when there is nothing to review (no messages, no AI message, no tool calls,
// or every candidate filtered out by When / a missing config).
func (m *HumanInTheLoopMiddleware) reviewableCalls(state map[string]any) (lastAI messages.Message, actionRequests []ActionRequest, reviewConfigs []ReviewConfig, interruptIndices []int, ok bool) {
	msgs, found := messagesFromState(state)
	if !found || len(msgs) == 0 {
		return messages.Message{}, nil, nil, nil, false
	}
	lastIdx := -1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleAI {
			lastIdx = i
			break
		}
	}
	if lastIdx < 0 || len(msgs[lastIdx].ToolCalls) == 0 {
		return messages.Message{}, nil, nil, nil, false
	}
	lastAI = msgs[lastIdx]

	actionRequests = []ActionRequest{}
	reviewConfigs = []ReviewConfig{}
	interruptIndices = []int{}
	for idx, call := range lastAI.ToolCalls {
		config, found := m.InterruptOn[call.Name]
		if !found {
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
		return messages.Message{}, nil, nil, nil, false
	}
	return lastAI, actionRequests, reviewConfigs, interruptIndices, true
}

// applyHITLDecisions validates the decision count against the number of
// interrupted tool calls, then rebuilds the AI message's tool calls in their
// original order — interrupted calls replaced per their decision via
// processHumanDecision, untouched calls kept verbatim — returning the update
// {"messages": [revisedAI, *artificialToolMessages]}. Shared by the Decide
// and interrupt modes so both apply identical semantics.
func (m *HumanInTheLoopMiddleware) applyHITLDecisions(lastAI messages.Message, interruptIndices []int, decisions []Decision) (map[string]any, error) {
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
	if !slices.Contains(config.AllowedDecisions, decision.Type) {
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
