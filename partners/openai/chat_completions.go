package openai

import (
	"context"
	"encoding/json"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// Chat Completions API types and conversion. The Chat Completions request and
// response shapes differ from the Responses API, so they are modeled here and
// converted back into the Responses-shaped responsePayload that `Invoke` /
// `toMessage` already consume.

type chatCompletionsRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Tools       []chatToolDef `json:"tools,omitempty"`
	ToolChoice  any           `json:"tool_choice,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	// ReasoningEffort (low|medium|high) steers reasoning models' effort on
	// the Chat Completions API (OpenAI o-series / gpt-5 family). Ignored by
	// non-reasoning models and gateways that don't know the field.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

// chatToolDef is the Chat Completions tools entry: the function descriptor
// is NESTED under "function", unlike the Responses API's flat toolSpec.
// OpenAI-compatible gateways (DeepSeek and friends) reject the flat shape
// with "missing field `function`".
type chatToolDef struct {
	Type     string           `json:"type"`
	Function chatFunctionSpec `json:"function"`
}

type chatFunctionSpec struct {
	Name        string        `json:"name"`
	Description string        `json:"description,omitempty"`
	Parameters  schema.Schema `json:"parameters,omitempty"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	Name       string         `json:"name,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatCompletionsResponse struct {
	ID      string                  `json:"id"`
	Model   string                  `json:"model"`
	Choices []chatCompletionsChoice `json:"choices"`
	Usage   usagePayload            `json:"usage"`
}

type chatCompletionsChoice struct {
	Message chatCompletionsMessage `json:"message"`
}

type chatCompletionsMessage struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	ToolCalls []chatToolCall `json:"tool_calls,omitempty"`
}

func (m ChatModel) buildChatCompletionsRequest(input []messages.Message) chatCompletionsRequest {
	payload := chatCompletionsRequest{
		Model:    m.config.Model,
		Messages: make([]chatMessage, 0, len(input)),
		Tools:    make([]chatToolDef, 0, len(m.boundTools)),
	}
	if m.config.Temperature != nil {
		payload.Temperature = m.config.Temperature
	}
	if m.config.MaxTokens != nil {
		payload.MaxTokens = m.config.MaxTokens
	}
	if m.reasoningEffort != "" {
		payload.ReasoningEffort = m.reasoningEffort
	}
	if m.toolChoice != nil {
		payload.ToolChoice = m.toolChoice.value
	}

	for _, message := range input {
		cm := chatMessage{Content: message.Content, Name: message.Name}
		switch message.Role {
		case messages.RoleSystem:
			cm.Role = "system"
		case messages.RoleHuman:
			cm.Role = "user"
		case messages.RoleAI:
			cm.Role = "assistant"
			for _, tc := range message.ToolCalls {
				argsJSON, _ := json.Marshal(tc.Args)
				cm.ToolCalls = append(cm.ToolCalls, chatToolCall{
					ID:       tc.ID,
					Type:     "function",
					Function: chatFunction{Name: tc.Name, Arguments: string(argsJSON)},
				})
			}
		case messages.RoleTool:
			cm.Role = "tool"
			cm.ToolCallID = message.ToolCallID
		}
		payload.Messages = append(payload.Messages, cm)
	}

	for _, tool := range m.boundTools {
		payload.Tools = append(payload.Tools, chatToolDef{
			Type: "function",
			Function: chatFunctionSpec{
				Name:        tool.Name(),
				Description: tool.Description(),
				Parameters:  tool.ArgsSchema(),
			},
		})
	}
	if len(payload.Tools) == 0 {
		payload.Tools = nil
	}
	return payload
}

func (m ChatModel) createChatCompletionsResponse(ctx context.Context, input []messages.Message) (responsePayload, error) {
	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	defer cancel()
	resp, err := postJSON[chatCompletionsResponse](ctx, m.config, "/chat/completions", m.buildChatCompletionsRequest(input))
	if err != nil {
		return responsePayload{}, err
	}
	return resp.toResponsesPayload(), nil
}

// toResponsesPayload converts a Chat Completions response into the
// Responses-shaped responsePayload so the shared `toMessage` path is reused.
func (r chatCompletionsResponse) toResponsesPayload() responsePayload {
	out := responsePayload{ID: r.ID, Model: r.Model, Usage: r.Usage}
	if len(r.Choices) == 0 {
		return out
	}
	msg := r.Choices[0].Message
	if msg.Content != "" {
		out.Output = append(out.Output, outputItem{
			Type:    "message",
			Role:    "assistant",
			Content: []contentOutput{{Type: "output_text", Text: msg.Content}},
		})
	}
	for _, tc := range msg.ToolCalls {
		out.Output = append(out.Output, outputItem{
			Type:      "function_call",
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}
	return out
}
