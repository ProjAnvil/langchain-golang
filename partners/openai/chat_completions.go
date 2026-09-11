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
	Model          string         `json:"model"`
	Messages       []chatMessage  `json:"messages"`
	Tools          []chatToolDef  `json:"tools,omitempty"`
	ToolChoice     any            `json:"tool_choice,omitempty"`
	ResponseFormat map[string]any `json:"response_format,omitempty"`
	Temperature    *float64       `json:"temperature,omitempty"`
	MaxTokens      *int           `json:"max_tokens,omitempty"`
	Stream         bool           `json:"stream,omitempty"`
	// Sampling knobs mirroring Python BaseChatOpenAI's optional fields,
	// forwarded by _default_params' exclude_if_none map
	// (chat_models/base.py:1340-1350): presence_penalty (:753),
	// frequency_penalty (:756), seed (:759), logprobs (:762),
	// top_logprobs (:765), logit_bias (:772), n (:778), top_p (:781),
	// stop (:963).
	TopP             *float64    `json:"top_p,omitempty"`
	Stop             []string    `json:"stop,omitempty"`
	Seed             *int        `json:"seed,omitempty"`
	PresencePenalty  *float64    `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64    `json:"frequency_penalty,omitempty"`
	LogitBias        map[int]int `json:"logit_bias,omitempty"`
	N                *int        `json:"n,omitempty"`
	Logprobs         *bool       `json:"logprobs,omitempty"`
	TopLogprobs      *int        `json:"top_logprobs,omitempty"`
	// StreamOptions opts into streaming usage accounting
	// ({"include_usage": true}); the API then appends a final choices-less
	// chunk carrying the usage object.
	StreamOptions *chatStreamOptions `json:"stream_options,omitempty"`
	// ReasoningEffort (low|medium|high) steers reasoning models' effort on
	// the Chat Completions API (OpenAI o-series / gpt-5 family). Ignored by
	// non-reasoning models and gateways that don't know the field.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

// chatStreamOptions is the Chat Completions stream_options object.
type chatStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
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
	Role string `json:"role"`
	// Content is a bare string for plain messages or a []any of structured
	// content parts (text/image_url/input_audio) for multimodal human
	// messages.
	Content    any            `json:"content,omitempty"`
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
	Usage   chatUsage               `json:"usage"`
}

// chatUsage models the Chat Completions usage object. The official API spells
// the counts prompt_tokens/completion_tokens; the Responses-style
// input_tokens/output_tokens spellings (emitted by this repo's CC test fakes
// and some OpenAI-compatible gateways) are accepted as fallbacks so either
// spelling parses. toUsagePayload normalizes the winner into the
// Responses-shaped usagePayload the shared toMessage path consumes.
type chatUsage struct {
	PromptTokens            int                    `json:"prompt_tokens"`
	CompletionTokens        int                    `json:"completion_tokens"`
	TotalTokens             int                    `json:"total_tokens"`
	InputTokens             int                    `json:"input_tokens"`
	OutputTokens            int                    `json:"output_tokens"`
	PromptTokensDetails     *chatPromptDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *chatCompletionDetails `json:"completion_tokens_details"`
	InputTokensDetails      *inputTokensDetails    `json:"input_tokens_details"`
	OutputTokensDetails     *outputTokensDetails   `json:"output_tokens_details"`
}

type chatPromptDetails struct {
	CachedTokens int `json:"cached_tokens"`
	AudioTokens  int `json:"audio_tokens"`
}

type chatCompletionDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
	AudioTokens     int `json:"audio_tokens"`
}

// toUsagePayload normalizes either field spelling into usagePayload.
func (u chatUsage) toUsagePayload() usagePayload {
	input, inputDetails := u.PromptTokens, any(u.PromptTokensDetails)
	if input == 0 && u.InputTokens != 0 {
		input, inputDetails = u.InputTokens, u.InputTokensDetails
	}
	output, outputDetails := u.CompletionTokens, any(u.CompletionTokensDetails)
	if output == 0 && u.OutputTokens != 0 {
		output, outputDetails = u.OutputTokens, u.OutputTokensDetails
	}
	out := usagePayload{
		InputTokens:  input,
		OutputTokens: output,
		TotalTokens:  u.TotalTokens,
	}
	switch d := inputDetails.(type) {
	case *chatPromptDetails:
		if d != nil {
			out.InputTokensDetails = &inputTokensDetails{CachedTokens: d.CachedTokens}
		}
	case *inputTokensDetails:
		if d != nil {
			out.InputTokensDetails = d
		}
	}
	switch d := outputDetails.(type) {
	case *chatCompletionDetails:
		if d != nil {
			out.OutputTokensDetails = &outputTokensDetails{ReasoningTokens: d.ReasoningTokens}
		}
	case *outputTokensDetails:
		if d != nil {
			out.OutputTokensDetails = d
		}
	}
	return out
}

// isZero reports whether the usage object carries no data at all, so streams
// can skip yielding a usage chunk when the payload has none.
func (u chatUsage) isZero() bool {
	return u == chatUsage{}
}

type chatCompletionsChoice struct {
	Message chatCompletionsMessage `json:"message"`
}

type chatCompletionsMessage struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	ToolCalls []chatToolCall `json:"tool_calls,omitempty"`
}

func (m ChatModel) buildChatCompletionsRequest(input []messages.Message) (chatCompletionsRequest, error) {
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
	payload.TopP = m.topP
	payload.Stop = m.stop
	payload.Seed = m.seed
	payload.PresencePenalty = m.presencePenalty
	payload.FrequencyPenalty = m.frequencyPenalty
	payload.LogitBias = m.logitBias
	payload.N = m.n
	payload.Logprobs = m.logprobs
	payload.TopLogprobs = m.topLogprobs
	if m.reasoningEffort != "" {
		payload.ReasoningEffort = m.reasoningEffort
	}
	if m.toolChoice != nil {
		payload.ToolChoice = m.toolChoice.value
	}
	// Same precedence as the Responses builder (buildRequest): a
	// structuredOutput binding (InvokeStructured / WithStructuredOutput) wins
	// over a raw responseFormat dict. On the wire the json_schema config takes
	// the CC-native nested form
	// {"type":"json_schema","json_schema":{name,schema,strict}}; strict is
	// omitted when false, mirroring the Responses path's responseFormat
	// serialization (json:"strict,omitempty"). json_mode / raw dicts
	// ({"type":"json_object"}, ...) pass through verbatim.
	if m.structuredOutput != nil {
		jsonSchema := map[string]any{
			"name":   m.structuredOutput.Name,
			"schema": m.structuredOutput.Schema,
		}
		if m.structuredOutput.Strict {
			jsonSchema["strict"] = true
		}
		payload.ResponseFormat = map[string]any{
			"type":        "json_schema",
			"json_schema": jsonSchema,
		}
	} else if m.responseFormat != nil {
		payload.ResponseFormat = m.responseFormat
	}

	for _, message := range input {
		cm := chatMessage{Content: message.Content, Name: message.Name}
		switch message.Role {
		case messages.RoleSystem:
			cm.Role = "system"
		case messages.RoleHuman:
			cm.Role = "user"
			// Structured content blocks become Chat Completions content
			// parts; plain text keeps a bare string content.
			if len(message.ContentBlocks) > 0 {
				parts, err := chatContentParts(message.ContentBlocks)
				if err != nil {
					return chatCompletionsRequest{}, err
				}
				cm.Content = parts
			}
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
	return payload, nil
}

func (m ChatModel) createChatCompletionsResponse(ctx context.Context, input []messages.Message) (responsePayload, error) {
	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	defer cancel()
	request, err := m.buildChatCompletionsRequest(input)
	if err != nil {
		return responsePayload{}, err
	}
	resp, err := postJSON[chatCompletionsResponse](ctx, m.config, "/chat/completions", request)
	if err != nil {
		return responsePayload{}, err
	}
	return resp.toResponsesPayload(), nil
}

// toResponsesPayload converts a Chat Completions response into the
// Responses-shaped responsePayload so the shared `toMessage` path is reused.
func (r chatCompletionsResponse) toResponsesPayload() responsePayload {
	out := responsePayload{ID: r.ID, Model: r.Model, Usage: r.Usage.toUsagePayload()}
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
