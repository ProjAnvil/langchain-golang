package messages

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Role identifies the semantic role of a chat message.
type Role string

const (
	RoleSystem Role = "system"
	RoleHuman  Role = "human"
	RoleAI     Role = "ai"
	RoleTool   Role = "tool"
)

// ContentBlock is the normalized representation for multimodal message content.
// Provider-specific fields can be stored alongside the standard "type" key.
type ContentBlock map[string]any

// ToolCall describes a model-requested tool invocation.
type ToolCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

// UsageMetadata contains token accounting returned by providers.
type UsageMetadata struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
}

// Message is the common chat message shape used across providers.
type Message struct {
	Role                Role           `json:"role"`
	Content             string         `json:"content,omitempty"`
	ContentBlocks       []ContentBlock `json:"content_blocks,omitempty"`
	Name                string         `json:"name,omitempty"`
	ID                  string         `json:"id,omitempty"`
	ToolCallID          string         `json:"tool_call_id,omitempty"`
	ToolCalls           []ToolCall     `json:"tool_calls,omitempty"`
	ResponseMetadata    map[string]any `json:"response_metadata,omitempty"`
	AdditionalKwargs    map[string]any `json:"additional_kwargs,omitempty"`
	UsageMetadata       UsageMetadata  `json:"usage_metadata,omitempty"`
	InvalidToolCalls    []ToolCall     `json:"invalid_tool_calls,omitempty"`
	ProviderNativeEvent map[string]any `json:"provider_native_event,omitempty"`
}

// System creates a system message.
func System(content string) Message {
	return Message{Role: RoleSystem, Content: content}
}

// Human creates a human message.
func Human(content string) Message {
	return Message{Role: RoleHuman, Content: content}
}

// AI creates an AI message.
func AI(content string) Message {
	return Message{Role: RoleAI, Content: content}
}

// Tool creates a tool-result message.
func Tool(toolCallID string, content string) Message {
	return Message{Role: RoleTool, ToolCallID: toolCallID, Content: content}
}

// WithContentBlocks returns a copy of the message with structured content.
func (m Message) WithContentBlocks(blocks []ContentBlock) Message {
	m.ContentBlocks = cloneBlocks(blocks)
	return m
}

// Text returns the textual content of a message. It mirrors Python's message
// text accessor for string content and standard text blocks.
func Text(message Message) string {
	if message.Content != "" {
		return message.Content
	}
	var out strings.Builder
	for _, block := range message.ContentBlocks {
		blockType, _ := block["type"].(string)
		if blockType == "" || blockType == "text" {
			if text, ok := block["text"].(string); ok {
				out.WriteString(text)
			}
		}
	}
	return out.String()
}

// Clone returns a defensive copy of a message.
func Clone(message Message) Message {
	message.ContentBlocks = cloneBlocks(message.ContentBlocks)
	message.ToolCalls = cloneToolCalls(message.ToolCalls)
	message.InvalidToolCalls = cloneToolCalls(message.InvalidToolCalls)
	message.ResponseMetadata = cloneMap(message.ResponseMetadata)
	message.AdditionalKwargs = cloneMap(message.AdditionalKwargs)
	message.ProviderNativeEvent = cloneMap(message.ProviderNativeEvent)
	return message
}

// MessagesToDict serializes messages to stable JSON-shaped maps.
func MessagesToDict(values []Message) ([]map[string]any, error) {
	out := make([]map[string]any, len(values))
	for i, message := range values {
		data, err := MarshalJSONStable(message)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &out[i]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// MessagesFromDict decodes stable JSON-shaped message maps.
func MessagesFromDict(values []map[string]any) ([]Message, error) {
	out := make([]Message, len(values))
	for i, value := range values {
		data, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		msg, err := UnmarshalJSONStable(data)
		if err != nil {
			return nil, err
		}
		out[i] = msg
	}
	return out, nil
}

// BufferString renders messages in a compact transcript form.
func BufferString(values []Message) string {
	lines := make([]string, 0, len(values))
	for _, message := range values {
		lines = append(lines, fmt.Sprintf("%s: %s", roleLabel(message.Role), Text(message)))
	}
	return strings.Join(lines, "\n")
}

// FilterOptions controls Filter.
type FilterOptions struct {
	IncludeRoles []Role
	ExcludeRoles []Role
	IncludeNames []string
	ExcludeNames []string
	IncludeIDs   []string
	ExcludeIDs   []string
}

// Filter returns messages matching include/exclude criteria.
func Filter(values []Message, opts FilterOptions) []Message {
	out := make([]Message, 0, len(values))
	for _, message := range values {
		if !containsRoleOrEmpty(opts.IncludeRoles, message.Role) || containsRole(opts.ExcludeRoles, message.Role) {
			continue
		}
		if !containsStringOrEmpty(opts.IncludeNames, message.Name) || containsString(opts.ExcludeNames, message.Name) {
			continue
		}
		if !containsStringOrEmpty(opts.IncludeIDs, message.ID) || containsString(opts.ExcludeIDs, message.ID) {
			continue
		}
		out = append(out, Clone(message))
	}
	return out
}

// MergeRuns merges consecutive messages with the same role and name. Tool
// messages are not merged because their tool_call_id is semantically important.
func MergeRuns(values []Message) []Message {
	out := make([]Message, 0, len(values))
	for _, message := range values {
		current := Clone(message)
		if len(out) == 0 || !canMerge(out[len(out)-1], current) {
			out = append(out, current)
			continue
		}
		last := &out[len(out)-1]
		last.Content = mergeText(last.Content, current.Content)
		last.ContentBlocks = append(last.ContentBlocks, cloneBlocks(current.ContentBlocks)...)
		last.ToolCalls = append(last.ToolCalls, cloneToolCalls(current.ToolCalls)...)
	}
	return out
}

// Trim keeps messages within an approximate character budget. If fromEnd is
// true, the newest messages are retained.
func Trim(values []Message, maxChars int, fromEnd bool) []Message {
	if maxChars <= 0 {
		return nil
	}
	out := []Message{}
	total := 0
	if fromEnd {
		for i := len(values) - 1; i >= 0; i-- {
			size := len(Text(values[i]))
			if total+size > maxChars {
				break
			}
			total += size
			out = append([]Message{Clone(values[i])}, out...)
		}
		return out
	}
	for _, message := range values {
		size := len(Text(message))
		if total+size > maxChars {
			break
		}
		total += size
		out = append(out, Clone(message))
	}
	return out
}

// MarshalJSONStable returns the canonical JSON representation used by golden
// tests and provider adapters.
func MarshalJSONStable(message Message) ([]byte, error) {
	return json.Marshal(message)
}

// UnmarshalJSONStable decodes a message serialized by MarshalJSONStable.
func UnmarshalJSONStable(data []byte) (Message, error) {
	var message Message
	err := json.Unmarshal(data, &message)
	return message, err
}

func roleLabel(role Role) string {
	switch role {
	case RoleSystem:
		return "System"
	case RoleHuman:
		return "Human"
	case RoleAI:
		return "AI"
	case RoleTool:
		return "Tool"
	default:
		return string(role)
	}
}

func canMerge(a Message, b Message) bool {
	return a.Role == b.Role && a.Name == b.Name && a.Role != RoleTool
}

func mergeText(a string, b string) string {
	if a == "" {
		return b
	}
	if b == "" {
		return a
	}
	return a + "\n" + b
}

func containsRole(values []Role, role Role) bool {
	for _, value := range values {
		if value == role {
			return true
		}
	}
	return false
}

func containsRoleOrEmpty(values []Role, role Role) bool {
	return len(values) == 0 || containsRole(values, role)
}

func containsString(values []string, value string) bool {
	for _, item := range values {
		if item == value {
			return true
		}
	}
	return false
}

func containsStringOrEmpty(values []string, value string) bool {
	return len(values) == 0 || containsString(values, value)
}

func cloneBlocks(values []ContentBlock) []ContentBlock {
	if values == nil {
		return nil
	}
	out := make([]ContentBlock, len(values))
	for i, block := range values {
		out[i] = cloneMap(block)
	}
	return out
}

func cloneToolCalls(values []ToolCall) []ToolCall {
	if values == nil {
		return nil
	}
	out := make([]ToolCall, len(values))
	for i, call := range values {
		call.Args = cloneMap(call.Args)
		out[i] = call
	}
	return out
}

func cloneMap[M ~map[string]any](values M) M {
	if values == nil {
		return nil
	}
	out := make(M, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}
