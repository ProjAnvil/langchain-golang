package messages

import (
	"bytes"
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

// ContentBlock is now defined in contentblocks.go as a sealed interface with
// concrete typed structs (TextBlock, ImageBlock, etc.). See contentblocks.go
// for the full type hierarchy, ParseContentBlock, BlockToMap, and CloneBlock.

// ToolCall describes a model-requested tool invocation.
type ToolCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

// InputTokenDetails breaks down input token counts. Mirrors Python's
// InputTokenDetails (langchain_core/messages/ai.py).
type InputTokenDetails struct {
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}

// OutputTokenDetails breaks down output token counts. Mirrors Python's
// OutputTokenDetails (langchain_core/messages/ai.py).
type OutputTokenDetails struct {
	ReasoningOutputTokens int `json:"reasoning_output_tokens,omitempty"`
}

// UsageMetadata contains token accounting returned by providers.
type UsageMetadata struct {
	InputTokens        int                 `json:"input_tokens,omitempty"`
	OutputTokens       int                 `json:"output_tokens,omitempty"`
	TotalTokens        int                 `json:"total_tokens,omitempty"`
	InputTokenDetails  *InputTokenDetails  `json:"input_token_details,omitempty"`
	OutputTokenDetails *OutputTokenDetails `json:"output_token_details,omitempty"`
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

// RemoveMessage is a sentinel used in message-list updates to mark a message
// (by ID) for removal during the merge. An empty ID means "remove all
// messages", mirroring Python's RemoveMessage(id=REMOVE_ALL_MESSAGES).
//
// It is intentionally not a Message: it only appears in update lists (as a
// MessageUpdate) and is consumed by the message-list reducer.
type RemoveMessage struct {
	ID string `json:"id"`
}

// IsRemoveMessage reports that this value is a removal sentinel.
func (RemoveMessage) IsRemoveMessage() bool { return true }

// MessageID returns the ID of the message to remove (empty means "all").
func (r RemoveMessage) MessageID() string { return r.ID }

// MessageUpdate is the element type of a message-list update: either a
// regular Message or a RemoveMessage sentinel. It is the Go analog of the
// mixed list[AnyMessage] that Python's add_messages consumes.
type MessageUpdate interface {
	isMessageUpdate()
}

func (Message) isMessageUpdate()       {}
func (RemoveMessage) isMessageUpdate() {}

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
		switch b := block.(type) {
		case TextBlock:
			out.WriteString(b.Text)
		case NonStandardContentBlock:
			// Handle legacy/provider text blocks with empty or "text" type.
			if b.Type == "" || b.Type == "text" {
				if text, ok := b.Value["text"].(string); ok {
					out.WriteString(text)
				}
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

// BufferStringXML renders messages using Python get_buffer_string's "xml"
// format: each message becomes a <message type="role">...</message> element,
// with AI tool calls nested under <content> and <tool_call> elements and all
// content escaped for XML.
func BufferStringXML(values []Message) string {
	lines := make([]string, 0, len(values))
	for _, message := range values {
		lines = append(lines, formatMessageXML(message))
	}
	return strings.Join(lines, "\n")
}

func formatMessageXML(message Message) string {
	msgType := strings.ToLower(string(message.Role))
	contentParts := xmlContentParts(message)

	if message.Role != RoleAI || len(message.ToolCalls) == 0 {
		return "<message type=" + quoteattr(msgType) + ">" +
			strings.Join(contentParts, " ") + "</message>"
	}

	parts := []string{"<message type=" + quoteattr(msgType) + ">"}
	if len(contentParts) > 0 {
		parts = append(parts, "  <content>"+strings.Join(contentParts, " ")+"</content>")
	}
	for _, tc := range message.ToolCalls {
		parts = append(parts, "  <tool_call id="+quoteattr(tc.ID)+" name="+quoteattr(tc.Name)+">"+
			escapeXML(toolCallArgsJSON(tc.Args))+"</tool_call>")
	}
	parts = append(parts, "</message>")
	return strings.Join(parts, "\n")
}

// xmlContentParts returns the escaped content parts for a message. String
// content is a single part; otherwise the supported content blocks are
// formatted in order (unknown blocks and base64 data blocks are skipped).
func xmlContentParts(message Message) []string {
	if message.Content != "" {
		return []string{escapeXML(message.Content)}
	}
	var parts []string
	for _, block := range message.ContentBlocks {
		if formatted := formatContentBlockXML(block); formatted != "" {
			parts = append(parts, formatted)
		}
	}
	return parts
}

// formatContentBlockXML mirrors Python's _format_content_block_xml: it renders
// the supported block types and returns "" for anything that should be
// skipped (unknown types and base64-encoded data).
func formatContentBlockXML(block ContentBlock) string {
	switch b := block.(type) {
	case TextBlock:
		if b.Text == "" {
			return ""
		}
		return escapeXML(b.Text)
	case ReasoningBlock:
		if b.Reasoning == "" {
			return ""
		}
		return "<reasoning>" + escapeXML(b.Reasoning) + "</reasoning>"
	case ImageBlock:
		if hasBase64Data(b.URL, b.Base64) {
			return ""
		}
		if b.URL != "" {
			return "<image url=" + quoteattr(b.URL) + " />"
		}
		if b.FileID != "" {
			return "<image file_id=" + quoteattr(b.FileID) + " />"
		}
		return ""
	case AudioBlock:
		if hasBase64Data(b.URL, b.Base64) {
			return ""
		}
		if b.URL != "" {
			return "<audio url=" + quoteattr(b.URL) + " />"
		}
		if b.FileID != "" {
			return "<audio file_id=" + quoteattr(b.FileID) + " />"
		}
		return ""
	case VideoBlock:
		if hasBase64Data(b.URL, b.Base64) {
			return ""
		}
		if b.URL != "" {
			return "<video url=" + quoteattr(b.URL) + " />"
		}
		if b.FileID != "" {
			return "<video file_id=" + quoteattr(b.FileID) + " />"
		}
		return ""
	case PlainTextBlock:
		if b.Text == "" {
			return ""
		}
		return escapeXML(truncateXML(b.Text))
	case ServerToolCall:
		return "<server_tool_call id=" + quoteattr(b.ID) + " name=" + quoteattr(b.Name) + ">" +
			escapeXML(truncateXML(toolCallArgsJSON(b.Args))) + "</server_tool_call>"
	case ServerToolResult:
		output := ""
		if b.Output != nil {
			output = escapeXML(truncateXML(anyToJSON(b.Output)))
		}
		return "<server_tool_result tool_call_id=" + quoteattr(b.ToolCallID) +
			" status=" + quoteattr(b.Status) + ">" + output + "</server_tool_result>"
	default:
		return ""
	}
}

func hasBase64Data(url, base64 string) bool {
	return base64 != "" || strings.HasPrefix(url, "data:")
}

func truncateXML(text string) string {
	const max = 500
	runes := []rune(text)
	if len(runes) <= max {
		return text
	}
	return string(runes[:max]) + "..."
}

// escapeXML escapes &, < and >, matching Python's xml.sax.saxutils.escape.
func escapeXML(text string) string {
	var b strings.Builder
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '&':
			b.WriteString("&amp;")
		case '<':
			b.WriteString("&lt;")
		case '>':
			b.WriteString("&gt;")
		default:
			b.WriteByte(text[i])
		}
	}
	return b.String()
}

// quoteattr escapes an XML attribute value and wraps it in double quotes,
// matching Python's xml.sax.saxutils.quoteattr.
func quoteattr(text string) string {
	escaped := escapeXML(text)
	return `"` + strings.ReplaceAll(escaped, `"`, "&quot;") + `"`
}

func toolCallArgsJSON(args map[string]any) string {
	if args == nil {
		return "{}"
	}
	return anyToJSON(args)
}

// anyToJSON returns compact JSON with HTML characters left unescaped,
// matching Python's json.dumps(..., ensure_ascii=False).
func anyToJSON(value any) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(value); err != nil {
		return "{}"
	}
	return strings.TrimSuffix(buf.String(), "\n")
}

// FilterOptions controls Filter.
type FilterOptions struct {
	IncludeRoles []Role
	ExcludeRoles []Role
	IncludeNames []string
	ExcludeNames []string
	IncludeIDs   []string
	ExcludeIDs   []string
	// ExcludeToolCalls, when true, removes tool_call content blocks from
	// AIMessage content while keeping every non-tool-call block.
	ExcludeToolCalls bool
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
		cloned := Clone(message)
		if opts.ExcludeToolCalls {
			cloned = excludeToolCallBlocks(cloned)
		}
		out = append(out, cloned)
	}
	return out
}

// excludeToolCallBlocks removes tool_call content blocks from an AIMessage's
// ContentBlocks, keeping every other block. Non-AI messages are returned
// unchanged. It mirrors the content-block pruning Python's
// filter_messages(..., exclude_tool_calls=True) performs, applied at the
// content-block level for the Go port (which stores tool calls both as legacy
// ToolCalls and as tool_call content blocks).
func excludeToolCallBlocks(message Message) Message {
	if message.Role != RoleAI || len(message.ContentBlocks) == 0 {
		return message
	}
	kept := make([]ContentBlock, 0, len(message.ContentBlocks))
	for _, block := range message.ContentBlocks {
		if block.BlockType() == "tool_call" {
			continue
		}
		kept = append(kept, block)
	}
	message.ContentBlocks = kept
	return message
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
	return message.MarshalJSON()
}

// UnmarshalJSONStable decodes a message serialized by MarshalJSONStable.
func UnmarshalJSONStable(data []byte) (Message, error) {
	var message Message
	err := message.UnmarshalJSON(data)
	return message, err
}

// MarshalJSON implements json.Marshaler. ContentBlocks (an interface slice)
// are converted to maps via BlockToMap so the wire format is unchanged from
// the old map[string]any representation.
func (m Message) MarshalJSON() ([]byte, error) {
	type alias Message // prevent recursion
	tmp := struct {
		alias
		ContentBlocks []map[string]any `json:"content_blocks,omitempty"`
	}{}
	tmp.alias = alias(m)
	if len(m.ContentBlocks) > 0 {
		tmp.ContentBlocks = make([]map[string]any, len(m.ContentBlocks))
		for i, b := range m.ContentBlocks {
			tmp.ContentBlocks[i] = BlockToMap(b)
		}
	}
	return json.Marshal(tmp)
}

// UnmarshalJSON implements json.Unmarshaler. ContentBlocks are decoded as
// raw maps and then normalized via ParseContentBlock.
func (m *Message) UnmarshalJSON(data []byte) error {
	type alias Message
	tmp := struct {
		alias
		ContentBlocks []map[string]any `json:"content_blocks,omitempty"`
	}{}
	if err := json.Unmarshal(data, &tmp); err != nil {
		return err
	}
	*m = Message(tmp.alias)
	if len(tmp.ContentBlocks) > 0 {
		m.ContentBlocks = make([]ContentBlock, len(tmp.ContentBlocks))
		for i, raw := range tmp.ContentBlocks {
			m.ContentBlocks[i] = ParseContentBlock(raw)
		}
	}
	return nil
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
		out[i] = CloneBlock(block)
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

// cloneAny deep-copies the common mutable any-typed fields (map[string]any).
// Scalar and other values are returned as-is, matching the shallow-clone depth
// used elsewhere in the package.
func cloneAny(value any) any {
	if m, ok := value.(map[string]any); ok {
		return cloneMap(m)
	}
	return value
}
