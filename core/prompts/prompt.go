package prompts

import (
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"text/template"

	"github.com/projanvil/langchain-golang/core/chathistory"
	"github.com/projanvil/langchain-golang/core/messages"
)

// PromptTemplate renders a string prompt from variables.
type PromptTemplate struct {
	name      string
	template  string
	parsed    *template.Template
	partials  map[string]any
	variables []string
}

// NewPromptTemplate creates a text/template-backed prompt template.
func NewPromptTemplate(name string, templateText string) (PromptTemplate, error) {
	return NewPromptTemplateWithPartials(name, templateText, nil)
}

// NewPromptTemplateWithPartials creates a prompt template with pre-bound
// partial variables.
func NewPromptTemplateWithPartials(name string, templateText string, partials map[string]any) (PromptTemplate, error) {
	if name == "" {
		name = "prompt"
	}
	parsed, err := template.New(name).Option("missingkey=error").Parse(templateText)
	if err != nil {
		return PromptTemplate{}, err
	}
	copiedPartials := cloneMapAny(partials)
	return PromptTemplate{
		name:      name,
		template:  templateText,
		parsed:    parsed,
		partials:  copiedPartials,
		variables: templateVariables(templateText, copiedPartials),
	}, nil
}

// Format renders the prompt with variables.
func (p PromptTemplate) Format(values map[string]any) (string, error) {
	merged := p.mergePartialAndUserVariables(values)
	var buffer bytes.Buffer
	if err := p.parsed.Execute(&buffer, merged); err != nil {
		return "", fmt.Errorf("format prompt %q: %w", p.name, err)
	}
	return buffer.String(), nil
}

// FormatPrompt renders the prompt as a prompt value.
func (p PromptTemplate) FormatPrompt(values map[string]any) (StringPromptValue, error) {
	text, err := p.Format(values)
	if err != nil {
		return StringPromptValue{}, err
	}
	return StringPromptValue{Text: text}, nil
}

// Template returns the original template text.
func (p PromptTemplate) Template() string {
	return p.template
}

// InputVariables returns variables required from the caller after applying
// partial variables.
func (p PromptTemplate) InputVariables() []string {
	return append([]string(nil), p.variables...)
}

// Partial returns a copy of the prompt with additional partial variables.
func (p PromptTemplate) Partial(values map[string]any) (PromptTemplate, error) {
	partials := cloneMapAny(p.partials)
	if partials == nil {
		partials = map[string]any{}
	}
	for key, value := range values {
		partials[key] = value
	}
	return NewPromptTemplateWithPartials(p.name, p.template, partials)
}

// Validate checks that the expected input variables match the template after
// partial variables are applied.
func (p PromptTemplate) Validate(expected []string) error {
	got := p.InputVariables()
	sort.Strings(got)
	want := append([]string(nil), expected...)
	sort.Strings(want)
	if len(got) != len(want) {
		return fmt.Errorf("prompt variables mismatch: got %v want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("prompt variables mismatch: got %v want %v", got, want)
		}
	}
	return nil
}

func (p PromptTemplate) mergePartialAndUserVariables(values map[string]any) map[string]any {
	merged := cloneMapAny(p.partials)
	if merged == nil {
		merged = map[string]any{}
	}
	for key, value := range merged {
		if fn, ok := value.(func() any); ok {
			merged[key] = fn()
		}
	}
	for key, value := range values {
		merged[key] = value
	}
	return merged
}

// ChatPromptPart renders zero or more chat messages.
type ChatPromptPart interface {
	FormatMessages(values map[string]any) ([]messages.Message, error)
}

// ChatMessageTemplate renders one chat message.
type ChatMessageTemplate struct {
	Role   messages.Role
	Prompt PromptTemplate
}

// NewChatMessageTemplate creates a chat message template.
func NewChatMessageTemplate(
	role messages.Role,
	name string,
	templateText string,
) (ChatMessageTemplate, error) {
	prompt, err := NewPromptTemplate(name, templateText)
	if err != nil {
		return ChatMessageTemplate{}, err
	}
	return ChatMessageTemplate{
		Role:   role,
		Prompt: prompt,
	}, nil
}

// NewChatMessageTemplateWithPartials creates a chat message template with
// pre-bound partial variables.
func NewChatMessageTemplateWithPartials(
	role messages.Role,
	name string,
	templateText string,
	partials map[string]any,
) (ChatMessageTemplate, error) {
	prompt, err := NewPromptTemplateWithPartials(name, templateText, partials)
	if err != nil {
		return ChatMessageTemplate{}, err
	}
	return ChatMessageTemplate{
		Role:   role,
		Prompt: prompt,
	}, nil
}

// Format renders the chat message.
func (t ChatMessageTemplate) Format(values map[string]any) (messages.Message, error) {
	content, err := t.Prompt.Format(values)
	if err != nil {
		return messages.Message{}, err
	}
	return messages.Message{
		Role:    t.Role,
		Content: content,
	}, nil
}

// FormatMessages renders the chat message as a single-message slice.
func (t ChatMessageTemplate) FormatMessages(values map[string]any) ([]messages.Message, error) {
	message, err := t.Format(values)
	if err != nil {
		return nil, err
	}
	return []messages.Message{message}, nil
}

// MessageContentPartTemplate renders one content block inside a rich chat
// message template.
type MessageContentPartTemplate interface {
	InputVariables() []string
	FormatContentBlock(values map[string]any) (messages.ContentBlock, bool, error)
}

// TextContentTemplate renders a text content block.
type TextContentTemplate struct {
	Prompt PromptTemplate
}

// NewTextContentTemplate creates a text content template.
func NewTextContentTemplate(name string, templateText string) (TextContentTemplate, error) {
	prompt, err := NewPromptTemplate(name, templateText)
	if err != nil {
		return TextContentTemplate{}, err
	}
	return TextContentTemplate{Prompt: prompt}, nil
}

func (t TextContentTemplate) InputVariables() []string {
	return t.Prompt.InputVariables()
}

func (t TextContentTemplate) FormatContentBlock(values map[string]any) (messages.ContentBlock, bool, error) {
	text, err := t.Prompt.Format(values)
	if err != nil {
		return nil, false, err
	}
	if text == "" {
		return nil, false, nil
	}
	return messages.ContentBlock{"type": "text", "text": text}, true, nil
}

// ImagePromptTemplate renders image_url content.
type ImagePromptTemplate struct {
	Template map[string]any
}

// NewImagePromptTemplate creates an image prompt template. Template string
// values may use the same Go text/template syntax as PromptTemplate.
func NewImagePromptTemplate(template map[string]any) (ImagePromptTemplate, error) {
	if _, ok := template["path"]; ok {
		return ImagePromptTemplate{}, fmt.Errorf("image prompt template path is not supported; use url")
	}
	if _, ok := template["url"]; !ok {
		return ImagePromptTemplate{}, fmt.Errorf("image prompt template requires url")
	}
	for key, value := range template {
		if text, ok := value.(string); ok {
			if _, err := NewPromptTemplate("image-"+key, text); err != nil {
				return ImagePromptTemplate{}, err
			}
		}
	}
	return ImagePromptTemplate{Template: cloneMapAny(template)}, nil
}

func (t ImagePromptTemplate) InputVariables() []string {
	seen := map[string]bool{}
	for _, value := range t.Template {
		text, ok := value.(string)
		if !ok {
			continue
		}
		for _, variable := range templateVariables(text, nil) {
			seen[variable] = true
		}
	}
	out := make([]string, 0, len(seen))
	for variable := range seen {
		out = append(out, variable)
	}
	sort.Strings(out)
	return out
}

// Format renders the image URL payload.
func (t ImagePromptTemplate) Format(values map[string]any) (map[string]any, error) {
	out := make(map[string]any, len(t.Template))
	for key, value := range t.Template {
		text, ok := value.(string)
		if !ok {
			out[key] = value
			continue
		}
		rendered, err := renderInlineTemplate("image-"+key, text, values)
		if err != nil {
			return nil, err
		}
		out[key] = rendered
	}
	return out, nil
}

func (t ImagePromptTemplate) FormatContentBlock(values map[string]any) (messages.ContentBlock, bool, error) {
	payload, err := t.Format(values)
	if err != nil {
		return nil, false, err
	}
	return messages.ContentBlock{"type": "image_url", "image_url": payload}, true, nil
}

// DictContentTemplate renders an arbitrary dictionary content block.
type DictContentTemplate struct {
	Prompt DictPromptTemplate
}

// NewDictContentTemplate creates a dictionary content template.
func NewDictContentTemplate(template map[string]any) DictContentTemplate {
	return DictContentTemplate{Prompt: NewDictPromptTemplate(template)}
}

func (t DictContentTemplate) InputVariables() []string {
	seen := map[string]bool{}
	collectTemplateVariables(t.Prompt.Template, seen)
	out := make([]string, 0, len(seen))
	for variable := range seen {
		out = append(out, variable)
	}
	sort.Strings(out)
	return out
}

func (t DictContentTemplate) FormatContentBlock(values map[string]any) (messages.ContentBlock, bool, error) {
	rendered, err := t.Prompt.Format(values)
	if err != nil {
		return nil, false, err
	}
	return messages.ContentBlock(rendered), true, nil
}

// RichChatMessageTemplate renders multimodal content blocks for one chat
// message.
type RichChatMessageTemplate struct {
	Role  messages.Role
	Parts []MessageContentPartTemplate
}

// NewRichChatMessageTemplate creates a multimodal chat message template.
func NewRichChatMessageTemplate(role messages.Role, parts ...MessageContentPartTemplate) RichChatMessageTemplate {
	return RichChatMessageTemplate{
		Role:  role,
		Parts: append([]MessageContentPartTemplate(nil), parts...),
	}
}

func (t RichChatMessageTemplate) Format(values map[string]any) (messages.Message, error) {
	blocks := make([]messages.ContentBlock, 0, len(t.Parts))
	for _, part := range t.Parts {
		block, ok, err := part.FormatContentBlock(values)
		if err != nil {
			return messages.Message{}, err
		}
		if ok {
			blocks = append(blocks, block)
		}
	}
	return messages.Message{Role: t.Role, ContentBlocks: blocks}, nil
}

func (t RichChatMessageTemplate) FormatMessages(values map[string]any) ([]messages.Message, error) {
	message, err := t.Format(values)
	if err != nil {
		return nil, err
	}
	return []messages.Message{message}, nil
}

// MessagesPlaceholder inserts a preformatted message list from input values.
type MessagesPlaceholder struct {
	VariableName string
	Optional     bool
	NMessages    int
}

// NewMessagesPlaceholder creates a chat message placeholder.
func NewMessagesPlaceholder(variableName string, optional bool, nMessages int) MessagesPlaceholder {
	return MessagesPlaceholder{
		VariableName: variableName,
		Optional:     optional,
		NMessages:    nMessages,
	}
}

// FormatMessages returns the provided message list, optionally truncated to the
// most recent N messages.
func (p MessagesPlaceholder) FormatMessages(values map[string]any) ([]messages.Message, error) {
	value, ok := values[p.VariableName]
	if !ok {
		if p.Optional {
			return nil, nil
		}
		return nil, fmt.Errorf("missing messages placeholder variable %q", p.VariableName)
	}
	out, err := convertToMessages(value)
	if err != nil {
		return nil, fmt.Errorf("format messages placeholder %q: %w", p.VariableName, err)
	}
	if p.NMessages > 0 && len(out) > p.NMessages {
		out = out[len(out)-p.NMessages:]
	}
	return out, nil
}

// ChatPromptTemplate renders a sequence of chat messages.
type ChatPromptTemplate struct {
	Messages []ChatMessageTemplate
	Parts    []ChatPromptPart
}

// NewChatPromptTemplate creates a chat prompt template.
func NewChatPromptTemplate(messageTemplates ...ChatMessageTemplate) ChatPromptTemplate {
	parts := make([]ChatPromptPart, len(messageTemplates))
	for i := range messageTemplates {
		parts[i] = messageTemplates[i]
	}
	return ChatPromptTemplate{
		Messages: append([]ChatMessageTemplate(nil), messageTemplates...),
		Parts:    parts,
	}
}

// NewChatPromptTemplateFromParts creates a chat prompt template from mixed
// message templates and placeholders.
func NewChatPromptTemplateFromParts(parts ...ChatPromptPart) ChatPromptTemplate {
	return ChatPromptTemplate{Parts: append([]ChatPromptPart(nil), parts...)}
}

// FormatMessages renders all messages in order.
func (p ChatPromptTemplate) FormatMessages(values map[string]any) ([]messages.Message, error) {
	if len(p.Parts) == 0 {
		out := make([]messages.Message, len(p.Messages))
		for i, messageTemplate := range p.Messages {
			message, err := messageTemplate.Format(values)
			if err != nil {
				return nil, err
			}
			out[i] = message
		}
		return out, nil
	}
	out := []messages.Message{}
	for _, part := range p.Parts {
		rendered, err := part.FormatMessages(values)
		if err != nil {
			return nil, err
		}
		out = append(out, rendered...)
	}
	return out, nil
}

// FormatPrompt renders all messages as a chat prompt value.
func (p ChatPromptTemplate) FormatPrompt(values map[string]any) (ChatPromptValue, error) {
	rendered, err := p.FormatMessages(values)
	if err != nil {
		return ChatPromptValue{}, err
	}
	return ChatPromptValue{Messages: rendered}, nil
}

func convertToMessages(value any) ([]messages.Message, error) {
	switch typed := value.(type) {
	case []messages.Message:
		return append([]messages.Message(nil), typed...), nil
	case messages.Message:
		return []messages.Message{typed}, nil
	case []any:
		out := make([]messages.Message, 0, len(typed))
		for _, item := range typed {
			message, err := convertOneMessage(item)
			if err != nil {
				return nil, err
			}
			out = append(out, message)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected a list of messages, got %T", value)
	}
}

func convertOneMessage(value any) (messages.Message, error) {
	switch typed := value.(type) {
	case messages.Message:
		return typed, nil
	case map[string]any:
		role, _ := typed["role"].(string)
		content, _ := typed["content"].(string)
		if role == "" {
			return messages.Message{}, fmt.Errorf("message map is missing role")
		}
		return messages.Message{Role: messages.Role(role), Content: content}, nil
	case []string:
		if len(typed) != 2 {
			return messages.Message{}, fmt.Errorf("message tuple should have role and content")
		}
		return messages.Message{Role: messages.Role(typed[0]), Content: typed[1]}, nil
	default:
		return messages.Message{}, fmt.Errorf("expected message, map, or role/content pair, got %T", value)
	}
}

func templateVariables(templateText string, partials map[string]any) []string {
	re := regexp.MustCompile(`{{\s*\.([A-Za-z_][A-Za-z0-9_]*)`)
	matches := re.FindAllStringSubmatch(templateText, -1)
	seen := map[string]bool{}
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		name := match[1]
		if _, partial := partials[name]; partial {
			continue
		}
		seen[name] = true
	}
	out := make([]string, 0, len(seen))
	for name := range seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func collectTemplateVariables(value any, seen map[string]bool) {
	switch typed := value.(type) {
	case string:
		for _, variable := range templateVariables(typed, nil) {
			seen[variable] = true
		}
	case map[string]any:
		for _, child := range typed {
			collectTemplateVariables(child, seen)
		}
	case []any:
		for _, child := range typed {
			collectTemplateVariables(child, seen)
		}
	}
}

// PromptValue is a formatted prompt that can be rendered as string or chat
// messages.
type PromptValue interface {
	ToString() string
	ToMessages() []messages.Message
}

// StringPromptValue is a formatted string prompt.
type StringPromptValue struct {
	Text string
}

// ToString returns the prompt text.
func (v StringPromptValue) ToString() string { return v.Text }

// ToMessages converts the prompt text to one human message.
func (v StringPromptValue) ToMessages() []messages.Message {
	return []messages.Message{messages.Human(v.Text)}
}

// ChatPromptValue is a formatted chat prompt.
type ChatPromptValue struct {
	Messages []messages.Message
}

// ToString returns a readable chat transcript.
func (v ChatPromptValue) ToString() string {
	return chathistory.BufferString(v.Messages)
}

// ToMessages returns a defensive copy of the chat messages.
func (v ChatPromptValue) ToMessages() []messages.Message {
	out := make([]messages.Message, len(v.Messages))
	copy(out, v.Messages)
	return out
}
