package prompts

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestPromptTemplateFormat(t *testing.T) {
	prompt, err := NewPromptTemplate("greeting", "Hello {{.name}}")
	if err != nil {
		t.Fatalf("new prompt: %v", err)
	}

	got, err := prompt.Format(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if got != "Hello Ada" {
		t.Fatalf("prompt: got %q", got)
	}
}

func TestPromptTemplateMissingVariable(t *testing.T) {
	prompt, err := NewPromptTemplate("greeting", "Hello {{.name}}")
	if err != nil {
		t.Fatalf("new prompt: %v", err)
	}

	_, err = prompt.Format(map[string]any{})
	if err == nil {
		t.Fatal("expected missing variable error")
	}
}

func TestPromptTemplatePartials(t *testing.T) {
	prompt, err := NewPromptTemplateWithPartials("greeting", "Hello {{.name}} from {{.place}}", map[string]any{
		"place": "Go",
	})
	if err != nil {
		t.Fatalf("new prompt: %v", err)
	}
	if got := prompt.InputVariables(); len(got) != 1 || got[0] != "name" {
		t.Fatalf("input variables: %#v", got)
	}
	rendered, err := prompt.Format(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if rendered != "Hello Ada from Go" {
		t.Fatalf("rendered: %q", rendered)
	}
	rendered, err = prompt.Format(map[string]any{"name": "Ada", "place": "user"})
	if err != nil {
		t.Fatalf("format override: %v", err)
	}
	if rendered != "Hello Ada from user" {
		t.Fatalf("override rendered: %q", rendered)
	}
}

func TestPromptTemplatePartialAddsValues(t *testing.T) {
	prompt, err := NewPromptTemplate("greeting", "Hello {{.name}} from {{.place}}")
	if err != nil {
		t.Fatalf("new prompt: %v", err)
	}
	partial, err := prompt.Partial(map[string]any{"place": "partial"})
	if err != nil {
		t.Fatalf("partial: %v", err)
	}
	if got := partial.InputVariables(); len(got) != 1 || got[0] != "name" {
		t.Fatalf("input variables: %#v", got)
	}
	rendered, err := partial.Format(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatalf("format partial: %v", err)
	}
	if rendered != "Hello Ada from partial" {
		t.Fatalf("rendered: %q", rendered)
	}
}

func TestPromptTemplateCallablePartial(t *testing.T) {
	prompt, err := NewPromptTemplateWithPartials("dynamic", "Now {{.timestamp}}", map[string]any{
		"timestamp": func() any { return "t1" },
	})
	if err != nil {
		t.Fatalf("new prompt: %v", err)
	}
	rendered, err := prompt.Format(nil)
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if rendered != "Now t1" {
		t.Fatalf("rendered: %q", rendered)
	}
}

func TestPromptTemplateValidate(t *testing.T) {
	prompt, err := NewPromptTemplateWithPartials("greeting", "Hello {{.name}} from {{.place}}", map[string]any{
		"place": "Go",
	})
	if err != nil {
		t.Fatalf("new prompt: %v", err)
	}
	if err := prompt.Validate([]string{"name"}); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if err := prompt.Validate([]string{"name", "place"}); err == nil {
		t.Fatal("expected validate mismatch")
	}
}

func TestChatPromptTemplateFormatMessages(t *testing.T) {
	system, err := NewChatMessageTemplate(messages.RoleSystem, "system", "Be {{.style}}")
	if err != nil {
		t.Fatalf("new system template: %v", err)
	}
	human, err := NewChatMessageTemplate(messages.RoleHuman, "human", "{{.question}}")
	if err != nil {
		t.Fatalf("new human template: %v", err)
	}

	prompt := NewChatPromptTemplate(system, human)
	got, err := prompt.FormatMessages(map[string]any{
		"style":    "concise",
		"question": "What is Go?",
	})
	if err != nil {
		t.Fatalf("format messages: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("messages: got %d want 2", len(got))
	}
	if got[0].Role != messages.RoleSystem || got[0].Content != "Be concise" {
		t.Fatalf("system message: %+v", got[0])
	}
	if got[1].Role != messages.RoleHuman || got[1].Content != "What is Go?" {
		t.Fatalf("human message: %+v", got[1])
	}
}

func TestMessagesPlaceholderRequired(t *testing.T) {
	placeholder := NewMessagesPlaceholder("history", false, 0)

	got, err := placeholder.FormatMessages(map[string]any{
		"history": []messages.Message{
			messages.Human("hi"),
			messages.AI("hello"),
		},
	})
	if err != nil {
		t.Fatalf("format messages: %v", err)
	}
	if len(got) != 2 || got[0].Content != "hi" || got[1].Role != messages.RoleAI {
		t.Fatalf("messages: %#v", got)
	}

	_, err = placeholder.FormatMessages(map[string]any{})
	if err == nil {
		t.Fatal("expected missing placeholder variable error")
	}
}

func TestMessagesPlaceholderOptionalAndLimit(t *testing.T) {
	optional := NewMessagesPlaceholder("history", true, 0)
	got, err := optional.FormatMessages(map[string]any{})
	if err != nil {
		t.Fatalf("optional format messages: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("optional messages: %#v", got)
	}

	limited := NewMessagesPlaceholder("history", false, 1)
	got, err = limited.FormatMessages(map[string]any{
		"history": []any{
			map[string]any{"role": "system", "content": "You are helpful."},
			[]string{"human", "Hello"},
		},
	})
	if err != nil {
		t.Fatalf("limited format messages: %v", err)
	}
	if len(got) != 1 || got[0].Role != messages.RoleHuman || got[0].Content != "Hello" {
		t.Fatalf("limited messages: %#v", got)
	}
}

func TestChatPromptTemplateWithMessagesPlaceholder(t *testing.T) {
	system, err := NewChatMessageTemplate(messages.RoleSystem, "system", "You are helpful.")
	if err != nil {
		t.Fatalf("new system template: %v", err)
	}
	human, err := NewChatMessageTemplate(messages.RoleHuman, "human", "{{.question}}")
	if err != nil {
		t.Fatalf("new human template: %v", err)
	}
	prompt := NewChatPromptTemplateFromParts(
		system,
		NewMessagesPlaceholder("history", false, 0),
		human,
	)

	got, err := prompt.FormatPrompt(map[string]any{
		"history": []messages.Message{
			messages.Human("what's 5 + 2"),
			messages.AI("5 + 2 is 7"),
		},
		"question": "now multiply that by 4",
	})
	if err != nil {
		t.Fatalf("format prompt: %v", err)
	}
	rendered := got.ToMessages()
	if len(rendered) != 4 {
		t.Fatalf("messages: %#v", rendered)
	}
	if rendered[0].Role != messages.RoleSystem || rendered[1].Content != "what's 5 + 2" || rendered[3].Content != "now multiply that by 4" {
		t.Fatalf("messages: %#v", rendered)
	}
	if got.ToString() == "" {
		t.Fatal("expected chat prompt value string")
	}
}

func TestImagePromptTemplateFormat(t *testing.T) {
	template, err := NewImagePromptTemplate(map[string]any{
		"url":    "https://example.com/{{.image_id}}.png",
		"detail": "{{.detail}}",
	})
	if err != nil {
		t.Fatalf("new image prompt: %v", err)
	}
	if got := template.InputVariables(); len(got) != 2 || got[0] != "detail" || got[1] != "image_id" {
		t.Fatalf("input variables: %#v", got)
	}
	formatted, err := template.Format(map[string]any{
		"image_id": "cat",
		"detail":   "high",
	})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if formatted["url"] != "https://example.com/cat.png" || formatted["detail"] != "high" {
		t.Fatalf("formatted: %#v", formatted)
	}
}

func TestImagePromptTemplateRejectsPath(t *testing.T) {
	_, err := NewImagePromptTemplate(map[string]any{"path": "/tmp/image.png"})
	if err == nil {
		t.Fatal("expected path error")
	}
}

func TestRichChatMessageTemplateFormat(t *testing.T) {
	textPart, err := NewTextContentTemplate("text", "Describe {{.subject}}")
	if err != nil {
		t.Fatalf("new text part: %v", err)
	}
	imagePart, err := NewImagePromptTemplate(map[string]any{
		"url":    "https://example.com/{{.image}}.png",
		"detail": "low",
	})
	if err != nil {
		t.Fatalf("new image part: %v", err)
	}
	dictPart := NewDictContentTemplate(map[string]any{
		"type": "input_audio",
		"id":   "{{.audio_id}}",
	})
	template := NewRichChatMessageTemplate(messages.RoleHuman, textPart, imagePart, dictPart)

	message, err := template.Format(map[string]any{
		"subject":  "this image",
		"image":    "diagram",
		"audio_id": "a1",
	})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if message.Role != messages.RoleHuman || len(message.ContentBlocks) != 3 {
		t.Fatalf("message: %#v", message)
	}
	if message.ContentBlocks[0]["type"] != "text" || message.ContentBlocks[0]["text"] != "Describe this image" {
		t.Fatalf("text block: %#v", message.ContentBlocks[0])
	}
	image := message.ContentBlocks[1]["image_url"].(map[string]any)
	if message.ContentBlocks[1]["type"] != "image_url" || image["url"] != "https://example.com/diagram.png" {
		t.Fatalf("image block: %#v", message.ContentBlocks[1])
	}
	if message.ContentBlocks[2]["type"] != "input_audio" || message.ContentBlocks[2]["id"] != "a1" {
		t.Fatalf("dict block: %#v", message.ContentBlocks[2])
	}
}

func TestChatPromptTemplateWithRichMessage(t *testing.T) {
	system, err := NewChatMessageTemplate(messages.RoleSystem, "system", "Be concise.")
	if err != nil {
		t.Fatalf("new system: %v", err)
	}
	imagePart, err := NewImagePromptTemplate(map[string]any{"url": "{{.url}}"})
	if err != nil {
		t.Fatalf("new image part: %v", err)
	}
	prompt := NewChatPromptTemplateFromParts(
		system,
		NewRichChatMessageTemplate(messages.RoleHuman, imagePart),
	)
	got, err := prompt.FormatMessages(map[string]any{"url": "https://example.com/x.png"})
	if err != nil {
		t.Fatalf("format messages: %v", err)
	}
	if len(got) != 2 || got[1].ContentBlocks[0]["type"] != "image_url" {
		t.Fatalf("messages: %#v", got)
	}
}

func TestPromptTemplateFormatPrompt(t *testing.T) {
	prompt, err := NewPromptTemplate("greeting", "Hello {{.name}}")
	if err != nil {
		t.Fatalf("new prompt: %v", err)
	}
	got, err := prompt.FormatPrompt(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatalf("format prompt: %v", err)
	}
	if got.ToString() != "Hello Ada" {
		t.Fatalf("string: %q", got.ToString())
	}
	rendered := got.ToMessages()
	if len(rendered) != 1 || rendered[0].Role != messages.RoleHuman || rendered[0].Content != "Hello Ada" {
		t.Fatalf("messages: %#v", rendered)
	}
}
