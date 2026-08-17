package prompts

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestStructuredPromptFromParts(t *testing.T) {
	human, err := NewChatMessageTemplate(messages.RoleHuman, "human", "Extract {{.thing}}")
	if err != nil {
		t.Fatal(err)
	}
	outSchema := schema.Object(map[string]schema.Schema{"name": schema.String("name")}, "name")
	prompt, err := NewStructuredPromptFromParts(outSchema, map[string]any{"method": "json_schema"}, human)
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := prompt.FormatMessages(map[string]any{"thing": "Ada"})
	if err != nil {
		t.Fatal(err)
	}
	if len(rendered) != 1 || rendered[0].Content != "Extract Ada" {
		t.Fatalf("messages: %#v", rendered)
	}
	if prompt.OutputSchema()["type"] != "object" {
		t.Fatalf("schema: %#v", prompt.OutputSchema())
	}
}

func TestStructuredPromptFromPartsRequiresSchema(t *testing.T) {
	human, err := NewChatMessageTemplate(messages.RoleHuman, "human", "Hi")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewStructuredPromptFromParts(nil, nil, human); err == nil {
		t.Fatal("expected missing schema error")
	}
}
