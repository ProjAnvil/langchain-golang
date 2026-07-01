package prompts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestLoadPromptFromConfig(t *testing.T) {
	loaded, err := LoadPromptFromConfig(map[string]any{
		"_type":    "prompt",
		"name":     "greeting",
		"template": "Hello {{.name}}",
	}, LoadPromptOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Format(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "Hello Ada" {
		t.Fatalf("Format = %q", got)
	}
}

func TestLoadPromptFromFileWithTemplatePath(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "template.txt"), []byte("Hi {{.name}}"), 0o600); err != nil {
		t.Fatal(err)
	}
	config := `{"_type":"prompt","template_path":"template.txt"}`
	path := filepath.Join(dir, "prompt.json")
	if err := os.WriteFile(path, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadPrompt(path, false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Format(map[string]any{"name": "Grace"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "Hi Grace" {
		t.Fatalf("Format = %q", got)
	}
}

func TestLoadPromptRejectsDangerousPaths(t *testing.T) {
	_, err := LoadPromptFromConfig(map[string]any{
		"_type":         "prompt",
		"template_path": "../template.txt",
	}, LoadPromptOptions{})
	if err == nil || !strings.Contains(err.Error(), "traversal") {
		t.Fatalf("expected traversal error, got %v", err)
	}
	_, err = LoadPrompt("lc://prompts/foo", false)
	if err == nil || !strings.Contains(err.Error(), "lc://") {
		t.Fatalf("expected lc error, got %v", err)
	}
}

func TestLoadFewShotPromptWithExamplesFile(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "examples.json"), []byte(`[{"q":"1+1","a":"2"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadPromptFromConfig(map[string]any{
		"_type": "few_shot",
		"example_prompt": map[string]any{
			"_type":    "prompt",
			"template": "Q: {{.q}}\nA: {{.a}}",
		},
		"examples":          "examples.json",
		"prefix":            "Answer.",
		"suffix":            "Q: {{.q}}\nA:",
		"example_separator": "\n---\n",
	}, LoadPromptOptions{BaseDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Format(map[string]any{"q": "2+2"})
	if err != nil {
		t.Fatal(err)
	}
	want := "Answer.\n---\nQ: 1+1\nA: 2\n---\nQ: 2+2\nA:"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestStructuredPrompt(t *testing.T) {
	human, err := NewChatMessageTemplate(messages.RoleHuman, "human", "Extract {{.thing}}")
	if err != nil {
		t.Fatal(err)
	}
	outSchema := schema.Object(map[string]schema.Schema{"name": schema.String("name")}, "name")
	prompt, err := NewStructuredPrompt(NewChatPromptTemplate(human), outSchema, map[string]any{"method": "json_schema"})
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
	copied := prompt.OutputSchema()
	copied["type"] = "changed"
	if prompt.OutputSchema()["type"] != "object" {
		t.Fatal("schema was not copied")
	}
	if prompt.StructuredOutputKwargs["method"] != "json_schema" {
		t.Fatalf("kwargs: %#v", prompt.StructuredOutputKwargs)
	}
}

func TestStructuredPromptRequiresSchema(t *testing.T) {
	_, err := NewStructuredPrompt(ChatPromptTemplate{}, nil, nil)
	if err == nil {
		t.Fatal("expected missing schema error")
	}
}
