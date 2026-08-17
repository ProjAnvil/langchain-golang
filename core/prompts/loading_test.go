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

func TestLoadPromptFileErrors(t *testing.T) {
	if _, err := LoadPrompt(filepath.Join(t.TempDir(), "missing.json"), false); err == nil {
		t.Fatal("expected missing file error")
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPrompt(path, false); err == nil {
		t.Fatal("expected invalid JSON error")
	}
}

func TestLoadPromptFromConfigTypes(t *testing.T) {
	if _, err := LoadPromptFromConfig(map[string]any{"_type": "unknown"}, LoadPromptOptions{}); err == nil {
		t.Fatal("expected unsupported type error")
	}

	loaded, err := LoadPromptFromConfig(map[string]any{"template": "Hi {{.name}}"}, LoadPromptOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Format(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "Hi Ada" {
		t.Fatalf("Format = %q", got)
	}
}

func TestLoadPromptTemplateValidation(t *testing.T) {
	if _, err := LoadPromptFromConfig(map[string]any{
		"_type":           "prompt",
		"template":        "Hi",
		"template_format": "jinja2",
	}, LoadPromptOptions{}); err == nil {
		t.Fatal("expected jinja2 error")
	}

	if _, err := LoadPromptFromConfig(map[string]any{"_type": "prompt"}, LoadPromptOptions{}); err == nil {
		t.Fatal("expected missing template error")
	}

	if _, err := LoadPromptFromConfig(map[string]any{
		"_type":         "prompt",
		"template":      "Hi",
		"template_path": "template.txt",
	}, LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "both") {
		t.Fatalf("expected both path and value error, got %v", err)
	}

	if _, err := LoadPromptFromConfig(map[string]any{
		"_type":         "prompt",
		"template_path": "template.md",
	}, LoadPromptOptions{AllowDangerousPaths: true}); err == nil || !strings.Contains(err.Error(), "unsupported template file format") {
		t.Fatalf("expected extension error, got %v", err)
	}

	dir := t.TempDir()
	if _, err := LoadPromptFromConfig(map[string]any{
		"_type":         "prompt",
		"template_path": "missing.txt",
	}, LoadPromptOptions{BaseDir: dir}); err == nil {
		t.Fatal("expected missing template file error")
	}
}

func TestLoadPromptTemplatePartials(t *testing.T) {
	loaded, err := LoadPromptFromConfig(map[string]any{
		"_type":    "prompt",
		"template": "Hello {{.name}} from {{.place}}",
		"partial_variables": map[string]any{
			"place": "Go",
		},
	}, LoadPromptOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Format(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "Hello Ada from Go" {
		t.Fatalf("Format = %q", got)
	}
}

func TestLoadPromptAbsolutePaths(t *testing.T) {
	dir := t.TempDir()
	templatePath := filepath.Join(dir, "template.txt")
	if err := os.WriteFile(templatePath, []byte("Hi {{.name}}"), 0o600); err != nil {
		t.Fatal(err)
	}
	config := map[string]any{
		"_type":         "prompt",
		"template_path": templatePath,
	}
	if _, err := LoadPromptFromConfig(config, LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "absolute") {
		t.Fatalf("expected absolute path error, got %v", err)
	}
	loaded, err := LoadPromptFromConfig(config, LoadPromptOptions{AllowDangerousPaths: true})
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Format(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "Hi Ada" {
		t.Fatalf("Format = %q", got)
	}
}

func fewShotConfig(overrides map[string]any) map[string]any {
	config := map[string]any{
		"_type": "few_shot",
		"example_prompt": map[string]any{
			"_type":    "prompt",
			"template": "Q: {{.q}}\nA: {{.a}}",
		},
		"examples": []map[string]any{{"q": "1+1", "a": "2"}},
		"prefix":   "Answer.",
		"suffix":   "Q: {{.q}}\nA:",
	}
	for key, value := range overrides {
		config[key] = value
	}
	return config
}

func TestLoadFewShotPromptWithInlineExamples(t *testing.T) {
	loaded, err := LoadPromptFromConfig(fewShotConfig(nil), LoadPromptOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Format(map[string]any{"q": "2+2"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "Q: 1+1\nA: 2") {
		t.Fatalf("Format = %q", got)
	}
}

func TestLoadFewShotPromptTemplatePathErrors(t *testing.T) {
	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"prefix_path": "prefix.txt",
	}), LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "both") {
		t.Fatalf("expected prefix path conflict, got %v", err)
	}
	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"prefix":      nil,
		"suffix_path": "suffix.txt",
	}), LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "both") {
		t.Fatalf("expected suffix path conflict, got %v", err)
	}
}

func TestLoadFewShotPromptExamplePromptErrors(t *testing.T) {
	config := fewShotConfig(nil)
	delete(config, "example_prompt")
	if _, err := LoadPromptFromConfig(config, LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "example_prompt") {
		t.Fatalf("expected example_prompt required error, got %v", err)
	}

	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"example_prompt": map[string]any{"_type": "prompt", "template": "{{.q"},
	}), LoadPromptOptions{}); err == nil {
		t.Fatal("expected example_prompt load error")
	}

	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"example_prompt": map[string]any{
			"_type":         "few_shot",
			"example_prompt": map[string]any{"_type": "prompt", "template": "{{.q}}"},
			"examples":       []any{map[string]any{"q": "1"}},
		},
	}), LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "must be a prompt template") {
		t.Fatalf("expected prompt template type error, got %v", err)
	}
}

func TestLoadFewShotPromptExampleErrors(t *testing.T) {
	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"examples": 42,
	}), LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "invalid examples format") {
		t.Fatalf("expected invalid examples format error, got %v", err)
	}

	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"examples": []any{"nope"},
	}), LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "must be an object") {
		t.Fatalf("expected example object error, got %v", err)
	}

	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"examples": "examples.txt",
	}), LoadPromptOptions{}); err == nil || !strings.Contains(err.Error(), "only json is supported") {
		t.Fatalf("expected examples extension error, got %v", err)
	}

	dir := t.TempDir()
	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"examples": "missing.json",
	}), LoadPromptOptions{BaseDir: dir}); err == nil {
		t.Fatal("expected missing examples file error")
	}

	if err := os.WriteFile(filepath.Join(dir, "bad.json"), []byte("["), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPromptFromConfig(fewShotConfig(map[string]any{
		"examples": "bad.json",
	}), LoadPromptOptions{BaseDir: dir}); err == nil {
		t.Fatal("expected invalid examples JSON error")
	}
}
