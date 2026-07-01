package prompts

import "testing"

func TestFewShotPromptTemplate(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "Q: {{.question}}\nA: {{.answer}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	prompt, err := NewFewShotPromptTemplate(
		[]map[string]any{{"question": "1+1?", "answer": "2"}},
		nil,
		examplePrompt,
		"Answer briefly.",
		"Q: {{.question}}\nA:",
		"\n---\n",
	)
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}

	got, err := prompt.Format(map[string]any{"question": "2+2?"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	want := "Answer briefly.\n---\nQ: 1+1?\nA: 2\n---\nQ: 2+2?\nA:"
	if got != want {
		t.Fatalf("prompt:\ngot  %q\nwant %q", got, want)
	}
}

func TestFewShotPromptTemplateSelector(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.input}} -> {{.output}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	prompt, err := NewFewShotPromptTemplate(nil, staticSelector{}, examplePrompt, "", "{{.input}} ->", "\n")
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	got, err := prompt.Format(map[string]any{"input": "b"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if got != "a -> A\nb ->" {
		t.Fatalf("got %q", got)
	}
}

func TestFewShotPromptTemplateInvalid(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.input}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	_, err = NewFewShotPromptTemplate(nil, nil, examplePrompt, "", "", "")
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestFewShotPromptWithTemplatesPythonFixture(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.question}}: {{.answer}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	prefix, err := NewPromptTemplate("prefix", "This is a test about {{.content}}.")
	if err != nil {
		t.Fatalf("new prefix: %v", err)
	}
	suffix, err := NewPromptTemplate("suffix", "Now you try to talk about {{.new_content}}.")
	if err != nil {
		t.Fatalf("new suffix: %v", err)
	}
	prompt, err := NewFewShotPromptWithTemplates(
		[]map[string]any{
			{"question": "foo", "answer": "bar"},
			{"question": "baz", "answer": "foo"},
		},
		nil,
		examplePrompt,
		&prefix,
		suffix,
		"\n",
		[]string{"content", "new_content"},
		true,
	)
	if err != nil {
		t.Fatalf("new few-shot with templates: %v", err)
	}
	got, err := prompt.Format(map[string]any{"content": "animals", "new_content": "party"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	want := "This is a test about animals.\nfoo: bar\nbaz: foo\nNow you try to talk about party."
	if got != want {
		t.Fatalf("prompt:\ngot  %q\nwant %q", got, want)
	}
}

func TestFewShotPromptWithTemplatesValidation(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.question}}: {{.answer}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	prefix, err := NewPromptTemplate("prefix", "This is a test about {{.content}}.")
	if err != nil {
		t.Fatalf("new prefix: %v", err)
	}
	suffix, err := NewPromptTemplate("suffix", "Now you try to talk about {{.new_content}}.")
	if err != nil {
		t.Fatalf("new suffix: %v", err)
	}
	_, err = NewFewShotPromptWithTemplates(
		[]map[string]any{{"question": "foo", "answer": "bar"}},
		nil,
		examplePrompt,
		&prefix,
		suffix,
		"\n",
		nil,
		true,
	)
	if err == nil {
		t.Fatal("expected validation error")
	}
	prompt, err := NewFewShotPromptWithTemplates(
		[]map[string]any{{"question": "foo", "answer": "bar"}},
		nil,
		examplePrompt,
		&prefix,
		suffix,
		"\n",
		nil,
		false,
	)
	if err != nil {
		t.Fatalf("new prompt: %v", err)
	}
	if len(prompt.InputVariables) != 2 || prompt.InputVariables[0] != "content" || prompt.InputVariables[1] != "new_content" {
		t.Fatalf("input variables: %#v", prompt.InputVariables)
	}
}

func TestDictPromptTemplate(t *testing.T) {
	prompt := NewDictPromptTemplate(map[string]any{
		"type": "text",
		"text": "Hello {{.name}}",
		"metadata": map[string]any{
			"source": "{{.source}}",
		},
		"items": []any{"{{.name}}", 42},
	})
	got, err := prompt.Format(map[string]any{"name": "Ada", "source": "docs"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if got["text"] != "Hello Ada" {
		t.Fatalf("text: %#v", got)
	}
	metadata := got["metadata"].(map[string]any)
	if metadata["source"] != "docs" {
		t.Fatalf("metadata: %#v", metadata)
	}
	items := got["items"].([]any)
	if items[0] != "Ada" || items[1] != 42 {
		t.Fatalf("items: %#v", items)
	}
}

type staticSelector struct{}

func (staticSelector) SelectExamples(map[string]any) ([]map[string]any, error) {
	return []map[string]any{{"input": "a", "output": "A"}}, nil
}
