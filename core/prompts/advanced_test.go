package prompts

import (
	"fmt"
	"testing"
)

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

func TestFewShotPromptTemplateRejectsBothExamplesAndSelector(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.input}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	_, err = NewFewShotPromptTemplate(
		[]map[string]any{{"input": "a"}},
		staticSelector{},
		examplePrompt,
		"",
		"",
		"",
	)
	if err == nil {
		t.Fatal("expected validation error for both examples and selector")
	}
}

func TestFewShotPromptTemplateDefaultSeparator(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.input}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	prompt, err := NewFewShotPromptTemplate(
		[]map[string]any{{"input": "a"}, {"input": "b"}},
		nil,
		examplePrompt,
		"",
		"{{.input}}",
		"",
	)
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if prompt.ExampleSeparator != "\n\n" {
		t.Fatalf("separator: %q", prompt.ExampleSeparator)
	}
	got, err := prompt.Format(map[string]any{"input": "c"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if got != "a\n\nb\n\nc" {
		t.Fatalf("got %q", got)
	}
}

type errorSelector struct{}

func (errorSelector) SelectExamples(map[string]any) ([]map[string]any, error) {
	return nil, fmt.Errorf("select failed")
}

func TestFewShotPromptTemplateSelectorError(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.input}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	prompt, err := NewFewShotPromptTemplate(nil, errorSelector{}, examplePrompt, "", "{{.input}}", "\n")
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if _, err := prompt.Format(map[string]any{"input": "x"}); err == nil {
		t.Fatal("expected selector error")
	}
}

func TestFewShotPromptTemplateFormatErrors(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.input}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	examples := []map[string]any{{"input": "a"}}

	badPrefix, err := NewFewShotPromptTemplate(examples, nil, examplePrompt, "{{.oops", "{{.input}}", "\n")
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if _, err := badPrefix.Format(map[string]any{"input": "x"}); err == nil {
		t.Fatal("expected prefix render error")
	}

	badSuffix, err := NewFewShotPromptTemplate(examples, nil, examplePrompt, "", "{{.oops", "\n")
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if _, err := badSuffix.Format(map[string]any{"input": "x"}); err == nil {
		t.Fatal("expected suffix render error")
	}

	twoVarExample, err := NewPromptTemplate("example", "{{.input}} {{.output}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	badExample, err := NewFewShotPromptTemplate(examples, nil, twoVarExample, "", "{{.input}}", "\n")
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if _, err := badExample.Format(map[string]any{"input": "x"}); err == nil {
		t.Fatal("expected example format error")
	}
}

func TestFewShotPromptWithTemplatesRejectsInvalidCombinations(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.input}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	suffix, err := NewPromptTemplate("suffix", "done")
	if err != nil {
		t.Fatalf("new suffix: %v", err)
	}
	_, err = NewFewShotPromptWithTemplates(
		[]map[string]any{{"input": "a"}},
		staticSelector{},
		examplePrompt,
		nil,
		suffix,
		"\n",
		nil,
		false,
	)
	if err == nil {
		t.Fatal("expected error for both examples and selector")
	}
	_, err = NewFewShotPromptWithTemplates(nil, nil, examplePrompt, nil, suffix, "\n", nil, false)
	if err == nil {
		t.Fatal("expected error for neither examples nor selector")
	}
}

func TestFewShotPromptWithTemplatesNilPrefixAndExtraValues(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.question}}: {{.answer}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	suffix, err := NewPromptTemplate("suffix", "Now {{.new_content}}.")
	if err != nil {
		t.Fatalf("new suffix: %v", err)
	}
	prompt, err := NewFewShotPromptWithTemplates(
		[]map[string]any{{"question": "foo", "answer": "bar"}},
		nil,
		examplePrompt,
		nil,
		suffix,
		"\n",
		nil,
		false,
	)
	if err != nil {
		t.Fatalf("new few-shot with templates: %v", err)
	}
	if len(prompt.InputVariables) != 1 || prompt.InputVariables[0] != "new_content" {
		t.Fatalf("input variables: %#v", prompt.InputVariables)
	}
	if prompt.Prefix != nil {
		t.Fatal("expected nil prefix")
	}
	got, err := prompt.Format(map[string]any{"new_content": "party", "extra": "leftover"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	if got != "foo: bar\nNow party." {
		t.Fatalf("got %q", got)
	}
}

func TestFewShotPromptWithTemplatesValidationSameLengthMismatch(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.question}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	prefix, err := NewPromptTemplate("prefix", "About {{.content}}.")
	if err != nil {
		t.Fatalf("new prefix: %v", err)
	}
	suffix, err := NewPromptTemplate("suffix", "Now {{.new_content}}.")
	if err != nil {
		t.Fatalf("new suffix: %v", err)
	}
	_, err = NewFewShotPromptWithTemplates(
		[]map[string]any{{"question": "foo"}},
		nil,
		examplePrompt,
		&prefix,
		suffix,
		"\n",
		[]string{"content", "wrong"},
		true,
	)
	if err == nil {
		t.Fatal("expected validation error for same-length mismatch")
	}
}

func TestFewShotPromptWithTemplatesFormatErrors(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.question}}: {{.answer}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	prefix, err := NewPromptTemplate("prefix", "About {{.content}}.")
	if err != nil {
		t.Fatalf("new prefix: %v", err)
	}
	suffix, err := NewPromptTemplate("suffix", "Now {{.new_content}}.")
	if err != nil {
		t.Fatalf("new suffix: %v", err)
	}
	examples := []map[string]any{{"question": "foo", "answer": "bar"}}

	selectorPrompt, err := NewFewShotPromptWithTemplates(nil, errorSelector{}, examplePrompt, &prefix, suffix, "\n", nil, false)
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if _, err := selectorPrompt.Format(nil); err == nil {
		t.Fatal("expected selector error")
	}

	prompt, err := NewFewShotPromptWithTemplates(examples, nil, examplePrompt, &prefix, suffix, "\n", nil, false)
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if _, err := prompt.Format(map[string]any{"new_content": "party"}); err == nil {
		t.Fatal("expected prefix format error")
	}
	if _, err := prompt.Format(map[string]any{"content": "animals"}); err == nil {
		t.Fatal("expected suffix format error")
	}

	badExamplesPrompt, err := NewFewShotPromptWithTemplates(
		[]map[string]any{{"question": "foo"}},
		nil,
		examplePrompt,
		&prefix,
		suffix,
		"\n",
		nil,
		false,
	)
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if _, err := badExamplesPrompt.Format(map[string]any{"content": "animals", "new_content": "party"}); err == nil {
		t.Fatal("expected example format error")
	}
}

func TestFewShotPromptWithTemplatesFinalRenderError(t *testing.T) {
	examplePrompt, err := NewPromptTemplate("example", "{{.answer}}")
	if err != nil {
		t.Fatalf("new example prompt: %v", err)
	}
	suffix, err := NewPromptTemplate("suffix", "done")
	if err != nil {
		t.Fatalf("new suffix: %v", err)
	}
	prompt, err := NewFewShotPromptWithTemplates(
		[]map[string]any{{"answer": "broken {{.x"}},
		nil,
		examplePrompt,
		nil,
		suffix,
		"\n",
		nil,
		false,
	)
	if err != nil {
		t.Fatalf("new few-shot: %v", err)
	}
	if _, err := prompt.Format(map[string]any{"extra": "leftover"}); err == nil {
		t.Fatal("expected final render error")
	}
}

func TestDictPromptTemplateFormatError(t *testing.T) {
	prompt := NewDictPromptTemplate(map[string]any{"text": "{{.missing}}"})
	if _, err := prompt.Format(nil); err == nil {
		t.Fatal("expected missing variable error")
	}
	nested := NewDictPromptTemplate(map[string]any{"items": []any{"{{"}})
	if _, err := nested.Format(nil); err == nil {
		t.Fatal("expected nested template error")
	}
	deep := NewDictPromptTemplate(map[string]any{"meta": map[string]any{"bad": "{{"}})
	if _, err := deep.Format(nil); err == nil {
		t.Fatal("expected deep template error")
	}
}

func TestDictPromptTemplateSliceValues(t *testing.T) {
	prompt := NewDictPromptTemplate(map[string]any{
		"tags":  []string{"{{.name}}", "static"},
		"count": 7,
	})
	got, err := prompt.Format(map[string]any{"name": "Ada"})
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	tags, ok := got["tags"].([]any)
	if !ok || len(tags) != 2 || tags[0] != "Ada" || tags[1] != "static" {
		t.Fatalf("tags: %#v", got["tags"])
	}
	if got["count"] != 7 {
		t.Fatalf("count: %#v", got["count"])
	}
}
