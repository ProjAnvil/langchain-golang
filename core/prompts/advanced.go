package prompts

import (
	"fmt"
	"reflect"
)

// ExampleSelector selects examples for a few-shot prompt.
type ExampleSelector interface {
	SelectExamples(inputs map[string]any) ([]map[string]any, error)
}

// FewShotPromptTemplate formats prefix, examples, and suffix into one prompt.
type FewShotPromptTemplate struct {
	Examples         []map[string]any
	ExampleSelector  ExampleSelector
	ExamplePrompt    PromptTemplate
	Prefix           string
	Suffix           string
	ExampleSeparator string
}

// FewShotPromptWithTemplates formats prefix, examples, and suffix where prefix
// and suffix are themselves prompt templates.
type FewShotPromptWithTemplates struct {
	Examples         []map[string]any
	ExampleSelector  ExampleSelector
	ExamplePrompt    PromptTemplate
	Prefix           *PromptTemplate
	Suffix           PromptTemplate
	ExampleSeparator string
	InputVariables   []string
}

// NewFewShotPromptTemplate creates a few-shot prompt template. Exactly one of
// examples or selector must be provided.
func NewFewShotPromptTemplate(
	examples []map[string]any,
	selector ExampleSelector,
	examplePrompt PromptTemplate,
	prefix string,
	suffix string,
	exampleSeparator string,
) (FewShotPromptTemplate, error) {
	if len(examples) > 0 && selector != nil {
		return FewShotPromptTemplate{}, fmt.Errorf("only one of examples and example selector should be provided")
	}
	if len(examples) == 0 && selector == nil {
		return FewShotPromptTemplate{}, fmt.Errorf("one of examples and example selector should be provided")
	}
	if exampleSeparator == "" {
		exampleSeparator = "\n\n"
	}
	return FewShotPromptTemplate{
		Examples:         cloneExamples(examples),
		ExampleSelector:  selector,
		ExamplePrompt:    examplePrompt,
		Prefix:           prefix,
		Suffix:           suffix,
		ExampleSeparator: exampleSeparator,
	}, nil
}

// NewFewShotPromptWithTemplates creates a few-shot prompt template whose prefix
// and suffix are PromptTemplate values. Exactly one of examples or selector must
// be provided.
func NewFewShotPromptWithTemplates(
	examples []map[string]any,
	selector ExampleSelector,
	examplePrompt PromptTemplate,
	prefix *PromptTemplate,
	suffix PromptTemplate,
	exampleSeparator string,
	inputVariables []string,
	validateTemplate bool,
) (FewShotPromptWithTemplates, error) {
	if len(examples) > 0 && selector != nil {
		return FewShotPromptWithTemplates{}, fmt.Errorf("only one of examples and example selector should be provided")
	}
	if len(examples) == 0 && selector == nil {
		return FewShotPromptWithTemplates{}, fmt.Errorf("one of examples and example selector should be provided")
	}
	if exampleSeparator == "" {
		exampleSeparator = "\n\n"
	}
	expected := promptWithTemplateVariables(prefix, suffix)
	if validateTemplate {
		if err := validatePromptVariables(inputVariables, expected); err != nil {
			return FewShotPromptWithTemplates{}, err
		}
	} else {
		inputVariables = expected
	}
	return FewShotPromptWithTemplates{
		Examples:         cloneExamples(examples),
		ExampleSelector:  selector,
		ExamplePrompt:    examplePrompt,
		Prefix:           clonePromptPointer(prefix),
		Suffix:           suffix,
		ExampleSeparator: exampleSeparator,
		InputVariables:   append([]string(nil), inputVariables...),
	}, nil
}

// Format renders the few-shot prompt.
func (p FewShotPromptTemplate) Format(values map[string]any) (string, error) {
	examples := p.Examples
	if p.ExampleSelector != nil {
		selected, err := p.ExampleSelector.SelectExamples(values)
		if err != nil {
			return "", err
		}
		examples = selected
	}
	pieces := []string{}
	if p.Prefix != "" {
		prefix, err := renderInlineTemplate("few-shot-prefix", p.Prefix, values)
		if err != nil {
			return "", err
		}
		pieces = append(pieces, prefix)
	}
	for _, example := range examples {
		formatted, err := p.ExamplePrompt.Format(example)
		if err != nil {
			return "", err
		}
		pieces = append(pieces, formatted)
	}
	if p.Suffix != "" {
		suffix, err := renderInlineTemplate("few-shot-suffix", p.Suffix, values)
		if err != nil {
			return "", err
		}
		pieces = append(pieces, suffix)
	}
	return joinNonEmpty(pieces, p.ExampleSeparator), nil
}

// Format renders the few-shot prompt with template prefix and suffix.
func (p FewShotPromptWithTemplates) Format(values map[string]any) (string, error) {
	examples := p.Examples
	if p.ExampleSelector != nil {
		selected, err := p.ExampleSelector.SelectExamples(values)
		if err != nil {
			return "", err
		}
		examples = selected
	}
	remaining := cloneMapAny(values)
	pieces := []string{}
	if p.Prefix != nil {
		prefixValues := takePromptValues(remaining, p.Prefix.InputVariables())
		prefix, err := p.Prefix.Format(prefixValues)
		if err != nil {
			return "", err
		}
		pieces = append(pieces, prefix)
	}
	for _, example := range examples {
		formatted, err := p.ExamplePrompt.Format(example)
		if err != nil {
			return "", err
		}
		pieces = append(pieces, formatted)
	}
	suffixValues := takePromptValues(remaining, p.Suffix.InputVariables())
	suffix, err := p.Suffix.Format(suffixValues)
	if err != nil {
		return "", err
	}
	pieces = append(pieces, suffix)
	template := joinNonEmpty(pieces, p.ExampleSeparator)
	if len(remaining) == 0 {
		return template, nil
	}
	return renderInlineTemplate("few-shot-with-templates", template, remaining)
}

// DictPromptTemplate recursively formats string leaves in a dictionary.
type DictPromptTemplate struct {
	Template map[string]any
}

// NewDictPromptTemplate creates a dictionary prompt template.
func NewDictPromptTemplate(template map[string]any) DictPromptTemplate {
	return DictPromptTemplate{Template: cloneMapAny(template)}
}

// Format renders the dictionary prompt.
func (p DictPromptTemplate) Format(values map[string]any) (map[string]any, error) {
	formatted, err := formatValue(p.Template, values)
	if err != nil {
		return nil, err
	}
	out, ok := formatted.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("formatted dictionary prompt is not an object")
	}
	return out, nil
}

func formatValue(value any, inputs map[string]any) (any, error) {
	switch v := value.(type) {
	case string:
		return renderInlineTemplate("dict-prompt", v, inputs)
	case map[string]any:
		out := make(map[string]any, len(v))
		for key, child := range v {
			formatted, err := formatValue(child, inputs)
			if err != nil {
				return nil, err
			}
			out[key] = formatted
		}
		return out, nil
	case []any:
		out := make([]any, len(v))
		for i, child := range v {
			formatted, err := formatValue(child, inputs)
			if err != nil {
				return nil, err
			}
			out[i] = formatted
		}
		return out, nil
	default:
		rv := reflect.ValueOf(value)
		if rv.IsValid() && rv.Kind() == reflect.Slice {
			out := make([]any, rv.Len())
			for i := 0; i < rv.Len(); i++ {
				formatted, err := formatValue(rv.Index(i).Interface(), inputs)
				if err != nil {
					return nil, err
				}
				out[i] = formatted
			}
			return out, nil
		}
		return value, nil
	}
}

func renderInlineTemplate(name, text string, values map[string]any) (string, error) {
	prompt, err := NewPromptTemplate(name, text)
	if err != nil {
		return "", err
	}
	return prompt.Format(values)
}

func joinNonEmpty(values []string, separator string) string {
	out := ""
	for _, value := range values {
		if value == "" {
			continue
		}
		if out != "" {
			out += separator
		}
		out += value
	}
	return out
}

func cloneExamples(examples []map[string]any) []map[string]any {
	out := make([]map[string]any, len(examples))
	for i, example := range examples {
		out[i] = cloneMapAny(example)
	}
	return out
}

func cloneMapAny(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func clonePromptPointer(prompt *PromptTemplate) *PromptTemplate {
	if prompt == nil {
		return nil
	}
	copied := *prompt
	return &copied
}

func promptWithTemplateVariables(prefix *PromptTemplate, suffix PromptTemplate) []string {
	seen := map[string]bool{}
	if prefix != nil {
		for _, variable := range prefix.InputVariables() {
			seen[variable] = true
		}
	}
	for _, variable := range suffix.InputVariables() {
		seen[variable] = true
	}
	out := make([]string, 0, len(seen))
	for variable := range seen {
		out = append(out, variable)
	}
	sortStrings(out)
	return out
}

func validatePromptVariables(got []string, want []string) error {
	got = append([]string(nil), got...)
	want = append([]string(nil), want...)
	sortStrings(got)
	sortStrings(want)
	if len(got) != len(want) {
		return fmt.Errorf("got input_variables=%v, but based on prefix/suffix expected %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("got input_variables=%v, but based on prefix/suffix expected %v", got, want)
		}
	}
	return nil
}

func takePromptValues(values map[string]any, variables []string) map[string]any {
	out := map[string]any{}
	for _, variable := range variables {
		if value, ok := values[variable]; ok {
			out[variable] = value
			delete(values, variable)
		}
	}
	return out
}

func sortStrings(values []string) {
	for i := 0; i < len(values); i++ {
		for j := i + 1; j < len(values); j++ {
			if values[j] < values[i] {
				values[i], values[j] = values[j], values[i]
			}
		}
	}
}
