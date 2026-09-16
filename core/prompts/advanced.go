package prompts

import (
	"fmt"
	"maps"
	"reflect"
	"slices"

	"github.com/projanvil/langchain-golang/core/messages"
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

// FewShotChatMessagePromptTemplate renders few-shot examples as alternating
// human/AI message pairs. It mirrors langchain_core's
// FewShotChatMessagePromptTemplate (few_shot.py:262): the canonical Python
// example prompt is a ChatPromptTemplate of ("human", ...) and ("ai", ...)
// messages; here each example is rendered by a human-side and an AI-side
// PromptTemplate instead. It satisfies ChatPromptPart, so it can be embedded
// directly in NewChatPromptTemplateFromParts like any other message part.
type FewShotChatMessagePromptTemplate struct {
	Examples        []map[string]any
	ExampleSelector ExampleSelector
	// HumanPrompt renders the human message of each example pair.
	HumanPrompt PromptTemplate
	// AIPrompt renders the AI message of each example pair.
	AIPrompt PromptTemplate
	// inputVariables are the variables passed to ExampleSelector in selector
	// mode (Python input_variables, few_shot.py:361-365). With fixed examples
	// it stays empty: rendering requires no input values.
	inputVariables []string
}

// NewFewShotChatMessagePromptTemplate creates a chat few-shot prompt template.
// Exactly one of examples or selector must be provided (few_shot.py:44-69):
// examples may be an empty non-nil slice (renders no messages, matching
// Python where examples=[] is accepted), while nil means "not provided" — and
// providing ANY non-nil examples (an empty list included) together with a
// selector is the both-provided error, since a non-None examples always wins
// in Python's either/or contract. humanTemplate and aiTemplate use Go
// text/template syntax, e.g. "What is {{.input}}?" / "{{.output}}".
// inputVariables is only meaningful in selector mode, where the render values
// are forwarded to the selector.
func NewFewShotChatMessagePromptTemplate(
	examples []map[string]any,
	selector ExampleSelector,
	humanTemplate string,
	aiTemplate string,
	inputVariables []string,
) (FewShotChatMessagePromptTemplate, error) {
	if examples != nil && selector != nil {
		return FewShotChatMessagePromptTemplate{}, fmt.Errorf("only one of examples and example selector should be provided")
	}
	if examples == nil && selector == nil {
		return FewShotChatMessagePromptTemplate{}, fmt.Errorf("one of examples and example selector should be provided")
	}
	humanPrompt, err := NewPromptTemplate("few-shot-example-human", humanTemplate)
	if err != nil {
		return FewShotChatMessagePromptTemplate{}, err
	}
	aiPrompt, err := NewPromptTemplate("few-shot-example-ai", aiTemplate)
	if err != nil {
		return FewShotChatMessagePromptTemplate{}, err
	}
	return FewShotChatMessagePromptTemplate{
		Examples:        cloneExamples(examples),
		ExampleSelector: selector,
		HumanPrompt:     humanPrompt,
		AIPrompt:        aiPrompt,
		inputVariables:  append([]string(nil), inputVariables...),
	}, nil
}

// FormatMessages renders each example as one human message followed by one AI
// message, flattened in order. This aligns with Python _format_messages
// (few_shot.py:393-412): examples come from the fixed list or from the
// example selector (which receives the render values, few_shot.py:71-89), and
// each example is formatted by the example prompt templates. An example
// missing a key required by the templates is an error — Python raises
// KeyError there ({k: e[k] for k in example_prompt.input_variables},
// few_shot.py:398-400). Empty examples render an empty message list.
func (t FewShotChatMessagePromptTemplate) FormatMessages(values map[string]any) ([]messages.Message, error) {
	examples := t.Examples
	if t.ExampleSelector != nil {
		selected, err := t.ExampleSelector.SelectExamples(values)
		if err != nil {
			return nil, err
		}
		examples = selected
	}
	out := []messages.Message{}
	for _, example := range examples {
		humanText, err := t.HumanPrompt.Format(example)
		if err != nil {
			return nil, fmt.Errorf("format few-shot example human message: %w", err)
		}
		aiText, err := t.AIPrompt.Format(example)
		if err != nil {
			return nil, fmt.Errorf("format few-shot example ai message: %w", err)
		}
		out = append(out, messages.Human(humanText), messages.AI(aiText))
	}
	return out, nil
}

// InputVariables returns the variables required by this template: the
// declared variables forwarded to the example selector in selector mode, and
// none with fixed examples (Python input_variables defaults to [],
// few_shot.py:361-365).
func (t FewShotChatMessagePromptTemplate) InputVariables() []string {
	return append([]string(nil), t.inputVariables...)
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
	return slices.Sorted(maps.Keys(seen))
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
