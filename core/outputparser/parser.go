package outputparser

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"regexp"
	"strings"

	"github.com/projanvil/langchain-golang/core/outputs"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/utils"
)

// Parser converts raw model output into a typed value.
type Parser[T any] interface {
	Parse(ctx context.Context, text string) (T, error)
	FormatInstructions() string
}

// StringParser returns model text unchanged.
type StringParser struct{}

// Parse returns text unchanged.
func (p StringParser) Parse(_ context.Context, text string) (string, error) {
	return text, nil
}

// FormatInstructions returns parser instructions.
func (p StringParser) FormatInstructions() string {
	return "Return plain text."
}

// JSONParser parses model output as JSON.
type JSONParser[T any] struct {
	instructions string
	schema       schema.Schema
}

// NewJSONParser creates a JSON parser.
func NewJSONParser[T any](instructions string) JSONParser[T] {
	if instructions == "" {
		instructions = "Return valid JSON."
	}
	return JSONParser[T]{instructions: instructions}
}

// NewJSONParserWithSchema creates a JSON parser with schema-aware format
// instructions.
func NewJSONParserWithSchema[T any](schemaValue schema.Schema) JSONParser[T] {
	return JSONParser[T]{
		instructions: JSONFormatInstructions(schemaValue),
		schema:       utils.CloneSchema(schemaValue),
	}
}

// Parse parses JSON into T.
func (p JSONParser[T]) Parse(_ context.Context, text string) (T, error) {
	var out T
	decoder := json.NewDecoder(strings.NewReader(extractJSONMarkdown(text)))
	decoder.UseNumber()
	if err := decoder.Decode(&out); err != nil {
		return out, fmt.Errorf("parse json output: %w", err)
	}
	return out, nil
}

// ParseResult parses the first generation in an LLM result.
func (p JSONParser[T]) ParseResult(ctx context.Context, result []outputs.Generation, partial bool) (T, bool, error) {
	var zero T
	if len(result) == 0 {
		return zero, false, fmt.Errorf("generation result is empty")
	}
	text := strings.TrimSpace(result[0].Text)
	if partial {
		parsed, ok, err := ParsePartialJSON(text)
		if err != nil || !ok {
			return zero, false, err
		}
		data, err := json.Marshal(parsed)
		if err != nil {
			return zero, false, err
		}
		out, err := p.Parse(ctx, string(data))
		return out, true, err
	}
	out, err := p.Parse(ctx, text)
	return out, true, err
}

// FormatInstructions returns parser instructions.
func (p JSONParser[T]) FormatInstructions() string {
	return p.instructions
}

// JSONFormatInstructions returns strict JSON format instructions with an
// optional schema.
func JSONFormatInstructions(schemaValue schema.Schema) string {
	if len(schemaValue) == 0 {
		return "Return a JSON object."
	}
	reduced := utils.RemoveJSONSchemaDefinitions(schemaValue)
	delete(reduced, "title")
	delete(reduced, "type")
	schemaText, err := json.Marshal(reduced)
	if err != nil {
		schemaText = []byte("{}")
	}
	return "STRICT OUTPUT FORMAT:\n" +
		"- Return only the JSON value that conforms to the schema.\n" +
		"- Do not wrap the JSON in Markdown or code fences.\n\n" +
		"The output should be formatted as a JSON instance that conforms to this JSON schema:\n" +
		"```\n" + string(schemaText) + "\n```"
}

// CommaSeparatedListParser parses comma-separated output into a string slice.
type CommaSeparatedListParser struct{}

// Parse parses CSV-style comma-separated text.
func (p CommaSeparatedListParser) Parse(_ context.Context, text string) ([]string, error) {
	reader := csv.NewReader(strings.NewReader(text))
	reader.TrimLeadingSpace = true
	reader.FieldsPerRecord = -1
	records, err := reader.ReadAll()
	if err != nil {
		parts := strings.Split(text, ",")
		out := make([]string, len(parts))
		for i, part := range parts {
			out[i] = strings.TrimSpace(part)
		}
		return out, nil
	}
	out := []string{}
	for _, record := range records {
		out = append(out, record...)
	}
	return out, nil
}

// FormatInstructions returns parser instructions.
func (p CommaSeparatedListParser) FormatInstructions() string {
	return "Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`."
}

// NumberedListParser parses lists like "1. foo".
type NumberedListParser struct {
	Pattern *regexp.Regexp
}

// Parse parses numbered list items.
func (p NumberedListParser) Parse(_ context.Context, text string) ([]string, error) {
	pattern := p.Pattern
	if pattern == nil {
		pattern = regexp.MustCompile(`(?m)\d+\.\s([^\n]+)`)
	}
	matches := pattern.FindAllStringSubmatch(text, -1)
	out := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			out = append(out, match[1])
		}
	}
	return out, nil
}

// FormatInstructions returns parser instructions.
func (p NumberedListParser) FormatInstructions() string {
	return "Your response should be a numbered list with each item on a new line."
}

// MarkdownListParser parses Markdown bullet lists.
type MarkdownListParser struct {
	Pattern *regexp.Regexp
}

// Parse parses Markdown list items.
func (p MarkdownListParser) Parse(_ context.Context, text string) ([]string, error) {
	pattern := p.Pattern
	if pattern == nil {
		pattern = regexp.MustCompile(`(?m)^\s*[-*]\s([^\n]+)$`)
	}
	matches := pattern.FindAllStringSubmatch(text, -1)
	out := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			out = append(out, match[1])
		}
	}
	return out, nil
}

// FormatInstructions returns parser instructions.
func (p MarkdownListParser) FormatInstructions() string {
	return "Your response should be a markdown list, eg: `- foo\n- bar\n- baz`."
}

// XMLParser parses XML output into a nested map representation.
type XMLParser struct {
	Tags []string
}

// Parse parses XML, including XML embedded in a fenced code block.
func (p XMLParser) Parse(_ context.Context, text string) (map[string]any, error) {
	xmlText := extractXML(text)
	root, err := parseXMLNode(xmlText)
	if err != nil {
		return nil, fmt.Errorf("parse xml output: %w", err)
	}
	return map[string]any{root.Name: root.Value()}, nil
}

// FormatInstructions returns parser instructions.
func (p XMLParser) FormatInstructions() string {
	tags := strings.Join(p.Tags, ", ")
	if tags == "" {
		tags = "make them on your own"
	}
	return fmt.Sprintf("The output should be formatted as XML. Always open and close all tags. Expected tags: %s.", tags)
}

type xmlNode struct {
	Name     string
	Text     string
	Children []xmlNode
}

func (n xmlNode) Value() any {
	if strings.TrimSpace(n.Text) != "" && len(n.Children) == 0 {
		return n.Text
	}
	children := make([]any, 0, len(n.Children))
	for _, child := range n.Children {
		children = append(children, map[string]any{child.Name: child.Value()})
	}
	return children
}

func extractXML(text string) string {
	re := regexp.MustCompile("(?s)```(?:xml)?(.*?)```")
	if match := re.FindStringSubmatch(text); len(match) > 1 {
		text = match[1]
	}
	encoding := regexp.MustCompile(`(?s)<([^>]*encoding[^>]*)>\n(.*)`)
	if match := encoding.FindStringSubmatch(text); len(match) > 2 {
		text = match[2]
	}
	return strings.TrimSpace(text)
}

func extractJSONMarkdown(text string) string {
	trimmed := strings.TrimSpace(text)
	re := regexp.MustCompile("(?s)```(?:json)?\\s*(.*?)\\s*```")
	if match := re.FindStringSubmatch(trimmed); len(match) > 1 {
		return strings.TrimSpace(match[1])
	}
	startObj := strings.Index(trimmed, "{")
	startArr := strings.Index(trimmed, "[")
	start := -1
	if startObj >= 0 && startArr >= 0 {
		if startObj < startArr {
			start = startObj
		} else {
			start = startArr
		}
	} else if startObj >= 0 {
		start = startObj
	} else {
		start = startArr
	}
	if start < 0 {
		return trimmed
	}
	endObj := strings.LastIndex(trimmed, "}")
	endArr := strings.LastIndex(trimmed, "]")
	end := endObj
	if endArr > end {
		end = endArr
	}
	if end >= start {
		return strings.TrimSpace(trimmed[start : end+1])
	}
	return trimmed
}

func parseXMLNode(text string) (xmlNode, error) {
	decoder := xml.NewDecoder(bytes.NewBufferString(text))
	stack := []xmlNode{}
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return xmlNode{}, err
		}
		switch t := token.(type) {
		case xml.StartElement:
			stack = append(stack, xmlNode{Name: t.Name.Local})
		case xml.CharData:
			if len(stack) > 0 {
				stack[len(stack)-1].Text += string([]byte(t))
			}
		case xml.EndElement:
			if len(stack) == 0 {
				return xmlNode{}, fmt.Errorf("unexpected closing tag %s", t.Name.Local)
			}
			node := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if node.Name != t.Name.Local {
				return xmlNode{}, fmt.Errorf("mismatched closing tag %s for %s", t.Name.Local, node.Name)
			}
			if len(stack) == 0 {
				return node, nil
			}
			stack[len(stack)-1].Children = append(stack[len(stack)-1].Children, node)
		}
	}
	return xmlNode{}, fmt.Errorf("missing XML root")
}
