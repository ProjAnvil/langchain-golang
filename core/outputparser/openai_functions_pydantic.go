package outputparser

import (
	"encoding/json"
	"fmt"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// PydanticOutputFunctionsParser parses OpenAI function_call arguments into a
// validated object. It mirrors Python's PydanticOutputFunctionsParser: the
// arguments JSON is validated against a JSON schema (the Go equivalent of a
// Pydantic model) and returned as a map.
//
// Schema is the single schema used in ArgsOnly mode. Schemas maps function
// names to schemas and is used when ArgsOnly is false; providing multiple
// schemas with ArgsOnly true is an error, matching Python's model validator.
type PydanticOutputFunctionsParser struct {
	Schema   schema.Schema
	Schemas  map[string]schema.Schema
	ArgsOnly bool
}

// ParseMessage extracts the function call, selects a schema, validates the
// arguments, and returns the parsed object.
func (p PydanticOutputFunctionsParser) ParseMessage(message messages.Message) (any, error) {
	call, err := functionCallFromMessage(message)
	if err != nil {
		return nil, err
	}
	selected, err := p.selectSchema(call.Name)
	if err != nil {
		return nil, err
	}
	return p.parseArguments(call.Arguments, selected)
}

// selectSchema resolves the schema for a function name, mirroring Python's
// validate_schema / parse_result contract.
func (p PydanticOutputFunctionsParser) selectSchema(name string) (schema.Schema, error) {
	if len(p.Schemas) > 0 {
		if p.ArgsOnly {
			return nil, fmt.Errorf("if multiple pydantic schemas are provided then ArgsOnly should be False")
		}
		selected, ok := p.Schemas[name]
		if !ok {
			return nil, fmt.Errorf("no pydantic schema for function %q", name)
		}
		return selected, nil
	}
	if len(p.Schema) == 0 {
		return nil, fmt.Errorf("pydantic schema is required")
	}
	return p.Schema, nil
}

// parseArguments decodes the arguments JSON, validates it against the schema
// using the package's shared validateSchema helper, and returns the object.
func (p PydanticOutputFunctionsParser) parseArguments(arguments string, spec schema.Schema) (map[string]any, error) {
	var raw any
	if err := json.Unmarshal([]byte(arguments), &raw); err != nil {
		return nil, &lcerrors.OutputParserException{
			Message:   fmt.Sprintf("parse function call arguments: %v", err),
			LLMOutput: arguments,
			Err:       err,
		}
	}
	if err := validateSchema(raw, spec, "$"); err != nil {
		return nil, &lcerrors.OutputParserException{
			Message:   fmt.Sprintf("function call arguments failed schema validation: %v", err),
			LLMOutput: arguments,
			Err:       err,
		}
	}
	object, ok := raw.(map[string]any)
	if !ok {
		return nil, &lcerrors.OutputParserException{
			Message:   "function call arguments are not a JSON object",
			LLMOutput: arguments,
		}
	}
	return object, nil
}

// PydanticAttrOutputFunctionsParser parses a function call and returns one
// attribute of the resulting object. It mirrors Python's
// PydanticAttrOutputFunctionsParser (getattr on the parsed model).
type PydanticAttrOutputFunctionsParser struct {
	PydanticOutputFunctionsParser
	AttrName string
}

// ParseMessage parses the function call and returns the configured attribute.
func (p PydanticAttrOutputFunctionsParser) ParseMessage(message messages.Message) (any, error) {
	result, err := p.PydanticOutputFunctionsParser.ParseMessage(message)
	if err != nil {
		return nil, err
	}
	object, ok := result.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("parsed function output is not an object")
	}
	value, ok := object[p.AttrName]
	if !ok {
		return nil, fmt.Errorf("attribute %q not found in parsed function output", p.AttrName)
	}
	return value, nil
}
