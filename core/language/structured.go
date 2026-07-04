package language

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// StructuredCaller is implemented by ChatModels that can produce output
// conforming to a JSON schema natively (provider response_format / tool-based).
// Models that do not implement it fall back to InvokeStructured's default
// JSON-decode-and-validate path.
type StructuredCaller interface {
	InvokeStructured(ctx context.Context, input []messages.Message, sch schema.Schema) (messages.Message, error)
}

// ErrSchemaViolation is returned when a model's response text does not parse
// as JSON or is missing keys the schema marks required.
var ErrSchemaViolation = errors.New("schema violation")

// InvokeStructured calls m (or a StructuredCaller it implements) to produce a
// message whose text is JSON conforming to sch. If m implements StructuredCaller,
// the native path is used; otherwise the model is invoked normally and the
// response text is validated against sch (best-effort JSON decode + required-key
// check). The returned message's text is the model's JSON text.
func InvokeStructured(
	ctx context.Context,
	m ChatModel,
	input []messages.Message,
	sch schema.Schema,
) (messages.Message, error) {
	// Prefer the provider-native structured-output path when available so that
	// partners (e.g. OpenAI response_format) can enforce the schema upstream.
	if native, ok := m.(StructuredCaller); ok {
		return native.InvokeStructured(ctx, input, sch)
	}

	response, err := m.Invoke(ctx, input)
	if err != nil {
		return messages.Message{}, fmt.Errorf("structured output: invoke model: %w", err)
	}

	text := messages.Text(response)
	var decoded map[string]any
	if err := json.Unmarshal([]byte(text), &decoded); err != nil {
		return messages.Message{}, fmt.Errorf("structured output: parse response JSON: %w: %v",
			ErrSchemaViolation, err)
	}

	// schema.Object encodes required field names under "required" as []string.
	for _, required := range requiredKeys(sch) {
		if _, present := decoded[required]; !present {
			return messages.Message{}, fmt.Errorf("structured output: missing required key %q: %w",
				required, ErrSchemaViolation)
		}
	}

	return response, nil
}

// requiredKeys returns the field names the schema marks required. The schema
// package encodes them under the "required" key as []string.
func requiredKeys(sch schema.Schema) []string {
	required, _ := sch["required"].([]string)
	return required
}
