package prebuilt

import (
	"encoding/json"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
)

// FormatErrorFunc formats a tool-call validation error into ToolMessage
// content, mirroring Python's `format_error` callable
// (tool_validator.py:120-121).
type FormatErrorFunc func(err error, call messages.ToolCall, tool tools.Tool) string

// validationNodeConfig holds the configuration assembled from
// ValidationNodeOptions.
type validationNodeConfig struct {
	messagesKey string
	formatError FormatErrorFunc
}

// ValidationNodeOption configures the NodeFunc returned by NewValidationNode.
type ValidationNodeOption func(*validationNodeConfig)

// WithValidationMessagesKey sets the state key the node reads the message list
// from and writes validation result messages to (default "messages"). Go
// extension: Python's ValidationNode has no counterpart option — it always
// reads state["messages"].
func WithValidationMessagesKey(key string) ValidationNodeOption {
	return func(c *validationNodeConfig) { c.messagesKey = key }
}

// WithFormatError overrides how validation errors are rendered into ToolMessage
// content. nil restores the default.
func WithFormatError(fn FormatErrorFunc) ValidationNodeOption {
	return func(c *validationNodeConfig) { c.formatError = fn }
}

// defaultFormatError mirrors Python's `_default_format_error`
// (tool_validator.py:34-40), with the Go error string in place of repr().
func defaultFormatError(err error, _ messages.ToolCall, _ tools.Tool) string {
	return fmt.Sprintf("%v\n\nRespond after fixing all validation errors.", err)
}

// NewValidationNode returns a graph.NodeFunc that validates every tool call on
// the last AI message in state[messagesKey] against the named tool's
// ArgsSchema, mirroring Python's `langgraph.prebuilt.ValidationNode`
// (tool_validator.py:47). The node does NOT run the tools: each call produces
// a ToolMessage with the validated args as JSON content, or — on a validation
// failure — a ToolMessage with the formatted error and
// AdditionalKwargs["is_error"]=true (so a downstream router can re-prompt the
// model, per the Python docstring example).
//
// Python marks ValidationNode @deprecated(LangGraphDeprecatedSinceV10)
// (tool_validator.py:43-46); this port exists for parity with the deprecated
// API.
//
// Divergences from Python (Go has no pydantic and this repo carries no
// JSON-schema validator dependency):
//
//   - Schemas are tools.Tool values only (Python also accepts BaseModel
//     classes and callables); the tool's ArgsSchema is the validation schema.
//   - Validation is a JSON-schema subset: `required` fields must be present,
//     and present properties typed string/integer/number/boolean/object/array
//     are type-checked. A tool with a nil ArgsSchema accepts any args (unlike
//     Python, which rejects schema-less tools at construction — Go's
//     core/tools.NewFunc explicitly permits a nil schema).
//   - Success content is json.Marshal of the provided args (Python dumps the
//     validated model, applying defaults/coercion; there is no equivalent
//     here).
//
// Input-shape errors mirror Python: a missing/empty/wrong-typed messages key
// is a "no message found in input" error (tool_validator.py:178), and a last
// message that is not an AI message is an error (:181). A call naming an
// unregistered tool is an error (Python raises KeyError at :191).
func NewValidationNode(toolList []tools.Tool, opts ...ValidationNodeOption) (graph.NodeFunc, error) {
	if len(toolList) == 0 {
		return nil, fmt.Errorf("prebuilt: ValidationNode requires at least one tool schema")
	}
	schemasByName := make(map[string]tools.Tool, len(toolList))
	for _, tool := range toolList {
		if tool == nil {
			return nil, fmt.Errorf("prebuilt: ValidationNode tool must not be nil")
		}
		name := tool.Name()
		if name == "" {
			return nil, fmt.Errorf("prebuilt: ValidationNode tool name must not be empty")
		}
		if _, exists := schemasByName[name]; exists {
			return nil, fmt.Errorf("prebuilt: ValidationNode duplicate tool name %q", name)
		}
		schemasByName[name] = tool
	}

	cfg := validationNodeConfig{messagesKey: defaultMessagesKey, formatError: defaultFormatError}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.messagesKey == "" {
		return nil, fmt.Errorf("prebuilt: ValidationNode messages key must not be empty")
	}
	if cfg.formatError == nil {
		cfg.formatError = defaultFormatError
	}

	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		raw, ok := state[cfg.messagesKey]
		if !ok {
			return nil, fmt.Errorf("prebuilt: ValidationNode: no message found in input (missing key %q)", cfg.messagesKey)
		}
		msgs, ok := raw.([]messages.Message)
		if !ok || len(msgs) == 0 {
			return nil, fmt.Errorf("prebuilt: ValidationNode: no message found in input (key %q holds %v)", cfg.messagesKey, raw)
		}
		last := msgs[len(msgs)-1]
		if last.Role != messages.RoleAI {
			return nil, fmt.Errorf("prebuilt: ValidationNode requires the last message to be an AI message, got role %q", last.Role)
		}

		outputs := make([]messages.Message, len(last.ToolCalls))
		for i, call := range last.ToolCalls {
			msg, err := validateToolCall(cfg, schemasByName, call)
			if err != nil {
				return nil, err
			}
			outputs[i] = msg
		}
		return map[string]any{cfg.messagesKey: outputs}, nil
	}, nil
}

// validateToolCall validates one tool call, returning either the validated
// ToolMessage or the error-flagged one. An unregistered tool name is a node
// error, not a per-call validation failure (mirroring Python's KeyError).
func validateToolCall(cfg validationNodeConfig, schemasByName map[string]tools.Tool, call messages.ToolCall) (messages.Message, error) {
	tool, ok := schemasByName[call.Name]
	if !ok {
		return messages.Message{}, fmt.Errorf("prebuilt: ValidationNode has no schema for tool %q", call.Name)
	}
	if err := validateArgsAgainstSchema(tool.ArgsSchema(), call.Args); err != nil {
		msg := messages.Tool(call.ID, cfg.formatError(err, call, tool))
		msg.Name = call.Name
		msg.AdditionalKwargs = map[string]any{"is_error": true}
		return msg, nil
	}
	content, err := json.Marshal(call.Args)
	if err != nil {
		return messages.Message{}, fmt.Errorf("prebuilt: ValidationNode could not serialize validated args for tool %q: %w", call.Name, err)
	}
	msg := messages.Tool(call.ID, string(content))
	msg.Name = call.Name
	return msg, nil
}

// validateArgsAgainstSchema checks args against the supported JSON-schema
// subset: required fields and per-property scalar/container types.
func validateArgsAgainstSchema(s schema.Schema, args map[string]any) error {
	if s == nil {
		return nil
	}
	for _, key := range requiredKeys(s) {
		if _, ok := args[key]; !ok {
			return fmt.Errorf("field %q is required but missing", key)
		}
	}
	props, _ := s["properties"].(map[string]any)
	for name, raw := range props {
		prop := propertySchema(raw)
		typ, _ := prop["type"].(string)
		value, present := args[name]
		if !present || typ == "" {
			continue
		}
		if !jsonTypeMatches(typ, value) {
			return fmt.Errorf("field %q: expected %s, got %T", name, typ, value)
		}
	}
	return nil
}

// propertySchema normalizes a "properties" entry to a plain map. Schemas built
// by schema.Object store each property as a schema.Schema (a named
// map[string]any type), while JSON-decoded schemas arrive as plain
// map[string]any values; both shapes must be type-checked.
func propertySchema(raw any) map[string]any {
	switch p := raw.(type) {
	case schema.Schema:
		return map[string]any(p)
	case map[string]any:
		return p
	default:
		return nil
	}
}

// requiredKeys normalizes the schema's "required" entry, accepting both the
// []string produced by schema.Object and the []any produced by JSON decoding.
func requiredKeys(s schema.Schema) []string {
	switch req := s["required"].(type) {
	case []string:
		return req
	case []any:
		keys := make([]string, 0, len(req))
		for _, k := range req {
			if str, ok := k.(string); ok {
				keys = append(keys, str)
			}
		}
		return keys
	default:
		return nil
	}
}

// jsonTypeMatches reports whether value satisfies a JSON-schema "type" name.
// Unknown type names are not checked. "integer" accepts Go ints and integral
// float64s (the shape JSON-decoded tool-call args take).
func jsonTypeMatches(typ string, value any) bool {
	switch typ {
	case "string":
		_, ok := value.(string)
		return ok
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "integer":
		switch v := value.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return true
		case float32:
			return v == float32(int32(v))
		case float64:
			return v == float64(int64(v))
		default:
			return false
		}
	case "number":
		switch value.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
			return true
		default:
			return false
		}
	case "object":
		_, ok := value.(map[string]any)
		return ok
	case "array":
		_, ok := value.([]any)
		return ok
	default:
		return true
	}
}
