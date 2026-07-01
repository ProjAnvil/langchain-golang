package schema

// Schema is the common JSON-schema-shaped contract used by prompts, tools,
// models, parsers, and structured output helpers.
type Schema map[string]any

// Object creates a JSON object schema with the provided properties and required
// field names.
func Object(properties map[string]Schema, required ...string) Schema {
	props := make(map[string]any, len(properties))
	for name, prop := range properties {
		props[name] = prop
	}

	out := Schema{
		"type":       "object",
		"properties": props,
	}
	if len(required) > 0 {
		req := make([]string, len(required))
		copy(req, required)
		out["required"] = req
	}

	return out
}

// String creates a JSON string schema.
func String(description string) Schema {
	return scalar("string", description)
}

// Integer creates a JSON integer schema.
func Integer(description string) Schema {
	return scalar("integer", description)
}

// Number creates a JSON number schema.
func Number(description string) Schema {
	return scalar("number", description)
}

// Boolean creates a JSON boolean schema.
func Boolean(description string) Schema {
	return scalar("boolean", description)
}

func scalar(kind string, description string) Schema {
	out := Schema{"type": kind}
	if description != "" {
		out["description"] = description
	}
	return out
}
