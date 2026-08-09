package channels

// OverwriteType is the discriminator value that marks an Overwrite after JSON
// round-trip, mirroring the `type: Literal["__overwrite__"]` field on Python's
// langgraph.types.Overwrite.
const OverwriteType = "__overwrite__"

// Overwrite bypasses a channel's reducer and writes Value directly, mirroring
// Python's langgraph.types.Overwrite dataclass. The Type field carries the
// OverwriteType discriminator so the value survives JSON serialization (e.g.
// through a checkpoint serde or API boundary) and can still be recognized by
// AsOverwrite / IsOverwrite after deserialization.
type Overwrite struct {
	Value any    `json:"value"`
	Type  string `json:"type"`
}

// NewOverwrite wraps value for a direct, reducer-bypassing write to a
// BinaryOperator channel. The returned Overwrite always carries Type =
// OverwriteType so it is recognized after JSON round-trip.
func NewOverwrite(value any) Overwrite {
	return Overwrite{Value: value, Type: OverwriteType}
}

// IsOverwrite reports whether v is an Overwrite value or a deserialized map
// carrying the __overwrite__ discriminator. It is a convenience wrapper around
// AsOverwrite.
func IsOverwrite(v any) bool {
	_, ok := AsOverwrite(v)
	return ok
}

// AsOverwrite unwraps v if it is an Overwrite (or a deserialized map form of
// one), returning the inner value and true. Otherwise it returns (nil, false).
//
// Three forms are recognized, mirroring Python's
// langgraph.channels.binop._get_overwrite:
//   - the typed Overwrite struct (or *Overwrite);
//   - the single-key sentinel map {"__overwrite__": value};
//   - the JSON-roundtripped map {"type": "__overwrite__", "value": ...} that
//     results from serializing an Overwrite through a JSON boundary.
func AsOverwrite(v any) (any, bool) {
	switch ow := v.(type) {
	case Overwrite:
		return ow.Value, true
	case *Overwrite:
		return ow.Value, true
	case map[string]any:
		// Sentinel-keyed form: {"__overwrite__": value}.
		if len(ow) == 1 {
			if inner, ok := ow[OverwriteType]; ok {
				return inner, true
			}
		}
		// JSON-roundtripped dataclass form:
		// {"type": "__overwrite__", "value": ...}.
		if t, ok := ow["type"].(string); ok && t == OverwriteType {
			if inner, ok := ow["value"]; ok {
				return inner, true
			}
		}
	}
	return nil, false
}
