package serde

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Type tags returned by DumpsTyped, mirroring Python's typed dumps/loads
// contract: "json" for JSON-native values, "json+envelope" for values
// encoded through the closed type registry.
const (
	tagJSON     = "json"
	tagEnvelope = "json+envelope"
)

// Envelope object keys for registered values: {"__type__": name, "data":
// payload}.
const (
	envelopeType = "__type__"
	envelopeData = "data"
)

// Registry envelope names (the closed set).
const (
	nameMessage   = "messages.Message"
	nameMessages  = "[]messages.Message"
	nameSend      = "types.Send"
	nameInterrupt = "types.Interrupt"
	nameTime      = "time.Time"
	nameBytes     = "[]byte"
	nameInt64     = "int64"
	nameInt       = "int"
	nameStrings   = "[]string"
)

// jsonSerializer implements checkpoint.Serializer as JSON plus the closed
// type registry documented in the package doc.
type jsonSerializer struct{}

var _ checkpoint.Serializer = jsonSerializer{}

// NewJSONSerializer returns a checkpoint.Serializer that encodes JSON-native
// values as plain JSON and registry members as typed envelopes.
func NewJSONSerializer() checkpoint.Serializer {
	return jsonSerializer{}
}

// DumpsTyped encodes v as plain JSON (tag "json") when v is JSON-native, or
// as a registry envelope (tag "json+envelope") when v is a registered
// concrete type. Encoding any other concrete type is an error — there is no
// silent lossy fallback.
func (jsonSerializer) DumpsTyped(v any) (string, []byte, error) {
	canonical := v
	tag := tagJSON
	if name, payload, ok, err := encodeRegistered(v); err != nil {
		return "", nil, err
	} else if ok {
		canonical = map[string]any{envelopeType: name, envelopeData: payload}
		tag = tagEnvelope
	} else if c, err := encodeValue(v); err != nil {
		return "", nil, err
	} else {
		canonical = c
	}
	data, err := json.Marshal(canonical)
	if err != nil {
		return "", nil, fmt.Errorf("serde: encode %T: %w", v, err)
	}
	return tag, data, nil
}

// LoadsTyped decodes data produced by DumpsTyped. Plain-JSON values decode
// with standard JSON semantics (numbers as float64); envelopes restore their
// registered Go types, recursively. An unknown tag or envelope name is an
// error.
func (jsonSerializer) LoadsTyped(typ string, data []byte) (any, error) {
	if typ != tagJSON && typ != tagEnvelope {
		return nil, fmt.Errorf("serde: unknown type tag %q", typ)
	}
	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		return nil, fmt.Errorf("serde: decode %q: %w", typ, err)
	}
	if typ == tagEnvelope {
		m, ok := decoded.(map[string]any)
		name, okName := m[envelopeType].(string)
		if !ok || !okName {
			return nil, fmt.Errorf("serde: malformed envelope: %s", data)
		}
		return decodeEnvelope(name, m[envelopeData])
	}
	return decodeValue(decoded)
}

// encodeValue converts v into a JSON-canonical tree: registered concrete
// types become envelope maps, maps and slices are converted recursively, and
// JSON-native primitives pass through. Any other type is an error.
func encodeValue(v any) (any, error) {
	switch t := v.(type) {
	case nil, string, bool, float64:
		return v, nil
	case map[string]any:
		m := make(map[string]any, len(t))
		for k, item := range t {
			c, err := encodeValue(item)
			if err != nil {
				return nil, err
			}
			m[k] = c
		}
		return m, nil
	case []any:
		s := make([]any, len(t))
		for i, item := range t {
			c, err := encodeValue(item)
			if err != nil {
				return nil, err
			}
			s[i] = c
		}
		return s, nil
	}
	name, payload, ok, err := encodeRegistered(v)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("serde: unregistered type %T (must be JSON-native or a registry member)", v)
	}
	return map[string]any{envelopeType: name, envelopeData: payload}, nil
}

// encodeRegistered returns the envelope name and canonical payload for a
// registered concrete type; ok is false when v is not a registry member.
// Payloads are values json.Marshal encodes directly; the any-typed fields of
// types.Send and types.Interrupt are converted recursively so nested
// registered values keep their types.
func encodeRegistered(v any) (name string, payload any, ok bool, err error) {
	switch t := v.(type) {
	case messages.Message:
		return nameMessage, t, true, nil
	case []messages.Message:
		return nameMessages, t, true, nil
	case types.Send:
		arg, err := encodeValue(t.Arg)
		if err != nil {
			return "", nil, false, err
		}
		return nameSend, map[string]any{"node": t.Node, "arg": arg}, true, nil
	case types.Interrupt:
		value, err := encodeValue(t.Value)
		if err != nil {
			return "", nil, false, err
		}
		return nameInterrupt, map[string]any{"value": value, "id": t.ID}, true, nil
	case time.Time:
		return nameTime, t.Format(time.RFC3339Nano), true, nil
	case []byte:
		return nameBytes, base64.StdEncoding.EncodeToString(t), true, nil
	case int64:
		return nameInt64, t, true, nil
	case int:
		return nameInt, t, true, nil
	case []string:
		return nameStrings, t, true, nil
	}
	return "", nil, false, nil
}

// decodeValue restores a JSON-decoded tree: envelope maps become their
// registered typed values, everything else passes through recursively.
func decodeValue(v any) (any, error) {
	switch t := v.(type) {
	case map[string]any:
		if name, ok := t[envelopeType].(string); ok {
			return decodeEnvelope(name, t[envelopeData])
		}
		m := make(map[string]any, len(t))
		for k, item := range t {
			d, err := decodeValue(item)
			if err != nil {
				return nil, err
			}
			m[k] = d
		}
		return m, nil
	case []any:
		s := make([]any, len(t))
		for i, item := range t {
			d, err := decodeValue(item)
			if err != nil {
				return nil, err
			}
			s[i] = d
		}
		return s, nil
	}
	return v, nil
}

// decodeEnvelope restores the registered Go value for an envelope name and
// its JSON-decoded payload. Unknown names and malformed payloads are errors.
func decodeEnvelope(name string, payload any) (any, error) {
	switch name {
	case nameMessage:
		var m messages.Message
		if err := remarshal(payload, &m); err != nil {
			return nil, fmt.Errorf("serde: decode %s: %w", name, err)
		}
		return m, nil
	case nameMessages:
		var ms []messages.Message
		if err := remarshal(payload, &ms); err != nil {
			return nil, fmt.Errorf("serde: decode %s: %w", name, err)
		}
		return ms, nil
	case nameSend:
		m, err := expectMap(name, payload)
		if err != nil {
			return nil, err
		}
		node, ok := m["node"].(string)
		if !ok {
			return nil, fmt.Errorf("serde: decode %s: node is not a string", name)
		}
		arg, err := decodeValue(m["arg"])
		if err != nil {
			return nil, err
		}
		send := types.Send{Node: node}
		if arg != nil {
			argMap, ok := arg.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("serde: decode %s: arg is %T, want map[string]any", name, arg)
			}
			send.Arg = argMap
		}
		return send, nil
	case nameInterrupt:
		m, err := expectMap(name, payload)
		if err != nil {
			return nil, err
		}
		id, ok := m["id"].(string)
		if !ok {
			return nil, fmt.Errorf("serde: decode %s: id is not a string", name)
		}
		value, err := decodeValue(m["value"])
		if err != nil {
			return nil, err
		}
		return types.Interrupt{Value: value, ID: id}, nil
	case nameTime:
		s, ok := payload.(string)
		if !ok {
			return nil, fmt.Errorf("serde: decode %s: payload is not a string", name)
		}
		ts, err := time.Parse(time.RFC3339Nano, s)
		if err != nil {
			return nil, fmt.Errorf("serde: decode %s: %w", name, err)
		}
		return ts, nil
	case nameBytes:
		s, ok := payload.(string)
		if !ok {
			return nil, fmt.Errorf("serde: decode %s: payload is not a string", name)
		}
		b, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			return nil, fmt.Errorf("serde: decode %s: %w", name, err)
		}
		return b, nil
	case nameInt64:
		f, err := expectNumber(name, payload)
		if err != nil {
			return nil, err
		}
		return int64(f), nil
	case nameInt:
		f, err := expectNumber(name, payload)
		if err != nil {
			return nil, err
		}
		return int(f), nil
	case nameStrings:
		items, ok := payload.([]any)
		if !ok {
			return nil, fmt.Errorf("serde: decode %s: payload is %T, want an array", name, payload)
		}
		ss := make([]string, len(items))
		for i, item := range items {
			s, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("serde: decode %s: element %d is not a string", name, i)
			}
			ss[i] = s
		}
		return ss, nil
	}
	return nil, fmt.Errorf("serde: unknown envelope type %q", name)
}

// expectMap asserts that a decoded envelope payload is a JSON object.
func expectMap(name string, payload any) (map[string]any, error) {
	m, ok := payload.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("serde: decode %s: payload is %T, want an object", name, payload)
	}
	return m, nil
}

// expectNumber asserts that a decoded envelope payload is a JSON number.
func expectNumber(name string, payload any) (float64, error) {
	f, ok := payload.(float64)
	if !ok {
		return 0, fmt.Errorf("serde: decode %s: payload is %T, want a number", name, payload)
	}
	return f, nil
}

// remarshal converts a JSON-decoded value back into a typed struct by
// round-tripping through JSON.
func remarshal(payload any, out any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, out)
}
