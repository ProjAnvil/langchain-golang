package serde

import (
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// roundTrip encodes v with DumpsTyped and decodes it back with LoadsTyped,
// returning the tag and the decoded value.
func roundTrip(t *testing.T, s checkpoint.Serializer, v any) (string, any) {
	t.Helper()
	typ, data, err := s.DumpsTyped(v)
	if err != nil {
		t.Fatalf("DumpsTyped(%T): %v", v, err)
	}
	got, err := s.LoadsTyped(typ, data)
	if err != nil {
		t.Fatalf("LoadsTyped(%q): %v", typ, err)
	}
	return typ, got
}

func TestJSONSerializerPlainJSONRoundTrip(t *testing.T) {
	s := NewJSONSerializer()
	cases := map[string]any{
		"nil":     nil,
		"string":  "hello",
		"float64": 3.14,
		"bool":    true,
		"map": map[string]any{
			"name":   "x",
			"count":  1.5,
			"nested": map[string]any{"ok": false},
		},
		"slice": []any{"a", 2.5, nil, map[string]any{"k": "v"}},
	}
	for name, v := range cases {
		t.Run(name, func(t *testing.T) {
			typ, got := roundTrip(t, s, v)
			if typ != "json" {
				t.Fatalf("tag = %q, want %q", typ, "json")
			}
			if !reflect.DeepEqual(got, v) {
				t.Fatalf("round trip = %#v, want %#v", got, v)
			}
		})
	}
}

func TestJSONSerializerRegistryRoundTrip(t *testing.T) {
	s := NewJSONSerializer()
	ts := time.Date(2026, 8, 8, 12, 34, 56, 789, time.UTC)
	msg := messages.Message{
		Role:    messages.RoleAI,
		Content: "hi",
		ID:      "m1",
		ToolCalls: []messages.ToolCall{
			{ID: "tc1", Name: "search", Args: map[string]any{"q": "go"}},
		},
	}
	cases := map[string]any{
		"messages.Message":   msg,
		"[]messages.Message": []messages.Message{messages.Human("q"), msg},
		"types.Send":         types.Send{Node: "worker", Arg: map[string]any{"task": "a", "n": 1.5}},
		"types.Interrupt":    types.Interrupt{Value: map[string]any{"reason": "approve?", "options": []any{"y", "n"}}, ID: "i1"},
		"time.Time":          ts,
		"[]byte":             []byte{0x00, 0x01, 0xff},
		"int64":              int64(1<<40 + 7),
		"int":                int(42),
		"[]string":           []string{"a", "b"},
	}
	for name, v := range cases {
		t.Run(name, func(t *testing.T) {
			typ, got := roundTrip(t, s, v)
			if typ != "json+envelope" {
				t.Fatalf("tag = %q, want %q", typ, "json+envelope")
			}
			if !reflect.DeepEqual(got, v) {
				t.Fatalf("round trip = %#v (%T), want %#v (%T)", got, got, v, v)
			}
		})
	}
}

func TestJSONSerializerPreservesIntTypes(t *testing.T) {
	s := NewJSONSerializer()

	if _, got := roundTrip(t, s, int64(9)); true {
		if _, ok := got.(int64); !ok {
			t.Fatalf("int64 decoded as %T, want int64", got)
		}
	}
	if _, got := roundTrip(t, s, int(9)); true {
		if _, ok := got.(int); !ok {
			t.Fatalf("int decoded as %T, want int", got)
		}
	}
	// A plain JSON number still decodes as float64.
	if _, got := roundTrip(t, s, float64(9)); true {
		if _, ok := got.(float64); !ok {
			t.Fatalf("float64 decoded as %T, want float64", got)
		}
	}
}

func TestJSONSerializerPreservesSliceAndBytesTypes(t *testing.T) {
	s := NewJSONSerializer()

	if _, got := roundTrip(t, s, []string{"x"}); true {
		if _, ok := got.([]string); !ok {
			t.Fatalf("[]string decoded as %T, want []string", got)
		}
	}
	if _, got := roundTrip(t, s, []byte("raw")); true {
		b, ok := got.([]byte)
		if !ok {
			t.Fatalf("[]byte decoded as %T, want []byte", got)
		}
		if string(b) != "raw" {
			t.Fatalf("[]byte decoded as %q, want %q", b, "raw")
		}
	}
}

// TestJSONSerializerNilSliceRoundTrip pins that typed nil slices encode a
// null envelope payload and decode back to empty non-nil slices, so a
// checkpoint containing them is restorable (regression: decode used to
// reject the null payload).
func TestJSONSerializerNilSliceRoundTrip(t *testing.T) {
	s := NewJSONSerializer()

	_, got := roundTrip(t, s, []string(nil))
	ss, ok := got.([]string)
	if !ok {
		t.Fatalf("[]string(nil) decoded as %T, want []string", got)
	}
	if ss == nil || len(ss) != 0 {
		t.Fatalf("[]string(nil) round trip = %#v, want empty non-nil []string", ss)
	}

	_, got = roundTrip(t, s, []messages.Message(nil))
	ms, ok := got.([]messages.Message)
	if !ok {
		t.Fatalf("[]messages.Message(nil) decoded as %T, want []messages.Message", got)
	}
	if ms == nil || len(ms) != 0 {
		t.Fatalf("[]messages.Message(nil) round trip = %#v, want empty non-nil []messages.Message", ms)
	}
}

func TestJSONSerializerNestedRegistryValues(t *testing.T) {
	s := NewJSONSerializer()

	// A registered value nested inside an envelope payload (Send.Arg) must
	// round-trip recursively, keeping its concrete type.
	send := types.Send{
		Node: "worker",
		Arg: map[string]any{
			"msgs":  []messages.Message{messages.Human("hi")},
			"tries": int(3),
			"tags":  []string{"x"},
		},
	}
	_, got := roundTrip(t, s, send)
	if !reflect.DeepEqual(got, send) {
		t.Fatalf("round trip = %#v, want %#v", got, send)
	}

	// The same applies to Interrupt.Value and to plain maps/slices.
	intr := types.Interrupt{Value: []messages.Message{messages.AI("stop")}, ID: "i9"}
	if _, got := roundTrip(t, s, intr); !reflect.DeepEqual(got, intr) {
		t.Fatalf("round trip = %#v, want %#v", got, intr)
	}

	m := map[string]any{"when": time.Date(2026, 1, 2, 3, 4, 5, 6, time.UTC)}
	if _, got := roundTrip(t, s, m); !reflect.DeepEqual(got, m) {
		t.Fatalf("round trip = %#v, want %#v", got, m)
	}

	sl := []any{int64(5), []byte("b")}
	if _, got := roundTrip(t, s, sl); !reflect.DeepEqual(got, sl) {
		t.Fatalf("round trip = %#v, want %#v", got, sl)
	}
}

func TestJSONSerializerEncodeUnregisteredType(t *testing.T) {
	s := NewJSONSerializer()
	type custom struct{ X int }
	for _, v := range []any{custom{X: 1}, float32(1.5), make(chan int)} {
		if _, _, err := s.DumpsTyped(v); err == nil {
			t.Fatalf("DumpsTyped(%T) succeeded, want error", v)
		}
	}
}

func TestJSONSerializerDecodeUnknownEnvelope(t *testing.T) {
	s := NewJSONSerializer()

	if _, err := s.LoadsTyped("json+envelope", []byte(`{"__type__":"no.Such","data":1}`)); err == nil {
		t.Fatal("LoadsTyped with unknown envelope name succeeded, want error")
	}
	if _, err := s.LoadsTyped("bogus-tag", []byte(`"x"`)); err == nil {
		t.Fatal("LoadsTyped with unknown tag succeeded, want error")
	}
}

func TestJSONSerializerInt64Precision(t *testing.T) {
	s := NewJSONSerializer()

	// Values above 2^53 must round-trip exactly, not via float64.
	big64 := int64(1<<60 + 12345)
	if _, got := roundTrip(t, s, big64); got != big64 {
		t.Fatalf("int64 round trip = %v, want %v", got, big64)
	}
	bigNeg := int(-(1 << 60))
	if _, got := roundTrip(t, s, bigNeg); got != bigNeg {
		t.Fatalf("int round trip = %v, want %v", got, bigNeg)
	}
	if _, got := roundTrip(t, s, int64(-9223372036854775808)); got != int64(-9223372036854775808) {
		t.Fatalf("int64 min round trip = %v, want %v", got, int64(-9223372036854775808))
	}
}

func TestJSONSerializerDecodeInvalidJSON(t *testing.T) {
	s := NewJSONSerializer()

	for _, tag := range []string{"json", "json+envelope"} {
		if _, err := s.LoadsTyped(tag, []byte("{not json")); err == nil {
			t.Fatalf("LoadsTyped(%q, invalid JSON) succeeded, want error", tag)
		}
	}
}

// TestJSONSerializerEncodeNestedUnregistered pins that unregistered values
// nested inside maps, slices, and envelope payloads propagate an encode
// error instead of being silently dropped.
func TestJSONSerializerEncodeNestedUnregistered(t *testing.T) {
	s := NewJSONSerializer()
	type custom struct{ X int }

	cases := map[string]any{
		"in map":        map[string]any{"ok": 1.5, "bad": custom{X: 1}},
		"in slice":      []any{"ok", custom{X: 2}},
		"send arg":      types.Send{Node: "n", Arg: map[string]any{"bad": make(chan int)}},
		"interrupt val": types.Interrupt{Value: custom{X: 3}, ID: "i"},
		// A registered envelope nested in a slice whose own payload fails.
		"nested envelope": []any{types.Send{Node: "n", Arg: map[string]any{"bad": custom{X: 4}}}},
	}
	for name, v := range cases {
		t.Run(name, func(t *testing.T) {
			if _, _, err := s.DumpsTyped(v); err == nil {
				t.Fatalf("DumpsTyped(%#v) succeeded, want error", v)
			}
		})
	}
}

// TestJSONSerializerMarshalError pins that a registered value whose fields
// are not JSON-marshalable (here a chan inside a message metadata map, which
// bypasses registry validation) surfaces a marshal error from DumpsTyped.
func TestJSONSerializerMarshalError(t *testing.T) {
	s := NewJSONSerializer()

	msg := messages.Message{
		Role:             messages.RoleAI,
		Content:          "hi",
		AdditionalKwargs: map[string]any{"bad": make(chan int)},
	}
	if _, _, err := s.DumpsTyped(msg); err == nil {
		t.Fatal("DumpsTyped(message with chan in AdditionalKwargs) succeeded, want error")
	}
}

// TestJSONSerializerDecodeNestedEnvelopeErrors pins that malformed envelopes
// nested inside plain-JSON maps and slices propagate a decode error.
func TestJSONSerializerDecodeNestedEnvelopeErrors(t *testing.T) {
	s := NewJSONSerializer()

	for _, doc := range []string{
		`{"a":{"__type__":"no.Such","data":1}}`,
		`[{"ok":1},{"__type__":"no.Such","data":1}]`,
	} {
		if _, err := s.LoadsTyped("json", []byte(doc)); err == nil {
			t.Fatalf("LoadsTyped(%s) succeeded, want error", doc)
		}
	}
}

// TestJSONSerializerMalformedSendInterrupt pins the field-level validation of
// the types.Send and types.Interrupt envelope payloads.
func TestJSONSerializerMalformedSendInterrupt(t *testing.T) {
	s := NewJSONSerializer()

	docs := []string{
		// Send: node must be a string.
		`{"__type__":"types.Send","data":{"node":1,"arg":{}}}`,
		// Send: a malformed envelope nested in arg propagates.
		`{"__type__":"types.Send","data":{"node":"n","arg":{"__type__":"no.Such","data":1}}}`,
		// Send: a non-nil arg that is not an object is an error.
		`{"__type__":"types.Send","data":{"node":"n","arg":[1]}}`,
		// Interrupt: id must be a string.
		`{"__type__":"types.Interrupt","data":{"id":1,"value":"x"}}`,
		// Interrupt: a malformed envelope nested in value propagates.
		`{"__type__":"types.Interrupt","data":{"id":"i","value":{"__type__":"no.Such","data":1}}}`,
	}
	for _, doc := range docs {
		if _, err := s.LoadsTyped("json+envelope", []byte(doc)); err == nil {
			t.Fatalf("LoadsTyped(%s) succeeded, want error", doc)
		}
	}
}

// TestJSONSerializerMalformedScalarPayloads pins payload validation for the
// scalar registry types: unparseable timestamps, invalid base64, and a
// non-array []string payload.
func TestJSONSerializerMalformedScalarPayloads(t *testing.T) {
	s := NewJSONSerializer()

	docs := []string{
		`{"__type__":"time.Time","data":"not-a-time"}`,
		`{"__type__":"[]byte","data":"!!!not-base64!!!"}`,
		`{"__type__":"[]string","data":"x"}`,
	}
	for _, doc := range docs {
		if _, err := s.LoadsTyped("json+envelope", []byte(doc)); err == nil {
			t.Fatalf("LoadsTyped(%s) succeeded, want error", doc)
		}
	}
}

// TestJSONSerializerSendNilArg pins that a Send with a nil Arg stays
// restorable: like typed nil slices, the typed nil map encodes as an empty
// object and decodes to an empty non-nil map (JSON's nil/empty asymmetry),
// so the non-map arg check must not fire on it.
func TestJSONSerializerSendNilArg(t *testing.T) {
	s := NewJSONSerializer()

	send := types.Send{Node: "worker"}
	_, got := roundTrip(t, s, send)
	decoded, ok := got.(types.Send)
	if !ok {
		t.Fatalf("round trip decoded as %T, want types.Send", got)
	}
	if decoded.Node != send.Node {
		t.Fatalf("round trip node = %q, want %q", decoded.Node, send.Node)
	}
	if decoded.Arg == nil || len(decoded.Arg) != 0 {
		t.Fatalf("round trip arg = %#v, want empty non-nil map", decoded.Arg)
	}
}

func TestJSONSerializerMalformedEnvelopes(t *testing.T) {
	s := NewJSONSerializer()

	names := []string{
		"messages.Message",
		"types.Send",
		"types.Interrupt",
		"time.Time",
		"[]byte",
		"int64",
		"int",
	}

	// Missing or null data is an error for every registry type except the
	// two slice types ([]messages.Message, []string), whose typed-nil
	// encoding is a null payload that decodes to the empty slice.
	for _, name := range names {
		for _, doc := range []string{
			`{"__type__":` + strconv.Quote(name) + `}`,
			`{"__type__":` + strconv.Quote(name) + `,"data":null}`,
		} {
			if _, err := s.LoadsTyped("json+envelope", []byte(doc)); err == nil {
				t.Fatalf("LoadsTyped(%s) succeeded, want error", doc)
			}
		}
	}

	// Wrongly shaped payloads are an error for every registry type.
	wrongShape := map[string]string{
		"messages.Message":   `{"__type__":"messages.Message","data":[]}`,
		"[]messages.Message": `{"__type__":"[]messages.Message","data":{}}`,
		"types.Send":         `{"__type__":"types.Send","data":"x"}`,
		"types.Interrupt":    `{"__type__":"types.Interrupt","data":"x"}`,
		"time.Time":          `{"__type__":"time.Time","data":123}`,
		"[]byte":             `{"__type__":"[]byte","data":123}`,
		"int64":              `{"__type__":"int64","data":1.5}`,
		"int":                `{"__type__":"int","data":"abc"}`,
		"[]string":           `{"__type__":"[]string","data":[1]}`,
	}
	for name, doc := range wrongShape {
		if _, err := s.LoadsTyped("json+envelope", []byte(doc)); err == nil {
			t.Fatalf("LoadsTyped(%s) succeeded, want error for %s", doc, name)
		}
	}

	// Unknown __type__ names are an error, nested or top-level.
	for _, doc := range []string{
		`{"__type__":"no.Such","data":1}`,
		`{"__type__":123,"data":1}`,
		`[1,2]`,
	} {
		if _, err := s.LoadsTyped("json+envelope", []byte(doc)); err == nil {
			t.Fatalf("LoadsTyped(%s) succeeded, want error", doc)
		}
	}
}
