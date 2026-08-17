package channels

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

// overwriteAppend is a tiny AppendSliceReducer-style helper local to these
// tests so the examples are self-contained (no dependency on the int-slice
// reducer machinery).
func overwriteAppend(existing any, update any) (any, error) {
	return AppendSliceReducer(existing, update)
}

func TestNewOverwriteType(t *testing.T) {
	// Mirror Python: Overwrite.type literal is always "__overwrite__".
	ow := NewOverwrite([]string{"b"})
	if ow.Type != OverwriteType {
		t.Fatalf("NewOverwrite(...).Type = %q, want %q", ow.Type, OverwriteType)
	}
	if !reflect.DeepEqual(ow.Value, []string{"b"}) {
		t.Fatalf("NewOverwrite(...).Value = %v, want [b]", ow.Value)
	}
}

func TestOverwriteReplacesValue(t *testing.T) {
	// (Gap-report baseline 1.) An Overwrite in the update slice REPLACES the
	// entire accumulated value, bypassing the reducer.
	ch := NewBinaryOperator(overwriteAppend)
	update(t, ch, []int{1, 2}) // seed + accumulate

	changed, err := ch.Update([]any{NewOverwrite([]int{99})})
	if err != nil {
		t.Fatalf("Update(Overwrite) error = %v", err)
	}
	if !changed {
		t.Fatal("Update(Overwrite) changed = false, want true")
	}
	if got := get(t, ch); !reflect.DeepEqual(got, []int{99}) {
		t.Fatalf("after Overwrite Get() = %v, want [99]", got)
	}
}

func TestOverwriteThenAccumulateAcrossSupersteps(t *testing.T) {
	// (Gap-report baseline 2.) Normal write then Overwrite (sequence across
	// super-steps): Overwrite wins (base reset), then later writes — in a
	// SUBSEQUENT super-step — accumulate via the reducer on the new base.
	ch := NewBinaryOperator(overwriteAppend)
	update(t, ch, []int{1, 2})             // super-step 1: [1 2]
	update(t, ch, NewOverwrite([]int{10})) // super-step 2: overwrite → [10]
	update(t, ch, []int{20})               // super-step 3: accumulate → [10 20]
	if got := get(t, ch); !reflect.DeepEqual(got, []int{10, 20}) {
		t.Fatalf("Get() = %v, want [10 20]", got)
	}
}

func TestOverwriteSkipsLaterWritesInSameSuperstep(t *testing.T) {
	// Python-accuracy: within one super-step, once an Overwrite is seen,
	// subsequent non-Overwrite values are SKIPPED (the `if not seen_overwrite`
	// guard in binop.update). They do NOT accumulate on the new base.
	ch := NewBinaryOperator(overwriteAppend)
	update(t, ch, []int{1}) // seed

	// In one super-step: a normal write, then Overwrite, then another normal
	// write. The Overwrite replaces; the trailing normal write is skipped.
	if _, err := ch.Update([]any{
		[]int{2},
		NewOverwrite([]int{99}),
		[]int{3}, // must be skipped — not applied via reducer
	}); err != nil {
		t.Fatalf("Update() error = %v", err)
	}
	if got := get(t, ch); !reflect.DeepEqual(got, []int{99}) {
		t.Fatalf("Get() = %v, want [99] (trailing write skipped)", got)
	}
}

func TestDoubleOverwriteErrors(t *testing.T) {
	// (Gap-report baseline 3.) Two Overwrites in one super-step is an error.
	ch := NewBinaryOperator(overwriteAppend)
	update(t, ch, []int{1}) // seed

	_, err := ch.Update([]any{
		NewOverwrite([]int{10}),
		NewOverwrite([]int{20}),
	})
	var iu *InvalidUpdateError
	if !errors.As(err, &iu) {
		t.Fatalf("Update(two Overwrites) error = %v, want *InvalidUpdateError", err)
	}
}

func TestOverwriteSurvivesJSONRoundtrip(t *testing.T) {
	// (Gap-report baseline 4.) Overwrite must survive JSON serialization
	// and deserialization: after round-trip the value arrives as a
	// map[string]any carrying the discriminator, and IsOverwrite /
	// AsOverwrite must still recognize it and recover the value.
	original := NewOverwrite([]int{7})
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("json.Marshal(Overwrite) error = %v", err)
	}

	// Confirm the wire form carries the discriminator fields.
	var wire map[string]any
	if err := json.Unmarshal(data, &wire); err != nil {
		t.Fatalf("json.Unmarshal into map error = %v", err)
	}
	if got, _ := wire["type"].(string); got != OverwriteType {
		t.Fatalf("wire type = %q, want %q", got, OverwriteType)
	}

	// Deserialize into a generic any (the checkpoint-serde path): the result
	// is a map[string]any, not a typed Overwrite.
	var generic any
	if err := json.Unmarshal(data, &generic); err != nil {
		t.Fatalf("json.Unmarshal into any error = %v", err)
	}

	if !IsOverwrite(generic) {
		t.Fatalf("IsOverwrite(json-roundtripped) = false, want true")
	}
	val, ok := AsOverwrite(generic)
	if !ok {
		t.Fatalf("AsOverwrite(json-roundtripped) ok = false, want true")
	}
	// JSON numbers deserialize as float64; compare the single element.
	slice, _ := val.([]any)
	if len(slice) != 1 {
		t.Fatalf("AsOverwrite value = %v, want 1-element slice", val)
	}
	if num, _ := slice[0].(float64); num != 7 {
		t.Fatalf("AsOverwrite value[0] = %v, want 7", slice[0])
	}

	// The round-tripped map form must drive the channel: it replaces the
	// accumulated value just like the typed struct.
	ch := NewBinaryOperator(overwriteAppend)
	update(t, ch, []int{1, 2})
	if _, err := ch.Update([]any{generic}); err != nil {
		t.Fatalf("Update(json-roundtripped Overwrite) error = %v", err)
	}
	if got := get(t, ch); !reflect.DeepEqual(got, []any{float64(7)}) {
		t.Fatalf("after round-tripped Overwrite Get() = %v, want [7]", got)
	}
}

func TestAsOverwriteRecognizesSentinelMap(t *testing.T) {
	// Python's _get_overwrite also recognizes the single-key sentinel map
	// {"__overwrite__": value}. Verify the Go port mirrors that.
	sentinel := map[string]any{OverwriteType: []int{42}}
	if !IsOverwrite(sentinel) {
		t.Fatal("IsOverwrite(sentinel map) = false, want true")
	}
	val, ok := AsOverwrite(sentinel)
	if !ok {
		t.Fatal("AsOverwrite(sentinel map) ok = false, want true")
	}
	if !reflect.DeepEqual(val, []int{42}) {
		t.Fatalf("AsOverwrite(sentinel) = %v, want [42]", val)
	}
}

func TestAsOverwritePointerForm(t *testing.T) {
	// The *Overwrite pointer form must be recognized just like the value form.
	ow := NewOverwrite([]int{5})
	val, ok := AsOverwrite(&ow)
	if !ok {
		t.Fatal("AsOverwrite(*Overwrite) ok = false, want true")
	}
	if !reflect.DeepEqual(val, []int{5}) {
		t.Fatalf("AsOverwrite(*Overwrite) = %v, want [5]", val)
	}
	if !IsOverwrite(&ow) {
		t.Fatal("IsOverwrite(*Overwrite) = false, want true")
	}
}

func TestAsOverwriteRejectsNonOverwrite(t *testing.T) {
	cases := []any{
		nil,
		42,
		"plain",
		[]int{1},
		map[string]any{"type": "other"},
		map[string]any{"value": 1},            // missing type discriminator
		map[string]any{"type": OverwriteType}, // missing value key
		map[string]any{"type": OverwriteType, "value": 1, "extra": 2}, // len != 1 but has both keys...
	}
	// NOTE: the last case {"type":"__overwrite__","value":1,"extra":2} is
	// recognized because both "type" and "value" keys are present; the
	// len==1 check only guards the sentinel form. Adjust the expectation
	// accordingly below.
	for i, v := range cases[:len(cases)-1] {
		if IsOverwrite(v) {
			t.Fatalf("case %d: IsOverwrite(%v) = true, want false", i, v)
		}
	}
	// The extra-key map is still recognized (type + value present).
	if !IsOverwrite(cases[len(cases)-1]) {
		t.Fatalf("IsOverwrite({type,value,extra}) = false, want true (type+value present)")
	}
}
