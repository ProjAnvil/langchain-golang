package tools

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

// echoArgs is the reflected argument struct shared across FromFunc tests.
type echoArgs struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

func TestFromFunc_StructArgs(t *testing.T) {
	tool, err := FromFunc("search", "search things", func(ctx context.Context, a echoArgs) (Result, error) {
		return Result{Content: a.Query}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	if tool.Name() != "search" {
		t.Errorf("name = %q", tool.Name())
	}
	if tool.Description() != "search things" {
		t.Errorf("description = %q", tool.Description())
	}
	s := tool.ArgsSchema()
	if s["type"] != "object" {
		t.Errorf("schema type = %v, want object", s["type"])
	}
	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties not map[string]any, got %T", s["properties"])
	}
	queryProp, ok := props["query"]
	if !ok {
		t.Fatalf("expected 'query' property, got %v", s)
	}
	if q, ok := queryProp.(schema.Schema); !ok || q["type"] != "string" {
		t.Errorf("query prop = %#v, want type=string", queryProp)
	}
	limitProp, ok := props["limit"]
	if !ok {
		t.Fatalf("expected 'limit' property, got %v", s)
	}
	if l, ok := limitProp.(schema.Schema); !ok || l["type"] != "integer" {
		t.Errorf("limit prop = %#v, want type=integer", limitProp)
	}
	// Both fields are non-pointer, non-omitempty → required.
	required, _ := s["required"].([]string)
	if !containsString(required, "query") || !containsString(required, "limit") {
		t.Errorf("required = %v, want [query limit]", required)
	}
	res, err := tool.Invoke(context.Background(), map[string]any{"query": "hi", "limit": 3})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "hi" {
		t.Errorf("content = %q", res.Content)
	}
}

func TestFromFunc_RejectsNonFunc(t *testing.T) {
	cases := []struct {
		name string
		fn   any
	}{
		{"int", 42},
		{"string", "hi"},
		{"nil", nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := FromFunc("x", "x", c.fn)
			if err == nil {
				t.Fatalf("expected error for %s", c.name)
			}
			if !errors.Is(err, ErrNotAFunc) {
				t.Errorf("expected ErrNotAFunc, got %v", err)
			}
		})
	}
}

func TestFromFunc_RejectsBadSignature(t *testing.T) {
	cases := []struct {
		name string
		fn   any
	}{
		{"too many user args", func(ctx context.Context, a, b echoArgs) (Result, error) { return Result{}, nil }},
		{"missing context", func(a echoArgs) (Result, error) { return Result{}, nil }},
		{"wrong return count", func(ctx context.Context, a echoArgs) Result { return Result{} }},
		{"non-error second return", func(ctx context.Context, a echoArgs) (Result, int) { return Result{}, 0 }},
		{"non-Result/any first return", func(ctx context.Context, a echoArgs) (int, error) { return 0, nil }},
		{"non-struct/map arg", func(ctx context.Context, a int) (Result, error) { return Result{}, nil }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := FromFunc("x", "x", c.fn); err == nil {
				t.Fatalf("expected error for %s", c.name)
			} else if !errors.Is(err, ErrInvalidSignature) {
				t.Errorf("expected ErrInvalidSignature for %s, got %v", c.name, err)
			}
		})
	}
}

func TestFromFunc_AnyReturn(t *testing.T) {
	// fn returns (any, error) — FromFunc must coerce into Result via JSON round-trip.
	tool, err := FromFunc("count", "counts", func(ctx context.Context, a echoArgs) (any, error) {
		return map[string]any{"content": a.Query}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	res, err := tool.Invoke(context.Background(), map[string]any{"query": "hello"})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "hello" {
		t.Errorf("content = %q, want hello", res.Content)
	}
}

func TestFromFunc_NoArg(t *testing.T) {
	tool, err := FromFunc("ping", "pings", func(ctx context.Context) (Result, error) {
		return Result{Content: "pong"}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	s := tool.ArgsSchema()
	if s["type"] != "object" {
		t.Errorf("schema type = %v, want object", s["type"])
	}
	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties not map[string]any: %#v", s["properties"])
	}
	if len(props) != 0 {
		t.Errorf("expected empty properties, got %v", props)
	}
	res, err := tool.Invoke(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "pong" {
		t.Errorf("content = %q", res.Content)
	}
}

func TestFromFunc_NoArgAnyReturn(t *testing.T) {
	tool, err := FromFunc("ping", "pings", func(ctx context.Context) (any, error) {
		return map[string]any{"content": "pong"}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	res, err := tool.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "pong" {
		t.Errorf("content = %q", res.Content)
	}
}

func TestFromFunc_AnyReturnScalar(t *testing.T) {
	// Non-object returns must land in Content (encoding/json refuses to
	// unmarshal a top-level scalar into a struct, so the invoker maps these
	// directly rather than round-tripping).
	cases := []struct {
		name string
		fn   any
		want string
	}{
		{"string", func(ctx context.Context) (any, error) { return "hi", nil }, "hi"},
		{"int", func(ctx context.Context) (any, error) { return 7, nil }, "7"},
		{"bool", func(ctx context.Context) (any, error) { return true, nil }, "true"},
		{"float", func(ctx context.Context) (any, error) { return 1.5, nil }, "1.5"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tool, err := FromFunc("scalar", "x", c.fn)
			if err != nil {
				t.Fatalf("FromFunc: %v", err)
			}
			res, err := tool.Invoke(context.Background(), nil)
			if err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			if res.Content != c.want {
				t.Errorf("content = %q, want %q", res.Content, c.want)
			}
		})
	}
}

func TestFromFunc_UnsupportedSliceElem(t *testing.T) {
	type badArgs struct {
		Ch chan int `json:"ch"`
	}
	if _, err := FromFunc("x", "x", func(ctx context.Context, a badArgs) (Result, error) {
		return Result{}, nil
	}); err == nil {
		t.Fatal("expected error for unsupported chan field")
	} else if !errors.Is(err, ErrUnsupportedType) {
		t.Errorf("expected ErrUnsupportedType, got %v", err)
	}
}

func TestFromFunc_PointerFieldOptional(t *testing.T) {
	type optArgs struct {
		Required string  `json:"required"`
		Optional *string `json:"optional"`
		Omit     string  `json:"omit,omitempty"`
	}
	tool, err := FromFunc("x", "x", func(ctx context.Context, a optArgs) (Result, error) {
		return Result{Content: a.Required}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	s := tool.ArgsSchema()
	props, _ := s["properties"].(map[string]any)
	if _, ok := props["required"]; !ok {
		t.Errorf("expected required field, got %v", props)
	}
	if _, ok := props["optional"]; !ok {
		t.Errorf("expected optional field in schema, got %v", props)
	}
	if _, ok := props["omit"]; !ok {
		t.Errorf("expected omit field in schema (omitempty must not drop it), got %v", props)
	}
	required, _ := s["required"].([]string)
	if !containsString(required, "required") {
		t.Errorf("required = %v, want [required]", required)
	}
	if containsString(required, "optional") {
		t.Errorf("optional (pointer) should not be required, got %v", required)
	}
	if containsString(required, "omit") {
		t.Errorf("omit (omitempty) should not be required, got %v", required)
	}
}

func TestFromFunc_MapArgs(t *testing.T) {
	tool, err := FromFunc("flex", "flex args", func(ctx context.Context, a map[string]any) (Result, error) {
		return Result{Content: "ok"}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	s := tool.ArgsSchema()
	if s["type"] != "object" {
		t.Errorf("schema type = %v, want object", s["type"])
	}
	res, err := tool.Invoke(context.Background(), map[string]any{"anything": 1})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "ok" {
		t.Errorf("content = %q", res.Content)
	}
}

func TestFromFunc_NestedStructAndSlice(t *testing.T) {
	type inner struct {
		Label string `json:"label"`
	}
	type args struct {
		Tags   []string `json:"tags"`
		Inner  inner    `json:"inner"`
		Counts []int    `json:"counts"`
	}
	tool, err := FromFunc("nested", "nested", func(ctx context.Context, a args) (Result, error) {
		return Result{Content: a.Inner.Label}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	s := tool.ArgsSchema()
	props, _ := s["properties"].(map[string]any)
	tags, ok := props["tags"].(schema.Schema)
	if !ok {
		t.Fatalf("tags prop missing or wrong type: %#v", props["tags"])
	}
	if tags["type"] != "array" {
		t.Errorf("tags type = %v, want array", tags["type"])
	}
	items, _ := tags["items"].(schema.Schema)
	if items["type"] != "string" {
		t.Errorf("tags items = %v, want string", items)
	}
	innerProp, ok := props["inner"].(schema.Schema)
	if !ok {
		t.Fatalf("inner prop missing or wrong type: %#v", props["inner"])
	}
	if innerProp["type"] != "object" {
		t.Errorf("inner type = %v, want object", innerProp["type"])
	}
	res, err := tool.Invoke(context.Background(), map[string]any{
		"tags":   []any{"a", "b"},
		"inner":  map[string]any{"label": "deep"},
		"counts": []any{1, 2, 3},
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "deep" {
		t.Errorf("content = %q, want deep", res.Content)
	}
}

func TestFromFunc_RejectsEmptyName(t *testing.T) {
	if _, err := FromFunc("", "x", func(ctx context.Context) (Result, error) { return Result{}, nil }); err == nil {
		t.Fatal("expected error for empty name")
	}
}

func TestFromFunc_InvokeMarshalsArgs(t *testing.T) {
	tool, err := FromFunc("echo", "echo", func(ctx context.Context, a echoArgs) (Result, error) {
		return Result{Content: a.Query, Metadata: map[string]any{"limit": a.Limit}}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	// Input map values are typed like JSON-decoded maps (e.g. float64 for numbers).
	// reflect-driven JSON round-trip must coerce them into int.
	res, err := tool.Invoke(context.Background(), map[string]any{
		"query": "hi",
		"limit": float64(7),
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "hi" {
		t.Errorf("content = %q", res.Content)
	}
	if v, _ := res.Metadata["limit"].(int); v != 7 {
		t.Errorf("metadata limit = %v (%T), want int 7", res.Metadata["limit"], res.Metadata["limit"])
	}
}

func TestFromFunc_UntaggedFieldUsesGoName(t *testing.T) {
	// A field with no json tag and one with an options-only tag
	// (`json:",omitempty"`) must fall back to the Go field name, mirroring
	// encoding/json. Without the fallback both would be keyed under "" and
	// produce a broken schema (a single property named "").
	type mixedArgs struct {
		Query    string `json:"query"`
		Untagged string
		OptsOnly string `json:",omitempty"`
	}
	tool, err := FromFunc("mixed", "mixed tags", func(ctx context.Context, a mixedArgs) (Result, error) {
		return Result{Content: a.Query + ":" + a.Untagged + ":" + a.OptsOnly}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	s := tool.ArgsSchema()
	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties not map[string]any: %#v", s["properties"])
	}
	// Each Go field name must be present as its own property key; the empty
	// key (the pre-fix bug) must NOT exist.
	if _, ok := props[""]; ok {
		t.Errorf("found empty-string property key (broken schema): %v", props)
	}
	for _, key := range []string{"query", "Untagged", "OptsOnly"} {
		prop, ok := props[key]
		if !ok {
			t.Errorf("expected property %q (Go field-name fallback), got %v", key, props)
			continue
		}
		if p, ok := prop.(schema.Schema); !ok || p["type"] != "string" {
			t.Errorf("property %q = %#v, want type=string", key, prop)
		}
	}
	// Untagged is non-pointer, non-omitempty → required; OptsOnly (omitempty)
	// → optional. Query is required.
	required, _ := s["required"].([]string)
	if !containsString(required, "Untagged") {
		t.Errorf("required = %v, want Untagged included", required)
	}
	if containsString(required, "OptsOnly") {
		t.Errorf("OptsOnly (omitempty) should not be required, got %v", required)
	}
	// Round-trip: invoking with the Go field names must populate the fields.
	res, err := tool.Invoke(context.Background(), map[string]any{
		"query":    "q",
		"Untagged": "u",
		"OptsOnly": "o",
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "q:u:o" {
		t.Errorf("content = %q, want q:u:o", res.Content)
	}
}

// containsString reports whether s contains target. Keeps the test file
// dependency-free.
func containsString(s []string, target string) bool {
	for _, v := range s {
		if v == target {
			return true
		}
	}
	return false
}

func TestFromFunc_RejectsInvalidMapArg(t *testing.T) {
	cases := []struct {
		name string
		fn   any
	}{
		{"non-string key", func(ctx context.Context, a map[int]any) (Result, error) { return Result{}, nil }},
		{"non-any value", func(ctx context.Context, a map[string]string) (Result, error) { return Result{}, nil }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := FromFunc("x", "x", c.fn); err == nil {
				t.Fatalf("expected error for %s", c.name)
			} else if !errors.Is(err, ErrInvalidSignature) {
				t.Errorf("expected ErrInvalidSignature for %s, got %v", c.name, err)
			}
		})
	}
}

func TestFromFunc_PointerArg(t *testing.T) {
	tool, err := FromFunc("ptr", "pointer args", func(ctx context.Context, a *echoArgs) (Result, error) {
		if a == nil {
			return Result{}, errors.New("expected non-nil args")
		}
		return Result{Content: a.Query}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	s := tool.ArgsSchema()
	if s["type"] != "object" {
		t.Errorf("schema type = %v, want object", s["type"])
	}
	res, err := tool.Invoke(context.Background(), map[string]any{"query": "hi", "limit": 1})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "hi" {
		t.Errorf("content = %q, want hi", res.Content)
	}
}

func TestFromFunc_ScalarFieldKinds(t *testing.T) {
	type scalarArgs struct {
		Ratio  float64 `json:"ratio"`
		Active bool    `json:"active"`
		Count  uint    `json:"count"`
	}
	tool, err := FromFunc("scalars", "scalar fields", func(ctx context.Context, a scalarArgs) (Result, error) {
		return Result{}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	props, _ := tool.ArgsSchema()["properties"].(map[string]any)
	want := map[string]string{"ratio": "number", "active": "boolean", "count": "integer"}
	for name, typ := range want {
		prop, ok := props[name].(schema.Schema)
		if !ok {
			t.Fatalf("property %q missing or wrong type: %#v", name, props[name])
		}
		if prop["type"] != typ {
			t.Errorf("property %q type = %v, want %v", name, prop["type"], typ)
		}
	}
}

func TestFromFunc_MapFieldSchemas(t *testing.T) {
	type mapArgs struct {
		Counts map[string]int `json:"counts"`
		Labels map[string]any `json:"labels"`
	}
	tool, err := FromFunc("maps", "map fields", func(ctx context.Context, a mapArgs) (Result, error) {
		return Result{}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	props, _ := tool.ArgsSchema()["properties"].(map[string]any)
	counts, ok := props["counts"].(schema.Schema)
	if !ok {
		t.Fatalf("counts property missing or wrong type: %#v", props["counts"])
	}
	if counts["type"] != "object" {
		t.Errorf("counts type = %v, want object", counts["type"])
	}
	addl, ok := counts["additionalProperties"].(schema.Schema)
	if !ok {
		t.Fatalf("counts additionalProperties missing: %#v", counts)
	}
	if addl["type"] != "integer" {
		t.Errorf("counts additionalProperties type = %v, want integer", addl["type"])
	}
	labels, ok := props["labels"].(schema.Schema)
	if !ok {
		t.Fatalf("labels property missing or wrong type: %#v", props["labels"])
	}
	if labels["type"] != "object" {
		t.Errorf("labels type = %v, want object", labels["type"])
	}
	// map[string]any stays open-ended: no additionalProperties constraint.
	if _, ok := labels["additionalProperties"]; ok {
		t.Errorf("labels should have no additionalProperties, got %#v", labels)
	}
}

func TestFromFunc_UnsupportedFieldTypes(t *testing.T) {
	t.Run("non-string map key", func(t *testing.T) {
		type badArgs struct {
			M map[int]string `json:"m"`
		}
		if _, err := FromFunc("x", "x", func(ctx context.Context, a badArgs) (Result, error) {
			return Result{}, nil
		}); err == nil {
			t.Fatal("expected error for non-string map key")
		} else if !errors.Is(err, ErrUnsupportedType) {
			t.Errorf("expected ErrUnsupportedType, got %v", err)
		}
	})
	t.Run("unsupported map value", func(t *testing.T) {
		type badArgs struct {
			M map[string]chan int `json:"m"`
		}
		if _, err := FromFunc("x", "x", func(ctx context.Context, a badArgs) (Result, error) {
			return Result{}, nil
		}); err == nil {
			t.Fatal("expected error for unsupported map value")
		} else if !errors.Is(err, ErrUnsupportedType) {
			t.Errorf("expected ErrUnsupportedType, got %v", err)
		}
	})
	t.Run("unsupported slice element", func(t *testing.T) {
		type badArgs struct {
			Chans []chan int `json:"chans"`
		}
		if _, err := FromFunc("x", "x", func(ctx context.Context, a badArgs) (Result, error) {
			return Result{}, nil
		}); err == nil {
			t.Fatal("expected error for unsupported slice element")
		} else if !errors.Is(err, ErrUnsupportedType) {
			t.Errorf("expected ErrUnsupportedType, got %v", err)
		}
	})
}

func TestFromFunc_DeeplyNestedStructRejected(t *testing.T) {
	type level4 struct {
		Leaf string `json:"leaf"`
	}
	type level3 struct {
		F level4 `json:"f"`
	}
	type level2 struct {
		F level3 `json:"f"`
	}
	type level1 struct {
		F level2 `json:"f"`
	}
	type root struct {
		F level1 `json:"f"`
	}
	if _, err := FromFunc("x", "x", func(ctx context.Context, a root) (Result, error) {
		return Result{}, nil
	}); err == nil {
		t.Fatal("expected error for struct nested deeper than maxStructDepth")
	} else if !errors.Is(err, ErrUnsupportedType) {
		t.Errorf("expected ErrUnsupportedType, got %v", err)
	}
}

func TestFromFunc_JSONTagSkip(t *testing.T) {
	type skipArgs struct {
		Shown  string `json:"shown"`
		Hidden string `json:"-"`
	}
	tool, err := FromFunc("skip", "skip fields", func(ctx context.Context, a skipArgs) (Result, error) {
		return Result{Content: a.Shown}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	props, _ := tool.ArgsSchema()["properties"].(map[string]any)
	if _, ok := props["shown"]; !ok {
		t.Errorf("expected shown property, got %v", props)
	}
	if _, ok := props["Hidden"]; ok {
		t.Errorf("json:\"-\" field should be skipped, got %v", props)
	}
	if _, ok := props["-"]; ok {
		t.Errorf("skip tag must not become a property key, got %v", props)
	}
	required, _ := tool.ArgsSchema()["required"].([]string)
	if containsString(required, "Hidden") || containsString(required, "-") {
		t.Errorf("skipped field should not be required, got %v", required)
	}
}

func TestFromFunc_UnexportedFieldsIgnored(t *testing.T) {
	// Unexported fields are invisible to the reflector and must not appear in
	// the schema, mirroring encoding/json's behaviour.
	type mixedVisArgs struct {
		Shown  string `json:"shown"`
		hidden string
	}
	tool, err := FromFunc("vis", "visibility", func(ctx context.Context, a mixedVisArgs) (Result, error) {
		return Result{Content: a.Shown}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	props, _ := tool.ArgsSchema()["properties"].(map[string]any)
	if len(props) != 1 {
		t.Fatalf("expected exactly one property, got %v", props)
	}
	if _, ok := props["shown"]; !ok {
		t.Errorf("expected shown property, got %v", props)
	}
}

func TestFromFunc_InvokeFuncError(t *testing.T) {
	want := errors.New("boom")
	tool, err := FromFunc("failer", "fails", func(ctx context.Context, a echoArgs) (Result, error) {
		return Result{}, want
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	if _, err := tool.Invoke(context.Background(), map[string]any{"query": "q", "limit": 1}); !errors.Is(err, want) {
		t.Fatalf("expected fn error to propagate, got %v", err)
	}
}

func TestFromFunc_InvokeArgErrors(t *testing.T) {
	tool, err := FromFunc("echo", "echo", func(ctx context.Context, a echoArgs) (Result, error) {
		return Result{Content: a.Query}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	// Unmarshal failure: query is a string field but receives a JSON object.
	if _, err := tool.Invoke(context.Background(), map[string]any{
		"query": map[string]any{"nested": true},
	}); err == nil {
		t.Fatal("expected unmarshal error for mismatched field type")
	}
	// Marshal failure: input contains a value encoding/json cannot represent.
	if _, err := tool.Invoke(context.Background(), map[string]any{
		"query": make(chan int),
	}); err == nil {
		t.Fatal("expected marshal error for unrepresentable input value")
	}
}

func TestFromFunc_AnyReturnBytesNilAndResult(t *testing.T) {
	cases := []struct {
		name    string
		fn      any
		want    string
		wantErr bool
	}{
		{"bytes", func(ctx context.Context) (any, error) { return []byte("raw"), nil }, "raw", false},
		{"nil", func(ctx context.Context) (any, error) { return nil, nil }, "", false},
		{"result passthrough", func(ctx context.Context) (any, error) {
			return Result{Content: "direct"}, nil
		}, "direct", false},
		{"json array cannot coerce", func(ctx context.Context) (any, error) {
			return []any{1, 2}, nil
		}, "", true},
		{"unmarshalable value", func(ctx context.Context) (any, error) {
			return func() {}, nil
		}, "", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tool, err := FromFunc("anyret", "x", c.fn)
			if err != nil {
				t.Fatalf("FromFunc: %v", err)
			}
			res, err := tool.Invoke(context.Background(), nil)
			if c.wantErr {
				if err == nil {
					t.Fatal("expected coercion error")
				}
				return
			}
			if err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			if res.Content != c.want {
				t.Errorf("content = %q, want %q", res.Content, c.want)
			}
		})
	}
}

func TestFromFunc_AnyReturnStructRoundTrip(t *testing.T) {
	// A struct with json tags round-trips into Result fields.
	type payload struct {
		Content  string         `json:"content"`
		Metadata map[string]any `json:"metadata"`
	}
	tool, err := FromFunc("structret", "x", func(ctx context.Context) (any, error) {
		return payload{Content: "body", Metadata: map[string]any{"k": "v"}}, nil
	})
	if err != nil {
		t.Fatalf("FromFunc: %v", err)
	}
	res, err := tool.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if res.Content != "body" {
		t.Errorf("content = %q, want body", res.Content)
	}
	if res.Metadata["k"] != "v" {
		t.Errorf("metadata = %v, want k=v", res.Metadata)
	}
}
