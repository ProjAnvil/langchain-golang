package schema

import (
	"reflect"
	"testing"
)

func TestObjectWithPropertiesAndRequired(t *testing.T) {
	props := map[string]Schema{
		"name": String("the name"),
		"age":  Integer(""),
	}
	got := Object(props, "name")

	if got["type"] != "object" {
		t.Errorf("type = %v, want object", got["type"])
	}
	gotProps, ok := got["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties has type %T, want map[string]any", got["properties"])
	}
	if len(gotProps) != 2 {
		t.Errorf("len(properties) = %d, want 2", len(gotProps))
	}
	if name, ok := gotProps["name"].(Schema); !ok || name["type"] != "string" {
		t.Errorf("properties[name] = %v, want a string schema", gotProps["name"])
	}
	req, ok := got["required"].([]string)
	if !ok {
		t.Fatalf("required has type %T, want []string", got["required"])
	}
	if !reflect.DeepEqual(req, []string{"name"}) {
		t.Errorf("required = %v, want [name]", req)
	}
}

func TestObjectWithoutRequiredOmitsKey(t *testing.T) {
	got := Object(map[string]Schema{})
	if _, present := got["required"]; present {
		t.Errorf("required should be omitted when no required fields are given, got %v", got["required"])
	}
}

func TestObjectCopiesRequiredSlice(t *testing.T) {
	required := []string{"a", "b"}
	got := Object(nil, required...)
	required[0] = "mutated"

	req := got["required"].([]string)
	if req[0] != "a" {
		t.Errorf("required[0] = %q, want %q: Object must copy the required slice", req[0], "a")
	}
}

func TestObjectCopiesPropertiesMap(t *testing.T) {
	props := map[string]Schema{"x": Boolean("")}
	got := Object(props)
	props["y"] = String("")

	gotProps := got["properties"].(map[string]any)
	if _, present := gotProps["y"]; present {
		t.Errorf("mutating the input properties map should not affect the returned schema")
	}
}

func TestScalarConstructors(t *testing.T) {
	tests := []struct {
		name     string
		build    func(string) Schema
		wantType string
	}{
		{"String", String, "string"},
		{"Integer", Integer, "integer"},
		{"Number", Number, "number"},
		{"Boolean", Boolean, "boolean"},
	}
	for _, tt := range tests {
		t.Run(tt.name+"WithDescription", func(t *testing.T) {
			got := tt.build("a description")
			if got["type"] != tt.wantType {
				t.Errorf("type = %v, want %v", got["type"], tt.wantType)
			}
			if got["description"] != "a description" {
				t.Errorf("description = %v, want %q", got["description"], "a description")
			}
		})
		t.Run(tt.name+"WithoutDescription", func(t *testing.T) {
			got := tt.build("")
			if got["type"] != tt.wantType {
				t.Errorf("type = %v, want %v", got["type"], tt.wantType)
			}
			if _, present := got["description"]; present {
				t.Errorf("description should be omitted for empty input, got %v", got["description"])
			}
		})
	}
}
