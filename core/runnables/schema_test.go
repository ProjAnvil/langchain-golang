package runnables

import "testing"

func TestConfigurableFieldSpec(t *testing.T) {
	field := ConfigurableField{
		ID:          "temperature",
		Annotation:  "float",
		Name:        "Temperature",
		Description: "model temperature",
		Default:     0.7,
		IsShared:    true,
	}
	spec := field.Spec()
	if spec.ID != field.ID ||
		spec.Annotation != field.Annotation ||
		spec.Name != field.Name ||
		spec.Description != field.Description ||
		spec.Default != field.Default ||
		spec.IsShared != field.IsShared {
		t.Fatalf("spec does not mirror field: %#v vs %#v", spec, field)
	}
	// Spec does not carry dependencies.
	if len(spec.Dependencies) != 0 {
		t.Fatalf("dependencies: %#v", spec.Dependencies)
	}
}
