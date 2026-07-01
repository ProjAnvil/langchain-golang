package modelprofiles

import (
	"reflect"
	"testing"
)

func TestIsDeclaredProfileField(t *testing.T) {
	declared := []string{
		FieldMaxInputTokens,
		FieldToolCalling,
		FieldTemperature,
		FieldImageURLInputs,
		FieldPDFToolMsg,
	}
	for _, field := range declared {
		if !IsDeclaredProfileField(field) {
			t.Errorf("expected %q to be a declared profile field", field)
		}
	}
	for _, field := range []string{"future_field", "another", "unknown_future_field"} {
		if IsDeclaredProfileField(field) {
			t.Errorf("expected %q to not be a declared profile field", field)
		}
	}
}

func TestDeclaredProfileFieldsSorted(t *testing.T) {
	fields := DeclaredProfileFields()
	if len(fields) != len(declaredProfileFields) {
		t.Fatalf("expected %d fields, got %d", len(declaredProfileFields), len(fields))
	}
	for i := 1; i < len(fields); i++ {
		if fields[i-1] >= fields[i] {
			t.Fatalf("fields not sorted: %q >= %q", fields[i-1], fields[i])
		}
	}
}

func TestUnknownProfileKeys(t *testing.T) {
	profile := Profile{
		FieldMaxInputTokens: 100,
		"future_field":      true,
		"another":           "val",
	}
	got := UnknownProfileKeys(profile)
	want := []string{"another", "future_field"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("UnknownProfileKeys = %v, want %v", got, want)
	}
}

func TestUnknownProfileKeysDeclaredOnly(t *testing.T) {
	profile := Profile{
		FieldMaxInputTokens: 100,
		FieldToolCalling:    true,
	}
	if got := UnknownProfileKeys(profile); len(got) != 0 {
		t.Fatalf("expected no unknown keys, got %v", got)
	}
}

func TestUnknownProfileKeysEmpty(t *testing.T) {
	if got := UnknownProfileKeys(Profile{}); len(got) != 0 {
		t.Fatalf("expected no unknown keys for empty profile, got %v", got)
	}
	if got := UnknownProfileKeys(nil); got != nil {
		t.Fatalf("expected nil for nil profile, got %v", got)
	}
}
