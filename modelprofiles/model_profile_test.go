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
		FieldMidConversationSystemMessages,
		FieldMidConversationTools,
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

// TestMidConversationFieldKeys pins the exact snake_case key strings added by
// Python #41180 (langchain-core 1.6.10) so a typo in the constants cannot slip
// through.
func TestMidConversationFieldKeys(t *testing.T) {
	if FieldMidConversationSystemMessages != "mid_conversation_system_messages" {
		t.Errorf("FieldMidConversationSystemMessages = %q, want %q",
			FieldMidConversationSystemMessages, "mid_conversation_system_messages")
	}
	if FieldMidConversationTools != "mid_conversation_tools" {
		t.Errorf("FieldMidConversationTools = %q, want %q",
			FieldMidConversationTools, "mid_conversation_tools")
	}
	for _, field := range []string{FieldMidConversationSystemMessages, FieldMidConversationTools} {
		if !IsDeclaredProfileField(field) {
			t.Errorf("expected %q to be a declared profile field", field)
		}
	}
	// The new keys must no longer surface as unknown profile keys.
	profile := Profile{
		FieldMidConversationSystemMessages: true,
		FieldMidConversationTools:          false,
	}
	if got := UnknownProfileKeys(profile); len(got) != 0 {
		t.Errorf("expected no unknown keys, got %v", got)
	}
}

func TestProfileSupports(t *testing.T) {
	profile := Profile{
		FieldMidConversationTools: true,
		FieldStructuredOutput:     false,
		FieldMaxInputTokens:       128000,
	}

	if !profile.Supports(FieldMidConversationTools) {
		t.Errorf("expected Supports to report true for a present-and-true field")
	}
	if profile.Supports(FieldStructuredOutput) {
		t.Errorf("expected Supports to report false for a present-but-false field")
	}
	if profile.Supports(FieldAttachment) {
		t.Errorf("expected Supports to report false for an absent field")
	}
	if profile.Supports(FieldMaxInputTokens) {
		t.Errorf("expected Supports to report false for a non-boolean field")
	}
	var nilProfile Profile
	if nilProfile.Supports(FieldMidConversationTools) {
		t.Errorf("expected Supports on a nil profile to report false")
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
