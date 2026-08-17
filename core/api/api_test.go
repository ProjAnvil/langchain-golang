package api

import "testing"

func TestDeprecationFormat(t *testing.T) {
	got := (Deprecation{
		Name:        "OldThing",
		Since:       "0.3.0",
		Removal:     "1.0",
		Alternative: "NewThing",
		Message:     "Extra context.",
	}).Format()
	want := "OldThing is deprecated since 0.3.0 and will be removed in 1.0; use NewThing instead. Extra context."
	if got != want {
		t.Fatalf("Format() = %q, want %q", got, want)
	}
}

func TestDeprecationFormatVariants(t *testing.T) {
	tests := []struct {
		name string
		dep  Deprecation
		want string
	}{
		{
			name: "empty",
			dep:  Deprecation{},
			want: "This API is deprecated",
		},
		{
			name: "name only",
			dep:  Deprecation{Name: "Old"},
			want: "Old is deprecated",
		},
		{
			name: "since only",
			dep:  Deprecation{Name: "Old", Since: "0.1.0"},
			want: "Old is deprecated since 0.1.0",
		},
		{
			name: "removal without since",
			dep:  Deprecation{Name: "Old", Removal: "2.0"},
			want: "Old is deprecated and will be removed in 2.0",
		},
		{
			name: "alternative without message",
			dep:  Deprecation{Name: "Old", Alternative: "New"},
			want: "Old is deprecated; use New instead",
		},
		{
			name: "message without alternative",
			dep:  Deprecation{Name: "Old", Message: "No longer maintained."},
			want: "Old is deprecated. No longer maintained.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.dep.Format(); got != tt.want {
				t.Fatalf("Format() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDeprecationError(t *testing.T) {
	dep := Deprecation{Name: "Old", Alternative: "New"}
	var err error = dep
	if err.Error() != dep.Format() {
		t.Fatalf("Error() = %q, want %q", err.Error(), dep.Format())
	}
	want := "Old is deprecated; use New instead"
	if err.Error() != want {
		t.Fatalf("Error() = %q, want %q", err.Error(), want)
	}
}

func TestStableAndBeta(t *testing.T) {
	stable := Stable("Runnable")
	if stable.Status != StatusStable || stable.Name != "Runnable" {
		t.Fatalf("unexpected stable metadata: %#v", stable)
	}
	if stable.Message != "" || stable.Deprecation != nil {
		t.Fatalf("stable metadata should have no message/deprecation: %#v", stable)
	}

	beta := Beta("Streaming", "subject to change")
	if beta.Status != StatusBeta || beta.Name != "Streaming" || beta.Message != "subject to change" {
		t.Fatalf("unexpected beta metadata: %#v", beta)
	}
	if beta.Deprecation != nil {
		t.Fatalf("beta metadata should have no deprecation: %#v", beta)
	}
}

func TestIsInternalPath(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		{"langchain_core/_api/deprecation", true},
		{"_private", true},
		{"langchain_core/messages", false},
		{"", false},
		{"_", false},
		{"a/_/b", false},
		{"a/__init__", true},
	}
	for _, tt := range tests {
		if got := IsInternalPath(tt.path); got != tt.want {
			t.Errorf("IsInternalPath(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

func TestMetadataHelpers(t *testing.T) {
	meta := Deprecated(Deprecation{Name: "Old", Alternative: "New"})
	if meta.Status != StatusDeprecated || meta.Deprecation == nil {
		t.Fatalf("unexpected metadata: %#v", meta)
	}
	if !IsInternalPath("langchain_core/_api/deprecation") {
		t.Fatal("expected private path")
	}
	if IsInternalPath("langchain_core/messages") {
		t.Fatal("public path reported internal")
	}
}
