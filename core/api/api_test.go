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
