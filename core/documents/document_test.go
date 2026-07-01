package documents

import "testing"

func TestDocumentCloneCopiesMetadata(t *testing.T) {
	doc := New("hello", map[string]any{"source": "unit"}).WithID("doc-1")
	clone := doc.Clone()
	clone.Metadata["source"] = "changed"

	if doc.Metadata["source"] != "unit" {
		t.Fatalf("metadata was mutated: %v", doc.Metadata["source"])
	}
	if clone.ID != "doc-1" {
		t.Fatalf("id: got %q want doc-1", clone.ID)
	}
}

func TestDocumentDefaultsMetadataToEmptyMap(t *testing.T) {
	doc := New("hello", nil)
	if doc.Metadata == nil || len(doc.Metadata) != 0 {
		t.Fatalf("metadata: %#v", doc.Metadata)
	}
	clone := doc.Clone()
	clone.Metadata["source"] = "unit"
	if len(doc.Metadata) != 0 {
		t.Fatalf("clone mutated original metadata: %#v", doc.Metadata)
	}
}

func TestDocumentHelpers(t *testing.T) {
	metadata := map[string]any{"source": "unit", "page": 1}
	doc := New("content", nil).WithID("doc-1").WithMetadata(metadata)
	metadata["source"] = "changed"

	if doc.ID != "doc-1" || doc.Source() != "unit" {
		t.Fatalf("unexpected doc helpers: %#v", doc)
	}
	value, ok := doc.MetadataValue("page")
	if !ok || value != 1 {
		t.Fatalf("metadata value = %v, %v", value, ok)
	}
}
