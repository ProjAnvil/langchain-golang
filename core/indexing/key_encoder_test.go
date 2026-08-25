package indexing

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestHashDocumentWithAlgorithmDeterministic(t *testing.T) {
	doc := documents.New("alpha", map[string]any{"source": "a"})

	first, err := HashDocumentWithAlgorithm(doc, KeyEncoderSHA256)
	if err != nil {
		t.Fatalf("HashDocumentWithAlgorithm(sha256) error = %v", err)
	}
	second, err := HashDocumentWithAlgorithm(doc, KeyEncoderSHA256)
	if err != nil {
		t.Fatalf("HashDocumentWithAlgorithm(sha256) error = %v", err)
	}
	if first != second {
		t.Fatalf("same doc hashed differently: %q vs %q", first, second)
	}
	if len(first) != 64 { // sha256 hex digest length
		t.Fatalf("sha256 digest length = %d, want 64", len(first))
	}
}

func TestHashDocumentWithAlgorithmLengths(t *testing.T) {
	doc := documents.New("alpha", nil)

	sha1Hash, err := HashDocumentWithAlgorithm(doc, KeyEncoderSHA1)
	if err != nil {
		t.Fatalf("sha1 error = %v", err)
	}
	if len(sha1Hash) != 40 {
		t.Fatalf("sha1 digest length = %d, want 40", len(sha1Hash))
	}

	sha512Hash, err := HashDocumentWithAlgorithm(doc, KeyEncoderSHA512)
	if err != nil {
		t.Fatalf("sha512 error = %v", err)
	}
	if len(sha512Hash) != 128 {
		t.Fatalf("sha512 digest length = %d, want 128", len(sha512Hash))
	}
}

func TestHashDocumentWithAlgorithmDistinct(t *testing.T) {
	doc := documents.New("alpha", nil)

	sha1Hash, err := HashDocumentWithAlgorithm(doc, KeyEncoderSHA1)
	if err != nil {
		t.Fatalf("sha1 error = %v", err)
	}
	sha256Hash, err := HashDocumentWithAlgorithm(doc, KeyEncoderSHA256)
	if err != nil {
		t.Fatalf("sha256 error = %v", err)
	}
	if sha1Hash == sha256Hash {
		t.Fatalf("different algorithms produced the same digest %q", sha1Hash)
	}
}

func TestHashDocumentWithAlgorithmDefaultMatchesHashDocument(t *testing.T) {
	doc := documents.New("alpha", map[string]any{"source": "a"})

	def, err := HashDocumentWithAlgorithm(doc, "")
	if err != nil {
		t.Fatalf("empty algorithm error = %v", err)
	}
	legacy, err := HashDocument(doc)
	if err != nil {
		t.Fatalf("HashDocument error = %v", err)
	}
	if def != legacy {
		t.Fatalf("empty algorithm %q != HashDocument %q", def, legacy)
	}
}

func TestHashDocumentWithAlgorithmBlake2b(t *testing.T) {
	doc := documents.New("alpha", nil)

	first, err := HashDocumentWithAlgorithm(doc, KeyEncoderBlake2b)
	if err != nil {
		t.Fatalf("blake2b error = %v", err)
	}
	if len(first) != 128 { // blake2b-512 hex digest length
		t.Fatalf("blake2b digest length = %d, want 128", len(first))
	}
	second, err := HashDocumentWithAlgorithm(doc, KeyEncoderBlake2b)
	if err != nil {
		t.Fatalf("blake2b error = %v", err)
	}
	if first != second {
		t.Fatalf("blake2b not deterministic: %q vs %q", first, second)
	}
	// Mirrors test_hashed_document.py::test_hashing: blake2b differs from
	// sha1/sha256/sha512 for the same document.
	for _, algorithm := range []KeyEncoder{KeyEncoderSHA1, KeyEncoderSHA256, KeyEncoderSHA512} {
		other, err := HashDocumentWithAlgorithm(doc, algorithm)
		if err != nil {
			t.Fatalf("%s error = %v", algorithm, err)
		}
		if other == first {
			t.Fatalf("blake2b and %s produced the same digest %q", algorithm, first)
		}
	}
}

func TestHashDocumentWithAlgorithmUnknown(t *testing.T) {
	doc := documents.New("alpha", nil)
	_, err := HashDocumentWithAlgorithm(doc, KeyEncoder("md5"))
	if err == nil {
		t.Fatalf("expected error for unknown algorithm, got nil")
	}
}
