package vectorstores

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
)

// Mirrors test_inmemory_dump_load (test_in_memory.py:86): search results are
// identical before and after a dump/load round trip.
func TestInMemoryDumpLoadRoundTrip(t *testing.T) {
	ctx := context.Background()
	embedding := embeddings.NewDeterministicFake(6)
	store, err := FromTexts(ctx, embedding, []string{"foo", "bar", "baz"})
	if err != nil {
		t.Fatalf("FromTexts: %v", err)
	}
	output, err := store.SimilaritySearchWithScore(ctx, "foo", 1)
	if err != nil {
		t.Fatalf("SimilaritySearchWithScore: %v", err)
	}

	testFile := filepath.Join(t.TempDir(), "test.json")
	if err := store.Dump(testFile); err != nil {
		t.Fatalf("Dump: %v", err)
	}

	loaded, err := LoadInMemory(testFile, embedding)
	if err != nil {
		t.Fatalf("LoadInMemory: %v", err)
	}
	loadedOutput, err := loaded.SimilaritySearchWithScore(ctx, "foo", 1)
	if err != nil {
		t.Fatalf("SimilaritySearchWithScore after load: %v", err)
	}
	if len(output) != len(loadedOutput) {
		t.Fatalf("result length: %d vs %d", len(output), len(loadedOutput))
	}
	for i := range output {
		if output[i].Document.ID != loadedOutput[i].Document.ID ||
			output[i].Document.PageContent != loadedOutput[i].Document.PageContent ||
			output[i].Score != loadedOutput[i].Score {
			t.Fatalf("result %d: %+v vs %+v", i, output[i], loadedOutput[i])
		}
	}
}

// The on-disk shape mirrors Python's dump: {"<id>": {"id", "vector", "text",
// "metadata"}} (in_memory.py:214-219, 537-546). Metadata survives the round
// trip, and Dump creates missing parent directories.
func TestInMemoryDumpFormatAndMetadata(t *testing.T) {
	ctx := context.Background()
	embedding := embeddings.NewDeterministicFake(6)
	store, err := FromTexts(ctx, embedding, []string{"foo"},
		WithIDs([]string{"doc-1"}), WithMetadatas([]map[string]any{{"source": "s3"}}))
	if err != nil {
		t.Fatalf("FromTexts: %v", err)
	}

	testFile := filepath.Join(t.TempDir(), "nested", "dir", "store.json")
	if err := store.Dump(testFile); err != nil {
		t.Fatalf("Dump: %v", err)
	}
	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	var records map[string]map[string]any
	if err := json.Unmarshal(data, &records); err != nil {
		t.Fatalf("dump is not a JSON object: %v", err)
	}
	record, ok := records["doc-1"]
	if !ok {
		t.Fatalf("missing record for doc-1: %s", data)
	}
	if record["id"] != "doc-1" || record["text"] != "foo" {
		t.Fatalf("record: %v", record)
	}
	if _, ok := record["vector"].([]any); !ok {
		t.Fatalf("record vector: %v", record["vector"])
	}
	metadata, ok := record["metadata"].(map[string]any)
	if !ok || metadata["source"] != "s3" {
		t.Fatalf("record metadata: %v", record["metadata"])
	}

	loaded, err := LoadInMemory(testFile, embedding)
	if err != nil {
		t.Fatalf("LoadInMemory: %v", err)
	}
	docs, err := loaded.GetByIDs(ctx, []string{"doc-1"})
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(docs) != 1 || docs[0].Metadata["source"] != "s3" {
		t.Fatalf("loaded doc: %+v", docs)
	}
}

// Load of a missing file surfaces a read error.
func TestLoadInMemoryMissingFile(t *testing.T) {
	_, err := LoadInMemory(filepath.Join(t.TempDir(), "does-not-exist.json"), embeddings.NewFake(4))
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

// Load of malformed JSON surfaces a decode error.
func TestLoadInMemoryMalformedJSON(t *testing.T) {
	testFile := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(testFile, []byte("{not json"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := LoadInMemory(testFile, embeddings.NewFake(4)); err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

// Dump error paths: encode failure (unmarshalable metadata), write failure
// (path is a directory), and parent-dir creation failure (a path component
// is an existing file). Also covers the stale-idSequence skip branch.
func TestInMemoryDumpErrors(t *testing.T) {
	ctx := context.Background()
	store, err := FromTexts(ctx, embeddings.NewFake(4), []string{"foo"},
		WithMetadatas([]map[string]any{{"bad": func() {}}}))
	if err != nil {
		t.Fatalf("FromTexts: %v", err)
	}
	if err := store.Dump(filepath.Join(t.TempDir(), "out.json")); err == nil {
		t.Fatal("expected encode error for unmarshalable metadata")
	}

	clean, err := FromTexts(ctx, embeddings.NewFake(4), []string{"foo"})
	if err != nil {
		t.Fatalf("FromTexts: %v", err)
	}
	if err := clean.Dump(t.TempDir()); err == nil {
		t.Fatal("expected write error when path is a directory")
	}

	blocker := filepath.Join(t.TempDir(), "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if err := clean.Dump(filepath.Join(blocker, "nested", "out.json")); err == nil {
		t.Fatal("expected MkdirAll error when a path component is a file")
	}

	// Stale idSequence entry (id without a document) is skipped.
	delete(clean.documents, clean.idSequence[0])
	stale := filepath.Join(t.TempDir(), "stale.json")
	if err := clean.Dump(stale); err != nil {
		t.Fatalf("Dump with stale idSequence: %v", err)
	}
	data, err := os.ReadFile(stale)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if string(data) != "{}" {
		t.Fatalf("expected empty dump, got %s", data)
	}
}
