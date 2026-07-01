package documentloaders

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestTextLoaderLoadsFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	if err := os.WriteFile(path, []byte("hello file"), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}
	loader := NewTextLoader(path)
	loader.Metadata = map[string]any{"kind": "note"}

	docs, err := Load(context.Background(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].PageContent != "hello file" {
		t.Fatalf("content: %q", docs[0].PageContent)
	}
	if docs[0].Metadata["source"] != path || docs[0].Metadata["kind"] != "note" {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}
}

func TestFileSystemBlobLoaderDirectoryOptions(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "a.txt"), "a")
	writeTestFile(t, filepath.Join(dir, "b.md"), "b")
	writeTestFile(t, filepath.Join(dir, ".hidden.txt"), "hidden")
	nested := filepath.Join(dir, "nested")
	if err := os.Mkdir(nested, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	writeTestFile(t, filepath.Join(nested, "c.txt"), "c")

	loader := NewFileSystemBlobLoader(dir)
	loader.Glob = "*.txt"
	blobs := collectBlobs(t, loader)
	if len(blobs) != 1 || blobs[0].Path != filepath.Join(dir, "a.txt") {
		t.Fatalf("non-recursive blobs: %#v", blobs)
	}
	if blobs[0].Mimetype != "text/plain; charset=utf-8" {
		t.Fatalf("mimetype: %q", blobs[0].Mimetype)
	}

	loader.Recursive = true
	blobs = collectBlobs(t, loader)
	if len(blobs) != 2 || blobs[0].Path != filepath.Join(dir, "a.txt") || blobs[1].Path != filepath.Join(nested, "c.txt") {
		t.Fatalf("recursive blobs: %#v", blobs)
	}

	loader.ShowHidden = true
	blobs = collectBlobs(t, loader)
	if len(blobs) != 3 || blobs[0].Path != filepath.Join(dir, ".hidden.txt") {
		t.Fatalf("hidden blobs: %#v", blobs)
	}
}

func TestGenericLoaderCombinesBlobLoaderAndParser(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "a.txt"), "alpha")
	writeTestFile(t, filepath.Join(dir, "b.txt"), "beta")

	blobLoader := NewFileSystemBlobLoader(dir)
	blobLoader.Glob = "*.txt"
	blobLoader.Metadata = map[string]any{"source": "override", "scope": "unit"}
	loader, err := NewGenericLoader(blobLoader, TextParser{})
	if err != nil {
		t.Fatalf("new generic loader: %v", err)
	}

	docs, err := Load(context.Background(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 2 || docs[0].PageContent != "alpha" || docs[1].PageContent != "beta" {
		t.Fatalf("docs: %#v", docs)
	}
	for _, doc := range docs {
		if doc.Metadata["source"] != "override" || doc.Metadata["scope"] != "unit" {
			t.Fatalf("metadata: %#v", doc.Metadata)
		}
	}
}

func TestGenericLoaderRequiresComponents(t *testing.T) {
	if _, err := NewGenericLoader(nil, TextParser{}); err == nil {
		t.Fatal("expected missing blob loader error")
	}
	if _, err := NewGenericLoader(NewFileSystemBlobLoader("x"), nil); err == nil {
		t.Fatal("expected missing blob parser error")
	}
}

func collectBlobs(t *testing.T, loader FileSystemBlobLoader) []Blob {
	t.Helper()
	iter, err := loader.YieldBlobs(context.Background())
	if err != nil {
		t.Fatalf("yield blobs: %v", err)
	}
	defer iter.Close()
	var blobs []Blob
	for {
		blob, ok, err := iter.Next(context.Background())
		if err != nil {
			t.Fatalf("next blob: %v", err)
		}
		if !ok {
			break
		}
		blobs = append(blobs, blob)
	}
	return blobs
}

func writeTestFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
