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

func TestFileSystemBlobLoaderPathErrors(t *testing.T) {
	loader := FileSystemBlobLoader{}
	if _, err := loader.YieldBlobs(context.Background()); err == nil {
		t.Fatal("expected missing path error")
	}

	loader = NewFileSystemBlobLoader(filepath.Join(t.TempDir(), "missing"))
	if _, err := loader.YieldBlobs(context.Background()); err == nil {
		t.Fatal("expected stat error")
	}
}

func TestFileSystemBlobLoaderHiddenSingleFile(t *testing.T) {
	dir := t.TempDir()
	hidden := filepath.Join(dir, ".hidden.txt")
	writeTestFile(t, hidden, "secret")

	loader := NewFileSystemBlobLoader(hidden)
	if blobs := collectBlobs(t, loader); len(blobs) != 0 {
		t.Fatalf("expected hidden file to be skipped: %#v", blobs)
	}

	loader.ShowHidden = true
	blobs := collectBlobs(t, loader)
	if len(blobs) != 1 || blobs[0].Path != hidden {
		t.Fatalf("hidden blobs: %#v", blobs)
	}
}

func TestFileSystemBlobLoaderInvalidGlob(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "a.txt"), "a")

	loader := NewFileSystemBlobLoader(dir)
	loader.Glob = "["
	if _, err := loader.YieldBlobs(context.Background()); err == nil {
		t.Fatal("expected bad pattern error (non-recursive)")
	}

	loader.Recursive = true
	if _, err := loader.YieldBlobs(context.Background()); err == nil {
		t.Fatal("expected bad pattern error (recursive)")
	}
}

func TestFileSystemBlobLoaderContextCancellation(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "a.txt"), "a")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	loader := NewFileSystemBlobLoader(dir)
	if _, err := loader.YieldBlobs(ctx); err == nil {
		t.Fatal("expected context error (non-recursive)")
	}

	loader.Recursive = true
	if _, err := loader.YieldBlobs(ctx); err == nil {
		t.Fatal("expected context error (recursive)")
	}
}

func TestFileSystemBlobLoaderSkipsHiddenDirAndSubdirs(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "a.txt"), "a")

	hiddenDir := filepath.Join(dir, ".config")
	if err := os.Mkdir(hiddenDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	writeTestFile(t, filepath.Join(hiddenDir, "b.txt"), "b")

	loader := NewFileSystemBlobLoader(dir)
	loader.Glob = "*"
	loader.Recursive = true
	blobs := collectBlobs(t, loader)
	if len(blobs) != 1 || blobs[0].Path != filepath.Join(dir, "a.txt") {
		t.Fatalf("blobs: %#v", blobs)
	}

	// Non-recursive glob "*" matches the hidden dir entry; it must be skipped.
	loader.Recursive = false
	loader.ShowHidden = true
	blobs = collectBlobs(t, loader)
	if len(blobs) != 1 || blobs[0].Path != filepath.Join(dir, "a.txt") {
		t.Fatalf("blobs: %#v", blobs)
	}
}

func TestFileSystemBlobLoaderExplicitMimetype(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "data.bin"), "x")

	loader := NewFileSystemBlobLoader(dir)
	loader.Mimetype = "application/octet-stream"
	blobs := collectBlobs(t, loader)
	if len(blobs) != 1 || blobs[0].Mimetype != "application/octet-stream" {
		t.Fatalf("blobs: %#v", blobs)
	}
}

func TestFileBlobIteratorErrors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.txt")
	writeTestFile(t, path, "a")

	loader := NewFileSystemBlobLoader(dir)
	iter, err := loader.YieldBlobs(context.Background())
	if err != nil {
		t.Fatalf("yield blobs: %v", err)
	}
	defer iter.Close()

	// Deleting the file after matching forces a read error in Next.
	if err := os.Remove(path); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if _, _, err := iter.Next(context.Background()); err == nil {
		t.Fatal("expected read error")
	}

	// A cancelled context surfaces from Next before reading.
	writeTestFile(t, path, "a")
	iter2, err := loader.YieldBlobs(context.Background())
	if err != nil {
		t.Fatalf("yield blobs: %v", err)
	}
	defer iter2.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := iter2.Next(ctx); err == nil {
		t.Fatal("expected context error")
	}
}

func TestFileBlobIteratorDefaultSourceMetadata(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a.txt")
	writeTestFile(t, path, "a")

	loader := NewFileSystemBlobLoader(dir)
	blobs := collectBlobs(t, loader)
	if len(blobs) != 1 {
		t.Fatalf("blobs: %#v", blobs)
	}
	if blobs[0].Metadata["source"] != path {
		t.Fatalf("metadata: %#v", blobs[0].Metadata)
	}
}

func TestTextParserLazyParseMetadata(t *testing.T) {
	ctx := context.Background()

	// Nil metadata and no path: no source key is injected.
	docs, err := Parse(ctx, TextParser{}, Blob{Data: []byte("plain")})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "plain" {
		t.Fatalf("docs: %#v", docs)
	}
	if _, ok := docs[0].Metadata["source"]; ok {
		t.Fatalf("unexpected source metadata: %#v", docs[0].Metadata)
	}

	// Path without source metadata: source falls back to the blob path.
	docs, err = Parse(ctx, TextParser{}, Blob{Data: []byte("x"), Path: "/tmp/x.txt"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if docs[0].Metadata["source"] != "/tmp/x.txt" {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}

	// Existing source metadata wins over the blob path.
	docs, err = Parse(ctx, TextParser{}, Blob{
		Data:     []byte("x"),
		Path:     "/tmp/x.txt",
		Metadata: map[string]any{"source": "kept"},
	})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if docs[0].Metadata["source"] != "kept" {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}
}

func TestGenericLoaderLazyLoadError(t *testing.T) {
	loader, err := NewGenericLoader(errBlobLoader{err: errTest}, TextParser{})
	if err != nil {
		t.Fatalf("new generic loader: %v", err)
	}
	if _, err := loader.LazyLoad(context.Background()); err == nil {
		t.Fatal("expected yield blobs error")
	}
}

func TestGenericLoaderIteratorErrors(t *testing.T) {
	ctx := context.Background()
	blob := NewBlobFromData([]byte("x"), "text/plain", nil)

	t.Run("blob iterator error", func(t *testing.T) {
		iter := &genericLoaderIterator{
			ctx:      ctx,
			blobIter: &fakeBlobIterator{nextErr: errTest},
			parser:   TextParser{},
		}
		if _, _, err := iter.Next(ctx); err == nil {
			t.Fatal("expected blob iterator error")
		}
	})

	t.Run("parser error", func(t *testing.T) {
		iter := &genericLoaderIterator{
			ctx:      ctx,
			blobIter: &fakeBlobIterator{blobs: []Blob{blob}},
			parser:   errParser{err: errTest},
		}
		if _, _, err := iter.Next(ctx); err == nil {
			t.Fatal("expected parser error")
		}
	})

	t.Run("document iterator next error", func(t *testing.T) {
		iter := &genericLoaderIterator{
			ctx:      ctx,
			blobIter: &fakeBlobIterator{blobs: []Blob{blob}},
			parser:   iterParser{iter: &errIterator{nextErr: errTest}},
		}
		if _, _, err := iter.Next(ctx); err == nil {
			t.Fatal("expected document iterator error")
		}
	})

	t.Run("document iterator close error", func(t *testing.T) {
		iter := &genericLoaderIterator{
			ctx:      ctx,
			blobIter: &fakeBlobIterator{blobs: []Blob{blob}},
			parser:   iterParser{iter: &errIterator{closeErr: errTest}},
		}
		if _, _, err := iter.Next(ctx); err == nil {
			t.Fatal("expected document iterator close error")
		}
	})

	t.Run("exhausted blob iterator ends iteration", func(t *testing.T) {
		iter := &genericLoaderIterator{
			ctx:      ctx,
			blobIter: &fakeBlobIterator{},
			parser:   TextParser{},
		}
		doc, ok, err := iter.Next(ctx)
		if err != nil || ok {
			t.Fatalf("expected end of iteration, got doc=%#v ok=%v err=%v", doc, ok, err)
		}
	})
}

func TestGenericLoaderIteratorClose(t *testing.T) {
	ctx := context.Background()
	blob := NewBlobFromData([]byte("x"), "text/plain", nil)

	// Close with an active document iterator reports its close error first.
	iter := &genericLoaderIterator{
		ctx:      ctx,
		blobIter: &fakeBlobIterator{blobs: []Blob{blob}, closeErr: errTest},
		parser:   TextParser{},
	}
	if _, ok, err := iter.Next(ctx); err != nil || !ok {
		t.Fatalf("next: ok=%v err=%v", ok, err)
	}
	if err := iter.Close(); err == nil {
		t.Fatal("expected blob iterator close error")
	}
	if iter.current != nil {
		t.Fatal("expected current iterator to be released")
	}

	// Close error from the active document iterator propagates.
	iter = &genericLoaderIterator{
		ctx:      ctx,
		blobIter: &fakeBlobIterator{},
		parser:   TextParser{},
		current:  &errIterator{closeErr: errTest},
	}
	if err := iter.Close(); err == nil {
		t.Fatal("expected document iterator close error")
	}
}

func TestTextLoaderMissingFile(t *testing.T) {
	loader := NewTextLoader(filepath.Join(t.TempDir(), "missing.txt"))
	if _, err := loader.LazyLoad(context.Background()); err == nil {
		t.Fatal("expected lazy load error")
	}
}

type errBlobLoader struct {
	err error
}

func (l errBlobLoader) YieldBlobs(context.Context) (BlobIterator, error) {
	return nil, l.err
}

// fakeBlobIterator is a BlobIterator over in-memory blobs that can fail on
// Next and Close.
type fakeBlobIterator struct {
	blobs    []Blob
	index    int
	nextErr  error
	closeErr error
}

func (i *fakeBlobIterator) Next(context.Context) (Blob, bool, error) {
	if i.nextErr != nil {
		return Blob{}, false, i.nextErr
	}
	if i.index >= len(i.blobs) {
		return Blob{}, false, nil
	}
	blob := i.blobs[i.index]
	i.index++
	return blob, true, nil
}

func (i *fakeBlobIterator) Close() error {
	return i.closeErr
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
