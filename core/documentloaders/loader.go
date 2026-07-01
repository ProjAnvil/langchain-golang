// Package documentloaders defines base document loading contracts.
package documentloaders

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/projanvil/langchain-golang/core/documents"
)

// TextSplitter is the document-splitting subset used by loaders.
type TextSplitter interface {
	SplitDocuments([]documents.Document) []documents.Document
}

// TextSplitterFactory creates a splitter for LoadAndSplit when callers do not
// provide one explicitly. Packages outside core can register a default without
// forcing documentloaders to depend on a concrete splitter package.
type TextSplitterFactory func() (TextSplitter, error)

var defaultTextSplitterFactory struct {
	sync.RWMutex
	fn TextSplitterFactory
}

// Loader eagerly loads documents.
type Loader interface {
	Load(ctx context.Context) ([]documents.Document, error)
}

// LazyLoader streams documents one at a time.
type LazyLoader interface {
	LazyLoad(ctx context.Context) (DocumentIterator, error)
}

// DocumentIterator iterates documents.
type DocumentIterator interface {
	Next(ctx context.Context) (documents.Document, bool, error)
	Close() error
}

// Load consumes a LazyLoader into a slice.
func Load(ctx context.Context, loader LazyLoader) ([]documents.Document, error) {
	iter, err := loader.LazyLoad(ctx)
	if err != nil {
		return nil, err
	}
	defer iter.Close()

	var docs []documents.Document
	for {
		doc, ok, err := iter.Next(ctx)
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

// LoadAndSplit loads documents and splits them. If splitter is nil, an error is
// returned unless a default splitter factory has been registered.
func LoadAndSplit(ctx context.Context, loader LazyLoader, splitter TextSplitter) ([]documents.Document, error) {
	if splitter == nil {
		var err error
		splitter, err = newDefaultTextSplitter()
		if err != nil {
			return nil, err
		}
	}
	docs, err := Load(ctx, loader)
	if err != nil {
		return nil, err
	}
	return splitter.SplitDocuments(docs), nil
}

// RegisterDefaultTextSplitterFactory registers a default splitter factory for
// LoadAndSplit. Passing nil clears the registered factory.
func RegisterDefaultTextSplitterFactory(factory TextSplitterFactory) {
	defaultTextSplitterFactory.Lock()
	defer defaultTextSplitterFactory.Unlock()
	defaultTextSplitterFactory.fn = factory
}

func newDefaultTextSplitter() (TextSplitter, error) {
	defaultTextSplitterFactory.RLock()
	factory := defaultTextSplitterFactory.fn
	defaultTextSplitterFactory.RUnlock()
	if factory == nil {
		return nil, fmt.Errorf("text splitter is required; pass a splitter or register a default text splitter factory")
	}
	splitter, err := factory()
	if err != nil {
		return nil, err
	}
	if splitter == nil {
		return nil, fmt.Errorf("default text splitter factory returned nil")
	}
	return splitter, nil
}

// SliceIterator is a simple in-memory DocumentIterator.
type SliceIterator struct {
	docs  []documents.Document
	index int
}

// NewSliceIterator creates an iterator over documents.
func NewSliceIterator(docs []documents.Document) *SliceIterator {
	copied := make([]documents.Document, len(docs))
	for i, doc := range docs {
		copied[i] = doc.Clone()
	}
	return &SliceIterator{docs: copied}
}

// Next returns the next document.
func (i *SliceIterator) Next(_ context.Context) (documents.Document, bool, error) {
	if i.index >= len(i.docs) {
		return documents.Document{}, false, nil
	}
	doc := i.docs[i.index].Clone()
	i.index++
	return doc, true, nil
}

// Close releases iterator resources.
func (i *SliceIterator) Close() error {
	i.index = len(i.docs)
	return nil
}

// Blob is raw data plus metadata for blob parsers.
type Blob struct {
	Data     []byte
	Path     string
	Mimetype string
	Metadata map[string]any
	Encoding string
}

// NewBlobFromData creates an in-memory blob.
func NewBlobFromData(data []byte, mimetype string, metadata map[string]any) Blob {
	return Blob{
		Data:     append([]byte(nil), data...),
		Mimetype: mimetype,
		Metadata: cloneMetadata(metadata),
		Encoding: "utf-8",
	}
}

// NewBlobFromPath reads a file into a blob.
func NewBlobFromPath(path string, mimetype string, metadata map[string]any) (Blob, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Blob{}, err
	}
	blob := NewBlobFromData(data, mimetype, metadata)
	blob.Path = path
	if blob.Mimetype == "" {
		blob.Mimetype = blobMimetype(path, "")
	}
	return blob, nil
}

// Reader returns a reader over blob data.
func (b Blob) Reader() io.Reader {
	return bytes.NewReader(b.Data)
}

// Source returns metadata["source"] when present, otherwise the blob path.
func (b Blob) Source() string {
	if source, ok := b.Metadata["source"].(string); ok {
		return source
	}
	return b.Path
}

// AsBytes returns a defensive copy of the blob data.
func (b Blob) AsBytes() []byte {
	return append([]byte(nil), b.Data...)
}

// AsString decodes blob data. The current Go loader stores bytes in memory and
// supports UTF-8 text by default.
func (b Blob) AsString() string {
	return string(b.Data)
}

// BlobParser lazily parses a blob into documents.
type BlobParser interface {
	LazyParse(ctx context.Context, blob Blob) (DocumentIterator, error)
}

// Parse consumes a BlobParser into a slice.
func Parse(ctx context.Context, parser BlobParser, blob Blob) ([]documents.Document, error) {
	iter, err := parser.LazyParse(ctx, blob)
	if err != nil {
		return nil, err
	}
	defer iter.Close()
	var docs []documents.Document
	for {
		doc, ok, err := iter.Next(ctx)
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		docs = append(docs, doc)
	}
	return docs, nil
}
