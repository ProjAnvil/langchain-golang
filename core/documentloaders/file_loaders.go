package documentloaders

import (
	"context"
	"fmt"
	"mime"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

// BlobLoader lazily loads raw blobs.
type BlobLoader interface {
	YieldBlobs(ctx context.Context) (BlobIterator, error)
}

// BlobIterator iterates blobs.
type BlobIterator interface {
	Next(ctx context.Context) (Blob, bool, error)
	Close() error
}

// FileSystemBlobLoader loads file blobs from a path or directory.
type FileSystemBlobLoader struct {
	Path       string
	Glob       string
	Recursive  bool
	ShowHidden bool
	Mimetype   string
	Metadata   map[string]any
}

// NewFileSystemBlobLoader creates a file-system blob loader.
func NewFileSystemBlobLoader(path string) FileSystemBlobLoader {
	return FileSystemBlobLoader{Path: path, Glob: "*"}
}

// YieldBlobs returns file blobs in deterministic path order.
func (l FileSystemBlobLoader) YieldBlobs(ctx context.Context) (BlobIterator, error) {
	paths, err := l.matchPaths(ctx)
	if err != nil {
		return nil, err
	}
	return &fileBlobIterator{loader: l, paths: paths}, nil
}

func (l FileSystemBlobLoader) matchPaths(ctx context.Context) ([]string, error) {
	if l.Path == "" {
		return nil, fmt.Errorf("path is required")
	}
	info, err := os.Stat(l.Path)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		if shouldSkipHidden(l.Path, l.ShowHidden) {
			return nil, nil
		}
		return []string{l.Path}, nil
	}

	pattern := l.Glob
	if pattern == "" {
		pattern = "*"
	}
	var paths []string
	if l.Recursive {
		err = filepath.WalkDir(l.Path, func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if path != l.Path && shouldSkipHidden(path, l.ShowHidden) {
				if entry.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			if entry.IsDir() {
				return nil
			}
			matched, err := filepath.Match(pattern, filepath.Base(path))
			if err != nil {
				return err
			}
			if matched {
				paths = append(paths, path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	} else {
		matches, err := filepath.Glob(filepath.Join(l.Path, pattern))
		if err != nil {
			return nil, err
		}
		for _, path := range matches {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			info, err := os.Stat(path)
			if err != nil {
				return nil, err
			}
			if info.IsDir() || shouldSkipHidden(path, l.ShowHidden) {
				continue
			}
			paths = append(paths, path)
		}
	}
	sort.Strings(paths)
	return paths, nil
}

type fileBlobIterator struct {
	loader FileSystemBlobLoader
	paths  []string
	index  int
}

func (i *fileBlobIterator) Next(ctx context.Context) (Blob, bool, error) {
	if i.index >= len(i.paths) {
		return Blob{}, false, nil
	}
	if err := ctx.Err(); err != nil {
		return Blob{}, false, err
	}
	path := i.paths[i.index]
	i.index++
	data, err := os.ReadFile(path)
	if err != nil {
		return Blob{}, false, err
	}
	metadata := cloneMetadata(i.loader.Metadata)
	if metadata == nil {
		metadata = map[string]any{}
	}
	if _, ok := metadata["source"]; !ok {
		metadata["source"] = path
	}
	blob := NewBlobFromData(data, blobMimetype(path, i.loader.Mimetype), metadata)
	blob.Path = path
	return blob, true, nil
}

func (i *fileBlobIterator) Close() error {
	i.index = len(i.paths)
	return nil
}

// TextParser parses text blobs into one document per blob.
type TextParser struct{}

// LazyParse parses a blob into a single text document.
func (p TextParser) LazyParse(_ context.Context, blob Blob) (DocumentIterator, error) {
	metadata := cloneMetadata(blob.Metadata)
	if metadata == nil {
		metadata = map[string]any{}
	}
	if _, ok := metadata["source"]; !ok && blob.Path != "" {
		metadata["source"] = blob.Path
	}
	return NewSliceIterator([]documents.Document{
		documents.New(string(blob.Data), metadata),
	}), nil
}

// GenericLoader composes a BlobLoader with a BlobParser.
type GenericLoader struct {
	BlobLoader BlobLoader
	BlobParser BlobParser
}

// NewGenericLoader creates a generic document loader.
func NewGenericLoader(blobLoader BlobLoader, blobParser BlobParser) (GenericLoader, error) {
	if blobLoader == nil {
		return GenericLoader{}, fmt.Errorf("blob loader is required")
	}
	if blobParser == nil {
		return GenericLoader{}, fmt.Errorf("blob parser is required")
	}
	return GenericLoader{BlobLoader: blobLoader, BlobParser: blobParser}, nil
}

// LazyLoad streams parsed documents from all blobs.
func (l GenericLoader) LazyLoad(ctx context.Context) (DocumentIterator, error) {
	blobIter, err := l.BlobLoader.YieldBlobs(ctx)
	if err != nil {
		return nil, err
	}
	return &genericLoaderIterator{
		ctx:      ctx,
		blobIter: blobIter,
		parser:   l.BlobParser,
	}, nil
}

type genericLoaderIterator struct {
	ctx      context.Context
	blobIter BlobIterator
	parser   BlobParser
	current  DocumentIterator
}

func (i *genericLoaderIterator) Next(ctx context.Context) (documents.Document, bool, error) {
	for {
		if i.current != nil {
			doc, ok, err := i.current.Next(ctx)
			if err != nil || ok {
				return doc, ok, err
			}
			if err := i.current.Close(); err != nil {
				return documents.Document{}, false, err
			}
			i.current = nil
		}
		blob, ok, err := i.blobIter.Next(ctx)
		if err != nil || !ok {
			return documents.Document{}, false, err
		}
		i.current, err = i.parser.LazyParse(i.ctx, blob)
		if err != nil {
			return documents.Document{}, false, err
		}
	}
}

func (i *genericLoaderIterator) Close() error {
	var errs []error
	if i.current != nil {
		errs = append(errs, i.current.Close())
		i.current = nil
	}
	errs = append(errs, i.blobIter.Close())
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// TextLoader loads one text file into one document.
type TextLoader struct {
	Path     string
	Metadata map[string]any
}

// NewTextLoader creates a text-file loader.
func NewTextLoader(path string) TextLoader {
	return TextLoader{Path: path}
}

// LazyLoad streams the loaded text document.
func (l TextLoader) LazyLoad(ctx context.Context) (DocumentIterator, error) {
	blobLoader := NewFileSystemBlobLoader(l.Path)
	blobLoader.Metadata = l.Metadata
	loader, err := NewGenericLoader(blobLoader, TextParser{})
	if err != nil {
		return nil, err
	}
	return loader.LazyLoad(ctx)
}

func shouldSkipHidden(path string, showHidden bool) bool {
	if showHidden {
		return false
	}
	return strings.HasPrefix(filepath.Base(path), ".")
}

func blobMimetype(path string, explicit string) string {
	if explicit != "" {
		return explicit
	}
	return mime.TypeByExtension(filepath.Ext(path))
}

func cloneMetadata(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}
