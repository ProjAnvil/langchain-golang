package documents

import (
	"context"
	"fmt"
)

// Transformer transforms a sequence of documents.
type Transformer interface {
	TransformDocuments(ctx context.Context, docs []Document) ([]Document, error)
}

// TransformerFunc adapts a function to Transformer.
type TransformerFunc func(context.Context, []Document) ([]Document, error)

// TransformDocuments transforms documents.
func (f TransformerFunc) TransformDocuments(ctx context.Context, docs []Document) ([]Document, error) {
	return f(ctx, docs)
}

// Compressor compresses retrieved documents with query context.
type Compressor interface {
	CompressDocuments(ctx context.Context, docs []Document, query string) ([]Document, error)
}

// CompressorFunc adapts a function to Compressor.
type CompressorFunc func(context.Context, []Document, string) ([]Document, error)

// CompressDocuments compresses documents.
func (f CompressorFunc) CompressDocuments(ctx context.Context, docs []Document, query string) ([]Document, error) {
	return f(ctx, docs, query)
}

// CompressorPipeline chains transformers and compressors.
type CompressorPipeline struct {
	Steps []any
}

// NewCompressorPipeline creates a compressor pipeline.
func NewCompressorPipeline(steps ...any) (CompressorPipeline, error) {
	for _, step := range steps {
		switch step.(type) {
		case Transformer, Compressor:
			continue
		default:
			return CompressorPipeline{}, fmt.Errorf("unexpected document pipeline step %T", step)
		}
	}
	return CompressorPipeline{Steps: append([]any(nil), steps...)}, nil
}

// CompressDocuments applies each transformer/compressor in order.
func (p CompressorPipeline) CompressDocuments(ctx context.Context, docs []Document, query string) ([]Document, error) {
	current := cloneDocuments(docs)
	for _, step := range p.Steps {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		switch typed := step.(type) {
		case Compressor:
			next, err := typed.CompressDocuments(ctx, current, query)
			if err != nil {
				return nil, err
			}
			current = cloneDocuments(next)
		case Transformer:
			next, err := typed.TransformDocuments(ctx, current)
			if err != nil {
				return nil, err
			}
			current = cloneDocuments(next)
		default:
			return nil, fmt.Errorf("unexpected document pipeline step %T", step)
		}
	}
	return current, nil
}

// TransformDocuments applies the pipeline without query context.
func (p CompressorPipeline) TransformDocuments(ctx context.Context, docs []Document) ([]Document, error) {
	return p.CompressDocuments(ctx, docs, "")
}

func cloneDocuments(docs []Document) []Document {
	out := make([]Document, len(docs))
	for i, doc := range docs {
		out[i] = doc.Clone()
	}
	return out
}
