package documents

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestCompressorPipelineTransformsAndCompresses(t *testing.T) {
	pipeline, err := NewCompressorPipeline(
		TransformerFunc(func(_ context.Context, docs []Document) ([]Document, error) {
			out := make([]Document, 0, len(docs))
			for _, doc := range docs {
				doc.PageContent = strings.ToUpper(doc.PageContent)
				doc.Metadata["stage"] = "transformed"
				out = append(out, doc)
			}
			return out, nil
		}),
		CompressorFunc(func(_ context.Context, docs []Document, query string) ([]Document, error) {
			if query != "keep" {
				t.Fatalf("query: got %q", query)
			}
			return docs[:1], nil
		}),
	)
	if err != nil {
		t.Fatalf("new pipeline: %v", err)
	}

	input := []Document{
		New("alpha", map[string]any{"source": "unit"}),
		New("beta", map[string]any{"source": "unit"}),
	}
	got, err := pipeline.CompressDocuments(context.Background(), input, "keep")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(got) != 1 || got[0].PageContent != "ALPHA" || got[0].Metadata["stage"] != "transformed" {
		t.Fatalf("got %#v", got)
	}
	if _, ok := input[0].Metadata["stage"]; ok {
		t.Fatalf("pipeline mutated input metadata: %#v", input[0].Metadata)
	}
}

func TestCompressorPipelineTransformDocuments(t *testing.T) {
	pipeline, err := NewCompressorPipeline(TransformerFunc(func(_ context.Context, docs []Document) ([]Document, error) {
		return append(docs, New("extra", nil)), nil
	}))
	if err != nil {
		t.Fatalf("new pipeline: %v", err)
	}
	got, err := pipeline.TransformDocuments(context.Background(), []Document{New("base", nil)})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if len(got) != 2 || got[1].PageContent != "extra" {
		t.Fatalf("got %#v", got)
	}
}

func TestCompressorPipelineRejectsInvalidStep(t *testing.T) {
	_, err := NewCompressorPipeline("bad")
	if err == nil {
		t.Fatal("expected invalid step error")
	}
}

func TestCompressorPipelinePropagatesErrors(t *testing.T) {
	wantErr := errors.New("failed")
	pipeline, err := NewCompressorPipeline(CompressorFunc(func(context.Context, []Document, string) ([]Document, error) {
		return nil, wantErr
	}))
	if err != nil {
		t.Fatalf("new pipeline: %v", err)
	}
	_, err = pipeline.CompressDocuments(context.Background(), []Document{New("x", nil)}, "q")
	if !errors.Is(err, wantErr) {
		t.Fatalf("err got %v want %v", err, wantErr)
	}
}
