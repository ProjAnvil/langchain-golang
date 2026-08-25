package openai

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// Embeddings adapts LangChain embedding calls to OpenAI's embeddings endpoint.
type Embeddings struct {
	config modelconfig.Config
}

const (
	embeddingDimensionsKey     = "openai_embedding_dimensions"
	embeddingEncodingFormatKey = "openai_embedding_encoding_format"
)

// WithEmbeddingDimensions sets the optional OpenAI embeddings dimensions
// request parameter.
func WithEmbeddingDimensions(dimensions int) modelconfig.Option {
	return modelconfig.WithExtra(embeddingDimensionsKey, dimensions)
}

// WithEmbeddingEncodingFormat sets the optional OpenAI embeddings
// encoding_format request parameter.
func WithEmbeddingEncodingFormat(format string) modelconfig.Option {
	return modelconfig.WithExtra(embeddingEncodingFormatKey, format)
}

const embeddingChunkSizeKey = "openai_embedding_chunk_size"

// WithEmbeddingChunkSize sets the maximum number of texts embedded per API
// request, mirroring Python OpenAIEmbeddings(chunk_size=...) (default 1000,
// embeddings/base.py:260). Non-positive values fall back to the default.
func WithEmbeddingChunkSize(size int) modelconfig.Option {
	return modelconfig.WithExtra(embeddingChunkSizeKey, size)
}

// chunkSize returns the configured batch size (default 1000).
func (e Embeddings) chunkSize() int {
	if size, ok := e.config.Extra[embeddingChunkSizeKey].(int); ok && size > 0 {
		return size
	}
	return 1000
}

// NewEmbeddings creates an OpenAI embeddings adapter.
func NewEmbeddings(opts ...modelconfig.Option) Embeddings {
	cfg := modelconfig.New(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "text-embedding-3-small"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return Embeddings{config: cfg}
}

// EmbedDocuments embeds all documents, batching inputs into chunks of
// chunkSize texts per API request (Python embeddings/base.py:731-745).
func (e Embeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	chunkSize := e.chunkSize()
	vectors := make([][]float64, 0, len(texts))
	for start := 0; start < len(texts); start += chunkSize {
		end := start + chunkSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[start:end]
		response, err := e.createEmbeddings(ctx, batch)
		if err != nil {
			return nil, err
		}

		sort.SliceStable(response.Data, func(i int, j int) bool {
			return response.Data[i].Index < response.Data[j].Index
		})
		if len(response.Data) != len(batch) {
			return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(response.Data), len(batch))
		}
		for _, item := range response.Data {
			if item.Index < 0 || item.Index >= len(batch) {
				return nil, fmt.Errorf("embedding index out of range: %d", item.Index)
			}
			vectors = append(vectors, append([]float64(nil), item.Embedding...))
		}
	}
	return vectors, nil
}

// EmbedQuery embeds a single query.
func (e Embeddings) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	vectors, err := e.EmbedDocuments(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}
	return vectors[0], nil
}

func (e Embeddings) createEmbeddings(
	ctx context.Context,
	texts []string,
) (embeddingResponsePayload, error) {
	ctx, cancel := context.WithTimeout(ctx, e.config.Timeout)
	defer cancel()
	payload := embeddingRequestPayload{
		Model: e.config.Model,
		Input: texts,
	}
	if dimensions, ok := e.config.Extra[embeddingDimensionsKey].(int); ok && dimensions > 0 {
		payload.Dimensions = &dimensions
	}
	if format, ok := e.config.Extra[embeddingEncodingFormatKey].(string); ok && format != "" {
		payload.EncodingFormat = format
	}
	return postJSON[embeddingResponsePayload](ctx, e.config, "/embeddings", payload)
}

type embeddingRequestPayload struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	Dimensions     *int     `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

type embeddingResponsePayload struct {
	Object string                 `json:"object"`
	Model  string                 `json:"model"`
	Data   []embeddingDataPayload `json:"data"`
	Usage  embeddingUsagePayload  `json:"usage"`
}

type embeddingDataPayload struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type embeddingUsagePayload struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
