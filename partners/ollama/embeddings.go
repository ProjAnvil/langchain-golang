package ollama

import (
	"context"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/core/modelconfig"
)

const embeddingDimensionsKey = "ollama_embedding_dimensions"

// WithEmbeddingDimensions sets the optional Ollama embeddings dimensions
// request parameter.
func WithEmbeddingDimensions(dimensions int) modelconfig.Option {
	return modelconfig.WithExtra(embeddingDimensionsKey, dimensions)
}

// Embeddings adapts LangChain embedding calls to the Ollama /api/embed endpoint.
type Embeddings struct {
	config modelconfig.Config
}

// NewEmbeddings creates an Ollama embeddings adapter.
func NewEmbeddings(opts ...modelconfig.Option) Embeddings {
	cfg := modelconfig.New(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "llama3"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return Embeddings{config: cfg}
}

// EmbedDocuments embeds all documents with one API request. Ollama returns
// embeddings in input order.
func (e Embeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	response, err := e.createEmbeddings(ctx, texts)
	if err != nil {
		return nil, err
	}
	if len(response.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama embedding count mismatch: got %d want %d", len(response.Embeddings), len(texts))
	}
	vectors := make([][]float64, len(response.Embeddings))
	for i, embedding := range response.Embeddings {
		vectors[i] = append([]float64(nil), embedding...)
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
		return nil, fmt.Errorf("empty ollama embedding response")
	}
	return vectors[0], nil
}

func (e Embeddings) createEmbeddings(
	ctx context.Context,
	texts []string,
) (embedResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, e.config.Timeout)
	defer cancel()
	payload := embedRequest{
		Model:     e.config.Model,
		Input:     texts,
		KeepAlive: e.config.Extra[keepAliveKey],
		Options:   e.buildOptions(),
	}
	if dimensions, ok := e.config.Extra[embeddingDimensionsKey].(int); ok && dimensions > 0 {
		payload.Dimensions = &dimensions
	}
	return postJSON[embedResponse](ctx, e.config, "/api/embed", payload)
}

func (e Embeddings) buildOptions() map[string]any {
	options := readSamplingOptions(e.config.Extra)
	if e.config.Temperature != nil {
		if options == nil {
			options = map[string]any{}
		}
		options["temperature"] = *e.config.Temperature
	}
	return options
}

type embedRequest struct {
	Model     string         `json:"model"`
	Input     []string       `json:"input"`
	Dimensions *int           `json:"dimensions,omitempty"`
	KeepAlive any            `json:"keep_alive,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
}

type embedResponse struct {
	Model        string      `json:"model"`
	Embeddings   [][]float64 `json:"embeddings"`
	TotalDuration int64      `json:"total_duration"`
}
