// Package cohere provides a Cohere rerank document compressor backed by the
// hosted Cohere Rerank API.
package cohere

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

const (
	providerName   = "cohere"
	defaultBaseURL = "https://api.cohere.com"
	defaultModel   = "rerank-v3.5"
	defaultTopN    = 3
)

// Reranker re-ranks documents through the hosted Cohere Rerank API. It
// implements retrievers.DocumentCompressor, so it can drive a
// retrievers.ContextualCompressionRetriever directly.
//
// Python parity: langchain_cohere.CohereRerank (POST /v2/rerank with
// model/query/documents/top_n; the response's results[].index reorder the
// input documents and relevance_score lands in the output metadata).
type Reranker struct {
	cfg  modelconfig.Config
	topN int
}

// Option configures a Reranker.
type Option func(*Reranker)

// WithAPIKey sets the Cohere API key used as Bearer authorization. When unset,
// New falls back to the COHERE_API_KEY environment variable.
func WithAPIKey(apiKey string) Option {
	return func(r *Reranker) {
		r.cfg.APIKey = apiKey
	}
}

// WithBaseURL overrides the Cohere API base URL (default
// https://api.cohere.com).
func WithBaseURL(baseURL string) Option {
	return func(r *Reranker) {
		r.cfg.BaseURL = baseURL
	}
}

// WithModel sets the rerank model (default "rerank-v3.5").
func WithModel(model string) Option {
	return func(r *Reranker) {
		r.cfg.Model = model
	}
}

// WithTopN sets how many reranked results the API should return (default 3,
// matching langchain_cohere.CohereRerank.top_n). Zero or negative values omit
// top_n from the request so the API returns all documents.
func WithTopN(topN int) Option {
	return func(r *Reranker) {
		r.topN = topN
	}
}

// WithMaxRetries sets the number of retries for retryable API failures.
func WithMaxRetries(maxRetries int) Option {
	return func(r *Reranker) {
		r.cfg.MaxRetries = maxRetries
	}
}

// WithHTTPClient sets the HTTP client used for rerank requests.
func WithHTTPClient(client *http.Client) Option {
	return func(r *Reranker) {
		r.cfg.HTTPClient = client
	}
}

// WithHeader adds a header to every rerank request.
func WithHeader(name, value string) Option {
	return func(r *Reranker) {
		if r.cfg.Headers == nil {
			r.cfg.Headers = map[string]string{}
		}
		r.cfg.Headers[name] = value
	}
}

// New creates a Cohere rerank compressor. The API key resolves from
// WithAPIKey first and the COHERE_API_KEY environment variable second.
func New(opts ...Option) *Reranker {
	reranker := &Reranker{
		cfg: modelconfig.New(
			modelconfig.WithBaseURL(defaultBaseURL),
		),
		topN: defaultTopN,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(reranker)
		}
	}
	if strings.TrimSpace(reranker.cfg.Model) == "" {
		reranker.cfg.Model = defaultModel
	}
	if strings.TrimSpace(reranker.cfg.BaseURL) == "" {
		reranker.cfg.BaseURL = defaultBaseURL
	}
	if reranker.cfg.APIKey == "" {
		if key := os.Getenv("COHERE_API_KEY"); key != "" {
			reranker.cfg.APIKey = key
		}
	}
	return reranker
}

// Compile-time guard: Reranker satisfies the document compressor interface
// used by retrievers.ContextualCompressionRetriever.
var _ retrievers.DocumentCompressor = (*Reranker)(nil)

// CompressDocuments re-ranks docs against query with the Cohere Rerank API.
// Empty input returns immediately without an API call; otherwise the input
// documents are reassembled in results[].index order with relevance_score
// recorded in each output document's metadata.
func (r *Reranker) CompressDocuments(
	ctx context.Context,
	docs []documents.Document,
	query string,
) ([]documents.Document, error) {
	if len(docs) == 0 {
		return []documents.Document{}, nil
	}

	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
	}
	cfg := r.cfg
	response, err := httpclient.PostJSON[rerankResponse](
		ctx,
		providerName,
		cfg,
		"/v2/rerank",
		rerankRequest{
			Model:     cfg.Model,
			Query:     query,
			Documents: texts,
			TopN:      r.topN,
		},
		func(req *http.Request) { configureRequest(req, cfg) },
	)
	if err != nil {
		return nil, err
	}

	out := make([]documents.Document, 0, len(response.Results))
	for _, result := range response.Results {
		if result.Index < 0 || result.Index >= len(docs) {
			return nil, fmt.Errorf(
				"cohere rerank result index %d outside input documents (0..%d)",
				result.Index,
				len(docs)-1,
			)
		}
		doc := docs[result.Index].Clone()
		if doc.Metadata == nil {
			doc.Metadata = map[string]any{}
		}
		doc.Metadata["relevance_score"] = result.RelevanceScore
		out = append(out, doc)
	}
	return out, nil
}

func configureRequest(req *http.Request, cfg modelconfig.Config) {
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	for name, value := range cfg.Headers {
		req.Header.Set(name, value)
	}
}

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

type rerankResponse struct {
	ID      string         `json:"id"`
	Results []rerankResult `json:"results"`
}

type rerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}
