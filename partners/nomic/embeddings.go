// Package nomic provides text embeddings through the Nomic Atlas API
// (https://api-atlas.nomic.ai).
package nomic

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	coreembeddings "github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

const (
	providerName       = "nomic"
	defaultBaseURL     = "https://api-atlas.nomic.ai/v1"
	defaultModel       = "nomic-embed-text-v1.5"
	embeddingsEndpoint = "/embedding/text"
)

// Task types accepted by the Atlas /embedding/text endpoint, mirroring the
// values documented by langchain_nomic.NomicEmbeddings.embed
// (langchain-nomic 1.0.1, embeddings.py:102-103).
const (
	TaskTypeSearchQuery    = "search_query"
	TaskTypeSearchDocument = "search_document"
	TaskTypeClassification = "classification"
	TaskTypeClustering     = "clustering"
)

// Embeddings produces text embeddings through the Nomic Atlas API and
// implements the core embeddings interface.
//
// Python parity: langchain_nomic.NomicEmbeddings (langchain-nomic 1.0.1) sends
// Bearer-authorized POSTs of {"model","texts","task_type","dimensionality"?}
// to /embedding/text and returns output["embeddings"]; embed_documents uses
// task_type "search_document" and embed_query uses "search_query"
// (embeddings.py:97-138), and the API key falls back to NOMIC_API_KEY
// (embeddings.py:88).
//
// Divergences vs langchain-nomic 1.0.1:
//   - an empty Model defaults to nomic-embed-text-v1.5 (Python requires the
//     model argument);
//   - only remote text inference is supported: no local/dynamic inference
//     modes (Embed4All device selection) and no embed_image/vision_model;
//   - an explicitly invalid TaskType fails fast with ErrInvalidInput instead
//     of forwarding the string to the API (Go addition);
//   - a missing API key fails fast with ErrInvalidInput instead of surfacing
//     the provider's 401 (Go addition).
type Embeddings struct {
	// Model is the embedding model name; empty defaults to
	// nomic-embed-text-v1.5.
	Model string
	// TaskType overrides the per-method default task type. Legal values are
	// the TaskType* constants; when empty, EmbedDocuments sends
	// "search_document" and EmbedQuery sends "search_query".
	TaskType string
	// Dimensionality truncates the vectors for Matryoshka-capable models;
	// nil sends full-size vectors.
	Dimensionality *int
	// APIKey authorizes requests; empty falls back to the NOMIC_API_KEY
	// environment variable.
	APIKey string
	// BaseURL is the Atlas API root; empty defaults to
	// https://api-atlas.nomic.ai/v1.
	BaseURL string
	// HTTPClient issues the requests; nil uses http.DefaultClient.
	HTTPClient *http.Client
}

// Compile-time guard: Embeddings satisfies the core embeddings interface.
var _ coreembeddings.Embeddings = (*Embeddings)(nil)

// EmbedDocuments embeds texts as searchable documents (task_type
// "search_document" unless TaskType overrides it) and returns the vectors in
// input order.
func (e *Embeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	return e.embed(ctx, texts, TaskTypeSearchDocument)
}

// EmbedQuery embeds one search query (task_type "search_query" unless
// TaskType overrides it) and returns the single resulting vector.
func (e *Embeddings) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	vectors, err := e.embed(ctx, []string{text}, TaskTypeSearchQuery)
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf(
			"%w: nomic %s response carried no embeddings",
			lcerrors.ErrSchemaValidation, embeddingsEndpoint,
		)
	}
	return vectors[0], nil
}

// embed posts one /embedding/text request; defaultTaskType applies when
// TaskType is empty.
func (e *Embeddings) embed(
	ctx context.Context,
	texts []string,
	defaultTaskType string,
) ([][]float64, error) {
	taskType := e.TaskType
	if taskType == "" {
		taskType = defaultTaskType
	}
	if !isValidTaskType(taskType) {
		return nil, fmt.Errorf(
			"%w: invalid nomic task type %q: want one of %s, %s, %s, %s",
			lcerrors.ErrInvalidInput, e.TaskType,
			TaskTypeSearchQuery, TaskTypeSearchDocument,
			TaskTypeClassification, TaskTypeClustering,
		)
	}
	apiKey := e.resolvedAPIKey()
	if apiKey == "" {
		return nil, fmt.Errorf(
			"%w: nomic API key is required: set Embeddings.APIKey or the NOMIC_API_KEY environment variable",
			lcerrors.ErrInvalidInput,
		)
	}

	response, err := httpclient.PostJSON[embeddingResponse](
		ctx,
		providerName,
		modelconfig.Config{
			BaseURL:    e.resolvedBaseURL(),
			APIKey:     apiKey,
			HTTPClient: e.HTTPClient,
		},
		embeddingsEndpoint,
		embeddingRequest{
			Model:          e.resolvedModel(),
			Texts:          texts,
			TaskType:       taskType,
			Dimensionality: e.Dimensionality,
		},
		func(req *http.Request) {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		},
	)
	if err != nil {
		return nil, err
	}
	return response.Embeddings, nil
}

// isValidTaskType reports whether taskType is one of the four values the
// Atlas API documents.
func isValidTaskType(taskType string) bool {
	switch taskType {
	case TaskTypeSearchQuery, TaskTypeSearchDocument,
		TaskTypeClassification, TaskTypeClustering:
		return true
	default:
		return false
	}
}

// resolvedModel returns the configured model or the nomic-embed-text-v1.5
// default (divergence: Python requires the model argument).
func (e *Embeddings) resolvedModel() string {
	if strings.TrimSpace(e.Model) == "" {
		return defaultModel
	}
	return e.Model
}

// resolvedBaseURL returns the configured base URL or the Atlas API default.
func (e *Embeddings) resolvedBaseURL() string {
	if strings.TrimSpace(e.BaseURL) == "" {
		return defaultBaseURL
	}
	return e.BaseURL
}

// resolvedAPIKey returns the configured API key or the NOMIC_API_KEY
// environment variable value.
func (e *Embeddings) resolvedAPIKey() string {
	if e.APIKey != "" {
		return e.APIKey
	}
	return os.Getenv("NOMIC_API_KEY")
}

// embeddingRequest is the /embedding/text request body (Atlas wire contract,
// langchain-nomic 1.0.1).
type embeddingRequest struct {
	Model          string   `json:"model"`
	Texts          []string `json:"texts"`
	TaskType       string   `json:"task_type"`
	Dimensionality *int     `json:"dimensionality,omitzero"`
}

// embeddingResponse is the /embedding/text response body. Usage is parsed to
// mirror the wire contract but is not exposed by the embeddings interface.
type embeddingResponse struct {
	Embeddings [][]float64    `json:"embeddings"`
	Usage      embeddingUsage `json:"usage"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
