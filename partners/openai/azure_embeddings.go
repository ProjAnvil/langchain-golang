package openai

import (
	"context"
	"fmt"
	"sort"

	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// AzureEmbeddings adapts LangChain embedding calls to the Azure OpenAI
// embeddings endpoint, mirroring Python's AzureOpenAIEmbeddings. The request/
// response bodies match the standard `/embeddings`; only the endpoint URL and
// `api-key` header differ.
type AzureEmbeddings struct {
	embed Embeddings
	az    azureClient
}

var _ embeddings.Embeddings = AzureEmbeddings{}

// NewAzureEmbeddings builds an Azure embeddings adapter.
func NewAzureEmbeddings(endpoint, deployment, apiVersion, apiKey string, opts ...modelconfig.Option) AzureEmbeddings {
	az := azureClient{endpoint: endpoint, deployment: deployment, apiVersion: apiVersion, apiKey: apiKey}.fromEnv()
	return AzureEmbeddings{embed: NewEmbeddings(opts...), az: az}
}

// EmbedDocuments embeds all documents with one Azure API request.
func (e AzureEmbeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	payload := embeddingRequestPayload{Model: e.embed.config.Model, Input: texts}
	if dimensions, ok := e.embed.config.Extra[embeddingDimensionsKey].(int); ok && dimensions > 0 {
		payload.Dimensions = &dimensions
	}
	if format, ok := e.embed.config.Extra[embeddingEncodingFormatKey].(string); ok && format != "" {
		payload.EncodingFormat = format
	}

	response, err := azurePost[embeddingResponsePayload](e.az, ctx, e.embed.config, "/embeddings", payload)
	if err != nil {
		return nil, err
	}
	sort.SliceStable(response.Data, func(i int, j int) bool {
		return response.Data[i].Index < response.Data[j].Index
	})
	vectors := make([][]float64, len(response.Data))
	for i, item := range response.Data {
		if item.Index < 0 || item.Index >= len(texts) {
			return nil, fmt.Errorf("embedding index out of range: %d", item.Index)
		}
		vectors[i] = append([]float64(nil), item.Embedding...)
	}
	if len(vectors) != len(texts) {
		return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(vectors), len(texts))
	}
	return vectors, nil
}

// EmbedQuery embeds a single query.
func (e AzureEmbeddings) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	vectors, err := e.EmbedDocuments(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}
	return vectors[0], nil
}
