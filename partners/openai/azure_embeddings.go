package openai

import (
	"context"
	"fmt"

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

// NewAzureEmbeddingsWithADToken is like NewAzureEmbeddings but authenticates
// with an Azure AD token (Authorization: Bearer) instead of an api-key,
// mirroring Python AzureOpenAIEmbeddings(azure_ad_token=...)
// (embeddings/azure.py:135).
func NewAzureEmbeddingsWithADToken(endpoint, deployment, apiVersion, adToken string, opts ...modelconfig.Option) AzureEmbeddings {
	az := azureClient{endpoint: endpoint, deployment: deployment, apiVersion: apiVersion, adToken: adToken}.fromEnv()
	return AzureEmbeddings{embed: NewEmbeddings(opts...), az: az}
}

// EmbedDocuments embeds all documents through the same context-length-safe
// pipeline as Embeddings.EmbedDocuments (Python's AzureOpenAIEmbeddings
// inherits embed_documents from OpenAIEmbeddings, so over-budget texts are
// tokenized, chunked, and merged here too); only the Azure endpoint and
// auth headers differ.
func (e AzureEmbeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	post := func(ctx context.Context, payload any) (embeddingResponsePayload, error) {
		switch typed := payload.(type) {
		case embeddingRequestPayload:
			if dimensions, ok := e.embed.config.Extra[embeddingDimensionsKey].(int); ok && dimensions > 0 {
				typed.Dimensions = &dimensions
				payload = typed
			}
			if format, ok := e.embed.config.Extra[embeddingEncodingFormatKey].(string); ok && format != "" {
				typed.EncodingFormat = format
				payload = typed
			}
		case embeddingTokenRequestPayload:
			if dimensions, ok := e.embed.config.Extra[embeddingDimensionsKey].(int); ok && dimensions > 0 {
				typed.Dimensions = &dimensions
				payload = typed
			}
			if format, ok := e.embed.config.Extra[embeddingEncodingFormatKey].(string); ok && format != "" {
				typed.EncodingFormat = format
				payload = typed
			}
		}
		return azurePost[embeddingResponsePayload](e.az, ctx, e.embed.config, "/embeddings", payload)
	}
	return e.embed.lenSafeEmbedDocuments(ctx, texts, post)
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
