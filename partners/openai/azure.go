package openai

import (
	"context"
	"net/http"
	"os"
	"strings"

	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// AzureChatModel adapts LangChain chat calls to the Azure OpenAI Chat
// Completions API, mirroring Python's AzureChatOpenAI. The request/response
// bodies are identical to the standard `/chat/completions`; only the endpoint
// URL (`{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=…`)
// and the `api-key` header (instead of `Bearer`) differ.
type AzureChatModel struct {
	chat       ChatModel
	endpoint   string
	deployment string
	apiVersion string
	apiKey     string
}

var _ language.ChatModel = AzureChatModel{}

// NewAzureChatModel builds an Azure chat model. Empty endpoint/apiKey/
// apiVersion fall back to the AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY /
// AZURE_OPENAI_API_VERSION environment variables (mirroring Python's
// `from_env` defaults).
func NewAzureChatModel(endpoint, deployment, apiVersion, apiKey string, opts ...modelconfig.Option) AzureChatModel {
	if endpoint == "" {
		endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	if apiKey == "" {
		apiKey = os.Getenv("AZURE_OPENAI_API_KEY")
	}
	if apiVersion == "" {
		apiVersion = os.Getenv("AZURE_OPENAI_API_VERSION")
	}
	return AzureChatModel{
		chat:       NewChatModel(opts...),
		endpoint:   endpoint,
		deployment: deployment,
		apiVersion: apiVersion,
		apiKey:     apiKey,
	}
}

func (m AzureChatModel) azureURL() string {
	return strings.TrimRight(m.endpoint, "/") +
		"/openai/deployments/" + m.deployment +
		"/chat/completions?api-version=" + m.apiVersion
}

func (m AzureChatModel) Invoke(ctx context.Context, input []messages.Message, opts ...runnables.Option) (messages.Message, error) {
	cfg := m.chat.config
	cfg.BaseURL = m.endpoint
	resp, err := httpclient.PostJSON[chatCompletionsResponse](
		ctx, providerName, cfg,
		"/openai/deployments/"+m.deployment+"/chat/completions?api-version="+m.apiVersion,
		m.chat.buildChatCompletionsRequest(input),
		func(req *http.Request) {
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("api-key", m.apiKey)
			for name, value := range m.chat.config.Headers {
				req.Header.Set(name, value)
			}
		},
	)
	if err != nil {
		return messages.Message{}, err
	}
	return resp.toResponsesPayload().toMessage(), nil
}

func (m AzureChatModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	runnable := runnables.NewFunc(m.Invoke, m.InputSchema(), m.OutputSchema())
	return runnable.Batch(ctx, inputs, opts...)
}

func (m AzureChatModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	// Streaming is not yet implemented for Azure; return an empty stream.
	return runnables.NewSliceStream([]messages.Message{}), nil
}

func (m AzureChatModel) InputSchema() schema.Schema  { return m.chat.InputSchema() }
func (m AzureChatModel) OutputSchema() schema.Schema { return m.chat.OutputSchema() }

func (m AzureChatModel) BindTools(bound []tools.Tool) (language.ChatModel, error) {
	next := m
	boundChat, err := m.chat.BindTools(bound)
	if err != nil {
		return nil, err
	}
	next.chat = boundChat.(ChatModel)
	return next, nil
}

func (m AzureChatModel) Capabilities() language.ChatModelCapabilities {
	return m.chat.Capabilities()
}

func (m AzureChatModel) LLMType() string { return "azure-openai-chat" }
