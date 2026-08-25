package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// azureClient carries the Azure-specific endpoint/auth needed to build Azure
// OpenAI URLs (`{endpoint}/openai/deployments/{deployment}/…?api-version=…`)
// and the `api-key` header (or `Authorization: Bearer <ad-token>` when an AD
// token is provided), mirroring Python's AzureChatOpenAI client construction.
type azureClient struct {
	endpoint   string
	deployment string
	apiVersion string
	apiKey     string
	adToken    string
}

// fromEnv fills empty endpoint/apiKey/apiVersion from the AZURE_OPENAI_*
// environment variables, mirroring Python's `from_env` defaults.
func (a azureClient) fromEnv() azureClient {
	if a.endpoint == "" {
		a.endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	if a.apiKey == "" {
		a.apiKey = os.Getenv("AZURE_OPENAI_API_KEY")
	}
	if a.adToken == "" {
		a.adToken = os.Getenv("AZURE_OPENAI_AD_TOKEN")
	}
	if a.apiVersion == "" {
		a.apiVersion = os.Getenv("AZURE_OPENAI_API_VERSION")
	}
	return a
}

func (a azureClient) url(endpointPath string) string {
	return "/openai/deployments/" + a.deployment + endpointPath + "?api-version=" + a.apiVersion
}

func (a azureClient) configure(req *http.Request, cfg modelconfig.Config) {
	req.Header.Set("Content-Type", "application/json")
	if a.adToken != "" {
		req.Header.Set("Authorization", "Bearer "+a.adToken)
	} else if a.apiKey != "" {
		req.Header.Set("api-key", a.apiKey)
	}
	for name, value := range cfg.Headers {
		req.Header.Set(name, value)
	}
}

func azurePost[T any](a azureClient, ctx context.Context, cfg modelconfig.Config, endpointPath string, payload any) (T, error) {
	cfg.BaseURL = a.endpoint
	return httpclient.PostJSON[T](ctx, providerName, cfg, a.url(endpointPath), payload, func(req *http.Request) {
		a.configure(req, cfg)
	})
}

// AzureChatModel adapts LangChain chat calls to the Azure OpenAI Chat
// Completions API, mirroring Python's AzureChatOpenAI. The request/response
// bodies are identical to the standard `/chat/completions`; only the endpoint
// URL and the `api-key` header differ.
type AzureChatModel struct {
	chat ChatModel
	az   azureClient
}

var _ language.ChatModel = AzureChatModel{}

// NewAzureChatModel builds an Azure chat model.
func NewAzureChatModel(endpoint, deployment, apiVersion, apiKey string, opts ...modelconfig.Option) AzureChatModel {
	az := azureClient{endpoint: endpoint, deployment: deployment, apiVersion: apiVersion, apiKey: apiKey}.fromEnv()
	return AzureChatModel{chat: NewChatModel(opts...), az: az}
}

// NewAzureChatModelWithADToken is like NewAzureChatModel but authenticates
// with an Azure AD token (Authorization: Bearer) instead of an api-key.
func NewAzureChatModelWithADToken(endpoint, deployment, apiVersion, adToken string, opts ...modelconfig.Option) AzureChatModel {
	az := azureClient{endpoint: endpoint, deployment: deployment, apiVersion: apiVersion, adToken: adToken}.fromEnv()
	return AzureChatModel{chat: NewChatModel(opts...), az: az}
}

// WithStreamChunkTimeout returns a copy of the model with the per-chunk
// stream timeout set, passing through to the embedded ChatModel that
// AzureChatModel.Stream delegates to (mirrors ChatModel.WithStreamChunkTimeout).
func (m AzureChatModel) WithStreamChunkTimeout(d time.Duration) AzureChatModel {
	next := m
	next.chat = m.chat.WithStreamChunkTimeout(d)
	return next
}

func (m AzureChatModel) Invoke(ctx context.Context, input []messages.Message, opts ...runnables.Option) (messages.Message, error) {
	resp, err := azurePost[chatCompletionsResponse](m.az, ctx, m.chat.config, "/chat/completions", m.chat.buildChatCompletionsRequest(input))
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
	cfg := runnables.NewConfig(opts...)
	return m.createStream(ctx, input, cfg)
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

// createStream mirrors ChatModel.createChatCompletionsStream but targets the
// Azure endpoint with the Azure auth headers.
func (m AzureChatModel) createStream(ctx context.Context, input []messages.Message, cfg runnables.Config) (*chatCompletionsStream, error) {
	requestPayload := m.chat.buildChatCompletionsRequest(input)
	requestPayload.Stream = true
	body, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(ctx, m.chat.config.Timeout)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(m.az.endpoint, "/")+m.az.url("/chat/completions"),
		bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, err
	}
	m.az.configure(req, m.chat.config)

	client := m.chat.config.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		cancel()
		return nil, lcerrors.WrapTransport(err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		cancel()
		return nil, httpclient.ResponseError(providerName, m.az.url("/chat/completions"), resp)
	}

	return &chatCompletionsStream{
		ctx:          ctx,
		cancel:       cancel,
		body:         resp.Body,
		scanner:      bufio.NewScanner(resp.Body),
		cfg:          cfg,
		toolCalls:    make(map[int]*streamToolCall),
		chunkTimeout: m.chat.streamChunkTimeout,
		model:        m.chat.config.Model,
	}, nil
}
