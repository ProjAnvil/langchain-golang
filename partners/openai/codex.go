package openai

import (
	"context"
	"net/http"

	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// CodexChatModel adapts LangChain chat calls to the experimental ChatGPT Codex
// backend (`https://chatgpt.com/backend-api/codex`), mirroring Python's
// `_ChatOpenAICodex`. It injects a refresh-aware `Authorization` bearer token
// and a `ChatGPT-Account-Id` header on every call. The browser login flow is
// out of scope (see chatgpt_oauth.go).
type CodexChatModel struct {
	chat      ChatModel
	baseURL   string
	accountID string
	provider  *TokenProvider
}

var _ language.ChatModel = CodexChatModel{}

const defaultCodexBaseURL = "https://chatgpt.com/backend-api/codex"

// NewCodexChatModel builds a Codex chat model backed by a token provider.
func NewCodexChatModel(accountID string, provider *TokenProvider, opts ...modelconfig.Option) CodexChatModel {
	return CodexChatModel{
		chat:      NewChatModel(opts...),
		baseURL:   defaultCodexBaseURL,
		accountID: accountID,
		provider:  provider,
	}
}

func (m CodexChatModel) Invoke(ctx context.Context, input []messages.Message, opts ...runnables.Option) (messages.Message, error) {
	accessToken := ""
	if m.provider != nil {
		token, err := m.provider.AccessToken(ctx)
		if err != nil {
			return messages.Message{}, err
		}
		accessToken = token
	}

	cfg := m.chat.config
	cfg.BaseURL = m.baseURL
	resp, err := httpclient.PostJSON[chatCompletionsResponse](
		ctx, providerName, cfg, "",
		m.chat.buildChatCompletionsRequest(input),
		func(req *http.Request) {
			req.Header.Set("Content-Type", "application/json")
			if accessToken != "" {
				req.Header.Set("Authorization", "Bearer "+accessToken)
			}
			if m.accountID != "" {
				req.Header.Set("ChatGPT-Account-Id", m.accountID)
			}
		},
	)
	if err != nil {
		return messages.Message{}, err
	}
	return resp.toResponsesPayload().toMessage(), nil
}

func (m CodexChatModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	runnable := runnables.NewFunc(m.Invoke, m.InputSchema(), m.OutputSchema())
	return runnable.Batch(ctx, inputs, opts...)
}

func (m CodexChatModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return runnables.NewSliceStream([]messages.Message{}), nil
}

func (m CodexChatModel) InputSchema() schema.Schema  { return m.chat.InputSchema() }
func (m CodexChatModel) OutputSchema() schema.Schema { return m.chat.OutputSchema() }

func (m CodexChatModel) BindTools(bound []tools.Tool) (language.ChatModel, error) {
	next := m
	boundChat, err := m.chat.BindTools(bound)
	if err != nil {
		return nil, err
	}
	next.chat = boundChat.(ChatModel)
	return next, nil
}

func (m CodexChatModel) Capabilities() language.ChatModelCapabilities {
	return m.chat.Capabilities()
}

func (m CodexChatModel) LLMType() string { return "openai-codex" }
