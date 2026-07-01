package chatmodels

import (
	"fmt"
	"sort"
	"strings"
)

type ProviderInfo struct {
	Package string
	Class   string
}

var BuiltinProviders = map[string]ProviderInfo{
	"anthropic":               {Package: "langchain_anthropic", Class: "ChatAnthropic"},
	"anthropic_bedrock":       {Package: "langchain_aws", Class: "ChatAnthropicBedrock"},
	"azure_ai":                {Package: "langchain_azure_ai.chat_models", Class: "AzureAIOpenAIApiChatModel"},
	"azure_openai":            {Package: "langchain_openai", Class: "AzureChatOpenAI"},
	"baseten":                 {Package: "langchain_baseten", Class: "ChatBaseten"},
	"bedrock":                 {Package: "langchain_aws", Class: "ChatBedrock"},
	"bedrock_converse":        {Package: "langchain_aws", Class: "ChatBedrockConverse"},
	"cohere":                  {Package: "langchain_cohere", Class: "ChatCohere"},
	"deepseek":                {Package: "langchain_deepseek", Class: "ChatDeepSeek"},
	"fireworks":               {Package: "langchain_fireworks", Class: "ChatFireworks"},
	"google_anthropic_vertex": {Package: "langchain_google_vertexai.model_garden", Class: "ChatAnthropicVertex"},
	"google_genai":            {Package: "langchain_google_genai", Class: "ChatGoogleGenerativeAI"},
	"google_vertexai":         {Package: "langchain_google_vertexai", Class: "ChatVertexAI"},
	"groq":                    {Package: "langchain_groq", Class: "ChatGroq"},
	"huggingface":             {Package: "langchain_huggingface", Class: "ChatHuggingFace"},
	"ibm":                     {Package: "langchain_ibm", Class: "ChatWatsonx"},
	"litellm":                 {Package: "langchain_litellm", Class: "ChatLiteLLM"},
	"mistralai":               {Package: "langchain_mistralai", Class: "ChatMistralAI"},
	"nvidia":                  {Package: "langchain_nvidia_ai_endpoints", Class: "ChatNVIDIA"},
	"ollama":                  {Package: "langchain_ollama", Class: "ChatOllama"},
	"openai":                  {Package: "langchain_openai", Class: "ChatOpenAI"},
	"openrouter":              {Package: "langchain_openrouter", Class: "ChatOpenRouter"},
	"perplexity":              {Package: "langchain_perplexity", Class: "ChatPerplexity"},
	"together":                {Package: "langchain_together", Class: "ChatTogether"},
	"upstage":                 {Package: "langchain_upstage", Class: "ChatUpstage"},
	"xai":                     {Package: "langchain_xai", Class: "ChatXAI"},
}

func BuiltinProviderNames() []string {
	names := make([]string, 0, len(BuiltinProviders))
	for name := range BuiltinProviders {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func NormalizeProvider(provider string) string {
	return strings.ToLower(strings.ReplaceAll(strings.TrimSpace(provider), "-", "_"))
}

func ProviderInfoFor(provider string) (ProviderInfo, bool) {
	info, ok := BuiltinProviders[NormalizeProvider(provider)]
	return info, ok
}

func SupportsProvider(provider string) bool {
	_, ok := ProviderInfoFor(provider)
	return ok
}

func InferModelProvider(modelName string) string {
	return attemptInferModelProvider(modelName)
}

func attemptInferModelProvider(modelName string) string {
	modelLower := strings.ToLower(modelName)

	if hasAnyPrefix(modelLower, "gpt-", "o1", "o3", "chatgpt", "text-davinci") {
		return "openai"
	}
	if strings.HasPrefix(modelLower, "claude") {
		return "anthropic"
	}
	if strings.HasPrefix(modelLower, "command") {
		return "cohere"
	}
	if strings.HasPrefix(modelLower, "accounts/fireworks") {
		return "fireworks"
	}
	if strings.HasPrefix(modelLower, "gemini") {
		return "google_vertexai"
	}
	if hasAnyPrefix(modelLower, "amazon.", "anthropic.", "meta.") {
		return "bedrock"
	}
	if hasAnyPrefix(modelLower, "mistral", "mixtral") {
		return "mistralai"
	}
	if strings.HasPrefix(modelLower, "deepseek") {
		return "deepseek"
	}
	if strings.HasPrefix(modelLower, "grok") {
		return "xai"
	}
	if strings.HasPrefix(modelLower, "sonar") {
		return "perplexity"
	}
	if strings.HasPrefix(modelLower, "solar") {
		return "upstage"
	}
	return ""
}

func parseModel(model string, modelProvider string) (string, string, error) {
	if modelProvider == "" && strings.Contains(model, ":") {
		parts := strings.SplitN(model, ":", 2)
		provider := NormalizeProvider(parts[0])
		if _, ok := BuiltinProviders[provider]; ok {
			modelProvider = provider
			model = parts[1]
		}
	}

	if modelProvider == "" {
		modelProvider = attemptInferModelProvider(model)
	}
	if modelProvider == "" {
		return "", "", fmt.Errorf(
			"Unable to infer model provider for model=%q. Please specify 'model_provider' directly.\n\nSupported providers: %s",
			model,
			strings.Join(BuiltinProviderNames(), ", "),
		)
	}

	modelProvider = NormalizeProvider(modelProvider)
	return model, modelProvider, nil
}

type ChatModelSpec struct {
	Model    string
	Provider string
}

type InitOption func(*initOptions)

type initOptions struct {
	modelProvider string
}

func WithModelProvider(provider string) InitOption {
	return func(opts *initOptions) {
		opts.modelProvider = provider
	}
}

func ParseModel(model string, opts ...InitOption) (ChatModelSpec, error) {
	options := initOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	parsedModel, provider, err := parseModel(model, options.modelProvider)
	if err != nil {
		return ChatModelSpec{}, err
	}
	if _, ok := BuiltinProviders[provider]; !ok {
		return ChatModelSpec{}, fmt.Errorf("Unsupported provider='%s'", provider)
	}
	return ChatModelSpec{Model: parsedModel, Provider: provider}, nil
}

func InitChatModel(model any, opts ...InitOption) (ChatModelSpec, error) {
	modelString, ok := model.(string)
	if !ok {
		return ChatModelSpec{}, fmt.Errorf("model must be a string")
	}

	return ParseModel(modelString, opts...)
}

func hasAnyPrefix(value string, prefixes ...string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(value, prefix) {
			return true
		}
	}
	return false
}
