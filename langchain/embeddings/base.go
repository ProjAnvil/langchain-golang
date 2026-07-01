package embeddings

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
	"azure_ai":        {Package: "langchain_azure_ai.embeddings", Class: "AzureAIOpenAIApiEmbeddingsModel"},
	"azure_openai":    {Package: "langchain_openai", Class: "AzureOpenAIEmbeddings"},
	"bedrock":         {Package: "langchain_aws", Class: "BedrockEmbeddings"},
	"cohere":          {Package: "langchain_cohere", Class: "CohereEmbeddings"},
	"google_genai":    {Package: "langchain_google_genai", Class: "GoogleGenerativeAIEmbeddings"},
	"google_vertexai": {Package: "langchain_google_vertexai", Class: "VertexAIEmbeddings"},
	"huggingface":     {Package: "langchain_huggingface", Class: "HuggingFaceEmbeddings"},
	"mistralai":       {Package: "langchain_mistralai", Class: "MistralAIEmbeddings"},
	"ollama":          {Package: "langchain_ollama", Class: "OllamaEmbeddings"},
	"openai":          {Package: "langchain_openai", Class: "OpenAIEmbeddings"},
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

func parseModelString(modelName string) (string, string, error) {
	if !strings.Contains(modelName, ":") {
		return "", "", fmt.Errorf(
			"Invalid model format %q. Model name must be in format 'provider:model-name'. Supported providers: %v",
			modelName,
			BuiltinProviderNames(),
		)
	}

	parts := strings.SplitN(modelName, ":", 2)
	provider := NormalizeProvider(parts[0])
	model := strings.TrimSpace(parts[1])

	if _, ok := BuiltinProviders[provider]; !ok {
		return "", "", unsupportedProviderError(provider)
	}
	if model == "" {
		return "", "", fmt.Errorf("Model name cannot be empty")
	}
	return provider, model, nil
}

func inferModelAndProvider(model string, provider string) (string, string, error) {
	if strings.TrimSpace(model) == "" {
		return "", "", fmt.Errorf("Model name cannot be empty")
	}
	provider = NormalizeProvider(provider)
	if provider == "" && strings.Contains(model, ":") {
		return parseModelString(model)
	}

	modelName := model
	if provider == "" {
		return "", "", fmt.Errorf(
			"Must specify either a model string in format 'provider:model-name' or explicitly set provider from: %v",
			BuiltinProviderNames(),
		)
	}
	if _, ok := BuiltinProviders[provider]; !ok {
		return "", "", unsupportedProviderError(provider)
	}
	return provider, modelName, nil
}

type EmbeddingsSpec struct {
	Model    string
	Provider string
}

type InitOption func(*initOptions)

type initOptions struct {
	provider string
}

func WithProvider(provider string) InitOption {
	return func(opts *initOptions) {
		opts.provider = provider
	}
}

func ParseModel(model string, opts ...InitOption) (EmbeddingsSpec, error) {
	options := initOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	provider, modelName, err := inferModelAndProvider(model, options.provider)
	if err != nil {
		return EmbeddingsSpec{}, err
	}
	return EmbeddingsSpec{Model: modelName, Provider: provider}, nil
}

func InitEmbeddings(model string, opts ...InitOption) (EmbeddingsSpec, error) {
	return ParseModel(model, opts...)
}

func unsupportedProviderError(provider string) error {
	return fmt.Errorf(
		"Provider '%s' is not supported. Supported providers and their required packages:\n%s",
		provider,
		providerList(),
	)
}

func providerList() string {
	lines := make([]string, 0, len(BuiltinProviders))
	for _, provider := range BuiltinProviderNames() {
		info := BuiltinProviders[provider]
		lines = append(lines, fmt.Sprintf("  - %s: %s", provider, strings.ReplaceAll(info.Package, "_", "-")))
	}
	return strings.Join(lines, "\n")
}
