package openai

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/modelprofiles"
)

// AzureTextModel adapts LangChain text-completion calls to the Azure OpenAI
// `/completions` endpoint, mirroring Python's AzureOpenAI (text LLM). The
// request/response bodies match the standard `/completions`; only the endpoint
// URL and `api-key` header differ.
type AzureTextModel struct {
	text TextModel
	az   azureClient
}

var _ language.LLM = AzureTextModel{}

// NewAzureTextModel builds an Azure text completion model.
func NewAzureTextModel(endpoint, deployment, apiVersion, apiKey string, opts ...modelconfig.Option) AzureTextModel {
	az := azureClient{endpoint: endpoint, deployment: deployment, apiVersion: apiVersion, apiKey: apiKey}.fromEnv()
	return AzureTextModel{text: NewTextModel(opts...), az: az}
}

func (m AzureTextModel) Invoke(ctx context.Context, prompt string, opts ...runnables.Option) (string, error) {
	resp, err := azurePost[completionResponse](m.az, ctx, m.text.config, "/completions", m.text.buildRequest(prompt))
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("azure openai %s: no choices in completion response", m.text.config.Model)
	}
	return resp.Choices[0].Text, nil
}

func (m AzureTextModel) Batch(ctx context.Context, inputs []string, opts ...runnables.Option) ([]string, error) {
	runnable := runnables.NewFunc(m.Invoke, m.InputSchema(), m.OutputSchema())
	return runnable.Batch(ctx, inputs, opts...)
}

func (m AzureTextModel) Stream(ctx context.Context, input string, opts ...runnables.Option) (runnables.Stream[string], error) {
	runnable := runnables.NewFunc(m.Invoke, m.InputSchema(), m.OutputSchema())
	return runnable.Stream(ctx, input, opts...)
}

func (m AzureTextModel) InputSchema() schema.Schema  { return schema.String("text prompt") }
func (m AzureTextModel) OutputSchema() schema.Schema { return schema.String("text completion") }

func (m AzureTextModel) ModelProfile() modelprofiles.Profile {
	return modelprofiles.Profile{"text_inputs": true, "text_outputs": true}
}
