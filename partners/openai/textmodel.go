package openai

import (
	"context"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/modelprofiles"
)

// TextModel adapts LangChain text-in, text-out calls to OpenAI's legacy
// Completions API (the non-chat `/completions` endpoint).
type TextModel struct {
	config modelconfig.Config
}

// Compile-time guard: TextModel (value receiver) satisfies language.LLM.
var _ language.LLM = TextModel{}

// NewTextModel creates an OpenAI text completion model adapter.
func NewTextModel(opts ...modelconfig.Option) TextModel {
	cfg := modelconfig.New(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "gpt-3.5-turbo-instruct"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return TextModel{config: cfg}
}

// Invoke calls the OpenAI Completions API and returns choices[0].text.
func (m TextModel) Invoke(
	ctx context.Context,
	prompt string,
	opts ...runnables.Option,
) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, m.config.Timeout)
	defer cancel()
	resp, err := postJSON[completionResponse](ctx, m.config, "/completions", m.buildRequest(prompt))
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("openai %s: no choices in completion response", m.config.Model)
	}
	return resp.Choices[0].Text, nil
}

// Batch invokes the model for each input while preserving order.
func (m TextModel) Batch(
	ctx context.Context,
	inputs []string,
	opts ...runnables.Option,
) ([]string, error) {
	runnable := runnables.NewFunc(m.Invoke, m.InputSchema(), m.OutputSchema())
	return runnable.Batch(ctx, inputs, opts...)
}

// Stream returns a single-chunk stream containing the Invoke response.
func (m TextModel) Stream(
	ctx context.Context,
	input string,
	opts ...runnables.Option,
) (runnables.Stream[string], error) {
	runnable := runnables.NewFunc(m.Invoke, m.InputSchema(), m.OutputSchema())
	return runnable.Stream(ctx, input, opts...)
}

// InputSchema returns the LLM input schema.
func (m TextModel) InputSchema() schema.Schema {
	return schema.String("text prompt")
}

// OutputSchema returns the LLM output schema.
func (m TextModel) OutputSchema() schema.Schema {
	return schema.String("text completion")
}

// ModelProfile returns a minimal text-only profile.
func (m TextModel) ModelProfile() modelprofiles.Profile {
	return modelprofiles.Profile{
		"text_inputs":  true,
		"text_outputs": true,
	}
}

func (m TextModel) buildRequest(prompt string) completionRequest {
	payload := completionRequest{
		Model:  m.config.Model,
		Prompt: prompt,
	}
	if m.config.MaxTokens != nil {
		payload.MaxTokens = m.config.MaxTokens
	}
	if m.config.Temperature != nil {
		payload.Temperature = m.config.Temperature
	}
	return payload
}

type completionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
}

type completionResponse struct {
	Choices []struct {
		Text string `json:"text"`
	} `json:"choices"`
	Usage usagePayload `json:"usage"`
}
