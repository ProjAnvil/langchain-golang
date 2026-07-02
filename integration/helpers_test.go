//go:build integration

package integration

import (
	"context"
	"os"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/partners/anthropic"
	"github.com/projanvil/langchain-golang/partners/openai"
)

// providerConfig is the env-derived configuration for one provider.
type providerConfig struct {
	enabled bool
	apiKey  string
	baseURL string
	model   string
}

func loadProviderConfig(prefix string) providerConfig {
	return providerConfig{
		enabled: os.Getenv(prefix+"_ENABLED") == "1",
		apiKey:  os.Getenv(prefix + "_API_KEY"),
		baseURL: os.Getenv(prefix + "_BASE_URL"),
		model:   os.Getenv(prefix + "_MODEL"),
	}
}

// require skips (or fails) the test according to whether the provider is
// configured. Unenabled → skip with a hint; enabled but missing key/model →
// fatal (config error, not a skip).
func (p providerConfig) require(t *testing.T, prefix string) {
	t.Helper()
	if !p.enabled {
		t.Skipf("%s_ENABLED!=1 — set it (and %s_API_KEY/%s_BASE_URL/%s_MODEL) in .env to run",
			prefix, prefix, prefix, prefix)
	}
	if p.apiKey == "" {
		t.Fatalf("%s_API_KEY is required when %s_ENABLED=1 (check .env)", prefix, prefix)
	}
	if p.model == "" {
		t.Fatalf("%s_MODEL is required when %s_ENABLED=1 (check .env)", prefix, prefix)
	}
}

func newOpenAIModel(t *testing.T) language.ChatModel {
	t.Helper()
	p := loadProviderConfig("OPENAI")
	p.require(t, "OPENAI")
	return openai.NewChatModel(
		modelconfig.WithAPIKey(p.apiKey),
		modelconfig.WithBaseURL(p.baseURL),
		modelconfig.WithModel(p.model),
	)
}

func newAnthropicModel(t *testing.T) language.ChatModel {
	t.Helper()
	p := loadProviderConfig("ANTHROPIC")
	p.require(t, "ANTHROPIC")
	return anthropic.NewChatModel(
		modelconfig.WithAPIKey(p.apiKey),
		modelconfig.WithBaseURL(p.baseURL),
		modelconfig.WithModel(p.model),
	)
}

// echoTool returns a trivial tool the agent can call end-to-end, so the
// model<->tools loop can be exercised against a real provider.
func echoTool(t *testing.T) tools.Tool {
	t.Helper()
	tool, err := tools.NewStructuredFunc(
		"echo",
		"Echo back the provided text verbatim.",
		schema.Object(map[string]schema.Schema{
			"text": schema.String("the text to echo back"),
		}, "text"),
		func(ctx context.Context, in map[string]any) (tools.Result, error) {
			text, _ := in["text"].(string)
			return tools.Result{Content: text}, nil
		},
	)
	if err != nil {
		t.Fatalf("build echo tool: %v", err)
	}
	return tool
}
