package gemini

import (
	"context"
	"net/http"

	"github.com/projanvil/langchain-golang/core/modelconfig"
	"google.golang.org/genai"
)

const (
	// providerName labels provider errors for this adapter.
	providerName = "gemini"
	// defaultBaseURL is the Gemini API (ML dev) endpoint the genai SDK posts
	// to when no override is configured. Requests hit
	// {base}/v1beta/models/{model}:generateContent (streaming appends
	// "?alt=sse").
	defaultBaseURL = "https://generativelanguage.googleapis.com/"
	// defaultModel mirrors Python ChatGoogleGenerativeAI's default model.
	defaultModel = "gemini-2.5-flash"
)

// newClient builds a genai SDK client from the provider config. The backend is
// pinned to BackendGeminiAPI so a stray GOOGLE_GENAI_USE_VERTEXAI environment
// variable cannot silently reroute this adapter onto the Vertex path (which
// needs project/location instead of an API key and is out of scope here).
//
// The SDK reads GEMINI_API_KEY / GOOGLE_API_KEY itself when APIKey is empty
// (client.go getAPIKeyFromEnv in the SDK: GOOGLE_API_KEY wins when both are
// set), which is exactly Python's google-genai behavior — so the env fallback
// the task requires comes for free on top of the explicit config.
func newClient(cfg modelconfig.Config) (*genai.Client, error) {
	clientCfg := &genai.ClientConfig{
		APIKey:  cfg.APIKey,
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: defaultBaseURL,
			Headers: http.Header{},
		},
	}
	if cfg.BaseURL != "" {
		clientCfg.HTTPOptions.BaseURL = cfg.BaseURL
	}
	for name, value := range cfg.Headers {
		clientCfg.HTTPOptions.Headers.Set(name, value)
	}
	if cfg.Timeout > 0 {
		clientCfg.HTTPOptions.Timeout = new(cfg.Timeout)
	}
	if cfg.HTTPClient != nil {
		clientCfg.HTTPClient = cfg.HTTPClient
	}
	return genai.NewClient(context.Background(), clientCfg)
}
