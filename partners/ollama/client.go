package ollama

import (
	"context"
	"net/http"

	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

const (
	defaultBaseURL = "http://localhost:11434"

	// providerName labels provider errors for this adapter.
	providerName = "ollama"
)

// postJSON posts a JSON request and decodes the JSON response, retrying
// retryable errors per cfg. Non-2xx responses and network timeouts surface as
// typed core/lcerrors values.
func postJSON[T any](
	ctx context.Context,
	cfg modelconfig.Config,
	endpoint string,
	requestPayload any,
) (T, error) {
	return httpclient.PostJSON[T](ctx, providerName, cfg, endpoint, requestPayload, func(req *http.Request) {
		configureRequest(req, cfg)
	})
}

// configureRequest sets Ollama custom headers on req.
func configureRequest(req *http.Request, cfg modelconfig.Config) {
	req.Header.Set("Content-Type", "application/json")
	for name, value := range cfg.Headers {
		req.Header.Set(name, value)
	}
}
