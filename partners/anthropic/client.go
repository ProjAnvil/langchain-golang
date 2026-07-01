package anthropic

import (
	"context"
	"net/http"

	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// providerName labels provider errors for this adapter.
const providerName = "anthropic"

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
		setHeaders(req, cfg)
	})
}

func setHeaders(req *http.Request, cfg modelconfig.Config) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")
	if cfg.APIKey != "" {
		req.Header.Set("x-api-key", cfg.APIKey)
	}
	for name, value := range cfg.Headers {
		req.Header.Set(name, value)
	}
}
