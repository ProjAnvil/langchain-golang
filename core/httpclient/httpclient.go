// Package httpclient provides a shared JSON-over-HTTP helper for provider
// adapters. It centralizes request/response handling, retry-policy wiring, and
// the translation of HTTP and network failures into typed core/lcerrors values,
// so provider packages do not each duplicate this logic.
package httpclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/retry"
)

// ConfigureRequest mutates an outgoing request to add provider-specific headers
// or authentication. It is invoked after Content-Type has been set.
type ConfigureRequest func(*http.Request)

// PostJSON posts a JSON-encoded payload to endpoint, decodes the JSON response
// into a value of type T, and retries retryable failures per cfg. Non-2xx
// responses and network timeouts are returned as typed lcerrors values.
//
// configure may be nil; otherwise it is called on each attempt to decorate the
// request with provider headers/auth derived from cfg.
func PostJSON[T any](
	ctx context.Context,
	provider string,
	cfg modelconfig.Config,
	endpoint string,
	payload any,
	configure ConfigureRequest,
) (T, error) {
	var zero T

	body, err := json.Marshal(payload)
	if err != nil {
		return zero, err
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	url := strings.TrimRight(cfg.BaseURL, "/") + endpoint

	var result T
	err = retry.Do(ctx, retry.Policy{
		MaxAttempts:       cfg.MaxRetries + 1,
		Delay:             cfg.RetryDelay,
		BackoffMultiplier: cfg.RetryBackoffMultiplier,
		MaxDelay:          cfg.RetryMaxDelay,
		ShouldRetry:       IsRetryable,
	}, func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		if configure != nil {
			configure(req)
		}

		resp, err := client.Do(req)
		if err != nil {
			return lcerrors.WrapTransport(err)
		}

		respBody, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return lcerrors.WrapTransport(readErr)
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return lcerrors.NewProviderError(provider, endpoint, resp.StatusCode, string(respBody), RetryAfter(resp))
		}
		var parsed T
		if err := json.Unmarshal(respBody, &parsed); err != nil {
			return fmt.Errorf("decode %s %s response: %w", provider, endpoint, err)
		}
		result = parsed
		return nil
	})
	if err != nil {
		return zero, err
	}
	return result, nil
}

// ResponseError reads and closes resp.Body and returns a typed
// lcerrors.ProviderError describing a non-2xx response. Callers should invoke it
// only when resp.StatusCode is outside the 2xx range; it consumes the body.
func ResponseError(provider, endpoint string, resp *http.Response) error {
	body, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil {
		return lcerrors.WrapTransport(err)
	}
	return lcerrors.NewProviderError(provider, endpoint, resp.StatusCode, string(body), RetryAfter(resp))
}

// IsRetryable reports whether err represents a failure worth retrying: HTTP 429,
// HTTP 408, any 5xx response, or a network/timeout failure classified as
// ErrTimeout.
func IsRetryable(err error) bool {
	var pe *lcerrors.ProviderError
	if errors.As(err, &pe) {
		return lcerrors.IsRetryableStatus(pe.StatusCode)
	}
	return errors.Is(err, lcerrors.ErrTimeout)
}

// RetryAfter parses the HTTP Retry-After response header, supporting both the
// delta-seconds and HTTP-date forms. It returns 0 when the header is absent or
// unparseable, and never returns a negative duration.
func RetryAfter(resp *http.Response) time.Duration {
	value := resp.Header.Get("Retry-After")
	if value == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(value); err == nil {
		if seconds < 0 {
			return 0
		}
		return time.Duration(seconds) * time.Second
	}
	if t, err := http.ParseTime(value); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}
