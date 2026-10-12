package typesafe

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
)

// APIError carries the safe summary format of Python's TypeSafeAPIError
// __str__ (client.py:88-100): "{endpoint}: {status} {phrase}
// (request_id=...)" with no body or header data.

func apiErrorForTest(status int, endpoint, requestID string) *APIError {
	pe := lcerrors.NewProviderError("typesafe", endpoint, status, "SECRET BODY", 0)
	return &APIError{ProviderError: *pe, RequestID: requestID}
}

func TestAPIErrorErrorFormat(t *testing.T) {
	cases := []struct {
		name      string
		err       *APIError
		wantExact string
	}{
		{
			name:      "full",
			err:       apiErrorForTest(401, "POST https://api.typesafe.ai/v1/systemone", "req-1"),
			wantExact: "POST https://api.typesafe.ai/v1/systemone: 401 Unauthorized (request_id=req-1)",
		},
		{
			name:      "no endpoint omits prefix",
			err:       apiErrorForTest(404, "", "req-2"),
			wantExact: "404 Not Found (request_id=req-2)",
		},
		{
			name:      "no request id omits suffix",
			err:       apiErrorForTest(500, "POST https://api.typesafe.ai/v1/systemone", ""),
			wantExact: "POST https://api.typesafe.ai/v1/systemone: 500 Internal Server Error",
		},
		{
			name:      "bare",
			err:       apiErrorForTest(418, "", ""),
			wantExact: "418 I'm a teapot",
		},
		{
			name:      "529 overloaded",
			err:       apiErrorForTest(529, "POST https://api.typesafe.ai/v1/systemone", "req-3"),
			wantExact: "POST https://api.typesafe.ai/v1/systemone: 529 Overloaded (request_id=req-3)",
		},
		{
			name:      "unknown status",
			err:       apiErrorForTest(599, "", ""),
			wantExact: "599 API request failed",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.err.Error(); got != tc.wantExact {
				t.Errorf("Error() mismatch:\n got %q\nwant %q", got, tc.wantExact)
			}
		})
	}
}

func TestAPIErrorErrorExcludesBodyAndHeaders(t *testing.T) {
	err := apiErrorForTest(422, "POST https://api.typesafe.ai/v1/systemone", "req-9")
	msg := err.Error()
	for _, secret := range []string{"SECRET BODY", "retry-after", "authorization", "Bearer"} {
		if strings.Contains(msg, secret) {
			t.Errorf("Error() leaked %q: %s", secret, msg)
		}
	}
}

func TestAPIErrorAsProviderErrorAndSentinels(t *testing.T) {
	err := apiErrorForTest(401, "POST https://api.typesafe.ai/v1/systemone", "req-1")

	var pe *lcerrors.ProviderError
	if !errors.As(err, &pe) {
		t.Fatal("errors.As(*ProviderError) failed")
	}
	if pe.StatusCode != 401 || pe.Provider != "typesafe" || pe.Endpoint != "POST https://api.typesafe.ai/v1/systemone" {
		t.Errorf("provider error fields mismatch: %#v", pe)
	}
	if pe.ModelKind != lcerrors.ModelKindAuth {
		t.Errorf("kind mismatch: %s", pe.ModelKind)
	}
	if !errors.Is(err, lcerrors.ErrModelAuth) {
		t.Error("errors.Is(err, ErrModelAuth) failed")
	}
	if !errors.Is(err, lcerrors.ErrProvider) {
		t.Error("errors.Is(err, ErrProvider) failed for subsumed sentinel")
	}
}

func TestResponseValidationErrorFormat(t *testing.T) {
	err := &ResponseValidationError{FieldPath: "answers.q.type"}
	want := "Invalid response data at 'answers.q.type'."
	if err.Error() != want {
		t.Errorf("Error() mismatch:\n got %q\nwant %q", err.Error(), want)
	}
	if !errors.Is(err, lcerrors.ErrSchemaValidation) {
		t.Error("errors.Is(err, ErrSchemaValidation) failed")
	}
}

func TestTimeoutAndConnectionErrorTypes(t *testing.T) {
	timeout := &timeoutError{timeout: 30 * time.Second}
	if got, want := timeout.Error(), "Request timed out (timeout=30s)."; got != want {
		t.Errorf("timeout message mismatch:\n got %q\nwant %q", got, want)
	}
	if !errors.Is(timeout, lcerrors.ErrModelTimeout) {
		t.Error("timeout error does not match ErrModelTimeout")
	}

	cause := errors.New("boom")
	conn := &connectionError{cause: cause}
	if !errors.Is(conn, lcerrors.ErrModelConnection) {
		t.Error("connection error does not match ErrModelConnection")
	}
	if !errors.Is(conn, cause) {
		t.Error("connection error lost its cause")
	}
}

func TestConnectionErrorMessage(t *testing.T) {
	conn := &connectionError{cause: errors.New("refused")}
	if got, want := conn.Error(), "Unable to connect to the TypeSafe API."; got != want {
		t.Errorf("message mismatch:\n got %q\nwant %q", got, want)
	}
}
