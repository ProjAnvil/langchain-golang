package typesafe

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
)

// Classifier mirrors langchain_typesafe's TypeSafeClassifier (classifier.py,
// #40542/#40570): a self-contained HTTP client for POST {base}/v1/systemone.
//
// Configuration precedence matches Python: explicit values win over
// environment variables (TYPESAFE_API_KEY, TYPESAFE_BASE_URL), the model
// defaults to jev-latest, the base URL to https://api.typesafe.ai, and the
// timeout to 30s. An injected HTTPClient is used as-is; Timeout only
// configures a client created by the classifier.
type Classifier struct {
	// Model is the TypeSafe model name (default jev-latest).
	Model string
	// APIKey authenticates requests (Bearer).
	APIKey string
	// BaseURL is the root URL for the TypeSafe API.
	BaseURL string
	// HTTPClient, when non-nil, is used as-is for requests.
	HTTPClient *http.Client
	// Timeout applies to clients created by the classifier (default 30s).
	Timeout time.Duration
}

const (
	defaultModel       = "jev-latest"
	defaultBaseURL     = "https://api.typesafe.ai"
	defaultTimeout     = 30 * time.Second
	requestIDHeader    = "x-typesafe-request-id"
	retryAfterHeader   = "retry-after"
	retryAfterMSHeader = "retry-after-ms"
	// userAgent is the Go client identifier (DIVERGENCES #5: Python sends
	// "langchain-typesafe/<version>", classifier.py:439).
	userAgent = "langchain-typesafe-go/0.0.1"
	provider  = "typesafe"
)

// ClassifierOption configures NewClassifier.
type ClassifierOption func(*Classifier)

// WithModel overrides the model name (stripped; empty rejected).
func WithModel(model string) ClassifierOption {
	return func(c *Classifier) { c.Model = model }
}

// WithAPIKey overrides TYPESAFE_API_KEY.
func WithAPIKey(apiKey string) ClassifierOption {
	return func(c *Classifier) { c.APIKey = apiKey }
}

// WithBaseURL overrides TYPESAFE_BASE_URL.
func WithBaseURL(baseURL string) ClassifierOption {
	return func(c *Classifier) { c.BaseURL = baseURL }
}

// WithHTTPClient injects a custom HTTP client (used as-is).
func WithHTTPClient(client *http.Client) ClassifierOption {
	return func(c *Classifier) { c.HTTPClient = client }
}

// WithTimeout overrides the request timeout for clients the classifier
// creates; it does not modify an injected HTTPClient.
func WithTimeout(timeout time.Duration) ClassifierOption {
	return func(c *Classifier) { c.Timeout = timeout }
}

// NewClassifier builds a Classifier from defaults, environment variables
// (TYPESAFE_API_KEY, TYPESAFE_BASE_URL), and options, mirroring Python's
// field validators (classifier.py:246-265).
func NewClassifier(opts ...ClassifierOption) (*Classifier, error) {
	classifier := &Classifier{
		Model:   defaultModel,
		APIKey:  os.Getenv("TYPESAFE_API_KEY"),
		BaseURL: cmp.Or(os.Getenv("TYPESAFE_BASE_URL"), defaultBaseURL),
		Timeout: defaultTimeout,
	}
	for _, opt := range opts {
		opt(classifier)
	}
	if err := applyDefaults(classifier); err != nil {
		return nil, err
	}
	return classifier, nil
}

// applyDefaults validates the model and API key and fills unset optional
// fields. It is shared by NewClassifier so manually constructed Classifier
// values can be normalized before use.
func applyDefaults(classifier *Classifier) error {
	if classifier.BaseURL == "" {
		classifier.BaseURL = cmp.Or(os.Getenv("TYPESAFE_BASE_URL"), defaultBaseURL)
	}
	if classifier.Timeout == 0 {
		classifier.Timeout = defaultTimeout
	}
	classifier.Model = strings.TrimSpace(classifier.Model)
	if classifier.Model == "" {
		return errors.New("TypeSafe model must not be empty.") //nolint:staticcheck // ST1005: verbatim Python text (classifier.py:251)
	}
	if strings.TrimSpace(classifier.APIKey) == "" {
		return errors.New( //nolint:staticcheck // ST1005: verbatim Python text (classifier.py:259-262)
			"TypeSafe API key is required. Pass `api_key` or set `TYPESAFE_API_KEY`.")
	}
	return nil
}

// Invoke classifies one request: it serializes the state, POSTs to
// {base}/v1/systemone, and returns the typed answers. Non-2xx responses
// yield *APIError, transport failures wrap ErrModelConnection (timeouts
// wrap ErrModelTimeout), and malformed 2xx bodies yield
// *ResponseValidationError.
func (c *Classifier) Invoke(ctx context.Context, req Request) (Response, error) {
	payload, err := c.buildPayload(req)
	if err != nil {
		return Response{}, err
	}
	endpoint := strings.TrimRight(c.BaseURL, "/") + "/v1/systemone"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return Response{}, &connectionError{cause: err}
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("User-Agent", userAgent)

	client := c.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: c.Timeout}
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		if isTimeoutError(err) {
			return Response{}, &timeoutError{timeout: c.effectiveTimeout(client)}
		}
		return Response{}, &connectionError{cause: err}
	}
	defer func() { _ = resp.Body.Close() }()
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return Response{}, &connectionError{cause: readErr}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Response{}, newAPIError(resp, string(body))
	}
	return parseResponse(resp, body)
}

// effectiveTimeout reports the timeout that governed the failed request:
// the client's own timeout when set, otherwise the classifier's.
func (c *Classifier) effectiveTimeout(client *http.Client) time.Duration {
	if client != nil && client.Timeout > 0 {
		return client.Timeout
	}
	return c.Timeout
}

// buildPayload writes {"state", "model", "questions"} in the Python dict
// insertion order (classifier.py:442-450). Questions are keyed by name in
// stable sorted order — Go map iteration is randomized, so the wire order
// is deterministic rather than insertion-ordered (DIVERGENCES #8).
func (c *Classifier) buildPayload(req Request) ([]byte, error) {
	state, err := SerializeState(req.State)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	buf.WriteString(`{"state":`)
	stateJSON, err := json.Marshal(state)
	if err != nil {
		return nil, fmt.Errorf("typesafe: marshal state: %w", err)
	}
	buf.Write(stateJSON)
	buf.WriteString(`,"model":`)
	modelJSON, err := json.Marshal(c.Model)
	if err != nil {
		return nil, fmt.Errorf("typesafe: marshal model: %w", err)
	}
	buf.Write(modelJSON)
	buf.WriteString(`,"questions":{`)
	for i, name := range slices.Sorted(maps.Keys(req.Questions)) {
		if i > 0 {
			buf.WriteByte(',')
		}
		nameJSON, err := json.Marshal(name)
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal question name: %w", err)
		}
		questionJSON, err := json.Marshal(req.Questions[name])
		if err != nil {
			return nil, fmt.Errorf("typesafe: marshal question %q: %w", name, err)
		}
		buf.Write(nameJSON)
		buf.WriteByte(':')
		buf.Write(questionJSON)
	}
	buf.WriteString(`}}`)
	return buf.Bytes(), nil
}

// newAPIError maps a non-2xx response onto APIError. The embedded
// ProviderError carries the sentinel classification (NewProviderError →
// ModelErrorKindForStatus), so a 400 whose body carries a context-overflow
// marker upgrades to ErrModelContextOverflow — an addition over Python,
// which classifies every 400 as invalid_request (DIVERGENCES #11).
func newAPIError(resp *http.Response, body string) *APIError {
	var retryAfter time.Duration
	if resp.StatusCode == http.StatusTooManyRequests {
		// Only the rate-limit branch parses retry hints, mirroring
		// _STATUS_ERROR_TYPES (client.py:288-295).
		retryAfter = parseRetryAfter(resp.Header)
	}
	providerError := lcerrors.NewProviderError(provider, sanitizedEndpoint(resp), resp.StatusCode, body, retryAfter)
	return &APIError{
		ProviderError: *providerError,
		RequestID:     resp.Header.Get(requestIDHeader),
	}
}

// sanitizedEndpoint mirrors _response_endpoint (client.py:269-285): the
// request method plus URL with query, fragment, and credentials removed,
// and the IPv6 host bracketed.
func sanitizedEndpoint(resp *http.Response) string {
	if resp.Request == nil || resp.Request.URL == nil {
		return ""
	}
	parts := resp.Request.URL
	host := parts.Hostname()
	if host == "" {
		return ""
	}
	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	if port := parts.Port(); port != "" {
		host += ":" + port
	}
	return resp.Request.Method + " " + parts.Scheme + "://" + host + parts.EscapedPath()
}

// isTimeoutError reports whether a transport failure is a timeout (connect,
// read, or request deadline), mirroring httpx2.TimeoutException.
func isTimeoutError(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	if netErr, ok := errors.AsType[net.Error](err); ok {
		return netErr.Timeout()
	}
	return false
}

// parseRetryAfter mirrors _parse_retry_after (client.py:231-257):
// retry-after-ms (milliseconds) wins over retry-after (seconds or
// HTTP-date). An empty header value counts as zero; negative or invalid
// values are ignored. Returns 0 when no valid hint exists.
func parseRetryAfter(header http.Header) time.Duration {
	for _, spec := range []struct {
		name       string
		multiplier time.Duration
	}{
		{retryAfterMSHeader, time.Millisecond},
		{retryAfterHeader, time.Second},
	} {
		raw, present := headerValue(header, spec.name)
		if !present {
			continue
		}
		trimmed := strings.TrimSpace(raw)
		value := 0.0
		parseErr := error(nil)
		if trimmed != "" {
			value, parseErr = strconv.ParseFloat(trimmed, 64)
		}
		if parseErr == nil && !math.IsInf(value, 0) && !math.IsNaN(value) && value >= 0 {
			delay := value * float64(spec.multiplier)
			if !math.IsInf(delay, 0) && delay <= float64(math.MaxInt64) {
				return time.Duration(delay)
			}
		}
		if parseErr != nil && spec.name == retryAfterHeader {
			// HTTP-date form (email.utils parsedate_to_datetime analog).
			if deadline, dateErr := http.ParseTime(raw); dateErr == nil {
				delta := max(time.Until(deadline).Milliseconds(), 0)
				return time.Duration(delta) * time.Millisecond
			}
			continue
		}
		if spec.name == retryAfterHeader {
			// A parseable-but-negative (or non-finite) seconds value ends
			// the search with no hint, matching Python's `return None`.
			return 0
		}
		// An unusable retry-after-ms value falls through to retry-after.
	}
	return 0
}

// headerValue reads a header distinguishing "absent" from "present but
// empty" (http.Header.Get collapses both).
func headerValue(header http.Header, name string) (string, bool) {
	if values, ok := header[http.CanonicalHeaderKey(name)]; ok && len(values) > 0 {
		return values[0], true
	}
	return "", false
}

// parseResponse validates a 2xx body and converts it into a Response
// (client.py:313-347). Validation is strict — an unknown answer type is an
// error, unlike the OpenAI Decisions client's skip behavior — and the
// returned error's FieldPath points at the first failing field.
func parseResponse(resp *http.Response, body []byte) (Response, error) {
	var raw any
	if err := json.Unmarshal(body, &raw); err != nil {
		return Response{}, &ResponseValidationError{FieldPath: "response"}
	}
	root, ok := raw.(map[string]any)
	if !ok {
		return Response{}, &ResponseValidationError{FieldPath: "response"}
	}
	model, ok := root["model"].(string)
	if !ok {
		return Response{}, &ResponseValidationError{FieldPath: "model"}
	}
	rawAnswers, ok := root["answers"].(map[string]any)
	if !ok {
		return Response{}, &ResponseValidationError{FieldPath: "answers"}
	}
	answers := make(map[string]Answer, len(rawAnswers))
	for _, name := range slices.Sorted(maps.Keys(rawAnswers)) {
		answer, err := parseAnswer(name, rawAnswers[name])
		if err != nil {
			return Response{}, err
		}
		answers[name] = answer
	}
	usage := Usage{}
	if rawUsage, present := root["usage"]; present {
		usageMap, ok := rawUsage.(map[string]any)
		if !ok {
			return Response{}, &ResponseValidationError{FieldPath: "usage"}
		}
		usage.InputTokens = optionalInt(usageMap["input_tokens"])
		usage.OutputTokens = optionalInt(usageMap["output_tokens"])
	}
	return Response{
		Model:     model,
		Answers:   answers,
		Usage:     usage,
		RequestID: resp.Header.Get(requestIDHeader),
	}, nil
}

// parseAnswer validates one answers[name] entry against the sealed union
// (types.py:273). Field paths mirror pydantic's error locations.
func parseAnswer(name string, raw any) (Answer, error) {
	field := func(leaf string) *ResponseValidationError {
		return &ResponseValidationError{FieldPath: "answers." + name + "." + leaf}
	}
	entry, ok := raw.(map[string]any)
	if !ok {
		return nil, &ResponseValidationError{FieldPath: "answers." + name}
	}
	switch entry["type"] {
	case "noul":
		value, ok := asFloat(entry["noul"])
		if !ok || value < 0 || value > 1 {
			return nil, field("noul")
		}
		return NoulAnswer{Noul: value}, nil
	case "choice":
		choice, ok := entry["choice"].(string)
		if !ok {
			return nil, field("choice")
		}
		rawProbabilities, ok := entry["probabilities"].(map[string]any)
		if !ok {
			return nil, field("probabilities")
		}
		probabilities := make(map[string]float64, len(rawProbabilities))
		for label, probability := range rawProbabilities {
			value, ok := asFloat(probability)
			if !ok {
				return nil, field("probabilities")
			}
			probabilities[label] = value
		}
		confidence, ok := asFloat(entry["confidence"])
		if !ok || confidence < 0 || confidence > 1 {
			return nil, field("confidence")
		}
		return ChoiceAnswer{Choice: choice, Probabilities: probabilities, Confidence: confidence}, nil
	case "score":
		score, ok := asFloat(entry["score"])
		if !ok {
			return nil, field("score")
		}
		rawLegend, ok := entry["legend"].(map[string]any)
		if !ok {
			return nil, field("legend")
		}
		legend := make(map[int]any, len(rawLegend))
		for key, description := range rawLegend {
			level, err := strconv.Atoi(key)
			if err != nil {
				return nil, field("legend")
			}
			legend[level] = description
		}
		rawProbabilities, ok := entry["probabilities"].(map[string]any)
		if !ok {
			return nil, field("probabilities")
		}
		probabilities := make(map[int]float64, len(rawProbabilities))
		for key, probability := range rawProbabilities {
			level, err := strconv.Atoi(key)
			if err != nil {
				return nil, field("probabilities")
			}
			value, ok := asFloat(probability)
			if !ok {
				return nil, field("probabilities")
			}
			probabilities[level] = value
		}
		confidence, ok := asFloat(entry["confidence"])
		if !ok || confidence < 0 || confidence > 1 {
			return nil, field("confidence")
		}
		return ScoreAnswer{Score: score, Legend: legend, Probabilities: probabilities, Confidence: confidence}, nil
	default:
		return nil, field("type")
	}
}

// asFloat accepts a JSON number (decoded as float64).
func asFloat(raw any) (float64, bool) {
	value, ok := raw.(float64)
	return value, ok
}

// optionalInt converts a JSON number to an int pointer, nil for null/absent.
func optionalInt(raw any) *int {
	if value, ok := raw.(float64); ok {
		intValue := int(value)
		return &intValue
	}
	return nil
}
