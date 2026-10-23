package decisions

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/lcerrors"
)

// defaultBaseURL is the OpenAI API root used when no base URL is set.
const defaultBaseURL = "https://api.openai.com/v1"

// Client posts decision requests to the OpenAI Decisions API, mirroring
// langchain_openai.decisions.OpenAIDecisions (PR #41125). It is deliberately
// self-contained (explicit fields rather than modelconfig.Config): the
// Decisions endpoint is not a chat/embeddings API and keeps its own error
// mapping (see the parity wave 3 design, §3.C).
//
// LangSmith gateway resolution is not implemented — only explicit fields
// plus the OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_BASE_URL environment
// variables are honored (DIVERGENCES #1). Usage is returned on the response
// only; it is not attached to any active run (DIVERGENCES #2).
type Client struct {
	// Model is the Decisions model name, e.g. "gpt-6-luna". Required.
	Model string
	// APIKey authenticates requests. NewClient reads OPENAI_API_KEY when
	// the field would otherwise be empty.
	APIKey string
	// BaseURL is the API root; requests go to {BaseURL}/decisions.
	// NewClient reads OPENAI_API_BASE, then OPENAI_BASE_URL, and defaults
	// to https://api.openai.com/v1.
	BaseURL string
	// HTTPClient performs the requests. Nil means &http.Client{}.
	HTTPClient *http.Client
	// Timeout bounds each Invoke when positive; it is applied via the
	// request context.
	Timeout time.Duration
}

// NewClient builds a Client for the given model, resolving the API key and
// base URL from the environment (parity wave 3 plan, Task 4).
func NewClient(model string) (*Client, error) {
	if model == "" {
		return nil, fmt.Errorf("openai decisions: model must not be empty")
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	if strings.TrimSpace(apiKey) == "" {
		// Python-aligned message with the LangSmith gateway sentence
		// trimmed (DIVERGENCES #1, PR #41125).
		return nil, fmt.Errorf("OpenAI API key is required. Pass an API key or set `OPENAI_API_KEY`.") //nolint:staticcheck // ST1005: verbatim Python message
	}
	baseURL := cmp.Or(
		firstNonEmptyEnv("OPENAI_API_BASE"),
		firstNonEmptyEnv("OPENAI_BASE_URL"),
		defaultBaseURL,
	)
	return &Client{Model: model, APIKey: apiKey, BaseURL: baseURL}, nil
}

func firstNonEmptyEnv(name string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return ""
}

// Invoke answers one request synchronously (Python OpenAIDecisions.invoke,
// base.py:244). The input is converted via toDecisionInput, questions are
// serialized sorted by name, and errors are mapped onto the lcerrors
// vocabulary: non-2xx responses become lcerrors.ProviderError (the 400
// context-overflow sniff applies automatically), malformed 2xx bodies wrap
// ErrSchemaValidation, and transport failures wrap the timeout/connection
// sentinels.
func (c *Client) Invoke(ctx context.Context, req Request) (Response, error) {
	payload, err := c.payload(req)
	if err != nil {
		return Response{}, err
	}
	body, err := marshalJSONText(payload)
	if err != nil {
		return Response{}, fmt.Errorf("%w: could not encode OpenAI Decisions request: %v", lcerrors.ErrInvalidInput, err)
	}

	ctx, cancel := c.withTimeout(ctx)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint(), strings.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("openai decisions: bad request URL: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)

	httpClient := cmp.Or(c.HTTPClient, &http.Client{})
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return Response{}, lcerrors.WrapTransport(err)
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return Response{}, lcerrors.WrapTransport(err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Response{}, lcerrors.NewProviderError("openai", "/decisions", resp.StatusCode, string(respBody), 0)
	}
	return parseResponse(resp.Header.Get("x-request-id"), respBody)
}

func (c *Client) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if c.Timeout > 0 {
		return context.WithTimeout(ctx, c.Timeout)
	}
	return context.WithCancel(ctx)
}

func (c *Client) endpoint() string {
	return strings.TrimSuffix(c.BaseURL, "/") + "/decisions"
}

// payload builds the request body: {"model","input","questions":[...]} with
// the question name injected from the map key and the array sorted by name
// (Go map iteration is randomized; Python preserves insertion order —
// DIVERGENCES #8, PR #41125).
func (c *Client) payload(req Request) (map[string]any, error) {
	input, err := toDecisionInput(req.Input)
	if err != nil {
		return nil, err
	}
	names := slices.Sorted(maps.Keys(req.Questions))
	questions := make([]any, 0, len(names))
	for _, name := range names {
		question := req.Questions[name]
		if err := validateQuestion(question); err != nil {
			return nil, err
		}
		questions = append(questions, question.toAPI(name))
	}
	return map[string]any{
		"model":     c.Model,
		"input":     input,
		"questions": questions,
	}, nil
}

// validateQuestion enforces the constraints pydantic declares on the Python
// models (min_length=1 instructions, min_length=1 choices, min_length=2
// levels — types.py:59,102,169).
func validateQuestion(q Question) error {
	switch v := q.(type) {
	case Predicate:
		if v.Instructions == "" {
			return fmt.Errorf("%w: predicate instructions must not be empty", lcerrors.ErrInvalidInput)
		}
	case Choice:
		if v.Instructions == "" {
			return fmt.Errorf("%w: choice instructions must not be empty", lcerrors.ErrInvalidInput)
		}
		if len(v.Choices) < 1 {
			return fmt.Errorf("%w: choice must have at least one option", lcerrors.ErrInvalidInput)
		}
	case Score:
		if v.Instructions == "" {
			return fmt.Errorf("%w: score instructions must not be empty", lcerrors.ErrInvalidInput)
		}
		if len(v.Levels) < 2 {
			return fmt.Errorf("%w: score must have at least two levels", lcerrors.ErrInvalidInput)
		}
	}
	return nil
}

// answerTypes mirrors Python's _ANSWER_TYPES (base.py:32): answers whose
// type is not in the set are skipped rather than rejected.
var answerTypes = []string{"predicate", "choice", "score", "refusal"}

type rawUsage struct {
	InputTokens  *int `json:"input_tokens"`
	OutputTokens *int `json:"output_tokens"`
}

// parseResponse validates and normalizes a 2xx response body (Python
// _parse_response/_normalize_answer, base.py:359-394): answers are keyed by
// name, unknown types are skipped, choice/score probability arrays become
// maps, and score levels additionally build the legend.
func parseResponse(requestID string, body []byte) (Response, error) {
	var raw struct {
		Model   string            `json:"model"`
		Answers *[]map[string]any `json:"answers"`
		Usage   *rawUsage         `json:"usage"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return Response{}, invalidResponseError(err)
	}
	if raw.Model == "" || raw.Answers == nil {
		return Response{}, invalidResponseError(fmt.Errorf("missing model or answers"))
	}
	answers := make(map[string]Answer, len(*raw.Answers))
	for _, entry := range *raw.Answers {
		answerType, _ := entry["type"].(string)
		if !slices.Contains(answerTypes, answerType) {
			continue
		}
		name, ok := entry["name"].(string)
		if !ok {
			return Response{}, invalidResponseError(fmt.Errorf("answer without a name"))
		}
		answer, err := normalizeAnswer(answerType, entry)
		if err != nil {
			return Response{}, err
		}
		answers[name] = answer
	}
	response := Response{Model: raw.Model, Answers: answers, RequestID: requestID}
	if raw.Usage != nil {
		response.Usage = Usage{InputTokens: raw.Usage.InputTokens, OutputTokens: raw.Usage.OutputTokens}
	}
	return response, nil
}

// invalidResponseError wraps every malformed 2xx body (Python raises
// openai.APIResponseValidationError with this exact message, base.py:375).
func invalidResponseError(cause error) error {
	return fmt.Errorf("%w: Invalid response from the OpenAI Decisions API. (%v)", //nolint:staticcheck // ST1005: verbatim Python message
		lcerrors.ErrSchemaValidation, cause)
}

func normalizeAnswer(answerType string, entry map[string]any) (Answer, error) {
	switch answerType {
	case "predicate":
		probability, err := unitIntervalField(entry, "probability")
		if err != nil {
			return nil, err
		}
		return PredicateAnswer{Probability: probability}, nil
	case "choice":
		return normalizeChoiceAnswer(entry)
	case "score":
		return normalizeScoreAnswer(entry)
	case "refusal":
		extra := make(map[string]any, len(entry))
		for key, value := range entry {
			if key != "name" && key != "type" {
				extra[key] = value
			}
		}
		return RefusalAnswer{Extra: extra}, nil
	default:
		return nil, invalidResponseError(fmt.Errorf("unhandled answer type %q", answerType))
	}
}

func normalizeChoiceAnswer(entry map[string]any) (Answer, error) {
	var choice any
	switch v := entry["choice"].(type) {
	case string, bool:
		choice = v
	default:
		return nil, invalidResponseError(fmt.Errorf("choice answer without a valid choice"))
	}
	items, err := probabilityItems(entry["probabilities"])
	if err != nil {
		return nil, err
	}
	probabilities := make(map[string]float64, len(items))
	for _, item := range items {
		probabilities[answerKey(item.value)] = item.probability
	}
	confidence, err := unitIntervalField(entry, "confidence")
	if err != nil {
		return nil, err
	}
	return ChoiceAnswer{Choice: choice, Probabilities: probabilities, Confidence: confidence}, nil
}

func normalizeScoreAnswer(entry map[string]any) (Answer, error) {
	score, err := numberField(entry, "score")
	if err != nil {
		return nil, err
	}
	items, err := probabilityItems(entry["probabilities"])
	if err != nil {
		return nil, err
	}
	legend := make(map[int]string, len(items))
	probabilities := make(map[int]float64, len(items))
	for _, item := range items {
		label, ok := item.entry["label"].(string)
		if !ok {
			return nil, invalidResponseError(fmt.Errorf("score level without a label"))
		}
		index, err := intValue(item.value)
		if err != nil {
			return nil, err
		}
		legend[index] = label
		probabilities[index] = item.probability
	}
	confidence, err := unitIntervalField(entry, "confidence")
	if err != nil {
		return nil, err
	}
	return ScoreAnswer{Score: score, Legend: legend, Probabilities: probabilities, Confidence: confidence}, nil
}

type probabilityItem struct {
	value       any
	probability float64
	entry       map[string]any
}

// probabilityItems converts the wire form [{"value": v, "probability": p}]
// used by choice and score answers.
func probabilityItems(raw any) ([]probabilityItem, error) {
	list, ok := raw.([]any)
	if !ok {
		return nil, invalidResponseError(fmt.Errorf("probabilities must be an array"))
	}
	items := make([]probabilityItem, 0, len(list))
	for _, element := range list {
		entry, ok := element.(map[string]any)
		if !ok {
			return nil, invalidResponseError(fmt.Errorf("probability entry must be an object"))
		}
		probability, err := numberField(entry, "probability")
		if err != nil {
			return nil, err
		}
		value, present := entry["value"]
		if !present {
			return nil, invalidResponseError(fmt.Errorf("probability entry without a value"))
		}
		items = append(items, probabilityItem{value: value, probability: probability, entry: entry})
	}
	return items, nil
}

func numberField(entry map[string]any, key string) (float64, error) {
	value, ok := entry[key].(float64)
	if !ok {
		return 0, invalidResponseError(fmt.Errorf("field %q must be a number", key))
	}
	return value, nil
}

func unitIntervalField(entry map[string]any, key string) (float64, error) {
	value, err := numberField(entry, key)
	if err != nil {
		return 0, err
	}
	if value < 0 || value > 1 {
		return 0, invalidResponseError(fmt.Errorf("field %q must be between 0 and 1", key))
	}
	return value, nil
}

// answerKey stringifies a probability value for map keys: strings stay
// themselves, bools become "true"/"false" (Python str(True) capitalizes —
// deliberate Go divergence, only observable for bool-valued choices).
func answerKey(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case bool:
		return strconv.FormatBool(v)
	default:
		return fmt.Sprint(v)
	}
}

// intValue converts a JSON number (decoded as float64) to an int index.
func intValue(value any) (int, error) {
	f, ok := value.(float64)
	if !ok || f != float64(int(f)) {
		return 0, invalidResponseError(fmt.Errorf("score level value must be an integer index"))
	}
	return int(f), nil
}
