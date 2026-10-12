package exa

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// Compile-time guards: both tools satisfy the shared tool contract.
var (
	_ tools.Tool = (*SearchResults)(nil)
	_ tools.Tool = (*FindSimilarResults)(nil)
)

// Python parity: langchain_exa.tools.ExaSearchResults keeps the historical
// "_json" tool names and description copy (tools.py:86-91, 167-172).
func TestSearchResultsNameAndDescription(t *testing.T) {
	tool := &SearchResults{}

	if got := tool.Name(); got != "exa_search_results_json" {
		t.Fatalf("name: got %q", got)
	}
	want := "Exa Search, one of the best web search APIs built for AI. " +
		"Input should be an Exa-optimized query. " +
		"Output is a JSON array of the query results"
	if got := tool.Description(); got != want {
		t.Fatalf("description: got %q want %q", got, want)
	}
}

func TestSearchResultsArgsSchema(t *testing.T) {
	argsSchema := (&SearchResults{}).ArgsSchema()

	if argsSchema["type"] != "object" {
		t.Fatalf("type: got %v", argsSchema["type"])
	}
	required, ok := argsSchema["required"].([]string)
	if !ok || len(required) != 1 || required[0] != "query" {
		t.Fatalf("required: got %v", argsSchema["required"])
	}
	props, ok := argsSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties: got %T", argsSchema["properties"])
	}
	for _, name := range []string{
		"query", "numResults", "livecrawl", "startPublishedDate", "endPublishedDate",
		"includeDomains", "excludeDomains", "contents",
	} {
		if _, ok := props[name]; !ok {
			t.Fatalf("properties missing %q: %v", name, props)
		}
	}
	livecrawl, ok := props["livecrawl"].(schema.Schema)
	if !ok {
		t.Fatalf("livecrawl property: got %T", props["livecrawl"])
	}
	enum, ok := livecrawl["enum"].([]string)
	if !ok || len(enum) != 3 || enum[0] != "always" || enum[1] != "fallback" || enum[2] != "never" {
		t.Fatalf("livecrawl enum: got %v", livecrawl["enum"])
	}
	domains, ok := props["includeDomains"].(schema.Schema)
	if !ok || domains["type"] != "array" {
		t.Fatalf("includeDomains property: got %v", props["includeDomains"])
	}
	items, ok := domains["items"].(schema.Schema)
	if !ok || items["type"] != "string" {
		t.Fatalf("includeDomains items: got %v", domains["items"])
	}
	contents, ok := props["contents"].(schema.Schema)
	if !ok {
		t.Fatalf("contents property: got %T", props["contents"])
	}
	contentsProps, ok := contents["properties"].(map[string]any)
	if !ok {
		t.Fatalf("contents properties: got %T", contents["properties"])
	}
	text, ok := contentsProps["text"].(schema.Schema)
	if !ok {
		t.Fatalf("contents.text: got %T", contentsProps["text"])
	}
	textProps, ok := text["properties"].(map[string]any)
	if !ok {
		t.Fatalf("contents.text properties: got %T", text["properties"])
	}
	if _, ok := textProps["maxCharacters"]; !ok {
		t.Fatalf("contents.text missing maxCharacters: %v", textProps)
	}
	highlights, ok := contentsProps["highlights"].(schema.Schema)
	if !ok {
		t.Fatalf("contents.highlights: got %T", contentsProps["highlights"])
	}
	highlightsProps, ok := highlights["properties"].(map[string]any)
	if !ok {
		t.Fatalf("contents.highlights properties: got %T", highlights["properties"])
	}
	for _, name := range []string{"query", "numSentences"} {
		if _, ok := highlightsProps[name]; !ok {
			t.Fatalf("contents.highlights missing %q: %v", name, highlightsProps)
		}
	}
}

// Python parity: ExaFindSimilarResults takes a required url first argument
// and adds the category filter on top of the shared search options.
func TestFindSimilarResultsNameAndArgsSchema(t *testing.T) {
	tool := &FindSimilarResults{}

	if got := tool.Name(); got != "exa_find_similar_results_json" {
		t.Fatalf("name: got %q", got)
	}
	want := "A wrapper around Exa Find Similar. " +
		"Input should be an Exa-optimized query. " +
		"Output is a JSON array of the query results"
	if got := tool.Description(); got != want {
		t.Fatalf("description: got %q want %q", got, want)
	}

	argsSchema := tool.ArgsSchema()
	required, ok := argsSchema["required"].([]string)
	if !ok || len(required) != 1 || required[0] != "url" {
		t.Fatalf("required: got %v", argsSchema["required"])
	}
	props, ok := argsSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties: got %T", argsSchema["properties"])
	}
	if _, ok := props["url"]; !ok {
		t.Fatalf("properties missing url: %v", props)
	}
	if _, ok := props["category"]; !ok {
		t.Fatalf("properties missing category: %v", props)
	}
	if _, ok := props["query"]; ok {
		t.Fatalf("find similar schema must not advertise query: %v", props)
	}
}

// Python parity: _run's num_results defaults to 10 and unset options are
// dropped from the request body entirely (exa_py omits None options).
func TestSearchResultsInvokeSendsDefaults(t *testing.T) {
	server := newExaServer(t)
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	result, err := tool.Invoke(t.Context(), map[string]any{"query": "what is the weather in SF"})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if server.lastPath != "/search" {
		t.Fatalf("path: got %q", server.lastPath)
	}
	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "query", "numResults")
	if body["numResults"] != float64(10) {
		t.Fatalf("numResults default: got %v want 10", body["numResults"])
	}
	if result.Content != "[]" {
		t.Fatalf("empty content: got %q", result.Content)
	}
}

func TestSearchResultsInvokeMapsResults(t *testing.T) {
	server := newExaServer(t)
	server.setResults(
		Result{
			Title:           "San Francisco, CA Weather Conditionsstar_ratehome",
			URL:             "https://www.wunderground.com/weather/37.8,-122.4",
			PublishedDate:   "2023-02-23T01:17:06.594Z",
			Author:          "wunderground",
			Score:           0.1843988299369812,
			Text:            "The time period when the sun is no more than 6 degrees below the horizon.",
			Highlights:      []string{"Civil twilight", "Sunset"},
			ID:              "https://www.wunderground.com/weather/37.8,-122.4",
			Summary:         "Weather conditions for San Francisco.",
			HighlightScores: []float64{0.91, 0.84},
		},
		Result{URL: "https://example.com/bare", Title: "Bare result"},
	)
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	result, err := tool.Invoke(t.Context(), map[string]any{"query": "weather"})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	var decoded []Result
	if err := json.Unmarshal([]byte(result.Content), &decoded); err != nil {
		t.Fatalf("content is not a JSON array of results: %v (%q)", err, result.Content)
	}
	if len(decoded) != 2 {
		t.Fatalf("results: got %d want 2", len(decoded))
	}
	first := decoded[0]
	if first.Title != "San Francisco, CA Weather Conditionsstar_ratehome" ||
		first.URL != "https://www.wunderground.com/weather/37.8,-122.4" ||
		first.PublishedDate != "2023-02-23T01:17:06.594Z" ||
		first.Author != "wunderground" || first.Score != 0.1843988299369812 ||
		first.Text != "The time period when the sun is no more than 6 degrees below the horizon." ||
		first.ID != "https://www.wunderground.com/weather/37.8,-122.4" ||
		first.Summary != "Weather conditions for San Francisco." {
		t.Fatalf("first result: got %+v", first)
	}
	if len(first.Highlights) != 2 || first.Highlights[0] != "Civil twilight" {
		t.Fatalf("highlights: got %v", first.Highlights)
	}
	if len(first.HighlightScores) != 2 || first.HighlightScores[0] != 0.91 {
		t.Fatalf("highlightScores: got %v", first.HighlightScores)
	}

	artifact, ok := result.Artifact.([]Result)
	if !ok {
		t.Fatalf("artifact type: got %T", result.Artifact)
	}
	if len(artifact) != 2 || artifact[1].URL != "https://example.com/bare" {
		t.Fatalf("artifact: got %+v", artifact)
	}
}

// Python parity: _run accepts the shared search options per invocation; the
// wire body only contains explicitly set pieces (contents nests
// text.maxCharacters and highlights.{query,numSentences}).
func TestSearchResultsInvokeInputOverrides(t *testing.T) {
	server := newExaServer(t)
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	_, err := tool.Invoke(t.Context(), map[string]any{
		"query":              "exa optimized query",
		"numResults":         float64(3),
		"livecrawl":          "always",
		"startPublishedDate": "2024-01-01",
		"endPublishedDate":   "2024-12-31",
		"includeDomains":     []any{"example.com"},
		"excludeDomains":     []any{"spam.example"},
		"contents": map[string]any{
			"text":       map[string]any{"maxCharacters": 1000},
			"highlights": map[string]any{"query": "exa", "numSentences": 3},
		},
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	body := server.lastBodyMap(t)
	assertExactKeys(t, body,
		"query", "numResults", "livecrawl", "startPublishedDate", "endPublishedDate",
		"includeDomains", "excludeDomains", "contents",
	)
	if body["numResults"] != float64(3) {
		t.Fatalf("numResults: got %v", body["numResults"])
	}
	if body["livecrawl"] != "always" {
		t.Fatalf("livecrawl: got %v", body["livecrawl"])
	}
	contents := subObject(t, body, "contents")
	text := subObject(t, contents, "text")
	if text["maxCharacters"] != float64(1000) {
		t.Fatalf("maxCharacters: got %v", text["maxCharacters"])
	}
	highlights := subObject(t, contents, "highlights")
	if highlights["query"] != "exa" || highlights["numSentences"] != float64(3) {
		t.Fatalf("highlights: got %v", highlights)
	}
}

// Tool-level struct options are the defaults; only the ones set appear on the
// wire (livecrawl stays a three-valued string, never a bool — audit M1).
func TestSearchResultsInvokeStructOptions(t *testing.T) {
	server := newExaServer(t)
	livecrawl := "never"
	tool := &SearchResults{
		Client:             &Client{APIKey: "key", BaseURL: server.URL},
		NumResults:         7,
		LiveCrawl:          &livecrawl,
		StartPublishedDate: "2024-02-02",
		IncludeDomains:     []string{"a.com"},
		Text:               &TextOptions{MaxCharacters: 250},
	}

	_, err := tool.Invoke(t.Context(), map[string]any{"query": "q"})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "query", "numResults", "livecrawl",
		"startPublishedDate", "includeDomains", "contents")
	if body["numResults"] != float64(7) || body["livecrawl"] != "never" {
		t.Fatalf("numResults/livecrawl: got %v / %v", body["numResults"], body["livecrawl"])
	}
	contents := subObject(t, body, "contents")
	assertExactKeys(t, contents, "text")
	text := subObject(t, contents, "text")
	if text["maxCharacters"] != float64(250) {
		t.Fatalf("maxCharacters: got %v", text["maxCharacters"])
	}
}

// Input contents merge over the struct defaults at the text/highlights level.
func TestSearchResultsInvokeContentsMerge(t *testing.T) {
	server := newExaServer(t)
	tool := &SearchResults{
		Client:     &Client{APIKey: "key", BaseURL: server.URL},
		Highlights: &HighlightOptions{NumSentences: 2},
	}

	_, err := tool.Invoke(t.Context(), map[string]any{
		"query": "q",
		"contents": map[string]any{
			"text": map[string]any{"maxCharacters": json.Number("42")},
		},
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	contents := subObject(t, server.lastBodyMap(t), "contents")
	assertExactKeys(t, contents, "text", "highlights")
	text := subObject(t, contents, "text")
	if text["maxCharacters"] != float64(42) {
		t.Fatalf("maxCharacters: got %v", text["maxCharacters"])
	}
	highlights := subObject(t, contents, "highlights")
	if _, ok := highlights["numSentences"]; !ok {
		t.Fatalf("struct highlights must survive the merge: %v", highlights)
	}
}

// An explicitly set contents object without text/highlights parts omits the
// contents key entirely, matching exa_py dropping empty ContentsRequest.
func TestSearchResultsInvokeEmptyContentsOmitted(t *testing.T) {
	server := newExaServer(t)
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	_, err := tool.Invoke(t.Context(), map[string]any{
		"query":    "q",
		"contents": map[string]any{},
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "query", "numResults")
}

func TestSearchResultsInvokeMissingQuery(t *testing.T) {
	server := newExaServer(t)
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	if _, err := tool.Invoke(t.Context(), map[string]any{}); err == nil ||
		!strings.Contains(err.Error(), "query") {
		t.Fatalf("missing query must error mentioning query: %v", err)
	}
	if _, err := tool.Invoke(t.Context(), map[string]any{"query": 42}); err == nil ||
		!strings.Contains(err.Error(), "query") {
		t.Fatalf("non-string query must error mentioning query: %v", err)
	}
	if server.requestCount() != 0 {
		t.Fatalf("no request may be sent, sent %d", server.requestCount())
	}
}

func TestSearchResultsInvokeInvalidLiveCrawl(t *testing.T) {
	server := newExaServer(t)
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	_, err := tool.Invoke(t.Context(), map[string]any{"query": "q", "livecrawl": "sometimes"})
	if err == nil || !strings.Contains(err.Error(), "livecrawl") {
		t.Fatalf("invalid livecrawl must error mentioning livecrawl: %v", err)
	}
	if server.requestCount() != 0 {
		t.Fatalf("no request may be sent, sent %d", server.requestCount())
	}
}

func TestSearchResultsInvokeInvalidTypes(t *testing.T) {
	server := newExaServer(t)
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	cases := []map[string]any{
		{"query": "q", "numResults": "ten"},
		{"query": "q", "startPublishedDate": 123},
		{"query": "q", "endPublishedDate": 123},
		{"query": "q", "includeDomains": "example.com"},
		{"query": "q", "includeDomains": []any{"example.com", 5}},
		{"query": "q", "excludeDomains": 42},
		{"query": "q", "livecrawl": true},
		{"query": "q", "contents": "text"},
		{"query": "q", "contents": map[string]any{"text": "yes"}},
		{"query": "q", "contents": map[string]any{"text": map[string]any{"maxCharacters": "many"}}},
		{"query": "q", "contents": map[string]any{"highlights": "yes"}},
		{"query": "q", "contents": map[string]any{"highlights": map[string]any{"query": 5}}},
		{"query": "q", "contents": map[string]any{"highlights": map[string]any{"numSentences": "two"}}},
		{"query": ""},
	}
	for _, input := range cases {
		if _, err := tool.Invoke(t.Context(), input); err == nil {
			t.Fatalf("input %v must error", input)
		}
	}
	if server.requestCount() != 0 {
		t.Fatalf("no request may be sent, sent %d", server.requestCount())
	}
}

func TestSearchResultsInvokeAPIError(t *testing.T) {
	server := newExaServer(t)
	server.setError(http.StatusTooManyRequests, "slow down")
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	_, err := tool.Invoke(t.Context(), map[string]any{"query": "q"})
	if !errors.Is(err, lcerrors.ErrModelRateLimit) {
		t.Fatalf("want rate limit sentinel, got %v", err)
	}
}

// Python parity: ExaFindSimilarResults._run posts the required url plus the
// category filter to /findSimilar.
func TestFindSimilarResultsInvoke(t *testing.T) {
	server := newExaServer(t)
	server.setResults(Result{Title: "similar", URL: "https://example.com/similar"})
	tool := &FindSimilarResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	result, err := tool.Invoke(t.Context(), map[string]any{
		"url":      "https://example.com/source",
		"category": "personal portfolio",
	})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if server.lastPath != "/findSimilar" {
		t.Fatalf("path: got %q", server.lastPath)
	}
	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "url", "numResults", "category")
	if body["numResults"] != float64(10) {
		t.Fatalf("numResults default: got %v", body["numResults"])
	}
	if !strings.Contains(result.Content, "https://example.com/similar") {
		t.Fatalf("content: got %q", result.Content)
	}
}

func TestFindSimilarResultsInvokeStructCategory(t *testing.T) {
	server := newExaServer(t)
	tool := &FindSimilarResults{
		Client:   &Client{APIKey: "key", BaseURL: server.URL},
		Category: "news",
	}

	if _, err := tool.Invoke(t.Context(), map[string]any{"url": "https://example.com"}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "url", "numResults", "category")
	if body["category"] != "news" {
		t.Fatalf("category: got %v", body["category"])
	}
}

func TestFindSimilarResultsInvokeMissingURL(t *testing.T) {
	server := newExaServer(t)
	tool := &FindSimilarResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	if _, err := tool.Invoke(t.Context(), map[string]any{"query": "q"}); err == nil ||
		!strings.Contains(err.Error(), "url") {
		t.Fatalf("missing url must error mentioning url: %v", err)
	}
	if server.requestCount() != 0 {
		t.Fatalf("no request may be sent, sent %d", server.requestCount())
	}
}

func TestFindSimilarResultsInvokeInvalidCategory(t *testing.T) {
	server := newExaServer(t)
	tool := &FindSimilarResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	if _, err := tool.Invoke(t.Context(), map[string]any{
		"url":      "https://example.com",
		"category": 7,
	}); err == nil || !strings.Contains(err.Error(), "category") {
		t.Fatalf("non-string category must error mentioning category: %v", err)
	}
}

// A nil Client falls back to the default client so the zero-value tool still
// resolves EXA_API_KEY / the public base URL.
func TestSearchResultsInvokeWithNilClientUsesEnv(t *testing.T) {
	t.Setenv("EXA_API_KEY", "")
	tool := &SearchResults{}

	if _, err := tool.Invoke(t.Context(), map[string]any{"query": "q"}); err == nil ||
		!strings.Contains(err.Error(), "EXA_API_KEY") {
		t.Fatalf("nil client without a key must error: %v", err)
	}
}

// Go-constructed invocations may carry integer-typed numbers directly.
func TestSearchResultsInvokeInt64NumResults(t *testing.T) {
	server := newExaServer(t)
	tool := &SearchResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	if _, err := tool.Invoke(t.Context(), map[string]any{
		"query":      "q",
		"numResults": int64(6),
	}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	body := server.lastBodyMap(t)
	if body["numResults"] != float64(6) {
		t.Fatalf("numResults: got %v want 6", body["numResults"])
	}
}

func TestFindSimilarResultsInvokeInvalidOptions(t *testing.T) {
	server := newExaServer(t)
	tool := &FindSimilarResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	if _, err := tool.Invoke(t.Context(), map[string]any{
		"url":       "https://example.com",
		"livecrawl": true,
	}); err == nil {
		t.Fatal("invalid livecrawl must error")
	}
	if server.requestCount() != 0 {
		t.Fatalf("no request may be sent, sent %d", server.requestCount())
	}
}

func TestFindSimilarResultsInvokeAPIError(t *testing.T) {
	server := newExaServer(t)
	server.setError(http.StatusTooManyRequests, "slow down")
	tool := &FindSimilarResults{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	_, err := tool.Invoke(t.Context(), map[string]any{"url": "https://example.com"})
	if !errors.Is(err, lcerrors.ErrModelRateLimit) {
		t.Fatalf("want rate limit sentinel, got %v", err)
	}
}

func TestFindSimilarResultsInvokeWithNilClientUsesEnv(t *testing.T) {
	t.Setenv("EXA_API_KEY", "")
	tool := &FindSimilarResults{}

	if _, err := tool.Invoke(t.Context(), map[string]any{"url": "https://example.com"}); err == nil ||
		!strings.Contains(err.Error(), "EXA_API_KEY") {
		t.Fatalf("nil client without a key must error: %v", err)
	}
}
