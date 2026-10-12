package exa

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

// Compile-time guard: SearchRetriever satisfies the core retriever contract.
var _ retrievers.Retriever = (*SearchRetriever)(nil)

// Python parity: ExaSearchRetriever defaults k=10 and requests text contents
// by default (text_contents_options=True), posting to /search.
func TestSearchRetrieverSearchMode(t *testing.T) {
	server := newExaServer(t)
	retriever := &SearchRetriever{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	if _, err := retriever.GetRelevantDocuments(t.Context(), "what is the weather in SF"); err != nil {
		t.Fatalf("get documents: %v", err)
	}

	if server.lastPath != "/search" {
		t.Fatalf("path: got %q", server.lastPath)
	}
	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "query", "numResults", "contents")
	if body["query"] != "what is the weather in SF" {
		t.Fatalf("query: got %v", body["query"])
	}
	if body["numResults"] != float64(10) {
		t.Fatalf("numResults default: got %v want 10", body["numResults"])
	}
	contents := subObject(t, body, "contents")
	if _, ok := contents["text"]; !ok {
		t.Fatalf("text contents must be requested by default: %v", contents)
	}
	if _, ok := contents["highlights"]; ok {
		t.Fatalf("highlights must not be requested by default: %v", contents)
	}
}

func TestSearchRetrieverOptions(t *testing.T) {
	server := newExaServer(t)
	livecrawl := "fallback"
	retriever := &SearchRetriever{
		Client:             &Client{APIKey: "key", BaseURL: server.URL},
		K:                  3,
		LiveCrawl:          &livecrawl,
		StartPublishedDate: "2024-01-01",
		EndPublishedDate:   "2024-06-30",
		IncludeDomains:     []string{"example.com"},
		ExcludeDomains:     []string{"spam.example"},
		Text:               &TextOptions{MaxCharacters: 500},
		Highlights:         &HighlightOptions{Query: "weather", NumSentences: 2},
	}

	if _, err := retriever.GetRelevantDocuments(t.Context(), "weather"); err != nil {
		t.Fatalf("get documents: %v", err)
	}

	body := server.lastBodyMap(t)
	assertExactKeys(t, body,
		"query", "numResults", "livecrawl", "startPublishedDate", "endPublishedDate",
		"includeDomains", "excludeDomains", "contents",
	)
	if body["numResults"] != float64(3) {
		t.Fatalf("numResults: got %v", body["numResults"])
	}
	contents := subObject(t, body, "contents")
	text := subObject(t, contents, "text")
	if text["maxCharacters"] != float64(500) {
		t.Fatalf("maxCharacters: got %v", text["maxCharacters"])
	}
	highlights := subObject(t, contents, "highlights")
	if highlights["query"] != "weather" || highlights["numSentences"] != float64(2) {
		t.Fatalf("highlights: got %v", highlights)
	}
}

// Python parity (plan Task 10): a URL switches the retriever to
// /findSimilar mode.
func TestSearchRetrieverFindSimilarMode(t *testing.T) {
	server := newExaServer(t)
	retriever := &SearchRetriever{
		Client:   &Client{APIKey: "key", BaseURL: server.URL},
		K:        4,
		URL:      "https://example.com/source",
		Category: "news",
	}

	if _, err := retriever.GetRelevantDocuments(t.Context(), "ignored query"); err != nil {
		t.Fatalf("get documents: %v", err)
	}

	if server.lastPath != "/findSimilar" {
		t.Fatalf("path: got %q want /findSimilar", server.lastPath)
	}
	body := server.lastBodyMap(t)
	assertExactKeys(t, body, "url", "numResults", "category", "contents")
	if body["url"] != "https://example.com/source" {
		t.Fatalf("url: got %v", body["url"])
	}
	if body["numResults"] != float64(4) {
		t.Fatalf("numResults: got %v", body["numResults"])
	}
}

// Python parity: page_content is the result text and metadata mirrors
// retrievers._get_metadata with underscore keys; fields absent from the API
// response are omitted, and a result without text falls back to its title.
func TestSearchRetrieverDocumentMapping(t *testing.T) {
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
		Result{Title: "Title only", URL: "https://example.com/bare", Score: 0.42},
	)
	retriever := &SearchRetriever{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	docs, err := retriever.GetRelevantDocuments(t.Context(), "weather")
	if err != nil {
		t.Fatalf("get documents: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("documents: got %d want 2", len(docs))
	}

	full := docs[0]
	if full.PageContent != "The time period when the sun is no more than 6 degrees below the horizon." {
		t.Fatalf("page content: got %q", full.PageContent)
	}
	for key, want := range map[string]any{
		"title":            "San Francisco, CA Weather Conditionsstar_ratehome",
		"url":              "https://www.wunderground.com/weather/37.8,-122.4",
		"published_date":   "2023-02-23T01:17:06.594Z",
		"author":           "wunderground",
		"score":            0.1843988299369812,
		"id":               "https://www.wunderground.com/weather/37.8,-122.4",
		"summary":          "Weather conditions for San Francisco.",
		"highlights":       []string{"Civil twilight", "Sunset"},
		"highlight_scores": []float64{0.91, 0.84},
	} {
		if _, ok := full.Metadata[key]; !ok {
			t.Fatalf("metadata missing %q: %v", key, full.Metadata)
		}
		if key == "highlights" || key == "highlight_scores" || key == "score" {
			continue // element types asserted below
		}
		if got := full.Metadata[key]; got != want {
			t.Fatalf("metadata[%q]: got %v want %v", key, got, want)
		}
	}
	if got, ok := full.Metadata["highlights"].([]string); !ok || len(got) != 2 || got[0] != "Civil twilight" {
		t.Fatalf("metadata[highlights]: got %v (%T)", full.Metadata["highlights"], full.Metadata["highlights"])
	}
	if got, ok := full.Metadata["highlight_scores"].([]float64); !ok || len(got) != 2 || got[0] != 0.91 {
		t.Fatalf("metadata[highlight_scores]: got %v (%T)", full.Metadata["highlight_scores"], full.Metadata["highlight_scores"])
	}
	if got, ok := full.Metadata["score"].(float64); !ok || got != 0.1843988299369812 {
		t.Fatalf("metadata[score]: got %v (%T)", full.Metadata["score"], full.Metadata["score"])
	}
	if len(full.Metadata) != 9 {
		t.Fatalf("metadata keys: got %v", keysOf(full.Metadata))
	}

	// Sparse result: no text -> title fallback, optional metadata omitted.
	bare := docs[1]
	if bare.PageContent != "Title only" {
		t.Fatalf("bare page content must fall back to the title: got %q", bare.PageContent)
	}
	for _, key := range []string{"published_date", "author", "highlights", "highlight_scores", "summary"} {
		if _, ok := bare.Metadata[key]; ok {
			t.Fatalf("bare metadata must omit %q: %v", key, bare.Metadata)
		}
	}
	if bare.Metadata["url"] != "https://example.com/bare" || bare.Metadata["title"] != "Title only" {
		t.Fatalf("bare metadata: got %v", bare.Metadata)
	}
	if len(bare.Metadata) != 3 {
		t.Fatalf("bare metadata keys: got %v", keysOf(bare.Metadata))
	}
}

func TestSearchRetrieverEmptyResults(t *testing.T) {
	server := newExaServer(t)
	retriever := &SearchRetriever{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	docs, err := retriever.GetRelevantDocuments(t.Context(), "weather")
	if err != nil {
		t.Fatalf("get documents: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("documents: got %d want 0", len(docs))
	}
}

func TestSearchRetrieverAPIError(t *testing.T) {
	server := newExaServer(t)
	server.setError(http.StatusInternalServerError, "exa exploded")
	retriever := &SearchRetriever{Client: &Client{APIKey: "key", BaseURL: server.URL}}

	if _, err := retriever.GetRelevantDocuments(t.Context(), "weather"); !errors.Is(err, lcerrors.ErrModelServer) {
		t.Fatalf("want server sentinel, got %v", err)
	}
}

// A nil Client falls back to the default client (public base URL + env key),
// so the zero-value retriever without a key fails before any request.
func TestSearchRetrieverNilClientWithoutKey(t *testing.T) {
	t.Setenv("EXA_API_KEY", "")
	retriever := &SearchRetriever{}

	_, err := retriever.GetRelevantDocuments(t.Context(), "weather")
	if err == nil || !strings.Contains(err.Error(), "EXA_API_KEY") {
		t.Fatalf("nil client without a key must error: %v", err)
	}
}
