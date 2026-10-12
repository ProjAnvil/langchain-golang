// Package exa provides search tools and a retriever backed by the Exa REST
// API (https://api.exa.ai).
//
// Python parity: langchain_exa 1.1.0 (ExaSearchResults / ExaFindSimilarResults
// tools and ExaSearchRetriever). Instead of wrapping the exa_py SDK, Go
// implements the REST endpoints directly (parity wave 3 design decision #6,
// docs/superpowers/specs/2026-10-12-parity-wave3-design.md): POST /search and
// POST /findSimilar with x-api-key auth and the API's camelCase wire fields.
package exa

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/projanvil/langchain-golang/core/lcerrors"
)

const (
	providerName   = "exa"
	defaultBaseURL = "https://api.exa.ai"
)

// defaultHTTPClient backs Client instances without an injected HTTPClient.
var defaultHTTPClient = &http.Client{}

// Client posts search and find-similar requests to the Exa REST API. The
// zero value resolves the API key from the EXA_API_KEY environment variable
// and the base URL to the public https://api.exa.ai endpoint (mirroring
// langchain_exa._utilities.initialize_client).
type Client struct {
	// APIKey authenticates every request via the x-api-key header. When
	// empty, the EXA_API_KEY environment variable is consulted.
	APIKey string
	// BaseURL targets the Exa API (default https://api.exa.ai).
	BaseURL string
	// HTTPClient is injected for tests; nil uses a shared default client.
	HTTPClient *http.Client
}

// Search runs a POST /search request and returns the API's results.
func (c *Client) Search(ctx context.Context, request searchRequest) ([]Result, error) {
	return c.post(ctx, "/search", request)
}

// FindSimilar runs a POST /findSimilar request and returns the API's results.
func (c *Client) FindSimilar(ctx context.Context, request searchRequest) ([]Result, error) {
	return c.post(ctx, "/findSimilar", request)
}

func (c *Client) post(ctx context.Context, path string, request searchRequest) ([]Result, error) {
	apiKey := c.resolveAPIKey()
	if apiKey == "" {
		return nil, fmt.Errorf(
			"exa API key is required: set Client.APIKey or the EXA_API_KEY environment variable",
		)
	}
	baseURL := strings.TrimSuffix(c.BaseURL, "/")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = defaultHTTPClient
	}

	payload, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("exa: encode %s request: %w", path, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("exa: build %s request: %w", path, err)
	}
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, lcerrors.WrapTransport(fmt.Errorf("exa: %s: %w", path, err))
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("exa: read %s response: %w", path, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, lcerrors.NewProviderError(providerName, path, resp.StatusCode, string(body), 0)
	}

	var parsed searchResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("exa: decode %s response: %w", path, err)
	}
	return parsed.Results, nil
}

func (c *Client) resolveAPIKey() string {
	return cmp.Or(c.APIKey, os.Getenv("EXA_API_KEY"))
}

// searchRequest is the wire body shared by /search and /findSimilar. Only
// explicitly configured options serialize (omitempty), mirroring how exa_py
// drops None options from the request; numResults is always present because
// both endpoints default it to 10.
type searchRequest struct {
	Query              string           `json:"query,omitempty"`
	URL                string           `json:"url,omitempty"`
	NumResults         int              `json:"numResults"`
	Category           string           `json:"category,omitempty"`
	LiveCrawl          *string          `json:"livecrawl,omitempty"`
	StartPublishedDate string           `json:"startPublishedDate,omitempty"`
	EndPublishedDate   string           `json:"endPublishedDate,omitempty"`
	IncludeDomains     []string         `json:"includeDomains,omitempty"`
	ExcludeDomains     []string         `json:"excludeDomains,omitempty"`
	Contents           *contentsRequest `json:"contents,omitempty"`
}

// contentsRequest selects which contents the API attaches to each result
// (exa_py ContentsRequest); it is omitted entirely when no part is set.
type contentsRequest struct {
	Text       *TextOptions      `json:"text,omitempty"`
	Highlights *HighlightOptions `json:"highlights,omitempty"`
}

// TextOptions configures the text contents attached to results (exa_py
// TextContentsOptions).
type TextOptions struct {
	MaxCharacters int `json:"maxCharacters,omitempty"`
}

// HighlightOptions configures the highlight contents attached to results
// (exa_py HighlightsContentsOptions).
type HighlightOptions struct {
	Query        string `json:"query,omitempty"`
	NumSentences int    `json:"numSentences,omitempty"`
}

// Result is one entry of the Exa API response. Field tags keep the API's
// camelCase wire names (langchain_exa SearchResult attributes).
type Result struct {
	Title           string    `json:"title"`
	URL             string    `json:"url"`
	PublishedDate   string    `json:"publishedDate"`
	Author          string    `json:"author"`
	Score           float64   `json:"score"`
	Text            string    `json:"text"`
	Highlights      []string  `json:"highlights"`
	ID              string    `json:"id"`
	Summary         string    `json:"summary"`
	HighlightScores []float64 `json:"highlightScores"`
}

// searchResponse is the top-level response envelope shared by both endpoints.
type searchResponse struct {
	Results          []Result `json:"results"`
	AutopromptString string   `json:"autopromptString"`
}
