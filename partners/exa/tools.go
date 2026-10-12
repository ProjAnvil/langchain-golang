package exa

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// defaultNumResults mirrors the num_results default of 10 in
// langchain_exa.tools (tools.py:104, tools.py:186).
const defaultNumResults = 10

// SearchResults is a tool that queries the Exa /search endpoint. Struct
// fields are the tool-level defaults; per-invocation arguments (as advertised
// by ArgsSchema) override them. Unlike the Python tool — which swallows
// exceptions into repr strings — Invoke returns API failures as typed
// errors (parity wave 3 plan, Task 10).
type SearchResults struct {
	Client             *Client
	NumResults         int
	LiveCrawl          *string
	StartPublishedDate string
	EndPublishedDate   string
	IncludeDomains     []string
	ExcludeDomains     []string
	Text               *TextOptions
	Highlights         *HighlightOptions
}

// FindSimilarResults is a tool that queries the Exa /findSimilar endpoint.
// The first argument is a required url; category filters similar pages and
// is only offered by this tool (Python tools.py:186-220).
type FindSimilarResults struct {
	Client             *Client
	NumResults         int
	LiveCrawl          *string
	StartPublishedDate string
	EndPublishedDate   string
	IncludeDomains     []string
	ExcludeDomains     []string
	Text               *TextOptions
	Highlights         *HighlightOptions
	Category           string
}

// Compile-time guards: both tools satisfy the shared tool contract.
var (
	_ tools.Tool = (*SearchResults)(nil)
	_ tools.Tool = (*FindSimilarResults)(nil)
)

// Name returns "exa_search_results_json" (Python tools.py:86).
func (t *SearchResults) Name() string { return "exa_search_results_json" }

// Description returns the Python tool description verbatim (tools.py:87-91).
func (t *SearchResults) Description() string {
	return "Exa Search, one of the best web search APIs built for AI. " +
		"Input should be an Exa-optimized query. " +
		"Output is a JSON array of the query results"
}

// ArgsSchema mirrors the Python _run signature: a required query plus the
// shared search options (tools.py:101-120).
func (t *SearchResults) ArgsSchema() schema.Schema {
	return schema.Object(searchArgsProperties(), "query")
}

// Invoke executes POST /search with the invocation arguments layered over the
// tool-level defaults. Content is a JSON array of the results and Artifact
// carries the typed []Result.
func (t *SearchResults) Invoke(ctx context.Context, input map[string]any) (tools.Result, error) {
	query, err := requiredString(input, "query")
	if err != nil {
		return tools.Result{}, err
	}
	request, err := buildSearchRequest(input, t.defaults(), query, "")
	if err != nil {
		return tools.Result{}, err
	}
	results, err := t.client().Search(ctx, request)
	if err != nil {
		return tools.Result{}, err
	}
	return resultsAsToolResult(results)
}

// Name returns "exa_find_similar_results_json" (Python tools.py:167).
func (t *FindSimilarResults) Name() string { return "exa_find_similar_results_json" }

// Description returns the Python tool description verbatim (tools.py:168-172).
func (t *FindSimilarResults) Description() string {
	return "A wrapper around Exa Find Similar. " +
		"Input should be an Exa-optimized query. " +
		"Output is a JSON array of the query results"
}

// ArgsSchema mirrors the Python _run signature: a required url plus the
// shared options and the category filter (tools.py:183-202).
func (t *FindSimilarResults) ArgsSchema() schema.Schema {
	properties := searchArgsProperties()
	delete(properties, "query")
	properties["url"] = schema.String("The URL to find similar pages for.")
	properties["category"] = schema.String("Filter for similar pages by category.")
	return schema.Object(properties, "url")
}

// Invoke executes POST /findSimilar with the invocation arguments layered
// over the tool-level defaults.
func (t *FindSimilarResults) Invoke(ctx context.Context, input map[string]any) (tools.Result, error) {
	url, err := requiredString(input, "url")
	if err != nil {
		return tools.Result{}, err
	}
	request, err := buildSearchRequest(input, t.defaults(), "", url)
	if err != nil {
		return tools.Result{}, err
	}
	if value, ok := input["category"]; ok && value != nil {
		category, ok := value.(string)
		if !ok {
			return tools.Result{}, fmt.Errorf("exa: category must be a string")
		}
		request.Category = category
	}
	results, err := t.client().FindSimilar(ctx, request)
	if err != nil {
		return tools.Result{}, err
	}
	return resultsAsToolResult(results)
}

func (t *SearchResults) client() *Client {
	if t.Client == nil {
		return &Client{}
	}
	return t.Client
}

func (t *FindSimilarResults) client() *Client {
	if t.Client == nil {
		return &Client{}
	}
	return t.Client
}

// searchDefaults aggregates the tool-level options shared by both tools so
// buildSearchRequest can layer invocation input over them.
type searchDefaults struct {
	numResults         int
	liveCrawl          *string
	startPublishedDate string
	endPublishedDate   string
	includeDomains     []string
	excludeDomains     []string
	text               *TextOptions
	highlights         *HighlightOptions
	category           string
}

func (t *SearchResults) defaults() searchDefaults {
	return searchDefaults{
		numResults:         t.NumResults,
		liveCrawl:          t.LiveCrawl,
		startPublishedDate: t.StartPublishedDate,
		endPublishedDate:   t.EndPublishedDate,
		includeDomains:     t.IncludeDomains,
		excludeDomains:     t.ExcludeDomains,
		text:               t.Text,
		highlights:         t.Highlights,
	}
}

func (t *FindSimilarResults) defaults() searchDefaults {
	return searchDefaults{
		numResults:         t.NumResults,
		liveCrawl:          t.LiveCrawl,
		startPublishedDate: t.StartPublishedDate,
		endPublishedDate:   t.EndPublishedDate,
		includeDomains:     t.IncludeDomains,
		excludeDomains:     t.ExcludeDomains,
		text:               t.Text,
		highlights:         t.Highlights,
		category:           t.Category,
	}
}

// buildSearchRequest layers the invocation input over the tool defaults.
// Only explicitly set pieces reach the wire: unset options keep their
// omitempty drop, and an empty contents selection omits the contents key
// entirely (exa_py sends no ContentsRequest when no part is configured).
func buildSearchRequest(
	input map[string]any,
	defaults searchDefaults,
	query string,
	url string,
) (searchRequest, error) {
	request := searchRequest{
		Query:              query,
		URL:                url,
		NumResults:         defaults.numResults,
		LiveCrawl:          defaults.liveCrawl,
		StartPublishedDate: defaults.startPublishedDate,
		EndPublishedDate:   defaults.endPublishedDate,
		IncludeDomains:     defaults.includeDomains,
		ExcludeDomains:     defaults.excludeDomains,
		Category:           defaults.category,
		Contents:           &contentsRequest{Text: defaults.text, Highlights: defaults.highlights},
	}

	if value, ok := input["numResults"]; ok && value != nil {
		number, ok := asInt(value)
		if !ok {
			return searchRequest{}, fmt.Errorf("exa: numResults must be an integer")
		}
		request.NumResults = number
	}
	if request.NumResults == 0 {
		request.NumResults = defaultNumResults
	}
	if value, ok := input["livecrawl"]; ok && value != nil {
		livecrawl, ok := value.(string)
		if !ok {
			return searchRequest{}, fmt.Errorf(`exa: livecrawl must be one of "always", "fallback", "never"`)
		}
		if err := validateLiveCrawl(livecrawl); err != nil {
			return searchRequest{}, err
		}
		request.LiveCrawl = &livecrawl
	}
	if err := applyString(input, "startPublishedDate", &request.StartPublishedDate); err != nil {
		return searchRequest{}, err
	}
	if err := applyString(input, "endPublishedDate", &request.EndPublishedDate); err != nil {
		return searchRequest{}, err
	}
	if err := applyStringSlice(input, "includeDomains", &request.IncludeDomains); err != nil {
		return searchRequest{}, err
	}
	if err := applyStringSlice(input, "excludeDomains", &request.ExcludeDomains); err != nil {
		return searchRequest{}, err
	}
	if value, ok := input["contents"]; ok && value != nil {
		contents, err := contentsFromInput(value)
		if err != nil {
			return searchRequest{}, err
		}
		if contents.Text != nil {
			request.Contents.Text = contents.Text
		}
		if contents.Highlights != nil {
			request.Contents.Highlights = contents.Highlights
		}
	}
	if request.Contents.Text == nil && request.Contents.Highlights == nil {
		request.Contents = nil
	}
	return request, nil
}

// validateLiveCrawl enforces Python's Literal["always", "fallback", "never"]
// input type (tools.py:117). The field is a *string on purpose — never a
// bool — because "fallback" is a third state (parity audit M1).
func validateLiveCrawl(value string) error {
	switch value {
	case "always", "fallback", "never":
		return nil
	default:
		return fmt.Errorf(`exa: livecrawl must be one of "always", "fallback", "never", got %q`, value)
	}
}

// contentsFromInput parses an invocation "contents" object. Missing parts
// stay nil so they never reach the wire.
func contentsFromInput(value any) (*contentsRequest, error) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("exa: contents must be an object")
	}
	out := &contentsRequest{}
	if text, ok := object["text"]; ok && text != nil {
		options, err := textOptionsFromInput(text)
		if err != nil {
			return nil, err
		}
		out.Text = options
	}
	if highlights, ok := object["highlights"]; ok && highlights != nil {
		options, err := highlightOptionsFromInput(highlights)
		if err != nil {
			return nil, err
		}
		out.Highlights = options
	}
	return out, nil
}

func textOptionsFromInput(value any) (*TextOptions, error) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("exa: contents.text must be an object")
	}
	options := &TextOptions{}
	if maxCharacters, ok := object["maxCharacters"]; ok && maxCharacters != nil {
		number, ok := asInt(maxCharacters)
		if !ok {
			return nil, fmt.Errorf("exa: contents.text.maxCharacters must be an integer")
		}
		options.MaxCharacters = number
	}
	return options, nil
}

func highlightOptionsFromInput(value any) (*HighlightOptions, error) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("exa: contents.highlights must be an object")
	}
	options := &HighlightOptions{}
	if query, ok := object["query"]; ok && query != nil {
		text, ok := query.(string)
		if !ok {
			return nil, fmt.Errorf("exa: contents.highlights.query must be a string")
		}
		options.Query = text
	}
	if numSentences, ok := object["numSentences"]; ok && numSentences != nil {
		number, ok := asInt(numSentences)
		if !ok {
			return nil, fmt.Errorf("exa: contents.highlights.numSentences must be an integer")
		}
		options.NumSentences = number
	}
	return options, nil
}

func requiredString(input map[string]any, key string) (string, error) {
	value, ok := input[key]
	if !ok || value == nil {
		return "", fmt.Errorf("exa: %s is required", key)
	}
	text, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("exa: %s must be a string", key)
	}
	if text == "" {
		return "", fmt.Errorf("exa: %s must not be empty", key)
	}
	return text, nil
}

func applyString(input map[string]any, key string, target *string) error {
	value, ok := input[key]
	if !ok || value == nil {
		return nil
	}
	text, ok := value.(string)
	if !ok {
		return fmt.Errorf("exa: %s must be a string", key)
	}
	*target = text
	return nil
}

func applyStringSlice(input map[string]any, key string, target *[]string) error {
	value, ok := input[key]
	if !ok || value == nil {
		return nil
	}
	items, ok := value.([]any)
	if !ok {
		return fmt.Errorf("exa: %s must be an array of strings", key)
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		text, ok := item.(string)
		if !ok {
			return fmt.Errorf("exa: %s must be an array of strings", key)
		}
		out = append(out, text)
	}
	*target = out
	return nil
}

// asInt coerces JSON-decoded numbers (float64, json.Number) and Go integers.
func asInt(value any) (int, bool) {
	switch number := value.(type) {
	case int:
		return number, true
	case int64:
		return int(number), true
	case float64:
		if number == float64(int(number)) {
			return int(number), true
		}
	case json.Number:
		if parsed, err := number.Int64(); err == nil {
			return int(parsed), true
		}
	}
	return 0, false
}

// resultsAsToolResult renders the API results for the tool contract: Content
// is the JSON array of results (matching the "Output is a JSON array of the
// query results" description) and Artifact carries the typed slice.
func resultsAsToolResult(results []Result) (tools.Result, error) {
	if results == nil {
		results = []Result{}
	}
	encoded, err := json.Marshal(results)
	if err != nil {
		return tools.Result{}, fmt.Errorf("exa: encode results: %w", err)
	}
	return tools.Result{Content: string(encoded), Artifact: results}, nil
}

// searchArgsProperties returns the shared tool argument schema properties
// (Python tools.py argument docs).
func searchArgsProperties() map[string]schema.Schema {
	return map[string]schema.Schema{
		"query": schema.String("The search query."),
		"numResults": schema.Integer(
			"The number of search results to return (1 to 100). Default: 10",
		),
		"livecrawl": {
			"type": "string",
			"enum": []string{"always", "fallback", "never"},
			"description": `Option to crawl live webpages if content is not in the index. ` +
				`Options: "always", "fallback", "never"`,
		},
		"startPublishedDate": schema.String(
			"The start date for when the document was published (in YYYY-MM-DD format).",
		),
		"endPublishedDate": schema.String(
			"The end date for when the document was published (in YYYY-MM-DD format).",
		),
		"includeDomains": stringArraySchema("A list of domains to include in the search."),
		"excludeDomains": stringArraySchema("A list of domains to exclude from the search."),
		"contents": schema.Object(map[string]schema.Schema{
			"text": schema.Object(map[string]schema.Schema{
				"maxCharacters": schema.Integer(
					"Maximum number of characters of text to attach to each result.",
				),
			}),
			"highlights": schema.Object(map[string]schema.Schema{
				"query":        schema.String("The query to focus highlights on."),
				"numSentences": schema.Integer("Number of sentences per highlight."),
			}),
		}),
	}
}

func stringArraySchema(description string) schema.Schema {
	return schema.Schema{
		"type":        "array",
		"items":       schema.String(""),
		"description": description,
	}
}
