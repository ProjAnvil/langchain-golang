package exa

import (
	"context"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

// SearchRetriever retrieves documents from the Exa API. The zero value
// behaves like Python's ExaSearchRetriever defaults: k=10 results with text
// contents requested. Setting URL switches the retriever to /findSimilar
// mode (parity wave 3 plan, Task 10); category applies in that mode.
type SearchRetriever struct {
	Client             *Client
	K                  int
	LiveCrawl          *string
	StartPublishedDate string
	EndPublishedDate   string
	IncludeDomains     []string
	ExcludeDomains     []string
	Text               *TextOptions
	Highlights         *HighlightOptions
	Category           string
	URL                string
}

// Compile-time guard: SearchRetriever satisfies the core retriever contract.
var _ retrievers.Retriever = (*SearchRetriever)(nil)

// GetRelevantDocuments runs a /search request for the query (or /findSimilar
// for the configured URL) and maps the API results to documents.
func (r *SearchRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	client := r.Client
	if client == nil {
		client = &Client{}
	}

	// Python's retriever defaults text_contents_options=True, so a nil Text
	// still requests text contents with the API defaults (retrievers.py:59).
	text := r.Text
	if text == nil {
		text = &TextOptions{}
	}
	request := searchRequest{
		NumResults:         r.K,
		LiveCrawl:          r.LiveCrawl,
		StartPublishedDate: r.StartPublishedDate,
		EndPublishedDate:   r.EndPublishedDate,
		IncludeDomains:     r.IncludeDomains,
		ExcludeDomains:     r.ExcludeDomains,
		Category:           r.Category,
		Contents:           &contentsRequest{Text: text, Highlights: r.Highlights},
	}
	if request.NumResults == 0 {
		request.NumResults = defaultNumResults
	}
	if r.URL != "" {
		request.URL = r.URL
	} else {
		request.Query = query
	}

	var results []Result
	var err error
	if r.URL != "" {
		results, err = client.FindSimilar(ctx, request)
	} else {
		results, err = client.Search(ctx, request)
	}
	if err != nil {
		return nil, err
	}

	docs := make([]documents.Document, 0, len(results))
	for _, result := range results {
		docs = append(docs, resultToDocument(result))
	}
	return docs, nil
}

// resultToDocument maps one API result to a document. PageContent is the
// result text, falling back to the title when the API returned none.
// Metadata mirrors langchain_exa retrievers._get_metadata with underscore
// keys; fields the API omitted are left out of the map.
func resultToDocument(result Result) documents.Document {
	content := result.Text
	if content == "" {
		content = result.Title
	}
	metadata := make(map[string]any)
	if result.Title != "" {
		metadata["title"] = result.Title
	}
	if result.URL != "" {
		metadata["url"] = result.URL
	}
	if result.PublishedDate != "" {
		metadata["published_date"] = result.PublishedDate
	}
	if result.Author != "" {
		metadata["author"] = result.Author
	}
	if result.Score != 0 {
		metadata["score"] = result.Score
	}
	if len(result.Highlights) > 0 {
		metadata["highlights"] = result.Highlights
	}
	if result.ID != "" {
		metadata["id"] = result.ID
	}
	if result.Summary != "" {
		metadata["summary"] = result.Summary
	}
	if len(result.HighlightScores) > 0 {
		metadata["highlight_scores"] = result.HighlightScores
	}
	return documents.New(content, metadata)
}
