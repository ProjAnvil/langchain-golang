package documentloaders

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/PuerkitoBio/goquery"

	"github.com/projanvil/langchain-golang/core/documents"
)

// DefaultWebMaxBodyBytes caps how much of an HTTP response body the loader
// reads before erroring out (10 MiB).
const DefaultWebMaxBodyBytes int64 = 10 << 20

// DefaultWebUserAgent is the User-Agent used when the USER_AGENT environment
// variable is unset, matching Python's
// langchain_community.utils.user_agent.get_user_agent.
const DefaultWebUserAgent = "DefaultLangchainUserAgent"

// WebLoaderOption configures a WebLoader.
type WebLoaderOption func(*webOptions)

type webOptions struct {
	maxBodyBytes int64
}

// WithMaxBodyBytes overrides the response body size cap (default
// DefaultWebMaxBodyBytes).
func WithMaxBodyBytes(max int64) WebLoaderOption {
	return func(opts *webOptions) {
		opts.maxBodyBytes = max
	}
}

// WebLoader fetches HTML pages over HTTP and extracts readable body text. It
// is the Go counterpart of Python's
// langchain_community.document_loaders.web_base.WebBaseLoader: GET with the
// same default header template, then the shared HTML extraction used by
// HTMLLoader.
type WebLoader struct {
	client       *http.Client
	url          string
	maxBodyBytes int64
}

// NewWebLoader creates a web loader. A nil client falls back to a default
// *http.Client.
func NewWebLoader(client *http.Client, opts ...WebLoaderOption) *WebLoader {
	options := webOptions{maxBodyBytes: DefaultWebMaxBodyBytes}
	for _, opt := range opts {
		opt(&options)
	}
	if client == nil {
		client = &http.Client{}
	}
	return &WebLoader{
		client:       client,
		maxBodyBytes: options.maxBodyBytes,
	}
}

// ForURL returns a loader bound to rawURL for use with LazyLoad,
// documentloaders.Load, and LoadAndSplit. The returned loader shares the
// client and options of the receiver.
func (l *WebLoader) ForURL(rawURL string) *WebLoader {
	bound := *l
	bound.url = rawURL
	return &bound
}

// Load fetches rawURL and returns one document for the page. Metadata follows
// Python's _build_metadata: source, title, description, and language.
func (l *WebLoader) Load(ctx context.Context, rawURL string) ([]documents.Document, error) {
	return Load(ctx, l.ForURL(rawURL))
}

// LazyLoad fetches the bound URL (see ForURL) and streams one document.
func (l *WebLoader) LazyLoad(ctx context.Context) (DocumentIterator, error) {
	if l.url == "" {
		return nil, fmt.Errorf("web loader requires a URL; use ForURL or Load")
	}
	body, err := l.fetch(ctx, l.url)
	if err != nil {
		return nil, err
	}
	return l.parse(body)
}

func (l *WebLoader) parse(body []byte) (DocumentIterator, error) {
	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("web loader: parse %s: %w", l.url, err)
	}

	// Metadata mirrors WebBaseLoader._build_metadata, minus the Python
	// sentinel fallbacks ("No description found." and friends): absent
	// values are omitted instead. source and title (via newHTMLDocument)
	// complete the set.
	metadata := map[string]any{"source": l.url}
	if description, ok := doc.Find(`meta[name="description"]`).First().Attr("content"); ok && description != "" {
		metadata["description"] = description
	}
	if language, ok := doc.Find("html").First().Attr("lang"); ok && language != "" {
		metadata["language"] = language
	}

	return NewSliceIterator([]documents.Document{newHTMLDocument(doc, metadata)}), nil
}

func (l *WebLoader) fetch(ctx context.Context, rawURL string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, fmt.Errorf("web loader: build request: %w", err)
	}
	for key, value := range defaultWebHeaders() {
		req.Header.Set(key, value)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("web loader: GET %s: %w", rawURL, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, fmt.Errorf("web loader: GET %s: unexpected status %d", rawURL, resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, l.maxBodyBytes+1))
	if err != nil {
		return nil, fmt.Errorf("web loader: GET %s: read body: %w", rawURL, err)
	}
	if int64(len(body)) > l.maxBodyBytes {
		return nil, fmt.Errorf("web loader: GET %s: body exceeds %d bytes", rawURL, l.maxBodyBytes)
	}
	return body, nil
}

// defaultWebHeaders mirrors the default_header_template of Python's
// WebBaseLoader, including the USER_AGENT environment override from
// langchain_community.utils.user_agent.get_user_agent.
func defaultWebHeaders() map[string]string {
	return map[string]string{
		"User-Agent":                webUserAgent(),
		"Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
		"Accept-Language":           "en-US,en;q=0.5",
		"Referer":                   "https://www.google.com/",
		"DNT":                       "1",
		"Connection":                "keep-alive",
		"Upgrade-Insecure-Requests": "1",
	}
}

func webUserAgent() string {
	if env := os.Getenv("USER_AGENT"); env != "" {
		return env
	}
	return DefaultWebUserAgent
}

var _ LazyLoader = (*WebLoader)(nil)
