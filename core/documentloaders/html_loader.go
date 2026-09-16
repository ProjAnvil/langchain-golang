package documentloaders

import (
	"context"
	"io"
	"strings"
	"unicode"

	"github.com/PuerkitoBio/goquery"
	"golang.org/x/net/html"

	"github.com/projanvil/langchain-golang/core/documents"
)

// htmlNonContentTags are removed before text extraction. This mirrors the
// reasonable-subset readability behavior used by Python's WebBaseLoader
// pipeline (langchain_community.document_loaders.web_base.WebBaseLoader wraps
// BeautifulSoup get_text; script/style/boilerplate tags never carry body
// content): script, style, noscript, template, iframe plus the readability
// boilerplate elements nav, aside, footer, header.
var htmlNonContentTags = []string{
	"script", "style", "noscript", "template", "iframe",
	"nav", "aside", "footer", "header",
}

// htmlBlockTags force a line break between their boundaries so extracted text
// keeps paragraph-level whitespace semantics (a reasonable stand-in for
// bs4/get_text separators in the Python WebBaseLoader).
var htmlBlockTags = map[string]bool{
	"address": true, "article": true, "aside": true, "blockquote": true,
	"body": true, "caption": true, "dd": true, "details": true,
	"dialog": true, "div": true, "dl": true, "dt": true, "fieldset": true,
	"figcaption": true, "figure": true, "footer": true, "form": true,
	"h1": true, "h2": true, "h3": true, "h4": true, "h5": true, "h6": true,
	"head": true, "header": true, "hgroup": true, "hr": true, "html": true,
	"legend": true, "li": true, "main": true, "menu": true, "nav": true,
	"ol": true, "p": true, "pre": true, "section": true, "summary": true,
	"table": true, "tbody": true, "td": true, "tfoot": true, "th": true,
	"thead": true, "tr": true, "ul": true,
}

// HTMLOption configures an HTMLLoader.
type HTMLOption func(*htmlOptions)

type htmlOptions struct {
	metadata map[string]any
}

// WithMetadata attaches additional metadata to the loaded document. Keys set
// here take precedence over values derived from the document itself (for
// example "title").
func WithMetadata(metadata map[string]any) HTMLOption {
	return func(opts *htmlOptions) {
		opts.metadata = metadata
	}
}

// HTMLLoader loads one HTML document from a reader and extracts readable body
// text. It is the Go counterpart of parsing with BeautifulSoup and taking the
// body text, as Python's WebBaseLoader does on fetched pages.
type HTMLLoader struct {
	source   io.Reader
	metadata map[string]any
}

// NewHTMLLoader creates a loader reading HTML from r.
func NewHTMLLoader(r io.Reader, opts ...HTMLOption) *HTMLLoader {
	options := htmlOptions{}
	for _, opt := range opts {
		opt(&options)
	}
	return &HTMLLoader{source: r, metadata: cloneMetadata(options.metadata)}
}

// LazyLoad parses the HTML and streams a single document. Parsing happens
// once up front; iterating the result is cheap.
func (l *HTMLLoader) LazyLoad(_ context.Context) (DocumentIterator, error) {
	doc, err := goquery.NewDocumentFromReader(l.source)
	if err != nil {
		return nil, err
	}
	return NewSliceIterator([]documents.Document{newHTMLDocument(doc, l.metadata)}), nil
}

// newHTMLDocument extracts body text and title metadata from a parsed
// document, honoring user metadata as an override.
func newHTMLDocument(doc *goquery.Document, metadata map[string]any) documents.Document {
	doc.Find(strings.Join(htmlNonContentTags, ",")).Remove()

	title := strings.TrimSpace(normalizeHTMLText(doc.Find("title").First().Text()))

	root := doc.Find("body")
	if root.Length() == 0 {
		root = doc.Selection
	}
	content := extractHTMLText(root)

	meta := cloneMetadata(metadata)
	if meta == nil {
		meta = map[string]any{}
	}
	if title != "" {
		if _, ok := meta["title"]; !ok {
			meta["title"] = title
		}
	}
	return documents.New(content, meta)
}

// extractHTMLText walks the DOM and returns block-structured text: text nodes
// are whitespace-normalized, block-level element boundaries become newlines,
// and empty lines are dropped.
func extractHTMLText(selection *goquery.Selection) string {
	var text strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		switch n.Type {
		case html.TextNode:
			text.WriteString(normalizeHTMLText(n.Data))
		case html.ElementNode:
			if n.Data == "br" {
				text.WriteString("\n")
				return
			}
			block := htmlBlockTags[n.Data]
			if block {
				text.WriteString("\n")
			}
			for child := n.FirstChild; child != nil; child = child.NextSibling {
				walk(child)
			}
			if block {
				text.WriteString("\n")
			}
		default:
			for child := n.FirstChild; child != nil; child = child.NextSibling {
				walk(child)
			}
		}
	}
	for _, node := range selection.Nodes {
		walk(node)
	}
	return joinHTMLLines(text.String())
}

// normalizeHTMLText collapses whitespace runs inside a text node to single
// spaces while preserving up to one leading and trailing space, so inline
// sibling text keeps its separation ("a <b>b</b>" stays "a b").
func normalizeHTMLText(s string) string {
	words := strings.Fields(s)
	if len(words) == 0 {
		if s == "" {
			return ""
		}
		return " "
	}
	var b strings.Builder
	if startsSpace(s) {
		b.WriteByte(' ')
	}
	b.WriteString(strings.Join(words, " "))
	if endsSpace(s) {
		b.WriteByte(' ')
	}
	return b.String()
}

func startsSpace(s string) bool {
	for _, r := range s {
		return unicode.IsSpace(r)
	}
	return false
}

func endsSpace(s string) bool {
	var last rune
	for _, r := range s {
		last = r
	}
	return last != 0 && unicode.IsSpace(last)
}

// joinHTMLLines trims each line, drops empty lines, and joins the rest with
// newlines, yielding stable paragraph-per-line output.
func joinHTMLLines(raw string) string {
	lines := strings.Split(raw, "\n")
	kept := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			kept = append(kept, line)
		}
	}
	return strings.Join(kept, "\n")
}

var _ LazyLoader = (*HTMLLoader)(nil)
