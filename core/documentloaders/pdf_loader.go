package documentloaders

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/ledongthuc/pdf"

	"github.com/projanvil/langchain-golang/core/documents"
)

// PDFLoader loads a PDF from a reader and yields one document per page. It
// is the Go counterpart of Python's PyPDFLoader
// (langchain_community.document_loaders.pdf.PyPDFParser): each page becomes
// a Document whose metadata records the 0-based "page" number and
// "total_pages", and whose content is the page text trimmed of surrounding
// whitespace. Encrypted or malformed PDFs fail the load with an error.
type PDFLoader struct {
	source io.Reader
}

// NewPDFLoader creates a loader reading PDF bytes from r.
func NewPDFLoader(r io.Reader) *PDFLoader {
	return &PDFLoader{source: r}
}

// LazyLoad parses the PDF and streams one document per page. Pages with no
// extractable text still yield a document with empty content, matching
// PyPDFLoader's per-page enumeration.
func (l *PDFLoader) LazyLoad(_ context.Context) (DocumentIterator, error) {
	data, err := io.ReadAll(l.source)
	if err != nil {
		return nil, err
	}
	reader, err := pdf.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, fmt.Errorf("pdf loader: %w", err)
	}

	total := reader.NumPage()
	fonts := make(map[string]*pdf.Font)
	docs := make([]documents.Document, 0, total)
	for number := 1; number <= total; number++ {
		page := reader.Page(number)
		// Share decoded font encodings across pages, the same cache
		// trick ledongthuc/pdf's own Reader.GetPlainText uses.
		for _, name := range page.Fonts() {
			if _, ok := fonts[name]; !ok {
				font := page.Font(name)
				fonts[name] = &font
			}
		}
		text, err := page.GetPlainText(fonts)
		if err != nil {
			return nil, fmt.Errorf("pdf loader: page %d: %w", number-1, err)
		}
		docs = append(docs, documents.New(strings.TrimSpace(text), map[string]any{
			"page":        number - 1,
			"total_pages": total,
		}))
	}
	return NewSliceIterator(docs), nil
}

var _ LazyLoader = (*PDFLoader)(nil)
