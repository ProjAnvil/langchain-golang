package textsplitters

import (
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

// SentenceTokenizer splits text into sentence strings for a language.
type SentenceTokenizer func(text string, language string) ([]string, error)

// SentenceSpan identifies a sentence range in the original text.
type SentenceSpan struct {
	Start int
	End   int
}

// SentenceSpanTokenizer returns sentence spans in the original text.
type SentenceSpanTokenizer func(text string, language string) ([]SentenceSpan, error)

// SentenceTextSplitter splits text on externally supplied sentence boundaries.
// It mirrors the NLTK/spaCy splitter shape without importing those optional
// libraries into this package.
type SentenceTextSplitter struct {
	TextSplitter
	separator     string
	language      string
	tokenizer     SentenceTokenizer
	spanTokenizer SentenceSpanTokenizer
}

// NewSentence creates a sentence splitter from a sentence tokenizer.
func NewSentence(tokenizer SentenceTokenizer, separator string, language string, cfg Config) (*SentenceTextSplitter, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("sentence tokenizer is required")
	}
	if separator == "" {
		separator = "\n\n"
	}
	if language == "" {
		language = "english"
	}
	if cfg.ChunkSize == 0 && cfg.ChunkOverlap == 0 && cfg.LengthFunc == nil {
		cfg.StripWhitespace = true
	}
	normalized, err := cfg.normalize()
	if err != nil {
		return nil, err
	}
	return &SentenceTextSplitter{
		TextSplitter: TextSplitter{cfg: normalized},
		separator:    separator,
		language:     language,
		tokenizer:    tokenizer,
	}, nil
}

// NewSentenceSpans creates a sentence splitter from original-text spans. Span
// mode preserves inter-sentence whitespace and therefore requires an empty
// merge separator, matching Python's NLTK span-tokenize constraint.
func NewSentenceSpans(tokenizer SentenceSpanTokenizer, language string, cfg Config) (*SentenceTextSplitter, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("sentence span tokenizer is required")
	}
	if language == "" {
		language = "english"
	}
	if cfg.ChunkSize == 0 && cfg.ChunkOverlap == 0 && cfg.LengthFunc == nil {
		cfg.StripWhitespace = true
	}
	normalized, err := cfg.normalize()
	if err != nil {
		return nil, err
	}
	return &SentenceTextSplitter{
		TextSplitter:  TextSplitter{cfg: normalized},
		language:      language,
		spanTokenizer: tokenizer,
	}, nil
}

// SplitText splits text into sentence-based chunks.
func (s *SentenceTextSplitter) SplitText(text string) ([]string, error) {
	var splits []string
	var err error
	if s.spanTokenizer != nil {
		splits, err = s.splitBySpans(text)
	} else {
		splits, err = s.tokenizer(text, s.language)
	}
	if err != nil {
		return nil, err
	}
	return s.mergeSplits(splits, s.separator), nil
}

// CreateDocuments splits texts into sentence-based documents.
func (s *SentenceTextSplitter) CreateDocuments(texts []string, metadatas []map[string]any) ([]documents.Document, error) {
	docs := make([]documents.Document, 0)
	for i, text := range texts {
		metadata := map[string]any(nil)
		if i < len(metadatas) {
			metadata = cloneMetadata(metadatas[i])
		}
		chunks, err := s.SplitText(text)
		if err != nil {
			return nil, err
		}
		index := 0
		previousChunkLen := 0
		for _, chunk := range chunks {
			chunkMetadata := cloneMetadata(metadata)
			if s.cfg.AddStartIndex {
				offset := index + previousChunkLen - s.cfg.ChunkOverlap
				if offset < 0 {
					offset = 0
				}
				found := strings.Index(text[offset:], chunk)
				if found >= 0 {
					index = offset + found
				} else {
					index = -1
				}
				chunkMetadata["start_index"] = index
				previousChunkLen = len(chunk)
			}
			docs = append(docs, documents.New(chunk, chunkMetadata))
		}
	}
	return docs, nil
}

// SplitDocuments splits existing documents and preserves metadata.
func (s *SentenceTextSplitter) SplitDocuments(docs []documents.Document) ([]documents.Document, error) {
	texts := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
		metadatas[i] = doc.Metadata
	}
	return s.CreateDocuments(texts, metadatas)
}

func (s *SentenceTextSplitter) splitBySpans(text string) ([]string, error) {
	spans, err := s.spanTokenizer(text, s.language)
	if err != nil {
		return nil, err
	}
	splits := make([]string, 0, len(spans))
	for i, span := range spans {
		if span.Start < 0 || span.End < span.Start || span.End > len(text) {
			return nil, fmt.Errorf("invalid sentence span [%d,%d) for text length %d", span.Start, span.End, len(text))
		}
		start := span.Start
		if i > 0 {
			start = spans[i-1].End
			if start > span.Start {
				return nil, fmt.Errorf("overlapping sentence spans at index %d", i)
			}
		}
		splits = append(splits, text[start:span.End])
	}
	return splits, nil
}
