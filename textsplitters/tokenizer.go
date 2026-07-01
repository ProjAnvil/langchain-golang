package textsplitters

import (
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

// Tokenizer converts text to token strings and back. It is intentionally small
// so provider or model-specific tokenizers can be adapted without adding a hard
// dependency to this package.
type Tokenizer interface {
	Encode(text string) ([]string, error)
	Decode(tokens []string) (string, error)
}

// WhitespaceTokenizer is a deterministic tokenizer useful for tests and
// whitespace-delimited text.
type WhitespaceTokenizer struct{}

// Encode splits text into whitespace-delimited tokens.
func (WhitespaceTokenizer) Encode(text string) ([]string, error) {
	return strings.Fields(text), nil
}

// Decode joins tokens with a single space.
func (WhitespaceTokenizer) Decode(tokens []string) (string, error) {
	return strings.Join(tokens, " "), nil
}

// TokenTextSplitter splits text by tokenizer token count with token overlap.
type TokenTextSplitter struct {
	TextSplitter
	tokenizer Tokenizer
}

// TokenIDTokenizer converts text to integer token IDs and back.
type TokenIDTokenizer interface {
	EncodeIDs(text string) ([]int, error)
	DecodeIDs(tokens []int) (string, error)
}

// SplitTextOnTokens splits text by token IDs using a tokenizer adapter.
func SplitTextOnTokens(text string, tokenizer TokenIDTokenizer, tokensPerChunk int, chunkOverlap int) ([]string, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("token ID tokenizer is required")
	}
	if tokensPerChunk <= chunkOverlap {
		return nil, fmt.Errorf("tokens_per_chunk must be greater than chunk_overlap")
	}
	inputIDs, err := tokenizer.EncodeIDs(text)
	if err != nil {
		return nil, fmt.Errorf("encode token IDs: %w", err)
	}
	return splitTokenIDs(inputIDs, tokenizer.DecodeIDs, tokensPerChunk, chunkOverlap, false)
}

// TokenIDTextSplitter splits text using integer token IDs. This adapts
// tiktoken, sentence-transformers, and similar libraries without forcing a
// concrete tokenizer dependency.
type TokenIDTextSplitter struct {
	TextSplitter
	tokenizer TokenIDTokenizer
}

// NewToken creates a token-count based splitter.
func NewToken(tokenizer Tokenizer, cfg Config) (*TokenTextSplitter, error) {
	if tokenizer == nil {
		tokenizer = WhitespaceTokenizer{}
	}
	if cfg.ChunkSize == 0 && cfg.ChunkOverlap == 0 && cfg.LengthFunc == nil {
		cfg.StripWhitespace = true
	}
	normalized, err := cfg.normalize()
	if err != nil {
		return nil, err
	}
	return &TokenTextSplitter{
		TextSplitter: TextSplitter{cfg: normalized},
		tokenizer:    tokenizer,
	}, nil
}

// NewTokenIDs creates an integer-token splitter.
func NewTokenIDs(tokenizer TokenIDTokenizer, cfg Config) (*TokenIDTextSplitter, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("token ID tokenizer is required")
	}
	if cfg.ChunkSize == 0 && cfg.ChunkOverlap == 0 && cfg.LengthFunc == nil {
		cfg.StripWhitespace = true
	}
	normalized, err := cfg.normalize()
	if err != nil {
		return nil, err
	}
	return &TokenIDTextSplitter{
		TextSplitter: TextSplitter{cfg: normalized},
		tokenizer:    tokenizer,
	}, nil
}

// SplitText splits text into token-count chunks.
func (s *TokenTextSplitter) SplitText(text string) ([]string, error) {
	tokens, err := s.tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("encode tokens: %w", err)
	}
	if len(tokens) == 0 {
		return nil, nil
	}
	step := s.cfg.ChunkSize - s.cfg.ChunkOverlap
	if step <= 0 {
		step = s.cfg.ChunkSize
	}
	out := []string{}
	for start := 0; start < len(tokens); start += step {
		end := start + s.cfg.ChunkSize
		if end > len(tokens) {
			end = len(tokens)
		}
		chunk, err := s.tokenizer.Decode(tokens[start:end])
		if err != nil {
			return nil, fmt.Errorf("decode tokens: %w", err)
		}
		if s.cfg.StripWhitespace {
			chunk = strings.TrimSpace(chunk)
		}
		if chunk != "" {
			out = append(out, chunk)
		}
		if end == len(tokens) {
			break
		}
	}
	return out, nil
}

// SplitText splits text into integer-token chunks.
func (s *TokenIDTextSplitter) SplitText(text string) ([]string, error) {
	tokens, err := s.tokenizer.EncodeIDs(text)
	if err != nil {
		return nil, fmt.Errorf("encode token IDs: %w", err)
	}
	return s.splitTokenIDs(tokens)
}

func (s *TokenIDTextSplitter) splitTokenIDs(tokens []int) ([]string, error) {
	return splitTokenIDs(tokens, s.tokenizer.DecodeIDs, s.cfg.ChunkSize, s.cfg.ChunkOverlap, s.cfg.StripWhitespace)
}

func splitTokenIDs(tokens []int, decode func([]int) (string, error), tokensPerChunk int, chunkOverlap int, stripWhitespace bool) ([]string, error) {
	if len(tokens) == 0 {
		return nil, nil
	}
	step := tokensPerChunk - chunkOverlap
	if step <= 0 {
		step = tokensPerChunk
	}
	out := []string{}
	for start := 0; start < len(tokens); start += step {
		end := start + tokensPerChunk
		if end > len(tokens) {
			end = len(tokens)
		}
		chunk, err := decode(tokens[start:end])
		if err != nil {
			return nil, fmt.Errorf("decode token IDs: %w", err)
		}
		if stripWhitespace {
			chunk = strings.TrimSpace(chunk)
		}
		if chunk != "" {
			out = append(out, chunk)
		}
		if end == len(tokens) {
			break
		}
	}
	return out, nil
}

// CreateDocuments splits texts into token-count documents.
func (s *TokenTextSplitter) CreateDocuments(texts []string, metadatas []map[string]any) ([]documents.Document, error) {
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

// CreateDocuments splits texts into integer-token documents.
func (s *TokenIDTextSplitter) CreateDocuments(texts []string, metadatas []map[string]any) ([]documents.Document, error) {
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
func (s *TokenTextSplitter) SplitDocuments(docs []documents.Document) ([]documents.Document, error) {
	texts := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
		metadatas[i] = doc.Metadata
	}
	return s.CreateDocuments(texts, metadatas)
}

// SplitDocuments splits existing documents using integer token IDs.
func (s *TokenIDTextSplitter) SplitDocuments(docs []documents.Document) ([]documents.Document, error) {
	texts := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
		metadatas[i] = doc.Metadata
	}
	return s.CreateDocuments(texts, metadatas)
}
