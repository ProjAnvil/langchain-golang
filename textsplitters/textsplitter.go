// Package textsplitters provides deterministic document chunking utilities.
package textsplitters

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/projanvil/langchain-golang/core/documents"
)

// KeepSeparator controls whether separators are preserved in split chunks.
type KeepSeparator string

const (
	// KeepSeparatorNone drops separators.
	KeepSeparatorNone KeepSeparator = ""
	// KeepSeparatorStart attaches separators to the start of the following split.
	KeepSeparatorStart KeepSeparator = "start"
	// KeepSeparatorEnd attaches separators to the end of the preceding split.
	KeepSeparatorEnd KeepSeparator = "end"
)

// LengthFunc returns the length of text for chunk-size accounting.
type LengthFunc func(string) int

// Config configures text splitting.
type Config struct {
	ChunkSize       int
	ChunkOverlap    int
	LengthFunc      LengthFunc
	KeepSeparator   KeepSeparator
	AddStartIndex   bool
	StripWhitespace bool
}

// normalize validates and fills defaults.
func (c Config) normalize() (Config, error) {
	if c.ChunkSize == 0 {
		c.ChunkSize = 4000
	}
	if c.ChunkSize <= 0 {
		return Config{}, fmt.Errorf("chunk size must be > 0, got %d", c.ChunkSize)
	}
	if c.ChunkOverlap < 0 {
		return Config{}, fmt.Errorf("chunk overlap must be >= 0, got %d", c.ChunkOverlap)
	}
	if c.ChunkOverlap > c.ChunkSize {
		return Config{}, fmt.Errorf("chunk overlap %d must be <= chunk size %d", c.ChunkOverlap, c.ChunkSize)
	}
	if c.LengthFunc == nil {
		c.LengthFunc = runeLen
	}
	if c.StripWhitespace == false {
		// False is a valid explicit value, but Go cannot distinguish it from
		// zero value. Keep Python's default by setting true in constructors when
		// callers pass an all-zero Config.
	}
	return c, nil
}

// TextSplitter contains shared document splitting behavior.
type TextSplitter struct {
	cfg Config
}

// CreateDocuments splits texts into documents with optional metadata.
func (s TextSplitter) CreateDocuments(texts []string, metadatas []map[string]any, split func(string) []string) []documents.Document {
	docs := make([]documents.Document, 0)
	for i, text := range texts {
		metadata := map[string]any(nil)
		if i < len(metadatas) {
			metadata = cloneMetadata(metadatas[i])
		}
		index := 0
		previousChunkLen := 0
		for _, chunk := range split(text) {
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
	return docs
}

// SplitDocuments splits documents and copies metadata to each chunk.
func (s TextSplitter) SplitDocuments(docs []documents.Document, split func(string) []string) []documents.Document {
	texts := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
		metadatas[i] = doc.Metadata
	}
	return s.CreateDocuments(texts, metadatas, split)
}

func (s TextSplitter) mergeSplits(splits []string, separator string) []string {
	separatorLen := s.cfg.LengthFunc(separator)
	docs := []string{}
	current := []string{}
	total := 0
	for _, split := range splits {
		length := s.cfg.LengthFunc(split)
		if total+length+separatorIfNeeded(separatorLen, len(current)) > s.cfg.ChunkSize {
			if len(current) > 0 {
				if doc := s.joinDocs(current, separator); doc != "" {
					docs = append(docs, doc)
				}
				for total > s.cfg.ChunkOverlap ||
					(total+length+separatorIfNeeded(separatorLen, len(current)) > s.cfg.ChunkSize && total > 0) {
					total -= s.cfg.LengthFunc(current[0])
					if len(current) > 1 {
						total -= separatorLen
					}
					current = current[1:]
				}
			}
		}
		current = append(current, split)
		total += length
		if len(current) > 1 {
			total += separatorLen
		}
	}
	if doc := s.joinDocs(current, separator); doc != "" {
		docs = append(docs, doc)
	}
	return docs
}

func (s TextSplitter) joinDocs(parts []string, separator string) string {
	text := strings.Join(parts, separator)
	if s.cfg.StripWhitespace {
		text = strings.TrimSpace(text)
	}
	return text
}

// CharacterTextSplitter splits text on one separator and merges pieces into
// chunks.
type CharacterTextSplitter struct {
	TextSplitter
	separator        string
	isSeparatorRegex bool
}

// NewCharacter creates a CharacterTextSplitter.
func NewCharacter(separator string, isSeparatorRegex bool, cfg Config) (*CharacterTextSplitter, error) {
	if cfg.ChunkSize == 0 && cfg.ChunkOverlap == 0 && cfg.LengthFunc == nil && cfg.KeepSeparator == "" {
		cfg.StripWhitespace = true
	}
	normalized, err := cfg.normalize()
	if err != nil {
		return nil, err
	}
	return &CharacterTextSplitter{
		TextSplitter:     TextSplitter{cfg: normalized},
		separator:        separator,
		isSeparatorRegex: isSeparatorRegex,
	}, nil
}

// SplitText splits text into chunks.
func (s *CharacterTextSplitter) SplitText(text string) []string {
	pattern := regexp.QuoteMeta(s.separator)
	if s.isSeparatorRegex {
		pattern = s.separator
	}
	splits := splitTextWithRegex(text, pattern, s.cfg.KeepSeparator)
	mergeSeparator := ""
	if s.cfg.KeepSeparator == KeepSeparatorNone && !isLookaround(s.separator, s.isSeparatorRegex) {
		mergeSeparator = s.separator
	}
	return s.mergeSplits(splits, mergeSeparator)
}

// CreateDocuments splits texts into documents.
func (s *CharacterTextSplitter) CreateDocuments(texts []string, metadatas []map[string]any) []documents.Document {
	return s.TextSplitter.CreateDocuments(texts, metadatas, s.SplitText)
}

// SplitDocuments splits documents.
func (s *CharacterTextSplitter) SplitDocuments(docs []documents.Document) []documents.Document {
	return s.TextSplitter.SplitDocuments(docs, s.SplitText)
}

// RecursiveCharacterTextSplitter recursively tries separators until chunks fit.
type RecursiveCharacterTextSplitter struct {
	TextSplitter
	separators       []string
	isSeparatorRegex bool
}

// NewRecursiveCharacter creates a RecursiveCharacterTextSplitter.
func NewRecursiveCharacter(separators []string, isSeparatorRegex bool, cfg Config) (*RecursiveCharacterTextSplitter, error) {
	if len(separators) == 0 {
		separators = []string{"\n\n", "\n", " ", ""}
	}
	if cfg.KeepSeparator == "" {
		cfg.KeepSeparator = KeepSeparatorStart
	}
	if cfg.ChunkSize == 0 && cfg.ChunkOverlap == 0 && cfg.LengthFunc == nil {
		cfg.StripWhitespace = true
	}
	normalized, err := cfg.normalize()
	if err != nil {
		return nil, err
	}
	return &RecursiveCharacterTextSplitter{
		TextSplitter:     TextSplitter{cfg: normalized},
		separators:       append([]string(nil), separators...),
		isSeparatorRegex: isSeparatorRegex,
	}, nil
}

// SplitText splits text into chunks.
func (s *RecursiveCharacterTextSplitter) SplitText(text string) []string {
	return s.splitText(text, s.separators)
}

// CreateDocuments splits texts into documents.
func (s *RecursiveCharacterTextSplitter) CreateDocuments(texts []string, metadatas []map[string]any) []documents.Document {
	return s.TextSplitter.CreateDocuments(texts, metadatas, s.SplitText)
}

// SplitDocuments splits documents.
func (s *RecursiveCharacterTextSplitter) SplitDocuments(docs []documents.Document) []documents.Document {
	return s.TextSplitter.SplitDocuments(docs, s.SplitText)
}

func (s *RecursiveCharacterTextSplitter) splitText(text string, separators []string) []string {
	separator := separators[len(separators)-1]
	newSeparators := []string{}
	for i, candidate := range separators {
		pattern := regexp.QuoteMeta(candidate)
		if s.isSeparatorRegex {
			pattern = candidate
		}
		if candidate == "" || regexpPatternMatches(text, pattern) {
			separator = candidate
			newSeparators = separators[i+1:]
			break
		}
	}

	pattern := regexp.QuoteMeta(separator)
	if s.isSeparatorRegex {
		pattern = separator
	}
	splits := splitTextWithRegex(text, pattern, s.cfg.KeepSeparator)
	goodSplits := []string{}
	finalChunks := []string{}
	mergeSeparator := ""
	if s.cfg.KeepSeparator == KeepSeparatorNone {
		mergeSeparator = separator
	}
	for _, split := range splits {
		if s.cfg.LengthFunc(split) < s.cfg.ChunkSize {
			goodSplits = append(goodSplits, split)
			continue
		}
		if len(goodSplits) > 0 {
			finalChunks = append(finalChunks, s.mergeSplits(goodSplits, mergeSeparator)...)
			goodSplits = nil
		}
		if len(newSeparators) == 0 {
			finalChunks = append(finalChunks, split)
		} else {
			finalChunks = append(finalChunks, s.splitText(split, newSeparators)...)
		}
	}
	if len(goodSplits) > 0 {
		finalChunks = append(finalChunks, s.mergeSplits(goodSplits, mergeSeparator)...)
	}
	return finalChunks
}

func splitTextWithRegex(text string, pattern string, keep KeepSeparator) []string {
	if pattern == "" {
		return splitRunes(text)
	}
	matches, ok := lookaroundSplitPositions(text, pattern)
	if !ok {
		re := regexp.MustCompile(pattern)
		matches = normalizeRegexpMatches(re.FindAllStringIndex(text, -1))
	}
	if len(matches) == 0 {
		if text == "" {
			return nil
		}
		return []string{text}
	}
	out := []string{}
	start := 0
	for _, match := range matches {
		before := text[start:match[0]]
		sep := text[match[0]:match[1]]
		switch keep {
		case KeepSeparatorStart:
			if before != "" {
				out = append(out, before)
			}
			if sep != "" {
				start = match[0]
			} else {
				start = match[1]
			}
		case KeepSeparatorEnd:
			if before+sep != "" {
				out = append(out, before+sep)
			}
			start = match[1]
		default:
			if before != "" {
				out = append(out, before)
			}
			start = match[1]
		}
		if keep == KeepSeparatorStart {
			start = match[0]
			if match[0] == match[1] {
				start = match[1]
			}
		}
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return compactStrings(out)
}

func normalizeRegexpMatches(matches [][]int) [][2]int {
	out := make([][2]int, 0, len(matches))
	for _, match := range matches {
		out = append(out, [2]int{match[0], match[1]})
	}
	return out
}

func regexpPatternMatches(text string, pattern string) bool {
	if matches, ok := lookaroundSplitPositions(text, pattern); ok {
		return len(matches) > 0
	}
	return regexp.MustCompile(pattern).FindStringIndex(text) != nil
}

func lookaroundSplitPositions(text string, pattern string) ([][2]int, bool) {
	var inner string
	var useEnd bool
	switch {
	case strings.HasPrefix(pattern, "(?=") && strings.HasSuffix(pattern, ")"):
		inner = strings.TrimSuffix(strings.TrimPrefix(pattern, "(?="), ")")
	case strings.HasPrefix(pattern, "(?<=") && strings.HasSuffix(pattern, ")"):
		inner = strings.TrimSuffix(strings.TrimPrefix(pattern, "(?<="), ")")
		useEnd = true
	default:
		return nil, false
	}

	re, err := regexp.Compile(inner)
	if err != nil {
		return nil, false
	}
	innerMatches := re.FindAllStringIndex(text, -1)
	matches := make([][2]int, 0, len(innerMatches))
	for _, match := range innerMatches {
		index := match[0]
		if useEnd {
			index = match[1]
		}
		matches = append(matches, [2]int{index, index})
	}
	return matches, true
}

func splitRunes(text string) []string {
	out := make([]string, 0, utf8.RuneCountInString(text))
	for _, r := range text {
		out = append(out, string(r))
	}
	return out
}

func compactStrings(values []string) []string {
	out := values[:0]
	for _, value := range values {
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func separatorIfNeeded(separatorLen int, currentLen int) int {
	if currentLen > 0 {
		return separatorLen
	}
	return 0
}

func isLookaround(separator string, isRegex bool) bool {
	if !isRegex {
		return false
	}
	return strings.HasPrefix(separator, "(?=") ||
		strings.HasPrefix(separator, "(?<!") ||
		strings.HasPrefix(separator, "(?<=") ||
		strings.HasPrefix(separator, "(?!")
}

func runeLen(text string) int {
	return utf8.RuneCountInString(text)
}

func cloneMetadata(metadata map[string]any) map[string]any {
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		out[key] = value
	}
	return out
}
