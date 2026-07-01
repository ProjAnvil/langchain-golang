package textsplitters

import (
	"encoding/json"
	"fmt"

	"github.com/projanvil/langchain-golang/core/documents"
)

// RecursiveJSONSplitter splits JSON-like maps into smaller JSON documents while
// preserving nested hierarchy.
type RecursiveJSONSplitter struct {
	MaxChunkSize int
	MinChunkSize int
}

// NewRecursiveJSON creates a RecursiveJSONSplitter.
func NewRecursiveJSON(maxChunkSize int, minChunkSize int) *RecursiveJSONSplitter {
	if maxChunkSize <= 0 {
		maxChunkSize = 2000
	}
	if minChunkSize <= 0 {
		minChunkSize = max(maxChunkSize-200, 50)
	}
	return &RecursiveJSONSplitter{
		MaxChunkSize: maxChunkSize,
		MinChunkSize: minChunkSize,
	}
}

// SplitJSON splits JSON data into smaller maps.
func (s *RecursiveJSONSplitter) SplitJSON(data map[string]any, convertLists bool) []map[string]any {
	var input any = data
	if convertLists {
		input = listToDict(input)
	}
	chunks := []map[string]any{{}}
	s.jsonSplit(input, nil, &chunks)
	if len(chunks) > 0 && len(chunks[len(chunks)-1]) == 0 {
		chunks = chunks[:len(chunks)-1]
	}
	return chunks
}

// SplitText splits JSON data into JSON strings.
func (s *RecursiveJSONSplitter) SplitText(data map[string]any, convertLists bool) ([]string, error) {
	chunks := s.SplitJSON(data, convertLists)
	out := make([]string, len(chunks))
	for i, chunk := range chunks {
		b, err := json.Marshal(chunk)
		if err != nil {
			return nil, fmt.Errorf("marshal json chunk: %w", err)
		}
		out[i] = string(b)
	}
	return out, nil
}

// CreateDocuments splits JSON maps into documents.
func (s *RecursiveJSONSplitter) CreateDocuments(
	texts []map[string]any,
	convertLists bool,
	metadatas []map[string]any,
) ([]documents.Document, error) {
	out := []documents.Document{}
	for i, text := range texts {
		chunks, err := s.SplitText(text, convertLists)
		if err != nil {
			return nil, err
		}
		var metadata map[string]any
		if i < len(metadatas) {
			metadata = metadatas[i]
		}
		for _, chunk := range chunks {
			out = append(out, documents.New(chunk, metadata))
		}
	}
	return out, nil
}

func (s *RecursiveJSONSplitter) jsonSplit(data any, currentPath []string, chunks *[]map[string]any) {
	if object, ok := data.(map[string]any); ok && len(object) > 0 {
		for key, value := range object {
			newPath := append(append([]string(nil), currentPath...), key)
			chunkSize := jsonSize((*chunks)[len(*chunks)-1])
			itemSize := jsonSize(map[string]any{key: value})
			remaining := s.MaxChunkSize - chunkSize
			if itemSize < remaining {
				setNested((*chunks)[len(*chunks)-1], newPath, value)
				continue
			}
			if chunkSize >= s.MinChunkSize {
				*chunks = append(*chunks, map[string]any{})
			}
			s.jsonSplit(value, newPath, chunks)
		}
		return
	}
	if len(currentPath) > 0 {
		setNested((*chunks)[len(*chunks)-1], currentPath, data)
	}
}

func jsonSize(data map[string]any) int {
	b, err := json.Marshal(data)
	if err != nil {
		return 0
	}
	return len(b)
}

func setNested(data map[string]any, path []string, value any) {
	current := data
	for _, key := range path[:len(path)-1] {
		next, ok := current[key].(map[string]any)
		if !ok {
			next = map[string]any{}
			current[key] = next
		}
		current = next
	}
	current[path[len(path)-1]] = value
}

func listToDict(data any) any {
	switch value := data.(type) {
	case map[string]any:
		out := make(map[string]any, len(value))
		for key, item := range value {
			out[key] = listToDict(item)
		}
		return out
	case []any:
		out := make(map[string]any, len(value))
		for i, item := range value {
			out[fmt.Sprintf("%d", i)] = listToDict(item)
		}
		return out
	default:
		return data
	}
}
