package textsplitters

import (
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

// Header maps a Markdown header marker to a metadata key.
type Header struct {
	Marker string
	Name   string
}

// MarkdownHeaderTextSplitter splits Markdown into documents carrying header
// metadata.
type MarkdownHeaderTextSplitter struct {
	headers        []Header
	returnEachLine bool
	stripHeaders   bool
}

// NewMarkdownHeader creates a MarkdownHeaderTextSplitter.
func NewMarkdownHeader(headers []Header, returnEachLine bool, stripHeaders bool) *MarkdownHeaderTextSplitter {
	copied := append([]Header(nil), headers...)
	sort.SliceStable(copied, func(i, j int) bool {
		return len(copied[i].Marker) > len(copied[j].Marker)
	})
	return &MarkdownHeaderTextSplitter{
		headers:        copied,
		returnEachLine: returnEachLine,
		stripHeaders:   stripHeaders,
	}
}

// SplitText splits Markdown and annotates chunks with active header metadata.
func (s *MarkdownHeaderTextSplitter) SplitText(text string) []documents.Document {
	lines := strings.Split(text, "\n")
	linesWithMetadata := []markdownLine{}
	currentContent := []string{}
	currentMetadata := map[string]any{}
	initialMetadata := map[string]any{}
	headerStack := []markdownHeader{}
	inCodeBlock := false
	openingFence := ""

	flush := func() {
		if len(currentContent) == 0 {
			return
		}
		linesWithMetadata = append(linesWithMetadata, markdownLine{
			content:  strings.Join(currentContent, "\n"),
			metadata: cloneMetadata(currentMetadata),
		})
		currentContent = nil
	}

	for _, line := range lines {
		stripped := printable(strings.TrimSpace(line))
		if !inCodeBlock {
			if strings.HasPrefix(stripped, "```") && strings.Count(stripped, "```") == 1 {
				inCodeBlock = true
				openingFence = "```"
			} else if strings.HasPrefix(stripped, "~~~") {
				inCodeBlock = true
				openingFence = "~~~"
			}
		} else if strings.HasPrefix(stripped, openingFence) {
			inCodeBlock = false
			openingFence = ""
		}

		if inCodeBlock {
			currentContent = append(currentContent, stripped)
			continue
		}

		matched := false
		for _, header := range s.headers {
			if isMarkdownHeader(stripped, header.Marker) {
				level := strings.Count(header.Marker, "#")
				for len(headerStack) > 0 && headerStack[len(headerStack)-1].level >= level {
					popped := headerStack[len(headerStack)-1]
					headerStack = headerStack[:len(headerStack)-1]
					delete(initialMetadata, popped.name)
				}
				headerText := strings.TrimSpace(stripped[len(header.Marker):])
				headerStack = append(headerStack, markdownHeader{
					level: level,
					name:  header.Name,
					data:  headerText,
				})
				initialMetadata[header.Name] = headerText
				flush()
				if !s.stripHeaders {
					currentContent = append(currentContent, stripped)
				}
				matched = true
				break
			}
		}
		if !matched {
			if stripped != "" {
				currentContent = append(currentContent, stripped)
			} else {
				flush()
			}
		}
		currentMetadata = cloneMetadata(initialMetadata)
	}
	flush()

	if s.returnEachLine {
		out := make([]documents.Document, 0, len(linesWithMetadata))
		for _, line := range linesWithMetadata {
			out = append(out, documents.New(line.content, line.metadata))
		}
		return out
	}
	return aggregateMarkdownLines(linesWithMetadata, s.stripHeaders)
}

type markdownLine struct {
	content  string
	metadata map[string]any
}

type markdownHeader struct {
	level int
	name  string
	data  string
}

func aggregateMarkdownLines(lines []markdownLine, stripHeaders bool) []documents.Document {
	chunks := []markdownLine{}
	for _, line := range lines {
		if len(chunks) > 0 && reflectMetadata(chunks[len(chunks)-1].metadata, line.metadata) {
			chunks[len(chunks)-1].content += "  \n" + line.content
			continue
		}
		if len(chunks) > 0 &&
			!reflectMetadata(chunks[len(chunks)-1].metadata, line.metadata) &&
			len(chunks[len(chunks)-1].metadata) < len(line.metadata) &&
			strings.HasPrefix(lastLine(chunks[len(chunks)-1].content), "#") &&
			!stripHeaders {
			chunks[len(chunks)-1].content += "  \n" + line.content
			chunks[len(chunks)-1].metadata = line.metadata
			continue
		}
		chunks = append(chunks, line)
	}
	out := make([]documents.Document, len(chunks))
	for i, chunk := range chunks {
		out[i] = documents.New(chunk.content, chunk.metadata)
	}
	return out
}

func isMarkdownHeader(line string, marker string) bool {
	if !strings.HasPrefix(line, marker) {
		return false
	}
	return len(line) == len(marker) || line[len(marker)] == ' '
}

func printable(text string) string {
	return strings.Map(func(r rune) rune {
		if r == '\t' || r == '\n' || r == '\r' || r >= 32 {
			return r
		}
		return -1
	}, text)
}

func reflectMetadata(a, b map[string]any) bool {
	if len(a) != len(b) {
		return false
	}
	for key, value := range a {
		if b[key] != value {
			return false
		}
	}
	return true
}

func lastLine(text string) string {
	parts := strings.Split(text, "\n")
	return parts[len(parts)-1]
}
