package textsplitters

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

var (
	markdownSyntaxHeaderRe = regexp.MustCompile(`^(#{1,6}) (.*)`)
	markdownSyntaxCodeRes  = []*regexp.Regexp{
		regexp.MustCompile(`^` + "```" + `(.*)`),
		regexp.MustCompile(`^~~~(.*)`),
	}
	markdownSyntaxHorzRes = []*regexp.Regexp{
		regexp.MustCompile(`^\*\*\*+\n`),
		regexp.MustCompile(`^---+\n`),
		regexp.MustCompile(`^___+\n`),
	}
)

// MarkdownSyntaxTextSplitter splits Markdown by syntax boundaries — headers,
// fenced code blocks, and horizontal rules — and annotates each chunk with the
// active headers and code language.
//
// It is a port of langchain's ExperimentalMarkdownSyntaxTextSplitter and
// retains the exact whitespace of the original text.
type MarkdownSyntaxTextSplitter struct {
	markerByName   map[string]string
	returnEachLine bool
	stripHeaders   bool
}

// NewMarkdownSyntax creates a MarkdownSyntaxTextSplitter.
//
// headers maps a header marker ("#", "##", ...) to a metadata key. When empty,
// the six ATX levels default to "Header 1".."Header 6".
func NewMarkdownSyntax(headers []Header, returnEachLine bool, stripHeaders bool) *MarkdownSyntaxTextSplitter {
	markerByName := map[string]string{}
	if len(headers) == 0 {
		for i := 1; i <= 6; i++ {
			markerByName[strings.Repeat("#", i)] = "Header " + strconv.Itoa(i)
		}
	} else {
		for _, header := range headers {
			markerByName[header.Marker] = header.Name
		}
	}
	return &MarkdownSyntaxTextSplitter{
		markerByName:   markerByName,
		returnEachLine: returnEachLine,
		stripHeaders:   stripHeaders,
	}
}

// SplitText splits Markdown into structured chunks. When returnEachLine is
// enabled, each non-empty line is returned as its own Document carrying the
// chunk's metadata.
func (s *MarkdownSyntaxTextSplitter) SplitText(text string) []documents.Document {
	lines := splitLinesKeepEnds(text)
	chunks := []documents.Document{}
	currentContent := ""
	currentMetadata := map[string]any{}
	headerStack := []markdownHeader{}

	completeChunk := func() {
		if strings.TrimSpace(currentContent) == "" {
			currentContent = ""
			currentMetadata = map[string]any{}
			return
		}
		for _, header := range headerStack {
			if key := s.markerByName[strings.Repeat("#", header.level)]; key != "" {
				currentMetadata[key] = header.data
			}
		}
		chunks = append(chunks, documents.New(currentContent, currentMetadata))
		currentContent = ""
		currentMetadata = map[string]any{}
	}

	for len(lines) > 0 {
		rawLine := lines[0]
		lines = lines[1:]

		if match := markdownSyntaxHeaderRe.FindStringSubmatch(rawLine); match != nil {
			marker := match[1]
			if _, configured := s.markerByName[marker]; !configured {
				currentContent += rawLine
				continue
			}
			completeChunk()
			if !s.stripHeaders {
				currentContent += rawLine
			}
			s.resolveHeaderStack(&headerStack, len(marker), match[2])
			continue
		}

		if language, ok := matchMarkdownSyntaxCode(rawLine); ok {
			completeChunk()
			currentContent = resolveMarkdownSyntaxCodeChunk(rawLine, &lines)
			currentMetadata["Code"] = language
			completeChunk()
			continue
		}

		if matchMarkdownSyntaxHorz(rawLine) {
			completeChunk()
			continue
		}

		currentContent += rawLine
	}
	completeChunk()

	if s.returnEachLine {
		out := []documents.Document{}
		for _, chunk := range chunks {
			for _, line := range splitLinesNoEnds(chunk.PageContent) {
				if strings.TrimSpace(line) == "" {
					continue
				}
				out = append(out, documents.New(line, chunk.Metadata))
			}
		}
		return out
	}
	return chunks
}

func (s *MarkdownSyntaxTextSplitter) resolveHeaderStack(stack *[]markdownHeader, depth int, text string) {
	for i, header := range *stack {
		if header.level >= depth {
			*stack = (*stack)[:i]
			break
		}
	}
	*stack = append(*stack, markdownHeader{level: depth, data: text})
}

func matchMarkdownSyntaxCode(line string) (string, bool) {
	for _, re := range markdownSyntaxCodeRes {
		if match := re.FindStringSubmatch(line); match != nil {
			return match[1], true
		}
	}
	return "", false
}

func resolveMarkdownSyntaxCodeChunk(currentLine string, lines *[]string) string {
	chunk := currentLine
	for len(*lines) > 0 {
		rawLine := (*lines)[0]
		*lines = (*lines)[1:]
		chunk += rawLine
		if _, ok := matchMarkdownSyntaxCode(rawLine); ok {
			return chunk
		}
	}
	return ""
}

func matchMarkdownSyntaxHorz(line string) bool {
	for _, re := range markdownSyntaxHorzRes {
		if re.MatchString(line) {
			return true
		}
	}
	return false
}

// splitLinesKeepEnds splits text into lines while retaining their line endings,
// mirroring Python's str.splitlines(keepends=True) for the common \n, \r, and
// \r\n boundaries.
func splitLinesKeepEnds(text string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '\n':
			lines = append(lines, text[start:i+1])
			start = i + 1
		case '\r':
			end := i + 1
			if end < len(text) && text[end] == '\n' {
				end++
			}
			lines = append(lines, text[start:end])
			i = end - 1
			start = end
		}
	}
	if start < len(text) {
		lines = append(lines, text[start:])
	}
	return lines
}

// splitLinesNoEnds splits text into lines without their endings, mirroring
// Python's str.splitlines() for the common \n, \r, and \r\n boundaries.
func splitLinesNoEnds(text string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '\n':
			lines = append(lines, text[start:i])
			start = i + 1
		case '\r':
			lines = append(lines, text[start:i])
			if i+1 < len(text) && text[i+1] == '\n' {
				i++
			}
			start = i + 1
		}
	}
	lines = append(lines, text[start:])
	return lines
}
