package textsplitters

import (
	"html"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

// HTMLSectionSplitter splits an HTML document into sections keyed by header
// tags (or by <body>, whose leading section is labeled with the "#TITLE#"
// placeholder).
//
// It is a port of langchain's HTMLSectionSplitter. The Python reference depends
// on lxml for a font-size XSLT transform and on BeautifulSoup for parsing. This
// Go port reproduces the observable behavior with a stdlib-only tokenizer and
// tree, so it adds no external dependency. Divergences:
//
//   - The lxml XSLT transform is reproduced by renaming elements whose inline
//     style carries a `font-size: Npx` declaration with N > 20 to <h1>, but the
//     lxml serialize/re-parse round trip (which can inject whitespace between
//     block elements) is not performed; whitespace already present in the input
//     is preserved subject to the same collapse rules as BeautifulSoup's
//     html.parser.
//   - Tag attributes containing a `>` character are not tokenized (a stdlib
//     limitation of the lightweight parser).
type HTMLSectionSplitter struct {
	headerNameByTag map[string]string
	headerTagSet    map[string]struct{}
}

// NewHTMLSection creates an HTMLSectionSplitter. Each Header.Marker is an HTML
// header tag such as "h1" and Header.Name is the metadata key to emit.
func NewHTMLSection(headers []Header) *HTMLSectionSplitter {
	nameByTag := make(map[string]string, len(headers))
	tagSet := make(map[string]struct{}, len(headers)+1)
	tagSet["body"] = struct{}{}
	for _, header := range headers {
		tag := strings.ToLower(strings.TrimSpace(header.Marker))
		if tag == "" {
			continue
		}
		nameByTag[tag] = header.Name
		tagSet[tag] = struct{}{}
	}
	return &HTMLSectionSplitter{
		headerNameByTag: nameByTag,
		headerTagSet:    tagSet,
	}
}

// SplitText splits an HTML string into sections keyed by the configured header
// tags. The section before the first header (inside <body>) carries the
// "#TITLE#" placeholder as its header value.
func (s *HTMLSectionSplitter) SplitText(text string) []documents.Document {
	root := parseHTMLTree(text)
	renameLargeFontSizeElements(root)

	headers := collectHTMLByName(root, s.headerTagSet)
	if len(headers) == 0 {
		return nil
	}

	flat := flattenHTMLTree(root)
	indexOf := make(map[*htmlNode]int, len(flat))
	for i, node := range flat {
		if node.kind == htmlElementNode {
			indexOf[node] = i
		}
	}

	out := []documents.Document{}
	for i, header := range headers {
		headerValue := "#TITLE#"
		tag := "h1"
		if i > 0 {
			headerValue = strings.TrimSpace(htmlElementText(header))
			tag = header.name
		}

		start := indexOf[header] + 1
		end := len(flat)
		if i+1 < len(headers) {
			end = indexOf[headers[i+1]]
		}

		var parts []string
		for _, node := range flat[start:end] {
			if node.kind == htmlTextNode {
				parts = append(parts, node.text)
			}
		}
		content := strings.TrimSpace(strings.Join(parts, " "))
		if content == "" {
			continue
		}

		key := s.headerNameByTag[tag]
		out = append(out, documents.New(content, map[string]any{key: headerValue}))
	}
	return out
}

// CreateDocuments splits texts into documents and merges source metadata into
// each section. A "#TITLE#" header placeholder is replaced with the source
// document's "Title" metadata value when present.
func (s *HTMLSectionSplitter) CreateDocuments(texts []string, metadatas []map[string]any) []documents.Document {
	out := []documents.Document{}
	for i, text := range texts {
		source := map[string]any(nil)
		if i < len(metadatas) {
			source = cloneMetadata(metadatas[i])
		}
		for _, chunk := range s.SplitText(text) {
			chunkMetadata := cloneMetadata(chunk.Metadata)
			for key, value := range chunkMetadata {
				if value == "#TITLE#" {
					if title, ok := source["Title"]; ok {
						chunkMetadata[key] = title
					}
				}
			}
			merged := cloneMetadata(source)
			for key, value := range chunkMetadata {
				merged[key] = value
			}
			out = append(out, documents.New(chunk.PageContent, merged))
		}
	}
	return out
}

// SplitDocuments splits existing documents and preserves their metadata.
//
// Unlike the Python reference, it does not additionally re-chunk each section
// with a RecursiveCharacterTextSplitter; callers that need size-based chunking
// can apply it to the results themselves.
func (s *HTMLSectionSplitter) SplitDocuments(docs []documents.Document) []documents.Document {
	texts := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
		metadatas[i] = doc.Metadata
	}
	return s.CreateDocuments(texts, metadatas)
}

// htmlNode is a lightweight HTML parse-tree node.
type htmlNode struct {
	kind     htmlNodeKind
	name     string
	text     string
	attrs    map[string]string
	children []*htmlNode
}

type htmlNodeKind int

const (
	htmlTextNode htmlNodeKind = iota
	htmlElementNode
)

// htmlToken is one lexical unit produced by the tokenizer.
type htmlToken struct {
	kind        htmlTokenKind
	name        string
	text        string
	attrs       string
	selfClosing bool
}

type htmlTokenKind int

const (
	htmlTokenText htmlTokenKind = iota
	htmlTokenOpen
	htmlTokenClose
)

func parseHTMLTree(text string) *htmlNode {
	root := &htmlNode{kind: htmlElementNode, name: "#root"}
	stack := []*htmlNode{root}

	for _, token := range tokenizeHTML(text) {
		switch token.kind {
		case htmlTokenText:
			if token.text == "" {
				continue
			}
			collapsed := token.text
			if !inWhitespacePreservingTag(stack) {
				collapsed = collapseHTMLWhitespace(token.text)
			}
			parent := stack[len(stack)-1]
			parent.children = append(parent.children, &htmlNode{kind: htmlTextNode, text: collapsed})
		case htmlTokenOpen:
			node := &htmlNode{
				kind:  htmlElementNode,
				name:  token.name,
				attrs: parseHTMLAttrs(token.attrs),
			}
			stack[len(stack)-1].children = append(stack[len(stack)-1].children, node)
			if !isHTMLVoidElement(token.name) && !token.selfClosing {
				stack = append(stack, node)
			}
		case htmlTokenClose:
			for j := len(stack) - 1; j > 0; j-- {
				if stack[j].name == token.name {
					stack = stack[:j]
					break
				}
			}
		}
	}
	return root
}

func tokenizeHTML(text string) []htmlToken {
	tokens := []htmlToken{}
	for i := 0; i < len(text); {
		if text[i] != '<' {
			j := i
			for j < len(text) && text[j] != '<' {
				j++
			}
			tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(text[i:j])})
			i = j
			continue
		}

		// A '<' that does not start a tag is literal text.
		if i+1 >= len(text) || !isHTMLTagStartChar(text[i+1]) {
			j := i
			for j < len(text) && text[j] != '<' {
				j++
			}
			tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(text[i:j])})
			i = j
			continue
		}

		switch {
		case strings.HasPrefix(text[i:], "<!--"):
			end := strings.Index(text[i+4:], "-->")
			if end < 0 {
				tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(text[i:])})
				return tokens
			}
			inner := text[i+4 : i+4+end]
			tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(inner)})
			i += 4 + end + 3
		case strings.HasPrefix(text[i:], "<![CDATA["):
			end := strings.Index(text[i+9:], "]]>")
			if end < 0 {
				tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(text[i:])})
				return tokens
			}
			inner := text[i+9 : i+9+end]
			tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(inner)})
			i += 9 + end + 3
		case strings.HasPrefix(text[i:], "<!") || strings.HasPrefix(text[i:], "<?"):
			end := findHTMLTagEnd(text[i:])
			if end < 0 {
				return tokens
			}
			i += end + 1
		case strings.HasPrefix(text[i:], "</"):
			end := findHTMLTagEnd(text[i:])
			if end < 0 {
				return tokens
			}
			name := strings.ToLower(htmlTagName(text[i+2 : i+end]))
			tokens = append(tokens, htmlToken{kind: htmlTokenClose, name: name})
			i += end + 1
		default:
			end := findHTMLTagEnd(text[i:])
			if end < 0 {
				tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(text[i:])})
				return tokens
			}
			raw := text[i+1 : i+end]
			name := strings.ToLower(htmlTagName(raw))
			selfClosing := strings.HasSuffix(strings.TrimSpace(raw), "/")
			attrs := strings.TrimSpace(raw[len(htmlTagName(raw)):])
			if selfClosing {
				attrs = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(attrs), "/"))
			}
			tokens = append(tokens, htmlToken{kind: htmlTokenOpen, name: name, attrs: attrs, selfClosing: selfClosing})
			i += end + 1

			// <script> and <style> bodies are raw text, not markup.
			if name == "script" || name == "style" {
				lower := strings.ToLower(text[i:])
				closeIdx := strings.Index(lower, "</"+name)
				if closeIdx < 0 {
					tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(text[i:])})
					return tokens
				}
				gt := strings.IndexByte(lower[closeIdx:], '>')
				if gt < 0 {
					tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(text[i:])})
					return tokens
				}
				rawContent := text[i : i+closeIdx]
				tokens = append(tokens, htmlToken{kind: htmlTokenText, text: html.UnescapeString(rawContent)})
				tokens = append(tokens, htmlToken{kind: htmlTokenClose, name: name})
				i += closeIdx + gt + 1
			}
		}
	}
	return tokens
}

func isHTMLTagStartChar(c byte) bool {
	switch {
	case c == '/', c == '!', c == '?':
		return true
	case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z':
		return true
	default:
		return false
	}
}

// htmlTagName returns the element name at the head of a raw tag body.
func htmlTagName(raw string) string {
	i := 0
	for i < len(raw) {
		switch raw[i] {
		case ' ', '\t', '\n', '\r', '\f', '/':
			return raw[:i]
		}
		i++
	}
	return raw
}

// findHTMLTagEnd returns the index of the closing '>' outside quoted attribute
// values, or -1.
func findHTMLTagEnd(s string) int {
	var quote byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if quote != 0 {
			if c == quote {
				quote = 0
			}
			continue
		}
		switch c {
		case '"', '\'':
			quote = c
		case '>':
			return i
		}
	}
	return -1
}

func inWhitespacePreservingTag(stack []*htmlNode) bool {
	for j := len(stack) - 1; j >= 0; j-- {
		switch stack[j].name {
		case "pre", "textarea":
			return true
		}
	}
	return false
}

// collapseHTMLWhitespace mimics BeautifulSoup's html.parser: an all-ASCII-space
// text node collapses to a single newline (when it contains one) or a single
// space.
func collapseHTMLWhitespace(text string) string {
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case ' ', '\n', '\t', '\f', '\r':
			continue
		default:
			return text
		}
	}
	if strings.Contains(text, "\n") {
		return "\n"
	}
	return " "
}

func isHTMLVoidElement(name string) bool {
	switch name {
	case "area", "base", "br", "col", "embed", "hr", "img", "input",
		"link", "meta", "param", "source", "track", "wbr":
		return true
	default:
		return false
	}
}

func renameLargeFontSizeElements(node *htmlNode) {
	if node.kind == htmlElementNode {
		if style, ok := node.attrs["style"]; ok && hasLargeFontSize(style) {
			node.name = "h1"
		}
		for _, child := range node.children {
			renameLargeFontSizeElements(child)
		}
	}
}

// hasLargeFontSize reports whether an inline style declares a font-size in px
// larger than 20, matching the Python XSLT transform's numeric comparison.
func hasLargeFontSize(style string) bool {
	idx := strings.Index(style, "font-size:")
	if idx < 0 {
		return false
	}
	rest := style[idx+len("font-size:"):]
	px := strings.Index(rest, "px")
	if px < 0 {
		return false
	}
	value, err := strconv.ParseFloat(strings.TrimSpace(rest[:px]), 64)
	if err != nil {
		return false
	}
	return value > 20
}

func collectHTMLByName(root *htmlNode, names map[string]struct{}) []*htmlNode {
	var out []*htmlNode
	var walk func(*htmlNode)
	walk = func(node *htmlNode) {
		if node.kind == htmlElementNode {
			if _, ok := names[node.name]; ok {
				out = append(out, node)
			}
			for _, child := range node.children {
				walk(child)
			}
		}
	}
	for _, child := range root.children {
		walk(child)
	}
	return out
}

func flattenHTMLTree(root *htmlNode) []*htmlNode {
	var out []*htmlNode
	var walk func(*htmlNode)
	walk = func(node *htmlNode) {
		out = append(out, node)
		for _, child := range node.children {
			walk(child)
		}
	}
	for _, child := range root.children {
		walk(child)
	}
	return out
}

func htmlElementText(node *htmlNode) string {
	var sb strings.Builder
	var walk func(*htmlNode)
	walk = func(n *htmlNode) {
		for _, child := range n.children {
			if child.kind == htmlTextNode {
				sb.WriteString(child.text)
			} else {
				walk(child)
			}
		}
	}
	walk(node)
	return sb.String()
}
