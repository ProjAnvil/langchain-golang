package textsplitters

import (
	"html"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

// HTMLHeaderTextSplitter splits HTML content into documents carrying active
// header metadata.
type HTMLHeaderTextSplitter struct {
	headers           []Header
	headerNameByTag   map[string]string
	headerLevelByName map[string]int
	returnEachElement bool
}

// HTMLSemanticPreservingSplitter splits HTML into semantic block documents.
// Header tags update hierarchy metadata, while configured semantic tags are
// preserved as document boundaries unless their text exceeds ChunkSize.
type HTMLSemanticPreservingSplitter struct {
	TextSplitter
	headers           []Header
	headerNameByTag   map[string]string
	headerLevelByName map[string]int
	semanticTags      map[string]struct{}
	options           *HTMLSemanticOptions
}

// HTMLCustomHandler extracts replacement text for a custom HTML element.
type HTMLCustomHandler func(attrs map[string]string, text string) string

// HTMLSemanticOptions configures Python-compatible semantic HTML splitting.
type HTMLSemanticOptions struct {
	SemanticTags     []string
	PreserveLinks    bool
	PreserveImages   bool
	PreserveVideos   bool
	PreserveAudio    bool
	AllowlistTags    []string
	DenylistTags     []string
	ExternalMetadata map[string]any
	NormalizeText    bool
	CustomHandlers   map[string]HTMLCustomHandler
}

// NewHTMLHeader creates an HTML header splitter. Header.Marker should be an
// HTML header tag such as "h1" or "h2".
func NewHTMLHeader(headers []Header, returnEachElement bool) *HTMLHeaderTextSplitter {
	copied, nameByTag, levelByName := normalizeHTMLHeaders(headers)
	return &HTMLHeaderTextSplitter{
		headers:           copied,
		headerNameByTag:   nameByTag,
		headerLevelByName: levelByName,
		returnEachElement: returnEachElement,
	}
}

// NewHTMLSemanticPreserving creates an HTML splitter that keeps semantic block
// boundaries. semanticTags defaults to common HTML content sectioning tags.
func NewHTMLSemanticPreserving(headers []Header, semanticTags []string, cfg Config) (*HTMLSemanticPreservingSplitter, error) {
	return newHTMLSemanticPreserving(headers, semanticTags, cfg, nil)
}

// NewHTMLSemanticPreservingWithOptions creates a Python-compatible semantic
// HTML splitter. It preserves the existing NewHTMLSemanticPreserving behavior
// while exposing Python beta splitter features such as links, media, custom
// handlers, allow/deny lists, external metadata, and text normalization.
func NewHTMLSemanticPreservingWithOptions(headers []Header, cfg Config, options HTMLSemanticOptions) (*HTMLSemanticPreservingSplitter, error) {
	semanticTags := options.SemanticTags
	return newHTMLSemanticPreserving(headers, semanticTags, cfg, &options)
}

func newHTMLSemanticPreserving(headers []Header, semanticTags []string, cfg Config, options *HTMLSemanticOptions) (*HTMLSemanticPreservingSplitter, error) {
	if cfg.ChunkSize == 0 && cfg.ChunkOverlap == 0 && cfg.LengthFunc == nil {
		cfg.StripWhitespace = true
	}
	normalized, err := cfg.normalize()
	if err != nil {
		return nil, err
	}
	copied, nameByTag, levelByName := normalizeHTMLHeaders(headers)
	if len(semanticTags) == 0 {
		semanticTags = []string{
			"article", "section", "main", "aside", "nav", "header", "footer",
			"p", "li", "blockquote", "pre", "code", "td", "th",
		}
	}
	tagSet := make(map[string]struct{}, len(semanticTags))
	for _, tag := range semanticTags {
		tag = strings.ToLower(strings.TrimSpace(tag))
		if tag != "" {
			tagSet[tag] = struct{}{}
		}
	}
	return &HTMLSemanticPreservingSplitter{
		TextSplitter:      TextSplitter{cfg: normalized},
		headers:           copied,
		headerNameByTag:   nameByTag,
		headerLevelByName: levelByName,
		semanticTags:      tagSet,
		options:           options,
	}, nil
}

func normalizeHTMLHeaders(headers []Header) ([]Header, map[string]string, map[string]int) {
	copied := append([]Header(nil), headers...)
	sort.SliceStable(copied, func(i, j int) bool {
		return htmlHeaderLevel(copied[i].Marker) < htmlHeaderLevel(copied[j].Marker)
	})
	nameByTag := make(map[string]string, len(copied))
	levelByName := make(map[string]int, len(copied))
	for _, header := range copied {
		tag := strings.ToLower(header.Marker)
		nameByTag[tag] = header.Name
		levelByName[header.Name] = htmlHeaderLevel(tag)
	}
	return copied, nameByTag, levelByName
}

// SplitText splits HTML into documents.
func (s *HTMLHeaderTextSplitter) SplitText(text string) []documents.Document {
	tokens := htmlElementTokens(text)
	active := map[string]any{}
	out := []documents.Document{}
	current := []string{}
	currentMetadata := map[string]any{}

	flush := func() {
		if len(current) == 0 {
			return
		}
		content := strings.Join(current, "  \n")
		if strings.TrimSpace(content) != "" {
			out = append(out, documents.New(content, currentMetadata))
		}
		current = nil
	}

	for _, token := range tokens {
		if token.text == "" {
			continue
		}
		if headerName, ok := s.headerNameByTag[token.tag]; ok {
			if !s.returnEachElement {
				flush()
			}
			level := htmlHeaderLevel(token.tag)
			for name := range active {
				if s.headerLevelByName[name] >= level {
					delete(active, name)
				}
			}
			active[headerName] = token.text
			metadata := cloneMetadata(active)
			out = append(out, documents.New(token.text, metadata))
			currentMetadata = metadata
			continue
		}
		metadata := cloneMetadata(active)
		if s.returnEachElement {
			out = append(out, documents.New(token.text, metadata))
			continue
		}
		if len(current) > 0 && !reflectMetadata(currentMetadata, metadata) {
			flush()
		}
		currentMetadata = metadata
		current = append(current, token.text)
	}
	if !s.returnEachElement {
		flush()
	}
	if len(out) == 0 {
		content := cleanHTMLText(stripHTMLTags(text))
		if content != "" {
			out = append(out, documents.New(content, nil))
		}
	}
	return out
}

// SplitText splits HTML into semantic block documents.
func (s *HTMLSemanticPreservingSplitter) SplitText(text string) []documents.Document {
	if s.options != nil {
		return s.splitTextWithOptions(text)
	}
	tokens := htmlElementTokensWithTags(text, s.semanticTagPattern())
	active := map[string]any{}
	out := []documents.Document{}
	for _, token := range tokens {
		if token.text == "" {
			continue
		}
		if headerName, ok := s.headerNameByTag[token.tag]; ok {
			level := htmlHeaderLevel(token.tag)
			for name := range active {
				if s.headerLevelByName[name] >= level {
					delete(active, name)
				}
			}
			active[headerName] = token.text
			metadata := cloneMetadata(active)
			metadata["tag"] = token.tag
			out = append(out, documents.New(token.text, metadata))
			continue
		}
		metadata := cloneMetadata(active)
		metadata["tag"] = token.tag
		if s.cfg.LengthFunc(token.text) <= s.cfg.ChunkSize {
			out = append(out, documents.New(token.text, metadata))
			continue
		}
		splitter, err := NewRecursiveCharacter(nil, false, s.cfg)
		if err != nil {
			out = append(out, documents.New(token.text, metadata))
			continue
		}
		for _, chunk := range splitter.SplitText(token.text) {
			if strings.TrimSpace(chunk) == "" {
				continue
			}
			out = append(out, documents.New(chunk, metadata))
		}
	}
	if len(out) == 0 {
		content := cleanHTMLText(stripHTMLTags(text))
		if content != "" {
			out = append(out, documents.New(content, nil))
		}
	}
	return out
}

func (s *HTMLSemanticPreservingSplitter) splitTextWithOptions(text string) []documents.Document {
	cleaned := removeHTMLBlocks(text, "script")
	cleaned = removeHTMLBlocks(cleaned, "style")
	headerRe := regexp.MustCompile(`(?is)<(h[1-6])\b[^>]*>(.*?)</\s*(h[1-6])\s*>`)
	matches := headerRe.FindAllStringSubmatchIndex(cleaned, -1)
	active := cloneMetadata(s.options.ExternalMetadata)
	out := []documents.Document{}
	start := 0
	for _, match := range matches {
		if match[2] < 0 || match[4] < 0 || match[6] < 0 {
			continue
		}
		out = append(out, s.documentsFromHTMLSegment(cleaned[start:match[0]], active)...)
		tag := strings.ToLower(cleaned[match[2]:match[3]])
		closeTag := strings.ToLower(cleaned[match[6]:match[7]])
		if tag == closeTag {
			level := htmlHeaderLevel(tag)
			for name := range active {
				if s.headerLevelByName[name] >= level {
					delete(active, name)
				}
			}
			if headerName, ok := s.headerNameByTag[tag]; ok {
				active[headerName] = cleanHTMLText(stripHTMLTags(cleaned[match[4]:match[5]]))
			}
		}
		start = match[1]
	}
	out = append(out, s.documentsFromHTMLSegment(cleaned[start:], active)...)
	if len(out) == 0 {
		content := s.extractOptionHTMLText(cleaned)
		if content != "" {
			out = append(out, documents.New(content, cloneMetadata(s.options.ExternalMetadata)))
		}
	}
	return out
}

func (s *HTMLSemanticPreservingSplitter) documentsFromHTMLSegment(segment string, metadata map[string]any) []documents.Document {
	content := s.extractOptionHTMLText(segment)
	if content == "" {
		return nil
	}
	docMetadata := cloneMetadata(metadata)
	if s.cfg.LengthFunc(content) <= s.cfg.ChunkSize {
		return []documents.Document{documents.New(content, docMetadata)}
	}
	splitter, err := NewRecursiveCharacter(nil, false, s.cfg)
	if err != nil {
		return []documents.Document{documents.New(content, docMetadata)}
	}
	out := []documents.Document{}
	for _, chunk := range splitter.SplitText(content) {
		if strings.TrimSpace(chunk) != "" {
			out = append(out, documents.New(chunk, docMetadata))
		}
	}
	return out
}

func (s *HTMLSemanticPreservingSplitter) extractOptionHTMLText(segment string) string {
	segment = removeConfiguredHTMLBlocks(segment, s.options.DenylistTags)
	if len(s.options.AllowlistTags) > 0 {
		segment = collectAllowedHTML(segment, s.options.AllowlistTags)
	}
	segment = s.applyCustomHTMLHandlers(segment)
	if s.options.PreserveLinks {
		segment = replaceHTMLLinks(segment)
	}
	if s.options.PreserveImages {
		segment = replaceHTMLMedia(segment, "img", "image")
	}
	if s.options.PreserveVideos {
		segment = replaceHTMLMedia(segment, "video", "video")
	}
	if s.options.PreserveAudio {
		segment = replaceHTMLMedia(segment, "audio", "audio")
	}
	content := cleanHTMLText(stripHTMLTags(segment))
	if s.options.NormalizeText {
		content = normalizeHTMLSemanticText(content)
	}
	return content
}

func (s *HTMLSemanticPreservingSplitter) applyCustomHTMLHandlers(segment string) string {
	for tag, handler := range s.options.CustomHandlers {
		if handler == nil {
			continue
		}
		tag = strings.ToLower(strings.TrimSpace(tag))
		if tag == "" {
			continue
		}
		segment = replaceHTMLPairedTag(segment, tag, func(attrs map[string]string, inner string) string {
			return handler(attrs, cleanHTMLText(stripHTMLTags(inner)))
		})
		segment = replaceHTMLSelfClosingTag(segment, tag, func(attrs map[string]string) string {
			return handler(attrs, "")
		})
	}
	return segment
}

// CreateDocuments splits raw HTML strings into semantic block documents.
func (s *HTMLSemanticPreservingSplitter) CreateDocuments(texts []string, metadatas []map[string]any) []documents.Document {
	out := []documents.Document{}
	for i, text := range texts {
		metadata := map[string]any(nil)
		if i < len(metadatas) {
			metadata = cloneMetadata(metadatas[i])
		}
		for _, doc := range s.SplitText(text) {
			docMetadata := cloneMetadata(metadata)
			for key, value := range doc.Metadata {
				docMetadata[key] = value
			}
			out = append(out, documents.New(doc.PageContent, docMetadata))
		}
	}
	return out
}

// SplitDocuments splits existing documents and preserves their metadata.
func (s *HTMLSemanticPreservingSplitter) SplitDocuments(docs []documents.Document) []documents.Document {
	texts := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
		metadatas[i] = doc.Metadata
	}
	return s.CreateDocuments(texts, metadatas)
}

func (s *HTMLSemanticPreservingSplitter) semanticTagPattern() string {
	tags := make([]string, 0, len(s.semanticTags)+6)
	for tag := range s.semanticTags {
		tags = append(tags, regexp.QuoteMeta(tag))
	}
	for i := 1; i <= 6; i++ {
		tags = append(tags, "h"+strconv.Itoa(i))
	}
	sort.Strings(tags)
	return strings.Join(tags, "|")
}

type htmlElementToken struct {
	tag  string
	text string
}

func htmlElementTokens(text string) []htmlElementToken {
	return htmlElementTokensWithTags(text, `h[1-6]|p|li|td|th|blockquote`)
}

func htmlElementTokensWithTags(text string, tagPattern string) []htmlElementToken {
	cleaned := removeHTMLBlocks(text, "script")
	cleaned = removeHTMLBlocks(cleaned, "style")
	re := regexp.MustCompile(`(?is)<(` + tagPattern + `)\b[^>]*>(.*?)</\s*(` + tagPattern + `)\s*>`)
	matches := re.FindAllStringSubmatch(cleaned, -1)
	out := make([]htmlElementToken, 0, len(matches))
	for _, match := range matches {
		if len(match) < 4 {
			continue
		}
		openTag := strings.ToLower(match[1])
		closeTag := strings.ToLower(match[3])
		if openTag != closeTag {
			continue
		}
		content := cleanHTMLText(stripHTMLTags(match[2]))
		if content == "" {
			continue
		}
		out = append(out, htmlElementToken{tag: openTag, text: content})
	}
	return out
}

func removeHTMLBlocks(text string, tag string) string {
	re := regexp.MustCompile(`(?is)<` + tag + `\b[^>]*>.*?</\s*` + tag + `\s*>`)
	return re.ReplaceAllString(text, "")
}

func removeConfiguredHTMLBlocks(text string, tags []string) string {
	for _, tag := range tags {
		tag = strings.ToLower(strings.TrimSpace(tag))
		if tag == "" {
			continue
		}
		text = removeHTMLBlocks(text, regexp.QuoteMeta(tag))
		re := regexp.MustCompile(`(?is)<` + regexp.QuoteMeta(tag) + `\b[^>]*/\s*>`)
		text = re.ReplaceAllString(text, "")
	}
	return text
}

func collectAllowedHTML(text string, tags []string) string {
	type indexedPart struct {
		index int
		text  string
	}
	parts := []indexedPart{}
	for _, tag := range tags {
		tag = strings.ToLower(strings.TrimSpace(tag))
		if tag == "" {
			continue
		}
		re := regexp.MustCompile(`(?is)<` + regexp.QuoteMeta(tag) + `\b[^>]*>.*?</\s*` + regexp.QuoteMeta(tag) + `\s*>`)
		for _, match := range re.FindAllStringIndex(text, -1) {
			parts = append(parts, indexedPart{index: match[0], text: text[match[0]:match[1]]})
		}
	}
	sort.SliceStable(parts, func(i, j int) bool { return parts[i].index < parts[j].index })
	out := make([]string, len(parts))
	for i, part := range parts {
		out[i] = part.text
	}
	return strings.Join(out, " ")
}

func replaceHTMLLinks(text string) string {
	return replaceHTMLPairedTag(text, "a", func(attrs map[string]string, inner string) string {
		label := cleanHTMLText(stripHTMLTags(inner))
		href := attrs["href"]
		if href == "" || label == "" {
			return label
		}
		return "[" + label + "](" + href + ")"
	})
}

func replaceHTMLMedia(text string, tag string, label string) string {
	replace := func(attrs map[string]string) string {
		src := attrs["src"]
		if src == "" {
			return ""
		}
		return "![" + label + ":" + src + "](" + src + ")"
	}
	text = replaceHTMLPairedTag(text, tag, func(attrs map[string]string, _ string) string {
		return replace(attrs)
	})
	return replaceHTMLSelfClosingTag(text, tag, replace)
}

func replaceHTMLPairedTag(text string, tag string, replace func(map[string]string, string) string) string {
	re := regexp.MustCompile(`(?is)<` + regexp.QuoteMeta(tag) + `\b([^>]*)>(.*?)</\s*` + regexp.QuoteMeta(tag) + `\s*>`)
	return re.ReplaceAllStringFunc(text, func(match string) string {
		parts := re.FindStringSubmatch(match)
		if len(parts) < 3 {
			return match
		}
		return replace(parseHTMLAttrs(parts[1]), parts[2])
	})
}

func replaceHTMLSelfClosingTag(text string, tag string, replace func(map[string]string) string) string {
	re := regexp.MustCompile(`(?is)<` + regexp.QuoteMeta(tag) + `\b([^>]*)/?>`)
	return re.ReplaceAllStringFunc(text, func(match string) string {
		parts := re.FindStringSubmatch(match)
		if len(parts) < 2 {
			return match
		}
		return replace(parseHTMLAttrs(parts[1]))
	})
}

func parseHTMLAttrs(text string) map[string]string {
	attrs := map[string]string{}
	re := regexp.MustCompile(`(?is)([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*("([^"]*)"|'([^']*)'|([^\s"'>/]+))`)
	for _, match := range re.FindAllStringSubmatch(text, -1) {
		value := ""
		for _, candidate := range match[3:] {
			if candidate != "" {
				value = candidate
				break
			}
		}
		attrs[strings.ToLower(match[1])] = html.UnescapeString(value)
	}
	return attrs
}

func stripHTMLTags(text string) string {
	re := regexp.MustCompile(`(?s)<[^>]+>`)
	return re.ReplaceAllString(text, " ")
}

func cleanHTMLText(text string) string {
	text = html.UnescapeString(text)
	fields := strings.Fields(text)
	return strings.Join(fields, " ")
}

func normalizeHTMLSemanticText(text string) string {
	text = strings.ToLower(text)
	re := regexp.MustCompile(`[^\pL\pN\s]+`)
	text = re.ReplaceAllString(text, " ")
	return strings.Join(strings.Fields(text), " ")
}

func htmlHeaderLevel(tag string) int {
	tag = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(tag)), "#")
	tag = strings.TrimPrefix(tag, "h")
	level, err := strconv.Atoi(tag)
	if err != nil || level <= 0 {
		return 9999
	}
	return level
}
