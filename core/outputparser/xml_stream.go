package outputparser

import (
	"context"
	"regexp"
	"strings"
)

// xmlStreamStartRe matches the first byte sequence that can begin a real XML
// element: a '<' followed by a letter, colon, or underscore. It deliberately
// does not match closing tags ('</'), comments ('<!'), or processing
// instructions ('<?'), mirroring Python's _StreamingParser.xml_start_re.
var xmlStreamStartRe = regexp.MustCompile(`<[a-zA-Z:_]`)

// StreamingXMLParser is an incremental XML parser fed a sequence of string
// chunks. It mirrors Python's langchain_core _StreamingParser: it buffers
// unprocessed text, waits until an opening tag appears, and emits one nested
// map for every completed leaf element. Trailing incomplete markup is held in
// the buffer and discarded by Close, so a truncated final chunk never causes
// an error.
type StreamingXMLParser struct {
	buffer                 string
	stack                  []*xmlFrame
	currentPathHasChildren bool
	xmlStarted             bool
}

// xmlFrame is an open element on the parser stack. text accumulates the raw
// character data that belongs to the element.
type xmlFrame struct {
	name string
	text strings.Builder
}

type xmlTokenKind int

const (
	xmlTokenIncomplete xmlTokenKind = iota // no complete token available yet
	xmlTokenText
	xmlTokenStart
	xmlTokenEnd
	xmlTokenSelfClose
)

type xmlEvent struct {
	kind xmlTokenKind
	name string
	text string
}

// NewStreamingXMLParser creates an empty incremental XML parser.
func NewStreamingXMLParser() *StreamingXMLParser {
	return &StreamingXMLParser{}
}

// NestedXMLElement builds the nested map emitted for a completed leaf element,
// mirroring Python's nested_element helper. path holds the ancestor tag names
// from outermost to innermost after the leaf has been popped; tag is the leaf
// tag name and text is its character data (nil when the element has no text).
func NestedXMLElement(path []string, tag string, text any) map[string]any {
	if len(path) == 0 {
		return map[string]any{tag: text}
	}
	return map[string]any{
		path[0]: []any{NestedXMLElement(path[1:], tag, text)},
	}
}

// Parse feeds a single chunk and returns every leaf element completed by it.
func (p *StreamingXMLParser) Parse(ctx context.Context, chunk string) ([]map[string]any, error) {
	return p.feed(ctx, chunk)
}

// Transform feeds all chunks in order, then finalizes the parser and returns
// every emitted element.
func (p *StreamingXMLParser) Transform(ctx context.Context, chunks []string) ([]map[string]any, error) {
	out := []map[string]any{}
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		emitted, err := p.feed(ctx, chunk)
		if err != nil {
			return nil, err
		}
		out = append(out, emitted...)
	}
	return out, p.Close()
}

// Close finalizes the parser, discarding buffered incomplete markup. It
// mirrors Python's _StreamingParser.close, which suppresses parse errors from
// trailing incomplete XML and never fails.
func (p *StreamingXMLParser) Close() error {
	p.buffer = ""
	p.stack = nil
	p.currentPathHasChildren = false
	p.xmlStarted = false
	return nil
}

// FormatInstructions returns the format instructions shared with XMLParser.
func (p *StreamingXMLParser) FormatInstructions() string {
	return (XMLParser{}).FormatInstructions()
}

func (p *StreamingXMLParser) feed(ctx context.Context, chunk string) ([]map[string]any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p.buffer += chunk
	return p.process(ctx)
}

// process scans complete tokens from the buffer and returns the leaf elements
// they close. Incomplete trailing tokens remain buffered for the next chunk.
func (p *StreamingXMLParser) process(ctx context.Context) ([]map[string]any, error) {
	out := []map[string]any{}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if !p.xmlStarted {
			loc := xmlStreamStartRe.FindStringIndex(p.buffer)
			if loc == nil {
				return out, nil
			}
			p.buffer = p.buffer[loc[0]:]
			p.xmlStarted = true
		}
		event := p.nextToken()
		if event.kind == xmlTokenIncomplete {
			return out, nil
		}
		p.apply(event, &out)
	}
}

// nextToken consumes one complete token from the front of the buffer, skipping
// comments, processing instructions, and declarations. It reports an
// incomplete token (and consumes nothing) when more data is required.
func (p *StreamingXMLParser) nextToken() xmlEvent {
	for {
		buf := p.buffer
		if buf == "" {
			return xmlEvent{kind: xmlTokenIncomplete}
		}
		if buf[0] != '<' {
			idx := strings.IndexByte(buf, '<')
			if idx < 0 {
				return xmlEvent{kind: xmlTokenIncomplete}
			}
			p.buffer = buf[idx:]
			return xmlEvent{kind: xmlTokenText, text: buf[:idx]}
		}

		switch {
		case strings.HasPrefix(buf, "<!--"):
			end := strings.Index(buf, "-->")
			if end < 0 {
				return xmlEvent{kind: xmlTokenIncomplete}
			}
			p.buffer = buf[end+3:]
		case strings.HasPrefix(buf, "<![CDATA["):
			end := strings.Index(buf, "]]>")
			if end < 0 {
				return xmlEvent{kind: xmlTokenIncomplete}
			}
			content := buf[len("<![CDATA["):end]
			p.buffer = buf[end+3:]
			return xmlEvent{kind: xmlTokenText, text: content}
		case strings.HasPrefix(buf, "<?"):
			end := strings.Index(buf, "?>")
			if end < 0 {
				return xmlEvent{kind: xmlTokenIncomplete}
			}
			p.buffer = buf[end+2:]
		case strings.HasPrefix(buf, "<!"):
			end := xmlDeclEnd(buf)
			if end < 0 {
				return xmlEvent{kind: xmlTokenIncomplete}
			}
			p.buffer = buf[end+1:]
		default:
			end := xmlTagEnd(buf)
			if end < 0 {
				return xmlEvent{kind: xmlTokenIncomplete}
			}
			inner := strings.TrimSpace(buf[1:end])
			p.buffer = buf[end+1:]
			if strings.HasPrefix(inner, "/") {
				return xmlEvent{kind: xmlTokenEnd, name: xmlTagName(inner[1:])}
			}
			selfClose := strings.HasSuffix(inner, "/")
			if selfClose {
				inner = strings.TrimSpace(inner[:len(inner)-1])
			}
			name := xmlTagName(inner)
			if selfClose {
				return xmlEvent{kind: xmlTokenSelfClose, name: name}
			}
			return xmlEvent{kind: xmlTokenStart, name: name}
		}
	}
}

func (p *StreamingXMLParser) apply(event xmlEvent, out *[]map[string]any) {
	switch event.kind {
	case xmlTokenText:
		if len(p.stack) > 0 {
			p.stack[len(p.stack)-1].text.WriteString(event.text)
		}
	case xmlTokenStart:
		p.stack = append(p.stack, &xmlFrame{name: event.name})
		p.currentPathHasChildren = false
	case xmlTokenSelfClose:
		p.stack = append(p.stack, &xmlFrame{name: event.name})
		p.currentPathHasChildren = false
		p.endElement(out)
	case xmlTokenEnd:
		p.endElement(out)
	}
}

func (p *StreamingXMLParser) endElement(out *[]map[string]any) {
	if len(p.stack) == 0 {
		return // stray closing tag: treated as trailing junk and ignored
	}
	frame := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]

	var text any
	if frame.text.Len() > 0 {
		text = frame.text.String()
	}
	path := make([]string, len(p.stack))
	for i, open := range p.stack {
		path[i] = open.name
	}
	if !p.currentPathHasChildren {
		*out = append(*out, NestedXMLElement(path, frame.name, text))
	}
	if len(p.stack) > 0 {
		p.currentPathHasChildren = true
	} else {
		p.xmlStarted = false
	}
}

// xmlTagEnd returns the index of the '>' that closes a tag, ignoring '>'
// characters that appear inside quoted attribute values. It returns -1 while
// the tag is still incomplete.
func xmlTagEnd(buf string) int {
	var quote byte
	for i := 1; i < len(buf); i++ {
		c := buf[i]
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

// xmlDeclEnd returns the index of the '>' that closes a <!...> declaration,
// ignoring '>' inside an internal subset ([...]) or a quoted string.
func xmlDeclEnd(buf string) int {
	depth := 0
	var quote byte
	for i := 2; i < len(buf); i++ {
		c := buf[i]
		if quote != 0 {
			if c == quote {
				quote = 0
			}
			continue
		}
		switch c {
		case '"', '\'':
			quote = c
		case '[':
			depth++
		case ']':
			if depth > 0 {
				depth--
			}
		case '>':
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// xmlTagName returns the leading tag name, stopping at whitespace or '/'.
func xmlTagName(s string) string {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ' ', '\t', '\r', '\n', '/':
			return s[:i]
		}
	}
	return s
}
