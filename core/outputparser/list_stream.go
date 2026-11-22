package outputparser

import (
	"context"
	"regexp"

	"github.com/projanvil/langchain-golang/core/utils"
)

var (
	numberedListStreamPattern = regexp.MustCompile(`(?m)\d+\.\s([^\n]+)`)
	markdownListStreamPattern = regexp.MustCompile(`(?m)^\s*[-*]\s([^\n]+)$`)
)

// DropLastN returns a lazy iterator that yields every item produced by source
// except its final n items. It is the Go equivalent of Python's droplastn used
// by list streaming parsers: the withheld tail may still be extended by future
// input, so it is not safe to emit until the source is exhausted. A negative n
// is treated as zero (yield everything).
func DropLastN[T any](source utils.Iterator[T], n int) utils.Iterator[T] {
	if n < 0 {
		n = 0
	}
	return &dropLastNIterator[T]{source: source, n: n}
}

type dropLastNIterator[T any] struct {
	source utils.Iterator[T]
	n      int
	buffer []T
	closed bool
}

func (i *dropLastNIterator[T]) Next(ctx context.Context) (T, bool, error) {
	var zero T
	if i.closed {
		return zero, false, nil
	}
	for {
		if err := ctx.Err(); err != nil {
			return zero, false, err
		}
		value, ok, err := i.source.Next(ctx)
		if err != nil {
			return zero, false, err
		}
		if !ok {
			return zero, false, nil // buffered tail is withheld
		}
		i.buffer = append(i.buffer, value)
		if len(i.buffer) > i.n {
			out := i.buffer[0]
			i.buffer = i.buffer[1:]
			return out, true, nil
		}
	}
}

func (i *dropLastNIterator[T]) Close() error {
	if i.closed {
		return nil
	}
	i.closed = true
	i.buffer = nil
	return i.source.Close()
}

// DropLastNChannel is the channel form of DropLastN. It forwards all but the
// final n received values and drops the buffered tail when source closes.
func DropLastNChannel[T any](ctx context.Context, source <-chan T, n int) <-chan T {
	if n < 0 {
		n = 0
	}
	out := make(chan T)
	go func() {
		defer close(out)
		buffer := make([]T, 0, n+1)
		for {
			select {
			case <-ctx.Done():
				return
			case value, ok := <-source:
				if !ok {
					return // buffered tail is withheld
				}
				buffer = append(buffer, value)
				for len(buffer) > n {
					item := buffer[0]
					buffer = buffer[1:]
					select {
					case out <- item:
					case <-ctx.Done():
						return
					}
				}
			}
		}
	}()
	return out
}

// NumberedListParserStream is the streaming counterpart of NumberedListParser.
// Each Parse call yields the list items that are safe to emit (withholding the
// last match via DropLastN), and Flush emits the remaining withheld items.
type NumberedListParserStream struct {
	Pattern *regexp.Regexp
	buffer  string
}

func (p *NumberedListParserStream) pattern() *regexp.Regexp {
	if p.Pattern != nil {
		return p.Pattern
	}
	return numberedListStreamPattern
}

// Parse feeds one chunk and returns the completed list items.
func (p *NumberedListParserStream) Parse(ctx context.Context, chunk string) ([]string, error) {
	return feedListStream(ctx, &p.buffer, p.pattern(), chunk)
}

// Flush finalizes the stream, returning the withheld last items.
func (p *NumberedListParserStream) Flush(ctx context.Context) ([]string, error) {
	return flushListStream(ctx, &p.buffer, p.pattern())
}

// Transform feeds all chunks and returns every list item in order.
func (p *NumberedListParserStream) Transform(ctx context.Context, chunks []string) ([]string, error) {
	return transformListStream(ctx, p, chunks)
}

// FormatInstructions returns the numbered-list format instructions.
func (p *NumberedListParserStream) FormatInstructions() string {
	return (NumberedListParser{}).FormatInstructions()
}

// MarkdownListParserStream is the streaming counterpart of MarkdownListParser
// and shares the DropLastN-based withholding semantics.
type MarkdownListParserStream struct {
	Pattern *regexp.Regexp
	buffer  string
}

func (p *MarkdownListParserStream) pattern() *regexp.Regexp {
	if p.Pattern != nil {
		return p.Pattern
	}
	return markdownListStreamPattern
}

// Parse feeds one chunk and returns the completed list items.
func (p *MarkdownListParserStream) Parse(ctx context.Context, chunk string) ([]string, error) {
	return feedListStream(ctx, &p.buffer, p.pattern(), chunk)
}

// Flush finalizes the stream, returning the withheld last items.
func (p *MarkdownListParserStream) Flush(ctx context.Context) ([]string, error) {
	return flushListStream(ctx, &p.buffer, p.pattern())
}

// Transform feeds all chunks and returns every list item in order.
func (p *MarkdownListParserStream) Transform(ctx context.Context, chunks []string) ([]string, error) {
	return transformListStream(ctx, p, chunks)
}

// FormatInstructions returns the Markdown-list format instructions.
func (p *MarkdownListParserStream) FormatInstructions() string {
	return (MarkdownListParser{}).FormatInstructions()
}

// listStream is the common surface shared by the streaming list parsers.
type listStream interface {
	Parse(ctx context.Context, chunk string) ([]string, error)
	Flush(ctx context.Context) ([]string, error)
}

func transformListStream(ctx context.Context, stream listStream, chunks []string) ([]string, error) {
	out := []string{}
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		items, err := stream.Parse(ctx, chunk)
		if err != nil {
			return nil, err
		}
		out = append(out, items...)
	}
	rest, err := stream.Flush(ctx)
	if err != nil {
		return nil, err
	}
	return append(out, rest...), nil
}

// listItemMatch is a single regex match with its captured item text and the
// end offset of the full match in the current buffer.
type listItemMatch struct {
	text string
	end  int
}

func findListItems(text string, pattern *regexp.Regexp) []listItemMatch {
	indices := pattern.FindAllStringSubmatchIndex(text, -1)
	out := make([]listItemMatch, 0, len(indices))
	for _, m := range indices {
		if len(m) < 4 || m[2] < 0 || m[3] < 0 {
			continue
		}
		out = append(out, listItemMatch{text: text[m[2]:m[3]], end: m[1]})
	}
	return out
}

// feedListStream implements Python's ListOutputParser._transform for one chunk:
// it reparses the accumulated buffer, yields all but the last match, then trims
// the buffer to the end of the last yielded match.
func feedListStream(ctx context.Context, buffer *string, pattern *regexp.Regexp, chunk string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	*buffer += chunk
	matches := findListItems(*buffer, pattern)
	kept, err := utils.CollectIterator(ctx, DropLastN(utils.NewSliceIterator(matches), 1))
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(kept))
	for _, m := range kept {
		out = append(out, m.text)
	}
	if len(kept) > 0 {
		*buffer = (*buffer)[kept[len(kept)-1].end:]
	}
	return out, nil
}

// flushListStream emits every remaining match once the input is complete.
func flushListStream(ctx context.Context, buffer *string, pattern *regexp.Regexp) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	matches := findListItems(*buffer, pattern)
	out := make([]string, 0, len(matches))
	for _, m := range matches {
		out = append(out, m.text)
	}
	*buffer = ""
	return out, nil
}
