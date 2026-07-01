package textsplitters

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestSentenceTextSplitter(t *testing.T) {
	splitter, err := NewSentence(simpleSentenceTokenizer, "\n\n", "english", Config{
		ChunkSize:       28,
		ChunkOverlap:    0,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new sentence splitter: %v", err)
	}

	got, err := splitter.SplitText("One sentence. Two sentence. Three sentence.")
	if err != nil {
		t.Fatalf("split text: %v", err)
	}
	want := []string{"One sentence.\n\nTwo sentence.", "Three sentence."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}
}

func TestSentenceTextSplitterSpansPreserveWhitespace(t *testing.T) {
	splitter, err := NewSentenceSpans(simpleSpanTokenizer, "english", Config{
		ChunkSize:       40,
		ChunkOverlap:    0,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new span splitter: %v", err)
	}

	got, err := splitter.SplitText("One sentence.   Two sentence.")
	if err != nil {
		t.Fatalf("split text: %v", err)
	}
	want := []string{"One sentence.   Two sentence."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}
}

func TestSentenceTextSplitterDocumentsAndErrors(t *testing.T) {
	splitter, err := NewSentence(simpleSentenceTokenizer, "\n\n", "", Config{
		ChunkSize:       5,
		ChunkOverlap:    0,
		AddStartIndex:   true,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new sentence splitter: %v", err)
	}

	docs, err := splitter.SplitDocuments([]documents.Document{
		documents.New("One. Two.", map[string]any{"source": "unit"}),
	})
	if err != nil {
		t.Fatalf("split documents: %v", err)
	}
	if len(docs) != 2 || docs[0].Metadata["source"] != "unit" || docs[0].Metadata["start_index"] != 0 || docs[1].Metadata["start_index"] != 5 {
		t.Fatalf("docs: %#v", docs)
	}

	_, err = NewSentence(nil, "\n\n", "", Config{})
	if err == nil {
		t.Fatal("expected missing tokenizer error")
	}

	bad, err := NewSentenceSpans(func(string, string) ([]SentenceSpan, error) {
		return []SentenceSpan{{Start: 5, End: 2}}, nil
	}, "", Config{})
	if err != nil {
		t.Fatalf("new bad span splitter: %v", err)
	}
	if _, err := bad.SplitText("short"); err == nil {
		t.Fatal("expected invalid span error")
	}

	failing, err := NewSentence(func(string, string) ([]string, error) {
		return nil, errors.New("tokenize failed")
	}, "\n\n", "", Config{})
	if err != nil {
		t.Fatalf("new failing splitter: %v", err)
	}
	if _, err := failing.SplitText("text"); err == nil {
		t.Fatal("expected tokenizer error")
	}
}

func simpleSentenceTokenizer(text string, _ string) ([]string, error) {
	parts := strings.SplitAfter(text, ".")
	out := []string{}
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out, nil
}

func simpleSpanTokenizer(text string, _ string) ([]SentenceSpan, error) {
	parts := strings.SplitAfter(text, ".")
	spans := []SentenceSpan{}
	start := 0
	for _, part := range parts {
		if part == "" {
			continue
		}
		end := start + len(part)
		spans = append(spans, SentenceSpan{Start: start, End: end})
		start = end
	}
	return spans, nil
}
