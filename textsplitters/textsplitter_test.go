package textsplitters

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestCharacterTextSplitter(t *testing.T) {
	splitter, err := NewCharacter(" ", false, Config{
		ChunkSize:       10,
		ChunkOverlap:    3,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}

	got := splitter.SplitText("alpha beta gamma delta")
	want := []string{"alpha beta", "gamma", "delta"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}
}

func TestCharacterTextSplitterPythonFixtures(t *testing.T) {
	tests := []struct {
		name          string
		text          string
		separator     string
		isRegex       bool
		keepSeparator KeepSeparator
		chunkSize     int
		chunkOverlap  int
		want          []string
	}{
		{
			name:         "overlapping words",
			text:         "foo bar baz 123",
			separator:    " ",
			chunkSize:    7,
			chunkOverlap: 3,
			want:         []string{"foo bar", "bar baz", "baz 123"},
		},
		{
			name:         "empty input",
			text:         "",
			separator:    " ",
			chunkSize:    5,
			chunkOverlap: 0,
			want:         []string{},
		},
		{
			name:         "whitespace only",
			text:         " ",
			separator:    " ",
			chunkSize:    5,
			chunkOverlap: 0,
			want:         []string{},
		},
		{
			name:          "literal separator at start",
			text:          "foo.bar.baz.123",
			separator:     ".",
			keepSeparator: KeepSeparatorStart,
			chunkSize:     1,
			chunkOverlap:  0,
			want:          []string{"foo", ".bar", ".baz", ".123"},
		},
		{
			name:          "regex separator at start",
			text:          "foo.bar.baz.123",
			separator:     `\.`,
			isRegex:       true,
			keepSeparator: KeepSeparatorStart,
			chunkSize:     1,
			chunkOverlap:  0,
			want:          []string{"foo", ".bar", ".baz", ".123"},
		},
		{
			name:          "literal separator at end",
			text:          "foo.bar.baz.123",
			separator:     ".",
			keepSeparator: KeepSeparatorEnd,
			chunkSize:     1,
			chunkOverlap:  0,
			want:          []string{"foo.", "bar.", "baz.", "123"},
		},
		{
			name:         "literal separator discarded",
			text:         "foo.bar.baz.123",
			separator:    ".",
			chunkSize:    1,
			chunkOverlap: 0,
			want:         []string{"foo", "bar", "baz", "123"},
		},
		{
			name:         "lookbehind split",
			text:         "abcmiddef",
			separator:    `(?<=mid)`,
			isRegex:      true,
			chunkSize:    5,
			chunkOverlap: 0,
			want:         []string{"abcmid", "def"},
		},
		{
			name:         "lookbehind merged without reinserting",
			text:         "abcmiddef",
			separator:    `(?<=mid)`,
			isRegex:      true,
			chunkSize:    100,
			chunkOverlap: 0,
			want:         []string{"abcmiddef"},
		},
		{
			name:         "empty separator",
			text:         "abc",
			separator:    "",
			chunkSize:    1,
			chunkOverlap: 0,
			want:         []string{"a", "b", "c"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			splitter, err := NewCharacter(tt.separator, tt.isRegex, Config{
				ChunkSize:       tt.chunkSize,
				ChunkOverlap:    tt.chunkOverlap,
				KeepSeparator:   tt.keepSeparator,
				StripWhitespace: true,
			})
			if err != nil {
				t.Fatalf("new splitter: %v", err)
			}
			if got := splitter.SplitText(tt.text); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("chunks: got %#v want %#v", got, tt.want)
			}
		})
	}
}

func TestRecursiveCharacterTextSplitter(t *testing.T) {
	splitter, err := NewRecursiveCharacter(nil, false, Config{
		ChunkSize:       12,
		ChunkOverlap:    0,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}

	got := splitter.SplitText("first paragraph\n\nsecond line")
	want := []string{"first", "paragraph", "second line"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}
}

func TestRecursiveCharacterTextSplitterPythonKeepSeparatorFixtures(t *testing.T) {
	splitter, err := NewRecursiveCharacter([]string{",", "."}, false, Config{
		ChunkSize:       10,
		ChunkOverlap:    0,
		KeepSeparator:   KeepSeparatorStart,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new start splitter: %v", err)
	}
	got := splitter.SplitText("Apple,banana,orange and tomato.")
	want := []string{"Apple", ",banana", ",orange and tomato", "."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("start chunks: got %#v want %#v", got, want)
	}

	splitter, err = NewRecursiveCharacter([]string{",", "."}, false, Config{
		ChunkSize:       10,
		ChunkOverlap:    0,
		KeepSeparator:   KeepSeparatorEnd,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new end splitter: %v", err)
	}
	got = splitter.SplitText("Apple,banana,orange and tomato.")
	want = []string{"Apple,", "banana,", "orange and tomato."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("end chunks: got %#v want %#v", got, want)
	}
}

func TestCreateDocumentsAddsStartIndexAndCopiesMetadata(t *testing.T) {
	splitter, err := NewCharacter(" ", false, Config{
		ChunkSize:       5,
		ChunkOverlap:    0,
		AddStartIndex:   true,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}

	metadata := map[string]any{"source": "unit"}
	docs := splitter.CreateDocuments([]string{"alpha beta"}, []map[string]any{metadata})
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2", len(docs))
	}
	if docs[0].Metadata["source"] != "unit" || docs[0].Metadata["start_index"] != 0 {
		t.Fatalf("first metadata: %#v", docs[0].Metadata)
	}
	if docs[1].Metadata["source"] != "unit" || docs[1].Metadata["start_index"] != 6 {
		t.Fatalf("second metadata: %#v", docs[1].Metadata)
	}
	docs[0].Metadata["source"] = "changed"
	if metadata["source"] != "unit" {
		t.Fatalf("metadata was not copied")
	}
}

func TestRecursiveCharacterCreateDocumentsRepeatedStartIndexPythonFixture(t *testing.T) {
	splitter, err := NewRecursiveCharacter([]string{"\n\n", "\n", " ", ""}, false, Config{
		ChunkSize:       6,
		ChunkOverlap:    0,
		AddStartIndex:   true,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}

	docs := splitter.CreateDocuments([]string{"w1 w1 w1 w1 w1 w1 w1 w1 w1"}, nil)
	got := make([]documents.Document, len(docs))
	copy(got, docs)
	want := []documents.Document{
		documents.New("w1 w1", map[string]any{"start_index": 0}),
		documents.New("w1 w1", map[string]any{"start_index": 6}),
		documents.New("w1 w1", map[string]any{"start_index": 12}),
		documents.New("w1 w1", map[string]any{"start_index": 18}),
		documents.New("w1", map[string]any{"start_index": 24}),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("docs: got %#v want %#v", got, want)
	}
}

func TestSplitDocuments(t *testing.T) {
	splitter, err := NewCharacter(" ", false, Config{
		ChunkSize:       5,
		ChunkOverlap:    0,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}

	docs := splitter.SplitDocuments([]documents.Document{
		documents.New("alpha beta", map[string]any{"source": "doc"}),
	})
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2", len(docs))
	}
	if docs[1].PageContent != "beta" || docs[1].Metadata["source"] != "doc" {
		t.Fatalf("second doc: %#v", docs[1])
	}
}

func TestTokenTextSplitter(t *testing.T) {
	splitter, err := NewToken(nil, Config{
		ChunkSize:       3,
		ChunkOverlap:    1,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new token splitter: %v", err)
	}

	got, err := splitter.SplitText("alpha beta gamma delta epsilon")
	if err != nil {
		t.Fatalf("split text: %v", err)
	}
	want := []string{"alpha beta gamma", "gamma delta epsilon"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}
}

func TestTokenTextSplitterDocumentsAndMetadata(t *testing.T) {
	splitter, err := NewToken(nil, Config{
		ChunkSize:       2,
		ChunkOverlap:    0,
		AddStartIndex:   true,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new token splitter: %v", err)
	}

	metadata := map[string]any{"source": "unit"}
	docs, err := splitter.CreateDocuments([]string{"alpha beta gamma"}, []map[string]any{metadata})
	if err != nil {
		t.Fatalf("create documents: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].PageContent != "alpha beta" || docs[0].Metadata["source"] != "unit" || docs[0].Metadata["start_index"] != 0 {
		t.Fatalf("first doc: %#v", docs[0])
	}
	if docs[1].PageContent != "gamma" || docs[1].Metadata["start_index"] != 11 {
		t.Fatalf("second doc: %#v", docs[1])
	}

	splitDocs, err := splitter.SplitDocuments([]documents.Document{documents.New("one two three", map[string]any{"id": "doc"})})
	if err != nil {
		t.Fatalf("split documents: %v", err)
	}
	if splitDocs[0].Metadata["id"] != "doc" {
		t.Fatalf("metadata: %#v", splitDocs[0].Metadata)
	}
}

func TestTokenTextSplitterCustomTokenizerError(t *testing.T) {
	splitter, err := NewToken(errorTokenizer{}, Config{ChunkSize: 2})
	if err != nil {
		t.Fatalf("new token splitter: %v", err)
	}
	_, err = splitter.SplitText("alpha")
	if err == nil {
		t.Fatal("expected tokenizer error")
	}

	custom, err := NewToken(commaTokenizer{}, Config{ChunkSize: 2, ChunkOverlap: 1})
	if err != nil {
		t.Fatalf("new custom splitter: %v", err)
	}
	got, err := custom.SplitText("a,b,c")
	if err != nil {
		t.Fatalf("split custom: %v", err)
	}
	want := []string{"a,b", "b,c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("custom chunks: got %#v want %#v", got, want)
	}
}

func TestTokenIDTextSplitter(t *testing.T) {
	splitter, err := NewTokenIDs(intTokenizer{}, Config{
		ChunkSize:       3,
		ChunkOverlap:    1,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new token ID splitter: %v", err)
	}

	got, err := splitter.SplitText("1 2 3 4 5")
	if err != nil {
		t.Fatalf("split token IDs: %v", err)
	}
	want := []string{"1 2 3", "3 4 5"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}

	docs, err := splitter.CreateDocuments([]string{"1 2 3 4"}, []map[string]any{{"source": "ids"}})
	if err != nil {
		t.Fatalf("create documents: %v", err)
	}
	if len(docs) != 2 || docs[0].Metadata["source"] != "ids" {
		t.Fatalf("docs: %#v", docs)
	}
}

func TestSplitTextOnTokensPythonFixture(t *testing.T) {
	got, err := SplitTextOnTokens("foo bar baz 123", asciiIDTokenizer{}, 7, 3)
	if err != nil {
		t.Fatalf("split text on tokens: %v", err)
	}
	want := []string{"foo bar", "bar baz", "baz 123"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}
}

func TestSplitTextOnTokensEmptyDecodePythonFixture(t *testing.T) {
	got, err := SplitTextOnTokens("foo bar baz 123", emptyDecodeIDTokenizer{}, 7, 3)
	if err != nil {
		t.Fatalf("split text on tokens: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("chunks: got %#v want empty", got)
	}
}

func TestSplitTextOnTokensInvalidOverlap(t *testing.T) {
	_, err := SplitTextOnTokens("foo", asciiIDTokenizer{}, 3, 3)
	if err == nil || !strings.Contains(err.Error(), "tokens_per_chunk must be greater than chunk_overlap") {
		t.Fatalf("error: %v", err)
	}
}

func TestTokenIDTextSplitterErrors(t *testing.T) {
	if _, err := NewTokenIDs(nil, Config{}); err == nil {
		t.Fatal("expected missing tokenizer error")
	}

	splitter, err := NewTokenIDs(errorIDTokenizer{}, Config{ChunkSize: 2})
	if err != nil {
		t.Fatalf("new token ID splitter: %v", err)
	}
	if _, err := splitter.SplitText("1 2"); err == nil {
		t.Fatal("expected encode error")
	}
}

func TestInvalidConfig(t *testing.T) {
	_, err := NewCharacter(" ", false, Config{ChunkSize: 10, ChunkOverlap: 11})
	if err == nil {
		t.Fatal("expected invalid overlap error")
	}
}

type commaTokenizer struct{}

func (commaTokenizer) Encode(text string) ([]string, error) {
	return splitComma(text), nil
}

func (commaTokenizer) Decode(tokens []string) (string, error) {
	return joinComma(tokens), nil
}

type errorTokenizer struct{}

func (errorTokenizer) Encode(string) ([]string, error) {
	return nil, errors.New("encode failed")
}

func (errorTokenizer) Decode([]string) (string, error) {
	return "", nil
}

type intTokenizer struct{}

func (intTokenizer) EncodeIDs(text string) ([]int, error) {
	fields := strings.Fields(text)
	out := make([]int, 0, len(fields))
	for _, field := range fields {
		switch field {
		case "1":
			out = append(out, 1)
		case "2":
			out = append(out, 2)
		case "3":
			out = append(out, 3)
		case "4":
			out = append(out, 4)
		case "5":
			out = append(out, 5)
		default:
			out = append(out, 0)
		}
	}
	return out, nil
}

func (intTokenizer) DecodeIDs(tokens []int) (string, error) {
	parts := make([]string, 0, len(tokens))
	for _, token := range tokens {
		switch token {
		case 1:
			parts = append(parts, "1")
		case 2:
			parts = append(parts, "2")
		case 3:
			parts = append(parts, "3")
		case 4:
			parts = append(parts, "4")
		case 5:
			parts = append(parts, "5")
		default:
			parts = append(parts, "0")
		}
	}
	return strings.Join(parts, " "), nil
}

type asciiIDTokenizer struct{}

func (asciiIDTokenizer) EncodeIDs(text string) ([]int, error) {
	out := make([]int, 0, len(text))
	for _, r := range text {
		out = append(out, int(r))
	}
	return out, nil
}

func (asciiIDTokenizer) DecodeIDs(tokens []int) (string, error) {
	var builder strings.Builder
	for _, token := range tokens {
		builder.WriteRune(rune(token))
	}
	return builder.String(), nil
}

type emptyDecodeIDTokenizer struct{}

func (emptyDecodeIDTokenizer) EncodeIDs(text string) ([]int, error) {
	return asciiIDTokenizer{}.EncodeIDs(text)
}

func (emptyDecodeIDTokenizer) DecodeIDs([]int) (string, error) {
	return "", nil
}

type errorIDTokenizer struct{}

func (errorIDTokenizer) EncodeIDs(string) ([]int, error) {
	return nil, errors.New("encode ids failed")
}

func (errorIDTokenizer) DecodeIDs([]int) (string, error) {
	return "", nil
}

func splitComma(text string) []string {
	if text == "" {
		return nil
	}
	out := []string{}
	start := 0
	for i, r := range text {
		if r == ',' {
			out = append(out, text[start:i])
			start = i + 1
		}
	}
	out = append(out, text[start:])
	return out
}

func joinComma(tokens []string) string {
	out := ""
	for i, token := range tokens {
		if i > 0 {
			out += ","
		}
		out += token
	}
	return out
}
