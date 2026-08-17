package textsplitters

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestSeparatorsForLanguageCoversAllLanguages(t *testing.T) {
	languages := []Language{
		LanguageC, LanguageCPP, LanguageGo, LanguageJava, LanguageJS, LanguageTS,
		LanguagePython, LanguageMarkdown, LanguageLatex, LanguageHTML, LanguageRust,
		LanguageRuby, LanguageR, LanguageElixir, LanguagePHP, LanguageSolidity,
		LanguageCSharp, LanguageCOBOL, LanguageScala, LanguageSwift, LanguageKotlin,
		LanguageLua, LanguageHaskell, LanguagePowerShell, LanguageProto, LanguageRST,
	}
	for _, language := range languages {
		t.Run(string(language), func(t *testing.T) {
			separators, err := SeparatorsForLanguage(language)
			if err != nil {
				t.Fatalf("separators: %v", err)
			}
			if len(separators) == 0 || separators[len(separators)-1] != "" {
				t.Fatalf("separators should end with the empty fallback: %#v", separators)
			}
		})
	}

	if _, err := SeparatorsForLanguage(Language("brainfuck")); err == nil {
		t.Fatal("expected unsupported language error")
	}
	if _, err := NewRecursiveCharacterFromLanguage(Language("brainfuck"), Config{}); err == nil {
		t.Fatal("expected unsupported language error from constructor")
	}
}

func TestNewMarkdownSplitter(t *testing.T) {
	splitter, err := NewMarkdown(Config{ChunkSize: 20, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new markdown splitter: %v", err)
	}
	chunks := splitter.SplitText("# Title\n\nBody text goes here.\n\n## Next\n\nMore text follows.")
	if len(chunks) < 2 {
		t.Fatalf("expected markdown-aware chunks, got %#v", chunks)
	}
	for _, chunk := range chunks {
		if len([]rune(chunk)) > 20 {
			t.Fatalf("chunk exceeds size: %#v", chunk)
		}
	}
}

func TestMarkdownHeaderTildeFenceAndControlChars(t *testing.T) {
	splitter := NewMarkdownHeader([]Header{{Marker: "#", Name: "Header 1"}}, false, true)
	docs := splitter.SplitText("# Title\n~~~\n# not a header\n~~~\nafter")
	if len(docs) != 1 {
		t.Fatalf("docs: %#v", docs)
	}
	want := "~~~\n# not a header\n~~~\nafter"
	if docs[0].PageContent != want || docs[0].Metadata["Header 1"] != "Title" {
		t.Fatalf("doc: %#v", docs[0])
	}

	docs = splitter.SplitText("# T\nva\x01lue")
	if len(docs) != 1 || docs[0].PageContent != "value" {
		t.Fatalf("control characters should be stripped: %#v", docs)
	}
}

func TestMarkdownHeaderNestedStackPop(t *testing.T) {
	splitter := NewMarkdownHeader([]Header{
		{Marker: "#", Name: "H1"},
		{Marker: "##", Name: "H2"},
	}, false, true)
	docs := splitter.SplitText("# A\n## B\nfirst\n# C\nsecond")
	if len(docs) != 2 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].Metadata["H1"] != "A" || docs[0].Metadata["H2"] != "B" {
		t.Fatalf("first metadata: %#v", docs[0].Metadata)
	}
	if !reflect.DeepEqual(docs[1].Metadata, map[string]any{"H1": "C"}) {
		t.Fatalf("higher-level header should pop nested metadata: %#v", docs[1].Metadata)
	}
}

func TestMarkdownHeaderKeepHeadersAggregatesWithFollowingChunk(t *testing.T) {
	splitter := NewMarkdownHeader([]Header{
		{Marker: "#", Name: "H1"},
		{Marker: "##", Name: "H2"},
	}, false, false)
	docs := splitter.SplitText("# Title\n## Sub\nbody")
	if len(docs) != 1 {
		t.Fatalf("docs: %#v", docs)
	}
	want := documents.New("# Title  \n## Sub\nbody", map[string]any{"H1": "Title", "H2": "Sub"})
	if !reflect.DeepEqual(docs[0], want) {
		t.Fatalf("doc:\n got %#v\nwant %#v", docs[0], want)
	}
}

func TestMarkdownHeaderReturnEachLine(t *testing.T) {
	splitter := NewMarkdownHeader([]Header{{Marker: "#", Name: "H1"}}, true, true)
	docs := splitter.SplitText("# T\na\n\nb")
	want := []documents.Document{
		documents.New("a", map[string]any{"H1": "T"}),
		documents.New("b", map[string]any{"H1": "T"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestMarkdownHeaderAggregatesSameMetadata(t *testing.T) {
	splitter := NewMarkdownHeader([]Header{{Marker: "#", Name: "H1"}}, false, true)
	docs := splitter.SplitText("# T\na\n\nb\n# U\nc")
	want := []documents.Document{
		documents.New("a  \nb", map[string]any{"H1": "T"}),
		documents.New("c", map[string]any{"H1": "U"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestMarkdownSyntaxUnclosedCodeFence(t *testing.T) {
	splitter := NewMarkdownSyntax(nil, false, true)
	docs := splitter.SplitText("# H\ntext\n```go\nfmt.Println()")
	if len(docs) != 1 || docs[0].PageContent != "text\n" {
		t.Fatalf("unclosed fence content should be dropped: %#v", docs)
	}
	if docs[0].Metadata["Header 1"] != "H" {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}
}

func TestMarkdownSyntaxCarriageReturnLineEndings(t *testing.T) {
	splitter := NewMarkdownSyntax(nil, true, true)
	docs := splitter.SplitText("a\rb\r\nc")
	want := []documents.Document{
		documents.New("a", map[string]any{}),
		documents.New("b", map[string]any{}),
		documents.New("c", map[string]any{}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestConfigValidationErrors(t *testing.T) {
	if _, err := NewCharacter(" ", false, Config{ChunkSize: -1}); err == nil {
		t.Fatal("expected negative chunk size error")
	}
	if _, err := NewCharacter(" ", false, Config{ChunkSize: 5, ChunkOverlap: -1}); err == nil {
		t.Fatal("expected negative chunk overlap error")
	}
	if _, err := NewRecursiveCharacter(nil, false, Config{ChunkSize: 5, ChunkOverlap: 6}); err == nil {
		t.Fatal("expected overlap greater than size error")
	}
}

func TestConstructorZeroConfigDefaults(t *testing.T) {
	character, err := NewCharacter(",", false, Config{})
	if err != nil {
		t.Fatalf("new character splitter: %v", err)
	}
	if got := character.SplitText(" a , b "); !reflect.DeepEqual(got, []string{"a , b"}) {
		t.Fatalf("character chunks: %#v", got)
	}

	recursive, err := NewRecursiveCharacter(nil, false, Config{})
	if err != nil {
		t.Fatalf("new recursive splitter: %v", err)
	}
	if got := recursive.SplitText("  hello world  "); !reflect.DeepEqual(got, []string{"hello world"}) {
		t.Fatalf("recursive chunks: %#v", got)
	}
}

func TestRecursiveCharacterSplitDocuments(t *testing.T) {
	splitter, err := NewRecursiveCharacter([]string{" "}, false, Config{ChunkSize: 5, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitDocuments([]documents.Document{
		documents.New("alpha beta", map[string]any{"source": "doc"}),
	})
	if len(docs) != 2 || docs[1].PageContent != " beta" || docs[1].Metadata["source"] != "doc" {
		t.Fatalf("docs: %#v", docs)
	}
}

func TestLookbehindRegexSeparator(t *testing.T) {
	splitter, err := NewRecursiveCharacter([]string{`(?<=,)`, ""}, true, Config{ChunkSize: 8, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	got := splitter.SplitText("aa,bb,cc")
	if !reflect.DeepEqual(got, []string{"aa,bb,cc"}) {
		t.Fatalf("chunks: %#v", got)
	}

	narrow, err := NewRecursiveCharacter([]string{`(?<=,)`, ""}, true, Config{ChunkSize: 4, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	got = narrow.SplitText("aa,bb,cc")
	if !reflect.DeepEqual(got, []string{"aa,", "bb,", "cc"}) {
		t.Fatalf("chunks: %#v", got)
	}
}

func TestLookaheadRegexSeparator(t *testing.T) {
	splitter, err := NewRecursiveCharacter([]string{`(?=,)`, ""}, true, Config{ChunkSize: 4, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	got := splitter.SplitText("aa,bb,cc")
	if !reflect.DeepEqual(got, []string{"aa", ",bb", ",cc"}) {
		t.Fatalf("chunks: %#v", got)
	}
}

func TestCreateDocumentsStartIndexEdgeCases(t *testing.T) {
	splitter, err := NewCharacter(`\s+`, true, Config{
		ChunkSize:       12,
		ChunkOverlap:    2,
		AddStartIndex:   true,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.CreateDocuments([]string{"alpha beta gamma"}, nil)
	if len(docs) != 2 {
		t.Fatalf("docs: %#v", docs)
	}
	// The first chunk is joined with the literal regex separator and therefore
	// cannot be found in the source text.
	if docs[0].Metadata["start_index"] != -1 {
		t.Fatalf("first start_index: %#v", docs[0].Metadata)
	}
	if docs[1].Metadata["start_index"] != 11 {
		t.Fatalf("second start_index: %#v", docs[1].Metadata)
	}
}

func TestSplitTextOnTokensTokenizerErrors(t *testing.T) {
	if _, err := SplitTextOnTokens("text", nil, 5, 1); err == nil {
		t.Fatal("expected missing tokenizer error")
	}
	if _, err := SplitTextOnTokens("text", errorIDTokenizer{}, 5, 1); err == nil {
		t.Fatal("expected encode error")
	}
}

func TestTokenConstructorsValidation(t *testing.T) {
	if _, err := NewToken(nil, Config{}); err != nil {
		t.Fatalf("zero config should default and succeed: %v", err)
	}
	if _, err := NewToken(nil, Config{ChunkSize: 5, ChunkOverlap: 6}); err == nil {
		t.Fatal("expected invalid config error")
	}
	if _, err := NewTokenIDs(intTokenizer{}, Config{}); err != nil {
		t.Fatalf("zero config should default and succeed: %v", err)
	}
	if _, err := NewTokenIDs(intTokenizer{}, Config{ChunkSize: 5, ChunkOverlap: 6}); err == nil {
		t.Fatal("expected invalid config error")
	}
}

func TestTokenTextSplitterEdgeCases(t *testing.T) {
	splitter, err := NewToken(nil, Config{ChunkSize: 2, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	got, err := splitter.SplitText("   ")
	if err != nil || len(got) != 0 {
		t.Fatalf("empty input should produce no chunks: %#v, %v", got, err)
	}

	equalOverlap, err := NewToken(nil, Config{ChunkSize: 2, ChunkOverlap: 2, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	got, err = equalOverlap.SplitText("a b c d")
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"a b", "c d"}) {
		t.Fatalf("chunks: %#v", got)
	}

	decoding, err := NewToken(decodeErrorTokenizer{}, Config{ChunkSize: 2})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	if _, err := decoding.SplitText("a b c"); err == nil {
		t.Fatal("expected decode error")
	}
	if _, err := decoding.CreateDocuments([]string{"a b c"}, nil); err == nil {
		t.Fatal("expected decode error from CreateDocuments")
	}
}

func TestTokenTextSplitterStartIndexNotFound(t *testing.T) {
	splitter, err := NewToken(upperTokenizer{}, Config{ChunkSize: 2, ChunkOverlap: 1, AddStartIndex: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs, err := splitter.CreateDocuments([]string{"alpha beta gamma"}, nil)
	if err != nil {
		t.Fatalf("create documents: %v", err)
	}
	if len(docs) == 0 || docs[0].Metadata["start_index"] != -1 {
		t.Fatalf("uppercased chunk should not be found: %#v", docs)
	}
}

func TestTokenIDSplitterEdgeCases(t *testing.T) {
	splitter, err := NewTokenIDs(intTokenizer{}, Config{ChunkSize: 2, ChunkOverlap: 2, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	got, err := splitter.SplitText("1 2 3 4")
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"1 2", "3 4"}) {
		t.Fatalf("chunks: %#v", got)
	}

	got, err = splitter.SplitText("")
	if err != nil || len(got) != 0 {
		t.Fatalf("empty input should produce no chunks: %#v, %v", got, err)
	}

	decoding, err := NewTokenIDs(decodeErrorIDTokenizer{}, Config{ChunkSize: 2})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	if _, err := decoding.SplitText("1 2 3"); err == nil {
		t.Fatal("expected decode error")
	}
	if _, err := decoding.CreateDocuments([]string{"1 2 3"}, nil); err == nil {
		t.Fatal("expected decode error from CreateDocuments")
	}
}

func TestTokenIDSplitDocumentsWithStartIndex(t *testing.T) {
	splitter, err := NewTokenIDs(intTokenizer{}, Config{
		ChunkSize:       3,
		ChunkOverlap:    1,
		AddStartIndex:   true,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs, err := splitter.SplitDocuments([]documents.Document{
		documents.New("1 2 3 4 5", map[string]any{"source": "ids"}),
	})
	if err != nil {
		t.Fatalf("split documents: %v", err)
	}
	if len(docs) != 2 || docs[0].Metadata["start_index"] != 0 || docs[1].Metadata["start_index"] != 4 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].Metadata["source"] != "ids" {
		t.Fatalf("metadata not preserved: %#v", docs[0])
	}

	unfindable, err := NewTokenIDs(intTokenizer{}, Config{ChunkSize: 2, ChunkOverlap: 1, AddStartIndex: true})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs, err = unfindable.SplitDocuments([]documents.Document{documents.New("a b c", nil)})
	if err != nil {
		t.Fatalf("split documents: %v", err)
	}
	if len(docs) == 0 || docs[0].Metadata["start_index"] != -1 {
		t.Fatalf("decoded digits should not be found in letters: %#v", docs)
	}
}

func TestJSFrameworkDefaultsAndErrors(t *testing.T) {
	splitter := NewJSFramework(nil, Config{})
	if splitter.Config.ChunkSize != 2000 {
		t.Fatalf("default chunk size: %d", splitter.Config.ChunkSize)
	}
	chunks, err := splitter.SplitText("function App() {\n  return <div><span>one</span><span>two</span></div>\n}")
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if len(chunks) == 0 || !strings.Contains(strings.Join(chunks, ""), "span") {
		t.Fatalf("chunks: %#v", chunks)
	}

	bad := NewJSFramework(nil, Config{ChunkSize: 5, ChunkOverlap: 10})
	if _, err := bad.SplitText("x"); err == nil {
		t.Fatal("expected invalid config error")
	}
}

func TestRecursiveJSONDefaultsAndScalarLeaf(t *testing.T) {
	splitter := NewRecursiveJSON(0, 0)
	if splitter.MaxChunkSize != 2000 || splitter.MinChunkSize != 1800 {
		t.Fatalf("defaults: %#v", splitter)
	}
	if chunks := splitter.SplitJSON(map[string]any{}, false); len(chunks) != 0 {
		t.Fatalf("empty input should produce no chunks: %#v", chunks)
	}

	small := NewRecursiveJSON(10, 1)
	value := strings.Repeat("x", 50)
	chunks := small.SplitJSON(map[string]any{"a": value}, false)
	if len(chunks) != 2 || chunks[1]["a"] != value {
		t.Fatalf("oversized scalar leaf: %#v", chunks)
	}
}

func TestRecursiveJSONMarshalErrors(t *testing.T) {
	splitter := NewRecursiveJSON(0, 0)
	data := map[string]any{"bad": math.NaN()}
	if _, err := splitter.SplitText(data, false); err == nil {
		t.Fatal("expected marshal error for NaN")
	}
	if _, err := splitter.CreateDocuments([]map[string]any{data}, false, nil); err == nil {
		t.Fatal("expected marshal error from CreateDocuments")
	}
}

func TestSentenceConstructorDefaultsAndValidation(t *testing.T) {
	splitter, err := NewSentence(simpleSentenceTokenizer, "", "english", Config{
		ChunkSize:       28,
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
		t.Fatalf("default separator chunks: got %#v want %#v", got, want)
	}

	if _, err := NewSentence(simpleSentenceTokenizer, "", "", Config{ChunkSize: 5, ChunkOverlap: 6}); err == nil {
		t.Fatal("expected invalid config error")
	}
	if _, err := NewSentenceSpans(nil, "", Config{}); err == nil {
		t.Fatal("expected missing span tokenizer error")
	}
	if _, err := NewSentenceSpans(simpleSpanTokenizer, "", Config{ChunkSize: 5, ChunkOverlap: 6}); err == nil {
		t.Fatal("expected invalid config error")
	}

	var gotLanguage string
	spans, err := NewSentenceSpans(func(text string, language string) ([]SentenceSpan, error) {
		gotLanguage = language
		return []SentenceSpan{{Start: 0, End: len(text)}}, nil
	}, "", Config{ChunkSize: 100})
	if err != nil {
		t.Fatalf("new span splitter: %v", err)
	}
	if _, err := spans.SplitText("hello"); err != nil {
		t.Fatalf("split: %v", err)
	}
	if gotLanguage != "english" {
		t.Fatalf("default language: %q", gotLanguage)
	}
}

func TestSentenceCreateDocumentsErrorAndStartIndex(t *testing.T) {
	failing, err := NewSentence(func(string, string) ([]string, error) {
		return nil, errors.New("tokenize failed")
	}, "", "", Config{})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	if _, err := failing.CreateDocuments([]string{"x"}, nil); err == nil {
		t.Fatal("expected tokenizer error from CreateDocuments")
	}

	splitter, err := NewSentence(simpleSentenceTokenizer, "\n\n", "", Config{
		ChunkSize:       15,
		ChunkOverlap:    3,
		AddStartIndex:   true,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs, err := splitter.CreateDocuments([]string{"One. Two. Three."}, nil)
	if err != nil {
		t.Fatalf("create documents: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].Metadata["start_index"] != -1 || docs[1].Metadata["start_index"] != 10 {
		t.Fatalf("start indexes: %#v", docs)
	}
}

func TestSentenceSpanOverlappingAndTokenizerError(t *testing.T) {
	overlapping, err := NewSentenceSpans(func(string, string) ([]SentenceSpan, error) {
		return []SentenceSpan{{Start: 0, End: 5}, {Start: 3, End: 6}}, nil
	}, "", Config{})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	if _, err := overlapping.SplitText("abcdef"); err == nil || !strings.Contains(err.Error(), "overlapping") {
		t.Fatalf("expected overlapping spans error, got %v", err)
	}

	failing, err := NewSentenceSpans(func(string, string) ([]SentenceSpan, error) {
		return nil, errors.New("spans failed")
	}, "", Config{})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	if _, err := failing.SplitText("text"); err == nil {
		t.Fatal("expected span tokenizer error")
	}
}

func TestHTMLSemanticConstructorValidation(t *testing.T) {
	if _, err := NewHTMLSemanticPreserving(nil, nil, Config{ChunkSize: 5, ChunkOverlap: 6}); err == nil {
		t.Fatal("expected invalid config error")
	}
	splitter, err := NewHTMLSemanticPreserving(nil, nil, Config{})
	if err != nil {
		t.Fatalf("zero config should default and succeed: %v", err)
	}
	docs := splitter.SplitText("<p>  spaced   text  </p>")
	if len(docs) != 1 || docs[0].PageContent != "spaced text" {
		t.Fatalf("docs: %#v", docs)
	}
}

func TestHTMLSemanticFallbackNoTokens(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving(nil, []string{"p"}, Config{ChunkSize: 100})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitText("<div>hello there</div>")
	if len(docs) != 1 || docs[0].PageContent != "hello there" || len(docs[0].Metadata) != 0 {
		t.Fatalf("fallback docs: %#v", docs)
	}
}

func TestHTMLSemanticSkipsWhitespaceOnlyChunks(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving(nil, []string{"pre"}, Config{ChunkSize: 5, ChunkOverlap: 0})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitText("<pre>aaaa  bbbb cccc</pre>")
	for _, doc := range docs {
		if strings.TrimSpace(doc.PageContent) == "" {
			t.Fatalf("whitespace-only chunk emitted: %#v", docs)
		}
	}
	if len(docs) < 2 {
		t.Fatalf("expected the oversized block to be split: %#v", docs)
	}
}

func TestHTMLHeaderMismatchedAndEmptyTags(t *testing.T) {
	splitter := NewHTMLHeader([]Header{{Marker: "h1", Name: "Header 1"}}, false)
	docs := splitter.SplitText(`<h1>kept</h2><p>body</p><h1></h1>`)
	if len(docs) != 1 || docs[0].PageContent != "body" {
		t.Fatalf("mismatched and empty tags should be skipped: %#v", docs)
	}
}

func TestHTMLHeaderNonNumericMarker(t *testing.T) {
	splitter := NewHTMLHeader([]Header{
		{Marker: "hx", Name: "Weird"},
		{Marker: "h1", Name: "Header 1"},
	}, false)
	docs := splitter.SplitText(`<h1>Title</h1><p>text</p>`)
	if len(docs) != 2 || docs[0].Metadata["Header 1"] != "Title" {
		t.Fatalf("docs: %#v", docs)
	}
}

func TestHTMLSemanticOptionsHeaderLevelClearing(t *testing.T) {
	splitter, err := NewHTMLSemanticPreservingWithOptions([]Header{
		{Marker: "h1", Name: "Header 1"},
		{Marker: "h2", Name: "Header 2"},
	}, Config{ChunkSize: 200}, HTMLSemanticOptions{})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitText(`<h1>A</h1><h2>B</h2><p>first</p><h1>C</h1><p>second</p>`)
	if len(docs) != 2 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].Metadata["Header 1"] != "A" || docs[0].Metadata["Header 2"] != "B" {
		t.Fatalf("first metadata: %#v", docs[0].Metadata)
	}
	if docs[1].Metadata["Header 1"] != "C" {
		t.Fatalf("second metadata: %#v", docs[1].Metadata)
	}
	if _, ok := docs[1].Metadata["Header 2"]; ok {
		t.Fatalf("h2 metadata should be cleared by a new h1: %#v", docs[1].Metadata)
	}
}

func TestHTMLSemanticOptionsEmptyOutputFallback(t *testing.T) {
	splitter, err := NewHTMLSemanticPreservingWithOptions(nil, Config{ChunkSize: 100}, HTMLSemanticOptions{
		ExternalMetadata: map[string]any{"source": "ext"},
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitText(`<h1>Only Title</h1>`)
	if len(docs) != 1 || docs[0].PageContent != "Only Title" || docs[0].Metadata["source"] != "ext" {
		t.Fatalf("fallback docs: %#v", docs)
	}
}

func TestHTMLSemanticOptionsChunkSplitting(t *testing.T) {
	splitter, err := NewHTMLSemanticPreservingWithOptions(nil, Config{ChunkSize: 6, ChunkOverlap: 0}, HTMLSemanticOptions{})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitText(`<p>aaaa bbbb cccc</p>`)
	if len(docs) < 2 {
		t.Fatalf("expected the oversized segment to be split: %#v", docs)
	}
	for _, doc := range docs {
		if len([]rune(doc.PageContent)) > 6 {
			t.Fatalf("chunk exceeds size: %#v", doc.PageContent)
		}
	}
}

func TestHTMLSemanticOptionsCustomHandlers(t *testing.T) {
	splitter, err := NewHTMLSemanticPreservingWithOptions(nil, Config{ChunkSize: 100}, HTMLSemanticOptions{
		CustomHandlers: map[string]HTMLCustomHandler{
			" ":    func(map[string]string, string) string { return "ignored" },
			"skip": nil,
			"br":   func(map[string]string, string) string { return " [break] " },
		},
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitText(`<p>a<br/>b</p>`)
	if len(docs) != 1 || docs[0].PageContent != "a [break] b" {
		t.Fatalf("custom handler docs: %#v", docs)
	}
}

func TestHTMLSemanticOptionsAllowAndDenyLists(t *testing.T) {
	allow, err := NewHTMLSemanticPreservingWithOptions(nil, Config{ChunkSize: 100}, HTMLSemanticOptions{
		AllowlistTags: []string{"", "p"},
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := allow.SplitText(`<div><p>one</p><span>mid</span><p>two</p></div>`)
	if len(docs) != 1 || docs[0].PageContent != "one two" {
		t.Fatalf("allowlist docs: %#v", docs)
	}

	deny, err := NewHTMLSemanticPreservingWithOptions(nil, Config{ChunkSize: 100}, HTMLSemanticOptions{
		DenylistTags: []string{"", "div"},
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs = deny.SplitText(`<div>drop</div><p>keep</p>`)
	if len(docs) != 1 || docs[0].PageContent != "keep" {
		t.Fatalf("denylist docs: %#v", docs)
	}
}

func TestHTMLSemanticOptionsLinksAndMediaEdgeCases(t *testing.T) {
	splitter, err := NewHTMLSemanticPreservingWithOptions(nil, Config{ChunkSize: 500}, HTMLSemanticOptions{
		PreserveLinks:  true,
		PreserveImages: true,
		PreserveVideos: true,
		PreserveAudio:  true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitText(`<p><a>label</a> <a href="u">go</a> <img> <img src="i.png"/> <video src="v.mp4"></video> <audio src="a.mp3"/></p>`)
	if len(docs) != 1 {
		t.Fatalf("docs: %#v", docs)
	}
	content := docs[0].PageContent
	for _, want := range []string{"label", "[go](u)", "![image:i.png](i.png)", "![video:v.mp4](v.mp4)", "![audio:a.mp3](a.mp3)"} {
		if !strings.Contains(content, want) {
			t.Fatalf("content %q missing %q", content, want)
		}
	}
}

func TestHTMLSemanticSplitDocuments(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving([]Header{{Marker: "h1", Name: "Header 1"}}, []string{"p"}, Config{ChunkSize: 100})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}
	docs := splitter.SplitDocuments([]documents.Document{
		documents.New("<h1>T</h1><p>body</p>", map[string]any{"source": "s"}),
	})
	if len(docs) != 2 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[1].Metadata["source"] != "s" || docs[1].Metadata["Header 1"] != "T" {
		t.Fatalf("metadata not merged: %#v", docs[1])
	}
}

func TestHTMLSectionNoHeadersAndEmptyMarker(t *testing.T) {
	splitter := NewHTMLSection([]Header{
		{Marker: " ", Name: "skipped"},
		{Marker: "h1", Name: "Header 1"},
	})
	if docs := splitter.SplitText("<div><p>no headers here</p></div>"); docs != nil {
		t.Fatalf("expected nil without header tags: %#v", docs)
	}
}

func TestHTMLSectionSplitDocuments(t *testing.T) {
	splitter := NewHTMLSection([]Header{{Marker: "h1", Name: "Header 1"}})
	docs := splitter.SplitDocuments([]documents.Document{
		documents.New("<body><h1>A</h1><p>x</p></body>", map[string]any{"source": "s"}),
	})
	if len(docs) != 1 || docs[0].Metadata["source"] != "s" || docs[0].Metadata["Header 1"] != "A" {
		t.Fatalf("docs: %#v", docs)
	}
}

func TestHTMLSectionNestedHeaderElement(t *testing.T) {
	splitter := NewHTMLSection([]Header{{Marker: "h1", Name: "Header 1"}})
	docs := splitter.SplitText(`<body><h1>A <b>B</b></h1><p>x</p></body>`)
	if len(docs) != 1 || docs[0].Metadata["Header 1"] != "A B" {
		t.Fatalf("nested header text should be flattened: %#v", docs)
	}
}

func TestHTMLSectionTokenizerEdgeCases(t *testing.T) {
	splitter := NewHTMLSection([]Header{{Marker: "h1", Name: "Header 1"}})
	tests := []struct {
		name     string
		html     string
		contains []string
	}{
		{name: "comment becomes text", html: `<body><h1>H</h1><!-- note --><p>x</p></body>`, contains: []string{"note", "x"}},
		{name: "unterminated comment", html: `<body><h1>H</h1><p>x</p></body><!-- tail`, contains: []string{"x", "tail"}},
		{name: "cdata", html: `<body><h1>H</h1><p><![CDATA[raw data]]></p></body>`, contains: []string{"raw data"}},
		{name: "unterminated cdata", html: `<body><h1>H</h1><![CDATA[never closed`, contains: []string{"never closed"}},
		{name: "doctype skipped", html: `<!DOCTYPE html><body><h1>H</h1><p>x</p></body>`, contains: []string{"x"}},
		{name: "unterminated processing instruction", html: `<body><h1>H</h1><p>x</p></body><?php echo`, contains: []string{"x"}},
		{name: "unterminated close tag", html: `<body><h1>H</h1><p>x</p></body></h1`, contains: []string{"x"}},
		{name: "unterminated open tag", html: `<body><h1>H</h1><p>x</p></body><div`, contains: []string{"x"}},
		{name: "self closing with attributes", html: `<body><h1>H</h1><br class="x" /><p>y</p></body>`, contains: []string{"y"}},
		{name: "script raw text", html: `<body><h1>H</h1><script>if (a < b) { c(); }</script><p>y</p></body>`, contains: []string{"if (a < b) { c(); }", "y"}},
		{name: "unterminated script", html: `<body><h1>H</h1><script>var a = 1;`, contains: []string{"var a = 1;"}},
		{name: "script close missing bracket", html: `<body><h1>H</h1><script>x</script`, contains: []string{"x"}},
		{name: "empty script body", html: `<body><h1>H</h1><script></script><p>y</p></body>`, contains: []string{"y"}},
		{name: "pre preserves whitespace", html: `<body><pre>keep   spaces</pre><h1>H</h1><p>y</p></body>`, contains: []string{"keep   spaces"}},
		{name: "whitespace only text node", html: `<body><h1>A</h1>   <h1>B</h1><p>z</p></body>`, contains: []string{"z"}},
		{name: "style without font size", html: `<body><span style="color:red">plain</span><h1>H</h1><p>y</p></body>`, contains: []string{"plain"}},
		{name: "font size without px", html: `<body><span style="font-size: large">nopx</span><h1>H</h1><p>y</p></body>`, contains: []string{"nopx"}},
		{name: "font size not numeric", html: `<body><span style="font-size: abcpx">bad</span><h1>H</h1><p>y</p></body>`, contains: []string{"bad"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			docs := splitter.SplitText(tt.html)
			joined := ""
			for _, doc := range docs {
				joined += "|" + doc.PageContent
			}
			for _, want := range tt.contains {
				if !strings.Contains(joined, want) {
					t.Fatalf("content %q missing %q (docs: %#v)", joined, want, docs)
				}
			}
		})
	}
}

type decodeErrorTokenizer struct{}

func (decodeErrorTokenizer) Encode(text string) ([]string, error) {
	return strings.Fields(text), nil
}

func (decodeErrorTokenizer) Decode([]string) (string, error) {
	return "", errors.New("decode failed")
}

type upperTokenizer struct{}

func (upperTokenizer) Encode(text string) ([]string, error) {
	return strings.Fields(text), nil
}

func (upperTokenizer) Decode(tokens []string) (string, error) {
	return strings.ToUpper(strings.Join(tokens, " ")), nil
}

type decodeErrorIDTokenizer struct{}

func (decodeErrorIDTokenizer) EncodeIDs(string) ([]int, error) {
	return []int{1, 2, 3}, nil
}

func (decodeErrorIDTokenizer) DecodeIDs([]int) (string, error) {
	return "", errors.New("decode ids failed")
}
