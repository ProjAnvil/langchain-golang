package textsplitters

import (
	"strings"
	"testing"
)

func TestPythonCodeTextSplitter(t *testing.T) {
	splitter, err := NewPythonCode(Config{ChunkSize: 20, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new python splitter: %v", err)
	}
	chunks := splitter.SplitText("class A:\n    pass\n\ndef b():\n    pass")
	if len(chunks) < 2 {
		t.Fatalf("chunks: %#v", chunks)
	}
}

func TestLatexAndHTMLTextSplitters(t *testing.T) {
	latex, err := NewLatex(Config{ChunkSize: 24, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new latex splitter: %v", err)
	}
	if chunks := latex.SplitText("\\section{Intro} hello \\subsection{Next} body"); len(chunks) == 0 {
		t.Fatal("expected latex chunks")
	}

	html, err := NewHTML(Config{ChunkSize: 16, StripWhitespace: true})
	if err != nil {
		t.Fatalf("new html splitter: %v", err)
	}
	if chunks := html.SplitText("<div><p>Hello</p><p>World</p></div>"); len(chunks) == 0 {
		t.Fatal("expected html chunks")
	}
}

func TestJSFrameworkTextSplitter(t *testing.T) {
	splitter := NewJSFramework(nil, Config{ChunkSize: 30, StripWhitespace: true})
	chunks, err := splitter.SplitText("export function App() {\nreturn <Panel><Button>Go</Button></Panel>\n}")
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("chunks: %#v", chunks)
	}
	foundComponent := false
	for _, chunk := range chunks {
		if strings.Contains(chunk, "<Panel") || strings.Contains(chunk, "<Button") {
			foundComponent = true
		}
	}
	if !foundComponent {
		t.Fatalf("component chunks not found: %#v", chunks)
	}
}
