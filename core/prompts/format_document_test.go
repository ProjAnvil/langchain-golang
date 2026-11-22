package prompts

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func sfDocPrompt(t *testing.T, template string) PromptTemplate {
	t.Helper()
	prompt, err := NewPromptTemplate("doc", template)
	if err != nil {
		t.Fatalf("NewPromptTemplate error: %v", err)
	}
	return prompt
}

func TestFormatDocument(t *testing.T) {
	prompt := sfDocPrompt(t, "{{.page}}: {{.page_content}}")
	doc := documents.New("This is a joke", map[string]any{"page": 1})
	got, err := FormatDocument(doc, prompt)
	if err != nil {
		t.Fatalf("FormatDocument error: %v", err)
	}
	if got != "1: This is a joke" {
		t.Fatalf("got %q", got)
	}
}

func TestGetDocumentInfo(t *testing.T) {
	prompt := sfDocPrompt(t, "{{.page}}: {{.page_content}}")
	doc := documents.New("body", map[string]any{"page": "7", "author": "anon"})
	info, err := GetDocumentInfo(doc, prompt)
	if err != nil {
		t.Fatalf("GetDocumentInfo error: %v", err)
	}
	if len(info) != 2 {
		t.Fatalf("info should contain exactly the input variables, got %#v", info)
	}
	if info["page_content"] != "body" || info["page"] != "7" {
		t.Fatalf("unexpected info: %#v", info)
	}
	// Extra metadata keys that the prompt does not use are excluded.
	if _, ok := info["author"]; ok {
		t.Fatalf("unexpected extra key in info: %#v", info)
	}
}

func TestGetDocumentInfoMetadataOverridesPageContent(t *testing.T) {
	// Python builds {"page_content": ..., **metadata}, so a metadata key named
	// page_content overrides the document's page content.
	prompt := sfDocPrompt(t, "{{.page_content}}")
	doc := documents.New("original", map[string]any{"page_content": "override"})
	got, err := FormatDocument(doc, prompt)
	if err != nil {
		t.Fatalf("FormatDocument error: %v", err)
	}
	if got != "override" {
		t.Fatalf("got %q, want metadata to win", got)
	}
}

func TestFormatDocumentMissingMetadata(t *testing.T) {
	prompt := sfDocPrompt(t, "{{.page}}: {{.page_content}}")
	doc := documents.New("body", nil) // no "page" metadata
	_, err := FormatDocument(doc, prompt)
	if err == nil {
		t.Fatal("expected an error for missing metadata")
	}
	if !strings.Contains(err.Error(), "missing metadata") || !strings.Contains(err.Error(), "page") {
		t.Fatalf("unexpected error: %q", err.Error())
	}
	// The error is deterministic across calls.
	_, err2 := FormatDocument(doc, prompt)
	if err2 == nil || err2.Error() != err.Error() {
		t.Fatalf("errors are not deterministic: %q vs %q", err, err2)
	}
}

func TestGetDocumentInfoMissingMetadataListsRequired(t *testing.T) {
	prompt := sfDocPrompt(t, "p={{.page}} src={{.source}} c={{.page_content}}")
	doc := documents.New("body", map[string]any{"page": 1})
	_, err := GetDocumentInfo(doc, prompt)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "source") {
		t.Fatalf("error should name the missing key: %q", err.Error())
	}
	// The required-metadata list excludes page_content, mirroring the Python
	// reference; it should list the other declared variables.
	if !strings.Contains(err.Error(), "[page source]") {
		t.Fatalf("error should list the required metadata: %q", err.Error())
	}
}

func TestGetDocumentInfoNilPrompt(t *testing.T) {
	doc := documents.New("body", nil)
	if _, err := GetDocumentInfo(doc, nil); err == nil {
		t.Fatal("expected an error for a nil prompt")
	}
}

func TestFormatDocumentWithOnlyPageContent(t *testing.T) {
	prompt := sfDocPrompt(t, "{{.page_content}}")
	doc := documents.New("just the content", map[string]any{"ignored": true})
	got, err := FormatDocument(doc, prompt)
	if err != nil {
		t.Fatalf("FormatDocument error: %v", err)
	}
	if got != "just the content" {
		t.Fatalf("got %q", got)
	}
}
