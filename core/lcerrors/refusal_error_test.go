package lcerrors

import (
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Ports core/tests/unit_tests/test_exceptions.py::test_model_refusal_error{,_defaults}（#41582）。

func TestModelRefusalError(t *testing.T) {
	raw := messages.AI("")
	raw.ResponseMetadata = map[string]any{"stop_reason": "refusal"}
	err := &ModelRefusalError{Message: "Blocked.", RawMessage: &raw}
	if err.Error() != "Blocked." {
		t.Fatalf("Error() = %q", err.Error())
	}
	if err.IsModelRetryable() {
		t.Fatal("refusal must not be retryable")
	}
	if err.RawMessage == nil || err.RawMessage.ResponseMetadata["stop_reason"] != "refusal" {
		t.Fatal("RawMessage must be preserved")
	}
	if !errors.Is(err, ErrModelRefusal) {
		t.Fatal("errors.Is(err, ErrModelRefusal) must hold")
	}
}

func TestModelRefusalErrorDefaults(t *testing.T) {
	err := &ModelRefusalError{}
	if err.Error() != "Model refused to produce the requested output" {
		t.Fatalf("default message = %q", err.Error())
	}
	if err.RawMessage != nil {
		t.Fatal("RawMessage defaults to nil")
	}
}

func TestModelRefusalFromBlock(t *testing.T) {
	raw := messages.AI("")
	// 无 category 无文本
	e1 := ModelRefusalFromBlock(messages.RefusalBlock{}, raw)
	if e1.Error() != "Model refused to produce the requested output" {
		t.Fatalf("bare: %q", e1.Error())
	}
	// 仅 category
	e2 := ModelRefusalFromBlock(messages.RefusalBlock{Extras: map[string]any{"category": "safety"}}, raw)
	if e2.Error() != "Model refused to produce the requested output (category='safety')" {
		t.Fatalf("category: %q", e2.Error())
	}
	// 全量
	e3 := ModelRefusalFromBlock(messages.RefusalBlock{
		Refusal: "Cannot comply.",
		Extras:  map[string]any{"category": "safety"},
	}, raw)
	want := "Model refused to produce the requested output (category='safety'): Cannot comply."
	if e3.Error() != want {
		t.Fatalf("full: %q want %q", e3.Error(), want)
	}
	if e3.RawMessage == nil {
		t.Fatal("RawMessage must be set")
	}
}
