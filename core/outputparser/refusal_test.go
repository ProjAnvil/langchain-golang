package outputparser

import (
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

// Ports core/tests/unit_tests/output_parsers/test_refusal.py (#41582).

func refusalMessage() messages.Message {
	m := messages.AI("")
	m.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "I can't do that."}}
	return m
}

func TestRaiseIfRefused(t *testing.T) {
	if err := RaiseIfRefused(refusalMessage()); err == nil {
		t.Fatal("refusal message must raise")
	} else {
		var mre *lcerrors.ModelRefusalError
		if !errors.As(err, &mre) {
			t.Fatalf("want *ModelRefusalError, got %T", err)
		}
	}
	plain := messages.AI("hello")
	plain.ContentBlocks = []messages.ContentBlock{messages.TextBlock{Text: "hello"}}
	if err := RaiseIfRefused(plain); err != nil {
		t.Fatalf("non-refusal message must pass, got %v", err)
	}
}

func TestJSONOutputToolsParserRefused(t *testing.T) {
	p := JSONOutputToolsParser{}
	_, err := p.ParseMessage(refusalMessage())
	var refusalErr *lcerrors.ModelRefusalError
	if !errors.As(err, &refusalErr) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}

func TestJSONOutputKeyToolsParserRefused(t *testing.T) {
	p := JSONOutputKeyToolsParser{KeyName: "anything"}
	_, err := p.ParseMessage(refusalMessage())
	var refusalErr *lcerrors.ModelRefusalError
	if !errors.As(err, &refusalErr) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}
