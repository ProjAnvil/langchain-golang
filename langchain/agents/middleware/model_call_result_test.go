package middleware

import (
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// TestNormalizeModelCallResultShortForms mirrors
// factory._normalize_to_model_response (factory.py:177-190).
func TestNormalizeModelCallResultShortForms(t *testing.T) {
	// AIMessage short form → ModelResponse{Result: [msg]}.
	resp, err := NormalizeModelCallResult(messages.AI("hello"))
	if err != nil {
		t.Fatalf("normalize AI message: %v", err)
	}
	if len(resp.Result) != 1 || resp.Result[0].Content != "hello" || resp.StructuredResponse != nil {
		t.Fatalf("AI short form mismatch: %#v", resp)
	}

	// ModelResponse passes through (value and pointer).
	in := ModelResponse{
		Result:             []messages.Message{messages.AI("x")},
		StructuredResponse: "s",
	}
	resp, err = NormalizeModelCallResult(in)
	if err != nil || !reflect.DeepEqual(resp, in) {
		t.Fatalf("ModelResponse value: resp=%#v err=%v", resp, err)
	}
	resp, err = NormalizeModelCallResult(&in)
	if err != nil || !reflect.DeepEqual(resp, in) {
		t.Fatalf("ModelResponse pointer: resp=%#v err=%v", resp, err)
	}

	// ExtendedModelResponse unwraps to its embedded ModelResponse.
	ext := ExtendedModelResponse{ModelResponse: in, Command: &Command{Update: map[string]any{"k": "v"}}}
	resp, err = NormalizeModelCallResult(ext)
	if err != nil || !reflect.DeepEqual(resp, in) {
		t.Fatalf("ExtendedModelResponse value: resp=%#v err=%v", resp, err)
	}
	resp, err = NormalizeModelCallResult(&ext)
	if err != nil || !reflect.DeepEqual(resp, in) {
		t.Fatalf("ExtendedModelResponse pointer: resp=%#v err=%v", resp, err)
	}
}

// TestNormalizeModelCallResultErrors pins the rejected shapes.
func TestNormalizeModelCallResultErrors(t *testing.T) {
	if _, err := NormalizeModelCallResult(messages.Human("nope")); err == nil ||
		!strings.Contains(err.Error(), "must have role") {
		t.Fatalf("non-AI message must error, got %v", err)
	}
	if _, err := NormalizeModelCallResult("nope"); err == nil ||
		!strings.Contains(err.Error(), "unsupported ModelCallResult type") {
		t.Fatalf("string must error, got %v", err)
	}
	if _, err := NormalizeModelCallResult(nil); err == nil ||
		!strings.Contains(err.Error(), "unsupported ModelCallResult type") {
		t.Fatalf("nil must error, got %v", err)
	}
	var nilResp *ModelResponse
	if _, err := NormalizeModelCallResult(nilResp); err == nil ||
		!strings.Contains(err.Error(), "nil *ModelResponse") {
		t.Fatalf("nil *ModelResponse must error, got %v", err)
	}
	var nilExt *ExtendedModelResponse
	if _, err := NormalizeModelCallResult(nilExt); err == nil ||
		!strings.Contains(err.Error(), "nil *ExtendedModelResponse") {
		t.Fatalf("nil *ExtendedModelResponse must error, got %v", err)
	}
}
