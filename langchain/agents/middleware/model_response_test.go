package middleware

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestModelResponse(t *testing.T) {
	response := ModelResponse{
		Result: []messages.Message{messages.AI("response")},
		StructuredResponse: map[string]any{
			"temperature": 75.0,
			"condition":   "sunny",
		},
	}

	if len(response.Result) != 1 {
		t.Fatalf("result count mismatch: got %d", len(response.Result))
	}
	if response.Result[0].Content != "response" {
		t.Fatalf("result content mismatch: got %q", response.Result[0].Content)
	}
	wantStructured := map[string]any{"temperature": 75.0, "condition": "sunny"}
	if !reflect.DeepEqual(response.StructuredResponse, wantStructured) {
		t.Fatalf("structured response mismatch: got %#v", response.StructuredResponse)
	}
}

func TestExtendedModelResponseWithCommand(t *testing.T) {
	modelResponse := ModelResponse{
		Result:             []messages.Message{messages.AI("response")},
		StructuredResponse: "structured",
	}
	command := Command{Update: map[string]any{"custom_state": "value"}}
	extended := ExtendedModelResponse{
		ModelResponse: modelResponse,
		Command:       &command,
	}

	if extended.ModelResponse.Result[0].Content != "response" {
		t.Fatalf("model response mismatch: %#v", extended.ModelResponse)
	}
	if extended.ModelResponse.StructuredResponse != "structured" {
		t.Fatalf("structured response mismatch: %#v", extended.ModelResponse.StructuredResponse)
	}
	if extended.Command == nil {
		t.Fatal("expected command")
	}
	if extended.Command.Update["custom_state"] != "value" {
		t.Fatalf("command update mismatch: %#v", extended.Command.Update)
	}
}

func TestExtendedModelResponseWithoutCommand(t *testing.T) {
	extended := ExtendedModelResponse{
		ModelResponse: ModelResponse{
			Result: []messages.Message{messages.AI("response")},
		},
	}

	if extended.Command != nil {
		t.Fatalf("expected nil command, got %#v", extended.Command)
	}
	if extended.ModelResponse.Result[0].Content != "response" {
		t.Fatalf("model response mismatch: %#v", extended.ModelResponse)
	}
}

func TestCommandRejectsUnsupportedControlFields(t *testing.T) {
	command := Command{
		Update: map[string]any{"x": 1},
		Goto:   "end",
	}
	if err := command.ValidateForWrapModelCall(); err == nil {
		t.Fatal("expected unsupported goto error")
	}

	command = Command{
		Update: map[string]any{"x": 1},
		Resume: "resume-token",
	}
	if err := command.ValidateForWrapModelCall(); err == nil {
		t.Fatal("expected unsupported resume error")
	}

	command = Command{
		Update: map[string]any{"x": 1},
		Graph:  "other-graph",
	}
	if err := command.ValidateForWrapModelCall(); err == nil {
		t.Fatal("expected unsupported graph error")
	}
}
