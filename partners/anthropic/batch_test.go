package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestChatModelBatchPreservesOrder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got requestPayload
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		prompt := got.Messages[0].Content[0].Text
		_, _ = fmt.Fprintf(w, `{
			"id":"msg_%s",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"stop_reason":"end_turn",
			"content":[{"type":"text","text":"echo:%s"}],
			"usage":{"input_tokens":1,"output_tokens":1}
		}`, prompt, prompt)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
	)
	outputs, err := model.Batch(context.Background(), [][]messages.Message{
		{messages.Human("first")},
		{messages.Human("second")},
		{messages.Human("third")},
	})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if len(outputs) != 3 {
		t.Fatalf("outputs: %+v", outputs)
	}
	for i, want := range []string{"echo:first", "echo:second", "echo:third"} {
		if outputs[i].Content != want {
			t.Fatalf("output %d: got %q want %q", i, outputs[i].Content, want)
		}
	}
}

func TestChatModelBatchPropagatesError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
		modelconfig.WithMaxRetries(0),
	)
	_, err := model.Batch(context.Background(), [][]messages.Message{
		{messages.Human("boom")},
	})
	if err == nil {
		t.Fatal("batch should propagate invoke errors")
	}
}

func TestChatModelSchemas(t *testing.T) {
	model := NewChatModel()
	input := model.InputSchema()
	if input["type"] != "array" {
		t.Fatalf("input schema: %+v", input)
	}
	output := model.OutputSchema()
	if output["type"] != "object" {
		t.Fatalf("output schema: %+v", output)
	}
	if _, ok := output["properties"]; !ok {
		t.Fatalf("output schema missing properties: %+v", output)
	}
}
