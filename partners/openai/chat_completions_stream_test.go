package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

func TestChatCompletionsStreamText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n"+
				"data: [DONE]\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()

	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	var got []string
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		got = append(got, chunk.Content)
	}
	if len(got) != 2 || got[0] != "Hello" || got[1] != " world" {
		t.Fatalf("streamed content = %#v, want [Hello, ' world']", got)
	}
}

func TestChatCompletionsStreamToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n"+
				"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"hi\\\"}\"}}]}}]}\n\n"+
				"data: [DONE]\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()

	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	var final messages.Message
	for {
		chunk, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		final = chunk
	}
	if len(final.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(final.ToolCalls))
	}
	tc := final.ToolCalls[0]
	if tc.ID != "call_1" || tc.Name != "search" || tc.Args["q"] != "hi" {
		t.Fatalf("tool call = %#v", tc)
	}
}

// TestChatCompletionsStreamToolsNestedFunctionShape locks the STREAM request
// payload to the Chat Completions tools shape: the function descriptor must be
// nested under "function" (regression test for the flat toolSpec leak that
// OpenAI-compatible gateways reject with 400 "missing field `function`").
func TestChatCompletionsStreamToolsNestedFunctionShape(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	tool, err := coretools.FromFunc("search", "search the web", func(ctx context.Context, args struct{ Q string }) (coretools.Result, error) {
		return coretools.Result{Content: args.Q}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	bound, err := model.BindTools([]coretools.Tool{tool})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := bound.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()
	for {
		_, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
	}

	if gotBody["stream"] != true {
		t.Fatalf("stream flag = %v, want true", gotBody["stream"])
	}
	toolsList, ok := gotBody["tools"].([]any)
	if !ok || len(toolsList) != 1 {
		t.Fatalf("tools = %v", gotBody["tools"])
	}
	entry, _ := toolsList[0].(map[string]any)
	if entry["type"] != "function" {
		t.Fatalf("tools[0].type = %v, want function", entry["type"])
	}
	fn, ok := entry["function"].(map[string]any)
	if !ok || fn["name"] != "search" {
		t.Fatalf("stream request tools[0] must nest the descriptor under function (Chat Completions shape), got %v", entry)
	}
	if _, flat := entry["name"]; flat {
		t.Fatalf("flat Responses-style toolSpec leaked into Chat Completions stream request: %v", entry)
	}
}
