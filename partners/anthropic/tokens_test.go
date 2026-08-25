package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

func TestGetNumTokensFromMessages(t *testing.T) {
	var got countTokensPayload
	var apiKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiKey = r.Header.Get("x-api-key")
		if r.URL.Path != "/messages/count_tokens" {
			t.Fatalf("path: got %q", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = fmt.Fprint(w, `{"input_tokens":42}`)
	}))
	defer server.Close()

	tool, err := tools.NewFunc(
		"search",
		"searches",
		schema.Object(map[string]schema.Schema{
			"q": schema.String("query"),
		}, "q"),
		func(_ context.Context, _ map[string]any) (tools.Result, error) {
			return tools.Result{Content: "ok"}, nil
		},
	)
	if err != nil {
		t.Fatalf("tool: %v", err)
	}
	base := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
		modelconfig.WithAPIKey("test-key"),
	)
	boundAny, err := base.BindTools([]tools.Tool{tool})
	if err != nil {
		t.Fatalf("bind tools: %v", err)
	}
	model, ok := boundAny.(ChatModel)
	if !ok {
		t.Fatalf("bound model is %T, not ChatModel", boundAny)
	}

	count, err := model.GetNumTokensFromMessages([]messages.Message{
		messages.System("be concise"),
		messages.Human("hello"),
	})
	if err != nil {
		t.Fatalf("count tokens: %v", err)
	}
	if count != 42 {
		t.Fatalf("count: got %d, want 42", count)
	}
	if apiKey != "test-key" {
		t.Fatalf("api key header: got %q", apiKey)
	}
	if got.Model != "claude-test" || got.System != "be concise" {
		t.Fatalf("request: %+v", got)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" || got.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("messages: %+v", got.Messages)
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "search" {
		t.Fatalf("tools: %+v", got.Tools)
	}
}

func TestGetNumTokensFromMessagesNoAPIKey(t *testing.T) {
	model := NewChatModel(modelconfig.WithModel("claude-test"))
	_, err := model.GetNumTokensFromMessages([]messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "API key") {
		t.Fatalf("expected API key error, got %v", err)
	}
}

func TestGetNumTokensFromMessagesHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
		modelconfig.WithAPIKey("bad-key"),
	)
	_, err := model.GetNumTokensFromMessages([]messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "count tokens") {
		t.Fatalf("expected wrapped count tokens error, got %v", err)
	}
}
