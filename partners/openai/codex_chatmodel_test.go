package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/tools"
)

const codexCompletionBody = `{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"codex ok"}}],"usage":{}}`

func freshCodexProvider() *TokenProvider {
	return NewTokenProvider(Token{
		AccessToken:  "tok-a",
		RefreshToken: "ref-a",
		ExpiresAt:    time.Now().Add(time.Hour),
	}, "", "client-1")
}

func codexTestModel(server *httptest.Server, provider *TokenProvider) CodexChatModel {
	model := NewCodexChatModel("acct-123", provider, modelconfig.WithModel("gpt-test"))
	model.baseURL = server.URL
	return model
}

func TestCodexChatModelBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(codexCompletionBody))
	}))
	defer server.Close()

	model := codexTestModel(server, freshCodexProvider())
	outputs, err := model.Batch(context.Background(), [][]messages.Message{
		{messages.Human("one")},
		{messages.Human("two")},
	})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if len(outputs) != 2 || outputs[0].Content != "codex ok" || outputs[1].Content != "codex ok" {
		t.Fatalf("outputs = %#v", outputs)
	}
}

func TestCodexChatModelStreamIsEmpty(t *testing.T) {
	model := NewCodexChatModel("acct-123", freshCodexProvider(), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()
	if _, ok, err := stream.Next(context.Background()); err != nil || ok {
		t.Fatalf("expected empty stream, ok=%v err=%v", ok, err)
	}
}

func TestCodexChatModelMeta(t *testing.T) {
	model := NewCodexChatModel("acct-123", nil, modelconfig.WithModel("gpt-test"))
	if model.LLMType() != "openai-codex" {
		t.Fatalf("LLMType = %q, want openai-codex", model.LLMType())
	}
	if !model.Capabilities().ToolCalling {
		t.Fatalf("Capabilities = %#v, want ToolCalling", model.Capabilities())
	}
	if model.InputSchema() == nil || model.OutputSchema() == nil {
		t.Fatal("schemas must be non-nil")
	}
}

func TestCodexChatModelBindTools(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(codexCompletionBody))
	}))
	defer server.Close()

	tool, err := tools.FromFunc("search", "search the web", func(ctx context.Context, args struct{ Q string }) (tools.Result, error) {
		return tools.Result{Content: args.Q}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	model := codexTestModel(server, freshCodexProvider())
	bound, err := model.BindTools([]tools.Tool{tool})
	if err != nil {
		t.Fatalf("BindTools: %v", err)
	}
	if _, err := bound.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	toolsList, ok := gotBody["tools"].([]any)
	if !ok || len(toolsList) != 1 {
		t.Fatalf("tools = %v", gotBody["tools"])
	}
	entry, _ := toolsList[0].(map[string]any)
	fn, ok := entry["function"].(map[string]any)
	if !ok || fn["name"] != "search" {
		t.Fatalf("tools[0] must nest the descriptor under function, got %v", entry)
	}
}

func TestCodexChatModelNilProviderSendsNoAuthHeader(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(codexCompletionBody))
	}))
	defer server.Close()

	model := codexTestModel(server, nil)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotAuth != "" {
		t.Fatalf("Authorization = %q, want empty with nil provider", gotAuth)
	}
}

func TestCodexChatModelTokenRefreshError(t *testing.T) {
	refreshServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "invalid_grant", http.StatusBadRequest)
	}))
	defer refreshServer.Close()

	provider := NewTokenProvider(Token{
		AccessToken:  "expired",
		RefreshToken: "ref-a",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, refreshServer.URL, "client-1")

	model := NewCodexChatModel("acct-123", provider, modelconfig.WithModel("gpt-test"))
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "oauth refresh") {
		t.Fatalf("expected refresh error, got %v", err)
	}
}
