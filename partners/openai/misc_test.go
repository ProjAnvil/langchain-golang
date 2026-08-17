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
)

// TestChatCompletionsSystemRoleMapping covers the system-role branch of
// buildChatCompletionsRequest.
func TestChatCompletionsSystemRoleMapping(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	if _, err := model.Invoke(context.Background(), []messages.Message{
		messages.System("be terse"),
		messages.Human("hi"),
	}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	msgs, ok := gotBody["messages"].([]any)
	if !ok || len(msgs) != 2 {
		t.Fatalf("messages = %v", gotBody["messages"])
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "be terse" {
		t.Fatalf("system message = %#v", first)
	}
}

func TestChatCompletionsInvokeHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
		modelconfig.WithMaxRetries(0),
	).WithChatCompletions()
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err == nil {
		t.Fatal("expected error for 500 response")
	}
}

// TestChatCompletionsEmptyChoices covers toResponsesPayload with no choices:
// the result is an empty AI message rather than an error.
func TestChatCompletionsEmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"x","model":"gpt-test","choices":[],"usage":{}}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.Content != "" || len(resp.ToolCalls) != 0 {
		t.Fatalf("response = %#v, want empty", resp)
	}
}

func TestAzureTextModelInvokeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer server.Close()

	m := NewAzureTextModel(server.URL, "txt-dep", "2024-01-01", "az-key",
		modelconfig.WithMaxRetries(0),
	)
	if _, err := m.Invoke(context.Background(), "prompt"); err == nil {
		t.Fatal("expected error for 500 response")
	}
}

// TestAzureEmbeddingsOrdersByIndex covers the sort branch: out-of-order
// response items must be reordered by their index field.
func TestAzureEmbeddingsOrdersByIndex(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":1,"embedding":[0.9]},{"index":0,"embedding":[0.1]}]}`))
	}))
	defer server.Close()

	e := NewAzureEmbeddings(server.URL, "emb-dep", "2024-01-01", "az-key")
	vectors, err := e.EmbedDocuments(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if len(vectors) != 2 || vectors[0][0] != 0.1 || vectors[1][0] != 0.9 {
		t.Fatalf("vectors not ordered by index: %#v", vectors)
	}
}

func TestRefreshBadTokenURL(t *testing.T) {
	provider := NewTokenProvider(Token{
		AccessToken:  "old-at",
		RefreshToken: "old-rt",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, "://bad-url", "client-1")

	if _, err := provider.AccessToken(context.Background()); err == nil {
		t.Fatal("expected request construction error")
	}
}

func TestRefreshTransportError(t *testing.T) {
	provider := NewTokenProvider(Token{
		AccessToken:  "old-at",
		RefreshToken: "old-rt",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, "http://127.0.0.1:1/oauth/token", "client-1")

	if _, err := provider.AccessToken(context.Background()); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestCodexChatModelInvokeHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadGateway)
	}))
	defer server.Close()

	model := codexTestModel(server, freshCodexProvider())
	model.chat.config.MaxRetries = 0
	_, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err == nil || !strings.Contains(err.Error(), "openai") {
		t.Fatalf("expected provider error, got %v", err)
	}
}
