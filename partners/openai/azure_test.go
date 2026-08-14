package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestAzureChatModelInvoke(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.RequestURI()
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-az-1",
			"model":"gpt-test",
			"choices":[{"message":{"role":"assistant","content":"hello from azure"}}],
			"usage":{}
		}`))
	}))
	defer server.Close()

	model := NewAzureChatModel(server.URL, "my-deployment", "2024-05-01-preview", "az-key",
		modelconfig.WithModel("gpt-test"),
	)

	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	wantPath := "/openai/deployments/my-deployment/chat/completions?api-version=2024-05-01-preview"
	if gotPath != wantPath {
		t.Fatalf("path = %q, want %q", gotPath, wantPath)
	}
	if resp.Content != "hello from azure" {
		t.Fatalf("content = %q", resp.Content)
	}
	if gotBody["model"] != "gpt-test" {
		t.Fatalf("model = %v", gotBody["model"])
	}
}

func TestAzureChatModelUsesAPIKeyHeader(t *testing.T) {
	var gotAuth string
	var gotAPIKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotAPIKey = r.Header.Get("api-key")
		_, _ = w.Write([]byte(`{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	}))
	defer server.Close()

	model := NewAzureChatModel(server.URL, "dep", "2024-01-01", "secret")
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotAPIKey != "secret" {
		t.Fatalf("api-key = %q, want secret", gotAPIKey)
	}
	if gotAuth != "" {
		t.Fatalf("Authorization = %q, want empty (Azure uses api-key)", gotAuth)
	}
}
