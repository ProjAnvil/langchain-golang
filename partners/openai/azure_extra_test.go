package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestAzureEmbeddingsEmbedDocuments(t *testing.T) {
	var gotPath, gotAPIKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.RequestURI()
		gotAPIKey = r.Header.Get("api-key")
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}`))
	}))
	defer server.Close()

	e := NewAzureEmbeddings(server.URL, "emb-dep", "2024-01-01", "az-key",
		modelconfig.WithModel("text-embedding-3-small"),
	)
	vectors, err := e.EmbedDocuments(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if len(vectors) != 1 || len(vectors[0]) != 3 {
		t.Fatalf("vectors = %#v", vectors)
	}
	if gotPath != "/openai/deployments/emb-dep/embeddings?api-version=2024-01-01" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotAPIKey != "az-key" {
		t.Fatalf("api-key = %q", gotAPIKey)
	}
}

func TestAzureTextModelInvoke(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.RequestURI()
		_, _ = w.Write([]byte(`{"choices":[{"text":"done"}]}`))
	}))
	defer server.Close()

	m := NewAzureTextModel(server.URL, "txt-dep", "2024-01-01", "az-key",
		modelconfig.WithModel("gpt-3.5-turbo-instruct"),
	)
	out, err := m.Invoke(context.Background(), "prompt")
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if out != "done" {
		t.Fatalf("output = %q, want done", out)
	}
	if gotPath != "/openai/deployments/txt-dep/completions?api-version=2024-01-01" {
		t.Fatalf("path = %q", gotPath)
	}
}

func TestAzureChatModelStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"+
				"data: [DONE]\n\n",
		)
	}))
	defer server.Close()

	m := NewAzureChatModel(server.URL, "dep", "2024-01-01", "az-key")
	stream, err := m.Stream(context.Background(), []messages.Message{messages.Human("x")})
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
	if len(got) != 1 || got[0] != "Hi" {
		t.Fatalf("streamed = %#v, want [Hi]", got)
	}
}

func TestAzureChatModelADTokenAuth(t *testing.T) {
	var gotAuth, gotAPIKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotAPIKey = r.Header.Get("api-key")
		_, _ = w.Write([]byte(`{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	}))
	defer server.Close()

	m := NewAzureChatModelWithADToken(server.URL, "dep", "2024-01-01", "ad-token-123")
	if _, err := m.Invoke(context.Background(), []messages.Message{messages.Human("x")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotAuth != "Bearer ad-token-123" {
		t.Fatalf("Authorization = %q, want 'Bearer ad-token-123'", gotAuth)
	}
	if gotAPIKey != "" {
		t.Fatalf("api-key = %q, want empty", gotAPIKey)
	}
}
