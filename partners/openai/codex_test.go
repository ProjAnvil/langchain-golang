package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestTokenProviderReturnsFreshToken(t *testing.T) {
	provider := NewTokenProvider(Token{
		AccessToken:  "tok-a",
		RefreshToken: "ref-a",
		ExpiresAt:    time.Now().Add(time.Hour),
	}, "", "client-1")

	got, err := provider.AccessToken(context.Background())
	if err != nil {
		t.Fatalf("AccessToken: %v", err)
	}
	if got != "tok-a" {
		t.Fatalf("access token = %q, want tok-a", got)
	}
}

func TestTokenProviderRefreshesWhenExpired(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/oauth/token" {
			t.Errorf("path = %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"access_token":"tok-b","refresh_token":"ref-b","expires_in":3600}`))
	}))
	defer server.Close()

	provider := NewTokenProvider(Token{
		AccessToken:  "tok-a",
		RefreshToken: "ref-a",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, server.URL+"/oauth/token", "client-1")

	got, err := provider.AccessToken(context.Background())
	if err != nil {
		t.Fatalf("AccessToken: %v", err)
	}
	if got != "tok-b" {
		t.Fatalf("access token = %q, want tok-b", got)
	}
	if provider.token.RefreshToken != "ref-b" {
		t.Fatalf("refresh token = %q, want ref-b", provider.token.RefreshToken)
	}
}

func TestCodexChatModelInjectsHeaders(t *testing.T) {
	var gotPath, gotAuth, gotAccount string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.RequestURI()
		gotAuth = r.Header.Get("Authorization")
		gotAccount = r.Header.Get("ChatGPT-Account-Id")
		_, _ = w.Write([]byte(`{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"codex ok"}}],"usage":{}}`))
	}))
	defer server.Close()

	provider := NewTokenProvider(Token{
		AccessToken:  "tok-a",
		RefreshToken: "ref-a",
		ExpiresAt:    time.Now().Add(time.Hour),
	}, "", "client-1")

	// Override the Codex base URL to the test server.
	model := NewCodexChatModel("acct-123", provider, modelconfig.WithModel("gpt-test"))
	model.baseURL = server.URL

	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if resp.Content != "codex ok" {
		t.Fatalf("content = %q", resp.Content)
	}
	if gotAuth != "Bearer tok-a" {
		t.Fatalf("Authorization = %q, want 'Bearer tok-a'", gotAuth)
	}
	if gotAccount != "acct-123" {
		t.Fatalf("ChatGPT-Account-Id = %q, want acct-123", gotAccount)
	}
	_ = gotPath
}
