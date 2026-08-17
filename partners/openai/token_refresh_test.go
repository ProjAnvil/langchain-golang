package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestTokenProviderDefaultTokenURL(t *testing.T) {
	provider := NewTokenProvider(Token{
		AccessToken: "tok",
		ExpiresAt:   time.Now().Add(time.Hour),
	}, "", "client-1")
	if provider.tokenURL != defaultChatGPTTokenURL {
		t.Fatalf("tokenURL = %q, want default %q", provider.tokenURL, defaultChatGPTTokenURL)
	}
}

func TestRefreshKeepsRefreshTokenWhenOmitted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Response carries no refresh_token: the stored one must be kept.
		_, _ = w.Write([]byte(`{"access_token":"new-at","expires_in":3600}`))
	}))
	defer server.Close()

	provider := NewTokenProvider(Token{
		AccessToken:  "old-at",
		RefreshToken: "keep-rt",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, server.URL, "client-1")

	got, err := provider.AccessToken(context.Background())
	if err != nil || got != "new-at" {
		t.Fatalf("AccessToken = %q, %v", got, err)
	}
	if provider.token.RefreshToken != "keep-rt" {
		t.Fatalf("refresh token = %q, want keep-rt", provider.token.RefreshToken)
	}
}

func TestRefreshInvalidJSONResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}))
	defer server.Close()

	provider := NewTokenProvider(Token{
		AccessToken:  "old-at",
		RefreshToken: "old-rt",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, server.URL, "client-1")

	if _, err := provider.AccessToken(context.Background()); err == nil {
		t.Fatal("expected decode error")
	}
}
