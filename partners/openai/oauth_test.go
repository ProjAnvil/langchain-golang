package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestTokenIsExpiredUsesSkew(t *testing.T) {
	token := Token{
		AccessToken:  "x",
		RefreshToken: "y",
		ExpiresAt:    time.Now().Add(time.Minute),
	}
	if !token.IsExpired(5 * time.Minute) {
		t.Fatal("token with 1min left should be expired under a 5min skew")
	}
	if token.IsExpired(0) {
		t.Fatal("token with 1min left should not be expired under zero skew")
	}
}

func TestRefreshFailurePreservesStoredToken(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "invalid_grant", http.StatusBadRequest)
	}))
	defer server.Close()

	provider := NewTokenProvider(Token{
		AccessToken:  "keep-at",
		RefreshToken: "keep-rt",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, server.URL+"/oauth/token", "client-1")

	_, err := provider.AccessToken(context.Background())
	if err == nil {
		t.Fatal("expected refresh error")
	}
	// The stored token must be preserved so a follow-up login is all that's
	// needed (mirroring Python's test_refresh_failure_preserves_stored_token).
	if provider.token.AccessToken != "keep-at" || provider.token.RefreshToken != "keep-rt" {
		t.Fatalf("token was mutated: %#v", provider.token)
	}
}

func TestRefreshMissingAccessToken(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"refresh_token":"r2","expires_in":3600}`))
	}))
	defer server.Close()

	provider := NewTokenProvider(Token{
		AccessToken:  "old-at",
		RefreshToken: "old-rt",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, server.URL+"/oauth/token", "client-1")

	_, err := provider.AccessToken(context.Background())
	if err == nil || !strings.Contains(err.Error(), "access_token") {
		t.Fatalf("expected missing access_token error, got %v", err)
	}
}

func TestAccessTokenRefreshesThenCaches(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{"access_token":"fresh","refresh_token":"fresh-rt","expires_in":3600}`))
	}))
	defer server.Close()

	provider := NewTokenProvider(Token{
		AccessToken:  "old-at",
		RefreshToken: "old-rt",
		ExpiresAt:    time.Now().Add(-time.Minute),
	}, server.URL+"/oauth/token", "client-1")

	got, err := provider.AccessToken(context.Background())
	if err != nil || got != "fresh" {
		t.Fatalf("first AccessToken = %q, %v", got, err)
	}
	// Second call uses the freshly cached (valid) token, no refresh.
	got, err = provider.AccessToken(context.Background())
	if err != nil || got != "fresh" {
		t.Fatalf("second AccessToken = %q, %v", got, err)
	}
	if calls != 1 {
		t.Fatalf("refresh calls = %d, want 1", calls)
	}
}
