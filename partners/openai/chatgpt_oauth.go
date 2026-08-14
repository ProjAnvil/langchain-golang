package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// ChatGPT OAuth token bundle + refreshable provider, mirroring the parts of
// Python's `langchain_openai.chatgpt_oauth` that are offline-testable. The
// browser login (device-authorization / PKCE `authorize`) flow is out of scope:
// callers obtain an initial token bundle themselves and hand it to a
// TokenProvider.

const defaultChatGPTTokenURL = "https://auth.openai.com/oauth/token"

// Token is a ChatGPT OAuth token bundle.
type Token struct {
	AccessToken  string
	RefreshToken string
	IDToken      string
	ExpiresAt    time.Time
}

// IsExpired reports whether the token is past (or within skew of) expiry.
func (t Token) IsExpired(skew time.Duration) bool {
	if skew <= 0 {
		skew = 30 * time.Second
	}
	return time.Now().Add(skew).After(t.ExpiresAt)
}

// TokenProvider returns a current access token, refreshing it via the OAuth
// token endpoint when it nears expiry. Concurrent callers are serialized.
type TokenProvider struct {
	mu         sync.Mutex
	token      Token
	tokenURL   string
	clientID   string
	httpClient *http.Client
}

// NewTokenProvider builds a provider around an initial token bundle.
func NewTokenProvider(token Token, tokenURL, clientID string) *TokenProvider {
	if tokenURL == "" {
		tokenURL = defaultChatGPTTokenURL
	}
	return &TokenProvider{token: token, tokenURL: tokenURL, clientID: clientID, httpClient: http.DefaultClient}
}

// AccessToken returns a valid access token, refreshing when expired.
func (p *TokenProvider) AccessToken(ctx context.Context) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.token.IsExpired(0) {
		return p.token.AccessToken, nil
	}
	refreshed, err := p.refresh(ctx)
	if err != nil {
		return "", err
	}
	p.token = refreshed
	return refreshed.AccessToken, nil
}

// refresh performs the OAuth refresh_token grant against the token URL.
func (p *TokenProvider) refresh(ctx context.Context) (Token, error) {
	body, err := json.Marshal(map[string]string{
		"grant_type":    "refresh_token",
		"refresh_token": p.token.RefreshToken,
		"client_id":     p.clientID,
	})
	if err != nil {
		return Token{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.tokenURL, bytes.NewReader(body))
	if err != nil {
		return Token{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return Token{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Token{}, fmt.Errorf("chatgpt oauth refresh: %s", resp.Status)
	}
	var payload struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		IDToken      string `json:"id_token"`
		ExpiresIn    int    `json:"expires_in"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return Token{}, err
	}
	if payload.AccessToken == "" {
		return Token{}, fmt.Errorf("chatgpt oauth refresh: no access_token in response")
	}
	if payload.RefreshToken == "" {
		payload.RefreshToken = p.token.RefreshToken
	}
	return Token{
		AccessToken:  payload.AccessToken,
		RefreshToken: payload.RefreshToken,
		IDToken:      payload.IDToken,
		ExpiresAt:    time.Now().Add(time.Duration(payload.ExpiresIn) * time.Second),
	}, nil
}
