package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/modelconfig"
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
	return time.Now().Add(skew).After(t.ExpiresAt)
}

// TokenProvider returns a current access token, refreshing it via the OAuth
// token endpoint when it nears expiry. Concurrent callers are serialized.
type TokenProvider struct {
	mu          sync.Mutex
	token       Token
	tokenURL    string
	clientID    string
	httpClient  *http.Client
	refreshSkew time.Duration
}

// NewTokenProvider builds a provider around an initial token bundle.
func NewTokenProvider(token Token, tokenURL, clientID string) *TokenProvider {
	if tokenURL == "" {
		tokenURL = defaultChatGPTTokenURL
	}
	return &TokenProvider{
		token:       token,
		tokenURL:    tokenURL,
		clientID:    clientID,
		httpClient:  http.DefaultClient,
		refreshSkew: 30 * time.Second,
	}
}

// AccessToken returns a valid access token, refreshing when expired (within
// the refresh skew).
func (p *TokenProvider) AccessToken(ctx context.Context) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.token.IsExpired(p.refreshSkew) {
		return p.token.AccessToken, nil
	}
	refreshed, err := p.refresh(ctx)
	if err != nil {
		return "", err
	}
	p.token = refreshed
	return refreshed.AccessToken, nil
}

// chatGPTTokenResponse is the OAuth token endpoint's JSON payload.
type chatGPTTokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
	ExpiresIn    int    `json:"expires_in"`
}

// refresh performs the OAuth refresh_token grant against the token URL via
// the shared httpclient helper, so failures come back as typed lcerrors
// values (ProviderError for non-2xx, wrapped transport errors) exactly like
// every other provider call. Retries stay disabled: the server may rotate
// the refresh token during a partially-failed attempt, and replaying an
// already-consumed refresh token can invalidate the login.
func (p *TokenProvider) refresh(ctx context.Context) (Token, error) {
	parsed, err := url.Parse(p.tokenURL)
	if err != nil {
		return Token{}, fmt.Errorf("chatgpt oauth refresh: parse token URL: %w", err)
	}
	endpoint := parsed.Path
	if parsed.RawQuery != "" {
		endpoint += "?" + parsed.RawQuery
	}
	// MaxRetries 0 means exactly one attempt (see the comment above).
	cfg := modelconfig.Config{BaseURL: parsed.Scheme + "://" + parsed.Host, HTTPClient: p.httpClient}
	payload, err := httpclient.PostJSON[chatGPTTokenResponse](
		ctx,
		providerName,
		cfg,
		endpoint,
		map[string]string{
			"grant_type":    "refresh_token",
			"refresh_token": p.token.RefreshToken,
			"client_id":     p.clientID,
		},
		nil,
	)
	if err != nil {
		return Token{}, fmt.Errorf("chatgpt oauth refresh: %w", err)
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
