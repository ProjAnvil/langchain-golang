package moderation

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

const defaultBaseURL = "https://api.openai.com/v1"

// Result is the outcome of a moderation check.
type Result struct {
	Flagged    bool
	Categories map[string]bool
	Scores     map[string]float64
}

// Client is a minimal OpenAI Moderations API client, mirroring
// `OpenAI.moderations.create(model=..., input=...)`.
type Client struct {
	config modelconfig.Config
}

// NewClient builds a moderation client. Defaults: base URL
// https://api.openai.com/v1 and model "omni-moderation-latest".
func NewClient(opts ...modelconfig.Option) Client {
	cfg := modelconfig.New(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "omni-moderation-latest"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return Client{config: cfg}
}

// Moderate flags text via the OpenAI Moderations API.
func (c Client) Moderate(ctx context.Context, text string) (Result, error) {
	ctx, cancel := context.WithTimeout(ctx, c.config.Timeout)
	defer cancel()

	resp, err := httpclient.PostJSON[moderationResponse](ctx, "openai", c.config, "/moderations",
		moderationRequest{Model: c.config.Model, Input: text},
		func(req *http.Request) {
			req.Header.Set("Content-Type", "application/json")
			if c.config.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
			}
			for name, value := range c.config.Headers {
				req.Header.Set(name, value)
			}
		},
	)
	if err != nil {
		return Result{}, err
	}
	if len(resp.Results) == 0 {
		return Result{}, fmt.Errorf("openai moderation: empty results")
	}
	r := resp.Results[0]
	return Result{Flagged: r.Flagged, Categories: r.Categories, Scores: r.CategoryScores}, nil
}

type moderationRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type moderationResponse struct {
	Results []moderationResult `json:"results"`
}

type moderationResult struct {
	Flagged        bool               `json:"flagged"`
	Categories     map[string]bool    `json:"categories"`
	CategoryScores map[string]float64 `json:"category_scores"`
}
