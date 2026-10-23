package modelconfig

import (
	"maps"
	"net/http"
	"time"
)

// Config is the normalized provider model configuration.
type Config struct {
	Model                  string
	BaseURL                string
	APIKey                 string
	Headers                map[string]string
	Temperature            *float64
	MaxTokens              *int
	Timeout                time.Duration
	MaxRetries             int
	RetryDelay             time.Duration
	RetryBackoffMultiplier float64
	RetryMaxDelay          time.Duration
	HTTPClient             *http.Client
	// TiktokenModelName overrides the model name used for token counting,
	// mirroring Python ChatOpenAI(tiktoken_model_name=...).
	TiktokenModelName string
	Extra             map[string]any
}

// Option configures a model Config.
type Option func(*Config)

// New creates a normalized config.
func New(opts ...Option) Config {
	cfg := Config{
		MaxRetries: 2,
		Headers:    map[string]string{},
		Extra:      map[string]any{},
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// WithModel sets the model name.
func WithModel(model string) Option {
	return func(cfg *Config) {
		cfg.Model = model
	}
}

// WithBaseURL sets the provider base URL.
func WithBaseURL(baseURL string) Option {
	return func(cfg *Config) {
		cfg.BaseURL = baseURL
	}
}

// WithAPIKey sets the provider API key.
func WithAPIKey(apiKey string) Option {
	return func(cfg *Config) {
		cfg.APIKey = apiKey
	}
}

// WithHeader sets one provider header.
func WithHeader(name string, value string) Option {
	return func(cfg *Config) {
		if cfg.Headers == nil {
			cfg.Headers = map[string]string{}
		}
		cfg.Headers[name] = value
	}
}

// WithTemperature sets model sampling temperature.
func WithTemperature(temperature float64) Option {
	return func(cfg *Config) {
		cfg.Temperature = &temperature
	}
}

// WithMaxTokens sets the maximum generated token count.
func WithMaxTokens(maxTokens int) Option {
	return func(cfg *Config) {
		cfg.MaxTokens = &maxTokens
	}
}

// WithTimeout sets request timeout.
func WithTimeout(timeout time.Duration) Option {
	return func(cfg *Config) {
		cfg.Timeout = timeout
	}
}

// WithMaxRetries sets retry count.
func WithMaxRetries(maxRetries int) Option {
	return func(cfg *Config) {
		cfg.MaxRetries = maxRetries
	}
}

// WithRetryDelay sets the initial delay between retry attempts.
func WithRetryDelay(delay time.Duration) Option {
	return func(cfg *Config) {
		cfg.RetryDelay = delay
	}
}

// WithRetryBackoffMultiplier sets the multiplier applied to each retry delay.
func WithRetryBackoffMultiplier(multiplier float64) Option {
	return func(cfg *Config) {
		cfg.RetryBackoffMultiplier = multiplier
	}
}

// WithRetryMaxDelay caps the delay between retry attempts.
func WithRetryMaxDelay(delay time.Duration) Option {
	return func(cfg *Config) {
		cfg.RetryMaxDelay = delay
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(cfg *Config) {
		cfg.HTTPClient = client
	}
}

// WithTiktokenModelName overrides the model name used to pick the token
// encoding, mirroring Python ChatOpenAI(tiktoken_model_name=...).
func WithTiktokenModelName(name string) Option {
	return func(cfg *Config) {
		cfg.TiktokenModelName = name
	}
}

// WithExtra stores a provider-specific option.
func WithExtra(name string, value any) Option {
	return func(cfg *Config) {
		if cfg.Extra == nil {
			cfg.Extra = map[string]any{}
		}
		cfg.Extra[name] = value
	}
}

// Clone returns a defensive copy with independent maps and scalar pointers.
// Nil Headers and Extra maps become writable empty maps.
func (cfg Config) Clone() Config {
	out := cfg
	out.Headers = make(map[string]string, len(cfg.Headers))
	maps.Copy(out.Headers, cfg.Headers)
	out.Extra = make(map[string]any, len(cfg.Extra))
	maps.Copy(out.Extra, cfg.Extra)
	if cfg.Temperature != nil {
		out.Temperature = new(*cfg.Temperature)
	}
	if cfg.MaxTokens != nil {
		out.MaxTokens = new(*cfg.MaxTokens)
	}
	return out
}
