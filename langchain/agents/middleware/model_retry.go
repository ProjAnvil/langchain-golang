package middleware

import (
	"context"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
)

type ModelHandler func(context.Context, ModelRequest) (ModelResponse, error)

type ModelRetryMiddleware struct {
	MaxRetries    int
	RetryOn       RetryPredicate
	OnFailure     string
	Formatter     FailureFormatter
	BackoffFactor float64
	InitialDelay  time.Duration
	MaxDelay      time.Duration
	Jitter        bool
	Sleep         func(time.Duration)
}

type ModelRetryOption func(*ModelRetryMiddleware)

func NewModelRetryMiddleware(opts ...ModelRetryOption) (*ModelRetryMiddleware, error) {
	m := &ModelRetryMiddleware{
		MaxRetries:    2,
		RetryOn:       func(error) bool { return true },
		OnFailure:     "continue",
		BackoffFactor: 2,
		InitialDelay:  time.Second,
		MaxDelay:      60 * time.Second,
		Jitter:        true,
		Sleep:         time.Sleep,
	}
	for _, opt := range opts {
		opt(m)
	}
	if m.RetryOn == nil {
		m.RetryOn = func(error) bool { return true }
	}
	if m.Sleep == nil {
		m.Sleep = time.Sleep
	}
	if m.OnFailure == "" {
		m.OnFailure = "continue"
	}
	if m.OnFailure != "continue" && m.OnFailure != "error" {
		return nil, fmt.Errorf("invalid on_failure: %s. Must be 'continue' or 'error'", m.OnFailure)
	}
	if err := validateRetryParams(m.MaxRetries, m.InitialDelay, m.MaxDelay, m.BackoffFactor); err != nil {
		return nil, err
	}
	return m, nil
}

func WithModelRetryMaxRetries(maxRetries int) ModelRetryOption {
	return func(m *ModelRetryMiddleware) {
		m.MaxRetries = maxRetries
	}
}

func WithModelRetryOn(predicate RetryPredicate) ModelRetryOption {
	return func(m *ModelRetryMiddleware) {
		m.RetryOn = predicate
	}
}

func WithModelRetryOnFailure(onFailure string) ModelRetryOption {
	return func(m *ModelRetryMiddleware) {
		m.OnFailure = onFailure
	}
}

func WithModelRetryFailureFormatter(formatter FailureFormatter) ModelRetryOption {
	return func(m *ModelRetryMiddleware) {
		m.Formatter = formatter
	}
}

func WithModelRetryBackoff(initialDelay, maxDelay time.Duration, backoffFactor float64, jitter bool) ModelRetryOption {
	return func(m *ModelRetryMiddleware) {
		m.InitialDelay = initialDelay
		m.MaxDelay = maxDelay
		m.BackoffFactor = backoffFactor
		m.Jitter = jitter
	}
}

func WithModelRetrySleep(sleep func(time.Duration)) ModelRetryOption {
	return func(m *ModelRetryMiddleware) {
		m.Sleep = sleep
	}
}

func (m *ModelRetryMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	for attempt := 0; attempt <= m.MaxRetries; attempt++ {
		response, err := handler(ctx, request)
		if err == nil {
			return response, nil
		}
		attemptsMade := attempt + 1
		if !m.RetryOn(err) || attempt == m.MaxRetries {
			return m.handleFailure(err, attemptsMade)
		}
		delay := calculateRetryDelay(attempt, m.BackoffFactor, m.InitialDelay, m.MaxDelay, m.Jitter)
		if delay > 0 {
			m.Sleep(delay)
		}
	}
	return ModelResponse{}, fmt.Errorf("retry loop completed without result")
}

func (m *ModelRetryMiddleware) handleFailure(err error, attemptsMade int) (ModelResponse, error) {
	if m.OnFailure == "error" {
		return ModelResponse{}, err
	}
	content := ""
	if m.Formatter != nil {
		content = m.Formatter(err)
	} else {
		word := "attempt"
		if attemptsMade != 1 {
			word = "attempts"
		}
		content = fmt.Sprintf("Model call failed after %d %s with %T: %v", attemptsMade, word, err, err)
	}
	return ModelResponse{Result: []messages.Message{messages.AI(content)}}, nil
}
