package middleware

import (
	"context"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/tools"
)

type ToolHandler func(context.Context, ToolCallRequest) (messages.Message, error)

type ToolRetryMiddleware struct {
	MaxRetries    int
	ToolFilter    map[string]bool
	RetryOn       RetryPredicate
	OnFailure     string
	Formatter     FailureFormatter
	BackoffFactor float64
	InitialDelay  time.Duration
	MaxDelay      time.Duration
	Jitter        bool
	Sleep         func(time.Duration)
}

type ToolRetryOption func(*ToolRetryMiddleware)

func NewToolRetryMiddleware(opts ...ToolRetryOption) (*ToolRetryMiddleware, error) {
	m := &ToolRetryMiddleware{
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
	switch m.OnFailure {
	case "":
		m.OnFailure = "continue"
	case "return_message":
		m.OnFailure = "continue"
	case "raise":
		m.OnFailure = "error"
	}
	if m.OnFailure != "continue" && m.OnFailure != "error" {
		return nil, fmt.Errorf("invalid on_failure: %s. Must be 'continue' or 'error'", m.OnFailure)
	}
	if err := validateRetryParams(m.MaxRetries, m.InitialDelay, m.MaxDelay, m.BackoffFactor); err != nil {
		return nil, err
	}
	return m, nil
}

func WithToolRetryMaxRetries(maxRetries int) ToolRetryOption {
	return func(m *ToolRetryMiddleware) {
		m.MaxRetries = maxRetries
	}
}

func WithToolRetryTools(names ...string) ToolRetryOption {
	return func(m *ToolRetryMiddleware) {
		if m.ToolFilter == nil {
			m.ToolFilter = map[string]bool{}
		}
		for _, name := range names {
			m.ToolFilter[name] = true
		}
	}
}

// WithToolRetryToolInstances restricts retries to the given tools, matched by
// name. This mirrors Python's `tools: list[BaseTool | str] | None`, which
// accepts BaseTool instances in addition to plain name strings.
func WithToolRetryToolInstances(instances ...tools.Tool) ToolRetryOption {
	return func(m *ToolRetryMiddleware) {
		if m.ToolFilter == nil {
			m.ToolFilter = map[string]bool{}
		}
		for _, t := range instances {
			if t != nil {
				m.ToolFilter[t.Name()] = true
			}
		}
	}
}

func WithToolRetryOn(predicate RetryPredicate) ToolRetryOption {
	return func(m *ToolRetryMiddleware) {
		m.RetryOn = predicate
	}
}

func WithToolRetryOnFailure(onFailure string) ToolRetryOption {
	return func(m *ToolRetryMiddleware) {
		m.OnFailure = onFailure
	}
}

func WithToolRetryFailureFormatter(formatter FailureFormatter) ToolRetryOption {
	return func(m *ToolRetryMiddleware) {
		m.Formatter = formatter
	}
}

func WithToolRetryBackoff(initialDelay, maxDelay time.Duration, backoffFactor float64, jitter bool) ToolRetryOption {
	return func(m *ToolRetryMiddleware) {
		m.InitialDelay = initialDelay
		m.MaxDelay = maxDelay
		m.BackoffFactor = backoffFactor
		m.Jitter = jitter
	}
}

func WithToolRetrySleep(sleep func(time.Duration)) ToolRetryOption {
	return func(m *ToolRetryMiddleware) {
		m.Sleep = sleep
	}
}

func (m *ToolRetryMiddleware) WrapToolCall(ctx context.Context, request ToolCallRequest, handler ToolHandler) (messages.Message, error) {
	toolName := request.ToolCall.Name
	if request.Tool != nil {
		toolName = request.Tool.Name()
	}
	if !m.shouldRetryTool(toolName) {
		return handler(ctx, request)
	}

	for attempt := 0; attempt <= m.MaxRetries; attempt++ {
		response, err := handler(ctx, request)
		if err == nil {
			return response, nil
		}
		attemptsMade := attempt + 1
		if !m.RetryOn(err) || attempt == m.MaxRetries {
			return m.handleFailure(toolName, request.ToolCall.ID, err, attemptsMade)
		}
		delay := calculateRetryDelay(attempt, m.BackoffFactor, m.InitialDelay, m.MaxDelay, m.Jitter)
		if delay > 0 {
			m.Sleep(delay)
		}
	}
	return messages.Message{}, fmt.Errorf("retry loop completed without result")
}

func (m *ToolRetryMiddleware) shouldRetryTool(toolName string) bool {
	if m.ToolFilter == nil {
		return true
	}
	return m.ToolFilter[toolName]
}

func (m *ToolRetryMiddleware) handleFailure(toolName, toolCallID string, err error, attemptsMade int) (messages.Message, error) {
	if m.OnFailure == "error" {
		return messages.Message{}, err
	}
	content := ""
	if m.Formatter != nil {
		content = m.Formatter(err)
	} else {
		word := "attempt"
		if attemptsMade != 1 {
			word = "attempts"
		}
		content = fmt.Sprintf("Tool '%s' failed after %d %s with %T: %v. Please try again.", toolName, attemptsMade, word, err, err)
	}
	return errorToolMessage(toolCallID, toolName, content), nil
}

func errorToolMessage(toolCallID, toolName, content string) messages.Message {
	message := messages.Tool(toolCallID, content)
	message.Name = toolName
	message.ResponseMetadata = map[string]any{"status": "error"}
	return message
}
