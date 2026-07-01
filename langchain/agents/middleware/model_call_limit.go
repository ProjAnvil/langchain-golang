package middleware

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
)

const (
	ThreadModelCallCountKey = "thread_model_call_count"
	RunModelCallCountKey    = "run_model_call_count"
)

type ModelCallLimitExceededError struct {
	ThreadCount int
	RunCount    int
	ThreadLimit *int
	RunLimit    *int
}

func (e ModelCallLimitExceededError) Error() string {
	return buildLimitExceededMessage(e.ThreadCount, e.RunCount, e.ThreadLimit, e.RunLimit)
}

type ModelCallLimitMiddleware struct {
	ThreadLimit  *int
	RunLimit     *int
	ExitBehavior string
}

func NewModelCallLimitMiddleware(threadLimit, runLimit *int, exitBehavior string) (*ModelCallLimitMiddleware, error) {
	if threadLimit == nil && runLimit == nil {
		return nil, fmt.Errorf("at least one limit must be specified (thread_limit or run_limit)")
	}
	if exitBehavior == "" {
		exitBehavior = "end"
	}
	if exitBehavior != "end" && exitBehavior != "error" {
		return nil, fmt.Errorf("invalid exit_behavior: %s. Must be 'end' or 'error'", exitBehavior)
	}
	return &ModelCallLimitMiddleware{
		ThreadLimit:  threadLimit,
		RunLimit:     runLimit,
		ExitBehavior: exitBehavior,
	}, nil
}

func (m *ModelCallLimitMiddleware) BeforeModel(ctx context.Context, state map[string]any) (*Command, error) {
	threadCount := intFromState(state, ThreadModelCallCountKey)
	runCount := intFromState(state, RunModelCallCountKey)

	threadExceeded := m.ThreadLimit != nil && threadCount >= *m.ThreadLimit
	runExceeded := m.RunLimit != nil && runCount >= *m.RunLimit
	if !threadExceeded && !runExceeded {
		return nil, nil
	}

	if m.ExitBehavior == "error" {
		return nil, ModelCallLimitExceededError{
			ThreadCount: threadCount,
			RunCount:    runCount,
			ThreadLimit: m.ThreadLimit,
			RunLimit:    m.RunLimit,
		}
	}

	return &Command{
		Goto: "end",
		Update: map[string]any{
			"messages": []messages.Message{
				messages.AI(buildLimitExceededMessage(threadCount, runCount, m.ThreadLimit, m.RunLimit)),
			},
		},
	}, nil
}

// AfterModel returns (map[string]any, error), matching the AfterModel hook
// signature used by every other middleware in this package (see
// langchain/agents/create_agent.go's AfterModelHook), so this middleware can
// be composed generically by CreateAgent alongside the rest.
func (m *ModelCallLimitMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	return map[string]any{
		ThreadModelCallCountKey: intFromState(state, ThreadModelCallCountKey) + 1,
		RunModelCallCountKey:    intFromState(state, RunModelCallCountKey) + 1,
	}, nil
}

func buildLimitExceededMessage(threadCount, runCount int, threadLimit, runLimit *int) string {
	limits := []string{}
	if threadLimit != nil && threadCount >= *threadLimit {
		limits = append(limits, fmt.Sprintf("thread limit (%d/%d)", threadCount, *threadLimit))
	}
	if runLimit != nil && runCount >= *runLimit {
		limits = append(limits, fmt.Sprintf("run limit (%d/%d)", runCount, *runLimit))
	}
	return fmt.Sprintf("Model call limits exceeded: %s", stringsJoin(limits, ", "))
}

func intFromState(state map[string]any, key string) int {
	if state == nil {
		return 0
	}
	switch value := state[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case float32:
		return int(value)
	default:
		return 0
	}
}

func stringsJoin(values []string, separator string) string {
	if len(values) == 0 {
		return ""
	}
	out := values[0]
	for _, value := range values[1:] {
		out += separator + value
	}
	return out
}
