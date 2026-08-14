package moderation

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

// ViolationError reports that moderated content was flagged.
type ViolationError struct {
	Categories []string
}

func (e *ViolationError) Error() string {
	return fmt.Sprintf("openai moderation flagged content: %s", strings.Join(e.Categories, ", "))
}

// Middleware flags user input (before the model call) and model output (after)
// via the OpenAI Moderations API, mirroring Python's OpenAIModerationMiddleware.
type Middleware struct {
	Client Client
}

// NewMiddleware builds a moderation middleware around a Client.
func NewMiddleware(client Client) *Middleware {
	return &Middleware{Client: client}
}

// BeforeModel moderates the last human message in state["messages"] and
// returns a ViolationError if it is flagged.
func (m *Middleware) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	text, ok := lastMessageText(state, messages.RoleHuman)
	if !ok {
		return state, nil
	}
	result, err := m.Client.Moderate(ctx, text)
	if err != nil {
		return nil, err
	}
	if result.Flagged {
		return nil, &ViolationError{Categories: flaggedCategories(result)}
	}
	return state, nil
}

// AfterModel moderates the last AI message in state["messages"] and returns a
// ViolationError if it is flagged.
func (m *Middleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	text, ok := lastMessageText(state, messages.RoleAI)
	if !ok {
		return state, nil
	}
	result, err := m.Client.Moderate(ctx, text)
	if err != nil {
		return nil, err
	}
	if result.Flagged {
		return nil, &ViolationError{Categories: flaggedCategories(result)}
	}
	return state, nil
}

func lastMessageText(state map[string]any, role messages.Role) (string, bool) {
	if state == nil {
		return "", false
	}
	msgs, ok := state["messages"].([]messages.Message)
	if !ok {
		return "", false
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == role {
			return msgs[i].Content, true
		}
	}
	return "", false
}

func flaggedCategories(result Result) []string {
	var cats []string
	for name, flagged := range result.Categories {
		if flagged {
			cats = append(cats, name)
		}
	}
	sort.Strings(cats)
	return cats
}
