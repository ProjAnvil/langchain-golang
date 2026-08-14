package moderation

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

// ViolationError reports that moderated content was flagged, with the stage
// where it occurred (mirroring Python's OpenAIModerationError.stage).
type ViolationError struct {
	Stage      string
	Categories []string
}

func (e *ViolationError) Error() string {
	stage := e.Stage
	if stage == "" {
		stage = "content"
	}
	return fmt.Sprintf("openai moderation flagged %s: %s", stage, strings.Join(e.Categories, ", "))
}

// Middleware flags user input, tool results, and model output via the OpenAI
// Moderations API, mirroring Python's OpenAIModerationMiddleware.
type Middleware struct {
	Client Client
	// CheckInput moderates the last human message before the model call.
	CheckInput bool
	// CheckOutput moderates the last AI message after the model call.
	CheckOutput bool
	// CheckToolResults moderates each tool-result message after the last AI
	// message before the next model call.
	CheckToolResults bool
}

// NewMiddleware builds a moderation middleware around a Client, enabling
// input and output checks by default (matching Python's defaults).
func NewMiddleware(client Client) *Middleware {
	return &Middleware{Client: client, CheckInput: true, CheckOutput: true}
}

// BeforeModel moderates tool results (if enabled) and the last human message,
// returning a ViolationError if any flagged content is found.
func (m *Middleware) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	msgs, ok := state["messages"].([]messages.Message)
	if !ok || len(msgs) == 0 {
		return state, nil
	}

	if m.CheckToolResults {
		ve, err := m.moderateToolMessages(ctx, msgs)
		if err != nil {
			return nil, err
		}
		if ve != nil {
			return nil, ve
		}
	}

	if m.CheckInput {
		if text, ok := lastMessageText(msgs, messages.RoleHuman); ok {
			ve, err := m.moderateText(ctx, text, "input")
			if err != nil {
				return nil, err
			}
			if ve != nil {
				return nil, ve
			}
		}
	}
	return state, nil
}

// AfterModel moderates the last AI message, returning a ViolationError if it
// is flagged.
func (m *Middleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	if !m.CheckOutput {
		return state, nil
	}
	if msgs, ok := state["messages"].([]messages.Message); ok {
		if text, ok := lastMessageText(msgs, messages.RoleAI); ok {
			ve, err := m.moderateText(ctx, text, "output")
			if err != nil {
				return nil, err
			}
			if ve != nil {
				return nil, ve
			}
		}
	}
	return state, nil
}

// moderateToolMessages moderates each tool-result message after the last AI
// message, returning the first violation (if any).
func (m *Middleware) moderateToolMessages(ctx context.Context, msgs []messages.Message) (*ViolationError, error) {
	lastAI := -1
	for i, msg := range msgs {
		if msg.Role == messages.RoleAI {
			lastAI = i
		}
	}
	if lastAI < 0 {
		return nil, nil
	}
	for i := lastAI + 1; i < len(msgs); i++ {
		if msgs[i].Role != messages.RoleTool || msgs[i].Content == "" {
			continue
		}
		ve, err := m.moderateText(ctx, msgs[i].Content, "tool")
		if err != nil {
			return nil, err
		}
		if ve != nil {
			return ve, nil
		}
	}
	return nil, nil
}

func (m *Middleware) moderateText(ctx context.Context, text string, stage string) (*ViolationError, error) {
	if text == "" {
		return nil, nil
	}
	result, err := m.Client.Moderate(ctx, text)
	if err != nil {
		return nil, err
	}
	if result.Flagged {
		return &ViolationError{Stage: stage, Categories: flaggedCategories(result)}, nil
	}
	return nil, nil
}

func lastMessageText(msgs []messages.Message, role messages.Role) (string, bool) {
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
