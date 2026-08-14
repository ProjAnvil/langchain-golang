package moderation

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

// ExitBehavior controls how a moderation violation is handled, mirroring
// Python's `exit_behavior` ("error", "end", "replace").
type ExitBehavior string

const (
	ExitError   ExitBehavior = "error"
	ExitEnd     ExitBehavior = "end"
	ExitReplace ExitBehavior = "replace"
)

const defaultViolationTemplate = "I'm sorry, but I can't comply with that request. It was flagged for {categories}."

// ViolationError reports that moderated content was flagged, with the stage
// where it occurred (mirroring Python's OpenAIModerationError).
type ViolationError struct {
	Stage      string
	Content    string
	Categories []string
	Result     Result
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
	// ExitBehavior controls how a violation is handled. Defaults to "end",
	// matching Python.
	ExitBehavior ExitBehavior
	// ViolationMessage overrides the default violation template. Supported
	// placeholders: {categories}, {category_scores}, {original_content}.
	ViolationMessage string
}

// NewMiddleware builds a moderation middleware around a Client, enabling
// input and output checks and the "end" exit behavior by default (matching
// Python's defaults).
func NewMiddleware(client Client) *Middleware {
	return &Middleware{
		Client:       client,
		CheckInput:   true,
		CheckOutput:  true,
		ExitBehavior: ExitEnd,
	}
}

// BeforeModel moderates tool results (if enabled) and the last human message.
// The return value is the state update (or an error, depending on the exit
// behavior).
func (m *Middleware) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	msgs, ok := state["messages"].([]messages.Message)
	if !ok || len(msgs) == 0 {
		return nil, nil
	}

	if m.CheckToolResults {
		if update, err := m.moderateToolMessages(ctx, msgs); err != nil || update != nil {
			return update, err
		}
	}

	if m.CheckInput {
		if idx := lastMessageIndex(msgs, messages.RoleHuman); idx >= 0 {
			return m.applyViolation(ctx, msgs, idx, "input", msgs[idx].Content)
		}
	}
	return nil, nil
}

// AfterModel moderates the last AI message.
func (m *Middleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	if !m.CheckOutput {
		return nil, nil
	}
	msgs, ok := state["messages"].([]messages.Message)
	if !ok || len(msgs) == 0 {
		return nil, nil
	}
	if idx := lastMessageIndex(msgs, messages.RoleAI); idx >= 0 {
		return m.applyViolation(ctx, msgs, idx, "output", msgs[idx].Content)
	}
	return nil, nil
}

// moderateToolMessages moderates each tool-result message after the last AI
// message, returning the first violation action (or nil).
func (m *Middleware) moderateToolMessages(ctx context.Context, msgs []messages.Message) (map[string]any, error) {
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
		if update, err := m.applyViolation(ctx, msgs, i, "tool", msgs[i].Content); err != nil || update != nil {
			return update, err
		}
	}
	return nil, nil
}

// applyViolation runs a moderation check on content and converts a flag into
// the exit-behavior-specific result.
func (m *Middleware) applyViolation(ctx context.Context, msgs []messages.Message, index int, stage string, content string) (map[string]any, error) {
	if content == "" {
		return nil, nil
	}
	result, err := m.Client.Moderate(ctx, content)
	if err != nil {
		return nil, err
	}
	if !result.Flagged {
		return nil, nil
	}

	categories := flaggedCategories(result)
	switch m.ExitBehavior {
	case ExitError:
		return nil, &ViolationError{Stage: stage, Content: content, Categories: categories, Result: result}
	case ExitEnd:
		return map[string]any{
			"jump_to": "end",
			"messages": []messages.Message{messages.AI(m.formatViolationMessage(content, result))},
		}, nil
	case ExitReplace:
		newMsgs := append([]messages.Message(nil), msgs...)
		orig := newMsgs[index]
		orig.Content = m.formatViolationMessage(content, result)
		newMsgs[index] = orig
		return map[string]any{"messages": newMsgs}, nil
	default:
		return nil, &ViolationError{Stage: stage, Content: content, Categories: categories, Result: result}
	}
}

// formatViolationMessage renders the violation template, mirroring Python's
// `_format_violation_message` (categories with "_"→" " and sorted JSON scores).
func (m *Middleware) formatViolationMessage(content string, result Result) string {
	categories := flaggedCategories(result)
	labels := make([]string, len(categories))
	for i, c := range categories {
		labels[i] = strings.ReplaceAll(c, "_", " ")
	}
	categoryLabel := strings.Join(labels, ", ")
	if categoryLabel == "" {
		categoryLabel = "OpenAI's safety policies"
	}
	scoresJSON, _ := json.Marshal(result.Scores)

	template := m.ViolationMessage
	if template == "" {
		template = defaultViolationTemplate
	}
	msg := strings.ReplaceAll(template, "{categories}", categoryLabel)
	msg = strings.ReplaceAll(msg, "{category_scores}", string(scoresJSON))
	msg = strings.ReplaceAll(msg, "{original_content}", content)
	return msg
}

func lastMessageIndex(msgs []messages.Message, role messages.Role) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == role {
			return i
		}
	}
	return -1
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
