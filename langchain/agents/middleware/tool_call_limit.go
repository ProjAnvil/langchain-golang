package middleware

import (
	"context"
	"fmt"
	"sort"

	"github.com/projanvil/langchain-golang/core/messages"
)

const (
	ThreadToolCallCountKey = "thread_tool_call_count"
	RunToolCallCountKey    = "run_tool_call_count"
	allToolsCountKey       = "__all__"
)

type ToolCallLimitExceededError struct {
	ThreadCount int
	RunCount    int
	ThreadLimit *int
	RunLimit    *int
	ToolName    string
}

func (e ToolCallLimitExceededError) Error() string {
	return buildFinalToolLimitMessage(e.ThreadCount, e.RunCount, e.ThreadLimit, e.RunLimit, e.ToolName)
}

type ToolCallLimitMiddleware struct {
	ToolName     string
	ThreadLimit  *int
	RunLimit     *int
	ExitBehavior string
}

func NewToolCallLimitMiddleware(toolName string, threadLimit, runLimit *int, exitBehavior string) (*ToolCallLimitMiddleware, error) {
	if threadLimit == nil && runLimit == nil {
		return nil, fmt.Errorf("at least one limit must be specified (thread_limit or run_limit)")
	}
	if exitBehavior == "" {
		exitBehavior = "continue"
	}
	if exitBehavior != "continue" && exitBehavior != "error" && exitBehavior != "end" {
		return nil, fmt.Errorf("invalid exit_behavior: %q. Must be one of [continue error end]", exitBehavior)
	}
	if threadLimit != nil && runLimit != nil && *runLimit > *threadLimit {
		return nil, fmt.Errorf("run_limit (%d) cannot exceed thread_limit (%d)", *runLimit, *threadLimit)
	}
	return &ToolCallLimitMiddleware{
		ToolName:     toolName,
		ThreadLimit:  threadLimit,
		RunLimit:     runLimit,
		ExitBehavior: exitBehavior,
	}, nil
}

func (m *ToolCallLimitMiddleware) Name() string {
	if m.ToolName == "" {
		return "ToolCallLimitMiddleware"
	}
	return "ToolCallLimitMiddleware[" + m.ToolName + "]"
}

func (m *ToolCallLimitMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	lastAI := lastAIMessage(state)
	if lastAI == nil || len(lastAI.ToolCalls) == 0 {
		return nil, nil
	}

	countKey := m.countKey()
	threadCounts := mapStringIntFromState(state, ThreadToolCallCountKey)
	runCounts := mapStringIntFromState(state, RunToolCallCountKey)
	threadCount := threadCounts[countKey]
	runCount := runCounts[countKey]

	allowed, blocked, newThreadCount, newRunCount := m.separateToolCalls(lastAI.ToolCalls, threadCount, runCount)
	threadCounts[countKey] = newThreadCount
	runCounts[countKey] = newRunCount + len(blocked)

	if len(blocked) == 0 {
		if len(allowed) == 0 {
			return nil, nil
		}
		return map[string]any{
			ThreadToolCallCountKey: threadCounts,
			RunToolCallCountKey:    runCounts,
		}, nil
	}

	finalThreadCount := threadCounts[countKey]
	finalRunCount := runCounts[countKey]
	if m.ExitBehavior == "error" {
		return nil, ToolCallLimitExceededError{
			ThreadCount: finalThreadCount + len(blocked),
			RunCount:    finalRunCount,
			ThreadLimit: m.ThreadLimit,
			RunLimit:    m.RunLimit,
			ToolName:    m.ToolName,
		}
	}

	artificialMessages := make([]messages.Message, 0, len(blocked)+1)
	for _, call := range blocked {
		artificialMessages = append(artificialMessages, errorToolMessage(call.ID, call.Name, buildToolLimitToolMessage(m.ToolName)))
	}

	update := map[string]any{
		ThreadToolCallCountKey: threadCounts,
		RunToolCallCountKey:    runCounts,
		"messages":             artificialMessages,
	}

	if m.ExitBehavior == "end" {
		if otherTools := m.otherPendingTools(lastAI.ToolCalls); len(otherTools) > 0 {
			return nil, fmt.Errorf("cannot end execution with other tool calls pending. Found calls to: %s", stringsJoin(otherTools, ", "))
		}
		artificialMessages = append(artificialMessages, messages.AI(buildFinalToolLimitMessage(finalThreadCount+len(blocked), finalRunCount, m.ThreadLimit, m.RunLimit, m.ToolName)))
		update["messages"] = artificialMessages
		update["jump_to"] = "end"
	}

	return update, nil
}

func (m *ToolCallLimitMiddleware) countKey() string {
	if m.ToolName == "" {
		return allToolsCountKey
	}
	return m.ToolName
}

func (m *ToolCallLimitMiddleware) matches(call messages.ToolCall) bool {
	return m.ToolName == "" || call.Name == m.ToolName
}

func (m *ToolCallLimitMiddleware) wouldExceed(threadCount, runCount int) bool {
	return (m.ThreadLimit != nil && threadCount+1 > *m.ThreadLimit) ||
		(m.RunLimit != nil && runCount+1 > *m.RunLimit)
}

func (m *ToolCallLimitMiddleware) separateToolCalls(calls []messages.ToolCall, threadCount, runCount int) ([]messages.ToolCall, []messages.ToolCall, int, int) {
	allowed := []messages.ToolCall{}
	blocked := []messages.ToolCall{}
	for _, call := range calls {
		if !m.matches(call) {
			continue
		}
		if m.wouldExceed(threadCount, runCount) {
			blocked = append(blocked, call)
			continue
		}
		allowed = append(allowed, call)
		threadCount++
		runCount++
	}
	return allowed, blocked, threadCount, runCount
}

func (m *ToolCallLimitMiddleware) otherPendingTools(calls []messages.ToolCall) []string {
	if m.ToolName == "" {
		return nil
	}
	seen := map[string]bool{}
	for _, call := range calls {
		if call.Name != m.ToolName {
			seen[call.Name] = true
		}
	}
	out := make([]string, 0, len(seen))
	for name := range seen {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func lastAIMessage(state map[string]any) *messages.Message {
	if state == nil {
		return nil
	}
	raw, ok := state["messages"]
	if !ok {
		return nil
	}
	msgs, ok := raw.([]messages.Message)
	if !ok {
		return nil
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleAI {
			return &msgs[i]
		}
	}
	return nil
}

func mapStringIntFromState(state map[string]any, key string) map[string]int {
	out := map[string]int{}
	if state == nil {
		return out
	}
	switch counts := state[key].(type) {
	case map[string]int:
		for name, count := range counts {
			out[name] = count
		}
	case map[string]any:
		for name, value := range counts {
			out[name] = intFromAny(value)
		}
	}
	return out
}

func intFromAny(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	default:
		return 0
	}
}

func buildToolLimitToolMessage(toolName string) string {
	if toolName != "" {
		return fmt.Sprintf("Tool call limit exceeded. Do not call '%s' again.", toolName)
	}
	return "Tool call limit exceeded. Do not make additional tool calls."
}

func buildFinalToolLimitMessage(threadCount, runCount int, threadLimit, runLimit *int, toolName string) string {
	toolDesc := "Tool"
	if toolName != "" {
		toolDesc = fmt.Sprintf("'%s' tool", toolName)
	}
	limits := []string{}
	if threadLimit != nil && threadCount > *threadLimit {
		limits = append(limits, fmt.Sprintf("thread limit exceeded (%d/%d calls)", threadCount, *threadLimit))
	}
	if runLimit != nil && runCount > *runLimit {
		limits = append(limits, fmt.Sprintf("run limit exceeded (%d/%d calls)", runCount, *runLimit))
	}
	return fmt.Sprintf("%s call limit reached: %s.", toolDesc, stringsJoin(limits, " and "))
}
