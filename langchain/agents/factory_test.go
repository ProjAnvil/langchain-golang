package agents

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestFetchLastAIAndToolMessagesNormal(t *testing.T) {
	input := []messages.Message{
		messages.Human("Hello"),
		{
			Role:    messages.RoleAI,
			Content: "Hi there!",
			ToolCalls: []messages.ToolCall{
				{Name: "test", ID: "1", Args: map[string]any{}},
			},
		},
		messages.Tool("1", "Tool result"),
	}

	aiMsg, toolMsgs := FetchLastAIAndToolMessages(input)
	if aiMsg == nil {
		t.Fatal("expected AI message")
	}
	if aiMsg.Content != "Hi there!" {
		t.Fatalf("AI content mismatch: got %q", aiMsg.Content)
	}
	if len(toolMsgs) != 1 {
		t.Fatalf("expected one tool message, got %d", len(toolMsgs))
	}
	if toolMsgs[0].Content != "Tool result" {
		t.Fatalf("tool content mismatch: got %q", toolMsgs[0].Content)
	}
}

func TestFetchLastAIAndToolMessagesMultipleAI(t *testing.T) {
	input := []messages.Message{
		messages.Human("First question"),
		{Role: messages.RoleAI, Content: "First answer", ID: "ai1"},
		messages.Human("Second question"),
		{Role: messages.RoleAI, Content: "Second answer", ID: "ai2"},
	}

	aiMsg, toolMsgs := FetchLastAIAndToolMessages(input)
	if aiMsg == nil {
		t.Fatal("expected AI message")
	}
	if aiMsg.Content != "Second answer" || aiMsg.ID != "ai2" {
		t.Fatalf("wrong AI message: %#v", *aiMsg)
	}
	if len(toolMsgs) != 0 {
		t.Fatalf("expected no tool messages, got %#v", toolMsgs)
	}
}

func TestFetchLastAIAndToolMessagesNoAIMessage(t *testing.T) {
	input := []messages.Message{
		messages.Human("Hello"),
		messages.System("You are a helpful assistant"),
	}

	aiMsg, toolMsgs := FetchLastAIAndToolMessages(input)
	if aiMsg != nil {
		t.Fatalf("expected nil AI message, got %#v", *aiMsg)
	}
	if len(toolMsgs) != 0 {
		t.Fatalf("expected no tool messages, got %#v", toolMsgs)
	}
}

func TestFetchLastAIAndToolMessagesEmptyList(t *testing.T) {
	aiMsg, toolMsgs := FetchLastAIAndToolMessages(nil)
	if aiMsg != nil {
		t.Fatalf("expected nil AI message, got %#v", *aiMsg)
	}
	if len(toolMsgs) != 0 {
		t.Fatalf("expected no tool messages, got %#v", toolMsgs)
	}
}

func TestFetchLastAIAndToolMessagesOnlyHumanMessages(t *testing.T) {
	input := []messages.Message{
		messages.Human("Hello"),
		messages.Human("Are you there?"),
	}

	aiMsg, toolMsgs := FetchLastAIAndToolMessages(input)
	if aiMsg != nil {
		t.Fatalf("expected nil AI message, got %#v", *aiMsg)
	}
	if len(toolMsgs) != 0 {
		t.Fatalf("expected no tool messages, got %#v", toolMsgs)
	}
}

func TestFetchLastAIAndToolMessagesAIWithoutToolCalls(t *testing.T) {
	input := []messages.Message{
		messages.Human("Hello"),
		messages.AI("Hi! How can I help you today?"),
	}

	aiMsg, toolMsgs := FetchLastAIAndToolMessages(input)
	if aiMsg == nil {
		t.Fatal("expected AI message")
	}
	if aiMsg.Content != "Hi! How can I help you today?" {
		t.Fatalf("AI content mismatch: got %q", aiMsg.Content)
	}
	if len(toolMsgs) != 0 {
		t.Fatalf("expected no tool messages, got %#v", toolMsgs)
	}
}

func TestSupportsProviderStrategyBlocksGeminiV2WithTools(t *testing.T) {
	model := ModelInfo{
		ModelName: "gemini-2.5-flash",
		Profile:   map[string]any{"structured_output": true},
	}
	if SupportsProviderStrategy(model, []any{"get_weather"}) {
		t.Fatal("expected Gemini 2 with tools to be blocked")
	}
}

func TestSupportsProviderStrategyAllowsGeminiV3WithTools(t *testing.T) {
	model := ModelInfo{
		ModelName: "gemini-3.1-pro-preview",
		Profile:   map[string]any{"structured_output": true},
	}
	if !SupportsProviderStrategy(model, []any{"get_weather"}) {
		t.Fatal("expected Gemini 3 with tools to be allowed")
	}
}

func TestSupportsProviderStrategyBlocksGeminiLatestAliasesWithTools(t *testing.T) {
	for _, alias := range []string{"gemini-flash-latest", "gemini-flash-lite-latest"} {
		t.Run(alias, func(t *testing.T) {
			model := ModelInfo{
				ModelName: alias,
				Profile:   map[string]any{"structured_output": true},
			}
			if SupportsProviderStrategy(model, []any{"get_weather"}) {
				t.Fatal("expected Gemini latest alias with tools to be blocked")
			}
		})
	}
}

func TestSupportsProviderStrategyFallbackAllowsKnownModels(t *testing.T) {
	modelNames := []string{
		"gpt-4.1",
		"gpt-4.1-mini",
		"gpt-4o",
		"gpt-4o-mini",
		"gpt-5",
		"gpt-5-mini",
		"gpt-5.1",
		"gpt-5.1-codex",
		"gpt-5.2",
		"gpt-5.2-2025-12-01",
		"gpt-5.2-chat-latest",
		"gpt-5.2-codex",
		"gpt-5.3",
		"gpt-5.3-codex-spark",
		"gpt-5.4",
		"gpt-5.4-2026-03-05",
		"gpt-5.4-mini",
		"gpt-5.4-nano",
		"gpt-5.5-pro",
		"openai:gpt-5.5",
		"openai/gpt-5-mini",
		"openai.gpt-5.4-mini",
		"claude-fable-5",
		"claude-mythos-5",
		"claude-haiku-4-5",
		"claude-haiku-4-5-20251001",
		"claude-opus-4-5",
		"claude-opus-4-5-20251101",
		"claude-opus-4-6",
		"claude-opus-4-7",
		"claude-opus-4-8",
		"claude-sonnet-4-5",
		"claude-sonnet-4-5-20250929",
		"claude-sonnet-4-6",
		"anthropic/claude-sonnet-4-5",
		"anthropic.claude-opus-4-6",
		"anthropic.claude-sonnet-4-5-20250929-v1:0",
		"grok-4.20-0309-reasoning",
		"grok-4.3",
		"grok-build-0.1",
	}

	for _, modelName := range modelNames {
		t.Run(modelName, func(t *testing.T) {
			if !SupportsProviderStrategy(modelName, nil) {
				t.Fatal("expected provider strategy support")
			}
		})
	}
}

func TestSupportsProviderStrategyFallbackBlocksOverbroadMatches(t *testing.T) {
	modelNames := []string{
		"gpt-5.2-pro",
		"gpt-5.4-pro",
		"gpt-oss-120b",
		"openai/gpt-oss-120b:free",
		"claude-3-5-sonnet-20241022",
		"claude-opus-4-1",
		"claude-opus-4-1-20250805",
		"claude-opus-4-20250514",
		"claude-opus-4-0",
		"grok-imagine-image",
		"grok-imagine-video",
		"solar-pro3",
		"sao10k/l3.1-70b-hanami-x1",
	}

	for _, modelName := range modelNames {
		t.Run(modelName, func(t *testing.T) {
			if SupportsProviderStrategy(modelName, nil) {
				t.Fatal("expected provider strategy support to be blocked")
			}
		})
	}
}

func TestSupportsProviderStrategyStringPathIgnoresTools(t *testing.T) {
	if !SupportsProviderStrategy("gpt-5.5", nil) {
		t.Fatal("expected gpt-5.5 to be supported")
	}
	if !SupportsProviderStrategy("gpt-5.5", []any{"get_weather"}) {
		t.Fatal("expected string fallback path to ignore tools")
	}
}

func TestSupportsProviderStrategyModelInfoPointer(t *testing.T) {
	var nilInfo *ModelInfo
	if SupportsProviderStrategy(nilInfo, nil) {
		t.Fatal("nil *ModelInfo must not support provider strategy")
	}

	// A ModelInfo with no identifying name falls through to "" and is rejected.
	if SupportsProviderStrategy(&ModelInfo{}, nil) {
		t.Fatal("empty ModelInfo must not support provider strategy")
	}

	// Name resolution order: ModelName, then Model, then ModelID.
	if !SupportsProviderStrategy(&ModelInfo{Model: "gpt-4o"}, nil) {
		t.Fatal("gpt-4o via Model field should match the fallback list")
	}
	if !SupportsProviderStrategy(&ModelInfo{ModelID: "openai:gpt-4.1"}, nil) {
		t.Fatal("gpt-4.1 via ModelID field should match the fallback list")
	}

	// A profile declaring structured_output support short-circuits the
	// fallback list, even for an unknown model name.
	supported := &ModelInfo{ModelName: "custom-model", Profile: map[string]any{"structured_output": true}}
	if !SupportsProviderStrategy(supported, nil) {
		t.Fatal("profile-declared structured_output support should be honored")
	}

	// Gemini (< 3) with tools blocks profile-declared support.
	blocked := &ModelInfo{ModelName: "gemini-2.5-pro", Profile: map[string]any{"structured_output": true}}
	if SupportsProviderStrategy(blocked, []any{"some-tool"}) {
		t.Fatal("gemini-2.x with tools must block provider strategy")
	}
	// Without tools the same model is allowed through the profile path.
	if !SupportsProviderStrategy(blocked, nil) {
		t.Fatal("gemini-2.x without tools should be allowed via the profile")
	}

	// A non-bool structured_output profile value is ignored; an unknown model
	// then fails the fallback list.
	notBool := &ModelInfo{ModelName: "custom-model", Profile: map[string]any{"structured_output": "yes"}}
	if SupportsProviderStrategy(notBool, nil) {
		t.Fatal("non-bool structured_output profile value must be ignored")
	}
}

func TestSupportsProviderStrategyModelInfoValue(t *testing.T) {
	info := ModelInfo{ModelName: "claude-sonnet-4-6", Profile: map[string]any{"structured_output": true}}
	if !SupportsProviderStrategy(info, nil) {
		t.Fatal("ModelInfo value with profile support should be accepted")
	}
	if !SupportsProviderStrategy(ModelInfo{ModelName: "grok-4"}, nil) {
		t.Fatal("grok-4 should match the fallback list")
	}
}

func TestFirstNonEmpty(t *testing.T) {
	if got := firstNonEmpty("", "", ""); got != "" {
		t.Fatalf("expected empty, got %q", got)
	}
	if got := firstNonEmpty("", "b", "c"); got != "b" {
		t.Fatalf("expected first non-empty, got %q", got)
	}
}
