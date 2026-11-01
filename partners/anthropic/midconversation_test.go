package anthropic

// Mid-conversation system message and tool-change placement, gated on the
// model profile (Python #41180, langchain-anthropic 1.7.7
// chat_models.py:_format_messages). Models are taken from the materialized
// modelprofiles bundle:
//
//   - claude-opus-5:     mid_conversation_system_messages=true, mid_conversation_tools=true
//   - claude-sonnet-5:   mid_conversation_system_messages=true, mid_conversation_tools=false
//   - claude-sonnet-4-5: neither key (legacy hoisting behavior)

import (
	"log/slog"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// captureAnthropicWarnings swaps the default slog logger for one writing into
// a buffer so tests can assert warning parity with Python's warnings.warn.
func captureAnthropicWarnings(t *testing.T) *strings.Builder {
	t.Helper()
	var buf strings.Builder
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return &buf
}

func midConvModel(t *testing.T, model string, request *map[string]any) ChatModel {
	t.Helper()
	server := newTestServer(t, request)
	t.Cleanup(server.Close)
	return NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel(model),
	)
}

func midConvInvoke(t *testing.T, model string, input []messages.Message) map[string]any {
	t.Helper()
	var request map[string]any
	m := midConvModel(t, model, &request)
	if _, err := m.Invoke(t.Context(), input); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	return request
}

// midConvRoles returns the wire role sequence of the messages array.
func midConvRoles(t *testing.T, request map[string]any) []string {
	t.Helper()
	raw, ok := request["messages"].([]any)
	if !ok {
		t.Fatalf("missing messages: %+v", request)
	}
	roles := make([]string, 0, len(raw))
	for _, item := range raw {
		message, ok := item.(map[string]any)
		if !ok {
			t.Fatalf("message not an object: %+v", item)
		}
		role, _ := message["role"].(string)
		roles = append(roles, role)
	}
	return roles
}

// midConvMessage returns the i-th wire message object.
func midConvMessage(t *testing.T, request map[string]any, i int) map[string]any {
	t.Helper()
	raw, ok := request["messages"].([]any)
	if !ok || i >= len(raw) {
		t.Fatalf("missing message %d: %+v", i, request)
	}
	message, ok := raw[i].(map[string]any)
	if !ok {
		t.Fatalf("message %d not an object: %+v", i, raw[i])
	}
	return message
}

// systemWithBlocks builds a system message carrying raw provider blocks.
func systemWithBlocks(blocks ...map[string]any) messages.Message {
	sys := messages.System("")
	for _, block := range blocks {
		sys.ContentBlocks = append(sys.ContentBlocks, messages.ParseContentBlock(block))
	}
	return sys
}

func toolRemovalBlock() map[string]any {
	return map[string]any{
		"type": "tool_removal",
		"tool": map[string]any{"type": "tool_reference", "name": "get_weather"},
	}
}

func inlineToolAdditionBlock() map[string]any {
	return map[string]any{
		"type": "tool_addition",
		"tool": map[string]any{
			"type": "tool_definition",
			"definition": map[string]any{
				"name":         "get_time",
				"description":  "Get the current time.",
				"input_schema": map[string]any{"type": "object", "properties": map[string]any{}},
			},
		},
	}
}

// TestMidConversationSystemInPlaceWhenProfileAllows mirrors Python
// test_sonnet55.test_mid_conversation_system: on a model whose profile allows
// mid-conversation system messages, a non-leading SystemMessage keeps its
// position as a system-role turn (string content stays a bare string) and no
// warnings fire.
func TestMidConversationSystemInPlaceWhenProfileAllows(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.System("initial"),
		messages.Human("hello"),
		messages.System("new instructions"),
		messages.AI("answer"),
		messages.Human("next"),
	})

	if request["system"] != "initial" {
		t.Fatalf("system field: %+v", request["system"])
	}
	roles := midConvRoles(t, request)
	want := []string{"user", "system", "assistant", "user"}
	if len(roles) != len(want) {
		t.Fatalf("roles: got %v want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("roles: got %v want %v", roles, want)
		}
	}
	if got := midConvMessage(t, request, 1)["content"]; got != "new instructions" {
		t.Fatalf("in-place system content should stay a bare string, got %#v", got)
	}
	if logs.Len() > 0 {
		t.Fatalf("in-place placement must not warn, got: %s", logs.String())
	}
}

// TestMidConversationSystemTrailingInPlace mirrors Python
// test__format_messages_non_leading_system_only_run_sent_in_place.
func TestMidConversationSystemTrailingInPlace(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("Review foo()"),
		messages.System("Be concise."),
	})

	if _, present := request["system"]; present {
		t.Fatalf("system field should be absent: %+v", request["system"])
	}
	roles := midConvRoles(t, request)
	if len(roles) != 2 || roles[0] != "user" || roles[1] != "system" {
		t.Fatalf("roles: %v", roles)
	}
	if got := midConvMessage(t, request, 1)["content"]; got != "Be concise." {
		t.Fatalf("trailing system content: %#v", got)
	}
	if logs.Len() > 0 {
		t.Fatalf("trailing in-place placement must not warn, got: %s", logs.String())
	}
}

// TestMidConversationSystemBeforeAITurnInPlace mirrors Python
// test__format_messages_system_between_user_and_ai_sent_in_place: a system
// run immediately before an assistant turn is a legal in-place position.
func TestMidConversationSystemBeforeAITurnInPlace(t *testing.T) {
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("Review foo()"),
		messages.System("Be concise."),
		messages.AI("Looks fine."),
	})

	roles := midConvRoles(t, request)
	want := []string{"user", "system", "assistant"}
	if len(roles) != len(want) {
		t.Fatalf("roles: got %v want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("roles: got %v want %v", roles, want)
		}
	}
}

// TestMidConversationSystemAfterToolTurnInPlace mirrors Python
// test__format_messages_system_after_tool_message_sent_in_place: a system
// message after a folded tool-result (user-role) turn is legal.
func TestMidConversationSystemAfterToolTurnInPlace(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{
		ID:   "toolu_1",
		Name: "search",
		Args: map[string]any{"q": "foo"},
	}}
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("Review foo()"),
		ai,
		messages.Tool("toolu_1", "results"),
		messages.System("Be concise."),
	})

	roles := midConvRoles(t, request)
	want := []string{"user", "assistant", "user", "system"}
	if len(roles) != len(want) {
		t.Fatalf("roles: got %v want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("roles: got %v want %v", roles, want)
		}
	}
	if _, present := request["system"]; present {
		t.Fatalf("system field should be absent: %+v", request["system"])
	}
}

// TestMidConversationSystemAfterPlainAITurnHoisted mirrors Python
// test__format_messages_system_after_plain_ai_turn_hoisted: an assistant turn
// that does not end in a server tool result is an illegal predecessor, so the
// system message is hoisted with a warning even on a supporting model.
func TestMidConversationSystemAfterPlainAITurnHoisted(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("Review foo()"),
		messages.AI("Looks fine."),
		messages.System("Be concise."),
	})

	if request["system"] != "Be concise." {
		t.Fatalf("system field: %+v", request["system"])
	}
	roles := midConvRoles(t, request)
	if len(roles) != 2 || roles[0] != "user" || roles[1] != "assistant" {
		t.Fatalf("roles: %v", roles)
	}
	if !strings.Contains(logs.String(), "hoisted") {
		t.Fatalf("expected hoist warning, got: %s", logs.String())
	}
}

// TestMidConversationSystemFollowedByUserTurnHoisted mirrors Python
// test__format_messages_system_followed_by_user_turn_hoisted: a legal-position
// system run followed by a user turn is hoisted with a warning.
func TestMidConversationSystemFollowedByUserTurnHoisted(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("Review foo()"),
		messages.System("Be concise."),
		messages.Human("Review bar()"),
	})

	if request["system"] != "Be concise." {
		t.Fatalf("system field: %+v", request["system"])
	}
	roles := midConvRoles(t, request)
	if len(roles) != 2 || roles[0] != "user" || roles[1] != "user" {
		t.Fatalf("roles: %v", roles)
	}
	if !strings.Contains(logs.String(), "hoisted") {
		t.Fatalf("expected hoist warning, got: %s", logs.String())
	}
}

// TestMidConversationSystemHoistedWhenProfileDoesNotAllow mirrors Python
// test_non_leading_system_hoisted_when_profile_does_not_declare_it: on a model
// whose profile does not declare the capability, the legacy hoisting behavior
// stays, plus the hoist warning.
func TestMidConversationSystemHoistedWhenProfileDoesNotAllow(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-4-5", []messages.Message{
		messages.Human("Review foo()"),
		messages.System("Be concise."),
	})

	if request["system"] != "Be concise." {
		t.Fatalf("system field: %+v", request["system"])
	}
	roles := midConvRoles(t, request)
	if len(roles) != 1 || roles[0] != "user" {
		t.Fatalf("roles: %v", roles)
	}
	if !strings.Contains(logs.String(), "hoisted") {
		t.Fatalf("expected hoist warning, got: %s", logs.String())
	}
}

// TestMidConversationSystemHoistStillCoalescesWhenProfileDoesNotAllow checks
// the compat side of the gate: without the capability, multiple hoisted system
// messages keep coalescing into the top-level system field (the pre-#41180
// behavior), unlike the profile-enabled path which errors.
func TestMidConversationSystemHoistStillCoalescesWhenProfileDoesNotAllow(t *testing.T) {
	captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-4-5", []messages.Message{
		messages.System("first"),
		messages.Human("Review foo()"),
		messages.System("second"),
	})

	if request["system"] != "first\nsecond" {
		t.Fatalf("system field should keep coalescing: %+v", request["system"])
	}
	roles := midConvRoles(t, request)
	if len(roles) != 1 || roles[0] != "user" {
		t.Fatalf("roles: %v", roles)
	}
}

// TestMidConversationSecondUnplaceableRunErrors mirrors Python
// test__format_messages_second_unplaceable_system_run_raises: on the
// profile-enabled path, hoisting a second system run while the top-level
// system field is already claimed is an error.
func TestMidConversationSecondUnplaceableRunErrors(t *testing.T) {
	var request map[string]any
	m := midConvModel(t, "claude-opus-5", &request)
	_, err := m.Invoke(t.Context(), []messages.Message{
		messages.System("You are a code reviewer."),
		messages.Human("Review foo()"),
		messages.System("Be concise."),
		messages.Human("Review bar()"),
	})
	if err == nil || !strings.Contains(err.Error(), "non-consecutive") {
		t.Fatalf("expected non-consecutive system error, got: %v", err)
	}
}

// TestMidConversationLeadingRunHoistedLaterRunInPlace mirrors Python
// test__format_messages_leading_run_hoisted_and_later_run_in_place.
func TestMidConversationLeadingRunHoistedLaterRunInPlace(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.System("You are a code reviewer."),
		messages.Human("Review foo()"),
		messages.System("Be concise."),
	})

	if request["system"] != "You are a code reviewer." {
		t.Fatalf("system field: %+v", request["system"])
	}
	roles := midConvRoles(t, request)
	want := []string{"user", "system"}
	if len(roles) != len(want) || roles[0] != want[0] || roles[1] != want[1] {
		t.Fatalf("roles: got %v want %v", roles, want)
	}
	if strings.Contains(logs.String(), "hoisted") {
		t.Fatalf("legal placements must not warn, got: %s", logs.String())
	}
}

// TestMidConversationToolChangeBlocksForwardedWhenProfileAllows mirrors
// Python test__format_messages_system_tool_change_block_sent_in_place and
// test__format_messages_system_tool_change_block_beside_text: tool_removal /
// tool_addition blocks on an in-place system message reach the wire verbatim.
func TestMidConversationToolChangeBlocksForwardedWhenProfileAllows(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("Review foo()"),
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			toolRemovalBlock(),
		),
		messages.AI("Looks fine."),
	})

	roles := midConvRoles(t, request)
	want := []string{"user", "system", "assistant"}
	if len(roles) != len(want) {
		t.Fatalf("roles: got %v want %v", roles, want)
	}
	content, ok := midConvMessage(t, request, 1)["content"].([]any)
	if !ok || len(content) != 2 {
		t.Fatalf("in-place system content: %+v", midConvMessage(t, request, 1)["content"])
	}
	text := content[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "Be concise." {
		t.Fatalf("text block: %+v", text)
	}
	removal := content[1].(map[string]any)
	if removal["type"] != "tool_removal" {
		t.Fatalf("tool_removal block: %+v", removal)
	}
	tool, _ := removal["tool"].(map[string]any)
	if tool["type"] != "tool_reference" || tool["name"] != "get_weather" {
		t.Fatalf("tool_removal tool map: %+v", tool)
	}
	if logs.Len() > 0 {
		t.Fatalf("forwarding tool changes must not warn, got: %s", logs.String())
	}
}

// TestMidConversationInlineToolAdditionForwarded mirrors Python
// test_inline_tool_definition_auto_appends_beta's payload side: an inline
// tool_definition inside a tool_addition block is forwarded unchanged (the
// docstring example for adding a tool mid-conversation).
func TestMidConversationInlineToolAdditionForwarded(t *testing.T) {
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("hello"),
		systemWithBlocks(inlineToolAdditionBlock()),
	})

	roles := midConvRoles(t, request)
	if len(roles) != 2 || roles[1] != "system" {
		t.Fatalf("roles: %v", roles)
	}
	content, ok := midConvMessage(t, request, 1)["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("in-place system content: %+v", midConvMessage(t, request, 1)["content"])
	}
	block := content[0].(map[string]any)
	if block["type"] != "tool_addition" {
		t.Fatalf("tool_addition block: %+v", block)
	}
	tool, _ := block["tool"].(map[string]any)
	if tool["type"] != "tool_definition" {
		t.Fatalf("inline tool_definition: %+v", tool)
	}
	definition, _ := tool["definition"].(map[string]any)
	if definition["name"] != "get_time" {
		t.Fatalf("inline definition: %+v", definition)
	}
}

// TestMidConversationToolChangeDroppedOnToolsFalseModel is the D8 gate: on
// claude-sonnet-5 the profile allows mid-conversation system messages but not
// tool changes, so the system turn keeps its position while the tool-change
// block is dropped with a warning (and no hoist warning fires).
func TestMidConversationToolChangeDroppedOnToolsFalseModel(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-5", []messages.Message{
		messages.Human("Review foo()"),
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			toolRemovalBlock(),
		),
		messages.AI("Looks fine."),
	})

	roles := midConvRoles(t, request)
	want := []string{"user", "system", "assistant"}
	if len(roles) != len(want) {
		t.Fatalf("roles: got %v want %v", roles, want)
	}
	content, ok := midConvMessage(t, request, 1)["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("tool-change block should be dropped: %+v", midConvMessage(t, request, 1)["content"])
	}
	if text := content[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("surviving text block: %+v", text)
	}
	logged := logs.String()
	if !strings.Contains(logged, "tool_removal") {
		t.Fatalf("expected tool-change drop warning, got: %s", logged)
	}
	if strings.Contains(logged, "hoisted") {
		t.Fatalf("system message stays in place; no hoist warning expected, got: %s", logged)
	}
}

// TestMidConversationToolChangeDroppedWhenHoistedOnAllowedModel mirrors
// Python test__format_messages_system_tool_change_block_stripped_on_illegal_position:
// a hoisted system message strips tool-change blocks even on a supporting
// model, warning about both the drop and the hoist.
func TestMidConversationToolChangeDroppedWhenHoistedOnAllowedModel(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("Review foo()"),
		messages.AI("Looks fine."),
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			toolRemovalBlock(),
		),
	})

	system, ok := request["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system field: %+v", request["system"])
	}
	if text := system[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("surviving system block: %+v", text)
	}
	logged := logs.String()
	if !strings.Contains(logged, "tool_removal") {
		t.Fatalf("expected tool-change drop warning, got: %s", logged)
	}
	if !strings.Contains(logged, "hoisted") {
		t.Fatalf("expected hoist warning, got: %s", logged)
	}
}

// TestMidConversationToolChangeDroppedOnUnsupportedModel mirrors Python
// test__format_messages_system_tool_change_block_stripped_on_unsupported_model:
// a model without the capability hoists the system message and strips the
// tool-change block.
func TestMidConversationToolChangeDroppedOnUnsupportedModel(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-4-5", []messages.Message{
		messages.Human("Review foo()"),
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			toolRemovalBlock(),
		),
		messages.AI("Looks fine."),
	})

	system, ok := request["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system field: %+v", request["system"])
	}
	if text := system[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("surviving system block: %+v", text)
	}
	logged := logs.String()
	if !strings.Contains(logged, "tool_removal") {
		t.Fatalf("expected tool-change drop warning, got: %s", logged)
	}
	if !strings.Contains(logged, "hoisted") {
		t.Fatalf("expected hoist warning, got: %s", logged)
	}
}

// TestMidConversationLeadingSystemToolChangeForwarded mirrors Python
// test__format_messages_leading_system_tool_change_block_forwarded: the
// leading system run may carry tool-change blocks into the top-level system
// field when the profile declares mid_conversation_tools (Anthropic validates
// them there).
func TestMidConversationLeadingSystemToolChangeForwarded(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		systemWithBlocks(toolRemovalBlock()),
		messages.Human("Review foo()"),
	})

	system, ok := request["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system field: %+v", request["system"])
	}
	block, _ := system[0].(map[string]any)
	if block["type"] != "tool_removal" {
		t.Fatalf("leading tool_removal block: %+v", block)
	}
	if tool, _ := block["tool"].(map[string]any); tool["name"] != "get_weather" {
		t.Fatalf("leading tool_removal tool map: %+v", tool)
	}
	if logs.Len() > 0 {
		t.Fatalf("forwarding a leading tool change must not warn, got: %s", logs.String())
	}
}

// TestMidConversationLeadingSystemToolChangeDroppedOnToolsFalseModel is the
// leading-run side of the mid_conversation_tools gate: on claude-sonnet-5 the
// block is dropped from the top-level system field with a warning, and no
// hoist warning fires (the leading run is never hoisted).
func TestMidConversationLeadingSystemToolChangeDroppedOnToolsFalseModel(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-5", []messages.Message{
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			toolRemovalBlock(),
		),
		messages.Human("Review foo()"),
	})

	system, ok := request["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system field: %+v", request["system"])
	}
	if text := system[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("surviving system block: %+v", text)
	}
	logged := logs.String()
	if !strings.Contains(logged, "tool_removal") {
		t.Fatalf("expected tool-change drop warning, got: %s", logged)
	}
	if strings.Contains(logged, "hoisted") {
		t.Fatalf("leading run is not hoisted; no hoist warning expected, got: %s", logged)
	}
}

// TestMidConversationSystemOmittedWhenNarrowedToNothing mirrors Python
// _format_in_place_system_messages: a system run whose blocks are all narrowed
// away is omitted entirely rather than sent as an empty system turn.
func TestMidConversationSystemOmittedWhenNarrowedToNothing(t *testing.T) {
	captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-5", []messages.Message{
		messages.Human("Review foo()"),
		systemWithBlocks(toolRemovalBlock()),
		messages.AI("Looks fine."),
	})

	roles := midConvRoles(t, request)
	want := []string{"user", "assistant"}
	if len(roles) != len(want) {
		t.Fatalf("roles: got %v want %v", roles, want)
	}
	if _, present := request["system"]; present {
		t.Fatalf("system field should be absent: %+v", request["system"])
	}
}

// TestMidConversationConsecutiveSystemRunMergesToOneTurn mirrors Python
// _merge_messages coalescing (test__format_messages_with_multiple_system): a
// run of consecutive system messages becomes one system turn whose content is
// a block array.
func TestMidConversationConsecutiveSystemRunMergesToOneTurn(t *testing.T) {
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("baz"),
		messages.System("bar"),
		messages.System("baz"),
	})

	roles := midConvRoles(t, request)
	if len(roles) != 2 || roles[1] != "system" {
		t.Fatalf("roles: %v", roles)
	}
	content, ok := midConvMessage(t, request, 1)["content"].([]any)
	if !ok || len(content) != 2 {
		t.Fatalf("merged run content: %+v", midConvMessage(t, request, 1)["content"])
	}
	for i, want := range []string{"bar", "baz"} {
		block, _ := content[i].(map[string]any)
		if block["type"] != "text" || block["text"] != want {
			t.Fatalf("merged block %d: %+v", i, block)
		}
	}
}
