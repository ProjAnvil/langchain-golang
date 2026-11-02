package anthropic

// Mid-conversation system message and tool-change placement, gated on the
// model profile (Python #41180, langchain-anthropic 1.7.7
// chat_models.py:_format_messages). Models are taken from the materialized
// modelprofiles bundle:
//
//   - claude-opus-5:     mid_conversation_system_messages=true, mid_conversation_tools=true
//   - claude-sonnet-5:   mid_conversation_system_messages=true, mid_conversation_tools=false
//   - claude-sonnet-4-5: neither key (legacy hoisting behavior)
//
// The profile gates only the in-place (non-leading) mechanism: Python formats
// the leading system run with preserve_tool_changes=True unconditionally
// (#40758, chat_models.py:741-747), so leading-run tool-change blocks forward
// for every model, profile or not.

import (
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
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
// field for every model, profile or not (Anthropic validates them there).
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

// TestMidConversationLeadingSystemToolChangeForwardedOnToolsFalseModel pins
// the leading run against Python's unconditional preserve_tool_changes
// (#40758, chat_models.py:741-747): the mid_conversation_tools profile key
// gates only the in-place path, so on claude-sonnet-5 (system messages
// allowed, tool changes not) the leading run's tool-change block still
// reaches the top-level system field, with no warnings (the leading run is
// never hoisted).
func TestMidConversationLeadingSystemToolChangeForwardedOnToolsFalseModel(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-5", []messages.Message{
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			toolRemovalBlock(),
		),
		messages.Human("Review foo()"),
	})

	system, ok := request["system"].([]any)
	if !ok || len(system) != 2 {
		t.Fatalf("system field: %+v", request["system"])
	}
	if text := system[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("leading text block: %+v", text)
	}
	removal, _ := system[1].(map[string]any)
	if removal["type"] != "tool_removal" {
		t.Fatalf("leading tool_removal block: %+v", removal)
	}
	if tool, _ := removal["tool"].(map[string]any); tool["name"] != "get_weather" {
		t.Fatalf("leading tool_removal tool map: %+v", tool)
	}
	if logs.Len() > 0 {
		t.Fatalf("forwarding a leading tool change must not warn, got: %s", logs.String())
	}
}

// TestMidConversationLeadingSystemToolChangeForwardedWithoutProfile covers
// the models Python's leading-run test never needed a capability flag for:
// the leading run's preserve_tool_changes has no profile gate at all
// (#40758, chat_models.py:741-747), so an ID whose profile carries no
// mid_conversation keys — and an ID absent from the bundle entirely —
// forward the block too.
func TestMidConversationLeadingSystemToolChangeForwardedWithoutProfile(t *testing.T) {
	models := map[string]string{
		"claude-sonnet-4-5": "in the bundle, no mid_conversation keys",
		"claude-future-9":   "absent from the bundle entirely",
	}
	for model, why := range models {
		t.Run(model+" ("+why+")", func(t *testing.T) {
			logs := captureAnthropicWarnings(t)
			request := midConvInvoke(t, model, []messages.Message{
				systemWithBlocks(toolRemovalBlock()),
				messages.Human("Review foo()"),
			})

			system, ok := request["system"].([]any)
			if !ok || len(system) != 1 {
				t.Fatalf("system field: %+v", request["system"])
			}
			if block, _ := system[0].(map[string]any); block["type"] != "tool_removal" {
				t.Fatalf("leading tool_removal block: %+v", block)
			}
			if logs.Len() > 0 {
				t.Fatalf("forwarding a leading tool change must not warn, got: %s", logs.String())
			}
		})
	}
}

// --- Beta-header auto-append for tool-change blocks (Python #41180,
// chat_models.py:2140-2161) ---

const (
	testToolChangeBeta  = "mid-conversation-tool-changes-2026-07-01"
	testInlineToolsBeta = "inline-tools-2026-09-15"
)

// midConvCapture records the wire request: parsed body, anthropic-beta header,
// and the request path (query included) so tests can assert endpoint routing.
type midConvCapture struct {
	body map[string]any
	beta string
	path string
}

// newMidConvCaptureServer serves the Messages API response while capturing
// header, path, and body of the incoming request.
func newMidConvCaptureServer(t *testing.T, captured *midConvCapture) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.path = r.URL.RequestURI()
		captured.beta = r.Header.Get("anthropic-beta")
		decodeRequest(t, w, r, &captured.body)
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`)
	}))
	t.Cleanup(server.Close)
	return server
}

func midConvInvokeCaptured(t *testing.T, model string, opts []modelconfig.Option, input []messages.Message) *midConvCapture {
	t.Helper()
	captured := &midConvCapture{}
	server := newMidConvCaptureServer(t, captured)
	options := append([]modelconfig.Option{
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel(model),
	}, opts...)
	if _, err := NewChatModel(options...).Invoke(t.Context(), input); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	return captured
}

var toolChangeConversation = []messages.Message{
	messages.Human("Review foo()"),
	systemWithBlocks(toolRemovalBlock()),
	messages.AI("Looks fine."),
}

// TestMidConversationToolChangeAutoAppendsBeta mirrors Python
// test_tool_change_block_auto_appends_beta (test_chat_models.py:5857): a
// tool-change block that reaches the wire as an in-place system turn enables
// the required beta header, and the request routes to the beta endpoint.
func TestMidConversationToolChangeAutoAppendsBeta(t *testing.T) {
	captured := midConvInvokeCaptured(t, "claude-opus-5", nil, toolChangeConversation)

	if captured.beta != testToolChangeBeta {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, testToolChangeBeta)
	}
	if captured.path != "/messages?beta=true" {
		t.Fatalf("request path: got %q want /messages?beta=true (Python routes betas-bearing payloads to the beta namespace)", captured.path)
	}
}

// TestMidConversationToolChangeBetaNotDuplicated mirrors Python
// test_tool_change_block_beta_not_duplicated (test_chat_models.py:5866): the
// auto-appended beta appears at most once when the caller already set it.
func TestMidConversationToolChangeBetaNotDuplicated(t *testing.T) {
	captured := midConvInvokeCaptured(t, "claude-opus-5",
		[]modelconfig.Option{WithBetaHeaders(testToolChangeBeta)},
		toolChangeConversation)

	if got := strings.Count(captured.beta, testToolChangeBeta); got != 1 {
		t.Fatalf("anthropic-beta header should contain the beta exactly once, got %q", captured.beta)
	}
}

// TestMidConversationToolChangeBetaMergesCallerSuppliedBetas checks the merge
// (not clobber) rule: caller-supplied beta flags are preserved alongside the
// auto-appended one.
func TestMidConversationToolChangeBetaMergesCallerSuppliedBetas(t *testing.T) {
	captured := midConvInvokeCaptured(t, "claude-opus-5",
		[]modelconfig.Option{WithBetaHeaders("context-1m-2025-08-07")},
		toolChangeConversation)

	want := "context-1m-2025-08-07," + testToolChangeBeta
	if captured.beta != want {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, want)
	}
}

// TestMidConversationInlineToolDefinitionAutoAppendsBeta mirrors Python
// test_inline_tool_definition_auto_appends_beta (test_chat_models.py:5875):
// an inline tool_definition uses the inline-tools beta.
func TestMidConversationInlineToolDefinitionAutoAppendsBeta(t *testing.T) {
	captured := midConvInvokeCaptured(t, "claude-opus-5", nil, []messages.Message{
		messages.Human("Review data"),
		systemWithBlocks(inlineToolAdditionBlock()),
	})

	if captured.beta != testInlineToolsBeta {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, testInlineToolsBeta)
	}
	if captured.path != "/messages?beta=true" {
		t.Fatalf("request path: got %q want /messages?beta=true", captured.path)
	}
}

// TestMidConversationInlineToolDefinitionSupersedesReferenceBeta mirrors
// Python test_inline_tool_definition_supersedes_reference_beta
// (test_chat_models.py:5891): one inline beta covers mixed inline and
// reference-based changes.
func TestMidConversationInlineToolDefinitionSupersedesReferenceBeta(t *testing.T) {
	captured := midConvInvokeCaptured(t, "claude-opus-5", nil, []messages.Message{
		messages.Human("Review data"),
		systemWithBlocks(toolRemovalBlock(), inlineToolAdditionBlock()),
	})

	if captured.beta != testInlineToolsBeta {
		t.Fatalf("anthropic-beta header: got %q want %q (inline beta must supersede the reference beta)", captured.beta, testInlineToolsBeta)
	}
}

// TestMidConversationExplicitInlineBetaSuppressesReferenceBeta ports Python's
// explicit-betas condition (chat_models.py:2152-2157): the reference beta is
// not appended when the inline beta is already caller-supplied, even with no
// inline block present.
func TestMidConversationExplicitInlineBetaSuppressesReferenceBeta(t *testing.T) {
	captured := midConvInvokeCaptured(t, "claude-opus-5",
		[]modelconfig.Option{WithBetaHeaders(testInlineToolsBeta)},
		toolChangeConversation)

	if captured.beta != testInlineToolsBeta {
		t.Fatalf("anthropic-beta header: got %q want %q (explicit inline beta suppresses the reference beta)", captured.beta, testInlineToolsBeta)
	}
}

// TestMidConversationLeadingToolChangeDoesNotAppendBeta pins the scan scope:
// Python scans only in-place system turns (payload messages with role
// "system"), so tool changes forwarded through the top-level system field do
// NOT trigger the beta, and the request keeps the standard endpoint.
func TestMidConversationLeadingToolChangeDoesNotAppendBeta(t *testing.T) {
	captureAnthropicWarnings(t)
	captured := midConvInvokeCaptured(t, "claude-opus-5", nil, []messages.Message{
		systemWithBlocks(toolRemovalBlock()),
		messages.Human("Review foo()"),
	})

	if captured.beta != "" {
		t.Fatalf("anthropic-beta header should stay unset, got %q", captured.beta)
	}
	if captured.path != "/messages" {
		t.Fatalf("request path: got %q want /messages", captured.path)
	}
}

// TestMidConversationPlainSystemTurnNoBeta guards the no-tool-change path: an
// in-place text system turn touches neither the header nor the endpoint.
func TestMidConversationPlainSystemTurnNoBeta(t *testing.T) {
	captured := midConvInvokeCaptured(t, "claude-opus-5", nil, []messages.Message{
		messages.Human("Review foo()"),
		messages.System("Be concise."),
	})

	if captured.beta != "" {
		t.Fatalf("anthropic-beta header should stay unset, got %q", captured.beta)
	}
	if captured.path != "/messages" {
		t.Fatalf("request path: got %q want /messages", captured.path)
	}
}

// --- thinking-display / user-profiles beta auto-append
// (chat_models.py:2130-2138, 2162-2169) ---

const (
	testThinkingDisplayBeta = "thinking-display-updates-2026-08-18"
	testUserProfilesBeta    = "user-profiles-2026-03-24"
)

// midConvInvokeChatModel invokes a caller-built ChatModel against a capture
// server, for options expressed as ChatModel methods (WithThinking,
// WithUserProfileID) rather than modelconfig.Option.
func midConvInvokeChatModel(t *testing.T, build func(baseURL string) ChatModel, input []messages.Message) *midConvCapture {
	t.Helper()
	captured := &midConvCapture{}
	server := newMidConvCaptureServer(t, captured)
	if _, err := build(server.URL).Invoke(t.Context(), input); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	return captured
}

// TestThinkingDisplayUpdatesAutoAppendsBeta mirrors Python
// test_thinking_display_updates_enables_beta (test_chat_models.py:6349): the
// `updates` thinking display mode is beta-gated, so its flag is auto-appended
// and the request routes to the beta endpoint.
func TestThinkingDisplayUpdatesAutoAppendsBeta(t *testing.T) {
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-sonnet-4-5"),
		).WithThinking(map[string]any{"type": "adaptive", "display": "updates"})
	}, []messages.Message{messages.Human("think")})

	if captured.beta != testThinkingDisplayBeta {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, testThinkingDisplayBeta)
	}
	if captured.path != "/messages?beta=true" {
		t.Fatalf("request path: got %q want /messages?beta=true", captured.path)
	}
	thinking, _ := captured.body["thinking"].(map[string]any)
	if thinking["display"] != "updates" {
		t.Fatalf("thinking should reach the wire unchanged: %+v", captured.body["thinking"])
	}
}

// TestThinkingDisplaySummarizedNoBeta mirrors Python
// test_thinking_display_summarized_does_not_enable_beta
// (test_chat_models.py:6358): other display values are generally available.
func TestThinkingDisplaySummarizedNoBeta(t *testing.T) {
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-sonnet-4-5"),
		).WithThinking(map[string]any{"type": "adaptive", "display": "summarized"})
	}, []messages.Message{messages.Human("think")})

	if captured.beta != "" {
		t.Fatalf("anthropic-beta header should stay unset, got %q", captured.beta)
	}
	if captured.path != "/messages" {
		t.Fatalf("request path: got %q want /messages", captured.path)
	}
}

// TestUserProfileIDAutoAppendsBeta mirrors Python test_user_profile_id_init_
// param (test_chat_models.py:3772): the id reaches the wire and its beta flag
// is auto-appended, routing to the beta endpoint.
func TestUserProfileIDAutoAppendsBeta(t *testing.T) {
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-sonnet-4-5"),
		).WithUserProfileID("uprof_1")
	}, []messages.Message{messages.Human("hello")})

	if captured.beta != testUserProfilesBeta {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, testUserProfilesBeta)
	}
	if captured.path != "/messages?beta=true" {
		t.Fatalf("request path: got %q want /messages?beta=true", captured.path)
	}
	if captured.body["user_profile_id"] != "uprof_1" {
		t.Fatalf("user_profile_id: %+v", captured.body["user_profile_id"])
	}
}

// TestUserProfileIDAbsentByDefault mirrors Python
// test_user_profile_id_absent_by_default: unset means neither the id nor any
// beta flag reaches the wire.
func TestUserProfileIDAbsentByDefault(t *testing.T) {
	captured := midConvInvokeCaptured(t, "claude-sonnet-4-5", nil, []messages.Message{messages.Human("hello")})

	if captured.beta != "" {
		t.Fatalf("anthropic-beta header should stay unset, got %q", captured.beta)
	}
	if _, present := captured.body["user_profile_id"]; present {
		t.Fatalf("user_profile_id should be absent: %+v", captured.body["user_profile_id"])
	}
}

// TestUserProfileBetaNotDuplicated mirrors Python's append-if-absent rule
// (chat_models.py:2165-2167): a caller-supplied flag is not repeated.
func TestUserProfileBetaNotDuplicated(t *testing.T) {
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-sonnet-4-5"),
			WithBetaHeaders(testUserProfilesBeta),
		).WithUserProfileID("uprof_1")
	}, []messages.Message{messages.Human("hello")})

	if got := strings.Count(captured.beta, testUserProfilesBeta); got != 1 {
		t.Fatalf("anthropic-beta header should contain the beta exactly once, got %q", captured.beta)
	}
}

// TestThinkingDisplayBetaPrecedesToolChangeBeta pins the append order per
// Python's payload build: thinking-display is appended (chat_models.py:2130)
// before the tool-change betas (:2140-2160), so both flags land in that
// order.
func TestThinkingDisplayBetaPrecedesToolChangeBeta(t *testing.T) {
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-opus-5"),
		).WithThinking(map[string]any{"type": "adaptive", "display": "updates"})
	}, toolChangeConversation)

	want := testThinkingDisplayBeta + "," + testToolChangeBeta
	if captured.beta != want {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, want)
	}
}

// TestUserProfileBetaFollowsToolChangeBeta pins the other side of the order:
// user-profiles is appended last (chat_models.py:2162), after the tool-change
// betas.
func TestUserProfileBetaFollowsToolChangeBeta(t *testing.T) {
	captured := midConvInvokeChatModel(t, func(baseURL string) ChatModel {
		return NewChatModel(
			modelconfig.WithBaseURL(baseURL),
			modelconfig.WithModel("claude-opus-5"),
		).WithUserProfileID("uprof_1")
	}, toolChangeConversation)

	want := testToolChangeBeta + "," + testUserProfilesBeta
	if captured.beta != want {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, want)
	}
}

// TestMidConversationStreamToolChangeAutoAppendsBeta checks the streaming
// path applies the same auto-append.
func TestMidConversationStreamToolChangeAutoAppendsBeta(t *testing.T) {
	captured := &midConvCapture{}
	streamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.path = r.URL.RequestURI()
		captured.beta = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: message_start\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_stop\n")
		_, _ = fmt.Fprint(w, `data: {"type":"content_block_stop","index":0}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: message_delta\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`+"\n\n")
		_, _ = fmt.Fprint(w, "event: message_stop\n")
		_, _ = fmt.Fprint(w, `data: {"type":"message_stop"}`+"\n\n")
	}))
	defer streamServer.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(streamServer.URL),
		modelconfig.WithModel("claude-opus-5"),
	)
	stream, err := model.Stream(t.Context(), toolChangeConversation)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	for {
		_, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
	}

	if captured.beta != testToolChangeBeta {
		t.Fatalf("anthropic-beta header: got %q want %q", captured.beta, testToolChangeBeta)
	}
	if captured.path != "/messages?beta=true" {
		t.Fatalf("request path: got %q want /messages?beta=true", captured.path)
	}
}

// --- Unrecognized system block narrowing on the in-place path
// (chat_models.py:635-642) ---

// TestMidConversationUnrecognizedSystemBlockDroppedInPlace mirrors Python
// test__format_messages_system_unrecognized_block_dropped_with_warning
// (test_chat_models.py:2435): Anthropic accepts only text (plus tool changes
// on mid-conversation turns) in a system message, so other blocks are dropped
// with a warning rather than forwarded to rejection.
func TestMidConversationUnrecognizedSystemBlockDroppedInPlace(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	captured := midConvInvokeCaptured(t, "claude-opus-5", nil, []messages.Message{
		messages.Human("Review foo()"),
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			map[string]any{"type": "made_up_block", "foo": "bar"},
		),
		messages.AI("Looks fine."),
	})

	turn := captured.body["messages"].([]any)[1].(map[string]any)
	content, ok := turn["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("unrecognized block should be dropped: %+v", turn["content"])
	}
	if text := content[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("surviving text block: %+v", text)
	}
	if !strings.Contains(logs.String(), "made_up_block") {
		t.Fatalf("expected unrecognized-block drop warning, got: %s", logs.String())
	}
}

// TestMidConversationUnrecognizedOnlySystemTurnOmitted extends the omit rule:
// an in-place turn whose blocks all narrow away disappears entirely.
func TestMidConversationUnrecognizedOnlySystemTurnOmitted(t *testing.T) {
	captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("Review foo()"),
		systemWithBlocks(map[string]any{"type": "made_up_block", "foo": "bar"}),
		messages.AI("Looks fine."),
	})

	roles := midConvRoles(t, request)
	want := []string{"user", "assistant"}
	if len(roles) != len(want) {
		t.Fatalf("roles: got %v want %v", roles, want)
	}
}

// --- Unrecognized system block narrowing on the hoisting paths: Python's
// _format_system_content (chat_models.py:635-642) drops non-text,
// non-tool-change blocks with a warning at EVERY call site — the leading run
// (chat_models.py:742), the not-allowed hoist (:759), and the pending-flush
// hoist (:1052) — not just the in-place path. ---

// TestMidConversationLeadingSystemUnrecognizedBlockDropped covers the leading
// run: an image block (valid in human content, unrecognized in a system
// message) is dropped with a warning, and no hoist warning fires because the
// leading run is never hoisted.
func TestMidConversationLeadingSystemUnrecognizedBlockDropped(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-4-5", []messages.Message{
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			map[string]any{
				"type": "image",
				"url":  "https://example.com/diagram.png",
			},
		),
		messages.Human("hi"),
	})

	system, ok := request["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("unrecognized block should be dropped from the leading run: %+v", request["system"])
	}
	if text := system[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("surviving system block: %+v", text)
	}
	logged := logs.String()
	if !strings.Contains(logged, "image") {
		t.Fatalf("expected unrecognized-block drop warning, got: %s", logged)
	}
	if strings.Contains(logged, "hoisted") {
		t.Fatalf("leading run is not hoisted; no hoist warning expected, got: %s", logged)
	}
}

// TestMidConversationHoistedSystemUnrecognizedBlockDropped covers the
// not-allowed hoist (model profile without mid_conversation_system_messages):
// the run is hoisted AND narrowed — both the unrecognized-block and the hoist
// warnings fire.
func TestMidConversationHoistedSystemUnrecognizedBlockDropped(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-sonnet-4-5", []messages.Message{
		messages.Human("Review foo()"),
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			map[string]any{"type": "made_up_block", "foo": "bar"},
		),
		messages.AI("Looks fine."),
	})

	system, ok := request["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("unrecognized block should be dropped from the hoisted run: %+v", request["system"])
	}
	if text := system[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("surviving system block: %+v", text)
	}
	logged := logs.String()
	if !strings.Contains(logged, "made_up_block") {
		t.Fatalf("expected unrecognized-block drop warning, got: %s", logged)
	}
	if !strings.Contains(logged, "hoisted") {
		t.Fatalf("expected hoist warning, got: %s", logged)
	}
}

// TestMidConversationPendingFlushHoistUnrecognizedBlockDropped covers the
// pending-flush hoist (chat_models.py:1047-1054): a deferred system run
// flushed ahead of a non-assistant turn is hoisted and narrowed the same way.
func TestMidConversationPendingFlushHoistUnrecognizedBlockDropped(t *testing.T) {
	logs := captureAnthropicWarnings(t)
	request := midConvInvoke(t, "claude-opus-5", []messages.Message{
		messages.Human("first"),
		systemWithBlocks(
			map[string]any{"type": "text", "text": "Be concise."},
			map[string]any{"type": "made_up_block", "foo": "bar"},
		),
		messages.Human("second"),
	})

	system, ok := request["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("unrecognized block should be dropped from the flushed run: %+v", request["system"])
	}
	if text := system[0].(map[string]any); text["text"] != "Be concise." {
		t.Fatalf("surviving system block: %+v", text)
	}
	logged := logs.String()
	if !strings.Contains(logged, "made_up_block") {
		t.Fatalf("expected unrecognized-block drop warning, got: %s", logged)
	}
	if !strings.Contains(logged, "hoisted") {
		t.Fatalf("expected hoist warning, got: %s", logged)
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
