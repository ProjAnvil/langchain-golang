package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

func TestRequestAIMessageWithToolCalls(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	ai := messages.AI("previous answer")
	ai.ToolCalls = []messages.ToolCall{{
		ID:   "toolu_1",
		Name: "search",
		Args: map[string]any{"q": "weather"},
	}}

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	if _, err := model.Invoke(context.Background(), []messages.Message{
		ai,
		messages.Human("and tomorrow?"),
	}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	rawMessages := request["messages"].([]any)
	assistant := rawMessages[0].(map[string]any)
	if assistant["role"] != "assistant" {
		t.Fatalf("assistant role: %v", assistant)
	}
	content := assistant["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("assistant content: %v", content)
	}
	text := content[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "previous answer" {
		t.Fatalf("assistant text block: %v", text)
	}
	toolUse := content[1].(map[string]any)
	if toolUse["type"] != "tool_use" || toolUse["id"] != "toolu_1" || toolUse["name"] != "search" {
		t.Fatalf("assistant tool_use block: %v", toolUse)
	}
	input := toolUse["input"].(map[string]any)
	if input["q"] != "weather" {
		t.Fatalf("tool_use input: %v", input)
	}
}

func TestRequestAIMessageEmptyGetsBlankTextBlock(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	if _, err := model.Invoke(context.Background(), []messages.Message{
		messages.AI(""),
	}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	content := request["messages"].([]any)[0].(map[string]any)["content"].([]any)
	block := content[0].(map[string]any)
	if block["type"] != "text" {
		t.Fatalf("empty assistant content: %v", block)
	}
	if text, ok := block["text"]; ok && text != "" {
		t.Fatalf("empty assistant text should be blank: %v", block)
	}
}

func TestRequestSystemTextAndBlocksCombine(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	withBlocks := messages.System("")
	withBlocks.ContentBlocks = []messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "text", "text": "from blocks"}),
	}

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	if _, err := model.Invoke(context.Background(), []messages.Message{
		messages.System("plain system"),
		withBlocks,
		messages.Human("hi"),
	}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	system, ok := request["system"].([]any)
	if !ok || len(system) != 2 {
		t.Fatalf("system should combine text and blocks: %+v", request["system"])
	}
	first := system[0].(map[string]any)
	if first["type"] != "text" || first["text"] != "plain system" {
		t.Fatalf("system text block should be prepended: %v", first)
	}
	second := system[1].(map[string]any)
	if second["text"] != "from blocks" {
		t.Fatalf("system block: %v", second)
	}
}

func TestRequestToolMessagePlainContent(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	if _, err := model.Invoke(context.Background(), []messages.Message{
		messages.Tool("toolu_1", "sunny"),
	}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	block := firstUserContent(t, request)[0]
	if block["type"] != "tool_result" || block["tool_use_id"] != "toolu_1" {
		t.Fatalf("tool_result block: %v", block)
	}
	if block["content"] != "sunny" {
		t.Fatalf("tool_result content should stay a string: %v", block["content"])
	}
}

func TestRequestToolMessageContentPrependedToBlocks(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	toolResult := messages.Tool("toolu_1", "header")
	toolResult.ContentBlocks = []messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "text", "text": "details"}),
	}

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	if _, err := model.Invoke(context.Background(), []messages.Message{toolResult}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	block := firstUserContent(t, request)[0]
	content := block["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("tool_result content: %v", content)
	}
	first := content[0].(map[string]any)
	if first["type"] != "text" || first["text"] != "header" {
		t.Fatalf("tool_result content should be prepended: %v", first)
	}
}

func TestRequestToolMessageInvalidBlockFails(t *testing.T) {
	model := NewChatModel(modelconfig.WithBaseURL("http://127.0.0.1:1"), modelconfig.WithModel("m"))
	toolResult := messages.Tool("toolu_1", "")
	toolResult.ContentBlocks = []messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "image"}),
	}
	if _, err := model.Invoke(context.Background(), []messages.Message{toolResult}); err == nil {
		t.Fatal("invoke with invalid tool_result block should fail")
	}
}

func TestRequestSystemInvalidBlockFails(t *testing.T) {
	model := NewChatModel(modelconfig.WithBaseURL("http://127.0.0.1:1"), modelconfig.WithModel("m"))
	sys := messages.System("")
	sys.ContentBlocks = []messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "image"}),
	}
	if _, err := model.Invoke(context.Background(), []messages.Message{sys}); err == nil {
		t.Fatal("invoke with invalid system block should fail")
	}
}

func TestInvokeUnknownResponseBlockPassesThrough(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","content":[{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"q":"news"}},{"type":"text","text":"done"}],"usage":{}}`)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("q")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	found := false
	for _, block := range resp.ContentBlocks {
		bm := messages.BlockToMap(block)
		if bm["type"] == "server_tool_use" && bm["name"] == "web_search" {
			found = true
		}
	}
	if !found {
		t.Fatalf("unknown block should pass through: %+v", resp.ContentBlocks)
	}
	if resp.Content != "done" {
		t.Fatalf("content: %q", resp.Content)
	}
}

func TestInvokeRedactedThinkingWithID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","content":[{"type":"redacted_thinking","data":"ZW5j","id":"rt_1"}],"usage":{}}`)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("q")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if len(resp.ContentBlocks) != 1 {
		t.Fatalf("content blocks: %+v", resp.ContentBlocks)
	}
	bm := messages.BlockToMap(resp.ContentBlocks[0])
	if bm["type"] != "reasoning" || bm["data"] != "ZW5j" || bm["id"] != "rt_1" {
		t.Fatalf("redacted thinking block: %+v", bm)
	}
}

func TestStructuredToolIsNotInvocable(t *testing.T) {
	tool := structuredTool{name: "response_format", desc: "desc"}
	if tool.Name() != "response_format" || tool.Description() != "desc" {
		t.Fatalf("structuredTool accessors: %+v", tool)
	}
	if _, err := tool.Invoke(context.Background(), map[string]any{}); err == nil ||
		!strings.Contains(err.Error(), "not invocable") {
		t.Fatalf("structuredTool.Invoke should fail: %v", err)
	}
}

func TestInvokeStructuredNoToolCallFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","content":[{"type":"text","text":"no tools here"}],"usage":{}}`)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	_, err := model.InvokeStructured(context.Background(), []messages.Message{
		messages.Human("q"),
	}, map[string]any{"type": "object"})
	if err == nil || !strings.Contains(err.Error(), "no tool_call") {
		t.Fatalf("InvokeStructured should fail without tool_call: %v", err)
	}
}

func TestRequestAIMessageWithContentBlocks(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	ai := messages.AI("prev")
	ai.ContentBlocks = []messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "text", "text": "extra detail"}),
	}

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	if _, err := model.Invoke(context.Background(), []messages.Message{ai}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	content := request["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("assistant content: %v", content)
	}
	second := content[1].(map[string]any)
	if second["type"] != "text" || second["text"] != "extra detail" {
		t.Fatalf("assistant content block: %v", second)
	}
}

func TestRequestAIMessageInvalidBlockFails(t *testing.T) {
	ai := messages.AI("")
	ai.ContentBlocks = []messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "image"}),
	}
	model := NewChatModel(modelconfig.WithBaseURL("http://127.0.0.1:1"), modelconfig.WithModel("m"))
	if _, err := model.Invoke(context.Background(), []messages.Message{ai}); err == nil {
		t.Fatal("invoke with invalid assistant block should fail")
	}
}

func TestCloneAnyMap(t *testing.T) {
	if cloneAnyMap(nil) != nil {
		t.Fatal("cloneAnyMap(nil) should be nil")
	}
	original := map[string]any{"k": "v"}
	clone := cloneAnyMap(original)
	if clone["k"] != "v" {
		t.Fatalf("clone: %v", clone)
	}
	clone["k"] = "changed"
	if original["k"] != "v" {
		t.Fatal("cloneAnyMap should be a defensive copy")
	}
}

func TestCloneMetadata(t *testing.T) {
	if cloneMetadata(nil) != nil {
		t.Fatal("cloneMetadata(nil) should be nil")
	}
	original := map[string]any{"k": "v"}
	clone := cloneMetadata(original)
	if clone["k"] != "v" {
		t.Fatalf("clone: %v", clone)
	}
	clone["k"] = "changed"
	if original["k"] != "v" {
		t.Fatal("cloneMetadata should be a defensive copy")
	}
}

func TestInvokeStructuredInvokeErrorPropagates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
		modelconfig.WithMaxRetries(0),
	)
	_, err := model.InvokeStructured(context.Background(), []messages.Message{
		messages.Human("q"),
	}, map[string]any{"type": "object", "title": "schema"})
	if err == nil || !strings.Contains(err.Error(), "structured output") {
		t.Fatalf("InvokeStructured should wrap the invoke error: %v", err)
	}
}
