package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Ports anthropic #41678 的 caller 用例与 #41637 的 compaction 用例
//（tests/unit_tests/test_chat_models.py 相应段）。

// compaction 块 replay 保留 content、不带 index（#41637；Go 构造无 index 字段，
// 本用例钉死 content 不丢）。
func TestCompactionBlockKeepsContentDropsIndex(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "compaction", Value: map[string]any{
			"content": "Summary of earlier turns.",
			"index":   0,
		}},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %+v", blocks)
	}
	encoded, _ := json.Marshal(blocks[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	if wire["type"] != "compaction" {
		t.Fatalf("type lost: %v", wire)
	}
	if wire["content"] != "Summary of earlier turns." {
		t.Fatalf("compaction content dropped: %v", wire)
	}
	if _, has := wire["index"]; has {
		t.Fatalf("index must never reach the wire: %v", wire)
	}
	// 调用方 map 不被原地修改（#40213 同类保证）。
	if msg.ContentBlocks[0].(messages.NonStandardContentBlock).Value["index"] != 0 {
		t.Fatal("caller map must not be mutated")
	}
}

// 非 direct 的嵌套 caller 透传（#41678）；direct 被剥（server 族）。
func TestNestedCallerPreservedServerToolResult(t *testing.T) {
	caller := map[string]any{"type": "code_execution_20250825", "tool_id": "srvtool_1"}
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "server_tool_result", Value: map[string]any{
			"type":        "server_tool_result",
			"tool_use_id": "srvtoolu_1",
			"content":     "42",
			"caller":      caller,
		}},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(blocks[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	got, ok := wire["caller"].(map[string]any)
	if !ok || got["type"] != "code_execution_20250825" {
		t.Fatalf("nested caller must be preserved: %v", wire)
	}
}

func TestDirectCallerStrippedOnServerBlocks(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "server_tool_use", Value: map[string]any{
			"type":   "server_tool_use",
			"id":     "srvtoolu_2",
			"name":   "web_search",
			"caller": map[string]any{"type": "direct"},
		}},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(blocks[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	if _, has := wire["caller"]; has {
		t.Fatalf("direct caller must be stripped on server blocks: %v", wire)
	}
}

// client tool_use 的 caller 无条件透传（上游 chat_models.py:834-835 的非对称行为）。
func TestClientToolUseCallerPassesVerbatim(t *testing.T) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{
		messages.NonStandardContentBlock{Type: "tool_use", Value: map[string]any{
			"type":   "tool_use",
			"id":     "toolu_1",
			"name":   "search",
			"input":  map[string]any{"q": "x"},
			"caller": map[string]any{"type": "direct"},
		}},
	}
	blocks, err := formatContentBlocks(msg.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(blocks[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	if got, ok := wire["caller"].(map[string]any); !ok || got["type"] != "direct" {
		t.Fatalf("tool_use caller passes verbatim: %v", wire)
	}
}
