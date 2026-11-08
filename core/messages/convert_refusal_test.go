package messages

import (
	"testing"
)

// Ports core/tests/unit_tests/messages/test_refusal.py 的 convert_to_openai_messages
// 用例（#41582，utils.py:1955-1959）：非空 refusal 文本发出，空文本不发。

func TestConvertToOpenAIMessagesRefusal(t *testing.T) {
	msg := AI("")
	msg.ContentBlocks = []ContentBlock{
		TextBlock{Text: "partial answer"},
		RefusalBlock{Refusal: "I cannot continue."},
	}
	out, err := ConvertToOpenAIMessages([]Message{msg})
	if err != nil {
		t.Fatal(err)
	}
	content, ok := out[0]["content"].([]map[string]any)
	if !ok {
		t.Fatalf("content is %T, want []map[string]any", out[0]["content"])
	}
	var refusal map[string]any
	for _, part := range content {
		if part["type"] == "refusal" {
			refusal = part
		}
	}
	if refusal == nil || refusal["refusal"] != "I cannot continue." {
		t.Fatalf("refusal part missing or wrong: %v", content)
	}
}

func TestConvertToOpenAIMessagesEmptyRefusalSkipped(t *testing.T) {
	msg := AI("")
	msg.ContentBlocks = []ContentBlock{
		TextBlock{Text: "ok"},
		RefusalBlock{Refusal: ""}, // Anthropic 式无文本 refusal
	}
	out, err := ConvertToOpenAIMessages([]Message{msg})
	if err != nil {
		t.Fatal(err)
	}
	content := out[0]["content"].([]map[string]any)
	for _, part := range content {
		if part["type"] == "refusal" {
			t.Fatalf("empty refusal must not be sent: %v", content)
		}
	}
}

// Ports test_refusal.py:200-207：id/index/extras 不上 wire，只发 type+refusal。
func TestConvertToOpenAIMessagesRefusalDropsExtras(t *testing.T) {
	msg := AI("")
	msg.ContentBlocks = []ContentBlock{
		RefusalBlock{Refusal: "no", ID: "r1", Extras: map[string]any{"category": "safety"}},
	}
	out, err := ConvertToOpenAIMessages([]Message{msg})
	if err != nil {
		t.Fatal(err)
	}
	content := out[0]["content"].([]map[string]any)
	if len(content) != 1 {
		t.Fatalf("want exactly 1 part, got %v", content)
	}
	if len(content[0]) != 2 || content[0]["type"] != "refusal" || content[0]["refusal"] != "no" {
		t.Fatalf("refusal part must carry exactly type+refusal, got %v", content[0])
	}
}
