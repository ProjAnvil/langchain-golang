package openai

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
)

// Ports partners/openai/tests/unit_tests/chat_models/test_refusal.py（#41582，
// openai 1.7.4）。roundTripFunc 复用 stream_errors_test.go 的定义。

// --- chat completions 非流式：message.refusal → AdditionalKwargs + RefusalBlock ---

func TestChatCompletionsRefusalNonStream(t *testing.T) {
	raw := `{
		"id": "chatcmpl-1", "model": "gpt-x",
		"choices": [{"message": {"role": "assistant", "content": "", "refusal": "I can't help with that."}}]
	}`
	var resp chatCompletionsResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.toResponsesPayload().toMessage()
	if msg.AdditionalKwargs["refusal"] != "I can't help with that." {
		t.Fatalf("additional_kwargs refusal missing: %v", msg.AdditionalKwargs)
	}
	var found bool
	for _, b := range msg.ContentBlocks {
		if rb, ok := b.(messages.RefusalBlock); ok {
			found = true
			if rb.Refusal != "I can't help with that." {
				t.Fatalf("refusal text = %q", rb.Refusal)
			}
		}
	}
	if !found {
		t.Fatalf("no RefusalBlock in %v", msg.ContentBlocks)
	}
}

// --- responses API 非流式：refusal content part → RefusalBlock ---
// id 挂在 message ITEM 上（上游 base.py:5269-5277 取 output.id），
// phase 同理（base.py:5250）；content part 自身不带 id。

func TestResponsesRefusalNonStream(t *testing.T) {
	raw := `{
		"id": "resp-1", "model": "gpt-x",
		"output": [{"type": "message", "role": "assistant", "id": "msg_1", "phase": "final_answer",
			"content": [{"type": "refusal", "refusal": "No."}]}]
	}`
	var resp responsePayload
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.toMessage()
	var rb *messages.RefusalBlock
	for _, b := range msg.ContentBlocks {
		if r, ok := b.(messages.RefusalBlock); ok {
			cpy := r
			rb = &cpy
		}
	}
	if rb == nil {
		t.Fatalf("no RefusalBlock in %v", msg.ContentBlocks)
	}
	if rb.Refusal != "No." {
		t.Fatalf("refusal text = %q", rb.Refusal)
	}
	if rb.ID != "msg_1" {
		t.Fatalf("refusal ID must come from the message item, got %q", rb.ID)
	}
	if rb.Extras["phase"] != "final_answer" {
		t.Fatalf("phase must land in extras, got %v", rb.Extras)
	}
}

// --- 回写方向（responses）：part 只带 type+refusal，id/phase 提升到 item 级 ---
// 上游 test_refusal.py:39-47 钉死的 wire 形状。

func TestResponsesRefusalReplayWireShape(t *testing.T) {
	ai := messages.AI("")
	ai.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{
		Refusal: "No.", ID: "msg_1", Extras: map[string]any{"phase": "final_answer"},
	}}
	m := ChatModel{}
	req, err := m.buildRequest([]messages.Message{ai})
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(req.Input)
	var items []map[string]any
	_ = json.Unmarshal(encoded, &items)
	if len(items) != 1 {
		t.Fatalf("want 1 input item, got %s", encoded)
	}
	item := items[0]
	if item["type"] != "message" || item["role"] != "assistant" {
		t.Fatalf("item shape: %v", item)
	}
	if item["id"] != "msg_1" || item["phase"] != "final_answer" {
		t.Fatalf("id/phase must be lifted to the item: %v", item)
	}
	parts, _ := item["content"].([]any)
	if len(parts) != 1 {
		t.Fatalf("want 1 refusal part, got %v", item["content"])
	}
	part, _ := parts[0].(map[string]any)
	if len(part) != 2 || part["type"] != "refusal" || part["refusal"] != "No." {
		t.Fatalf("refusal part must carry exactly type+refusal: %v", part)
	}
}

// 空文本 refusal 不发送（_compat.py:451-455）。
func TestResponsesEmptyRefusalSkipped(t *testing.T) {
	ai := messages.AI("")
	ai.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: ""}}
	m := ChatModel{}
	req, err := m.buildRequest([]messages.Message{ai})
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(req.Input)
	if strings.Contains(string(encoded), "refusal") {
		t.Fatalf("empty refusal must not be sent: %s", encoded)
	}
}

// --- CC 回写方向：refusal 块 → content parts（上游 _compat.py:179-184 形状）---

func TestChatCompletionsRefusalReplayWireShape(t *testing.T) {
	ai := messages.AI("partial")
	ai.ContentBlocks = []messages.ContentBlock{
		messages.TextBlock{Text: "partial"},
		messages.RefusalBlock{Refusal: "declined"},
	}
	m := ChatModel{}
	req, err := m.buildChatCompletionsRequest([]messages.Message{ai})
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(req.Messages[0])
	var wire map[string]any
	_ = json.Unmarshal(encoded, &wire)
	parts, ok := wire["content"].([]any)
	if !ok {
		t.Fatalf("assistant content must become parts when refusal present: %s", encoded)
	}
	var sawText, sawRefusal bool
	for _, p := range parts {
		part, _ := p.(map[string]any)
		switch part["type"] {
		case "text":
			sawText = part["text"] == "partial"
		case "refusal":
			sawRefusal = part["refusal"] == "declined"
		}
	}
	if !sawText || !sawRefusal {
		t.Fatalf("want text+refusal parts: %s", encoded)
	}
}

// --- responses 流式：response.refusal.done → RefusalBlock chunk ---

func TestResponsesStreamRefusalBlock(t *testing.T) {
	sse := "event: response.refusal.done\n" +
		`data: {"type":"response.refusal.done","refusal":"I can't help with that."}` + "\n\n" +
		"data: [DONE]\n\n"
	m := NewChatModel(
		modelconfig.WithBaseURL("http://stream.test"),
		modelconfig.WithModel("gpt-test"),
		func(c *modelconfig.Config) {
			c.HTTPClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body:       io.NopCloser(strings.NewReader(sse)),
					Request:    req,
				}, nil
			})}
		},
	)
	stream, err := m.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(t.Context())
	if err != nil || !ok {
		t.Fatalf("first Next = (%v, %v), want refusal chunk", ok, err)
	}
	var found bool
	for _, b := range chunk.ContentBlocks {
		if rb, isRefusal := b.(messages.RefusalBlock); isRefusal {
			found = true
			if rb.Refusal != "I can't help with that." {
				t.Fatalf("refusal text = %q", rb.Refusal)
			}
		}
	}
	if !found {
		t.Fatalf("want RefusalBlock chunk, got %v", chunk.ContentBlocks)
	}
}

// --- 结构化输出：refusal → ModelRefusalError（fake HTTPClient 端到端）---
// config.HTTPClient 可注入 fake RoundTripper（stream.go:54-57 先例）。

func refusalResponseBody() string {
	return `{"id":"resp-r","model":"gpt-x","output":[{"type":"message","role":"assistant",
		"content":[{"type":"refusal","refusal":"I cannot provide that."}]}],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
}

func TestInvokeStructuredRaisesModelRefusal(t *testing.T) {
	fake := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(refusalResponseBody())),
			Request:    r,
		}, nil
	})}
	m := NewChatModel(
		modelconfig.WithBaseURL("http://structured.test"),
		modelconfig.WithModel("gpt-test"),
		func(c *modelconfig.Config) { c.HTTPClient = fake },
	)
	_, err := m.InvokeStructured(t.Context(), []messages.Message{messages.Human("hi")},
		schema.Schema{"type": "object", "title": "Out"})
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}
