package messages

import "testing"

func TestRemoveMessageSentinel(t *testing.T) {
	rm := RemoveMessage{ID: "msg_1"}
	if !rm.IsRemoveMessage() {
		t.Fatalf("RemoveMessage.IsRemoveMessage() = false, want true")
	}
	if rm.MessageID() != "msg_1" {
		t.Fatalf("RemoveMessage.MessageID() = %q, want %q", rm.MessageID(), "msg_1")
	}

	// Both Message and RemoveMessage satisfy the MessageUpdate union.
	var _ MessageUpdate = rm
	var _ MessageUpdate = Human("hi")
}

func TestFilterExcludeToolCalls(t *testing.T) {
	ai := AI("").WithContentBlocks([]ContentBlock{
		TextBlock{Text: "intro"},
		ToolCallBlock{ID: "call_1", Name: "search", Args: map[string]any{"q": "x"}},
		TextBlock{Text: "outro"},
	})

	got := Filter([]Message{ai}, FilterOptions{ExcludeToolCalls: true})
	if len(got) != 1 {
		t.Fatalf("Filter() returned %d messages, want 1", len(got))
	}
	blocks := got[0].ContentBlocks
	if len(blocks) != 2 {
		t.Fatalf("Filter() returned %d content blocks, want 2 (tool_call pruned)", len(blocks))
	}
	if blocks[0].BlockType() != "text" || blocks[1].BlockType() != "text" {
		t.Fatalf("Filter() kept non-text blocks after tool-call pruning: %#v", blocks)
	}

	// Tool-call pruning applies only to AIMessages.
	human := Human("hi").WithContentBlocks([]ContentBlock{
		TextBlock{Text: "hi"},
		ToolCallBlock{ID: "call_2", Name: "search"},
	})
	gotHuman := Filter([]Message{human}, FilterOptions{ExcludeToolCalls: true})
	if len(gotHuman[0].ContentBlocks) != 2 {
		t.Fatalf("Filter() pruned tool-call blocks from non-AI message: %#v", gotHuman[0].ContentBlocks)
	}

	// Without the flag, tool-call blocks are preserved.
	gotAll := Filter([]Message{ai}, FilterOptions{})
	if len(gotAll[0].ContentBlocks) != 3 {
		t.Fatalf("Filter() with ExcludeToolCalls=false returned %d blocks, want 3", len(gotAll[0].ContentBlocks))
	}
}

func TestBufferStringXML(t *testing.T) {
	got := BufferStringXML([]Message{
		Human("Hi, how are you?"),
		AI("Good, how are you?"),
	})
	want := `<message type="human">Hi, how are you?</message>` +
		"\n" + `<message type="ai">Good, how are you?</message>`
	if got != want {
		t.Fatalf("BufferStringXML() = %q, want %q", got, want)
	}
}

func TestBufferStringXMLEscaping(t *testing.T) {
	got := BufferStringXML([]Message{Human(`Is 5 < 10 & 10 > 5?`)})
	want := `<message type="human">Is 5 &lt; 10 &amp; 10 &gt; 5?</message>`
	if got != want {
		t.Fatalf("BufferStringXML() = %q, want %q", got, want)
	}
}

func TestBufferStringXMLToolCalls(t *testing.T) {
	ai := AI("I'll search for that.")
	ai.ToolCalls = []ToolCall{
		{ID: "call_123", Name: "search", Args: map[string]any{"query": "weather"}},
	}
	got := BufferStringXML([]Message{ai})
	want := `<message type="ai">` + "\n" +
		`  <content>I'll search for that.</content>` + "\n" +
		`  <tool_call id="call_123" name="search">{"query":"weather"}</tool_call>` + "\n" +
		`</message>`
	if got != want {
		t.Fatalf("BufferStringXML() = %q, want %q", got, want)
	}
}

func TestBufferStringXMLContentBlocks(t *testing.T) {
	msg := AI("").WithContentBlocks([]ContentBlock{
		TextBlock{Text: "hello <world>"},
		ReasoningBlock{Reasoning: "thinking & stuff"},
	})
	got := BufferStringXML([]Message{msg})
	want := `<message type="ai">hello &lt;world&gt; <reasoning>thinking &amp; stuff</reasoning></message>`
	if got != want {
		t.Fatalf("BufferStringXML() = %q, want %q", got, want)
	}
}

func TestMessageUpdateSeal(t *testing.T) {
	// Both members of the MessageUpdate union implement the seal method.
	Message{}.isMessageUpdate()
	RemoveMessage{}.isMessageUpdate()
}
