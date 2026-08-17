package messages

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMessageJSONRoundTrip(t *testing.T) {
	original := AI("use the search tool")
	original.ID = "msg_123"
	original.ToolCalls = []ToolCall{
		{
			ID:   "call_123",
			Name: "search",
			Args: map[string]any{"query": "langchain go"},
		},
	}
	original.UsageMetadata = UsageMetadata{
		InputTokens:  10,
		OutputTokens: 5,
		TotalTokens:  15,
	}
	original.ResponseMetadata = map[string]any{"model": "fake-chat"}

	data, err := MarshalJSONStable(original)
	if err != nil {
		t.Fatalf("marshal message: %v", err)
	}

	decoded, err := UnmarshalJSONStable(data)
	if err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}

	if decoded.Role != RoleAI {
		t.Fatalf("role mismatch: got %q", decoded.Role)
	}
	if decoded.ToolCalls[0].Name != "search" {
		t.Fatalf("tool call name mismatch: got %q", decoded.ToolCalls[0].Name)
	}
	if decoded.UsageMetadata.TotalTokens != 15 {
		t.Fatalf("usage mismatch: got %d", decoded.UsageMetadata.TotalTokens)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw json: %v", err)
	}
	if raw["role"] != string(RoleAI) {
		t.Fatalf("serialized role mismatch: got %v", raw["role"])
	}
}

func TestMessageUtilities(t *testing.T) {
	msgs := []Message{
		System("rules"),
		Human("hello"),
		Human("again"),
		AI("").WithContentBlocks([]ContentBlock{TextBlock{Text: "answer"}}),
		Tool("call-1", "result"),
	}

	if got := Text(msgs[3]); got != "answer" {
		t.Fatalf("Text = %q, want answer", got)
	}
	if got := BufferString(msgs[:2]); got != "System: rules\nHuman: hello" {
		t.Fatalf("BufferString = %q", got)
	}
	filtered := Filter(msgs, FilterOptions{IncludeRoles: []Role{RoleHuman}})
	if len(filtered) != 2 {
		t.Fatalf("filtered len = %d, want 2", len(filtered))
	}
	merged := MergeRuns(msgs)
	if len(merged) != 4 || merged[1].Content != "hello\nagain" {
		t.Fatalf("unexpected merged messages: %#v", merged)
	}
	trimmed := Trim(msgs, len("result"), true)
	if len(trimmed) != 1 || trimmed[0].Role != RoleTool {
		t.Fatalf("unexpected trimmed messages: %#v", trimmed)
	}
}

func TestUsageMetadataTokenDetailsJSON(t *testing.T) {
	usage := UsageMetadata{
		InputTokens:        350,
		OutputTokens:       240,
		TotalTokens:        590,
		InputTokenDetails:  &InputTokenDetails{CacheReadInputTokens: 100, CacheCreationInputTokens: 200},
		OutputTokenDetails: &OutputTokenDetails{ReasoningOutputTokens: 30},
	}

	data, err := json.Marshal(usage)
	if err != nil {
		t.Fatalf("marshal usage metadata: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw json: %v", err)
	}

	inputDetails, ok := raw["input_token_details"].(map[string]any)
	if !ok {
		t.Fatalf("missing input_token_details: %#v", raw)
	}
	if inputDetails["cache_read_input_tokens"] != float64(100) {
		t.Fatalf("cache_read_input_tokens mismatch: %#v", inputDetails)
	}
	if inputDetails["cache_creation_input_tokens"] != float64(200) {
		t.Fatalf("cache_creation_input_tokens mismatch: %#v", inputDetails)
	}

	outputDetails, ok := raw["output_token_details"].(map[string]any)
	if !ok {
		t.Fatalf("missing output_token_details: %#v", raw)
	}
	if outputDetails["reasoning_output_tokens"] != float64(30) {
		t.Fatalf("reasoning_output_tokens mismatch: %#v", outputDetails)
	}
}

func TestMessagesDictRoundTripAndClone(t *testing.T) {
	original := []Message{{Role: RoleAI, ContentBlocks: []ContentBlock{TextBlock{Text: "x"}}}}
	dicts, err := MessagesToDict(original)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := MessagesFromDict(dicts)
	if err != nil {
		t.Fatal(err)
	}
	if Text(decoded[0]) != "x" {
		t.Fatalf("decoded text = %q", Text(decoded[0]))
	}
	clone := Clone(original[0])
	clone.ContentBlocks[0] = TextBlock{Text: "changed"}
	if Text(original[0]) != "x" {
		t.Fatal("clone mutation changed original")
	}
}

func TestTextContentBlockVariants(t *testing.T) {
	// String content always wins over blocks.
	withBoth := Message{Content: "plain", ContentBlocks: []ContentBlock{TextBlock{Text: "block"}}}
	if got := Text(withBoth); got != "plain" {
		t.Fatalf("Text() = %q, want plain", got)
	}

	msg := Message{Role: RoleAI, ContentBlocks: []ContentBlock{
		TextBlock{Text: "a"},
		NonStandardContentBlock{Type: "", Value: map[string]any{"text": "legacy"}},
		NonStandardContentBlock{Type: "text", Value: map[string]any{"text": "b"}},
		NonStandardContentBlock{Type: "tool_use", Value: map[string]any{"text": "ignored"}},
		NonStandardContentBlock{Type: "", Value: map[string]any{}},
		NonStandardContentBlock{Type: "", Value: map[string]any{"text": 42}},
	}}
	if got := Text(msg); got != "alegacyb" {
		t.Fatalf("Text() = %q, want %q", got, "alegacyb")
	}
}

func TestBufferStringUnknownRole(t *testing.T) {
	got := BufferString([]Message{{Role: "custom", Content: "hi"}})
	if got != "custom: hi" {
		t.Fatalf("BufferString() = %q, want %q", got, "custom: hi")
	}

	got = BufferString([]Message{System("s"), Human("h"), AI("a"), Tool("c1", "t")})
	if got != "System: s\nHuman: h\nAI: a\nTool: t" {
		t.Fatalf("BufferString() = %q", got)
	}
}

func TestBufferStringXMLAIWithToolCalls(t *testing.T) {
	msg := AI("thinking")
	msg.ToolCalls = []ToolCall{
		{ID: "c1", Name: "search", Args: map[string]any{"q": "x"}},
		{ID: "c2", Name: "noop"},
	}
	want := "<message type=\"ai\">\n" +
		"  <content>thinking</content>\n" +
		"  <tool_call id=\"c1\" name=\"search\">{\"q\":\"x\"}</tool_call>\n" +
		"  <tool_call id=\"c2\" name=\"noop\">{}</tool_call>\n" +
		"</message>"
	if got := BufferStringXML([]Message{msg}); got != want {
		t.Fatalf("BufferStringXML() =\n%s\nwant:\n%s", got, want)
	}

	// Tool calls without any content: no <content> element.
	noContent := AI("")
	noContent.ToolCalls = []ToolCall{{ID: "c1", Name: "t"}}
	want = "<message type=\"ai\">\n" +
		"  <tool_call id=\"c1\" name=\"t\">{}</tool_call>\n" +
		"</message>"
	if got := BufferStringXML([]Message{noContent}); got != want {
		t.Fatalf("BufferStringXML() =\n%s\nwant:\n%s", got, want)
	}
}

func TestBufferStringXMLAllBlockKinds(t *testing.T) {
	msg := Human("").WithContentBlocks([]ContentBlock{
		TextBlock{Text: "a<b>&c"},
		TextBlock{},
		ReasoningBlock{Reasoning: "why"},
		ReasoningBlock{},
		ImageBlock{URL: "http://x/img.png"},
		ImageBlock{FileID: "f1"},
		ImageBlock{Base64: "zzz", MimeType: "image/png"},
		ImageBlock{URL: "data:image/png;base64,AA"},
		ImageBlock{URL: "http://x/\"quoted\".png"},
		AudioBlock{URL: "http://x/a.mp3"},
		AudioBlock{FileID: "af1"},
		AudioBlock{Base64: "q"},
		VideoBlock{URL: "http://x/v.mp4"},
		VideoBlock{FileID: "vf1"},
		VideoBlock{Base64: "q"},
		PlainTextBlock{Text: "plain"},
		PlainTextBlock{},
		ServerToolCall{ID: "s1", Name: "code_exec", Args: map[string]any{"code": "1+1"}},
		ServerToolCall{ID: "s3", Name: "bad", Args: map[string]any{"f": func() {}}},
		ServerToolResult{ToolCallID: "s1", Status: "success", Output: map[string]any{"n": 2}},
		ServerToolResult{ToolCallID: "s2", Status: "error"},
		NonStandardContentBlock{Type: "tool_use", Value: map[string]any{"text": "ignored"}},
	})

	parts := []string{
		"a&lt;b&gt;&amp;c",
		"<reasoning>why</reasoning>",
		`<image url="http://x/img.png" />`,
		`<image file_id="f1" />`,
		`<image url="http://x/&quot;quoted&quot;.png" />`,
		`<audio url="http://x/a.mp3" />`,
		`<audio file_id="af1" />`,
		`<video url="http://x/v.mp4" />`,
		`<video file_id="vf1" />`,
		"plain",
		`<server_tool_call id="s1" name="code_exec">{"code":"1+1"}</server_tool_call>`,
		`<server_tool_call id="s3" name="bad">{}</server_tool_call>`,
		`<server_tool_result tool_call_id="s1" status="success">{"n":2}</server_tool_result>`,
		`<server_tool_result tool_call_id="s2" status="error"></server_tool_result>`,
	}
	want := `<message type="human">` + strings.Join(parts, " ") + `</message>`
	if got := BufferStringXML([]Message{msg}); got != want {
		t.Fatalf("BufferStringXML() =\n%s\nwant:\n%s", got, want)
	}
}

func TestBufferStringXMLStringContentEscaped(t *testing.T) {
	got := BufferStringXML([]Message{Human("a & <b>")})
	want := `<message type="human">a &amp; &lt;b&gt;</message>`
	if got != want {
		t.Fatalf("BufferStringXML() = %q, want %q", got, want)
	}
}

func TestBufferStringXMLTruncatesLongPlainText(t *testing.T) {
	long := strings.Repeat("x", 501)
	msg := Human("").WithContentBlocks([]ContentBlock{PlainTextBlock{Text: long}})
	got := BufferStringXML([]Message{msg})
	want := `<message type="human">` + strings.Repeat("x", 500) + `...</message>`
	if got != want {
		t.Fatalf("BufferStringXML() length = %d, want truncated output", len(got))
	}
}

func TestTrimVariants(t *testing.T) {
	msgs := []Message{Human("aaaa"), Human("bbbb"), Human("cccc")}

	if got := Trim(msgs, 0, true); got != nil {
		t.Fatalf("Trim(maxChars=0) = %#v, want nil", got)
	}
	if got := Trim(msgs, -1, false); got != nil {
		t.Fatalf("Trim(maxChars=-1) = %#v, want nil", got)
	}

	// fromEnd=false keeps the leading messages within budget.
	got := Trim(msgs, 8, false)
	if len(got) != 2 || got[0].Content != "aaaa" || got[1].Content != "bbbb" {
		t.Fatalf("Trim(fromEnd=false) = %#v, want first two messages", got)
	}
}

func TestFilterNamesAndIDs(t *testing.T) {
	alice := Human("from alice")
	alice.Name = "alice"
	alice.ID = "m1"
	bob := Human("from bob")
	bob.Name = "bob"
	bob.ID = "m2"
	anon := AI("answer")
	anon.ID = "m3"
	msgs := []Message{alice, bob, anon}

	if got := Filter(msgs, FilterOptions{IncludeNames: []string{"alice"}}); len(got) != 1 || got[0].Name != "alice" {
		t.Fatalf("IncludeNames filter = %#v", got)
	}
	if got := Filter(msgs, FilterOptions{ExcludeNames: []string{"bob"}}); len(got) != 2 {
		t.Fatalf("ExcludeNames filter = %#v", got)
	}
	if got := Filter(msgs, FilterOptions{IncludeIDs: []string{"m2", "m3"}}); len(got) != 2 || got[0].ID != "m2" {
		t.Fatalf("IncludeIDs filter = %#v", got)
	}
	if got := Filter(msgs, FilterOptions{ExcludeIDs: []string{"m1", "m2"}}); len(got) != 1 || got[0].ID != "m3" {
		t.Fatalf("ExcludeIDs filter = %#v", got)
	}
	if got := Filter(msgs, FilterOptions{ExcludeRoles: []Role{RoleHuman}}); len(got) != 1 || got[0].Role != RoleAI {
		t.Fatalf("ExcludeRoles filter = %#v", got)
	}
}

func TestCloneDeepCopiesMaps(t *testing.T) {
	msg := AI("x")
	msg.ToolCalls = []ToolCall{{ID: "c1", Name: "t", Args: map[string]any{"a": 1}}}
	msg.InvalidToolCalls = []ToolCall{{ID: "c2", Name: "bad", Args: map[string]any{"b": 2}}}
	msg.ResponseMetadata = map[string]any{"k": "v"}
	msg.AdditionalKwargs = map[string]any{"k2": "v2"}
	msg.ProviderNativeEvent = map[string]any{"k3": "v3"}

	clone := Clone(msg)
	clone.ToolCalls[0].Args["a"] = 99
	clone.InvalidToolCalls[0].Args["b"] = 99
	clone.ResponseMetadata["k"] = "changed"
	clone.AdditionalKwargs["k2"] = "changed"
	clone.ProviderNativeEvent["k3"] = "changed"

	if msg.ToolCalls[0].Args["a"] != 1 {
		t.Fatal("clone ToolCalls mutation changed original")
	}
	if msg.InvalidToolCalls[0].Args["b"] != 2 {
		t.Fatal("clone InvalidToolCalls mutation changed original")
	}
	if msg.ResponseMetadata["k"] != "v" {
		t.Fatal("clone ResponseMetadata mutation changed original")
	}
	if msg.AdditionalKwargs["k2"] != "v2" {
		t.Fatal("clone AdditionalKwargs mutation changed original")
	}
	if msg.ProviderNativeEvent["k3"] != "v3" {
		t.Fatal("clone ProviderNativeEvent mutation changed original")
	}
}

func TestMergeRunsBoundaries(t *testing.T) {
	// Tool messages are never merged, even with the same role.
	tools := []Message{Tool("c1", "r1"), Tool("c2", "r2")}
	if got := MergeRuns(tools); len(got) != 2 {
		t.Fatalf("MergeRuns(tool messages) = %#v, want 2 messages", got)
	}

	// Empty content merges without introducing blank lines.
	empties := []Message{Human(""), Human("x"), Human("")}
	got := MergeRuns(empties)
	if len(got) != 1 || got[0].Content != "x" {
		t.Fatalf("MergeRuns(empty contents) = %#v, want single message with content x", got)
	}

	// Same role but different names are not merged.
	named := []Message{{Role: RoleHuman, Content: "a", Name: "alice"}, {Role: RoleHuman, Content: "b", Name: "bob"}}
	if got := MergeRuns(named); len(got) != 2 {
		t.Fatalf("MergeRuns(different names) = %#v, want 2 messages", got)
	}
}

func TestMessagesDictErrorPaths(t *testing.T) {
	// A dict value that json.Marshal cannot encode fails MessagesFromDict.
	if _, err := MessagesFromDict([]map[string]any{{"role": func() {}}}); err == nil {
		t.Fatal("expected error for unmarshalable dict value")
	}
	// A dict that marshals but is not a valid message fails too.
	if _, err := MessagesFromDict([]map[string]any{{"role": 123}}); err == nil {
		t.Fatal("expected error for non-string role")
	}
	// A message whose fields json.Marshal cannot encode fails MessagesToDict.
	bad := AI("x")
	bad.AdditionalKwargs = map[string]any{"f": func() {}}
	if _, err := MessagesToDict([]Message{bad}); err == nil {
		t.Fatal("expected error for unmarshalable message field")
	}
	// Invalid JSON fails UnmarshalJSONStable.
	if _, err := UnmarshalJSONStable([]byte("{invalid")); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}
