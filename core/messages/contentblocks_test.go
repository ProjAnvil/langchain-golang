package messages

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestParseContentBlockServerToolCall(t *testing.T) {
	input := map[string]any{
		"type":  "server_tool_call",
		"id":    "stc_1",
		"name":  "web_search",
		"args":  map[string]any{"query": "langchain go"},
		"index": 2,
	}
	block := ParseContentBlock(input)
	stc, ok := block.(ServerToolCall)
	if !ok {
		t.Fatalf("expected ServerToolCall, got %T", block)
	}
	if stc.ID != "stc_1" {
		t.Fatalf("id mismatch: %#v", stc)
	}
	if stc.Name != "web_search" {
		t.Fatalf("name mismatch: %#v", stc)
	}
	if got := stc.Args["query"]; got != "langchain go" {
		t.Fatalf("args mismatch: %#v", stc.Args)
	}
	if stc.Index != 2 {
		t.Fatalf("index mismatch: %#v", stc)
	}

	out := BlockToMap(stc)
	if out["type"] != "server_tool_call" {
		t.Fatalf("type not preserved: %#v", out)
	}
	if out["id"] != "stc_1" || out["name"] != "web_search" {
		t.Fatalf("round-trip mismatch: %#v", out)
	}
}

func TestParseContentBlockServerToolCallChunk(t *testing.T) {
	input := map[string]any{
		"type":  "server_tool_call_chunk",
		"id":    "stc_2",
		"name":  "code_exec",
		"args":  "{\"code\": \"print(",
		"index": 3,
	}
	block := ParseContentBlock(input)
	chunk, ok := block.(ServerToolCallChunk)
	if !ok {
		t.Fatalf("expected ServerToolCallChunk, got %T", block)
	}
	if chunk.ID != "stc_2" || chunk.Name != "code_exec" {
		t.Fatalf("field mismatch: %#v", chunk)
	}
	if chunk.Args != "{\"code\": \"print(" {
		t.Fatalf("args mismatch: %#v", chunk)
	}
	if chunk.Index != 3 {
		t.Fatalf("index mismatch: %#v", chunk)
	}

	out := BlockToMap(chunk)
	if out["type"] != "server_tool_call_chunk" {
		t.Fatalf("type not preserved: %#v", out)
	}
	if out["id"] != "stc_2" || out["name"] != "code_exec" {
		t.Fatalf("round-trip mismatch: %#v", out)
	}
}

func TestParseContentBlockServerToolResult(t *testing.T) {
	input := map[string]any{
		"type":         "server_tool_result",
		"id":           "str_1",
		"tool_call_id": "stc_1",
		"status":       "success",
		"output":       "42",
		"index":        4,
	}
	block := ParseContentBlock(input)
	result, ok := block.(ServerToolResult)
	if !ok {
		t.Fatalf("expected ServerToolResult, got %T", block)
	}
	if result.ID != "str_1" {
		t.Fatalf("id mismatch: %#v", result)
	}
	if result.ToolCallID != "stc_1" {
		t.Fatalf("tool_call_id mismatch: %#v", result)
	}
	if result.Status != "success" {
		t.Fatalf("status mismatch: %#v", result)
	}
	if result.Output != "42" {
		t.Fatalf("output mismatch: %#v", result)
	}
	if result.Index != 4 {
		t.Fatalf("index mismatch: %#v", result)
	}

	out := BlockToMap(result)
	if out["type"] != "server_tool_result" {
		t.Fatalf("type not preserved: %#v", out)
	}
	if out["tool_call_id"] != "stc_1" || out["status"] != "success" {
		t.Fatalf("round-trip mismatch: %#v", out)
	}
}

func TestCloneBlockServerToolTypes(t *testing.T) {
	call := ServerToolCall{
		ID:     "stc_1",
		Name:   "web_search",
		Args:   map[string]any{"query": "go"},
		Index:  1,
		Extras: map[string]any{"trace": "t1"},
	}
	callClone := CloneBlock(call).(ServerToolCall)
	callClone.Args["query"] = "changed"
	callClone.Extras["trace"] = "changed"
	if call.Args["query"] != "go" || call.Extras["trace"] != "t1" {
		t.Fatal("ServerToolCall clone mutation changed original")
	}

	result := ServerToolResult{
		ID:         "str_1",
		ToolCallID: "stc_1",
		Status:     "success",
		Output:     map[string]any{"n": 1},
		Extras:     map[string]any{"trace": "t2"},
	}
	resultClone := CloneBlock(result).(ServerToolResult)
	resultClone.Output.(map[string]any)["n"] = 2
	resultClone.Extras["trace"] = "changed"
	if result.Output.(map[string]any)["n"] != 1 || result.Extras["trace"] != "t2" {
		t.Fatal("ServerToolResult clone mutation changed original")
	}
}

func TestMessageJSONServerToolCallRoundTrip(t *testing.T) {
	msg := Message{
		Role: RoleAI,
		ContentBlocks: []ContentBlock{
			ServerToolCall{ID: "stc_1", Name: "web_search", Args: map[string]any{"query": "go"}},
		},
	}

	data, err := MarshalJSONStable(msg)
	if err != nil {
		t.Fatalf("marshal message: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw json: %v", err)
	}
	blocks, ok := raw["content_blocks"].([]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("unexpected content_blocks: %#v", raw["content_blocks"])
	}
	first, ok := blocks[0].(map[string]any)
	if !ok {
		t.Fatalf("unexpected block shape: %#v", blocks[0])
	}
	if first["type"] != "server_tool_call" {
		t.Fatalf("type not preserved: %#v", first)
	}

	decoded, err := UnmarshalJSONStable(data)
	if err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	stc, ok := decoded.ContentBlocks[0].(ServerToolCall)
	if !ok {
		t.Fatalf("expected ServerToolCall, got %T", decoded.ContentBlocks[0])
	}
	if stc.ID != "stc_1" || stc.Name != "web_search" {
		t.Fatalf("round-trip mismatch: %#v", stc)
	}
	if got := stc.Args["query"]; got != "go" {
		t.Fatalf("args mismatch: %#v", stc.Args)
	}
}

func TestBlockTypeDiscriminators(t *testing.T) {
	// Seal methods carry no behavior, but they are part of each type's method
	// set; invoke them alongside BlockType so the interface contract is
	// exercised for every concrete block type.
	cases := []struct {
		name  string
		block ContentBlock
		seal  func()
		want  string
	}{
		{"text", TextBlock{}, func() { TextBlock{}.isContentBlock() }, "text"},
		{"image", ImageBlock{}, func() { ImageBlock{}.isContentBlock() }, "image"},
		{"video", VideoBlock{}, func() { VideoBlock{}.isContentBlock() }, "video"},
		{"audio", AudioBlock{}, func() { AudioBlock{}.isContentBlock() }, "audio"},
		{"file", FileBlock{}, func() { FileBlock{}.isContentBlock() }, "file"},
		{"text-plain", PlainTextBlock{}, func() { PlainTextBlock{}.isContentBlock() }, "text-plain"},
		{"reasoning", ReasoningBlock{}, func() { ReasoningBlock{}.isContentBlock() }, "reasoning"},
		{"tool_call", ToolCallBlock{}, func() { ToolCallBlock{}.isContentBlock() }, "tool_call"},
		{"tool_call_chunk", ToolCallChunkBlock{}, func() { ToolCallChunkBlock{}.isContentBlock() }, "tool_call_chunk"},
		{"invalid_tool_call", InvalidToolCallBlock{}, func() { InvalidToolCallBlock{}.isContentBlock() }, "invalid_tool_call"},
		{"server_tool_call", ServerToolCall{}, func() { ServerToolCall{}.isContentBlock() }, "server_tool_call"},
		{"server_tool_call_chunk", ServerToolCallChunk{}, func() { ServerToolCallChunk{}.isContentBlock() }, "server_tool_call_chunk"},
		{"server_tool_result", ServerToolResult{}, func() { ServerToolResult{}.isContentBlock() }, "server_tool_result"},
		{"nonstandard", NonStandardContentBlock{Type: "tool_use"}, func() { NonStandardContentBlock{}.isContentBlock() }, "tool_use"},
	}
	for _, tc := range cases {
		if got := tc.block.BlockType(); got != tc.want {
			t.Errorf("%s: BlockType() = %q, want %q", tc.name, got, tc.want)
		}
		tc.seal()
	}

	citation := Citation{}
	citation.isAnnotation()
	if got := citation.AnnotationType(); got != "citation" {
		t.Errorf("Citation.AnnotationType() = %q, want citation", got)
	}
	nonStandard := NonStandardAnnotation{}
	nonStandard.isAnnotation()
	if got := nonStandard.AnnotationType(); got != "non_standard_annotation" {
		t.Errorf("NonStandardAnnotation.AnnotationType() = %q, want non_standard_annotation", got)
	}
}

func TestParseContentBlockNil(t *testing.T) {
	if block := ParseContentBlock(nil); block != nil {
		t.Fatalf("ParseContentBlock(nil) = %#v, want nil", block)
	}
}

func TestParseContentBlockStandardTypes(t *testing.T) {
	cases := []struct {
		name  string
		input map[string]any
		check func(t *testing.T, block ContentBlock)
	}{
		{
			name: "text with annotations and extras",
			input: map[string]any{
				"type":        "text",
				"text":        "hello",
				"id":          "t1",
				"index":       1,
				"annotations": []Annotation{Citation{ID: "cit1", URL: "http://x/doc"}},
				"custom":      "extra-value",
			},
			check: func(t *testing.T, block ContentBlock) {
				tb, ok := block.(TextBlock)
				if !ok {
					t.Fatalf("expected TextBlock, got %T", block)
				}
				if tb.Text != "hello" || tb.ID != "t1" || tb.Index != 1 {
					t.Fatalf("field mismatch: %#v", tb)
				}
				if len(tb.Annotations) != 1 {
					t.Fatalf("annotations = %#v", tb.Annotations)
				}
				if cit, ok := tb.Annotations[0].(Citation); !ok || cit.ID != "cit1" {
					t.Fatalf("annotation mismatch: %#v", tb.Annotations[0])
				}
				if tb.Extras["custom"] != "extra-value" {
					t.Fatalf("extras mismatch: %#v", tb.Extras)
				}
			},
		},
		{
			name:  "video",
			input: map[string]any{"type": "video", "id": "v1", "url": "http://x/v.mp4", "mime_type": "video/mp4", "custom": 1},
			check: func(t *testing.T, block ContentBlock) {
				vb, ok := block.(VideoBlock)
				if !ok {
					t.Fatalf("expected VideoBlock, got %T", block)
				}
				if vb.ID != "v1" || vb.URL != "http://x/v.mp4" || vb.MimeType != "video/mp4" {
					t.Fatalf("field mismatch: %#v", vb)
				}
				if vb.Extras["custom"] != 1 {
					t.Fatalf("extras mismatch: %#v", vb.Extras)
				}
			},
		},
		{
			name:  "audio",
			input: map[string]any{"type": "audio", "file_id": "af1", "base64": "QUJD"},
			check: func(t *testing.T, block ContentBlock) {
				ab, ok := block.(AudioBlock)
				if !ok {
					t.Fatalf("expected AudioBlock, got %T", block)
				}
				if ab.FileID != "af1" || ab.Base64 != "QUJD" {
					t.Fatalf("field mismatch: %#v", ab)
				}
			},
		},
		{
			name:  "file with legacy data fallback",
			input: map[string]any{"type": "file", "data": "AAAA", "mime_type": "application/pdf"},
			check: func(t *testing.T, block ContentBlock) {
				fb, ok := block.(FileBlock)
				if !ok {
					t.Fatalf("expected FileBlock, got %T", block)
				}
				if fb.Base64 != "AAAA" {
					t.Fatalf("legacy data not mapped to base64: %#v", fb)
				}
			},
		},
		{
			name:  "file base64 wins over legacy data",
			input: map[string]any{"type": "file", "base64": "BB", "data": "AAAA"},
			check: func(t *testing.T, block ContentBlock) {
				fb, ok := block.(FileBlock)
				if !ok {
					t.Fatalf("expected FileBlock, got %T", block)
				}
				if fb.Base64 != "BB" {
					t.Fatalf("base64 should win over data: %#v", fb)
				}
			},
		},
		{
			name:  "text-plain with default mime type",
			input: map[string]any{"type": "text-plain", "text": "body", "title": "doc", "context": "ctx"},
			check: func(t *testing.T, block ContentBlock) {
				pb, ok := block.(PlainTextBlock)
				if !ok {
					t.Fatalf("expected PlainTextBlock, got %T", block)
				}
				if pb.MimeType != "text/plain" {
					t.Fatalf("default mime_type = %q, want text/plain", pb.MimeType)
				}
				if pb.Text != "body" || pb.Title != "doc" || pb.Context != "ctx" {
					t.Fatalf("field mismatch: %#v", pb)
				}
			},
		},
		{
			name:  "reasoning",
			input: map[string]any{"type": "reasoning", "id": "r1", "reasoning": "because", "index": 0},
			check: func(t *testing.T, block ContentBlock) {
				rb, ok := block.(ReasoningBlock)
				if !ok {
					t.Fatalf("expected ReasoningBlock, got %T", block)
				}
				if rb.ID != "r1" || rb.Reasoning != "because" || rb.Index != 0 {
					t.Fatalf("field mismatch: %#v", rb)
				}
			},
		},
		{
			name:  "tool_call",
			input: map[string]any{"type": "tool_call", "id": "tc1", "name": "search", "args": map[string]any{"q": "x"}},
			check: func(t *testing.T, block ContentBlock) {
				tc, ok := block.(ToolCallBlock)
				if !ok {
					t.Fatalf("expected ToolCallBlock, got %T", block)
				}
				if tc.ID != "tc1" || tc.Name != "search" || tc.Args["q"] != "x" {
					t.Fatalf("field mismatch: %#v", tc)
				}
			},
		},
		{
			name:  "tool_call_chunk",
			input: map[string]any{"type": "tool_call_chunk", "id": "tcc1", "name": "search", "args": "{\"q\":"},
			check: func(t *testing.T, block ContentBlock) {
				tc, ok := block.(ToolCallChunkBlock)
				if !ok {
					t.Fatalf("expected ToolCallChunkBlock, got %T", block)
				}
				if tc.ID != "tcc1" || tc.Name != "search" || tc.Args != "{\"q\":" {
					t.Fatalf("field mismatch: %#v", tc)
				}
			},
		},
		{
			name:  "invalid_tool_call",
			input: map[string]any{"type": "invalid_tool_call", "id": "itc1", "name": "search", "args": "{bad", "error": "parse error"},
			check: func(t *testing.T, block ContentBlock) {
				tc, ok := block.(InvalidToolCallBlock)
				if !ok {
					t.Fatalf("expected InvalidToolCallBlock, got %T", block)
				}
				if tc.ID != "itc1" || tc.Name != "search" || tc.Args != "{bad" || tc.Error != "parse error" {
					t.Fatalf("field mismatch: %#v", tc)
				}
			},
		},
		{
			name:  "unknown type uses escape hatch",
			input: map[string]any{"type": "tool_use", "id": "u1", "input": "raw"},
			check: func(t *testing.T, block ContentBlock) {
				ns, ok := block.(NonStandardContentBlock)
				if !ok {
					t.Fatalf("expected NonStandardContentBlock, got %T", block)
				}
				if ns.Type != "tool_use" {
					t.Fatalf("type mismatch: %#v", ns)
				}
				if _, hasType := ns.Value["type"]; hasType {
					t.Fatalf("value should not retain type key: %#v", ns.Value)
				}
				if ns.Value["id"] != "u1" || ns.Value["input"] != "raw" {
					t.Fatalf("value mismatch: %#v", ns.Value)
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.check(t, ParseContentBlock(tc.input))
		})
	}
}

func TestBlockToMapNil(t *testing.T) {
	if got := BlockToMap(nil); got != nil {
		t.Fatalf("BlockToMap(nil) = %#v, want nil", got)
	}
}

func TestBlockToMapAllTypes(t *testing.T) {
	cases := []struct {
		name  string
		block ContentBlock
		want  map[string]any
	}{
		{
			name:  "text full",
			block: TextBlock{ID: "t1", Text: "hello", Index: 1, Extras: map[string]any{"extra": "e"}},
			want:  map[string]any{"type": "text", "text": "hello", "id": "t1", "index": 1, "extra": "e"},
		},
		{
			name:  "text with annotations",
			block: TextBlock{Text: "hi", Annotations: []Annotation{Citation{ID: "c1"}}},
			want:  map[string]any{"type": "text", "text": "hi", "annotations": []Annotation{Citation{ID: "c1"}}},
		},
		{
			name:  "extras cannot overwrite known keys",
			block: TextBlock{Text: "hello", Extras: map[string]any{"type": "evil", "text": "evil"}},
			want:  map[string]any{"type": "text", "text": "hello"},
		},
		{
			name:  "image full",
			block: ImageBlock{ID: "i1", FileID: "f1", MimeType: "image/png", Index: 2, URL: "http://x/i.png", Base64: "QQ", Extras: map[string]any{"w": 100}},
			want:  map[string]any{"type": "image", "id": "i1", "file_id": "f1", "mime_type": "image/png", "index": 2, "url": "http://x/i.png", "base64": "QQ", "w": 100},
		},
		{
			name:  "image empty",
			block: ImageBlock{},
			want:  map[string]any{"type": "image"},
		},
		{
			name:  "video",
			block: VideoBlock{URL: "http://x/v.mp4"},
			want:  map[string]any{"type": "video", "url": "http://x/v.mp4"},
		},
		{
			name:  "audio",
			block: AudioBlock{Base64: "QQ", MimeType: "audio/mp3"},
			want:  map[string]any{"type": "audio", "base64": "QQ", "mime_type": "audio/mp3"},
		},
		{
			name:  "file",
			block: FileBlock{FileID: "ff1"},
			want:  map[string]any{"type": "file", "file_id": "ff1"},
		},
		{
			name:  "text-plain full",
			block: PlainTextBlock{ID: "p1", FileID: "pf1", MimeType: "text/markdown", Index: 3, URL: "http://x/d", Base64: "QQ", Text: "body", Title: "title", Context: "ctx", Extras: map[string]any{"e": 1}},
			want:  map[string]any{"type": "text-plain", "mime_type": "text/markdown", "id": "p1", "file_id": "pf1", "index": 3, "url": "http://x/d", "base64": "QQ", "text": "body", "title": "title", "context": "ctx", "e": 1},
		},
		{
			name:  "text-plain minimal keeps mime_type key",
			block: PlainTextBlock{},
			want:  map[string]any{"type": "text-plain", "mime_type": ""},
		},
		{
			name:  "reasoning",
			block: ReasoningBlock{ID: "r1", Reasoning: "why", Index: 0},
			want:  map[string]any{"type": "reasoning", "id": "r1", "reasoning": "why", "index": 0},
		},
		{
			name:  "tool_call",
			block: ToolCallBlock{ID: "tc1", Name: "search", Args: map[string]any{"q": "x"}, Index: 1},
			want:  map[string]any{"type": "tool_call", "name": "search", "id": "tc1", "args": map[string]any{"q": "x"}, "index": 1},
		},
		{
			name:  "tool_call_chunk",
			block: ToolCallChunkBlock{ID: "tcc1", Name: "search", Args: "{\"q\":", Index: 2},
			want:  map[string]any{"type": "tool_call_chunk", "id": "tcc1", "name": "search", "args": "{\"q\":", "index": 2},
		},
		{
			name:  "invalid_tool_call",
			block: InvalidToolCallBlock{ID: "itc1", Name: "search", Args: "{bad", Error: "parse error"},
			want:  map[string]any{"type": "invalid_tool_call", "id": "itc1", "name": "search", "args": "{bad", "error": "parse error"},
		},
		{
			name:  "server_tool_call",
			block: ServerToolCall{ID: "s1", Name: "web_search", Args: map[string]any{"q": "x"}},
			want:  map[string]any{"type": "server_tool_call", "name": "web_search", "id": "s1", "args": map[string]any{"q": "x"}},
		},
		{
			name:  "server_tool_call_chunk",
			block: ServerToolCallChunk{ID: "sc1", Name: "code_exec", Args: "{\"code\":"},
			want:  map[string]any{"type": "server_tool_call_chunk", "id": "sc1", "name": "code_exec", "args": "{\"code\":"},
		},
		{
			name:  "server_tool_result",
			block: ServerToolResult{ID: "sr1", ToolCallID: "s1", Status: "success", Output: "42", Index: 4},
			want:  map[string]any{"type": "server_tool_result", "tool_call_id": "s1", "status": "success", "id": "sr1", "output": "42", "index": 4},
		},
		{
			name:  "nonstandard round trip",
			block: NonStandardContentBlock{Type: "tool_use", Value: map[string]any{"id": "u1", "input": "raw"}},
			want:  map[string]any{"type": "tool_use", "id": "u1", "input": "raw"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := BlockToMap(tc.block)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("BlockToMap() = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// blockExtras returns the Extras map of a typed block so tests can verify
// that CloneBlock performs an independent copy.
func blockExtras(b ContentBlock) map[string]any {
	switch v := b.(type) {
	case TextBlock:
		return v.Extras
	case ImageBlock:
		return v.Extras
	case VideoBlock:
		return v.Extras
	case AudioBlock:
		return v.Extras
	case FileBlock:
		return v.Extras
	case PlainTextBlock:
		return v.Extras
	case ReasoningBlock:
		return v.Extras
	case ToolCallBlock:
		return v.Extras
	case ToolCallChunkBlock:
		return v.Extras
	case InvalidToolCallBlock:
		return v.Extras
	case ServerToolCall:
		return v.Extras
	case ServerToolCallChunk:
		return v.Extras
	case ServerToolResult:
		return v.Extras
	default:
		return nil
	}
}

func TestCloneBlockNil(t *testing.T) {
	if got := CloneBlock(nil); got != nil {
		t.Fatalf("CloneBlock(nil) = %#v, want nil", got)
	}
}

func TestCloneBlockAllTypes(t *testing.T) {
	blocks := []ContentBlock{
		TextBlock{ID: "t1", Text: "x", Annotations: []Annotation{Citation{ID: "c1"}}, Extras: map[string]any{"k": "v"}},
		ImageBlock{ID: "i1", URL: "http://x/i.png", Extras: map[string]any{"k": "v"}},
		VideoBlock{ID: "v1", URL: "http://x/v.mp4", Extras: map[string]any{"k": "v"}},
		AudioBlock{ID: "a1", Base64: "QQ", Extras: map[string]any{"k": "v"}},
		FileBlock{ID: "f1", FileID: "ff1", Extras: map[string]any{"k": "v"}},
		PlainTextBlock{ID: "p1", Text: "body", Title: "t", Context: "c", Extras: map[string]any{"k": "v"}},
		ReasoningBlock{ID: "r1", Reasoning: "why", Extras: map[string]any{"k": "v"}},
		ToolCallBlock{ID: "tc1", Name: "search", Args: map[string]any{"q": "x"}, Extras: map[string]any{"k": "v"}},
		ToolCallChunkBlock{ID: "tcc1", Name: "search", Args: "{\"q\":", Extras: map[string]any{"k": "v"}},
		InvalidToolCallBlock{ID: "itc1", Name: "search", Args: "{bad", Error: "e", Extras: map[string]any{"k": "v"}},
		ServerToolCall{ID: "s1", Name: "web_search", Args: map[string]any{"q": "x"}, Extras: map[string]any{"k": "v"}},
		ServerToolCallChunk{ID: "sc1", Name: "code_exec", Args: "{\"code\":", Extras: map[string]any{"k": "v"}},
		ServerToolResult{ID: "sr1", ToolCallID: "s1", Status: "success", Output: map[string]any{"n": 1}, Extras: map[string]any{"k": "v"}},
		ServerToolResult{ID: "sr2", ToolCallID: "s2", Status: "error", Output: "scalar"},
		NonStandardContentBlock{Type: "tool_use", Value: map[string]any{"k": "v"}},
	}
	for i, block := range blocks {
		clone := CloneBlock(block)
		if !reflect.DeepEqual(clone, block) {
			t.Fatalf("block %d (%T): clone = %#v, want %#v", i, block, clone, block)
		}
		// Mutating the clone's extras must not leak into the original.
		if extras := blockExtras(clone); extras != nil {
			extras["k"] = "changed"
			if got := blockExtras(block)["k"]; got != "v" {
				t.Fatalf("block %d (%T): clone extras mutation leaked into original: %v", i, block, got)
			}
		}
	}

	// Args maps are cloned independently too.
	tc := ToolCallBlock{Name: "search", Args: map[string]any{"q": "x"}}
	tcClone := CloneBlock(tc).(ToolCallBlock)
	tcClone.Args["q"] = "changed"
	if tc.Args["q"] != "x" {
		t.Fatal("ToolCallBlock clone Args mutation changed original")
	}

	// NonStandardContentBlock values are cloned independently.
	ns := NonStandardContentBlock{Type: "tool_use", Value: map[string]any{"k": "v"}}
	nsClone := CloneBlock(ns).(NonStandardContentBlock)
	nsClone.Value["k"] = "changed"
	if ns.Value["k"] != "v" {
		t.Fatal("NonStandardContentBlock clone Value mutation changed original")
	}

	// Annotations slice is copied: appending to the clone does not grow the original.
	tb := TextBlock{Text: "x", Annotations: []Annotation{Citation{ID: "c1"}}}
	tbClone := CloneBlock(tb).(TextBlock)
	tbClone.Annotations = append(tbClone.Annotations, Citation{ID: "c2"})
	if len(tb.Annotations) != 1 {
		t.Fatal("TextBlock clone annotations mutation changed original")
	}
}
