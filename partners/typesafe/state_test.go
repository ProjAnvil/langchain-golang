package typesafe

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// SerializeState mirrors langchain_typesafe/_state.py (#40542): the root must
// be a string, message, sequence, or object; messages anywhere in the tree
// become OpenAI role/content dicts with IDs omitted.

const rootStateError = "TypeSafe state must be a string, object, array, BaseMessage, or sequence of BaseMessage objects."

func TestSerializeStateStringRoot(t *testing.T) {
	got, err := SerializeState("hello")
	if err != nil {
		t.Fatalf("serialize string: %v", err)
	}
	if got != "hello" {
		t.Errorf("got %#v, want \"hello\"", got)
	}
}

func TestSerializeStateScalarRootsRejected(t *testing.T) {
	for _, root := range []any{nil, 42, int64(7), 3.14, true} {
		_, err := SerializeState(root)
		if err == nil {
			t.Errorf("expected error for root %#v", root)
			continue
		}
		if err.Error() != rootStateError {
			t.Errorf("root %#v error mismatch:\n got %q\nwant %q", root, err.Error(), rootStateError)
		}
	}
}

func TestSerializeStateMessageSequence(t *testing.T) {
	ai := messages.AI("let me check")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "analyze", Args: map[string]any{"baz": "buz"}}}
	tool := messages.Tool("1", "foobar")
	tool.Name = "bar"

	got, err := SerializeState([]messages.Message{
		messages.System("sys"),
		messages.Human("hi"),
		ai,
		tool,
	})
	if err != nil {
		t.Fatalf("serialize sequence: %v", err)
	}
	want := []any{
		map[string]any{"role": "system", "content": "sys"},
		map[string]any{"role": "user", "content": "hi"},
		map[string]any{
			"role":    "assistant",
			"content": "let me check",
			"tool_calls": []any{map[string]any{
				"type": "function",
				"id":   "1",
				"function": map[string]any{
					"name":      "analyze",
					"arguments": `{"baz":"buz"}`,
				},
			}},
		},
		map[string]any{"role": "tool", "content": "foobar", "tool_call_id": "1", "name": "bar"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("sequence mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestSerializeStateSingleMessageRoot(t *testing.T) {
	msg := messages.Human("hi")
	msg.ID = "abc-123"
	got, err := SerializeState(msg)
	if err != nil {
		t.Fatalf("serialize single message: %v", err)
	}
	// A single BaseMessage converts to one role/content dict (Python's
	// convert_to_openai_messages returns the dict for a single message), and
	// the message ID is omitted.
	want := map[string]any{"role": "user", "content": "hi"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("single message mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestSerializeStatePointerMessage(t *testing.T) {
	msg := messages.System("sys")
	got, err := SerializeState(&msg)
	if err != nil {
		t.Fatalf("serialize pointer message: %v", err)
	}
	want := map[string]any{"role": "system", "content": "sys"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("pointer message mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestSerializeStateNested(t *testing.T) {
	got, err := SerializeState(map[string]any{
		"k": []any{messages.Human("m"), 1, "x", nil, 2.5, false},
		"s": "str",
	})
	if err != nil {
		t.Fatalf("serialize nested: %v", err)
	}
	want := map[string]any{
		"k": []any{map[string]any{"role": "user", "content": "m"}, 1, "x", nil, 2.5, false},
		"s": "str",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("nested mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestSerializeStateMixedBlocks(t *testing.T) {
	msg := messages.Human("")
	msg = msg.WithContentBlocks([]messages.ContentBlock{
		messages.TextBlock{Text: "look"},
		messages.ImageBlock{Base64: "abc", MimeType: "image/png"},
		messages.ImageBlock{URL: "https://example.com/y.png"},
	})
	got, err := SerializeState(msg)
	if err != nil {
		t.Fatalf("serialize mixed blocks: %v", err)
	}
	want := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "text", "text": "look"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,abc"}},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/y.png"}},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("mixed blocks mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestSerializeStateAllTextBlocksJoin(t *testing.T) {
	msg := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.TextBlock{Text: "a"},
		messages.TextBlock{Text: "b"},
	})
	got, err := SerializeState(msg)
	if err != nil {
		t.Fatalf("serialize text blocks: %v", err)
	}
	want := map[string]any{"role": "user", "content": "a\nb"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("text join mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestSerializeStateImageDefaultsAndAudio(t *testing.T) {
	msg := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{Base64: "zz"},                         // mime defaults to image/jpeg
		messages.AudioBlock{Base64: "xyz", MimeType: "audio/wav"}, // input_audio shape
	})
	got, err := SerializeState(msg)
	if err != nil {
		t.Fatalf("serialize image/audio: %v", err)
	}
	want := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,zz"}},
			map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "xyz", "format": "wav"}},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("image/audio mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestSerializeStateUnsupportedBlocks(t *testing.T) {
	msg := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.VideoBlock{URL: "https://example.com/v.mp4"},
	})
	if _, err := SerializeState(msg); err == nil {
		t.Fatal("expected error for video block in state message")
	}

	audioNoData := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.AudioBlock{URL: "https://example.com/a.wav"},
	})
	if _, err := SerializeState(audioNoData); err == nil {
		t.Fatal("expected error for audio block without base64")
	}
}

func TestSerializeStateNonStringMapKeys(t *testing.T) {
	_, err := SerializeState(map[int]string{1: "a"})
	if err == nil {
		t.Fatal("expected error for non-string map keys")
	}
	if err.Error() != "TypeSafe state object keys must be strings." {
		t.Errorf("error mismatch: %q", err.Error())
	}
}

type widget struct{}

func TestSerializeStateUnsupportedValues(t *testing.T) {
	errMsg := "Unsupported TypeSafe state value: widget."
	if _, err := SerializeState(widget{}); err == nil || err.Error() != errMsg {
		t.Errorf("root widget error mismatch: %v", err)
	}
	if _, err := SerializeState([]any{widget{}}); err == nil || err.Error() != errMsg {
		t.Errorf("nested widget error mismatch: %v", err)
	}
	if _, err := SerializeState(map[string]any{"w": widget{}}); err == nil || err.Error() != errMsg {
		t.Errorf("map widget error mismatch: %v", err)
	}
	// Python excludes bytes/bytearray from the Sequence branch.
	if _, err := SerializeState([]byte("ab")); err == nil || err.Error() != "Unsupported TypeSafe state value: []uint8." {
		t.Errorf("bytes error mismatch: %v", err)
	}
}

func TestSerializeStateNonStandardPassthrough(t *testing.T) {
	msg := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.NonStandardContentBlock{
			Type:  "image_url",
			Value: map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/i.png"}},
		},
	})
	got, err := SerializeState(msg)
	if err != nil {
		t.Fatalf("serialize non-standard: %v", err)
	}
	want := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/i.png"}},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("non-standard mismatch:\n got %#v\nwant %#v", got, want)
	}
}

func TestSerializeStateAudioFormatVariants(t *testing.T) {
	msg := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.AudioBlock{Base64: "a", Extras: map[string]any{"format": "mp3"}},
		messages.AudioBlock{Base64: "b", MimeType: "mpeg"}, // no slash: whole mime is the format
	})
	got, err := SerializeState(msg)
	if err != nil {
		t.Fatalf("serialize audio: %v", err)
	}
	content := got.(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["input_audio"].(map[string]any)["format"] != "mp3" {
		t.Errorf("explicit format extra ignored: %#v", content[0])
	}
	if content[1].(map[string]any)["input_audio"].(map[string]any)["format"] != "mpeg" {
		t.Errorf("suffix-less mime not used as format: %#v", content[1])
	}
}

func TestSerializeStateImageWithoutSource(t *testing.T) {
	msg := messages.Human("").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{FileID: "file-1"},
	})
	if _, err := SerializeState(msg); err == nil {
		t.Fatal("expected error for image block with only file_id")
	}
}

func TestSerializeStateUnknownRole(t *testing.T) {
	msg := messages.Message{Role: messages.Role("bogus"), Content: "x"}
	if _, err := SerializeState(msg); err == nil {
		t.Fatal("expected error for unknown message role")
	}
}

func TestSerializeStateNilToolCallArgs(t *testing.T) {
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "t"}}
	got, err := SerializeState(ai)
	if err != nil {
		t.Fatalf("serialize nil args: %v", err)
	}
	toolCalls := got.(map[string]any)["tool_calls"].([]any)
	if got := toolCalls[0].(map[string]any)["function"].(map[string]any)["arguments"]; got != "{}" {
		t.Errorf("nil args arguments = %q, want {}", got)
	}
}

func TestSerializeStateNilMessagePointer(t *testing.T) {
	_, err := SerializeState((*messages.Message)(nil))
	if err == nil || err.Error() != "Unsupported TypeSafe state value: Message." {
		t.Errorf("nil message pointer error mismatch: %v", err)
	}
}
