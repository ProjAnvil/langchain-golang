package openai

import (
	"cmp"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Multimodal input conversion. Human messages that carry ContentBlocks are
// serialized as structured content on both APIs (mirroring Python's
// _convert_message_to_dict for Chat Completions and the Responses payload
// construction in chat_models/base.py); plain-text messages keep a bare
// string content so payloads do not grow a parts array for text-only chats.
//
// Shapes:
//
//	Responses      Chat Completions
//	input_text     {"type":"text","text":...}
//	input_image    {"type":"image_url","image_url":{"url":...}}
//	(image_url is  (image_url wraps the URL in an object;
//	 a bare string)  base64 images become data: URIs)
//	input_audio    input_audio (identical on both)

// inputContentPart is one structured Responses input content part.
type inputContentPart struct {
	Type       string      `json:"type"`
	Text       string      `json:"text,omitempty"`
	ImageURL   string      `json:"image_url,omitempty"`
	FileID     string      `json:"file_id,omitempty"`
	Detail     string      `json:"detail,omitempty"`
	InputAudio *inputAudio `json:"input_audio,omitempty"`
}

// inputAudio is the audio payload shared by both APIs
// ({"data": <base64>, "format": "mp3"|"wav"|...}).
type inputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format,omitempty"`
}

// chatContentPart is one structured Chat Completions content part.
type chatContentPart struct {
	Type       string        `json:"type"`
	Text       string        `json:"text,omitempty"`
	ImageURL   *chatImageURL `json:"image_url,omitempty"`
	InputAudio *inputAudio   `json:"input_audio,omitempty"`
}

// chatImageURL is the Chat Completions image_url object (the URL is nested,
// unlike the Responses API's bare string).
type chatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// responsesContentParts converts human-message content blocks into Responses
// API input content parts. NonStandard blocks pass through verbatim except
// OpenAI-native "image_url" blocks, which are rewritten to input_image the
// way Python's _convert_chat_completions_blocks_to_responses does.
func responsesContentParts(blocks []messages.ContentBlock) ([]any, error) {
	parts := make([]any, 0, len(blocks))
	for _, block := range blocks {
		switch b := block.(type) {
		case messages.TextBlock:
			parts = append(parts, inputContentPart{Type: "input_text", Text: b.Text})
		case messages.ImageBlock:
			part, err := responsesImagePart(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		case messages.AudioBlock:
			audio, err := audioPart(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, inputContentPart{Type: "input_audio", InputAudio: audio})
		case messages.NonStandardContentBlock:
			parts = append(parts, passthroughResponsesPart(b))
		default:
			return nil, fmt.Errorf("openai: unsupported content block type %q in human message", block.BlockType())
		}
	}
	return parts, nil
}

// chatContentParts converts human-message content blocks into Chat
// Completions content parts. NonStandard blocks pass through verbatim
// (OpenAI-native {"type":"image_url",...} blocks are already in API shape).
func chatContentParts(blocks []messages.ContentBlock) ([]any, error) {
	parts := make([]any, 0, len(blocks))
	for _, block := range blocks {
		switch b := block.(type) {
		case messages.TextBlock:
			parts = append(parts, chatContentPart{Type: "text", Text: b.Text})
		case messages.ImageBlock:
			if b.FileID != "" && b.URL == "" && b.Base64 == "" {
				return nil, fmt.Errorf("openai: chat completions image content block requires url or base64 (file_id is Responses-only)")
			}
			url, err := imageSourceURL(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, chatContentPart{
				Type:     "image_url",
				ImageURL: &chatImageURL{URL: url, Detail: imageDetail(b)},
			})
		case messages.AudioBlock:
			audio, err := audioPart(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, chatContentPart{Type: "input_audio", InputAudio: audio})
		case messages.NonStandardContentBlock:
			parts = append(parts, messages.BlockToMap(b))
		default:
			return nil, fmt.Errorf("openai: unsupported content block type %q in human message", block.BlockType())
		}
	}
	return parts, nil
}

// responsesImagePart maps an ImageBlock onto an input_image part: url wins,
// then base64 (as a data: URI), then file_id.
func responsesImagePart(b messages.ImageBlock) (inputContentPart, error) {
	url, useURL, err := responsesImageSource(b)
	if err != nil {
		return inputContentPart{}, err
	}
	part := inputContentPart{Type: "input_image", Detail: imageDetail(b)}
	switch {
	case useURL:
		part.ImageURL = url
	default:
		part.FileID = b.FileID
	}
	return part, nil
}

// responsesImageSource resolves the image source for the Responses API,
// reporting whether the source is a URL (true) or a file_id (false).
func responsesImageSource(b messages.ImageBlock) (source string, isURL bool, err error) {
	switch {
	case b.URL != "":
		return b.URL, true, nil
	case b.Base64 != "":
		return imageDataURL(b), true, nil
	case b.FileID != "":
		return b.FileID, false, nil
	default:
		return "", false, fmt.Errorf("openai: image content block requires url, base64, or file_id")
	}
}

// imageSourceURL resolves the image source for the Chat Completions API
// (no file_id variant exists there).
func imageSourceURL(b messages.ImageBlock) (string, error) {
	if b.URL != "" {
		return b.URL, nil
	}
	if b.Base64 != "" {
		return imageDataURL(b), nil
	}
	return "", fmt.Errorf("openai: image content block requires url or base64")
}

// imageDataURL builds a "data:<mime>;base64,<data>" URI, defaulting the media
// type to image/jpeg when the block does not carry one.
func imageDataURL(b messages.ImageBlock) string {
	mime := cmp.Or(b.MimeType, "image/jpeg")
	return "data:" + mime + ";base64," + b.Base64
}

// imageDetail surfaces the optional "detail" hint from block extras.
func imageDetail(b messages.ImageBlock) string {
	detail, _ := b.Extras["detail"].(string)
	return detail
}

// audioPart maps an AudioBlock onto the input_audio payload. Base64 data is
// required; the format prefers an explicit "format" extra and falls back to
// the suffix of mime_type ("audio/wav" -> "wav"), mirroring Python.
func audioPart(b messages.AudioBlock) (*inputAudio, error) {
	if b.Base64 == "" {
		return nil, fmt.Errorf("openai: audio content block requires base64 data")
	}
	format, _ := b.Extras["format"].(string)
	if format == "" && b.MimeType != "" {
		if idx := strings.LastIndex(b.MimeType, "/"); idx >= 0 && idx+1 < len(b.MimeType) {
			format = b.MimeType[idx+1:]
		} else {
			format = b.MimeType
		}
	}
	return &inputAudio{Data: b.Base64, Format: format}, nil
}

// passthroughResponsesPart forwards a provider-native block to the Responses
// API, rewriting the OpenAI-native spellings Python converts
// (_convert_chat_completions_blocks_to_responses): "text" -> input_text and
// "image_url" -> input_image with the URL unwrapped to a bare string.
func passthroughResponsesPart(b messages.NonStandardContentBlock) any {
	m := messages.BlockToMap(b)
	switch m["type"] {
	case "text":
		return inputContentPart{Type: "input_text", Text: stringOf(m["text"])}
	case "image_url":
		part := inputContentPart{Type: "input_image"}
		if wrapper, ok := m["image_url"].(map[string]any); ok {
			part.ImageURL = stringOf(wrapper["url"])
			part.Detail = stringOf(wrapper["detail"])
		}
		return part
	default:
		return m
	}
}

func stringOf(v any) string {
	s, _ := v.(string)
	return s
}
