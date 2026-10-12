package decisions

import (
	"cmp"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

// This file maps core message content onto OpenAI Decisions input parts.
// The mapping is deliberately self-contained (parity wave 3 design decision
// #7 / DIVERGENCES #13): core/messages.ConvertToOpenAIMessages emits
// LangChain-shape image blocks rather than the OpenAI image_url shape the
// Decisions API expects, and the partners/openai contentblocks.go precedent
// (responsesContentParts/imageDataURL) is unexported and therefore not
// importable from this package.
//
// Image semantics (audit minor #5, PR #41125): both base64 and hosted-URL
// image blocks become {"type":"input_image","image_url":...} without error;
// only audio/video/file and other non-convertible blocks are rejected with
// ErrInvalidInput. (Python's _input.py:61-68 rejects hosted URLs; this port
// accepts them — see DIVERGENCES.md #13.)

// messageToInputParts converts one message's content into Decisions input
// parts. A message without content blocks becomes a single input_text part
// carrying its plain-text content.
func messageToInputParts(msg messages.Message) ([]map[string]any, error) {
	if len(msg.ContentBlocks) == 0 {
		return []map[string]any{{"type": "input_text", "text": msg.Content}}, nil
	}
	parts := make([]map[string]any, 0, len(msg.ContentBlocks))
	for _, block := range msg.ContentBlocks {
		switch b := block.(type) {
		case messages.TextBlock:
			parts = append(parts, map[string]any{"type": "input_text", "text": b.Text})
		case messages.ImageBlock:
			url, err := imageSourceURL(b)
			if err != nil {
				return nil, err
			}
			parts = append(parts, map[string]any{"type": "input_image", "image_url": url})
		default:
			return nil, unsupportedBlockError(block.BlockType())
		}
	}
	return parts, nil
}

// imageSourceURL resolves an image block to the URL sent as image_url: the
// URL wins, then base64 (as a data: URI with the mime type defaulting to
// image/jpeg), mirroring the partners/openai imageDataURL precedent.
func imageSourceURL(b messages.ImageBlock) (string, error) {
	if b.URL != "" {
		return b.URL, nil
	}
	if b.Base64 != "" {
		return "data:" + cmp.Or(b.MimeType, "image/jpeg") + ";base64," + b.Base64, nil
	}
	return "", unsupportedBlockError("image")
}

// IsBase64ImageBlock reports whether block is an image whose source is
// base64 data — a data: URL or a base64 payload — mirroring Python's
// _is_base64_image data:-prefix judgment (model_router.py:259). Hosted-URL
// images and non-image blocks report false. It is exported because the
// ModelRouter middleware (partners/openai/middleware) applies it to decide
// which image blocks survive routing.
func IsBase64ImageBlock(block messages.ContentBlock) bool {
	switch b := block.(type) {
	case messages.ImageBlock:
		return b.Base64 != "" || strings.HasPrefix(b.URL, "data:")
	case messages.NonStandardContentBlock:
		if b.Type != "image" && b.Type != "image_url" {
			return false
		}
		if url, ok := b.Value["url"].(string); ok {
			return strings.HasPrefix(url, "data:")
		}
		if base64, ok := b.Value["base64"].(string); ok && base64 != "" {
			return true
		}
		if wrapper, ok := b.Value["image_url"].(map[string]any); ok {
			if url, ok := wrapper["url"].(string); ok {
				return strings.HasPrefix(url, "data:")
			}
		}
		return false
	default:
		return false
	}
}

// unsupportedBlockError rejects content the Decisions API cannot accept
// (Python _input.py:70-74; the Go message is a paraphrase that keeps the
// block type, per the parity wave 3 plan's Global Constraints).
func unsupportedBlockError(blockType string) error {
	return fmt.Errorf(
		"%w: unsupported content block type for the OpenAI Decisions API: %q. Only text and images are supported",
		lcerrors.ErrInvalidInput, blockType,
	)
}
