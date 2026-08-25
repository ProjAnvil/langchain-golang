package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	// Register decoders so image.DecodeConfig can size jpeg/png/gif inputs,
	// the Go equivalent of Python's PIL-backed sizing.
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/tiktoken-go/tokenizer"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
)

// Compile-time guards: ChatModel provides the provider-specific token
// counters the language package dispatches to.
var (
	_ language.TokenCounter        = ChatModel{}
	_ language.MessageTokenCounter = ChatModel{}
)

// tiktokenCodecs caches constructed BPE codecs by encoding so the (embedded)
// vocabulary and split regexp are built once per process.
var tiktokenCodecs sync.Map // tokenizer.Encoding -> tokenizer.Codec

// getCodec returns the cached codec for encoding, constructing it on first
// use. The tiktoken-go vocabularies are embedded in the library, so this does
// no network I/O.
func getCodec(encoding tokenizer.Encoding) (tokenizer.Codec, error) {
	if cached, ok := tiktokenCodecs.Load(encoding); ok {
		return cached.(tokenizer.Codec), nil
	}
	codec, err := tokenizer.Get(encoding)
	if err != nil {
		return nil, err
	}
	actual, _ := tiktokenCodecs.LoadOrStore(encoding, codec)
	return actual.(tokenizer.Codec), nil
}

// encodingForModel mirrors Python ChatOpenAI._get_encoding_model
// (chat_models/base.py:2077): tiktoken_model_name overrides the model name,
// then the tokenizer's model table resolves the encoding; unknown models fall
// back to o200k_base for gpt-4o/gpt-4.1/gpt-5 prefixes and cl100k_base
// otherwise.
func encodingForModel(model string) tokenizer.Encoding {
	if codec, err := tokenizer.ForModel(tokenizer.Model(model)); err == nil {
		return tokenizer.Encoding(codec.GetName())
	}
	lower := strings.ToLower(model)
	if strings.HasPrefix(lower, "gpt-4o") ||
		strings.HasPrefix(lower, "gpt-4.1") ||
		strings.HasPrefix(lower, "gpt-4.5") || // tiktoken-go v0.7.0's table lacks gpt-4.5; Python tiktoken maps it to o200k_base
		strings.HasPrefix(lower, "gpt-5") {
		return tokenizer.O200kBase
	}
	return tokenizer.Cl100kBase
}

// getEncodingModel resolves the effective model name (honoring
// TiktokenModelName) and its codec, mirroring _get_encoding_model.
func (m ChatModel) getEncodingModel() (string, tokenizer.Codec, error) {
	model := m.config.Model
	if m.config.TiktokenModelName != "" {
		model = m.config.TiktokenModelName
	}
	codec, err := getCodec(encodingForModel(model))
	return model, codec, err
}

// GetTokenIDs mirrors ChatOpenAI.get_token_ids (chat_models/base.py:2093) using
// a real tiktoken BPE encoding (cl100k_base/o200k_base, selected like Python's
// _get_encoding_model). The signature has no error return, so on an
// (unexpected) codec load failure it falls back to
// language.DefaultGetTokenIDs' 4-rune approximation.
func (m ChatModel) GetTokenIDs(text string) []int {
	_, codec, err := m.getEncodingModel()
	if err != nil {
		return language.DefaultGetTokenIDs(text)
	}
	ids, _, err := codec.Encode(text)
	if err != nil {
		return language.DefaultGetTokenIDs(text)
	}
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}

// GetNumTokens returns the number of tokens in text (len of GetTokenIDs),
// mirroring BaseLanguageModel.get_num_tokens.
func (m ChatModel) GetNumTokens(text string) int {
	return len(m.GetTokenIDs(text))
}

// GetNumTokensFromMessages mirrors ChatOpenAI.get_num_tokens_from_messages
// (chat_models/base.py:2103), including the per-message/per-name overhead, the
// +3 reply primer, and OpenAI's image token formula. Remaining divergences
// from Python: tool schemas are ignored (Python warns and ignores them), and
// images that cannot be sized (fetch/decode failure) are ignored, matching
// Python's behavior when PIL/httpx are missing or the fetch fails. Like
// Python, it errors for models outside the gpt-3.5-turbo / gpt-4 / gpt-5
// families.
func (m ChatModel) GetNumTokensFromMessages(msgs []messages.Message) (int, error) {
	model, _, _ := m.getEncodingModel()
	tokensPerMessage := 3
	tokensPerName := 1
	switch {
	case strings.HasPrefix(model, "gpt-3.5-turbo-0301"):
		// Every message follows <|im_start|>{role/name}\n{content}<|im_end|>\n.
		tokensPerMessage = 4
		tokensPerName = -1 // if there's a name, the role is omitted
	case strings.HasPrefix(model, "gpt-3.5-turbo"),
		strings.HasPrefix(model, "gpt-4"),
		strings.HasPrefix(model, "gpt-5"):
	default:
		return 0, fmt.Errorf(
			"GetNumTokensFromMessages() is not presently implemented for model %s. "+
				"See https://platform.openai.com/docs/guides/text-generation/managing-tokens "+
				"for information on how messages are converted to tokens",
			model)
	}

	total := 0
	for _, msg := range msgs {
		total += tokensPerMessage
		total += m.GetNumTokens(openAIWireRole(msg.Role))
		if msg.Content != "" {
			total += m.GetNumTokens(msg.Content)
		}
		for _, block := range msg.ContentBlocks {
			switch typed := block.(type) {
			case messages.TextBlock:
				total += m.GetNumTokens(typed.Text)
			case messages.ImageBlock:
				total += imageBlockTokens(typed)
			}
		}
		if msg.Name != "" {
			total += m.GetNumTokens(msg.Name) + tokensPerName
		}
		// Tool call token counting is not documented by OpenAI; this mirrors
		// Python's approximation of counting function name + arguments.
		for _, call := range msg.ToolCalls {
			total += m.GetNumTokens(call.Name)
			if data, err := json.Marshal(call.Args); err == nil {
				total += m.GetNumTokens(string(data))
			}
		}
		if msg.ToolCallID != "" {
			// Inferred approximation from Python: tool_call_id contributes 3.
			total += 3
		}
	}
	// Every reply is primed with <|im_start|>assistant.
	total += 3
	return total, nil
}

const (
	// imageFetchTimeout matches Python _url_to_size's 5-second timeout.
	imageFetchTimeout = 5 * time.Second
	// maxImageBytes matches OpenAI's 50 MB payload limit, enforced by
	// Python _url_to_size.
	maxImageBytes = 50 * 1024 * 1024
)

// imageBlockTokens returns the token cost of one image block following
// OpenAI's formula: detail "low" is a flat 85 (exact match, as in Python);
// otherwise the image is sized (base64 payload or URL fetch) and counted via
// countImageTokens. Images that cannot be sized contribute 0, matching
// Python's behavior when PIL/httpx are missing or the fetch fails. (Python
// raises on undecodable base64; the Go port tolerates it and counts 0.)
func imageBlockTokens(block messages.ImageBlock) int {
	if detail, _ := block.Extras["detail"].(string); detail == "low" {
		return 85
	}
	width, height, ok := imageBlockSize(block)
	if !ok {
		return 0
	}
	return countImageTokens(width, height)
}

// imageBlockSize mirrors Python _url_to_size (chat_models/base.py:3953): it
// resolves the pixel size of an image block from its base64 payload or URL,
// reporting ok=false on any failure (bad base64, undecodable image, fetch
// error/timeout, oversized payload).
func imageBlockSize(block messages.ImageBlock) (width, height int, ok bool) {
	if block.Base64 != "" {
		return base64ImageSize(block.Base64)
	}
	if block.URL == "" {
		return 0, 0, false
	}
	if strings.HasPrefix(block.URL, "data:") {
		return base64ImageSize(block.URL)
	}
	return fetchImageSize(block.URL)
}

// base64ImageSize sizes a base64-encoded image, accepting both raw base64 and
// data URLs ("data:image/png;base64,...").
func base64ImageSize(data string) (int, int, bool) {
	if strings.HasPrefix(data, "data:") {
		comma := strings.Index(data, ",")
		if comma < 0 {
			return 0, 0, false
		}
		data = data[comma+1:]
	}
	raw, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return 0, 0, false
	}
	return decodeImageSize(raw)
}

// decodeImageSize reads the dimensions of an encoded image via
// image.DecodeConfig (jpeg/png/gif registered above).
func decodeImageSize(raw []byte) (int, int, bool) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return 0, 0, false
	}
	return cfg.Width, cfg.Height, true
}

// fetchImageSize GETs url with a 5s timeout and a 50 MB cap, then decodes the
// dimensions. Any failure (timeout, non-200, oversize, undecodable) reports
// ok=false so the image is ignored, like Python's _url_to_size.
func fetchImageSize(url string) (int, int, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), imageFetchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, 0, false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		slog.Warn("openai: failed to fetch image for token counting",
			slog.String("url", url), slog.String("error", err.Error()))
		return 0, 0, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, 0, false
	}
	if resp.ContentLength > maxImageBytes {
		return 0, 0, false
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxImageBytes+1))
	if err != nil || len(data) > maxImageBytes {
		return 0, 0, false
	}
	return decodeImageSize(data)
}

// resizeImage mirrors Python _resize (chat_models/base.py:4036): the larger
// side is capped at 2048, then the smaller side at 768, preserving aspect
// ratio with integer division.
func resizeImage(width, height int) (int, int) {
	if width > 2048 || height > 2048 {
		if width > height {
			height = height * 2048 / width
			width = 2048
		} else {
			width = width * 2048 / height
			height = 2048
		}
	}
	if width > 768 && height > 768 {
		if width > height {
			width = width * 768 / height
			height = 768
		} else {
			height = height * 768 / width
			width = 768
		}
	}
	return width, height
}

// countImageTokens mirrors Python _count_image_tokens
// (chat_models/base.py:4012): resize, then 170 * ceil(h/512) * ceil(w/512) + 85.
func countImageTokens(width, height int) int {
	width, height = resizeImage(width, height)
	h := (height + 511) / 512
	w := (width + 511) / 512
	return 170*h*w + 85
}

// openAIWireRole maps a message role to the string Python's message dict
// carries ("system"/"user"/"assistant"/"tool"), which is what gets counted.
func openAIWireRole(role messages.Role) string {
	switch role {
	case messages.RoleSystem:
		return "system"
	case messages.RoleHuman:
		return "user"
	case messages.RoleAI:
		return "assistant"
	case messages.RoleTool:
		return "tool"
	default:
		return string(role)
	}
}
