package middleware

import (
	"context"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// ModelFallbackMiddleware retries a failed model call on a sequence of
// fallback models, mirroring Python's `ModelFallbackMiddleware`.
type ModelFallbackMiddleware struct {
	Models []any
}

func NewModelFallbackMiddleware(firstModel any, additionalModels ...any) *ModelFallbackMiddleware {
	models := make([]any, 0, 1+len(additionalModels))
	models = append(models, firstModel)
	models = append(models, additionalModels...)
	return &ModelFallbackMiddleware{Models: models}
}

func (m *ModelFallbackMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	response, err := handler(ctx, request)
	if err == nil {
		return response, nil
	}
	lastErr := err
	for _, fallbackModel := range m.Models {
		model := fallbackModel
		if spec, ok := fallbackModel.(string); ok {
			// Resolve `provider:model` string specs (Python's `init_chat_model`).
			// A string that is not in `provider:model` form is passed through
			// unchanged, preserving the pre-existing passthrough behavior for
			// opaque model markers.
			if parsed, parseErr := chatmodels.ParseModelString(spec); parseErr == nil {
				resolved, resolveErr := chatmodels.Resolve(parsed)
				if resolveErr != nil {
					return ModelResponse{}, resolveErr
				}
				model = resolved
			}
		}

		// Sanitize provider-specific Anthropic cache_control markers only when
		// the fallback model cannot accept them (Python's
		// `_supports_anthropic_cache_control` / `_sanitize_request_for_fallback`).
		req := request
		if !supportsAnthropicCacheControl(model) {
			req = sanitizeRequestForFallback(request)
		}

		next, overrideErr := req.Override(WithModel(model))
		if overrideErr != nil {
			return ModelResponse{}, overrideErr
		}
		response, err = handler(ctx, next)
		if err == nil {
			return response, nil
		}
		lastErr = err
	}
	return ModelResponse{}, lastErr
}

// supportsAnthropicCacheControl reports whether model accepts Anthropic
// `cache_control` markers, decided by provider (`LLMType`) rather than model
// name, mirroring Python's `_supports_anthropic_cache_control` which checks
// `_llm_type` against the Anthropic set.
func supportsAnthropicCacheControl(model any) bool {
	if lt, ok := model.(LLMTypeProvider); ok {
		return strings.HasPrefix(lt.LLMType(), "anthropic-chat")
	}
	return false
}

// sanitizeRequestForFallback returns a copy of request with Anthropic
// `cache_control` markers removed from model settings and message content
// blocks, mirroring Python's `_sanitize_request_for_fallback`. The input is
// not mutated.
func sanitizeRequestForFallback(request ModelRequest) ModelRequest {
	out := request
	out.ModelSettings = stripCacheControlFromSettings(request.ModelSettings)
	if request.SystemMessage != nil {
		sys := *request.SystemMessage
		sys.ContentBlocks = sanitizeContentBlocks(request.SystemMessage.ContentBlocks)
		out.SystemMessage = &sys
	}
	out.Messages = make([]messages.Message, len(request.Messages))
	for i, msg := range request.Messages {
		msg.ContentBlocks = sanitizeContentBlocks(msg.ContentBlocks)
		out.Messages[i] = msg
	}
	return out
}

func stripCacheControlFromSettings(settings map[string]any) map[string]any {
	if _, ok := settings["cache_control"]; !ok {
		return settings
	}
	out := make(map[string]any, len(settings))
	for k, v := range settings {
		if k != "cache_control" {
			out[k] = v
		}
	}
	return out
}

func sanitizeContentBlocks(blocks []messages.ContentBlock) []messages.ContentBlock {
	if len(blocks) == 0 {
		return blocks
	}
	out := make([]messages.ContentBlock, 0, len(blocks))
	for _, block := range blocks {
		out = append(out, sanitizeContentBlock(block))
	}
	return out
}

func sanitizeContentBlock(block messages.ContentBlock) messages.ContentBlock {
	m := messages.BlockToMap(block)
	if !stripCacheControl(m) {
		return block
	}
	return messages.ParseContentBlock(m)
}

// stripCacheControl removes a `cache_control` key from a content-block map,
// both at the top level and nested under `metadata`/`extras`, reporting
// whether anything changed.
func stripCacheControl(m map[string]any) bool {
	changed := false
	if _, ok := m["cache_control"]; ok {
		delete(m, "cache_control")
		changed = true
	}
	for _, key := range []string{"metadata", "extras"} {
		if nested, ok := m[key].(map[string]any); ok {
			if _, ok := nested["cache_control"]; ok {
				delete(nested, "cache_control")
				changed = true
			}
		}
	}
	return changed
}
