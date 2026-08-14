package middleware

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

func cacheControlMarker() map[string]any {
	return map[string]any{"type": "ephemeral"}
}

// contentBlockWithCacheControl builds a text content block whose extras carry
// the Anthropic cache_control marker at the top level and nested under
// "metadata", mirroring the block shape an outer caching middleware produces.
func contentBlockWithCacheControl(text string) messages.ContentBlock {
	return messages.ParseContentBlock(map[string]any{
		"type":          "text",
		"text":          text,
		"cache_control": cacheControlMarker(),
		"metadata":      map[string]any{"cache_control": cacheControlMarker()},
	})
}

// requestWithCacheControl builds a request carrying Anthropic cache_control
// markers in model_settings, the system message, and the conversation messages.
func requestWithCacheControl(model any) (ModelRequest, error) {
	system := messages.Message{
		Role:          messages.RoleSystem,
		ContentBlocks: []messages.ContentBlock{contentBlockWithCacheControl("system prompt")},
	}
	return NewModelRequest(ModelRequest{
		Model:         model,
		SystemMessage: &system,
		ModelSettings: map[string]any{"cache_control": cacheControlMarker()},
		Messages: []messages.Message{{
			Role:          messages.RoleHuman,
			ContentBlocks: []messages.ContentBlock{contentBlockWithCacheControl("hi")},
		}},
	})
}

func blockHasCacheControl(block messages.ContentBlock) bool {
	m := messages.BlockToMap(block)
	if _, ok := m["cache_control"]; ok {
		return true
	}
	for _, nestedKey := range []string{"extras", "metadata"} {
		if nested, ok := m[nestedKey].(map[string]any); ok {
			if _, ok := nested["cache_control"]; ok {
				return true
			}
		}
	}
	return false
}

func requestBlocksHaveCacheControl(req ModelRequest) bool {
	if _, ok := req.ModelSettings["cache_control"]; ok {
		return true
	}
	if req.SystemMessage != nil {
		for _, block := range req.SystemMessage.ContentBlocks {
			if blockHasCacheControl(block) {
				return true
			}
		}
	}
	for _, msg := range req.Messages {
		for _, block := range msg.ContentBlocks {
			if blockHasCacheControl(block) {
				return true
			}
		}
	}
	return false
}

func TestModelFallbackMiddlewareStripsCacheControlForNonAnthropicFallback(t *testing.T) {
	// A non-Anthropic model (no LLMTypeProvider) must receive a sanitized request.
	fallback := NewModelFallbackMiddleware(language.NewFakeChatModel())
	request, err := requestWithCacheControl("primary")
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	var fallbackReq ModelRequest
	calls := 0
	_, err = fallback.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if calls == 0 {
			calls++
			return ModelResponse{}, errors.New("primary failed")
		}
		fallbackReq = req
		return ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}

	if requestBlocksHaveCacheControl(fallbackReq) {
		t.Fatalf("fallback request still carries cache_control markers: %#v", fallbackReq)
	}
}

func TestModelFallbackMiddlewareKeepsCacheControlForAnthropicFallback(t *testing.T) {
	// An Anthropic model (LLMType "anthropic-chat") keeps prompt caching intact.
	anthropic := fakeLLMTypeModel{llmType: "anthropic-chat"}
	fallback := NewModelFallbackMiddleware(anthropic)
	request, err := requestWithCacheControl("primary")
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	var fallbackReq ModelRequest
	calls := 0
	_, err = fallback.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if calls == 0 {
			calls++
			return ModelResponse{}, errors.New("primary failed")
		}
		fallbackReq = req
		return ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}

	if !requestBlocksHaveCacheControl(fallbackReq) {
		t.Fatalf("fallback request lost cache_control markers but should have kept them: %#v", fallbackReq)
	}
}

func TestModelFallbackMiddlewareResolvesStringFallbackSpec(t *testing.T) {
	const provider = "middleware-parity-test"
	wantModel := language.NewFakeChatModel()

	var capturedModel string
	chatmodels.RegisterProvider(provider, func(model string, opts map[string]any) (language.ChatModel, error) {
		capturedModel = model
		return wantModel, nil
	})

	fallback := NewModelFallbackMiddleware(provider + ":resolved-model")

	request, err := NewModelRequest(ModelRequest{Model: "primary"})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}

	var fallbackModel any
	calls := 0
	_, err = fallback.WrapModelCall(context.Background(), request, func(ctx context.Context, req ModelRequest) (ModelResponse, error) {
		if calls == 0 {
			calls++
			return ModelResponse{}, errors.New("primary failed")
		}
		fallbackModel = req.Model
		return ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil {
		t.Fatalf("wrap model call: %v", err)
	}
	if fallbackModel != wantModel {
		t.Fatalf("fallback model = %#v, want resolved model %#v", fallbackModel, wantModel)
	}
	if capturedModel != "resolved-model" {
		t.Fatalf("factory received model %q, want %q", capturedModel, "resolved-model")
	}
}
