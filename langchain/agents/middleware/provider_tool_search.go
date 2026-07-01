package middleware

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

var ServerToolSearchTools = map[string]map[string]any{
	"anthropic": {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
	"openai":    {"type": "tool_search"},
}

type DeferredTool struct {
	Tool   tools.Tool
	Extras map[string]any
}

func NewDeferredTool(tool tools.Tool) DeferredTool {
	return DeferredTool{Tool: tool, Extras: map[string]any{"defer_loading": true}}
}

func (t DeferredTool) Name() string {
	return t.Tool.Name()
}

func (t DeferredTool) Description() string {
	return t.Tool.Description()
}

func (t DeferredTool) ArgsSchema() schema.Schema {
	return t.Tool.ArgsSchema()
}

func (t DeferredTool) Invoke(ctx context.Context, input map[string]any) (tools.Result, error) {
	return t.Tool.Invoke(ctx, input)
}

func (t DeferredTool) DeferLoading() bool {
	value, _ := t.Extras["defer_loading"].(bool)
	return value
}

type ProviderToolSearchMiddleware struct {
	SearchableToolNames map[string]bool
}

func NewProviderToolSearchMiddleware(searchableTools ...string) *ProviderToolSearchMiddleware {
	names := map[string]bool{}
	for _, name := range searchableTools {
		names[name] = true
	}
	return &ProviderToolSearchMiddleware{SearchableToolNames: names}
}

func (m *ProviderToolSearchMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	next, err := m.prepareRequest(request)
	if err != nil {
		return ModelResponse{}, err
	}
	return handler(ctx, next)
}

func (m *ProviderToolSearchMiddleware) prepareRequest(request ModelRequest) (ModelRequest, error) {
	if len(m.SearchableToolNames) > 0 {
		available := map[string]bool{}
		for _, entry := range request.Tools {
			if tool, ok := asTool(entry); ok {
				available[tool.Name()] = true
			}
		}
		unknown := []string{}
		for name := range m.SearchableToolNames {
			if !available[name] {
				unknown = append(unknown, name)
			}
		}
		if len(unknown) > 0 {
			sort.Strings(unknown)
			return ModelRequest{}, fmt.Errorf("ProviderToolSearchMiddleware: searchable_tools references tool(s) not bound to the model: %s", strings.Join(unknown, ", "))
		}
	}

	needsSearch := false
	for _, entry := range request.Tools {
		if m.isDeferredTool(entry) {
			needsSearch = true
			break
		}
	}
	if !needsSearch {
		return request, nil
	}

	provider := InferProvider(request.Model, request.Runtime)
	if provider == "" {
		return ModelRequest{}, fmt.Errorf("ProviderToolSearchMiddleware could not determine the provider for model %T; server-side tool search supports: anthropic, openai", request.Model)
	}
	spec, ok := ServerToolSearchTools[provider]
	if !ok {
		return ModelRequest{}, fmt.Errorf("ProviderToolSearchMiddleware requires a provider with server-side tool search, but got %q; supported providers: anthropic, openai", provider)
	}

	nextTools := make([]any, 0, len(request.Tools)+1)
	for _, entry := range request.Tools {
		if m.isDeferredTool(entry) {
			tool, _ := asTool(entry)
			nextTools = append(nextTools, NewDeferredTool(tool))
			continue
		}
		nextTools = append(nextTools, entry)
	}
	nextTools = append(nextTools, cloneAnyMap(spec))
	return request.Override(WithTools(nextTools))
}

func (m *ProviderToolSearchMiddleware) isDeferredTool(entry any) bool {
	if deferred, ok := entry.(DeferredTool); ok && deferred.DeferLoading() {
		return true
	}
	tool, ok := asTool(entry)
	if !ok {
		return false
	}
	return m.SearchableToolNames[tool.Name()]
}

func asTool(entry any) (tools.Tool, bool) {
	switch typed := entry.(type) {
	case DeferredTool:
		return typed.Tool, true
	case tools.Tool:
		return typed, true
	default:
		return nil, false
	}
}

func InferProvider(model any, runtime any) string {
	if provider := providerFromValue(model); provider != "" {
		return provider
	}
	if provider := providerFromValue(runtime); provider != "" {
		return provider
	}
	return ""
}

func providerFromValue(value any) string {
	if value == nil {
		return ""
	}
	if text, ok := value.(string); ok {
		return providerFromString(text)
	}
	if config, ok := value.(map[string]any); ok {
		if provider, ok := config["model_provider"].(string); ok {
			return normalizeProvider(provider)
		}
		if model, ok := config["model"].(string); ok {
			return providerFromString(model)
		}
		if provider, ok := config["ls_provider"].(string); ok {
			return normalizeProvider(provider)
		}
	}
	typeName := reflect.TypeOf(value).String()
	return providerFromClassName(typeName)
}

func providerFromString(value string) string {
	if before, after, ok := strings.Cut(value, ":"); ok && after != "" {
		return normalizeProvider(before)
	}
	lower := strings.ToLower(value)
	switch {
	case strings.Contains(lower, "claude"), strings.Contains(lower, "anthropic"):
		return "anthropic"
	case strings.Contains(lower, "gpt"), strings.Contains(lower, "openai"):
		return "openai"
	default:
		return ""
	}
}

func providerFromClassName(typeName string) string {
	switch {
	case strings.Contains(typeName, "Anthropic"):
		return "anthropic"
	case strings.Contains(typeName, "OpenAI"):
		return "openai"
	default:
		return ""
	}
}

func normalizeProvider(provider string) string {
	return strings.ToLower(strings.ReplaceAll(provider, "-", "_"))
}
