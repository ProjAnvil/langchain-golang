package openaicompat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// TestGroqWireChatCompletions samples the full Chat Completions payload shape
// for one provider (groq): the resolved model must hit {GROQ_API_BASE}/chat/
// completions with an OpenAI CC body (model + messages[{role,content}] and no
// Responses-only fields like input/instructions) and bearer auth from
// GROQ_API_KEY. The other providers share the same request builder, so this
// stands in for the wire shape of all seven.
func TestGroqWireChatCompletions(t *testing.T) {
	var gotPath string
	var gotAuth string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-groq-1",
			"model":"llama-3.3-70b-versatile",
			"choices":[{"message":{"role":"assistant","content":"groq says hi"}}],
			"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
		}`))
	}))
	defer server.Close()
	t.Setenv("GROQ_API_KEY", "groq-test-key")
	t.Setenv("GROQ_API_BASE", server.URL)

	spec, err := chatmodels.ParseModelString("groq:llama-3.3-70b-versatile")
	if err != nil {
		t.Fatalf("ParseModelString: %v", err)
	}
	model, err := chatmodels.Resolve(spec)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	resp, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hello")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	if gotPath != "/chat/completions" {
		t.Fatalf("path = %q want /chat/completions", gotPath)
	}
	if gotAuth != "Bearer groq-test-key" {
		t.Fatalf("Authorization = %q want Bearer groq-test-key", gotAuth)
	}
	if gotBody["model"] != "llama-3.3-70b-versatile" {
		t.Fatalf("body model = %v", gotBody["model"])
	}
	msgs, ok := gotBody["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("body messages = %v want 1 entry", gotBody["messages"])
	}
	first, ok := msgs[0].(map[string]any)
	if !ok {
		t.Fatalf("messages[0] = %T want object", msgs[0])
	}
	if first["role"] != "user" || first["content"] != "hello" {
		t.Fatalf("messages[0] = %v want {role:user content:hello}", first)
	}
	// Chat Completions shape only: the Responses API fields must be absent.
	for _, forbidden := range []string{"input", "instructions", "max_output_tokens"} {
		if _, present := gotBody[forbidden]; present {
			t.Fatalf("body must not carry Responses-API field %q: %v", forbidden, gotBody[forbidden])
		}
	}
	if resp.Content != "groq says hi" {
		t.Fatalf("content = %q", resp.Content)
	}
}

// TestOpenRouterAttributionHeaders covers the one provider with
// provider-specific default headers: OpenRouter's app-attribution headers
// (HTTP-Referer from OPENROUTER_APP_URL, X-Title from OPENROUTER_APP_TITLE)
// mirror langchain_openrouter's defaults (https://docs.langchain.com and
// "LangChain") and are env-overridable.
func TestOpenRouterAttributionHeaders(t *testing.T) {
	newServer := func(capture *map[string]any, headers *http.Header) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			*capture = map[string]any{}
			_ = json.NewDecoder(r.Body).Decode(capture)
			*headers = r.Header.Clone()
			_, _ = w.Write([]byte(`{
				"id":"gen-or-1",
				"model":"anthropic/claude-sonnet-4-5",
				"choices":[{"message":{"role":"assistant","content":"or ok"}}],
				"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}
			}`))
		}))
	}

	invokeResolved := func(t *testing.T, model string) (map[string]any, http.Header) {
		t.Helper()
		var body map[string]any
		var hdrs http.Header
		server := newServer(&body, &hdrs)
		defer server.Close()
		t.Setenv("OPENROUTER_API_KEY", "or-test-key")
		t.Setenv("OPENROUTER_API_BASE", server.URL)

		resolved, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
			Provider: "openrouter",
			Model:    model,
		})
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		if _, err := resolved.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
			t.Fatalf("Invoke: %v", err)
		}
		return body, hdrs
	}

	t.Run("default attribution headers", func(t *testing.T) {
		t.Setenv("OPENROUTER_APP_URL", "")
		t.Setenv("OPENROUTER_APP_TITLE", "")
		body, hdrs := invokeResolved(t, "anthropic/claude-sonnet-4-5")
		if got := hdrs.Get("HTTP-Referer"); got != "https://docs.langchain.com" {
			t.Fatalf("HTTP-Referer = %q want https://docs.langchain.com", got)
		}
		if got := hdrs.Get("X-Title"); got != "LangChain" {
			t.Fatalf("X-Title = %q want LangChain", got)
		}
		if got := hdrs.Get("Authorization"); got != "Bearer or-test-key" {
			t.Fatalf("Authorization = %q want Bearer or-test-key", got)
		}
		if body["model"] != "anthropic/claude-sonnet-4-5" {
			t.Fatalf("body model = %v", body["model"])
		}
	})

	t.Run("env overrides attribution headers", func(t *testing.T) {
		t.Setenv("OPENROUTER_APP_URL", "https://myapp.example")
		t.Setenv("OPENROUTER_APP_TITLE", "My App")
		_, hdrs := invokeResolved(t, "anthropic/claude-sonnet-4-5")
		if got := hdrs.Get("HTTP-Referer"); got != "https://myapp.example" {
			t.Fatalf("HTTP-Referer = %q want https://myapp.example", got)
		}
		if got := hdrs.Get("X-Title"); got != "My App" {
			t.Fatalf("X-Title = %q want My App", got)
		}
	})
}
