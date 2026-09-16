package openaicompat

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// The compat factory builds partners/openai ChatModel (providers.go), so
// ParallelToolCalls must flow through to the Chat Completions payload with
// zero openaicompat changes.

func TestCompatInheritsParallelToolCalls(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-1","model":"llama-test",
			"choices":[{"message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()
	t.Setenv("GROQ_API_KEY", "test-key")
	t.Setenv("GROQ_API_BASE", server.URL)

	spec, err := chatmodels.ParseModelString("groq:llama-test")
	if err != nil {
		t.Fatalf("ParseModelString: %v", err)
	}
	model, err := chatmodels.Resolve(spec)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	tool, err := coretools.FromFunc("GetWeather", "Get weather.", func(ctx context.Context, args struct{ City string }) (coretools.Result, error) {
		return coretools.Result{Content: args.City}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	binder, ok := model.(language.ToolBinder)
	if !ok {
		t.Fatal("resolved model does not implement language.ToolBinder")
	}
	no := false
	bound, err := binder.BindToolsWithOptions([]coretools.Tool{tool}, language.BindToolsOptions{ParallelToolCalls: &no})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}
	if _, err := bound.Invoke(t.Context(), []messages.Message{messages.Human("hello")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if gotBody["parallel_tool_calls"] != false {
		t.Fatalf("parallel_tool_calls = %v, want false", gotBody["parallel_tool_calls"])
	}
}
