package gemini

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/structuredoutput"
)

// jokeSchema mirrors the standardtests Joke fixture.
func jokeSchema() schema.Schema {
	sch := schema.Object(map[string]schema.Schema{
		"setup":     schema.String("question to set up a joke"),
		"punchline": schema.String("answer to resolve a joke"),
	}, "setup", "punchline")
	sch["title"] = "Joke"
	sch["description"] = "Joke to tell user."
	return sch
}

// TestInvokeStructuredResponseJsonSchema mirrors Python
// with_structured_output(method="json_schema") (chat_models.py:2545-2581,
// 3755-3770): the request carries generationConfig.responseMimeType
// "application/json" plus responseJsonSchema with the standard JSON schema
// passed through, and the returned message text is the model's JSON.
func TestInvokeStructuredResponseJsonSchema(t *testing.T) {
	var generation map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		generation, _ = body["generationConfig"].(map[string]any)
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[
				{"text":"{\"setup\":\"Why go?\",\"punchline\":\"No NULLs.\"}"}
			]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":6,"candidatesTokenCount":9,"totalTokenCount":15}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	response, err := model.InvokeStructured(t.Context(), []messages.Message{
		messages.Human("Tell me a joke."),
	}, jokeSchema())
	if err != nil {
		t.Fatalf("InvokeStructured: %v", err)
	}

	if generation == nil {
		t.Fatalf("generationConfig missing")
	}
	if generation["responseMimeType"] != "application/json" {
		t.Errorf("responseMimeType: %v", generation["responseMimeType"])
	}
	jsonSchema, ok := generation["responseJsonSchema"].(map[string]any)
	if !ok {
		t.Fatalf("responseJsonSchema: %v", generation["responseJsonSchema"])
	}
	// The standard JSON schema passes through untouched (lowercase types,
	// required list, title) — Python's response_json_schema behavior.
	if jsonSchema["type"] != "object" {
		t.Errorf("responseJsonSchema type: %v", jsonSchema["type"])
	}
	if required, _ := jsonSchema["required"].([]any); len(required) != 2 {
		t.Errorf("responseJsonSchema required: %v", jsonSchema["required"])
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(response.Content), &parsed); err != nil {
		t.Fatalf("structured content not JSON: %v (%q)", err, response.Content)
	}
	if parsed["setup"] != "Why go?" || parsed["punchline"] != "No NULLs." {
		t.Errorf("structured content: %v", parsed)
	}
	if response.UsageMetadata.TotalTokens != 15 {
		t.Errorf("usage passthrough: %+v", response.UsageMetadata)
	}
}

// TestStructuredOutputBindingComposable verifies the model satisfies the
// structuredoutput.BindJSON generic (JSONSchemaModel), the typed counterpart
// of Python's with_structured_output.
func TestStructuredOutputBindingComposable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[
			{"text":"{\"setup\":\"a\",\"punchline\":\"b\"}"}
		]}}]}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	runnable, err := structuredoutput.BindJSON[ChatModel, modelT](
		model,
		"Joke",
		jokeSchema(),
		true,
	)
	if err != nil {
		t.Fatalf("BindJSON: %v", err)
	}
	result, err := runnable.Invoke(t.Context(), []messages.Message{messages.Human("joke")})
	if err != nil {
		t.Fatalf("BindJSON invoke: %v", err)
	}
	if result.Setup != "a" || result.Punchline != "b" {
		t.Errorf("parsed result: %+v", result)
	}
	if !strings.HasPrefix(model.LLMType(), "chat-google") {
		t.Errorf("llm type: %q", model.LLMType())
	}
}

type modelT struct {
	Setup     string `json:"setup"`
	Punchline string `json:"punchline"`
}
