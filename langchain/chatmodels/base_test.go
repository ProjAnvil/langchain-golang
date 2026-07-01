package chatmodels

import (
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestBuiltinProvidersSorted(t *testing.T) {
	names := BuiltinProviderNames()
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	if !reflect.DeepEqual(names, sorted) {
		t.Fatalf("providers are not sorted: got %#v want %#v", names, sorted)
	}
}

func TestAttemptInferModelProvider(t *testing.T) {
	tests := []struct {
		modelName        string
		expectedProvider string
	}{
		{"gpt-5.5", "openai"},
		{"o3", "openai"},
		{"text-davinci-003", "openai"},
		{"claude-3-haiku-20240307", "anthropic"},
		{"command-r-plus", "cohere"},
		{"accounts/fireworks/models/mixtral-8x7b-instruct", "fireworks"},
		{"Accounts/Fireworks/models/mixtral-8x7b-instruct", "fireworks"},
		{"gemini-1.5-pro", "google_vertexai"},
		{"gemini-2.5-pro", "google_vertexai"},
		{"gemini-3.1-pro-preview", "google_vertexai"},
		{"amazon.titan-text-express-v1", "bedrock"},
		{"Amazon.Titan-Text-Express-v1", "bedrock"},
		{"anthropic.claude-v2", "bedrock"},
		{"Anthropic.Claude-V2", "bedrock"},
		{"mistral-small", "mistralai"},
		{"mixtral-8x7b", "mistralai"},
		{"deepseek-v3", "deepseek"},
		{"grok-beta", "xai"},
		{"sonar-small", "perplexity"},
		{"solar-pro", "upstage"},
	}

	for _, tt := range tests {
		t.Run(tt.modelName, func(t *testing.T) {
			got := attemptInferModelProvider(tt.modelName)
			if got != tt.expectedProvider {
				t.Fatalf("provider mismatch: got %q want %q", got, tt.expectedProvider)
			}
		})
	}
}

func TestInferModelProvider(t *testing.T) {
	if got := InferModelProvider("gpt-5.5"); got != "openai" {
		t.Fatalf("provider mismatch: got %q want openai", got)
	}
}

func TestAttemptInferModelProviderUnknown(t *testing.T) {
	if got := attemptInferModelProvider("unknown-model"); got != "" {
		t.Fatalf("expected no provider, got %q", got)
	}
}

func TestProviderInfoFor(t *testing.T) {
	info, ok := ProviderInfoFor("azure-openai")
	if !ok {
		t.Fatal("expected azure-openai to be supported")
	}
	if info.Package != "langchain_openai" || info.Class != "AzureChatOpenAI" {
		t.Fatalf("unexpected provider info: %#v", info)
	}
	if !SupportsProvider("Azure-OpenAI") {
		t.Fatal("expected mixed-case hyphenated provider to be supported")
	}
	if SupportsProvider("unknown") {
		t.Fatal("expected unknown provider to be unsupported")
	}
}

func TestParseModel(t *testing.T) {
	tests := []struct {
		name             string
		model            string
		modelProvider    string
		expectedModel    string
		expectedProvider string
	}{
		{
			name:             "provider prefix",
			model:            "openai:gpt-5.5",
			expectedModel:    "gpt-5.5",
			expectedProvider: "openai",
		},
		{
			name:             "provider prefix normalized",
			model:            "azure-openai:gpt-5.5",
			expectedModel:    "gpt-5.5",
			expectedProvider: "azure_openai",
		},
		{
			name:             "explicit provider",
			model:            "gpt-5.5",
			modelProvider:    "openai",
			expectedModel:    "gpt-5.5",
			expectedProvider: "openai",
		},
		{
			name:             "provider normalized",
			model:            "gpt-5.5",
			modelProvider:    "azure-openai",
			expectedModel:    "gpt-5.5",
			expectedProvider: "azure_openai",
		},
		{
			name:             "inferred provider",
			model:            "claude-3-haiku-20240307",
			expectedModel:    "claude-3-haiku-20240307",
			expectedProvider: "anthropic",
		},
		{
			name:             "colon in model after provider prefix",
			model:            "openai:ft:gpt-5.5",
			expectedModel:    "ft:gpt-5.5",
			expectedProvider: "openai",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, provider, err := parseModel(tt.model, tt.modelProvider)
			if err != nil {
				t.Fatalf("parse model: %v", err)
			}
			if model != tt.expectedModel || provider != tt.expectedProvider {
				t.Fatalf("got (%q, %q), want (%q, %q)", model, provider, tt.expectedModel, tt.expectedProvider)
			}
		})
	}
}

func TestParseModelPublic(t *testing.T) {
	spec, err := ParseModel("azure-openai:gpt-5.5")
	if err != nil {
		t.Fatalf("parse model: %v", err)
	}
	if spec.Model != "gpt-5.5" || spec.Provider != "azure_openai" {
		t.Fatalf("unexpected spec: %#v", spec)
	}
}

func TestParseModelErrorsWhenProviderCannotBeInferred(t *testing.T) {
	_, _, err := parseModel("unknown-model", "")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "Unable to infer model provider") {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, provider := range BuiltinProviderNames() {
		if !strings.Contains(err.Error(), provider) {
			t.Fatalf("error %q missing supported provider %q", err.Error(), provider)
		}
	}
}

func TestInitChatModelRejectsModelObject(t *testing.T) {
	_, err := InitChatModel(123)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "must be a string") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestInitChatModelUnknownProvider(t *testing.T) {
	_, err := InitChatModel("foo", WithModelProvider("bar"))
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "Unsupported provider='bar'") {
		t.Fatalf("unexpected error: %v", err)
	}
}
