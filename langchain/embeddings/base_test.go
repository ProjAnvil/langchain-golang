package embeddings

import (
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestParseModelString(t *testing.T) {
	tests := []struct {
		modelString      string
		expectedProvider string
		expectedModel    string
	}{
		{"openai:text-embedding-3-small", "openai", "text-embedding-3-small"},
		{"bedrock:amazon.titan-embed-text-v1", "bedrock", "amazon.titan-embed-text-v1"},
		{"huggingface:BAAI/bge-base-en:v1.5", "huggingface", "BAAI/bge-base-en:v1.5"},
		{"google_genai:gemini-embedding-001", "google_genai", "gemini-embedding-001"},
		{"azure-openai:text-embedding-3-large", "azure_openai", "text-embedding-3-large"},
	}

	for _, tt := range tests {
		t.Run(tt.modelString, func(t *testing.T) {
			provider, model, err := parseModelString(tt.modelString)
			if err != nil {
				t.Fatalf("parse model string: %v", err)
			}
			if provider != tt.expectedProvider || model != tt.expectedModel {
				t.Fatalf("got (%q, %q), want (%q, %q)", provider, model, tt.expectedProvider, tt.expectedModel)
			}
		})
	}
}

func TestProviderInfoFor(t *testing.T) {
	info, ok := ProviderInfoFor("azure-openai")
	if !ok {
		t.Fatal("expected azure-openai to be supported")
	}
	if info.Package != "langchain_openai" || info.Class != "AzureOpenAIEmbeddings" {
		t.Fatalf("unexpected provider info: %#v", info)
	}
	if !SupportsProvider("Azure-OpenAI") {
		t.Fatal("expected mixed-case hyphenated provider to be supported")
	}
	if SupportsProvider("unknown") {
		t.Fatal("expected unknown provider to be unsupported")
	}
}

func TestParseModelStringErrors(t *testing.T) {
	tests := []struct {
		name        string
		modelString string
		want        string
	}{
		{"missing provider separator", "just-a-model-name", "Model name must be"},
		{"empty model string", "", "Invalid model format"},
		{"empty provider", ":model-name", "is not supported"},
		{"empty model", "openai:", "Model name cannot be empty"},
		{"invalid provider", "invalid-provider:model-name", "Provider 'invalid_provider' is not supported"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := parseModelString(tt.modelString)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tt.want)
			}
		})
	}
}

func TestInferModelAndProvider(t *testing.T) {
	tests := []struct {
		name             string
		model            string
		provider         string
		expectedProvider string
		expectedModel    string
	}{
		{
			name:             "model string",
			model:            "openai:text-embedding-3-small",
			expectedProvider: "openai",
			expectedModel:    "text-embedding-3-small",
		},
		{
			name:             "explicit provider",
			model:            "text-embedding-3-small",
			provider:         "openai",
			expectedProvider: "openai",
			expectedModel:    "text-embedding-3-small",
		},
		{
			name:             "explicit provider normalized",
			model:            "text-embedding-3-large",
			provider:         "azure-openai",
			expectedProvider: "azure_openai",
			expectedModel:    "text-embedding-3-large",
		},
		{
			name:             "explicit provider preserves colon in model",
			model:            "ft:text-embedding-3-small",
			provider:         "openai",
			expectedProvider: "openai",
			expectedModel:    "ft:text-embedding-3-small",
		},
		{
			name:             "model string preserves later colons",
			model:            "openai:ft:text-embedding-3-small",
			expectedProvider: "openai",
			expectedModel:    "ft:text-embedding-3-small",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, model, err := inferModelAndProvider(tt.model, tt.provider)
			if err != nil {
				t.Fatalf("infer model and provider: %v", err)
			}
			if provider != tt.expectedProvider || model != tt.expectedModel {
				t.Fatalf("got (%q, %q), want (%q, %q)", provider, model, tt.expectedProvider, tt.expectedModel)
			}
		})
	}
}

func TestParseModelPublic(t *testing.T) {
	spec, err := ParseModel("azure-openai:text-embedding-3-large")
	if err != nil {
		t.Fatalf("parse model: %v", err)
	}
	if spec.Model != "text-embedding-3-large" || spec.Provider != "azure_openai" {
		t.Fatalf("unexpected spec: %#v", spec)
	}
}

func TestInitEmbeddings(t *testing.T) {
	spec, err := InitEmbeddings("text-embedding-3-small", WithProvider("openai"))
	if err != nil {
		t.Fatalf("init embeddings: %v", err)
	}
	if spec.Model != "text-embedding-3-small" || spec.Provider != "openai" {
		t.Fatalf("unexpected spec: %#v", spec)
	}
}

func TestInferModelAndProviderErrors(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		provider string
		want     string
	}{
		{"missing provider", "text-embedding-3-small", "", "Must specify either"},
		{"empty model", "", "", "Model name cannot be empty"},
		{"empty provider with model", "model", "", "Must specify either"},
		{"invalid provider", "model", "invalid", "Provider 'invalid' is not supported"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := inferModelAndProvider(tt.model, tt.provider)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tt.want)
			}
			if tt.provider == "invalid" {
				for _, provider := range BuiltinProviderNames() {
					if !strings.Contains(err.Error(), provider) {
						t.Fatalf("error %q missing supported provider %q", err.Error(), provider)
					}
				}
			}
		})
	}
}

func TestSupportedProvidersPackageNames(t *testing.T) {
	for _, provider := range BuiltinProviderNames() {
		info := BuiltinProviders[provider]
		if strings.Contains(info.Package, "-") {
			t.Fatalf("package for %q contains dash: %q", provider, info.Package)
		}
		if !strings.HasPrefix(info.Package, "langchain_") {
			t.Fatalf("package for %q should start with langchain_: %q", provider, info.Package)
		}
		if info.Package != strings.ToLower(info.Package) {
			t.Fatalf("package for %q should be lowercase: %q", provider, info.Package)
		}
	}
}

func TestBuiltinProvidersSorted(t *testing.T) {
	names := BuiltinProviderNames()
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	if !reflect.DeepEqual(names, sorted) {
		t.Fatalf("providers are not sorted: got %#v want %#v", names, sorted)
	}
}
