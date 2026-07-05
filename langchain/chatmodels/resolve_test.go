package chatmodels

import (
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
)

// TestParseModelString covers the strict 'provider:model' parser. It is
// intentionally distinct from ParseModel: ParseModelString does no inference
// and no validation against BuiltinProviders or the factory registry.
func TestParseModelString(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		wantSpec   ChatModelSpec
		wantErr    bool
		errSubstr  string
	}{
		{
			name:     "simple provider:model",
			input:    "openai:gpt-4o-mini",
			wantSpec: ChatModelSpec{Provider: "openai", Model: "gpt-4o-mini"},
		},
		{
			name:     "mixed-case provider normalizes to lowercase",
			input:    "OpenAI:gpt-4o",
			wantSpec: ChatModelSpec{Provider: "openai", Model: "gpt-4o"},
		},
		{
			name:     "dashed provider normalizes dash to underscore",
			input:    "Azure-OpenAI:gpt-4",
			wantSpec: ChatModelSpec{Provider: "azure_openai", Model: "gpt-4"},
		},
		{
			name:     "colon in model half preserved (split on first colon only)",
			input:    "openai:ft:gpt-4o",
			wantSpec: ChatModelSpec{Provider: "openai", Model: "ft:gpt-4o"},
		},
		{
			name:      "no colon rejected",
			input:     "gpt-4o-mini",
			wantErr:   true,
			errSubstr: "must be in 'provider:model' form",
		},
		{
			name:      "empty model half rejected",
			input:     "openai:",
			wantErr:   true,
			errSubstr: "must be in 'provider:model' form",
		},
		{
			name:      "empty provider half rejected",
			input:     ":gpt-4o-mini",
			wantErr:   true,
			errSubstr: "must be in 'provider:model' form",
		},
		{
			name:      "empty string rejected",
			input:     "",
			wantErr:   true,
			errSubstr: "must be in 'provider:model' form",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec, err := ParseModelString(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil with spec=%#v", spec)
				}
				if tt.errSubstr != "" && !strings.Contains(err.Error(), tt.errSubstr) {
					t.Fatalf("error %q does not contain %q", err.Error(), tt.errSubstr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if spec != tt.wantSpec {
				t.Fatalf("spec mismatch: got %+v want %+v", spec, tt.wantSpec)
			}
		})
	}
}

// TestResolve_RegisteredFactory registers a fake factory under a unique,
// non-real provider name (so it cannot collide with Task 5.6's real openai
// registration), then asserts Resolve returns the factory's product, that the
// model string is passed through, and that opts is nil.
func TestResolve_RegisteredFactory(t *testing.T) {
	const provider = "test-fake-provider"

	var capturedModel string
	var capturedOpts map[string]any

	wantModel := language.NewFakeChatModel()

	RegisterProvider(provider, func(model string, opts map[string]any) (language.ChatModel, error) {
		capturedModel = model
		capturedOpts = opts
		return wantModel, nil
	})

	got, err := Resolve(ChatModelSpec{Provider: provider, Model: "passed-through-model"})
	if err != nil {
		t.Fatalf("Resolve returned unexpected error: %v", err)
	}
	if got != wantModel {
		t.Fatalf("Resolve returned a different ChatModel than the factory's product")
	}
	if capturedModel != "passed-through-model" {
		t.Fatalf("factory received model %q, want %q", capturedModel, "passed-through-model")
	}
	if capturedOpts != nil {
		t.Fatalf("factory received opts=%#v, want nil", capturedOpts)
	}
}

// TestResolve_NormalizesProvider verifies Resolve normalizes the provider name
// before lookup (so "Test-Fake-Provider" hits the registration made above).
func TestResolve_NormalizesProvider(t *testing.T) {
	const provider = "test-fake-provider-normalized"

	RegisterProvider(provider, func(model string, opts map[string]any) (language.ChatModel, error) {
		return language.NewFakeChatModel(), nil
	})

	// Mixed case + dash should normalize to the same key.
	if _, err := Resolve(ChatModelSpec{Provider: "Test-Fake-Provider-Normalized", Model: "m"}); err != nil {
		t.Fatalf("Resolve with denormalized provider name failed: %v", err)
	}
}

// TestResolve_UnknownProvider asserts that an unregistered provider surfaces a
// typed *UnknownProviderError so callers can errors.As it.
func TestResolve_UnknownProvider(t *testing.T) {
	_, err := Resolve(ChatModelSpec{Provider: "definitely-not-registered-xyz", Model: "x"})
	if err == nil {
		t.Fatal("expected error for unknown provider, got nil")
	}

	var unknownErr *UnknownProviderError
	if !errors.As(err, &unknownErr) {
		t.Fatalf("expected *UnknownProviderError, got %T: %v", err, err)
	}
	if unknownErr.Provider != "definitely-not-registered-xyz" {
		t.Fatalf("UnknownProviderError.Provider = %q, want %q",
			unknownErr.Provider, "definitely-not-registered-xyz")
	}
}
