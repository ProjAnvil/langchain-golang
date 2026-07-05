package middleware

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestModelRequestOverride_MissingKeys(t *testing.T) {
	base := ModelRequest{
		Model:          "m",
		Messages:       []messages.Message{messages.Human("hi")},
		ToolChoice:     "auto",
		Tools:          []any{},
		ResponseFormat: "rf",
		State:          map[string]any{"k": "v"},
		ModelSettings:  map[string]any{"temp": 0.5},
	}

	next, err := base.Override(
		WithToolChoice(map[string]any{"type": "any"}),
		WithResponseFormat("rf2"),
		WithModelSettings(map[string]any{"temp": 0.9}),
		WithState(map[string]any{"k2": "v2"}),
	)
	if err != nil {
		t.Fatalf("Override: %v", err)
	}
	if tc, _ := next.ToolChoice.(map[string]any); tc["type"] != "any" {
		t.Fatalf("ToolChoice not overridden: %#v", next.ToolChoice)
	}
	if next.ResponseFormat != "rf2" {
		t.Fatalf("ResponseFormat not overridden: %#v", next.ResponseFormat)
	}
	if next.ModelSettings["temp"] != 0.9 {
		t.Fatalf("ModelSettings not overridden: %#v", next.ModelSettings)
	}
	if next.State["k2"] != "v2" {
		t.Fatalf("State not overridden: %#v", next.State)
	}
	// Overriding State/ModelSettings must not mutate the original request.
	if _, stillThere := base.State["k"]; !stillThere {
		t.Fatalf("Override mutated original State: %#v", base.State)
	}
}
