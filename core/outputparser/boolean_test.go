package outputparser

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
)

func TestBooleanOutputParserParse(t *testing.T) {
	parser := NewBooleanOutputParser()
	cases := []struct {
		name string
		text string
		want bool
	}{
		{name: "uppercase yes", text: "YES", want: true},
		{name: "lowercase yes", text: "yes", want: true},
		{name: "mixed case yes in sentence", text: "Yes, this is relevant.", want: true},
		{name: "uppercase no", text: "NO", want: false},
		{name: "lowercase no", text: "no", want: false},
		{name: "no in sentence", text: "I think NO it is not relevant", want: false},
		{name: "no with leading text", text: "definitely NO", want: false},
		{name: "yes with trailing punctuation", text: "The answer is YES.", want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parser.Parse(t.Context(), tc.text)
			if err != nil {
				t.Fatalf("Parse(%q) unexpected error: %v", tc.text, err)
			}
			if got != tc.want {
				t.Fatalf("Parse(%q) = %v, want %v", tc.text, got, tc.want)
			}
		})
	}
}

func TestBooleanOutputParserInvalidInput(t *testing.T) {
	parser := NewBooleanOutputParser()
	cases := []struct {
		name string
		text string
	}{
		{name: "empty string", text: ""},
		{name: "unrelated text", text: "maybe"},
		{name: "ambiguous both tokens", text: "YES and NO"},
		{name: "ambiguous mixed case", text: "well, yes but also no"},
		{name: "token embedded in word is not a match", text: "NO_OUTPUT"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parser.Parse(t.Context(), tc.text)
			if err == nil {
				t.Fatalf("Parse(%q) expected an error", tc.text)
			}
			var ope *lcerrors.OutputParserException
			if !errors.As(err, &ope) {
				t.Fatalf("Parse(%q) error %T does not wrap *lcerrors.OutputParserException", tc.text, err)
			}
		})
	}
}

func TestBooleanOutputParserZeroValueAndCustomTokens(t *testing.T) {
	// Zero value falls back to YES/NO defaults.
	zero := BooleanOutputParser{}
	if got, err := zero.Parse(t.Context(), "YES"); err != nil || !got {
		t.Fatalf("zero-value Parse(YES) = %v, %v; want true, nil", got, err)
	}

	custom := BooleanOutputParser{TrueVal: "YAY", FalseVal: "NAY"}
	if got, err := custom.Parse(t.Context(), "the verdict is yay"); err != nil || !got {
		t.Fatalf("custom Parse = %v, %v; want true, nil", got, err)
	}
	if got, err := custom.Parse(t.Context(), "NAY"); err != nil || got {
		t.Fatalf("custom Parse = %v, %v; want false, nil", got, err)
	}
	if got := custom.FormatInstructions(); got != "Your response should be either YAY or NAY." {
		t.Fatalf("FormatInstructions() = %q", got)
	}
}

func TestBooleanOutputParserCanceledContext(t *testing.T) {
	parser := NewBooleanOutputParser()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := parser.Parse(ctx, "YES"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Parse with canceled context = %v, want context.Canceled", err)
	}
}

func TestNoOutputParserParse(t *testing.T) {
	parser := NewNoOutputParser()
	cases := []struct {
		name string
		text string
		want string
	}{
		{name: "no-output sentinel", text: "NO_OUTPUT", want: ""},
		{name: "no-output sentinel with whitespace", text: "  NO_OUTPUT \n", want: ""},
		{name: "real content is trimmed", text: "  relevant snippet  ", want: "relevant snippet"},
		{name: "sentinel embedded is kept", text: "NO_OUTPUT is a token", want: "NO_OUTPUT is a token"},
		{name: "lowercase sentinel is kept", text: "no_output", want: "no_output"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parser.Parse(t.Context(), tc.text)
			if err != nil {
				t.Fatalf("Parse(%q) unexpected error: %v", tc.text, err)
			}
			if got != tc.want {
				t.Fatalf("Parse(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestNoOutputParserZeroValueAndCustomMarker(t *testing.T) {
	zero := NoOutputParser{}
	if got, err := zero.Parse(t.Context(), "NO_OUTPUT"); err != nil || got != "" {
		t.Fatalf("zero-value Parse = %q, %v; want empty, nil", got, err)
	}
	custom := NoOutputParser{NoOutputStr: "NONE"}
	if got, err := custom.Parse(t.Context(), "NONE"); err != nil || got != "" {
		t.Fatalf("custom Parse = %q, %v; want empty, nil", got, err)
	}
	if got, err := custom.Parse(t.Context(), "NO_OUTPUT"); err != nil || got != "NO_OUTPUT" {
		t.Fatalf("custom Parse(NO_OUTPUT) = %q, %v; want NO_OUTPUT, nil", got, err)
	}
	if got := custom.FormatInstructions(); got != "Return the extracted text, or NONE if nothing is relevant." {
		t.Fatalf("FormatInstructions() = %q", got)
	}
}

func TestNoOutputParserCanceledContext(t *testing.T) {
	parser := NewNoOutputParser()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := parser.Parse(ctx, "anything"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Parse with canceled context = %v, want context.Canceled", err)
	}
}
