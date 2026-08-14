package lcerrors_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/outputparser"
)

// TestCreateMessageMatchesPythonFormat locks CreateMessage to Python's
// create_message output, including the trailing space and the error-code path.
func TestCreateMessageMatchesPythonFormat(t *testing.T) {
	got := lcerrors.CreateMessage("Failed to parse output", lcerrors.ErrorCodeOutputParsingFailure)
	want := "Failed to parse output\n" +
		"For troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE "
	if got != want {
		t.Fatalf("CreateMessage() = %q, want %q", got, want)
	}
}

// TestOutputParserExceptionSatisfiesErrorsAPI verifies that OutputParserException
// behaves like a proper wrapped error: errors.As/errors.Is/Unwrap all work, and
// the plain-message constructor appends the troubleshooting line like Python.
func TestOutputParserExceptionSatisfiesErrorsAPI(t *testing.T) {
	root := errors.New("boom")

	ope := lcerrors.NewOutputParserExceptionFromError(root)
	if ope == nil {
		t.Fatal("NewOutputParserExceptionFromError(root) returned nil")
	}

	// Unwrap must return the wrapped error directly.
	if ope.Unwrap() != root {
		t.Fatalf("Unwrap() = %v, want %v", ope.Unwrap(), root)
	}
	// errors.Is must reach the wrapped error through Unwrap.
	if !errors.Is(ope, root) {
		t.Fatal("errors.Is(ope, root) = false, want true")
	}
	// errors.As must extract *OutputParserException.
	var target *lcerrors.OutputParserException
	if !errors.As(ope, &target) {
		t.Fatal("errors.As(ope, &target) = false, want true")
	}
	if target.Err != root {
		t.Fatalf("target.Err = %v, want %v", target.Err, root)
	}

	// Both behaviors must survive an additional wrapping layer.
	wrapped := fmt.Errorf("context: %w", ope)
	var target2 *lcerrors.OutputParserException
	if !errors.As(wrapped, &target2) {
		t.Fatal("errors.As(wrapped, &target2) = false, want true")
	}
	if !errors.Is(wrapped, root) {
		t.Fatal("errors.Is(wrapped, root) = false, want true")
	}

	// The plain-message constructor mirrors Python's str path by appending the
	// OUTPUT_PARSING_FAILURE troubleshooting link.
	msg := lcerrors.NewOutputParserException("parse failed").Error()
	if !strings.Contains(msg, "For troubleshooting, visit:") {
		t.Fatalf("NewOutputParserException message missing troubleshooting line: %q", msg)
	}
	if !strings.Contains(msg, "OUTPUT_PARSING_FAILURE") {
		t.Fatalf("NewOutputParserException message missing error code: %q", msg)
	}
}

// TestJSONParserReturnsOutputParserException ensures that a JSON parse failure
// surfaces as *lcerrors.OutputParserException carrying the raw model output and
// the underlying parse error.
func TestJSONParserReturnsOutputParserException(t *testing.T) {
	parser := outputparser.NewJSONParser[map[string]any]("")
	_, err := parser.Parse(context.Background(), "{bad json}")
	if err == nil {
		t.Fatal("expected parse error")
	}

	var target *lcerrors.OutputParserException
	if !errors.As(err, &target) {
		t.Fatalf("errors.As into *lcerrors.OutputParserException = false, got %T: %v", err, err)
	}
	if target.LLMOutput != "{bad json}" {
		t.Fatalf("LLMOutput = %q, want %q", target.LLMOutput, "{bad json}")
	}
	if target.Err == nil {
		t.Fatal("Err should be set to the underlying parse error")
	}
}
