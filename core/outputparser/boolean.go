package outputparser

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/projanvil/langchain-golang/core/lcerrors"
)

// Default boolean tokens mirror Python's BooleanOutputParser defaults
// (langchain_classic.output_parsers.boolean.BooleanOutputParser): the true
// token is "YES" and the false token is "NO".
const (
	defaultBooleanTrueVal  = "YES"
	defaultBooleanFalseVal = "NO"
)

// DefaultNoOutputString mirrors Python's NoOutputParser.no_output_str
// (langchain_classic.retrievers.document_compressors.chain_extract): the
// sentinel an LLM returns when none of a document's content is relevant.
const DefaultNoOutputString = "NO_OUTPUT"

// BooleanOutputParser parses the output of an LLM call into a boolean by
// looking for whole-word YES/NO tokens (case-insensitive). It mirrors Python's
// langchain_classic.output_parsers.boolean.BooleanOutputParser. When both a
// true and a false token are present, or neither is, Parse returns an error.
//
// The zero value is usable and defaults to the tokens "YES" and "NO". Custom
// tokens can be set through the exported fields or NewBooleanOutputParser.
type BooleanOutputParser struct {
	// TrueVal is the token parsed as true (default "YES").
	TrueVal string
	// FalseVal is the token parsed as false (default "NO").
	FalseVal string
}

// NewBooleanOutputParser creates a BooleanOutputParser with the default
// "YES"/"NO" tokens.
func NewBooleanOutputParser() BooleanOutputParser {
	return BooleanOutputParser{
		TrueVal:  defaultBooleanTrueVal,
		FalseVal: defaultBooleanFalseVal,
	}
}

func (p BooleanOutputParser) trueVal() string {
	if p.TrueVal == "" {
		return defaultBooleanTrueVal
	}
	return p.TrueVal
}

func (p BooleanOutputParser) falseVal() string {
	if p.FalseVal == "" {
		return defaultBooleanFalseVal
	}
	return p.FalseVal
}

// Parse converts YES/NO model output into a boolean. Matching is
// case-insensitive and uses whole-word boundaries, so a token embedded in a
// longer word (for example "NO_OUTPUT") is not treated as a boolean token.
// Ambiguous output containing both tokens, and output containing neither,
// produce an *lcerrors.OutputParserException.
func (p BooleanOutputParser) Parse(ctx context.Context, text string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}

	trueVal := p.trueVal()
	falseVal := p.falseVal()
	pattern := fmt.Sprintf(`(?i)\b(%s|%s)\b`, regexp.QuoteMeta(trueVal), regexp.QuoteMeta(falseVal))
	re, err := regexp.Compile(pattern)
	if err != nil {
		return false, &lcerrors.OutputParserException{
			Message:   fmt.Sprintf("compile boolean output pattern: %v", err),
			LLMOutput: text,
			Err:       err,
		}
	}

	truthy := make(map[string]struct{})
	for _, match := range re.FindAllStringSubmatch(text, -1) {
		if len(match) > 1 {
			truthy[strings.ToUpper(match[1])] = struct{}{}
		}
	}

	_, hasTrue := truthy[strings.ToUpper(trueVal)]
	_, hasFalse := truthy[strings.ToUpper(falseVal)]
	switch {
	case hasTrue && hasFalse:
		return false, &lcerrors.OutputParserException{
			Message: fmt.Sprintf(
				"Ambiguous response. Both %s and %s in received: %s.",
				trueVal, falseVal, text,
			),
			LLMOutput: text,
		}
	case hasTrue:
		return true, nil
	case hasFalse:
		return false, nil
	default:
		return false, &lcerrors.OutputParserException{
			Message: fmt.Sprintf(
				"BooleanOutputParser expected output value to include either %s or %s. Received %s.",
				trueVal, falseVal, text,
			),
			LLMOutput: text,
		}
	}
}

// FormatInstructions returns parser instructions describing the expected
// YES/NO output.
func (p BooleanOutputParser) FormatInstructions() string {
	return fmt.Sprintf("Your response should be either %s or %s.", p.trueVal(), p.falseVal())
}

// NoOutputParser parses outputs that may contain a null-string sentinel. It
// mirrors Python's NoOutputParser used by LLMChainExtractor: a response equal
// to the sentinel (after trimming) parses to an empty string, which the
// extractor treats as "document has no relevant content" and drops. Any other
// response is returned trimmed.
//
// The zero value is usable and defaults to the "NO_OUTPUT" sentinel.
type NoOutputParser struct {
	// NoOutputStr is the sentinel parsed to an empty string (default "NO_OUTPUT").
	NoOutputStr string
}

// NewNoOutputParser creates a NoOutputParser with the default "NO_OUTPUT"
// sentinel.
func NewNoOutputParser() NoOutputParser {
	return NoOutputParser{NoOutputStr: DefaultNoOutputString}
}

func (p NoOutputParser) marker() string {
	if p.NoOutputStr == "" {
		return DefaultNoOutputString
	}
	return p.NoOutputStr
}

// Parse returns an empty string when the trimmed text equals the no-output
// sentinel; otherwise it returns the trimmed text.
func (p NoOutputParser) Parse(ctx context.Context, text string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	cleaned := strings.TrimSpace(text)
	if cleaned == p.marker() {
		return "", nil
	}
	return cleaned, nil
}

// FormatInstructions returns parser instructions describing the no-output
// sentinel.
func (p NoOutputParser) FormatInstructions() string {
	return fmt.Sprintf("Return the extracted text, or %s if nothing is relevant.", p.marker())
}

var (
	_ Parser[bool]   = BooleanOutputParser{}
	_ Parser[string] = NoOutputParser{}
)
