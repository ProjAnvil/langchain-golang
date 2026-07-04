package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestRedactionDetectsAndAppliesStrategies(t *testing.T) {
	emailText := "Contact me at a@example.com"
	emails := DetectEmail(emailText)
	if len(emails) != 1 || emails[0].Value != "a@example.com" {
		t.Fatalf("email matches: %#v", emails)
	}
	redacted, err := ApplyRedactionStrategy(emailText, emails, RedactionRedact)
	if err != nil {
		t.Fatalf("redact: %v", err)
	}
	if redacted != "Contact me at [REDACTED_EMAIL]" {
		t.Fatalf("redacted mismatch: %q", redacted)
	}

	cards := DetectCreditCard("card 4111-1111-1111-1111")
	if len(cards) != 1 {
		t.Fatalf("credit card matches: %#v", cards)
	}
	masked, err := ApplyRedactionStrategy("card 4111-1111-1111-1111", cards, RedactionMask)
	if err != nil {
		t.Fatalf("mask: %v", err)
	}
	if masked != "card ****-****-****-1111" {
		t.Fatalf("masked mismatch: %q", masked)
	}

	hashed, err := ApplyRedactionStrategy(emailText, emails, RedactionHash)
	if err != nil {
		t.Fatalf("hash: %v", err)
	}
	if !strings.Contains(hashed, "<email_hash:") {
		t.Fatalf("hash mismatch: %q", hashed)
	}
}

func TestPIIMiddlewareBeforeModelRedactsInputAndToolResults(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIApplyToToolResults(true))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "lookup"}}
	update, err := middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{
		messages.Human("email me at user@example.com"),
		ai,
		messages.Tool("1", "tool saw tool@example.com"),
	}})
	if err != nil {
		t.Fatalf("before model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if !strings.Contains(msgs[0].Content, "[REDACTED_EMAIL]") {
		t.Fatalf("human not redacted: %#v", msgs[0])
	}
	if !strings.Contains(msgs[2].Content, "[REDACTED_EMAIL]") {
		t.Fatalf("tool not redacted: %#v", msgs[2])
	}
}

func TestPIIMiddlewareAfterModelRedactsOutputAndToolArgs(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIApplyToInput(false), WithPIIApplyToOutput(true))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	ai := messages.AI("send to bot@example.com")
	ai.ToolCalls = []messages.ToolCall{{ID: "1", Name: "send", Args: map[string]any{"to": "user@example.com"}}}
	update, err := middleware.AfterModel(context.Background(), map[string]any{"messages": []messages.Message{messages.Human("hi"), ai}})
	if err != nil {
		t.Fatalf("after model: %v", err)
	}
	msgs := update["messages"].([]messages.Message)
	if !strings.Contains(msgs[1].Content, "[REDACTED_EMAIL]") {
		t.Fatalf("ai content not redacted: %#v", msgs[1])
	}
	if msgs[1].ToolCalls[0].Args["to"] != "[REDACTED_EMAIL]" {
		t.Fatalf("tool args not redacted: %#v", msgs[1].ToolCalls[0].Args)
	}
}

func TestPIIMiddlewareBlockRaisesDetectionError(t *testing.T) {
	middleware, err := NewPIIMiddleware("email", WithPIIStrategy(RedactionBlock))
	if err != nil {
		t.Fatalf("new pii middleware: %v", err)
	}
	_, err = middleware.BeforeModel(context.Background(), map[string]any{"messages": []messages.Message{messages.Human("user@example.com")}})
	var piiErr PIIDetectionError
	if !errors.As(err, &piiErr) {
		t.Fatalf("expected PIIDetectionError, got %v", err)
	}
}

func TestPIIMiddlewareUnknownTypeRequiresDetector(t *testing.T) {
	_, err := NewPIIMiddleware("api_key")
	if err == nil {
		t.Fatal("expected unknown type error")
	}
}

// TestPIIStreamTransformer_BoundaryStraddle verifies that a PII pattern split
// across two streaming deltas is still redacted. The lookback buffer must hold
// back the trailing tail of the first delta so the regex can complete against
// the next delta's prefix.
func TestPIIStreamTransformer_BoundaryStraddle(t *testing.T) {
	// Pattern "SSN-123" split across two deltas: "SSN" + "-123".
	patterns := []string{`SSN-\d+`}
	xform := NewPIIStreamTransformer(patterns) // implements WrapModelStreamHook
	tf := xform.TransformModelStream(func(s string) string { return s })

	got := tf("prefix SSN") + tf("-123 suffix")
	if strings.Contains(got, "SSN-123") {
		t.Errorf("pattern spanning chunks not redacted: %q", got)
	}
	if !strings.Contains(got, "[REDACTED") {
		t.Errorf("expected redaction token, got %q", got)
	}
}

// TestPIIStreamTransformer_MultiCallNoCorruption verifies that the transformer
// is robust to Task 3.1's multi-call pattern: the same composed DeltaTransform
// is invoked once per delta, then once on the content-block-finish assembled
// text, then once on the model_end assembled text. A naive append-only buffer
// would duplicate the leading fragment on the full-text call. This test
// simulates that sequence directly and asserts the model_end text is the
// redacted full text with no duplication.
func TestPIIStreamTransformer_MultiCallNoCorruption(t *testing.T) {
	patterns := []string{`SSN-\d+`}
	xform := NewPIIStreamTransformer(patterns)
	tf := xform.TransformModelStream(func(s string) string { return s })

	// Per-delta path: three small chunks whose concatenation contains the PII.
	deltaOut := tf("pre ") + tf("SSN") + tf("-1 suffix")
	// content-block-finish: full assembled text.
	finishOut := tf("pre SSN-1 suffix")
	// model_end: same full assembled text again.
	endOut := tf("pre SSN-1 suffix")

	if strings.Contains(deltaOut, "SSN-1") {
		t.Errorf("delta leaked raw PII: %q", deltaOut)
	}
	if !strings.Contains(deltaOut, "[REDACTED") {
		t.Errorf("delta missing redaction token: %q", deltaOut)
	}
	// The full-text calls must NOT begin with the delta-emitted prefix
	// (which would indicate the buffer appended the full text to a leftover
	// tail). They must equal the cleanly-redacted full text.
	if strings.Contains(finishOut, "SSN-1") {
		t.Errorf("finish leaked raw PII: %q", finishOut)
	}
	if !strings.Contains(finishOut, "[REDACTED") {
		t.Errorf("finish missing redaction token: %q", finishOut)
	}
	if finishOut != endOut {
		t.Errorf("finish vs model_end mismatch (corruption): %q vs %q", finishOut, endOut)
	}
	// Hard corruption guard: the full-text output must contain exactly one
	// "pre " prefix, not the doubled "pre pre " that a naive append-only
	// buffer would produce.
	if strings.Count(finishOut, "pre ") != 1 {
		t.Errorf("model_end text corrupted (prefix duplicated/missing): %q", finishOut)
	}
	// And it must end with "suffix", not be truncated.
	if !strings.HasSuffix(finishOut, "suffix") {
		t.Errorf("model_end text truncated: %q", finishOut)
	}
}

// TestPIIStreamTransformer_Flush verifies that Flush emits the held tail at
// stream end, redacted, so callers reading only deltas + Flush see the full
// redacted text.
func TestPIIStreamTransformer_Flush(t *testing.T) {
	xform := NewPIIStreamTransformer([]string{`SSN-\d+`})
	tf := xform.TransformModelStream(func(s string) string { return s })

	out := tf("a SSN-1 b")
	flushed := xform.Flush()
	assembled := out + flushed

	if strings.Contains(assembled, "SSN-1") {
		t.Errorf("flushed stream leaked raw PII: %q", assembled)
	}
	if !strings.Contains(assembled, "[REDACTED") {
		t.Errorf("flushed stream missing redaction: %q", assembled)
	}
	if !strings.HasSuffix(assembled, "b") {
		t.Errorf("flushed stream lost trailing text: %q", assembled)
	}
}

// TestPIIStreamTransformer_LongPatternStraddle verifies that a long fixed
// pattern (whose match length exceeds its source length) is still redacted
// when split across deltas + Flush. The lookback is sized to the pattern
// source length, so the constructor must pick the larger of (source length,
// any other pattern) — this test asserts that sizing and the end-to-end
// redaction both work.
func TestPIIStreamTransformer_LongPatternStraddle(t *testing.T) {
	longPat := `TOKEN-[A-Z]{20}`
	xform := NewPIIStreamTransformer([]string{longPat})
	if xform.Lookback() < len(longPat) {
		t.Fatalf("lookback %d < pattern length %d", xform.Lookback(), len(longPat))
	}
	tf := xform.TransformModelStream(func(s string) string { return s })

	got := tf("lead TOKEN") + tf("-ABCDEFGHIJKLMNOPQRST trail") + xform.Flush()
	if strings.Contains(got, "TOKEN-") {
		t.Errorf("long pattern leaked: %q", got)
	}
	if !strings.Contains(got, "[REDACTED") {
		t.Errorf("long pattern not redacted: %q", got)
	}
}

// TestPIIStreamTransformer_CoincidentalPrefixNoHeldLoss is the covering test
// for review Fix 1: a per-delta call whose text happens to start with the
// accumulated rawSeen triggers the terminal branch (the same signature as a
// real content-block-finish / model_end full-text call). With the OLD design
// (clear held on reset) the buffered tail is discarded, so a PII pattern
// straddling the reset boundary (start in the held tail, completion in a
// later delta) leaks. With the preserve-held fix the tail survives and the
// next delta still concatenates with it, so the straddling PII is redacted.
//
// Sequence: delta "pre SSN" (held buffers "SSN"), then a coincidental delta
// equal to rawSeen ("pre SSN") that false-triggers terminal, then "-123 post"
// which completes the PII. The pre-fix output contains the raw "SSN-123"; the
// fixed output redacts it.
func TestPIIStreamTransformer_CoincidentalPrefixNoHeldLoss(t *testing.T) {
	xform := NewPIIStreamTransformer([]string{`SSN-\d+`})
	tf := xform.TransformModelStream(func(s string) string { return s })

	// Delta 1: "SSN" is buffered into the held tail (no complete match yet).
	d1 := tf("pre SSN")
	// Delta 2: coincidental — equals rawSeen, so it false-triggers the
	// terminal branch. This is NOT a real finish; more deltas follow.
	d2 := tf("pre SSN")
	// Delta 3: completes the PII ("-123"). With preserve-held, the buffered
	// "SSN" concatenates with "-123" and the regex matches; without it, the
	// "SSN" was discarded and "-123" leaks unredacted.
	d3 := tf("-123 post")
	got := d1 + d2 + d3 + xform.Flush()

	if strings.Contains(got, "SSN-123") {
		t.Errorf("coincidental-prefix reset leaked straddling PII (held lost): %q", got)
	}
	// The PII completion must have produced a redaction token via the
	// preserved held-tail path. (The token is absent in the pre-fix code,
	// which discards held and redacts "-123 post" as a no-match.)
	if !strings.Contains(got, "[REDACTED]") {
		t.Errorf("expected [REDACTED] from preserved held-tail redaction, got %q", got)
	}
}

// TestPIIStreamTransformer_ModelEndWithoutFinishIsComplete is the covering
// test for review Fix 2: when the model_end full-text call arrives after
// deltas WITHOUT an intervening content-block-finish event (or in any
// multi-call ordering where the terminal text equals the accumulated raw
// deltas), the transform must return the fully-redacted complete text — not
// a lookback-truncated tail. A naive delta-path execution on the full text
// would withhold the trailing `lookback` bytes and truncate the final
// message. This test asserts the terminal branch fires and the suffix is
// intact.
func TestPIIStreamTransformer_ModelEndWithoutFinishIsComplete(t *testing.T) {
	xform := NewPIIStreamTransformer([]string{`SSN-\d+`})
	tf := xform.TransformModelStream(func(s string) string { return s })

	// Per-delta path: the raw PII is split across the deltas.
	deltaOut := tf("alpha SSN") + tf("-99 beta gamma delta")
	// model_end full text arrives with NO prior content-block-finish. It
	// equals the concatenation of the deltas, so the terminal branch must
	// fire and return the cleanly-redacted FULL text.
	endOut := tf("alpha SSN-99 beta gamma delta")

	// The delta stream must not leak the raw PII.
	if strings.Contains(deltaOut, "SSN-99") {
		t.Errorf("delta leaked raw PII: %q", deltaOut)
	}
	// The model_end text must be the fully-redacted complete message: no
	// raw PII, a redaction token present, AND the trailing suffix intact
	// (not truncated by a withheld lookback tail).
	if strings.Contains(endOut, "SSN-99") {
		t.Errorf("model_end leaked raw PII: %q", endOut)
	}
	if !strings.Contains(endOut, "[REDACTED]") {
		t.Errorf("model_end missing redaction token: %q", endOut)
	}
	if !strings.HasSuffix(endOut, "delta") {
		t.Errorf("model_end truncated (lookback tail withheld): %q", endOut)
	}
	if c := strings.Count(endOut, "alpha"); c != 1 {
		t.Errorf("model_end prefix corrupted (count(alpha)=%d): %q", c, endOut)
	}
}

// TestPIIStreamTransformer_FlushConcurrentWithDeltas verifies the Flush race
// fix (review Minor Fix): a concurrent Flush during streaming must not race
// with the delta path's held read/write. Run under `go test -race` to catch
// the data race that the pre-fix code (no synchronization on held) exhibited.
func TestPIIStreamTransformer_FlushConcurrentWithDeltas(t *testing.T) {
	xform := NewPIIStreamTransformer([]string{`SSN-\d+`})
	tf := xform.TransformModelStream(func(s string) string { return s })

	deltas := []string{"a SSN", "-1 b", "c SSN", "-2 d", "e SSN", "-3 f"}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 200 {
			// Hammer Flush from another goroutine while deltas flow.
			xform.Flush()
		}
	}()
	for _, d := range deltas {
		tf(d)
	}
	xform.Flush()
	<-done
}
