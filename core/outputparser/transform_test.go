package outputparser

import (
	"context"
	"math"
	"testing"
)

func TestTransformParsesIndependentChunks(t *testing.T) {
	parser := StringParser{}
	got, err := Transform(context.Background(), parser, []string{"a", "b"})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("got %#v", got)
	}
}

func TestCumulativeJSONParserTransform(t *testing.T) {
	parser := CumulativeJSONParser{}
	got, err := parser.Transform(context.Background(), []string{
		`{"answer":`,
		` 1,`,
		` "ok": true}`,
	})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %#v", got)
	}
	first := got[0].(map[string]any)
	if first["answer"].(jsonNumber).String() != "1" {
		t.Fatalf("first: %#v", first)
	}
	second := got[1].(map[string]any)
	if second["ok"] != true {
		t.Fatalf("second: %#v", second)
	}
}

func TestCumulativeJSONParserDiff(t *testing.T) {
	parser := CumulativeJSONParser{Diff: true}
	got, err := parser.Transform(context.Background(), []string{
		`{"answer":1`,
		`, "ok": true`,
		`, "answer":2}`,
	})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %#v", got)
	}
	first := got[0].([]JSONPatchOperation)
	if len(first) != 1 || first[0].Op != "add" || first[0].Path != "" {
		t.Fatalf("first patch: %#v", first)
	}
	second := got[1].([]JSONPatchOperation)
	if len(second) != 1 || second[0].Op != "add" || second[0].Path != "/ok" {
		t.Fatalf("second patch: %#v", second)
	}
	third := got[2].([]JSONPatchOperation)
	if len(third) != 1 || third[0].Op != "replace" || third[0].Path != "/answer" {
		t.Fatalf("third patch: %#v", third)
	}
}

func TestParsePartialJSONFence(t *testing.T) {
	got, ok, err := ParsePartialJSON("```json\n{\"name\":\"Ada\"}\n```")
	if err != nil {
		t.Fatalf("parse partial: %v", err)
	}
	if !ok {
		t.Fatal("expected parse")
	}
	if got.(map[string]any)["name"] != "Ada" {
		t.Fatalf("got %#v", got)
	}
}

func TestParsePartialJSONIncompleteToken(t *testing.T) {
	_, ok, err := ParsePartialJSON(`{"answer": tru`)
	if err != nil {
		t.Fatalf("parse partial: %v", err)
	}
	if ok {
		t.Fatal("expected incomplete token not to parse")
	}
}

func TestTransformContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Transform(ctx, StringParser{}, []string{"a"})
	if err == nil {
		t.Fatal("expected context error")
	}
}

func TestTransformPropagatesParseError(t *testing.T) {
	parser := NewJSONParser[map[string]any]("")
	_, err := Transform(context.Background(), parser, []string{"not json"})
	if err == nil {
		t.Fatal("expected parse error")
	}
}

func TestCumulativeJSONParserContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	parser := CumulativeJSONParser{}
	if _, err := parser.Transform(ctx, []string{"{}"}); err == nil {
		t.Fatal("expected context error")
	}
}

func TestCumulativeJSONParserSkipsUnparseable(t *testing.T) {
	parser := CumulativeJSONParser{}
	got, err := parser.Transform(context.Background(), []string{
		`{"a": tru`,
		`e}`,
	})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %#v", got)
	}
	if got[0].(map[string]any)["a"] != true {
		t.Fatalf("value: %#v", got[0])
	}
}

func TestParsePartialJSONEdgeCases(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{"empty", ""},
		{"whitespace", "   "},
		{"mismatched closing", `{"a":1]`},
		{"incomplete token", `tru`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok, err := ParsePartialJSON(tc.input)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if ok || got != nil {
				t.Fatalf("expected no parse, got %#v ok=%v", got, ok)
			}
		})
	}
}

func TestParsePartialJSONEscapedQuote(t *testing.T) {
	got, ok, err := ParsePartialJSON(`{"a":"x\"`)
	if err != nil || !ok {
		t.Fatalf("parse: ok=%v err=%v", ok, err)
	}
	if got.(map[string]any)["a"] != `x"` {
		t.Fatalf("got %#v", got)
	}
}

func TestStripJSONFenceEdgeCases(t *testing.T) {
	if got := stripJSONFence("no fence"); got != "no fence" {
		t.Fatalf("no fence: %q", got)
	}
	if got := stripJSONFence("```json"); got != "```json" {
		t.Fatalf("single line fence: %q", got)
	}
	if got := stripJSONFence("```json\n{}"); got != "{}" {
		t.Fatalf("unclosed fence: %q", got)
	}
}

func TestMakeJSONPatchRemoveAndNested(t *testing.T) {
	previous := map[string]any{
		"keep":   map[string]any{"n": float64(1)},
		"remove": "gone",
	}
	next := map[string]any{
		"keep": map[string]any{"n": float64(2)},
	}
	ops := MakeJSONPatch(previous, next)
	var removeOp, replaceOp *JSONPatchOperation
	for i := range ops {
		switch {
		case ops[i].Op == "remove" && ops[i].Path == "/remove":
			removeOp = &ops[i]
		case ops[i].Op == "replace" && ops[i].Path == "/keep/n":
			replaceOp = &ops[i]
		}
	}
	if removeOp == nil {
		t.Fatalf("missing remove op: %#v", ops)
	}
	if replaceOp == nil {
		t.Fatalf("missing nested replace op: %#v", ops)
	}
}

func TestMakeJSONPatchEscapedPath(t *testing.T) {
	ops := MakeJSONPatch(nil, map[string]any{"a/b~c": 1})
	if len(ops) != 1 || ops[0].Op != "add" || ops[0].Path != "" {
		t.Fatalf("root add: %#v", ops)
	}
	ops = MakeJSONPatch(map[string]any{}, map[string]any{"a/b~c": 1})
	if len(ops) != 1 || ops[0].Path != "/a~1b~0c" {
		t.Fatalf("escaped path: %#v", ops)
	}
}

func TestCloneJSONValueUnmarshalable(t *testing.T) {
	if got := cloneJSONValue(math.Inf(1)); got != math.Inf(1) {
		t.Fatalf("unmarshalable value should be returned as-is, got %#v", got)
	}
	if got := cloneJSONValue(map[string]any{"a": float64(1)}); got.(map[string]any)["a"] == nil {
		t.Fatalf("round trip: %#v", got)
	}
}
