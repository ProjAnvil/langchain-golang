package outputparser

import (
	"context"
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
