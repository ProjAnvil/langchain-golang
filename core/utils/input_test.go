package utils

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

// smallBFailingWriter is an io.Writer that always fails, used to verify that
// PrintText propagates write errors instead of panicking.
type smallBFailingWriter struct{}

func (smallBFailingWriter) Write(p []byte) (int, error) {
	return 0, errors.New("write failed")
}

func TestGetColorMappingRotation(t *testing.T) {
	items := []string{"a", "b", "c", "d", "e", "f"}
	colors := []string{"blue", "yellow", "pink"}
	want := map[string]string{
		"a": "blue",
		"b": "yellow",
		"c": "pink",
		"d": "blue",
		"e": "yellow",
		"f": "pink",
	}
	got := GetColorMapping(items, colors)
	if len(got) != len(want) {
		t.Fatalf("GetColorMapping returned %d entries, want %d", len(got), len(want))
	}
	for key, value := range want {
		if got[key] != value {
			t.Errorf("GetColorMapping[%q] = %q, want %q", key, got[key], value)
		}
	}
	again := GetColorMapping(items, colors)
	for key := range want {
		if again[key] != got[key] {
			t.Errorf("GetColorMapping is not deterministic for %q: %q vs %q", key, again[key], got[key])
		}
	}
}

func TestGetColorMappingEmptyItems(t *testing.T) {
	got := GetColorMapping(nil, []string{"blue", "green"})
	if len(got) != 0 {
		t.Fatalf("GetColorMapping with empty items = %v, want empty map", got)
	}
}

func TestGetColorMappingEmptyColors(t *testing.T) {
	got := GetColorMapping([]string{"a", "b", "c"}, nil)
	if len(got) != 0 {
		t.Fatalf("GetColorMapping with empty colors = %v, want empty map", got)
	}
}

func TestGetColorMappingDuplicateItems(t *testing.T) {
	got := GetColorMapping([]string{"a", "b", "a"}, []string{"blue", "green"})
	if got["a"] != "blue" || got["b"] != "green" {
		t.Fatalf("GetColorMapping duplicate items = %v", got)
	}
}

func TestGetColoredTextKnownColors(t *testing.T) {
	cases := map[string]string{
		"blue":   "36;1",
		"yellow": "33;1",
		"pink":   "38;5;200",
		"green":  "32;1",
		"red":    "31;1",
	}
	for color, code := range cases {
		want := "\x1b[" + code + "m\x1b[1;3mhello\x1b[0m"
		if got := GetColoredText("hello", color); got != want {
			t.Errorf("GetColoredText(%q) = %q, want %q", color, got, want)
		}
	}
}

func TestGetColoredTextRawCode(t *testing.T) {
	want := "\x1b[31;1m\x1b[1;3mhello\x1b[0m"
	if got := GetColoredText("hello", "31;1"); got != want {
		t.Fatalf("GetColoredText raw code = %q, want %q", got, want)
	}
}

func TestGetColoredTextUnknownColorNoPanic(t *testing.T) {
	got := GetColoredText("hello", "purple")
	if !strings.HasPrefix(got, "\x1b[") || !strings.HasSuffix(got, "\x1b[0m") || !strings.Contains(got, "hello") {
		t.Fatalf("GetColoredText unknown color = %q", got)
	}
}

func TestGetColoredTextEmptyText(t *testing.T) {
	want := "\x1b[32;1m\x1b[1;3m\x1b[0m"
	if got := GetColoredText("", "green"); got != want {
		t.Fatalf("GetColoredText empty text = %q, want %q", got, want)
	}
}

func TestGetBoldedText(t *testing.T) {
	want := "\x1b[1mhello\x1b[0m"
	if got := GetBoldedText("hello"); got != want {
		t.Fatalf("GetBoldedText = %q, want %q", got, want)
	}
}

func TestPrintTextPlain(t *testing.T) {
	var buf bytes.Buffer
	if err := PrintText("hello", "", false, &buf); err != nil {
		t.Fatalf("PrintText returned error: %v", err)
	}
	if buf.String() != "hello" {
		t.Fatalf("PrintText plain = %q, want %q", buf.String(), "hello")
	}
}

func TestPrintTextColored(t *testing.T) {
	var buf bytes.Buffer
	if err := PrintText("hello", "green", false, &buf); err != nil {
		t.Fatalf("PrintText returned error: %v", err)
	}
	if buf.String() != GetColoredText("hello", "green") {
		t.Fatalf("PrintText colored = %q", buf.String())
	}
}

func TestPrintTextBold(t *testing.T) {
	var buf bytes.Buffer
	if err := PrintText("hello", "", true, &buf); err != nil {
		t.Fatalf("PrintText returned error: %v", err)
	}
	if buf.String() != GetBoldedText("hello") {
		t.Fatalf("PrintText bold = %q", buf.String())
	}
}

func TestPrintTextColoredBold(t *testing.T) {
	var buf bytes.Buffer
	if err := PrintText("hello", "red", true, &buf); err != nil {
		t.Fatalf("PrintText returned error: %v", err)
	}
	want := GetColoredText(GetBoldedText("hello"), "red")
	if buf.String() != want {
		t.Fatalf("PrintText colored bold = %q, want %q", buf.String(), want)
	}
}

func TestPrintTextNoTrailingNewline(t *testing.T) {
	var buf bytes.Buffer
	if err := PrintText("hello", "red", true, &buf); err != nil {
		t.Fatalf("PrintText returned error: %v", err)
	}
	if strings.HasSuffix(buf.String(), "\n") {
		t.Fatalf("PrintText appended a newline: %q", buf.String())
	}
}

func TestPrintTextNilWriter(t *testing.T) {
	if err := PrintText("hello", "", false, nil); err == nil {
		t.Fatal("PrintText with nil writer should return an error")
	}
}

func TestPrintTextWriteError(t *testing.T) {
	if err := PrintText("hello", "", false, smallBFailingWriter{}); err == nil {
		t.Fatal("PrintText should propagate the writer error")
	}
}
