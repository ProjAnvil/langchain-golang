package outputparser

import (
	"context"
	"reflect"
	"regexp"
	"testing"

	"github.com/projanvil/langchain-golang/core/utils"
)

func TestDropLastNIterator(t *testing.T) {
	cases := []struct {
		name   string
		values []int
		n      int
		want   []int
	}{
		{"drop two", []int{1, 2, 3, 4, 5}, 2, []int{1, 2, 3}},
		{"drop one", []int{1, 2, 3}, 1, []int{1, 2}},
		{"drop zero", []int{1, 2, 3}, 0, []int{1, 2, 3}},
		{"drop more than length", []int{1, 2, 3}, 5, []int{}},
		{"negative drops nothing", []int{1, 2, 3}, -3, []int{1, 2, 3}},
		{"empty source", []int{}, 1, []int{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := utils.CollectIterator(t.Context(), DropLastN(utils.NewSliceIterator(tc.values), tc.n))
			if err != nil {
				t.Fatalf("collect: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestDropLastNIteratorCloseIdempotent(t *testing.T) {
	iter := DropLastN(utils.NewSliceIterator([]int{1, 2, 3}), 1)
	if err := iter.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := iter.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	if _, ok, err := iter.Next(t.Context()); err != nil || ok {
		t.Fatalf("next after close: ok=%v err=%v", ok, err)
	}
}

func TestDropLastNIteratorContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := utils.CollectIterator(ctx, DropLastN(utils.NewSliceIterator([]int{1, 2, 3}), 1))
	if err == nil {
		t.Fatal("expected context error")
	}
}

func TestDropLastNChannel(t *testing.T) {
	cases := []struct {
		name string
		n    int
		want []int
	}{
		{"drop two", 2, []int{1, 2, 3}},
		{"drop zero", 0, []int{1, 2, 3, 4, 5}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := make(chan int, 5)
			for i := 1; i <= 5; i++ {
				src <- i
			}
			close(src)
			got := []int{}
			for value := range DropLastNChannel(t.Context(), src, tc.n) {
				got = append(got, value)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestNumberedListParserStreamWithholdsLastItem(t *testing.T) {
	parser := &NumberedListParserStream{}
	got, err := parser.Parse(t.Context(), "1. alpha\n2. beta\n3. gamma")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if want := []string{"alpha", "beta"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("parse got %#v want %#v", got, want)
	}
	rest, err := parser.Flush(t.Context())
	if err != nil {
		t.Fatalf("flush: %v", err)
	}
	if want := []string{"gamma"}; !reflect.DeepEqual(rest, want) {
		t.Fatalf("flush got %#v want %#v", rest, want)
	}
}

func TestNumberedListParserStreamChunkBoundaries(t *testing.T) {
	chunks := []string{"1. al", "pha\n2. be", "ta\n3. ga", "mma"}
	parser := &NumberedListParserStream{}
	all := []string{}
	for _, chunk := range chunks {
		items, err := parser.Parse(t.Context(), chunk)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		all = append(all, items...)
	}
	rest, err := parser.Flush(t.Context())
	if err != nil {
		t.Fatalf("flush: %v", err)
	}
	all = append(all, rest...)
	if want := []string{"alpha", "beta", "gamma"}; !reflect.DeepEqual(all, want) {
		t.Fatalf("got %#v want %#v", all, want)
	}
}

func TestMarkdownListParserStreamWithheldAndChunkBoundaries(t *testing.T) {
	chunks := []string{"- al", "pha\n- be", "ta\n- ga", "mma"}
	parser := &MarkdownListParserStream{}
	all := []string{}
	for _, chunk := range chunks {
		items, err := parser.Parse(t.Context(), chunk)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		all = append(all, items...)
	}
	rest, err := parser.Flush(t.Context())
	if err != nil {
		t.Fatalf("flush: %v", err)
	}
	all = append(all, rest...)
	if want := []string{"alpha", "beta", "gamma"}; !reflect.DeepEqual(all, want) {
		t.Fatalf("got %#v want %#v", all, want)
	}
}

func TestListStreamTransformMatchesBatchParser(t *testing.T) {
	numberedFull := "1. alpha\n2. beta\n3. gamma\nnot an item"
	numberedStream := &NumberedListParserStream{}
	gotNumbered, err := numberedStream.Transform(t.Context(), []string{numberedFull})
	if err != nil {
		t.Fatalf("numbered transform: %v", err)
	}
	wantNumbered, err := (NumberedListParser{}).Parse(t.Context(), numberedFull)
	if err != nil {
		t.Fatalf("numbered batch: %v", err)
	}
	if !reflect.DeepEqual(gotNumbered, wantNumbered) {
		t.Fatalf("numbered got %#v want %#v", gotNumbered, wantNumbered)
	}

	markdownFull := "- alpha\n* beta\nplain line\n- gamma"
	markdownStream := &MarkdownListParserStream{}
	gotMarkdown, err := markdownStream.Transform(t.Context(), []string{markdownFull})
	if err != nil {
		t.Fatalf("markdown transform: %v", err)
	}
	wantMarkdown, err := (MarkdownListParser{}).Parse(t.Context(), markdownFull)
	if err != nil {
		t.Fatalf("markdown batch: %v", err)
	}
	if !reflect.DeepEqual(gotMarkdown, wantMarkdown) {
		t.Fatalf("markdown got %#v want %#v", gotMarkdown, wantMarkdown)
	}
}

func TestListStreamCustomPattern(t *testing.T) {
	parser := &NumberedListParserStream{Pattern: regexp.MustCompile(`(?m)^item: (.+)$`)}
	got, err := parser.Transform(t.Context(), []string{"item: one\nitem: two"})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if want := []string{"one", "two"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v want %#v", got, want)
	}
}

func TestListStreamEmptyAndNoMatches(t *testing.T) {
	parser := &NumberedListParserStream{}
	got, err := parser.Transform(t.Context(), nil)
	if err != nil {
		t.Fatalf("empty transform: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("empty got %#v", got)
	}

	got, err = parser.Transform(t.Context(), []string{"no list items here"})
	if err != nil {
		t.Fatalf("no matches transform: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("no matches got %#v", got)
	}
}

func TestListStreamFormatInstructions(t *testing.T) {
	if got := (&NumberedListParserStream{}).FormatInstructions(); got == "" {
		t.Fatal("numbered instructions empty")
	}
	if got := (&MarkdownListParserStream{}).FormatInstructions(); got == "" {
		t.Fatal("markdown instructions empty")
	}
}

func TestListStreamContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := (&NumberedListParserStream{}).Parse(ctx, "1. a"); err == nil {
		t.Fatal("expected context error from Parse")
	}
	if _, err := (&NumberedListParserStream{}).Transform(ctx, []string{"1. a"}); err == nil {
		t.Fatal("expected context error from Transform")
	}
	if _, err := (&MarkdownListParserStream{}).Flush(ctx); err == nil {
		t.Fatal("expected context error from Flush")
	}
}
