package backends

import "testing"

func TestMatchGlob(t *testing.T) {
	tests := []struct {
		name    string
		pattern string
		path    string
		want    bool
	}{
		{"single segment wildcard matches", "a/*.go", "a/b.go", true},
		{"single segment wildcard does not span slash", "a/*.go", "a/b/c.go", false},
		{"double star spans directories", "a/**/*.go", "a/b/c.go", true},
		{"double star matches zero directories", "a/**/*.go", "a/c.go", true},
		{"double star alone matches everything", "**", "a/b/c.go", true},
		{"question mark matches one char", "a?c/go.mod", "abc/go.mod", true},
		{"question mark requires a char", "a?c/go.mod", "ac/go.mod", false},
		{"character class matches member", "[abc]/x.txt", "b/x.txt", true},
		{"character class rejects non member", "[abc]/x.txt", "d/x.txt", false},
		{"braces match literally", "a.{x,y}", "a.{x,y}", true},
		{"no brace expansion", "a.{x,y}", "a.x", false},
		{"full path must match", "a/*.go", "b/b.go", false},
		{"rooted pattern matches rooted path", "/a/*.go", "/a/b.go", true},
		{"rooted pattern rejects unrooted path", "/a/*.go", "a/b.go", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchGlob(tt.pattern, tt.path); got != tt.want {
				t.Errorf("MatchGlob(%q, %q) = %v, want %v", tt.pattern, tt.path, got, tt.want)
			}
		})
	}
}

func TestGrepLines(t *testing.T) {
	content := "abc\nnope\na.c\nxa.cy\nanother a.c line"
	tests := []struct {
		name     string
		pattern  string
		maxCount int
		want     []GrepLine
	}{
		{
			name:     "literal substring is not regex",
			pattern:  "a.c",
			maxCount: 0,
			want: []GrepLine{
				{LineNumber: 3, Text: "a.c"},
				{LineNumber: 4, Text: "xa.cy"},
				{LineNumber: 5, Text: "another a.c line"},
			},
		},
		{
			name:     "maxCount limits results",
			pattern:  "a.c",
			maxCount: 1,
			want:     []GrepLine{{LineNumber: 3, Text: "a.c"}},
		},
		{
			name:     "negative maxCount is unlimited",
			pattern:  "a.c",
			maxCount: -1,
			want: []GrepLine{
				{LineNumber: 3, Text: "a.c"},
				{LineNumber: 4, Text: "xa.cy"},
				{LineNumber: 5, Text: "another a.c line"},
			},
		},
		{
			name:     "no matches returns empty",
			pattern:  "zzz",
			maxCount: 0,
			want:     nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GrepLines(content, tt.pattern, tt.maxCount)
			if len(got) != len(tt.want) {
				t.Fatalf("GrepLines(%q) returned %d lines, want %d: %+v", tt.pattern, len(got), len(tt.want), got)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("GrepLines(%q)[%d] = %+v, want %+v", tt.pattern, i, got[i], tt.want[i])
				}
			}
		})
	}
}
