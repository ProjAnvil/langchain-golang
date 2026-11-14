package backends

import (
	"errors"
	"strings"
	"testing"
)

func TestNormalizePath(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    string
		wantErr bool
	}{
		{"relative becomes rooted", "x/y", "/x/y", false},
		{"dotdot within root resolves", "/a/../b", "/b", false},
		{"dotdot escaping root rejected", "a/../../x", "", true},
		{"double slash collapses", "//a", "/a", false},
		{"leading dotdot rejected", "../etc/passwd", "", true},
		{"root stays root", "/", "/", false},
		{"dot segments and duplicate slashes", "/./foo//bar", "/foo/bar", false},
		{"backslashes become slashes", "a\\b\\c", "/a/b/c", false},
		{"mixed separators", "foo/bar\\baz/qux", "/foo/bar/baz/qux", false},
		{"empty normalizes like upstream", "", "/.", false},
		{"dot normalizes like upstream", ".", "/.", false},
		{"dotdot substring in filename kept", "foo..bar.txt", "/foo..bar.txt", false},
		{"tilde prefix rejected", "~/secret.txt", "", true},
		{"windows drive rejected", "D:/data/file.txt", "", true},
		{"windows drive with backslashes rejected", "C:\\Users\\file.txt", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NormalizePath(tt.in)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("NormalizePath(%q) = %q, want error", tt.in, got)
				}
				var opErr *Error
				if !errors.As(err, &opErr) {
					t.Fatalf("NormalizePath(%q) error %v is not *backends.Error", tt.in, err)
				}
				if opErr.Code != ErrInvalidPath {
					t.Errorf("error code = %q, want %q", opErr.Code, ErrInvalidPath)
				}
				if !strings.Contains(err.Error(), ErrInvalidPath) {
					t.Errorf("error %q does not contain literal %q", err, ErrInvalidPath)
				}
				return
			}
			if err != nil {
				t.Fatalf("NormalizePath(%q) unexpected error: %v", tt.in, err)
			}
			if got != tt.want {
				t.Errorf("NormalizePath(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestNormalizeReadBounds(t *testing.T) {
	tests := []struct {
		name      string
		offset    int
		limit     int
		wantStart int
		wantCount int
	}{
		{"passthrough", 5, 10, 5, 10},
		{"negative offset clamps to zero", -3, 10, 0, 10},
		{"zero limit is empty window", 5, 0, 5, 0},
		{"negative limit is empty window", 5, -2, 5, 0},
		{"both invalid clamp", -1, -1, 0, 0},
		{"zero offset and positive limit", 0, 3, 0, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			start, count := NormalizeReadBounds(tt.offset, tt.limit)
			if start != tt.wantStart || count != tt.wantCount {
				t.Errorf("NormalizeReadBounds(%d, %d) = (%d, %d), want (%d, %d)",
					tt.offset, tt.limit, start, count, tt.wantStart, tt.wantCount)
			}
		})
	}
}
