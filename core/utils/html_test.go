package utils

import (
	"reflect"
	"regexp"
	"sort"
	"testing"
)

func sortedCopy(values []string) []string {
	out := append([]string(nil), values...)
	sort.Strings(out)
	return out
}

// Mirrors test_html.py::test_find_all_links_none.
func TestFindAllLinksNone(t *testing.T) {
	if got := FindAllLinks("<span>Hello world</span>", nil); len(got) != 0 {
		t.Fatalf("links: %v, want none", got)
	}
}

// Mirrors test_find_all_links_single.
func TestFindAllLinksSingle(t *testing.T) {
	htmls := []string{
		"href='foobar.com'",
		`href="foobar.com"`,
		`<div><a class="blah" href="foobar.com">hullo</a></div>`,
	}
	for _, html := range htmls {
		got := FindAllLinks(html, nil)
		if !reflect.DeepEqual(got, []string{"foobar.com"}) {
			t.Fatalf("FindAllLinks(%q) = %v, want [foobar.com]", html, got)
		}
	}
}

// Mirrors test_find_all_links_multiple.
func TestFindAllLinksMultiple(t *testing.T) {
	html := `<div><a class="blah" href="https://foobar.com">hullo</a></div>` +
		`<div><a class="bleh" href="/baz/cool">buhbye</a></div>`
	got := sortedCopy(FindAllLinks(html, nil))
	want := []string{"/baz/cool", "https://foobar.com"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("links: %v, want %v", got, want)
	}
}

// Mirrors test_find_all_links_ignore_suffix.
func TestFindAllLinksIgnoreSuffix(t *testing.T) {
	for _, suffix := range SuffixesToIgnore {
		if got := FindAllLinks(`href="foobar`+suffix+`"`, nil); len(got) != 0 {
			t.Fatalf("suffix %q: links %v, want none", suffix, got)
		}
		// Don't ignore if the suffix doesn't occur at the end of the link.
		got := FindAllLinks(`href="foobar`+suffix+`more"`, nil)
		if !reflect.DeepEqual(got, []string{"foobar" + suffix + "more"}) {
			t.Fatalf("suffix %q: links %v", suffix, got)
		}
	}
}

// Mirrors test_find_all_links_ignore_prefix.
func TestFindAllLinksIgnorePrefix(t *testing.T) {
	for _, prefix := range PrefixesToIgnore {
		if got := FindAllLinks(`href="`+prefix+`foobar"`, nil); len(got) != 0 {
			t.Fatalf("prefix %q: links %v, want none", prefix, got)
		}
		// Pound signs are split on when not prefixes.
		if prefix == "#" {
			continue
		}
		got := FindAllLinks(`href="foobar`+prefix+`more"`, nil)
		if !reflect.DeepEqual(got, []string{"foobar" + prefix + "more"}) {
			t.Fatalf("prefix %q: links %v", prefix, got)
		}
	}
}

// Mirrors test_find_all_links_drop_fragment.
func TestFindAllLinksDropFragment(t *testing.T) {
	got := FindAllLinks(`href="foobar.com/woah#section_one"`, nil)
	if !reflect.DeepEqual(got, []string{"foobar.com/woah"}) {
		t.Fatalf("links: %v", got)
	}
}

// A custom pattern replaces the default (Python's pattern= parameter); the
// link must be in capture group 1.
func TestFindAllLinksCustomPattern(t *testing.T) {
	pattern := regexp.MustCompile(`data-link="([^"]+)"`)
	got := FindAllLinks(`<a data-link="x.com">x</a>`, pattern)
	if !reflect.DeepEqual(got, []string{"x.com"}) {
		t.Fatalf("links: %v", got)
	}
}

const extractTestHTML = `<a href="https://foobar.com">one</a>` +
	`<a href="http://baz.net">two</a>` +
	`<a href="//foobar.com/hello">three</a>` +
	`<a href="/how/are/you/doing">four</a>`

// Mirrors test_extract_sub_links (all three scenarios).
func TestExtractSubLinks(t *testing.T) {
	got, err := ExtractSubLinks(extractTestHTML, "https://foobar.com", ExtractSubLinksOptions{})
	if err != nil {
		t.Fatalf("ExtractSubLinks: %v", err)
	}
	want := []string{
		"https://foobar.com",
		"https://foobar.com/hello",
		"https://foobar.com/how/are/you/doing",
	}
	if !reflect.DeepEqual(sortedCopy(got), want) {
		t.Fatalf("links: %v, want %v", sortedCopy(got), want)
	}

	got, err = ExtractSubLinks(extractTestHTML, "https://foobar.com/hello", ExtractSubLinksOptions{})
	if err != nil {
		t.Fatalf("ExtractSubLinks: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"https://foobar.com/hello"}) {
		t.Fatalf("links: %v", got)
	}

	got, err = ExtractSubLinks(extractTestHTML, "https://foobar.com/hello", ExtractSubLinksOptions{AllowOutside: true})
	if err != nil {
		t.Fatalf("ExtractSubLinks: %v", err)
	}
	want = []string{
		"http://baz.net",
		"https://foobar.com",
		"https://foobar.com/hello",
		"https://foobar.com/how/are/you/doing",
	}
	if !reflect.DeepEqual(sortedCopy(got), want) {
		t.Fatalf("links: %v, want %v", sortedCopy(got), want)
	}
}

// Mirrors test_extract_sub_links_base (relative link joins against url, not
// base_url).
func TestExtractSubLinksBase(t *testing.T) {
	html := extractTestHTML + `<a href="alexis.html"</a>`
	got, err := ExtractSubLinks(html, "https://foobar.com/hello/bill.html", ExtractSubLinksOptions{
		BaseURL: "https://foobar.com",
	})
	if err != nil {
		t.Fatalf("ExtractSubLinks: %v", err)
	}
	want := []string{
		"https://foobar.com",
		"https://foobar.com/hello",
		"https://foobar.com/hello/alexis.html",
		"https://foobar.com/how/are/you/doing",
	}
	if !reflect.DeepEqual(sortedCopy(got), want) {
		t.Fatalf("links: %v, want %v", sortedCopy(got), want)
	}
}

// Mirrors test_extract_sub_links_exclude.
func TestExtractSubLinksExclude(t *testing.T) {
	html := extractTestHTML + `<a href="alexis.html"</a>`
	got, err := ExtractSubLinks(html, "https://foobar.com/hello/bill.html", ExtractSubLinksOptions{
		BaseURL:         "https://foobar.com",
		AllowOutside:    true,
		ExcludePrefixes: []string{"https://foobar.com/how", "http://baz.org"},
	})
	if err != nil {
		t.Fatalf("ExtractSubLinks: %v", err)
	}
	want := []string{
		"http://baz.net",
		"https://foobar.com",
		"https://foobar.com/hello",
		"https://foobar.com/hello/alexis.html",
	}
	if !reflect.DeepEqual(sortedCopy(got), want) {
		t.Fatalf("links: %v, want %v", sortedCopy(got), want)
	}
}

// Mirrors test_prevent_outside: netloc and scheme changes are outside.
func TestExtractSubLinksPreventOutside(t *testing.T) {
	html := `<a href="https://foobar.comic.com">BAD</a>` +
		`<a href="https://foobar.comic:9999">BAD</a>` +
		`<a href="https://foobar.com:9999">BAD</a>` +
		`<a href="http://foobar.com:9999/">BAD</a>` +
		`<a href="https://foobar.com/OK">OK</a>` +
		`<a href="http://foobar.com/BAD">BAD</a>`
	got, err := ExtractSubLinks(html, "https://foobar.com/hello/bill.html", ExtractSubLinksOptions{
		BaseURL: "https://foobar.com",
	})
	if err != nil {
		t.Fatalf("ExtractSubLinks: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"https://foobar.com/OK"}) {
		t.Fatalf("links: %v", got)
	}
}

// Mirrors test_extract_sub_links_with_query.
func TestExtractSubLinksWithQuery(t *testing.T) {
	html := `<a href="https://foobar.com?query=123">one</a>` +
		`<a href="/hello?query=456">two</a>` +
		`<a href="//foobar.com/how/are/you?query=789">three</a>` +
		`<a href="doing?query=101112"></a>`
	got, err := ExtractSubLinks(html, "https://foobar.com/hello/bill.html", ExtractSubLinksOptions{
		BaseURL: "https://foobar.com",
	})
	if err != nil {
		t.Fatalf("ExtractSubLinks: %v", err)
	}
	want := []string{
		"https://foobar.com/hello/doing?query=101112",
		"https://foobar.com/hello?query=456",
		"https://foobar.com/how/are/you?query=789",
		"https://foobar.com?query=123",
	}
	if !reflect.DeepEqual(sortedCopy(got), want) {
		t.Fatalf("links: %v, want %v", sortedCopy(got), want)
	}
}

// Unparseable links raise by default and are skipped with
// ContinueOnFailure (Python's continue_on_failure).
func TestExtractSubLinksFailureHandling(t *testing.T) {
	html := `href="http://foo bar/baz"`
	if _, err := ExtractSubLinks(html, "https://foobar.com", ExtractSubLinksOptions{}); err == nil {
		t.Fatal("expected error for unparseable link")
	}
	got, err := ExtractSubLinks(html, "https://foobar.com", ExtractSubLinksOptions{ContinueOnFailure: true})
	if err != nil {
		t.Fatalf("ExtractSubLinks: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("links: %v, want none", got)
	}
}

// An unparseable url/base URL surfaces an error.
func TestExtractSubLinksInvalidURL(t *testing.T) {
	if _, err := ExtractSubLinks("", "http://foo bar", ExtractSubLinksOptions{}); err == nil {
		t.Fatal("expected error for invalid url")
	}
	if _, err := ExtractSubLinks("", "https://foobar.com", ExtractSubLinksOptions{BaseURL: "http://foo bar"}); err == nil {
		t.Fatal("expected error for invalid base URL")
	}
}

// With a valid BaseURL and an invalid rawURL, the raw-URL parse error
// surfaces (base parses first, rawURL second).
func TestExtractSubLinksInvalidRawURLWithValidBase(t *testing.T) {
	_, err := ExtractSubLinks("", "http://foo bar", ExtractSubLinksOptions{BaseURL: "https://foobar.com"})
	if err == nil {
		t.Fatal("expected error for invalid raw url")
	}
}
