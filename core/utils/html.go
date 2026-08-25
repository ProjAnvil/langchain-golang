package utils

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

// PrefixesToIgnore mirrors langchain_core.utils.html.PREFIXES_TO_IGNORE.
var PrefixesToIgnore = []string{"javascript:", "mailto:", "#"}

// SuffixesToIgnore mirrors langchain_core.utils.html.SUFFIXES_TO_IGNORE.
var SuffixesToIgnore = []string{
	".css", ".js", ".ico", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".csv",
	".bz2", ".zip", ".epub", ".webp", ".pdf", ".docx", ".xlsx", ".pptx", ".pptm",
}

// defaultLinkRegexp captures href targets up to the first terminator
// (#, ', "). Port note: Python's DEFAULT_LINK_REGEX uses negative lookaheads
// for the prefix/suffix exclusions, which Go's RE2 engine does not support.
// The exclusions are applied as post-match filters in FindAllLinks instead,
// which is equivalent once empty captures are dropped: terminator chars
// never appear inside a capture, so an ignored suffix only matters at the
// capture's end, and an ignored prefix only at its start.
var defaultLinkRegexp = regexp.MustCompile(`href=["']([^"'#]*)["'#]`)

// FindAllLinks extracts unique links from raw HTML, mirroring Python's
// find_all_links (utils/html.py:46). A nil pattern uses the default matcher
// (with the prefix/suffix exclusions); a custom pattern replaces it and must
// capture the link in capture group 1. Links are deduplicated and returned
// in first-occurrence order (Python returns list(set(...)), whose order is
// unspecified).
func FindAllLinks(rawHTML string, pattern *regexp.Regexp) []string {
	var raw []string
	if pattern != nil {
		for _, match := range pattern.FindAllStringSubmatch(rawHTML, -1) {
			if len(match) > 1 {
				raw = append(raw, match[1])
			}
		}
	} else {
		for _, match := range defaultLinkRegexp.FindAllStringSubmatch(rawHTML, -1) {
			link := match[1]
			// An href ending immediately at '#' (e.g. href="#foobar") captures
			// the empty string, which would pass the filters below; Python's
			// lookahead pattern never matches those and returns [], so drop
			// empty captures explicitly.
			if link == "" {
				continue
			}
			if hasAnyPrefix(link, PrefixesToIgnore) || hasAnySuffix(link, SuffixesToIgnore) {
				continue
			}
			raw = append(raw, link)
		}
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(raw))
	for _, link := range raw {
		if !seen[link] {
			seen[link] = true
			out = append(out, link)
		}
	}
	return out
}

// ExtractSubLinksOptions configures ExtractSubLinks.
type ExtractSubLinksOptions struct {
	// BaseURL checks outside links against this URL; defaults to the page url
	// (Python base_url=None).
	BaseURL string
	// Pattern overrides the default link matcher (Python pattern=).
	Pattern *regexp.Regexp
	// AllowOutside keeps external links. It is the inverse of Python's
	// prevent_outside=True default (Go has no non-zero-value option defaults).
	AllowOutside bool
	// ExcludePrefixes drops URLs starting with any of these prefixes.
	ExcludePrefixes []string
	// ContinueOnFailure skips links that fail to parse instead of erroring
	// (Python continue_on_failure).
	ContinueOnFailure bool
}

// ExtractSubLinks extracts all links from raw HTML and converts them to
// absolute URLs, mirroring Python's extract_sub_links (utils/html.py:62).
// Absolute http(s) links pass through; scheme-relative links adopt the page
// scheme; relative links resolve against the page url (Python urljoin).
// Unless AllowOutside is set, links whose host differs from the base URL's
// or that are not under the base URL prefix are dropped.
func ExtractSubLinks(rawHTML string, rawURL string, options ExtractSubLinksOptions) ([]string, error) {
	baseURLToUse := options.BaseURL
	if baseURLToUse == "" {
		baseURLToUse = rawURL
	}
	parsedBaseURL, err := url.Parse(baseURLToUse)
	if err != nil {
		return nil, fmt.Errorf("utils: extract sub links: parse base URL %q: %w", baseURLToUse, err)
	}
	parsedURL, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("utils: extract sub links: parse URL %q: %w", rawURL, err)
	}

	links := FindAllLinks(rawHTML, options.Pattern)
	seen := map[string]bool{}
	absolutePaths := []string{}
	for _, link := range links {
		absolutePath, err := resolveLink(parsedURL, link)
		if err != nil {
			if options.ContinueOnFailure {
				continue
			}
			return nil, fmt.Errorf("utils: extract sub links: parse link %q: %w", link, err)
		}
		if !seen[absolutePath] {
			seen[absolutePath] = true
			absolutePaths = append(absolutePaths, absolutePath)
		}
	}

	results := []string{}
	for _, path := range absolutePaths {
		if hasAnyPrefix(path, options.ExcludePrefixes) {
			continue
		}
		if !options.AllowOutside {
			parsedPath, err := url.Parse(path)
			if err != nil {
				if options.ContinueOnFailure {
					continue
				}
				return nil, fmt.Errorf("utils: extract sub links: parse path %q: %w", path, err)
			}
			if parsedBaseURL.Host != parsedPath.Host {
				continue
			}
			// Verifies the rest of the path after the host when the base URL
			// is more specific (Python's startswith check).
			if !strings.HasPrefix(path, baseURLToUse) {
				continue
			}
		}
		results = append(results, path)
	}
	return results, nil
}

// resolveLink converts one extracted link to an absolute URL
// (utils/html.py:93-106). Fragments never occur: the default matcher stops at
// '#'.
func resolveLink(base *url.URL, link string) (string, error) {
	parsedLink, err := url.Parse(link)
	if err != nil {
		return "", err
	}
	// Absolute links like https://to/path.
	if parsedLink.Scheme == "http" || parsedLink.Scheme == "https" {
		return link, nil
	}
	// Protocol-relative links like //to/path.
	if strings.HasPrefix(link, "//") {
		return base.Scheme + ":" + link, nil
	}
	return base.ResolveReference(parsedLink).String(), nil
}

func hasAnyPrefix(value string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(value, prefix) {
			return true
		}
	}
	return false
}

func hasAnySuffix(value string, suffixes []string) bool {
	for _, suffix := range suffixes {
		if strings.HasSuffix(value, suffix) {
			return true
		}
	}
	return false
}
