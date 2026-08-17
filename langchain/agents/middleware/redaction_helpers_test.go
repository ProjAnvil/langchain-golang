package middleware

import (
	"errors"
	"strings"
	"testing"
)

func TestPIIDetectionErrorMessage(t *testing.T) {
	err := PIIDetectionError{PIIType: "email", Matches: []PIIMatch{{Value: "a@b.com"}, {Value: "c@d.com"}}}
	if got := err.Error(); got != "Detected 2 instance(s) of email in text content" {
		t.Fatalf("error message mismatch: %q", got)
	}
}

func TestDetectIPFiltersInvalidAddresses(t *testing.T) {
	matches := DetectIP("ping 10.0.0.1 and 999.1.2.3")
	if len(matches) != 1 || matches[0].Value != "10.0.0.1" {
		t.Fatalf("ip matches: %#v", matches)
	}
}

func TestDetectMACAddress(t *testing.T) {
	matches := DetectMACAddress("mac aa:bb:cc:dd:ee:ff here")
	if len(matches) != 1 || matches[0].Value != "aa:bb:cc:dd:ee:ff" {
		t.Fatalf("mac matches: %#v", matches)
	}
	if got := DetectMACAddress("not a mac zz:11"); len(got) != 0 {
		t.Fatalf("expected no mac matches, got %#v", got)
	}
}

func TestDetectURLVariants(t *testing.T) {
	matches := DetectURL("visit https://example.com/page and www.foo.org plus bare.com/path end")
	values := []string{}
	for _, match := range matches {
		values = append(values, match.Value)
	}
	joined := strings.Join(values, ",")
	if !strings.Contains(joined, "https://example.com/page") {
		t.Fatalf("missing http url: %#v", values)
	}
	if !strings.Contains(joined, "www.foo.org") {
		t.Fatalf("missing www url: %#v", values)
	}
	if !strings.Contains(joined, "bare.com/path") {
		t.Fatalf("missing bare url with path: %#v", values)
	}
	// A bare domain without www. prefix and without a path is not a URL.
	if strings.Contains(joined, "end") {
		t.Fatalf("unexpected match: %#v", values)
	}
	// Matches must be sorted by start offset.
	for i := 1; i < len(matches); i++ {
		if matches[i-1].Start >= matches[i].Start {
			t.Fatalf("matches not sorted: %#v", matches)
		}
	}
}

func TestDetectURLSkipsDuplicatesOverlappingHTTPMatches(t *testing.T) {
	matches := DetectURL("https://example.com")
	if len(matches) != 1 {
		t.Fatalf("expected the bare matcher to skip the overlapping http match, got %#v", matches)
	}
}

func TestDetectURLNoMatches(t *testing.T) {
	if got := DetectURL("plain text without links"); len(got) != 0 {
		t.Fatalf("expected no url matches, got %#v", got)
	}
}

func TestApplyRedactionStrategyEdgeCases(t *testing.T) {
	content, err := ApplyRedactionStrategy("unchanged", nil, RedactionRedact)
	if err != nil || content != "unchanged" {
		t.Fatalf("empty matches should return content unchanged: %q, %v", content, err)
	}

	matches := []PIIMatch{{Type: "email", Value: "a@b.com", Start: 0, End: 7}}
	if _, err := ApplyRedactionStrategy("a@b.com", matches, RedactionBlock); err == nil {
		t.Fatal("expected block strategy to return PIIDetectionError")
	} else {
		var piiErr PIIDetectionError
		if !errors.As(err, &piiErr) {
			t.Fatalf("expected PIIDetectionError, got %v", err)
		}
	}

	if _, err := ApplyRedactionStrategy("a@b.com", matches, RedactionStrategy("bogus")); err == nil ||
		!strings.Contains(err.Error(), "unknown redaction strategy") {
		t.Fatalf("expected unknown strategy error, got %v", err)
	}
}

func TestRedactionRuleResolveWithPattern(t *testing.T) {
	rule, err := (RedactionRule{PIIType: "ticket", Pattern: `TICKET-\d+`}).Resolve()
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if rule.Strategy != RedactionRedact {
		t.Fatalf("default strategy mismatch: %q", rule.Strategy)
	}
	updated, matches, err := rule.Apply("see TICKET-42 now")
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if len(matches) != 1 || matches[0].Type != "ticket" || matches[0].Value != "TICKET-42" {
		t.Fatalf("pattern matches: %#v", matches)
	}
	if updated != "see [REDACTED_TICKET] now" {
		t.Fatalf("updated mismatch: %q", updated)
	}
}

func TestRedactionRuleResolveInvalidPattern(t *testing.T) {
	if _, err := (RedactionRule{PIIType: "ticket", Pattern: "["}).Resolve(); err == nil {
		t.Fatal("expected invalid pattern error")
	}
}

func TestRedactionRuleResolveUnknownType(t *testing.T) {
	if _, err := (RedactionRule{PIIType: "not-a-builtin"}).Resolve(); err == nil ||
		!strings.Contains(err.Error(), "unknown PII type") {
		t.Fatalf("expected unknown PII type error, got %v", err)
	}
}

func TestResolvedRedactionRuleApplyNoMatches(t *testing.T) {
	rule, err := (RedactionRule{PIIType: "email"}).Resolve()
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	updated, matches, err := rule.Apply("nothing sensitive")
	if err != nil || updated != "nothing sensitive" || matches != nil {
		t.Fatalf("no-match apply mismatch: %q %#v %v", updated, matches, err)
	}
}

func TestPassesLuhnLengthConstraints(t *testing.T) {
	if passesLuhn("411111") {
		t.Fatal("expected too-short number to fail luhn")
	}
	if !passesLuhn("4111-1111-1111-1111") {
		t.Fatal("expected valid card number to pass luhn")
	}
	if passesLuhn("4111-1111-1111-1112") {
		t.Fatal("expected invalid card number to fail luhn")
	}
}

func TestMaskMatchVariants(t *testing.T) {
	cases := []struct {
		name  string
		match PIIMatch
		want  string
	}{
		{"email standard", PIIMatch{Type: "email", Value: "user@example.com"}, "user@****.com"},
		{"email no tld dot", PIIMatch{Type: "email", Value: "user@localhost"}, "user@****"},
		{"email malformed", PIIMatch{Type: "email", Value: "noatsign"}, "****sign"},
		{"credit card dashes", PIIMatch{Type: "credit_card", Value: "4111-1111-1111-1111"}, "****-****-****-1111"},
		{"credit card spaces", PIIMatch{Type: "credit_card", Value: "4111 1111 1111 1111"}, "**** **** **** 1111"},
		{"credit card plain", PIIMatch{Type: "credit_card", Value: "4111111111111111"}, "************1111"},
		{"ip", PIIMatch{Type: "ip", Value: "10.0.0.7"}, "*.*.*.7"},
		{"ip malformed", PIIMatch{Type: "ip", Value: "10.0"}, "****"},
		{"mac colons", PIIMatch{Type: "mac_address", Value: "aa:bb:cc:dd:ee:ff"}, "**:**:**:**:**:ff"},
		{"mac dashes", PIIMatch{Type: "mac_address", Value: "aa-bb-cc-dd-ee-ff"}, "**-**-**-**-**-ff"},
		{"url", PIIMatch{Type: "url", Value: "https://example.com"}, "[MASKED_URL]"},
		{"generic long", PIIMatch{Type: "custom", Value: "secretvalue"}, "****alue"},
		{"generic short", PIIMatch{Type: "custom", Value: "abc"}, "****"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := maskMatch(tt.match); got != tt.want {
				t.Fatalf("maskMatch(%#v) = %q, want %q", tt.match, got, tt.want)
			}
		})
	}
}

func TestLastN(t *testing.T) {
	if got := lastN("ab", 4); got != "ab" {
		t.Fatalf("lastN short value: %q", got)
	}
	if got := lastN("abcdef", 3); got != "def" {
		t.Fatalf("lastN long value: %q", got)
	}
}

func TestReplaceMatchesMultiple(t *testing.T) {
	matches := []PIIMatch{
		{Type: "email", Value: "a@x.com", Start: 0, End: 7},
		{Type: "email", Value: "b@y.com", Start: 12, End: 19},
	}
	got := replaceMatches("a@x.com and b@y.com", matches, func(m PIIMatch) string {
		return "[" + m.Value + "]"
	})
	if got != "[a@x.com] and [b@y.com]" {
		t.Fatalf("replaceMatches mismatch: %q", got)
	}
}
