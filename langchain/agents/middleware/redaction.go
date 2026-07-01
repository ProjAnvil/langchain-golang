package middleware

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net"
	"net/url"
	"regexp"
	"sort"
	"strings"
)

type RedactionStrategy string

const (
	RedactionBlock  RedactionStrategy = "block"
	RedactionRedact RedactionStrategy = "redact"
	RedactionMask   RedactionStrategy = "mask"
	RedactionHash   RedactionStrategy = "hash"
)

type PIIMatch struct {
	Type  string
	Value string
	Start int
	End   int
}

type PIIDetectionError struct {
	PIIType string
	Matches []PIIMatch
}

func (e PIIDetectionError) Error() string {
	return fmt.Sprintf("Detected %d instance(s) of %s in text content", len(e.Matches), e.PIIType)
}

type Detector func(string) []PIIMatch

var BuiltinDetectors = map[string]Detector{
	"email":       DetectEmail,
	"credit_card": DetectCreditCard,
	"ip":          DetectIP,
	"mac_address": DetectMACAddress,
	"url":         DetectURL,
}

func DetectEmail(content string) []PIIMatch {
	return regexMatches("email", `\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`, content)
}

func DetectCreditCard(content string) []PIIMatch {
	matches := regexMatches("credit_card", `\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b`, content)
	out := matches[:0]
	for _, match := range matches {
		if passesLuhn(match.Value) {
			out = append(out, match)
		}
	}
	return out
}

func DetectIP(content string) []PIIMatch {
	candidates := regexMatches("ip", `\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b`, content)
	out := candidates[:0]
	for _, match := range candidates {
		if parsed := net.ParseIP(match.Value); parsed != nil {
			out = append(out, match)
		}
	}
	return out
}

func DetectMACAddress(content string) []PIIMatch {
	return regexMatches("mac_address", `\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b`, content)
}

func DetectURL(content string) []PIIMatch {
	out := []PIIMatch{}
	for _, match := range regexMatches("url", `https?://[^\s<>"{}|\\^`+"`"+`\[\]]+`, content) {
		parsed, err := url.Parse(match.Value)
		if err == nil && (parsed.Scheme == "http" || parsed.Scheme == "https") && parsed.Host != "" {
			out = append(out, match)
		}
	}
	bare := regexMatches("url", `\b(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?:/[^\s]*)?`, content)
	for _, match := range bare {
		if overlapsAny(match, out) {
			continue
		}
		if !strings.HasPrefix(match.Value, "www.") && !strings.Contains(match.Value, "/") {
			continue
		}
		parsed, err := url.Parse("http://" + match.Value)
		if err == nil && parsed.Host != "" && strings.Contains(parsed.Host, ".") {
			out = append(out, match)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Start < out[j].Start })
	return out
}

func ApplyRedactionStrategy(content string, matches []PIIMatch, strategy RedactionStrategy) (string, error) {
	if len(matches) == 0 {
		return content, nil
	}
	switch strategy {
	case RedactionRedact:
		return replaceMatches(content, matches, func(match PIIMatch) string {
			return "[REDACTED_" + strings.ToUpper(match.Type) + "]"
		}), nil
	case RedactionMask:
		return replaceMatches(content, matches, maskMatch), nil
	case RedactionHash:
		return replaceMatches(content, matches, func(match PIIMatch) string {
			sum := sha256.Sum256([]byte(match.Value))
			return fmt.Sprintf("<%s_hash:%s>", match.Type, hex.EncodeToString(sum[:])[:8])
		}), nil
	case RedactionBlock:
		return "", PIIDetectionError{PIIType: matches[0].Type, Matches: matches}
	default:
		return "", fmt.Errorf("unknown redaction strategy: %s", strategy)
	}
}

type RedactionRule struct {
	PIIType  string
	Strategy RedactionStrategy
	Detector Detector
	Pattern  string
}

type ResolvedRedactionRule struct {
	PIIType  string
	Strategy RedactionStrategy
	Detector Detector
}

func (r RedactionRule) Resolve() (ResolvedRedactionRule, error) {
	strategy := r.Strategy
	if strategy == "" {
		strategy = RedactionRedact
	}
	detector := r.Detector
	if detector == nil && r.Pattern != "" {
		compiled, err := regexp.Compile(r.Pattern)
		if err != nil {
			return ResolvedRedactionRule{}, err
		}
		detector = func(content string) []PIIMatch {
			out := []PIIMatch{}
			for _, loc := range compiled.FindAllStringIndex(content, -1) {
				out = append(out, PIIMatch{Type: r.PIIType, Value: content[loc[0]:loc[1]], Start: loc[0], End: loc[1]})
			}
			return out
		}
	}
	if detector == nil {
		var ok bool
		detector, ok = BuiltinDetectors[r.PIIType]
		if !ok {
			return ResolvedRedactionRule{}, fmt.Errorf("unknown PII type: %s", r.PIIType)
		}
	}
	return ResolvedRedactionRule{PIIType: r.PIIType, Strategy: strategy, Detector: detector}, nil
}

func (r ResolvedRedactionRule) Apply(content string) (string, []PIIMatch, error) {
	matches := r.Detector(content)
	if len(matches) == 0 {
		return content, nil, nil
	}
	updated, err := ApplyRedactionStrategy(content, matches, r.Strategy)
	return updated, matches, err
}

func regexMatches(kind, pattern, content string) []PIIMatch {
	re := regexp.MustCompile(pattern)
	out := []PIIMatch{}
	for _, loc := range re.FindAllStringIndex(content, -1) {
		out = append(out, PIIMatch{Type: kind, Value: content[loc[0]:loc[1]], Start: loc[0], End: loc[1]})
	}
	return out
}

func passesLuhn(cardNumber string) bool {
	digits := []int{}
	for _, r := range cardNumber {
		if r >= '0' && r <= '9' {
			digits = append(digits, int(r-'0'))
		}
	}
	if len(digits) < 13 || len(digits) > 19 {
		return false
	}
	checksum := 0
	for i := len(digits) - 1; i >= 0; i-- {
		value := digits[i]
		if (len(digits)-1-i)%2 == 1 {
			value *= 2
			if value > 9 {
				value -= 9
			}
		}
		checksum += value
	}
	return checksum%10 == 0
}

func replaceMatches(content string, matches []PIIMatch, replacement func(PIIMatch) string) string {
	sort.Slice(matches, func(i, j int) bool { return matches[i].Start > matches[j].Start })
	out := content
	for _, match := range matches {
		out = out[:match.Start] + replacement(match) + out[match.End:]
	}
	return out
}

func maskMatch(match PIIMatch) string {
	switch match.Type {
	case "email":
		parts := strings.Split(match.Value, "@")
		if len(parts) == 2 {
			domainParts := strings.Split(parts[1], ".")
			if len(domainParts) > 1 {
				return parts[0] + "@****." + domainParts[len(domainParts)-1]
			}
			return parts[0] + "@****"
		}
	case "credit_card":
		digits := onlyDigits(match.Value)
		sep := ""
		if strings.Contains(match.Value, "-") {
			sep = "-"
		} else if strings.Contains(match.Value, " ") {
			sep = " "
		}
		last := lastN(digits, 4)
		if sep != "" {
			return "****" + sep + "****" + sep + "****" + sep + last
		}
		return "************" + last
	case "ip":
		parts := strings.Split(match.Value, ".")
		if len(parts) == 4 {
			return "*.*.*." + parts[3]
		}
	case "mac_address":
		sep := ":"
		if strings.Contains(match.Value, "-") {
			sep = "-"
		}
		return "**" + sep + "**" + sep + "**" + sep + "**" + sep + "**" + sep + lastN(match.Value, 2)
	case "url":
		return "[MASKED_URL]"
	}
	if len(match.Value) > 4 {
		return "****" + lastN(match.Value, 4)
	}
	return "****"
}

func onlyDigits(value string) string {
	var b strings.Builder
	for _, r := range value {
		if r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func lastN(value string, n int) string {
	if len(value) <= n {
		return value
	}
	return value[len(value)-n:]
}

func overlapsAny(match PIIMatch, matches []PIIMatch) bool {
	for _, existing := range matches {
		if (existing.Start <= match.Start && match.Start < existing.End) ||
			(existing.Start < match.End && match.End <= existing.End) {
			return true
		}
	}
	return false
}
