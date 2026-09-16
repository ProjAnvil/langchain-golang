package sqltoolkit

import (
	"fmt"
	"regexp"
	"strings"
)

// analyzedSQL carries the views the read-only guardrails need after comments
// have been stripped from the raw query.
type analyzedSQL struct {
	// statements are the non-empty semicolon-separated statements found in
	// the comment-stripped query. The guardrail requires exactly one, and
	// statements[0] is what gets executed.
	statements []string
	// plain is the query with comments, string literals, dollar-quoted
	// strings, and quoted identifiers blanked out. Keyword analysis (the
	// leading SELECT/WITH keyword and forbidden write keywords) runs against
	// this view so 'DELETE' or 'INTO' appearing as a value cannot trip the
	// guard, and quoted identifiers that merely contain keywords are not
	// misread as writes.
	plain string
}

// analyzeSQL strips SQL comments (replacing each with a single space) and
// splits the result into statements on semicolons that appear outside string
// literals, dollar-quoted strings, and quoted identifiers. Block comments may
// nest (a sqlite extension; treating them as nested is the safe superset for
// postgres too, since an unclosed comment is a syntax error there anyway).
//
// dollarQuotes enables postgres-style $tag$...$tag$ string handling. When it
// is false (sqlite), '$' is copied through so named parameters like $name
// keep working.
//
// It reports an error for unterminated string literals, quoted identifiers,
// dollar-quoted strings, or block comments.
func analyzeSQL(query string, dollarQuotes bool) (analyzedSQL, error) {
	var plain strings.Builder
	var current strings.Builder
	var statements []string

	flush := func() {
		if statement := strings.TrimSpace(current.String()); statement != "" {
			statements = append(statements, statement)
		}
		current.Reset()
	}

	for i := 0; i < len(query); i++ {
		c := query[i]
		switch {
		case c == '\'':
			// String literal; '' is an escaped quote. The literal passes
			// through to the executed statement verbatim but is blanked in
			// the keyword-analysis view.
			j, ok := scanQuoted(query, i, '\'')
			if !ok {
				return analyzedSQL{}, fmt.Errorf("unterminated string literal")
			}
			current.WriteString(query[i : j+1])
			plain.WriteByte(' ')
			i = j

		case c == '"':
			// Quoted identifier; "" is an escaped double quote.
			j, ok := scanQuoted(query, i, '"')
			if !ok {
				return analyzedSQL{}, fmt.Errorf("unterminated quoted identifier")
			}
			current.WriteString(query[i : j+1])
			plain.WriteByte(' ')
			i = j

		case c == '`':
			// Backtick-quoted identifier (sqlite extension).
			j := strings.IndexByte(query[i+1:], '`')
			if j < 0 {
				return analyzedSQL{}, fmt.Errorf("unterminated backtick-quoted identifier")
			}
			current.WriteString(query[i : i+j+2])
			plain.WriteByte(' ')
			i += j + 1

		case c == '$' && dollarQuotes && i+1 < len(query) && isDollarTagStart(query[i+1]):
			// Postgres dollar-quoted string: $tag$ ... $tag$ where the tag
			// follows unquoted-identifier rules (so positional parameters
			// like $1 never open a dollar quote). A bare '$' falls through
			// to the default case and is copied verbatim.
			end, err := scanDollarQuoted(query, i)
			if err != nil {
				return analyzedSQL{}, err
			}
			current.WriteString(query[i:end])
			plain.WriteByte(' ')
			i = end - 1

		case c == '-' && i+1 < len(query) && query[i+1] == '-':
			// Line comment: strip through end of line, keep the newline.
			j := strings.IndexByte(query[i:], '\n')
			end := len(query)
			if j >= 0 {
				end = i + j
			}
			current.WriteByte(' ')
			plain.WriteByte(' ')
			i = end - 1

		case c == '/' && i+1 < len(query) && query[i+1] == '*':
			// Nested-capable block comment.
			depth := 1
			j := i + 2
			for j < len(query) && depth > 0 {
				switch {
				case query[j] == '/' && j+1 < len(query) && query[j+1] == '*':
					depth++
					j += 2
				case query[j] == '*' && j+1 < len(query) && query[j+1] == '/':
					depth--
					j += 2
				default:
					j++
				}
			}
			if depth > 0 {
				return analyzedSQL{}, fmt.Errorf("unterminated block comment")
			}
			plain.WriteByte(' ')
			i = j - 1

		case c == ';':
			// Statement separator outside any literal.
			flush()
			plain.WriteByte(' ')

		default:
			current.WriteByte(c)
			plain.WriteByte(c)
		}
	}
	flush()

	return analyzedSQL{statements: statements, plain: plain.String()}, nil
}

// scanQuoted returns the index of the closing quote for the quoted token that
// starts at query[start], honoring the doubled-quote escape (” or ""), and
// reports whether a closer was found.
func scanQuoted(query string, start int, quote byte) (int, bool) {
	for j := start + 1; j < len(query); j++ {
		if query[j] != quote {
			continue
		}
		if j+1 < len(query) && query[j+1] == quote { // doubled quote escape
			j++
			continue
		}
		return j, true
	}
	return 0, false
}

// scanDollarQuoted returns the exclusive end index of the dollar-quoted string
// opening at query[start] (the caller has verified the '$' is followed by a
// tag-start character, so this really is a dollar-quote opener) or an error
// when the opening tag never closes.
func scanDollarQuoted(query string, start int) (int, error) {
	j := start + 1
	for j < len(query) && isDollarTagChar(query[j]) {
		j++
	}
	if j >= len(query) || query[j] != '$' {
		return 0, fmt.Errorf("unterminated dollar-quoted string")
	}
	tag := query[start : j+1]
	rest := strings.Index(query[j+1:], tag)
	if rest < 0 {
		return 0, fmt.Errorf("unterminated dollar-quoted string")
	}
	return j + 1 + rest + len(tag), nil
}

func isDollarTagStart(c byte) bool {
	return c == '_' || c == '$' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isDollarTagChar(c byte) bool {
	// Tag characters follow unquoted-identifier rules: no '$' (that closes
	// the opening delimiter) and no digits in the first position.
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

// forbiddenKeywordRe matches write-capable keywords that must not appear
// anywhere in a read-only SELECT/WITH statement: INTO covers SELECT ... INTO
// table (postgres), INSERT/REPLACE/MERGE ... INTO, and UPDATE/DELETE cover
// data-modifying CTEs (postgres WITH x AS (UPDATE/DELETE ... RETURNING ...))
// and SELECT ... FOR UPDATE row locks.
var forbiddenKeywordRe = regexp.MustCompile(`(?i)\b(INTO|UPDATE|DELETE)\b`)

// guardReadOnly enforces the toolkit's read-only contract on a raw query:
//
//  1. comments are stripped first, so they cannot disguise the leading
//     keyword or hide a second statement;
//  2. exactly one statement may remain (a single trailing semicolon is fine);
//  3. the statement must start with SELECT or WITH;
//  4. the statement body must not contain a write-capable keyword (INTO,
//     UPDATE, DELETE) outside string literals and quoted identifiers.
//
// It returns the validated statement ready for execution.
func guardReadOnly(dialect Dialect, query string) (string, error) {
	analyzed, err := analyzeSQL(query, dialect == Postgres)
	if err != nil {
		return "", fmt.Errorf("read-only guardrail rejected the query: %w", err)
	}
	if len(analyzed.statements) != 1 {
		return "", fmt.Errorf(
			"read-only guardrail rejected the query: expected exactly one SQL statement, found %d",
			len(analyzed.statements))
	}
	statement := analyzed.statements[0]

	keyword := leadingKeyword(analyzed.plain)
	if keyword != "select" && keyword != "with" {
		return "", fmt.Errorf(
			"read-only guardrail rejected the query: statement must start with SELECT or WITH, got %q",
			keyword)
	}
	if match := forbiddenKeywordRe.FindString(analyzed.plain); match != "" {
		return "", fmt.Errorf(
			"read-only guardrail rejected the query: read-only SELECT/WITH statements must not contain the keyword %q (SELECT ... INTO, data-modifying CTEs, and FOR UPDATE are rejected)",
			strings.ToUpper(match))
	}
	return statement, nil
}

// leadingKeyword returns the first run of ASCII letters in s, lowercased.
func leadingKeyword(s string) string {
	s = strings.TrimSpace(s)
	letters := make([]byte, 0, 8)
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'a' && c <= 'z':
			letters = append(letters, c)
		case c >= 'A' && c <= 'Z':
			letters = append(letters, c-'A'+'a')
		default:
			if len(letters) > 0 {
				return string(letters)
			}
		}
	}
	return string(letters)
}
