package sqltoolkit

import (
	"strings"
	"testing"
)

// TestAnalyzeSQLDollarQuotedStrings exercises the postgres-only $tag$...$tag$
// lexer path: contents (including semicolons and quotes) are preserved
// verbatim for execution but blanked out of the keyword-analysis view.
func TestAnalyzeSQLDollarQuotedStrings(t *testing.T) {
	cases := []struct {
		name  string
		query string
		// wantStatements are the expected non-empty statements after splitting.
		wantStatements []string
		// wantPlain is the keyword-analysis view (literals blanked).
		wantPlain string
	}{
		{
			name:           "empty tag dollar quote keeps semicolon literal",
			query:          "SELECT $$a;b$$",
			wantStatements: []string{"SELECT $$a;b$$"},
			wantPlain:      "SELECT  ",
		},
		{
			name:           "named tag round trips",
			query:          "SELECT $tag$he said " + "`hi`" + "$tag$",
			wantStatements: []string{"SELECT $tag$he said `hi`$tag$"},
			wantPlain:      "SELECT  ",
		},
		{
			name:           "underscore and digits allowed after first tag char",
			query:          "SELECT $_a1$xy$_a1$",
			wantStatements: []string{"SELECT $_a1$xy$_a1$"},
			wantPlain:      "SELECT  ",
		},
		{
			name:           "semicolon inside dollar quote does not split",
			query:          "SELECT $q$a;b$q$; SELECT 2",
			wantStatements: []string{"SELECT $q$a;b$q$", "SELECT 2"},
			wantPlain:      "SELECT    SELECT 2",
		},
		{
			name:           "write keywords inside dollar quote are blanked",
			query:          "SELECT $q$DELETE FROM users$q$",
			wantStatements: []string{"SELECT $q$DELETE FROM users$q$"},
			wantPlain:      "SELECT  ",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			analyzed, err := analyzeSQL(tc.query, true)
			if err != nil {
				t.Fatalf("analyzeSQL(%q, true): %v", tc.query, err)
			}
			if strings.Join(analyzed.statements, "\x00") != strings.Join(tc.wantStatements, "\x00") {
				t.Errorf("statements = %q, want %q", analyzed.statements, tc.wantStatements)
			}
			if analyzed.plain != tc.wantPlain {
				t.Errorf("plain = %q, want %q", analyzed.plain, tc.wantPlain)
			}
		})
	}
}

// TestAnalyzeSQLDollarQuotedErrors covers the two unterminated dollar-quote
// failure modes: the opening tag never terminating in '$' (e.g. `$tag `) and
// the closing tag never appearing.
func TestAnalyzeSQLDollarQuotedErrors(t *testing.T) {
	for _, query := range []string{
		"SELECT $tag oops", // '$' followed by tag start, but no closing '$' before a non-tag char
		"SELECT $tag$abc",  // opening tag never closed
		"SELECT $$_",       // '$$' opener, '_' cannot close it, EOF
		"SELECT $$abc $x$", // no $$ closer ever appears in the remainder
	} {
		if _, err := analyzeSQL(query, true); err == nil {
			t.Errorf("analyzeSQL(%q, true) succeeded, want unterminated dollar-quoted error", query)
		} else if !strings.Contains(err.Error(), "unterminated dollar-quoted string") {
			t.Errorf("analyzeSQL(%q, true) error = %v, want dollar-quote error", query, err)
		}
	}
}

// TestAnalyzeSQLPositionalAndNamedParameters proves that with dollarQuotes
// disabled (sqlite) '$' is copied through verbatim so $name parameters keep
// working, and even with dollarQuotes enabled positional $1 stays literal
// because a digit is not a valid tag start.
func TestAnalyzeSQLPositionalAndNamedParameters(t *testing.T) {
	for _, dollarQuotes := range []bool{false, true} {
		analyzed, err := analyzeSQL("SELECT $1", dollarQuotes)
		if err != nil {
			t.Fatalf("analyzeSQL($1, %v): %v", dollarQuotes, err)
		}
		if len(analyzed.statements) != 1 || analyzed.statements[0] != "SELECT $1" {
			t.Errorf("dollarQuotes=%v statements = %q, want [SELECT $1]", dollarQuotes, analyzed.statements)
		}
		if analyzed.plain != "SELECT $1" {
			t.Errorf("dollarQuotes=%v plain = %q, want %q", dollarQuotes, analyzed.plain, "SELECT $1")
		}
	}

	// sqlite mode: a named parameter like $name never opens a dollar quote.
	analyzed, err := analyzeSQL("SELECT name FROM users WHERE id = $id", false)
	if err != nil {
		t.Fatalf("analyzeSQL named param: %v", err)
	}
	if !strings.Contains(analyzed.plain, "$id") {
		t.Errorf("plain = %q, want $id preserved", analyzed.plain)
	}
}

// TestAnalyzeSQLUnterminatedQuotedTokens covers the error returns for
// unterminated string literals, quoted identifiers, and backtick identifiers.
func TestAnalyzeSQLUnterminatedQuotedTokens(t *testing.T) {
	cases := []struct {
		query string
		want  string
	}{
		{"SELECT 'oops", "unterminated string literal"},
		{`SELECT "oops`, "unterminated quoted identifier"},
		{"SELECT `oops", "unterminated backtick-quoted identifier"},
	}
	for _, tc := range cases {
		if _, err := analyzeSQL(tc.query, false); err == nil {
			t.Errorf("analyzeSQL(%q) succeeded, want %q", tc.query, tc.want)
		} else if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("analyzeSQL(%q) error = %v, want %q", tc.query, err, tc.want)
		}
	}
}

// TestAnalyzeSQLDoubledQuoteEscapes covers scanQuoted's doubled-quote escape:
// ” inside a string literal and "" inside a quoted identifier.
func TestAnalyzeSQLDoubledQuoteEscapes(t *testing.T) {
	analyzed, err := analyzeSQL("SELECT 'it''s'; SELECT 2", false)
	if err != nil {
		t.Fatalf("analyzeSQL escaped single quotes: %v", err)
	}
	if len(analyzed.statements) != 2 || analyzed.statements[0] != "SELECT 'it''s'" {
		t.Errorf("statements = %q, want split outside the escaped literal", analyzed.statements)
	}

	analyzed, err = analyzeSQL(`SELECT "we""ird" FROM users`, false)
	if err != nil {
		t.Fatalf("analyzeSQL escaped double quotes: %v", err)
	}
	if analyzed.statements[0] != `SELECT "we""ird" FROM users` {
		t.Errorf("statements = %q, want identifier preserved verbatim", analyzed.statements)
	}
	if strings.Contains(analyzed.plain, "ird") {
		t.Errorf("plain = %q, want the quoted identifier blanked", analyzed.plain)
	}
}

// TestAnalyzeSQLNestedBlockComments covers the depth-tracking scanner:
// nested opens, characters that merely look like closers, and an outer
// comment left open by a closed inner one.
func TestAnalyzeSQLNestedBlockComments(t *testing.T) {
	cases := []struct {
		name  string
		query string
		// wantPlain is the keyword-analysis view after comment removal.
		wantPlain string
	}{
		{
			name:      "nested comment fully closed",
			query:     "SELECT /* a /* b */ c */ 1",
			wantPlain: "SELECT   1",
		},
		{
			name:      "star slash with space does not close",
			query:     "SELECT /* a * / b */ 1",
			wantPlain: "SELECT   1",
		},
		{
			name:      "slash star inside line comment is inert",
			query:     "SELECT 1 -- /* not open\n+ 2",
			wantPlain: "SELECT 1  \n+ 2",
		},
		{
			name:      "empty block comment",
			query:     "SELECT/**/1",
			wantPlain: "SELECT 1",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			analyzed, err := analyzeSQL(tc.query, false)
			if err != nil {
				t.Fatalf("analyzeSQL(%q): %v", tc.query, err)
			}
			if analyzed.plain != tc.wantPlain {
				t.Errorf("plain = %q, want %q", analyzed.plain, tc.wantPlain)
			}
		})
	}

	// The inner close balances only one level: the outer comment stays open.
	if _, err := analyzeSQL("SELECT /* a /* b */", false); err == nil {
		t.Error("analyzeSQL(outer comment left open) succeeded, want unterminated block comment")
	} else if !strings.Contains(err.Error(), "unterminated block comment") {
		t.Errorf("error = %v, want unterminated block comment", err)
	}
}

// TestGuardReadOnlyPostgresFlavors drives guardReadOnly's postgres flavor
// directly (no server needed): dollar-quoted and E'...' string literals must
// not trip the forbidden-keyword or multi-statement guards, and lexer errors
// must surface wrapped in the guardrail message.
func TestGuardReadOnlyPostgresFlavors(t *testing.T) {
	t.Run("dollar quoted write keyword passes", func(t *testing.T) {
		statement, err := guardReadOnly(Postgres, "SELECT $$DELETE FROM users$$ AS payload")
		if err != nil {
			t.Fatalf("guardReadOnly: %v", err)
		}
		if statement != "SELECT $$DELETE FROM users$$ AS payload" {
			t.Errorf("statement = %q, want verbatim", statement)
		}
	})

	t.Run("E escape string literal passes", func(t *testing.T) {
		// postgres E'...' strings: the E is an ordinary token to the lexer and
		// the quoted body is blanked before keyword analysis.
		statement, err := guardReadOnly(Postgres, `SELECT E'a\nb' AS s`)
		if err != nil {
			t.Fatalf("guardReadOnly: %v", err)
		}
		if statement != `SELECT E'a\nb' AS s` {
			t.Errorf("statement = %q, want verbatim", statement)
		}
	})

	t.Run("write keyword in E string passes", func(t *testing.T) {
		if _, err := guardReadOnly(Postgres, `SELECT E'INTO' AS s`); err != nil {
			t.Errorf("write keyword inside E-string rejected: %v", err)
		}
	})

	t.Run("positional parameter passes", func(t *testing.T) {
		statement, err := guardReadOnly(Postgres, "SELECT * FROM users WHERE id = $1")
		if err != nil {
			t.Fatalf("guardReadOnly: %v", err)
		}
		if statement != "SELECT * FROM users WHERE id = $1" {
			t.Errorf("statement = %q, want $1 preserved", statement)
		}
	})

	t.Run("unterminated dollar quote is wrapped", func(t *testing.T) {
		_, err := guardReadOnly(Postgres, "SELECT $tag$")
		if err == nil {
			t.Fatal("guardReadOnly accepted an unterminated dollar quote")
		}
		if !strings.Contains(err.Error(), "read-only guardrail rejected the query") ||
			!strings.Contains(err.Error(), "unterminated dollar-quoted string") {
			t.Errorf("error = %v, want wrapped guardrail error", err)
		}
	})

	t.Run("unterminated quoted identifier is wrapped", func(t *testing.T) {
		_, err := guardReadOnly(SQLite, `SELECT "oops`)
		if err == nil {
			t.Fatal("guardReadOnly accepted an unterminated quoted identifier")
		}
		if !strings.Contains(err.Error(), "unterminated quoted identifier") {
			t.Errorf("error = %v, want unterminated quoted identifier", err)
		}
	})
}

// TestLeadingKeyword checks the keyword extractor used for the leading
// SELECT/WITH requirement, including mixed case and non-letter leaders.
func TestLeadingKeyword(t *testing.T) {
	cases := []struct{ input, want string }{
		{"  SELECT 1", "select"},
		{"with x as (select 1)", "with"},
		{"SeLeCt", "select"},
		{"(select 1)", "select"}, // non-letter leader is skipped until letters start
		{"123abc", "abc"},        // digits never join the run, letters after them do
		{"", ""},
		{"\t\n  drop table", "drop"},
	}
	for _, tc := range cases {
		if got := leadingKeyword(tc.input); got != tc.want {
			t.Errorf("leadingKeyword(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
