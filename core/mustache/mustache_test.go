package mustache

import (
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

func mustRender(t *testing.T, template string, data map[string]any) string {
	t.Helper()
	got, err := Render(template, data)
	if err != nil {
		t.Fatalf("Render(%q) returned error: %v", template, err)
	}
	return got
}

func renderWithPartials(t *testing.T, template string, data map[string]any, partials map[string]string) string {
	t.Helper()
	got, err := RenderWithPartials(template, data, partials)
	if err != nil {
		t.Fatalf("RenderWithPartials(%q) returned error: %v", template, err)
	}
	return got
}

func TestRenderVariables(t *testing.T) {
	cases := []struct {
		name     string
		template string
		data     map[string]any
		want     string
	}{
		{"plain", "Hello {{name}}!", map[string]any{"name": "Alice"}, "Hello Alice!"},
		{"missing is empty", "a{{missing}}b", nil, "ab"},
		{"nil data", "x{{name}}y", nil, "xy"},
		{"dotted name", "{{user.name}} is {{user.age}}", map[string]any{
			"user": map[string]any{"name": "Bob", "age": 42},
		}, "Bob is 42"},
		{"dotted missing leaf", "[{{user.nope}}]", map[string]any{"user": map[string]any{}}, "[]"},
		{"numeric value", "{{n}}", map[string]any{"n": 7}, "7"},
		{"zero renders", "{{n}}", map[string]any{"n": 0}, "0"},
		{"dot current scope", "{{#s}}{{.}}{{/s}}", map[string]any{"s": "hi"}, "hi"},
		{"list index", "{{items.1}}", map[string]any{"items": []any{"a", "b", "c"}}, "b"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := mustRender(t, tc.template, tc.data); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestRenderHTMLEscaping(t *testing.T) {
	if got := mustRender(t, "{{v}}", map[string]any{"v": `<>&"`}); got != "&lt;&gt;&amp;&quot;" {
		t.Fatalf("escaped = %q", got)
	}
	// Single quotes are intentionally NOT escaped, matching the reference.
	if got := mustRender(t, "{{v}}", map[string]any{"v": "'"}); got != "'" {
		t.Fatalf("single quote escaped unexpectedly: %q", got)
	}
	if got := mustRender(t, "{{v}}", map[string]any{"v": "a & b"}); got != "a &amp; b" {
		t.Fatalf("ampersand = %q", got)
	}
}

func TestRenderUnescaped(t *testing.T) {
	tests := []struct {
		name     string
		template string
	}{
		{"triple braces", "{{{html}}}"},
		{"ampersand", "{{&html}}"},
		{"ampersand with space", "{{& html}}"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := mustRender(t, tc.template, map[string]any{"html": "<b>\"&"})
			if got != `<b>"&` {
				t.Fatalf("got %q want unescaped", got)
			}
		})
	}
}

func TestRenderSections(t *testing.T) {
	tests := []struct {
		name     string
		template string
		data     map[string]any
		want     string
	}{
		{
			"list loop with dot",
			"{{#items}}[{{.}}]{{/items}}",
			map[string]any{"items": []any{"a", "b"}},
			"[a][b]",
		},
		{
			"list loop of maps",
			"{{#people}}{{name}} {{/people}}",
			map[string]any{"people": []any{
				map[string]any{"name": "Ann"},
				map[string]any{"name": "Bob"},
			}},
			"Ann Bob ",
		},
		{
			"empty list skipped",
			"A{{#items}}x{{/items}}B",
			map[string]any{"items": []any{}},
			"AB",
		},
		{
			"map pushes scope",
			"{{#user}}Hi {{name}}{{/user}}",
			map[string]any{"user": map[string]any{"name": "Bob"}},
			"Hi Bob",
		},
		{
			"falsey section skipped",
			"A{{#missing}}X{{/missing}}B",
			nil,
			"AB",
		},
		{
			"false bool skipped",
			"{{#flag}}shown{{/flag}}",
			map[string]any{"flag": false},
			"",
		},
		{
			"true bool renders with parent scope",
			"{{#flag}}Yes {{name}}{{/flag}}",
			map[string]any{"flag": true, "name": "X"},
			"Yes X",
		},
		{
			"zero skipped",
			"{{#count}}n{{/count}}",
			map[string]any{"count": 0},
			"",
		},
		{
			"nested sections",
			"{{#groups}}{{#items}}{{.}}{{/items}}{{/groups}}",
			map[string]any{"groups": []any{
				map[string]any{"items": []any{"1", "2"}},
				map[string]any{"items": []any{"3"}},
			}},
			"123",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := mustRender(t, tc.template, tc.data); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestRenderInvertedSections(t *testing.T) {
	tests := []struct {
		name     string
		template string
		data     map[string]any
		want     string
	}{
		{"missing renders", "{{^missing}}fallback{{/missing}}", nil, "fallback"},
		{"present suppressed", "{{^present}}fallback{{/present}}", map[string]any{"present": true}, ""},
		{"empty string renders", "{{^x}}empty{{/x}}", map[string]any{"x": ""}, "empty"},
		{"inverted resolves parent scope", "{{^missing}}{{name}}{{/missing}}", map[string]any{"name": "Bob"}, "Bob"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := mustRender(t, tc.template, tc.data); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestRenderComments(t *testing.T) {
	// A standalone comment tag and its whole line are removed.
	if got := mustRender(t, "Begin\n{{! hidden }}\nEnd\n", nil); got != "Begin\nEnd\n" {
		t.Fatalf("standalone comment = %q", got)
	}
	// An inline comment leaves the surrounding whitespace intact.
	if got := mustRender(t, "A {{! c }} B", nil); got != "A  B" {
		t.Fatalf("inline comment = %q", got)
	}
}

func TestRenderPartials(t *testing.T) {
	// Partials are resolved from string values in the data map.
	if got := mustRender(t, "{{>greet}} {{name}}", map[string]any{
		"greet": "Hello {{name}}",
		"name":  "Bob",
	}); got != "Hello Bob Bob" {
		t.Fatalf("data-based partial = %q", got)
	}
	if got := mustRender(t, "A{{>missing}}B", nil); got != "AB" {
		t.Fatalf("missing partial = %q", got)
	}
	// Explicit partials via RenderWithPartials.
	if got := renderWithPartials(t, "A{{>p}}B", map[string]any{"x": "!"}, map[string]string{"p": "-{{x}}-"}); got != "A-!-B" {
		t.Fatalf("explicit partial = %q", got)
	}
	// A standalone partial inherits its indentation on each line.
	if got := renderWithPartials(t, "  {{>p}}\n", nil, map[string]string{"p": "a\nb"}); got != "  a\n  b" {
		t.Fatalf("indented partial = %q", got)
	}
}

func TestRenderDelimiterChanges(t *testing.T) {
	if got := mustRender(t, "{{=<% %>=}}Hello <%name%>!", map[string]any{"name": "Bob"}); got != "Hello Bob!" {
		t.Fatalf("delimiter change = %q", got)
	}
	// Change delimiters and then change them back to the defaults.
	if got := mustRender(t, "{{=<% %>=}}<%name%><%={{ }}=%>{{name}}", map[string]any{"name": "Bob"}); got != "BobBob" {
		t.Fatalf("delimiter round trip = %q", got)
	}
	// Triple braces only work with the default curly delimiters.
	if got := mustRender(t, "{{=<% %>=}}<%&name%>", map[string]any{"name": "<x>"}); got != "<x>" {
		t.Fatalf("ampersand with changed delimiters = %q", got)
	}
}

func TestRenderStandaloneSections(t *testing.T) {
	template := "Names:\n{{#people}}\n- {{name}}\n{{/people}}\nEnd"
	data := map[string]any{"people": []any{
		map[string]any{"name": "Ann"},
		map[string]any{"name": "Bob"},
	}}
	want := "Names:\n- Ann\n- Bob\nEnd"
	if got := mustRender(t, template, data); got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestRenderErrorsAreDeterministic(t *testing.T) {
	tests := []struct {
		name     string
		template string
		contains string
	}{
		{"unclosed tag", "{{name", "unclosed tag"},
		{"empty tag", "{{}}", "empty tag"},
		{"unclosed set delimiter", "{{=<% %>}}", "unclosed set delimiter tag"},
		{"unbalanced section", "{{#a}}x", "never closed"},
		{"closing never opened", "{{/a}}", "never opened"},
		{"mismatched close", "{{#a}}x{{/b}}", "does not match"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Render(tc.template, nil)
			if err == nil {
				t.Fatalf("expected an error for %q", tc.template)
			}
			if !strings.Contains(err.Error(), tc.contains) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.contains)
			}
			var target *Error
			if !errors.As(err, &target) {
				t.Fatalf("error %v is not a *Error", err)
			}
			// Rendering the same malformed template twice yields the same message.
			_, err2 := Render(tc.template, nil)
			if err2 == nil || err2.Error() != err.Error() {
				t.Fatalf("errors are not deterministic: %q vs %q", err, err2)
			}
		})
	}
}

func TestRenderWithPartialsRejectsMalformedPartial(t *testing.T) {
	_, err := RenderWithPartials("{{>p}}", nil, map[string]string{"p": "{{#unclosed"})
	if err == nil {
		t.Fatal("expected an error for a malformed partial")
	}
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("error %v is not a *Error", err)
	}
}

func TestParseTag(t *testing.T) {
	tests := []struct {
		name      string
		text      string
		wantType  TagType
		wantKey   string
		remaining string
	}{
		{"variable", "name}}rest", TagVariable, "name", "rest"},
		{"comment", "! note }}rest", TagComment, "note", "rest"},
		{"section", "#x}}rest", TagSection, "x", "rest"},
		{"inverted", "^x}}rest", TagInvertedSection, "x", "rest"},
		{"end", "/x}}rest", TagEnd, "x", "rest"},
		{"partial", ">p}}rest", TagPartial, "p", "rest"},
		{"ampersand", "&x}}rest", TagNoEscape, "x", "rest"},
		{"set delimiter", "=<% %>=}}rest", TagSetDelimiter, "<% %>", "rest"},
		{"triple brace", "{name}}}", TagNoEscape, "name", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tok, remaining, err := ParseTag(tc.text, "{{", "}}")
			if err != nil {
				t.Fatalf("ParseTag error: %v", err)
			}
			if tok.Type != tc.wantType || tok.Key != tc.wantKey {
				t.Fatalf("token = %#v want {%q %q}", tok, tc.wantType, tc.wantKey)
			}
			if remaining != tc.remaining {
				t.Fatalf("remaining = %q want %q", remaining, tc.remaining)
			}
		})
	}
}

func TestParseTagErrors(t *testing.T) {
	for _, text := range []string{"name", ""} {
		_, _, err := ParseTag(text, "{{", "}}")
		if err == nil {
			t.Fatalf("expected error for %q", text)
		}
		var target *Error
		if !errors.As(err, &target) {
			t.Fatalf("error %v is not a *Error", err)
		}
	}
}

func TestTokenize(t *testing.T) {
	tokens, err := Tokenize("Hi {{#x}}a{{/x}}")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}
	want := []Token{
		{Type: TagLiteral, Key: "Hi "},
		{Type: TagSection, Key: "x"},
		{Type: TagLiteral, Key: "a"},
		{Type: TagEnd, Key: "x"},
	}
	if len(tokens) != len(want) {
		t.Fatalf("tokens = %#v want %#v", tokens, want)
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("token[%d] = %#v want %#v", i, tokens[i], want[i])
		}
	}
}

func TestTokenizeEmptyAndLiteral(t *testing.T) {
	if tokens, err := Tokenize(""); err != nil || len(tokens) != 0 {
		t.Fatalf("empty template tokens = %#v err = %v", tokens, err)
	}
	tokens, err := Tokenize("just text")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}
	if len(tokens) != 1 || tokens[0] != (Token{Type: TagLiteral, Key: "just text"}) {
		t.Fatalf("tokens = %#v", tokens)
	}
}

func TestTokenizeStandaloneCommentDropsTag(t *testing.T) {
	tokens, err := Tokenize("Begin\n{{! c }}\nEnd\n")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}
	for _, tok := range tokens {
		if tok.Type == TagComment {
			t.Fatalf("comment token should be dropped: %#v", tokens)
		}
	}
	if got := mustRender(t, "Begin\n{{! c }}\nEnd\n", nil); got != "Begin\nEnd\n" {
		t.Fatalf("comment render = %q", got)
	}
}

func TestTokenizeDelimiterChangeEmitsToken(t *testing.T) {
	tokens, err := Tokenize("{{=<% %>=}}<%x%>")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}
	found := false
	for _, tok := range tokens {
		if tok.Type == TagSetDelimiter && tok.Key == "<% %>" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a set delimiter token, got %#v", tokens)
	}
}

func schemaProps(t *testing.T, s schema.Schema) map[string]any {
	t.Helper()
	typ, _ := s["type"].(string)
	if typ != "object" {
		t.Fatalf("schema type = %q want object", typ)
	}
	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties: %#v", s)
	}
	return props
}

func propSchema(t *testing.T, props map[string]any, key string) schema.Schema {
	t.Helper()
	raw, ok := props[key]
	if !ok {
		t.Fatalf("missing property %q in %#v", key, props)
	}
	ps, ok := raw.(schema.Schema)
	if !ok {
		t.Fatalf("property %q is %T, want schema.Schema", key, raw)
	}
	return ps
}

func TestSchemaSimpleVariables(t *testing.T) {
	s, err := Schema("{{name}} {{age}}")
	if err != nil {
		t.Fatalf("Schema error: %v", err)
	}
	props := schemaProps(t, s)
	if len(props) != 2 {
		t.Fatalf("properties = %#v", props)
	}
	for _, key := range []string{"name", "age"} {
		ps := propSchema(t, props, key)
		if typ, _ := ps["type"].(string); typ != "string" {
			t.Fatalf("property %q type = %q", key, typ)
		}
	}
	required, _ := s["required"].([]string)
	if len(required) != 2 || required[0] != "age" || required[1] != "name" {
		t.Fatalf("required = %#v want [age name]", required)
	}
}

func TestSchemaNestedSection(t *testing.T) {
	s, err := Schema("{{#user}}{{name}}{{/user}}")
	if err != nil {
		t.Fatalf("Schema error: %v", err)
	}
	props := schemaProps(t, s)
	user := propSchema(t, props, "user")
	userProps := schemaProps(t, user)
	name := propSchema(t, userProps, "name")
	if typ, _ := name["type"].(string); typ != "string" {
		t.Fatalf("user.name type = %q", name["type"])
	}
	if _, exists := userProps["user"]; exists {
		t.Fatalf("unexpected nested user property")
	}
}

func TestSchemaDottedVariableAbsorbsRoot(t *testing.T) {
	s, err := Schema("{{a}} {{a.b}}")
	if err != nil {
		t.Fatalf("Schema error: %v", err)
	}
	root := schemaProps(t, s)
	a := propSchema(t, root, "a")
	aProps := schemaProps(t, a)
	b := propSchema(t, aProps, "b")
	if typ, _ := b["type"].(string); typ != "string" {
		t.Fatalf("a.b type = %q", b["type"])
	}
}

func TestSchemaSkipsDot(t *testing.T) {
	s, err := Schema("{{#items}}{{.}}{{/items}}")
	if err != nil {
		t.Fatalf("Schema error: %v", err)
	}
	// "items" is a section (container); the "." current-scope tag is skipped.
	props := schemaProps(t, s)
	items := propSchema(t, props, "items")
	if typ, _ := items["type"].(string); typ != "object" {
		t.Fatalf("items type = %q", items["type"])
	}
	child := schemaProps(t, items)
	if len(child) != 0 {
		t.Fatalf("items should have no leaf children, got %#v", child)
	}
}

func TestSchemaPropagatesTokenizeError(t *testing.T) {
	_, err := Schema("{{#unclosed")
	if err == nil {
		t.Fatal("expected Schema to fail on an unbalanced section")
	}
	var target *Error
	if !errors.As(err, &target) {
		t.Fatalf("error %v is not a *Error", err)
	}
}
