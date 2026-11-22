package prompts

import (
	"strings"
	"testing"
)

func sfJinja(t *testing.T, template string, values map[string]any) string {
	t.Helper()
	got, err := Jinja2Formatter(template, values)
	if err != nil {
		t.Fatalf("Jinja2Formatter(%q) error: %v", template, err)
	}
	return got
}

func sfFString(t *testing.T, template string, values map[string]any) string {
	t.Helper()
	got, err := FStringFormatter(template, values)
	if err != nil {
		t.Fatalf("FStringFormatter(%q) error: %v", template, err)
	}
	return got
}

// ---------------------------------------------------------------------------
// Jinja2Formatter: supported subset
// ---------------------------------------------------------------------------

func TestJinja2FormatterVariables(t *testing.T) {
	cases := []struct {
		name     string
		template string
		values   map[string]any
		want     string
	}{
		{"plain", "Hello {{ name }}!", map[string]any{"name": "Alice"}, "Hello Alice!"},
		{"dotted", "{{ user.name }}", map[string]any{"user": map[string]any{"name": "Bob"}}, "Bob"},
		{"missing is empty", "a{{ missing }}b", nil, "ab"},
		{"nil context", "x{{ name }}y", nil, "xy"},
		{"number", "{{ n }}", map[string]any{"n": 5}, "5"},
		{"no autoescape", "{{ v }}", map[string]any{"v": "<b>&"}, "<b>&"},
		{"comment dropped", "A{# hidden #}B", nil, "AB"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sfJinja(t, tc.template, tc.values); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestJinja2FormatterFilters(t *testing.T) {
	cases := []struct {
		name     string
		template string
		values   map[string]any
		want     string
	}{
		{"upper", "{{ name | upper }}", map[string]any{"name": "Alice"}, "ALICE"},
		{"lower", "{{ name | lower }}", map[string]any{"name": "Alice"}, "alice"},
		{"default substitutes missing", `{{ missing | default("N/A") }}`, nil, "N/A"},
		{"default keeps value", `{{ name | default("N/A") }}`, map[string]any{"name": "Bob"}, "Bob"},
		{"chained filters", "{{ name | lower | upper }}", map[string]any{"name": "Alice"}, "ALICE"},
		{"pipe inside quoted default", `{{ x | default("a|b") }}`, nil, "a|b"},
		{"default from a path", `{{ missing | default(fallback) }}`, map[string]any{"fallback": "F"}, "F"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sfJinja(t, tc.template, tc.values); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestJinja2FormatterIf(t *testing.T) {
	cases := []struct {
		name     string
		template string
		values   map[string]any
		want     string
	}{
		{"true branch", "{% if show %}YES{% else %}NO{% endif %}", map[string]any{"show": true}, "YES"},
		{"false branch", "{% if show %}YES{% else %}NO{% endif %}", map[string]any{"show": false}, "NO"},
		{"no else", "{% if show %}YES{% endif %}", map[string]any{"show": false}, ""},
		{"not", "{% if not show %}HIDDEN{% endif %}", map[string]any{"show": false}, "HIDDEN"},
		{"truthy string", "{% if name %}Hi{% endif %}", map[string]any{"name": "x"}, "Hi"},
		{"equality", `{% if status == "ok" %}GOOD{% else %}BAD{% endif %}`, map[string]any{"status": "ok"}, "GOOD"},
		{"inequality", `{% if status != "ok" %}BAD{% endif %}`, map[string]any{"status": "bad"}, "BAD"},
		{"comparison with filter", `{% if name | lower == "bob" %}YES{% endif %}`, map[string]any{"name": "Bob"}, "YES"},
		{"nested if", "{% if a %}{% if b %}AB{% endif %}{% endif %}", map[string]any{"a": true, "b": true}, "AB"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sfJinja(t, tc.template, tc.values); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestJinja2FormatterFor(t *testing.T) {
	cases := []struct {
		name     string
		template string
		values   map[string]any
		want     string
	}{
		{"list", "{% for x in items %}[{{ x }}]{% endfor %}", map[string]any{"items": []any{"a", "b"}}, "[a][b]"},
		{"empty list", "A{% for x in items %}{{ x }}{% endfor %}B", map[string]any{"items": []any{}}, "AB"},
		{"loop index", "{% for x in items %}{{ loop.index }}:{{ x }} {% endfor %}", map[string]any{"items": []any{"a", "b"}}, "1:a 2:b "},
		{"loop flags", "{% for x in items %}{{ loop.first }},{{ loop.last }};{% endfor %}", map[string]any{"items": []any{"a", "b"}}, "true,false;false,true;"},
		{"map iterates sorted keys", "{% for k in obj %}{{ k }}-{% endfor %}", map[string]any{"obj": map[string]any{"b": 1, "a": 2}}, "a-b-"},
		{"string iterates runes", "{% for c in word %}[{{ c }}]{% endfor %}", map[string]any{"word": "hi"}, "[h][i]"},
		{"dotted iterable", "{% for x in data.items %}{{ x }}{% endfor %}", map[string]any{"data": map[string]any{"items": []any{"1", "2"}}}, "12"},
		{"for inside if", "{% if ok %}{% for x in items %}{{ x }}{% endfor %}{% endif %}", map[string]any{"ok": true, "items": []any{"a", "b"}}, "ab"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sfJinja(t, tc.template, tc.values); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Jinja2Formatter: unsupported constructs must return explicit errors
// ---------------------------------------------------------------------------

func TestJinja2FormatterUnsupported(t *testing.T) {
	cases := []struct {
		name     string
		template string
		contains string
	}{
		{"set tag", "{% set x = 1 %}{{ x }}", "unsupported construct"},
		{"macro", "{% macro m() %}x{% endmacro %}", "unsupported construct"},
		{"include", `{% include "x" %}`, "unsupported construct"},
		{"elif", "{% if a %}A{% elif b %}B{% endif %}", "unexpected"},
		{"and condition", "{% if a and b %}x{% endif %}", "unsupported if condition"},
		{"greater than", "{% if a > 1 %}x{% endif %}", "unsupported"},
		{"unknown filter", `{{ name | join(",") }}`, "unsupported filter"},
		{"filter with unexpected argument", `{{ name | lower("x") }}`, "does not take arguments"},
		{"function call", "{{ fn() }}", "unsupported expression"},
		{"arithmetic expression", "{{ a + b }}", "unsupported expression"},
		{"whitespace control var", "{{- name -}}", "unsupported"},
		{"unclosed variable", "{{ name", "unclosed"},
		{"unclosed block delimiter", "{% if x", "unclosed"},
		{"missing endif", "{% if x %}unclosed", "missing endif"},
		{"missing endfor", "{% for x in items %}unclosed", "missing endfor"},
		{"stray endif", "{% endif %}", "unexpected"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Jinja2Formatter(tc.template, map[string]any{"a": true, "b": true, "name": "x", "x": "y", "items": []any{}})
			if err == nil {
				t.Fatalf("expected an error for %q", tc.template)
			}
			if !strings.Contains(err.Error(), tc.contains) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.contains)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Jinja2TemplateVariables / ValidateJinja2
// ---------------------------------------------------------------------------

func sfVars(t *testing.T, call func() ([]string, error)) []string {
	t.Helper()
	got, err := call()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return got
}

func TestJinja2TemplateVariables(t *testing.T) {
	cases := []struct {
		name     string
		template string
		want     []string
	}{
		{"variables", "{{ name }} {{ user.age }}", []string{"name", "user"}},
		{"if condition and branches", "{% if show %}{{ a }}{% else %}{{ b }}{% endif %}", []string{"a", "b", "show"}},
		{"for excludes loop vars", "{% for x in items %}{{ x }}{{ loop.index }}{{ outer }}{% endfor %}", []string{"items", "outer"}},
		{"filters contribute default arg", `{{ missing | default(fallback) }}`, []string{"fallback", "missing"}},
		{"literal operand excluded", `{% if status == "ok" %}x{% endif %}`, []string{"status"}},
		{"not condition", "{% if not flag %}x{% endif %}", []string{"flag"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := sfVars(t, func() ([]string, error) { return Jinja2TemplateVariables(tc.template) })
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}

func TestJinja2TemplateVariablesRejectsUnsupported(t *testing.T) {
	if _, err := Jinja2TemplateVariables("{% set x = 1 %}"); err == nil {
		t.Fatal("expected an error for an unsupported construct")
	}
}

func TestValidateJinja2(t *testing.T) {
	if err := ValidateJinja2("{{ name }}", []string{"name"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err := ValidateJinja2("{{ name }}", []string{"other"})
	if err == nil || !strings.Contains(err.Error(), "missing variables") {
		t.Fatalf("expected a missing-variables error, got %v", err)
	}
	if err := ValidateJinja2("{% set x = 1 %}", []string{}); err == nil {
		t.Fatal("expected an error for unsupported construct")
	}
}

// ---------------------------------------------------------------------------
// FStringFormatter / ValidateFStringTemplate
// ---------------------------------------------------------------------------

func TestFStringFormatter(t *testing.T) {
	cases := []struct {
		name     string
		template string
		values   map[string]any
		want     string
	}{
		{"plain", "Hello {name}!", map[string]any{"name": "Alice"}, "Hello Alice!"},
		{"braces escaped", "{{literal}} {name}", map[string]any{"name": "Alice"}, "{literal} Alice"},
		{"repeated field", "{a}-{a}", map[string]any{"a": "x"}, "x-x"},
		{"integer spec", "{count:d}", map[string]any{"count": 7}, "7"},
		{"float spec", "{price:.2f}", map[string]any{"price": 3.14159}, "3.14"},
		{"conversion ignored", "{name!r}", map[string]any{"name": "Bob"}, "Bob"},
		{"empty template", "", nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sfFString(t, tc.template, tc.values); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestFStringFormatterErrors(t *testing.T) {
	errors := []struct {
		name     string
		template string
		values   map[string]any
		contains string
	}{
		{"missing variable", "{name}", nil, "missing variable"},
		{"attribute access", "{user.name}", map[string]any{"user.name": "x"}, "attribute access"},
		{"indexing", "{items[0]}", map[string]any{"items[0]": "x"}, "indexing"},
		{"all digits", "{0}", map[string]any{"0": "x"}, "all-digit"},
		{"nested field", "{a:{b}}", nil, "nested"},
		{"single closing brace", "oops }", nil, "single '}'"},
		{"unmatched opening brace", "{name", nil, "unmatched"},
		{"unsupported spec", "{name:>10}", map[string]any{"name": "x"}, "unsupported format spec"},
	}
	for _, tc := range errors {
		t.Run(tc.name, func(t *testing.T) {
			_, err := FStringFormatter(tc.template, tc.values)
			if err == nil {
				t.Fatalf("expected error for %q", tc.template)
			}
			if !strings.Contains(err.Error(), tc.contains) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.contains)
			}
		})
	}
}

func TestValidateFStringTemplate(t *testing.T) {
	cases := []struct {
		name     string
		template string
		want     []string
	}{
		{"dedup and sort", "{b} {a} {b}", []string{"a", "b"}},
		{"no fields", "just text", []string{}},
		{"dotted is rejected", "{a.b}", nil},
		{"index is rejected", "{a[0]}", nil},
		{"digits rejected", "{1}", nil},
		{"nested rejected", "{a:{b}}", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ValidateFStringTemplate(tc.template)
			if tc.want == nil {
				if err == nil {
					t.Fatalf("expected error for %q", tc.template)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// GetTemplateVariables / CheckValidTemplate / IsSubsequence
// ---------------------------------------------------------------------------

func TestGetTemplateVariables(t *testing.T) {
	cases := []struct {
		name     string
		template string
		format   string
		want     []string
	}{
		{"f-string", "{a} {b}", "f-string", []string{"a", "b"}},
		{"mustache top level", "{{a}} {{#b}}{{c}}{{/b}}", "mustache", []string{"a", "b"}},
		{"mustache dotted", "{{user.name}} {{x}}", "mustache", []string{"user", "x"}},
		{"jinja2", "{{ x }} {{ y.z }}", "jinja2", []string{"x", "y"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := GetTemplateVariables(tc.template, tc.format)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
	if _, err := GetTemplateVariables("x", "unknown"); err == nil {
		t.Fatal("expected error for unsupported format")
	}
}

func TestCheckValidTemplate(t *testing.T) {
	valid := []struct {
		template string
		format   string
		vars     []string
	}{
		{"Hi {name}", "f-string", []string{"name"}},
		{"{{ x }}", "jinja2", []string{"x"}},
		{"{{a}}", "mustache", []string{"a"}},
	}
	for _, tc := range valid {
		if err := CheckValidTemplate(tc.template, tc.format, tc.vars); err != nil {
			t.Fatalf("CheckValidTemplate(%q, %q) error: %v", tc.template, tc.format, err)
		}
	}
	if err := CheckValidTemplate("Hi {name}", "f-string", []string{"other"}); err == nil {
		t.Fatal("expected error for variable mismatch")
	}
	if err := CheckValidTemplate("x", "unknown", []string{}); err == nil {
		t.Fatal("expected error for unknown format")
	}
	if err := CheckValidTemplate("{a.b}", "f-string", []string{"a.b"}); err == nil {
		t.Fatal("expected error for invalid f-string template")
	}
}

func TestIsSubsequence(t *testing.T) {
	cases := []struct {
		name   string
		child  []string
		parent []string
		want   bool
	}{
		{"prefix", []string{"a", "b"}, []string{"a", "b", "c"}, true},
		{"mismatch", []string{"a", "c"}, []string{"a", "b", "c"}, false},
		{"empty child", []string{}, []string{"a"}, false},
		{"empty parent", []string{"a"}, []string{}, false},
		{"parent shorter", []string{"a", "b"}, []string{"a"}, false},
		{"equal", []string{"a"}, []string{"a"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsSubsequence(tc.child, tc.parent); got != tc.want {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
	if !IsSubsequence([]int{1, 2}, []int{1, 2, 3}) {
		t.Fatal("generic int instantiation failed")
	}
}

// ---------------------------------------------------------------------------
// FormatterMapping / DefaultFormatterMapping / Mustache helpers
// ---------------------------------------------------------------------------

func TestDefaultFormatterMapping(t *testing.T) {
	if len(DefaultFormatterMapping) != 3 {
		t.Fatalf("mapping has %d entries, want 3", len(DefaultFormatterMapping))
	}
	for _, key := range []string{"f-string", "mustache", "jinja2"} {
		if _, ok := DefaultFormatterMapping[key]; !ok {
			t.Fatalf("missing formatter %q", key)
		}
	}
	values := map[string]any{"name": "X"}
	cases := []struct {
		key      string
		template string
		want     string
	}{
		{"f-string", "Hi {name}", "Hi X"},
		{"mustache", "Hi {{name}}", "Hi X"},
		{"jinja2", "Hi {{ name }}", "Hi X"},
	}
	for _, tc := range cases {
		got, err := DefaultFormatterMapping[tc.key](tc.template, values)
		if err != nil {
			t.Fatalf("formatter %q error: %v", tc.key, err)
		}
		if got != tc.want {
			t.Fatalf("formatter %q got %q want %q", tc.key, got, tc.want)
		}
	}
}

func TestMustacheFormatterAndSchema(t *testing.T) {
	if got, err := MustacheFormatter("Hi {{name}}", map[string]any{"name": "X"}); err != nil || got != "Hi X" {
		t.Fatalf("MustacheFormatter got %q err %v", got, err)
	}
	s, err := MustacheSchema("{{name}} {{#user}}{{age}}{{/user}}")
	if err != nil {
		t.Fatalf("MustacheSchema error: %v", err)
	}
	if typ, _ := s["type"].(string); typ != "object" {
		t.Fatalf("schema type = %q", typ)
	}
	props, ok := s["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties: %#v", s)
	}
	if _, ok := props["name"]; !ok {
		t.Fatalf("missing name property: %#v", props)
	}
	if _, ok := props["user"]; !ok {
		t.Fatalf("missing user property: %#v", props)
	}
}
