package prompts

import (
	"fmt"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/mustache"
	"github.com/projanvil/langchain-golang/core/schema"
)

// FormatterMapping maps a template format name to the function that renders
// that format. The formatter signature is shared by FStringFormatter,
// MustacheFormatter, and Jinja2Formatter.
type FormatterMapping map[string]func(string, map[string]any) (string, error)

// DefaultFormatterMapping is the default set of template formatters keyed by
// format name, mirroring langchain_core.prompts.string.DEFAULT_FORMATTER_MAPPING.
// Treat it as read-only; callers that need to add formats should copy it.
var DefaultFormatterMapping = FormatterMapping{
	"f-string": FStringFormatter,
	"mustache": MustacheFormatter,
	"jinja2":   Jinja2Formatter,
}

// ---------------------------------------------------------------------------
// Mustache formatting
// ---------------------------------------------------------------------------

// MustacheFormatter renders a Mustache template with the given values. It
// delegates to the standalone core/mustache package.
func MustacheFormatter(template string, values map[string]any) (string, error) {
	return mustache.Render(template, values)
}

// MustacheSchema returns a JSON-schema-shaped description of the variables a
// Mustache template expects. It delegates to core/mustache.Schema.
func MustacheSchema(template string) (schema.Schema, error) {
	return mustache.Schema(template)
}

// sfMustacheTemplateVars returns the sorted top-level variables referenced by
// a Mustache template, mirroring langchain_core's mustache_template_vars. Only
// variables at section depth zero are reported, and dotted names collapse to
// their top-level key.
func sfMustacheTemplateVars(template string) ([]string, error) {
	tokens, err := mustache.Tokenize(template)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	depth := 0
	for _, tok := range tokens {
		switch tok.Type {
		case mustache.TagEnd:
			depth--
		case mustache.TagVariable, mustache.TagSection, mustache.TagInvertedSection, mustache.TagNoEscape:
			if depth == 0 && tok.Key != "." {
				name := strings.Split(tok.Key, ".")[0]
				if name != "" {
					seen[name] = true
				}
			}
		}
		if tok.Type == mustache.TagSection || tok.Type == mustache.TagInvertedSection {
			depth++
		}
	}
	return sfSortedKeys(seen), nil
}

// ---------------------------------------------------------------------------
// f-string formatting
// ---------------------------------------------------------------------------

// sfFStringToken is one piece of a parsed f-string template: either a literal
// text chunk or a replacement field.
type sfFStringToken struct {
	isField bool
	text    string
	name    string
	spec    string
	hasSpec bool
}

// sfParseFString parses a str.format-style template into literal and field
// tokens. It understands {{ and }} escapes, {name} fields, {name!conv}
// conversions (accepted but normalized to the plain string form), and
// {name:spec} format specs. Nested replacement fields are rejected.
func sfParseFString(template string) ([]sfFStringToken, error) {
	tokens := []sfFStringToken{}
	var text strings.Builder
	flush := func() {
		if text.Len() > 0 {
			tokens = append(tokens, sfFStringToken{text: text.String()})
			text.Reset()
		}
	}
	for i := 0; i < len(template); {
		ch := template[i]
		switch ch {
		case '{':
			if i+1 < len(template) && template[i+1] == '{' {
				text.WriteByte('{')
				i += 2
				continue
			}
			end := strings.IndexByte(template[i+1:], '}')
			if end < 0 {
				return nil, fmt.Errorf("f-string: unmatched '{' in template")
			}
			content := template[i+1 : i+1+end]
			if strings.Contains(content, "{") {
				return nil, fmt.Errorf("f-string: nested replacement fields are not allowed")
			}
			namePart := content
			spec := ""
			hasSpec := false
			if c := strings.IndexByte(content, ':'); c >= 0 {
				namePart = content[:c]
				spec = content[c+1:]
				hasSpec = true
			}
			if e := strings.IndexByte(namePart, '!'); e >= 0 {
				// Conversion flags (!r, !s, !a) are normalized to the plain
				// string form; Python's repr/str distinction is not preserved.
				namePart = namePart[:e]
			}
			name := strings.TrimSpace(namePart)
			if err := sfValidateFStringName(name); err != nil {
				return nil, err
			}
			if hasSpec && (strings.Contains(spec, "{") || strings.Contains(spec, "}")) {
				return nil, fmt.Errorf("f-string: nested replacement fields are not allowed")
			}
			flush()
			tokens = append(tokens, sfFStringToken{isField: true, name: name, spec: spec, hasSpec: hasSpec})
			i = i + 1 + end + 1
		case '}':
			if i+1 < len(template) && template[i+1] == '}' {
				text.WriteByte('}')
				i += 2
				continue
			}
			return nil, fmt.Errorf("f-string: single '}' in template is not allowed")
		default:
			text.WriteByte(ch)
			i++
		}
	}
	flush()
	return tokens, nil
}

// sfValidateFStringName rejects variable names with attribute access,
// indexing, or all-digit names, matching validate_f_string_template.
func sfValidateFStringName(name string) error {
	if name == "" {
		return fmt.Errorf("f-string: empty variable name")
	}
	if strings.ContainsAny(name, ".[]") {
		return fmt.Errorf("f-string: invalid variable name %q; attribute access (.) and indexing ([]) are not allowed", name)
	}
	if sfAllDigits(name) {
		return fmt.Errorf("f-string: invalid variable name %q; all-digit names are interpreted as positional arguments", name)
	}
	return nil
}

func sfAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// FStringFormatter renders a str.format-style template. Missing variables and
// unsupported format specs return an error. A format spec, when present, is
// applied with fmt.Sprintf using "%"+spec; specs that fmt cannot honor (for
// example Python's ">10" alignment) return an explicit error.
func FStringFormatter(template string, values map[string]any) (string, error) {
	tokens, err := sfParseFString(template)
	if err != nil {
		return "", err
	}
	if values == nil {
		values = map[string]any{}
	}
	var out strings.Builder
	for _, tok := range tokens {
		if !tok.isField {
			out.WriteString(tok.text)
			continue
		}
		value, ok := values[tok.name]
		if !ok {
			return "", fmt.Errorf("f-string: missing variable %q", tok.name)
		}
		if tok.hasSpec && tok.spec != "" {
			rendered := fmt.Sprintf("%"+tok.spec, value)
			if strings.Contains(rendered, "%!") {
				return "", fmt.Errorf("f-string: unsupported format spec %q", tok.spec)
			}
			out.WriteString(rendered)
		} else {
			out.WriteString(sfStringify(value))
		}
	}
	return out.String(), nil
}

// ValidateFStringTemplate validates a str.format-style template and returns
// its sorted, de-duplicated input variable names. It mirrors langchain_core's
// validate_f_string_template.
func ValidateFStringTemplate(template string) ([]string, error) {
	tokens, err := sfParseFString(template)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	for _, tok := range tokens {
		if tok.isField {
			seen[tok.name] = true
		}
	}
	return sfSortedKeys(seen), nil
}

// ---------------------------------------------------------------------------
// jinja2 subset
// ---------------------------------------------------------------------------
//
// The following is a deliberately small, documented Jinja2 subset implemented
// with the standard library only (no third-party Jinja dependency). Supported
// constructs:
//   - variable interpolation {{ expr }} with dotted paths and literal values
//   - filters: lower, upper, default(value)
//   - {% if cond %} ... {% else %} ... {% endif %} where cond is a bare
//     variable, "not <cond>", or "<a> == <b>" / "<a> != <b>"
//   - {% for x in items %} ... {% endfor %}, with loop.index, loop.index0,
//     loop.first, and loop.last available inside the loop
//   - comments {# ... #}
//
// Everything else (set, macro, include, extends, elif, boolean expressions
// with and/or, comparisons other than ==/!=, whitespace control, function
// calls, unknown filters, etc.) returns an explicit "unsupported" error.

type sfJinjaTokenKind int

const (
	sfJinjaText sfJinjaTokenKind = iota
	sfJinjaVar
	sfJinjaBlock
	sfJinjaComment
)

type sfJinjaToken struct {
	kind    sfJinjaTokenKind
	content string
}

type sfJinjaNode struct {
	kind string // "text", "var", "if", "for"
	text string // literal text
	expr string // variable expression

	cond     string // if condition
	thenBody []*sfJinjaNode
	elseBody []*sfJinjaNode

	varName  string // for
	iterable string
	loopBody []*sfJinjaNode
}

var (
	sfIdentRe   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$`)
	sfVarNameRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
)

// sfTokenizeJinja splits a template into text, variable, block, and comment
// tokens. Unclosed delimiters return an error.
func sfTokenizeJinja(template string) ([]sfJinjaToken, error) {
	type opener struct {
		open  string
		close string
		kind  sfJinjaTokenKind
	}
	openers := []opener{
		{"{{", "}}", sfJinjaVar},
		{"{%", "%}", sfJinjaBlock},
		{"{#", "#}", sfJinjaComment},
	}
	tokens := []sfJinjaToken{}
	for i := 0; i < len(template); {
		bestIdx := len(template)
		best := -1
		for k, o := range openers {
			if idx := strings.Index(template[i:], o.open); idx >= 0 && i+idx < bestIdx {
				bestIdx = i + idx
				best = k
			}
		}
		if best < 0 {
			tokens = append(tokens, sfJinjaToken{kind: sfJinjaText, content: template[i:]})
			break
		}
		if bestIdx > i {
			tokens = append(tokens, sfJinjaToken{kind: sfJinjaText, content: template[i:bestIdx]})
		}
		o := openers[best]
		start := bestIdx + len(o.open)
		closeIdx := strings.Index(template[start:], o.close)
		if closeIdx < 0 {
			return nil, fmt.Errorf("jinja2: unclosed %s tag", o.open)
		}
		tokens = append(tokens, sfJinjaToken{kind: o.kind, content: template[start : start+closeIdx]})
		i = start + closeIdx + len(o.close)
	}
	return tokens, nil
}

// sfJinjaParser holds token state while building the AST.
type sfJinjaParser struct {
	tokens []sfJinjaToken
	pos    int
}

// sfParseJinja parses a template into an AST, validating every expression
// and condition so that unsupported constructs are rejected consistently by
// every public entry point.
func sfParseJinja(template string) ([]*sfJinjaNode, error) {
	tokens, err := sfTokenizeJinja(template)
	if err != nil {
		return nil, err
	}
	p := &sfJinjaParser{tokens: tokens}
	stops := map[string]bool{"else": true, "endif": true, "endfor": true, "elif": true}
	nodes, stop, err := p.parseSequence(stops)
	if err != nil {
		return nil, err
	}
	if stop != "" {
		return nil, fmt.Errorf("jinja2: unexpected {%% %s %%} tag", stop)
	}
	return nodes, nil
}

// parseSequence parses nodes until a block tag whose keyword is in stops is
// reached (that token is left unconsumed) or tokens run out.
func (p *sfJinjaParser) parseSequence(stops map[string]bool) ([]*sfJinjaNode, string, error) {
	nodes := []*sfJinjaNode{}
	for p.pos < len(p.tokens) {
		tok := p.tokens[p.pos]
		switch tok.kind {
		case sfJinjaText:
			if tok.content != "" {
				nodes = append(nodes, &sfJinjaNode{kind: "text", text: tok.content})
			}
			p.pos++
		case sfJinjaComment:
			p.pos++
		case sfJinjaVar:
			expr := strings.TrimSpace(tok.content)
			if err := sfCheckExpr(expr); err != nil {
				return nil, "", err
			}
			nodes = append(nodes, &sfJinjaNode{kind: "var", expr: expr})
			p.pos++
		case sfJinjaBlock:
			tag := strings.TrimSpace(tok.content)
			fields := strings.Fields(tag)
			if len(fields) == 0 {
				return nil, "", fmt.Errorf("jinja2: empty {%% %%} tag")
			}
			keyword := fields[0]
			if stops[keyword] {
				return nodes, keyword, nil
			}
			switch keyword {
			case "if":
				cond := strings.TrimSpace(tag[len(keyword):])
				if err := sfCheckCond(cond); err != nil {
					return nil, "", err
				}
				p.pos++ // consume if
				thenBody, stop, err := p.parseSequence(map[string]bool{"else": true, "endif": true})
				if err != nil {
					return nil, "", err
				}
				node := &sfJinjaNode{kind: "if", cond: cond, thenBody: thenBody}
				switch stop {
				case "endif":
					p.pos++
				case "else":
					p.pos++ // consume else
					elseBody, stop2, err := p.parseSequence(map[string]bool{"endif": true})
					if err != nil {
						return nil, "", err
					}
					if stop2 != "endif" {
						return nil, "", fmt.Errorf("jinja2: missing endif")
					}
					p.pos++ // consume endif
					node.elseBody = elseBody
				default:
					return nil, "", fmt.Errorf("jinja2: missing endif")
				}
				nodes = append(nodes, node)
			case "for":
				rest := strings.TrimSpace(tag[len(keyword):])
				idx := strings.Index(rest, " in ")
				if idx < 0 {
					return nil, "", fmt.Errorf("jinja2: unsupported for tag {%% %s %%}; expected 'for x in items'", tag)
				}
				varName := strings.TrimSpace(rest[:idx])
				iterable := strings.TrimSpace(rest[idx+len(" in "):])
				if !sfVarNameRe.MatchString(varName) {
					return nil, "", fmt.Errorf("jinja2: invalid for variable %q", varName)
				}
				if err := sfCheckBase(iterable); err != nil {
					return nil, "", err
				}
				p.pos++ // consume for
				body, stop, err := p.parseSequence(map[string]bool{"endfor": true})
				if err != nil {
					return nil, "", err
				}
				if stop != "endfor" {
					return nil, "", fmt.Errorf("jinja2: missing endfor")
				}
				p.pos++ // consume endfor
				nodes = append(nodes, &sfJinjaNode{kind: "for", varName: varName, iterable: iterable, loopBody: body})
			default:
				if sfIsControlKeyword(keyword) {
					return nil, "", fmt.Errorf("jinja2: unexpected {%% %s %%} tag", tag)
				}
				return nil, "", fmt.Errorf("jinja2: unsupported construct {%% %s %%}", tag)
			}
		}
	}
	return nodes, "", nil
}

func sfIsControlKeyword(keyword string) bool {
	switch keyword {
	case "else", "endif", "endfor", "elif", "endblock", "endmacro",
		"endwith", "endfilter", "endraw", "endset", "endcall", "endtrans":
		return true
	default:
		return false
	}
}

// sfSplitPipe splits an expression on "|" while ignoring "|" characters
// inside single- or double-quoted strings.
func sfSplitPipe(expr string) []string {
	parts := []string{}
	var b strings.Builder
	var quote rune
	for _, r := range expr {
		switch {
		case quote != 0:
			b.WriteRune(r)
			if r == quote {
				quote = 0
			}
		case r == '\'' || r == '"':
			quote = r
			b.WriteRune(r)
		case r == '|':
			parts = append(parts, strings.TrimSpace(b.String()))
			b.Reset()
		default:
			b.WriteRune(r)
		}
	}
	parts = append(parts, strings.TrimSpace(b.String()))
	return parts
}

// sfParseFilter splits a filter string into its name and argument content.
func sfParseFilter(filter string) (string, string) {
	filter = strings.TrimSpace(filter)
	if i := strings.Index(filter, "("); i >= 0 {
		name := strings.TrimSpace(filter[:i])
		arg := strings.TrimSpace(filter[i+1:])
		arg = strings.TrimSpace(strings.TrimSuffix(arg, ")"))
		return name, arg
	}
	return filter, ""
}

// sfCheckBase validates a bare expression (path, quoted string, boolean,
// none, or number) without a data context.
func sfCheckBase(base string) error {
	base = strings.TrimSpace(base)
	if base == "" {
		return fmt.Errorf("jinja2: empty expression")
	}
	if sfIsQuoted(base) {
		return nil
	}
	if sfIdentRe.MatchString(base) {
		return nil
	}
	switch strings.ToLower(base) {
	case "true", "false", "none", "null":
		return nil
	}
	if _, err := strconv.ParseFloat(base, 64); err == nil {
		return nil
	}
	return fmt.Errorf("jinja2: unsupported expression %q", base)
}

// sfCheckExpr validates a full variable expression (base plus filters).
func sfCheckExpr(expr string) error {
	parts := sfSplitPipe(expr)
	if err := sfCheckBase(parts[0]); err != nil {
		return err
	}
	for _, filter := range parts[1:] {
		if err := sfCheckFilter(filter); err != nil {
			return err
		}
	}
	return nil
}

// sfCheckFilter validates a filter name and its arguments.
func sfCheckFilter(filter string) error {
	name, arg := sfParseFilter(filter)
	switch name {
	case "lower", "upper":
		if arg != "" {
			return fmt.Errorf("jinja2: filter %q does not take arguments", name)
		}
	case "default":
		if arg == "" {
			return fmt.Errorf("jinja2: filter default requires an argument")
		}
		if err := sfCheckBase(arg); err != nil {
			return err
		}
	default:
		return fmt.Errorf("jinja2: unsupported filter %q", name)
	}
	return nil
}

// sfCheckCond validates an if condition without a data context.
func sfCheckCond(cond string) error {
	cond = strings.TrimSpace(cond)
	if cond == "" {
		return fmt.Errorf("jinja2: empty if condition")
	}
	if strings.HasPrefix(cond, "not ") {
		return sfCheckCond(strings.TrimSpace(cond[len("not "):]))
	}
	if strings.Contains(cond, " and ") || strings.Contains(cond, " or ") {
		return fmt.Errorf("jinja2: unsupported if condition %q; only bare variables, not, == and != are supported", cond)
	}
	for _, op := range []string{"==", "!="} {
		if i := strings.Index(cond, op); i >= 0 {
			left := strings.TrimSpace(cond[:i])
			right := strings.TrimSpace(cond[i+len(op):])
			if left == "" || right == "" {
				return fmt.Errorf("jinja2: unsupported if condition %q", cond)
			}
			if err := sfCheckExpr(left); err != nil {
				return err
			}
			return sfCheckExpr(right)
		}
	}
	return sfCheckExpr(cond)
}

// sfIsQuoted reports whether s is a simple single- or double-quoted literal.
func sfIsQuoted(s string) bool {
	if len(s) < 2 {
		return false
	}
	first, last := s[0], s[len(s)-1]
	return (first == '"' && last == '"') || (first == '\'' && last == '\'')
}

func sfUnquote(s string) string {
	if sfIsQuoted(s) {
		return s[1 : len(s)-1]
	}
	return s
}

// sfIndexValue resolves one dotted-path part into a value, supporting
// string-keyed maps, slices, and arrays.
func sfIndexValue(current any, part string) (any, bool) {
	if current == nil {
		return nil, false
	}
	rv := reflect.ValueOf(current)
	switch rv.Kind() {
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return nil, false
		}
		key := reflect.ValueOf(part)
		if key.Type() != rv.Type().Key() {
			key = key.Convert(rv.Type().Key())
		}
		value := rv.MapIndex(key)
		if !value.IsValid() {
			return nil, false
		}
		return value.Interface(), true
	case reflect.Slice, reflect.Array:
		idx, err := strconv.Atoi(part)
		if err != nil || idx < 0 || idx >= rv.Len() {
			return nil, false
		}
		return rv.Index(idx).Interface(), true
	default:
		return nil, false
	}
}

// sfResolvePath resolves a dotted path against the context. Missing keys
// resolve to nil (rendered as an empty string), matching Jinja's undefined.
func sfResolvePath(path string, ctx map[string]any) (any, error) {
	var current any = ctx
	for _, part := range strings.Split(path, ".") {
		next, ok := sfIndexValue(current, part)
		if !ok {
			return nil, nil
		}
		current = next
	}
	return current, nil
}

// sfEvalBase evaluates a bare expression to a value.
func sfEvalBase(base string, ctx map[string]any) (any, error) {
	if err := sfCheckBase(base); err != nil {
		return nil, err
	}
	base = strings.TrimSpace(base)
	if sfIsQuoted(base) {
		return sfUnquote(base), nil
	}
	if sfIdentRe.MatchString(base) {
		return sfResolvePath(base, ctx)
	}
	switch strings.ToLower(base) {
	case "true":
		return true, nil
	case "false":
		return false, nil
	case "none", "null":
		return nil, nil
	}
	if i, err := strconv.ParseInt(base, 10, 64); err == nil {
		return i, nil
	}
	if f, err := strconv.ParseFloat(base, 64); err == nil {
		return f, nil
	}
	return nil, fmt.Errorf("jinja2: unsupported expression %q", base)
}

// sfApplyFilter applies one filter to a value.
func sfApplyFilter(value any, filter string, ctx map[string]any) (any, error) {
	name, arg := sfParseFilter(filter)
	switch name {
	case "lower":
		return strings.ToLower(sfStringify(value)), nil
	case "upper":
		return strings.ToUpper(sfStringify(value)), nil
	case "default":
		if value == nil {
			return sfEvalBase(arg, ctx)
		}
		return value, nil
	default:
		return nil, fmt.Errorf("jinja2: unsupported filter %q", name)
	}
}

// sfEvalExpr evaluates a full variable expression (base plus filters).
func sfEvalExpr(expr string, ctx map[string]any) (any, error) {
	if err := sfCheckExpr(expr); err != nil {
		return nil, err
	}
	parts := sfSplitPipe(expr)
	value, err := sfEvalBase(parts[0], ctx)
	if err != nil {
		return nil, err
	}
	for _, filter := range parts[1:] {
		value, err = sfApplyFilter(value, filter, ctx)
		if err != nil {
			return nil, err
		}
	}
	return value, nil
}

// sfEvalCond evaluates an if condition to a boolean.
func sfEvalCond(cond string, ctx map[string]any) (bool, error) {
	if err := sfCheckCond(cond); err != nil {
		return false, err
	}
	cond = strings.TrimSpace(cond)
	if strings.HasPrefix(cond, "not ") {
		value, err := sfEvalCond(strings.TrimSpace(cond[len("not "):]), ctx)
		return !value, err
	}
	for _, op := range []string{"==", "!="} {
		if i := strings.Index(cond, op); i >= 0 {
			left := strings.TrimSpace(cond[:i])
			right := strings.TrimSpace(cond[i+len(op):])
			lv, err := sfEvalExpr(left, ctx)
			if err != nil {
				return false, err
			}
			rv, err := sfEvalExpr(right, ctx)
			if err != nil {
				return false, err
			}
			equal := sfStringify(lv) == sfStringify(rv)
			if op == "==" {
				return equal, nil
			}
			return !equal, nil
		}
	}
	value, err := sfEvalExpr(cond, ctx)
	if err != nil {
		return false, err
	}
	return sfTruthy(value), nil
}

// sfToIterable converts a value into the list a for-loop iterates. Nil yields
// an empty list; maps iterate sorted keys; strings iterate runes.
func sfToIterable(value any) ([]any, error) {
	if value == nil {
		return []any{}, nil
	}
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Slice, reflect.Array:
		out := make([]any, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			out[i] = rv.Index(i).Interface()
		}
		return out, nil
	case reflect.Map:
		keys := make([]string, 0, rv.Len())
		for _, key := range rv.MapKeys() {
			keys = append(keys, fmt.Sprint(key.Interface()))
		}
		sort.Strings(keys)
		out := make([]any, len(keys))
		for i, key := range keys {
			out[i] = key
		}
		return out, nil
	case reflect.String:
		runes := []rune(rv.String())
		out := make([]any, len(runes))
		for i, r := range runes {
			out[i] = string(r)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("jinja2: value of type %T is not iterable", value)
	}
}

// sfRenderJinja renders a parsed AST with the given context.
func sfRenderJinja(nodes []*sfJinjaNode, ctx map[string]any) (string, error) {
	var out strings.Builder
	if err := sfRenderJinjaInto(nodes, ctx, &out); err != nil {
		return "", err
	}
	return out.String(), nil
}

func sfRenderJinjaInto(nodes []*sfJinjaNode, ctx map[string]any, out *strings.Builder) error {
	for _, node := range nodes {
		switch node.kind {
		case "text":
			out.WriteString(node.text)
		case "var":
			value, err := sfEvalExpr(node.expr, ctx)
			if err != nil {
				return err
			}
			out.WriteString(sfStringify(value))
		case "if":
			cond, err := sfEvalCond(node.cond, ctx)
			if err != nil {
				return err
			}
			if cond {
				if err := sfRenderJinjaInto(node.thenBody, ctx, out); err != nil {
					return err
				}
			} else if err := sfRenderJinjaInto(node.elseBody, ctx, out); err != nil {
				return err
			}
		case "for":
			value, err := sfEvalBase(node.iterable, ctx)
			if err != nil {
				return err
			}
			items, err := sfToIterable(value)
			if err != nil {
				return err
			}
			for i, item := range items {
				child := make(map[string]any, len(ctx)+2)
				for k, v := range ctx {
					child[k] = v
				}
				child[node.varName] = item
				child["loop"] = map[string]any{
					"index":  i + 1,
					"index0": i,
					"first":  i == 0,
					"last":   i == len(items)-1,
				}
				if err := sfRenderJinjaInto(node.loopBody, child, out); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// Jinja2Formatter renders a template using the supported Jinja2 subset.
// Unsupported constructs return an explicit error; no third-party Jinja
// dependency is used.
func Jinja2Formatter(template string, values map[string]any) (string, error) {
	nodes, err := sfParseJinja(template)
	if err != nil {
		return "", err
	}
	if values == nil {
		values = map[string]any{}
	}
	return sfRenderJinja(nodes, values)
}

// Jinja2TemplateVariables returns the sorted, de-duplicated variable names a
// Jinja2 template references. Loop variables and the special "loop" object are
// excluded. Unsupported constructs return an error, mirroring Jinja2Formatter.
func Jinja2TemplateVariables(template string) ([]string, error) {
	nodes, err := sfParseJinja(template)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var walk func([]*sfJinjaNode, map[string]bool)
	walk = func(list []*sfJinjaNode, loopVars map[string]bool) {
		for _, node := range list {
			switch node.kind {
			case "var":
				sfCollectExprVars(node.expr, seen, loopVars)
			case "if":
				sfCollectCondVars(node.cond, seen, loopVars)
				walk(node.thenBody, loopVars)
				walk(node.elseBody, loopVars)
			case "for":
				sfCollectBaseVars(node.iterable, seen, loopVars)
				child := make(map[string]bool, len(loopVars)+2)
				for k, v := range loopVars {
					child[k] = v
				}
				child[node.varName] = true
				child["loop"] = true
				walk(node.loopBody, child)
			}
		}
	}
	walk(nodes, map[string]bool{})
	return sfSortedKeys(seen), nil
}

func sfCollectExprVars(expr string, seen, loopVars map[string]bool) {
	parts := sfSplitPipe(expr)
	sfCollectBaseVars(parts[0], seen, loopVars)
	for _, filter := range parts[1:] {
		name, arg := sfParseFilter(filter)
		if name == "default" {
			sfCollectBaseVars(arg, seen, loopVars)
		}
	}
}

func sfCollectCondVars(cond string, seen, loopVars map[string]bool) {
	cond = strings.TrimSpace(cond)
	if strings.HasPrefix(cond, "not ") {
		sfCollectCondVars(strings.TrimSpace(cond[len("not "):]), seen, loopVars)
		return
	}
	for _, op := range []string{"==", "!="} {
		if i := strings.Index(cond, op); i >= 0 {
			sfCollectExprVars(strings.TrimSpace(cond[:i]), seen, loopVars)
			sfCollectExprVars(strings.TrimSpace(cond[i+len(op):]), seen, loopVars)
			return
		}
	}
	sfCollectExprVars(cond, seen, loopVars)
}

func sfCollectBaseVars(base string, seen, loopVars map[string]bool) {
	base = strings.TrimSpace(base)
	if !sfIdentRe.MatchString(base) {
		return // literals, booleans, numbers
	}
	root := strings.Split(base, ".")[0]
	if root == "loop" || loopVars[root] {
		return
	}
	seen[root] = true
}

// ValidateJinja2 checks that a Jinja2 template only uses supported constructs
// and that its variables exactly match the provided input variables. It returns
// a descriptive error describing any missing or extra variables.
func ValidateJinja2(template string, inputVariables []string) error {
	vars, err := Jinja2TemplateVariables(template)
	if err != nil {
		return err
	}
	missing, extra := sfDiffVars(vars, inputVariables)
	if len(missing) > 0 || len(extra) > 0 {
		return fmt.Errorf("jinja2: invalid prompt schema: missing variables %v, extra variables %v", missing, extra)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// GetTemplateVariables returns the sorted variable names referenced by a
// template for one of the supported formats: "f-string", "mustache", or
// "jinja2".
func GetTemplateVariables(template, format string) ([]string, error) {
	switch format {
	case "jinja2":
		return Jinja2TemplateVariables(template)
	case "f-string":
		return ValidateFStringTemplate(template)
	case "mustache":
		return sfMustacheTemplateVars(template)
	default:
		return nil, fmt.Errorf("unsupported template format %q", format)
	}
}

// CheckValidTemplate checks that a template is well formed for the given format
// and that its variables exactly match inputVariables. An unknown format, a
// template parse error, or a variable mismatch returns an error.
func CheckValidTemplate(template, format string, inputVariables []string) error {
	if _, ok := DefaultFormatterMapping[format]; !ok {
		return fmt.Errorf("invalid template format %q, expected one of %v", format, sfFormatNames())
	}
	vars, err := GetTemplateVariables(template, format)
	if err != nil {
		return err
	}
	missing, extra := sfDiffVars(vars, inputVariables)
	if len(missing) > 0 || len(extra) > 0 {
		return fmt.Errorf("invalid prompt schema: missing variables %v, extra variables %v", missing, extra)
	}
	return nil
}

// IsSubsequence reports whether child is a subsequence (prefix) of parent:
// both must be non-empty, parent must be at least as long as child, and the
// leading elements must match. It mirrors langchain_core's is_subsequence.
func IsSubsequence[T comparable](child, parent []T) bool {
	if len(child) == 0 || len(parent) == 0 || len(parent) < len(child) {
		return false
	}
	for i := range child {
		if child[i] != parent[i] {
			return false
		}
	}
	return true
}

// sfStringify formats a value for interpolation. Booleans and numbers use
// Go's fmt representation; nil renders as an empty string.
func sfStringify(value any) string {
	if value == nil {
		return ""
	}
	switch v := value.(type) {
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprint(v)
	}
}

// sfTruthy mirrors Python truthiness for prompt values.
func sfTruthy(value any) bool {
	if value == nil {
		return false
	}
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Bool:
		return rv.Bool()
	case reflect.String:
		return rv.Len() != 0
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int() != 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return rv.Uint() != 0
	case reflect.Float32, reflect.Float64:
		return rv.Float() != 0
	case reflect.Slice, reflect.Array, reflect.Map:
		return rv.Len() != 0
	case reflect.Ptr, reflect.Interface:
		return !rv.IsNil()
	default:
		return true
	}
}

// sfDiffVars returns the template variables missing from inputVariables and
// the inputVariables not used by the template, both sorted for determinism.
func sfDiffVars(templateVars, inputVariables []string) ([]string, []string) {
	inputSet := map[string]bool{}
	for _, v := range inputVariables {
		inputSet[v] = true
	}
	templateSet := map[string]bool{}
	for _, v := range templateVars {
		templateSet[v] = true
	}
	missing := []string{}
	for _, v := range templateVars {
		if !inputSet[v] {
			missing = append(missing, v)
		}
	}
	extra := []string{}
	for _, v := range inputVariables {
		if !templateSet[v] {
			extra = append(extra, v)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	return missing, extra
}

func sfSortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for key := range set {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func sfFormatNames() []string {
	out := make([]string, 0, len(DefaultFormatterMapping))
	for key := range DefaultFormatterMapping {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}
