// Package mustache implements a spec-compliant subset of the Mustache
// templating language, adapted from LangChain's chevron-based
// langchain_core.utils.mustache module.
//
// It is a standalone package: the only non-standard-library import is
// core/schema, which provides the Schema return type. It intentionally has no
// dependency on core/retrievers or core/language.
//
// Supported syntax:
//   - variables {{name}} with HTML escaping and dotted names {{user.name}}
//   - unescaped variables {{{name}}} and {{& name}}
//   - sections {{#x}}...{{/x}} and inverted sections {{^x}}...{{/x}}
//   - comments {{! ... }}
//   - partials {{>p}}
//   - delimiter changes {{=<% %>=}}
//   - standalone-tag whitespace handling
//
// Partials are resolved from string values in the data map (see Render and
// RenderWithPartials). Malformed tags and unbalanced sections return a
// deterministic *Error value; the package never panics.
package mustache

import (
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/projanvil/langchain-golang/core/schema"
)

// TagType identifies the kind of a Token produced by Tokenize.
type TagType string

// Tag type values emitted by Tokenize. Their string values mirror the tag
// names used by the Python chevron reference so that they can be compared
// directly to string literals when convenient.
const (
	TagLiteral         TagType = "literal"
	TagVariable        TagType = "variable"
	TagSection         TagType = "section"
	TagInvertedSection TagType = "inverted section"
	TagEnd             TagType = "end"
	TagPartial         TagType = "partial"
	TagNoEscape        TagType = "no escape"
	TagComment         TagType = "comment"
	TagSetDelimiter    TagType = "set delimiter"
)

// Inert tag types that are produced while parsing but never affect rendering.
// "no escape?" is a lone "{" tag that is not a confirmed triple-brace tag.
const (
	tagNoEscapeQ     TagType = "no escape?"
	tagSetDelimiterQ TagType = "set delimiter?"
)

// Token is a single unit of a tokenized Mustache template. Key holds the
// literal text for TagLiteral tokens, or the tag key (with sigils stripped)
// for every other tag type.
type Token struct {
	Type TagType
	Key  string
}

// Error describes a Mustache syntax error such as an unclosed tag, an empty
// tag, or an unbalanced section. All errors returned by this package are
// *Error values and can be inspected with errors.As.
type Error struct {
	// Message is a stable, deterministic description of the problem.
	Message string
	// Line is the 1-based line where the problem was detected, or 0 when the
	// location is unknown (for example when calling ParseTag directly).
	Line int
}

// Error implements the error interface.
func (e *Error) Error() string {
	if e == nil {
		return "mustache: <nil>"
	}
	if e.Line > 0 {
		return fmt.Sprintf("mustache: %s (line %d)", e.Message, e.Line)
	}
	return "mustache: " + e.Message
}

// ParseTag parses the tag at the start of text, which must begin immediately
// after the left delimiter. It returns the parsed token, the remaining text
// after the right delimiter, and an error for malformed tags. The lDel and
// rDel arguments are the active delimiters; they are needed to recognize
// triple-brace unescaped tags.
func ParseTag(text, lDel, rDel string) (Token, string, error) {
	return parseTag(text, lDel, rDel, 0)
}

// Tokenize splits a Mustache template into tokens using the default "{{" and
// "}}" delimiters. Delimiter-change tags inside the template are honored for
// all subsequent tags. Comments are dropped, and standalone-tag whitespace is
// stripped following the Mustache spec. Unbalanced sections and malformed
// tags return a deterministic *Error.
func Tokenize(template string) ([]Token, error) {
	return tokenize(template, "{{", "}}")
}

// tokenize is the delimiter-parameterized tokenizer used by Tokenize and by
// the package's own recursive rendering.
type tokenizer struct {
	line         int
	lastTagLine  int
	lDel         string
	rDel         string
	openSections []string
}

func tokenize(template, defLDel, defRDel string) ([]Token, error) {
	tk := &tokenizer{line: 1, lDel: defLDel, rDel: defRDel}
	tokens := []Token{}
	isStandalone := true

	for template != "" {
		literal, rest := grabLiteral(template, tk.lDel, &tk.line)
		template = rest

		// No further left delimiter (or it sits at EOF): the rest is a
		// literal. The Python reference yields this trailing literal here;
		// empty literals are skipped so the token stream stays clean.
		if rest == "" {
			if literal != "" {
				tokens = append(tokens, Token{Type: TagLiteral, Key: literal})
			}
			break
		}

		isStandalone = lSACheck(literal, isStandalone)

		tok, remainder, err := parseTag(template, tk.lDel, tk.rDel, tk.line)
		if err != nil {
			return nil, err
		}
		template = remainder
		tagType, tagKey := tok.Type, tok.Key

		switch tagType {
		case TagSetDelimiter:
			parts := strings.Fields(tagKey)
			if len(parts) == 0 {
				return nil, &Error{Message: "empty set delimiter tag", Line: tk.line}
			}
			tk.lDel, tk.rDel = parts[0], parts[len(parts)-1]
		case TagSection, TagInvertedSection:
			tk.openSections = append(tk.openSections, tagKey)
			tk.lastTagLine = tk.line
		case TagEnd:
			if len(tk.openSections) == 0 {
				return nil, &Error{
					Message: fmt.Sprintf("closing tag %q was never opened", tagKey),
					Line:    tk.line,
				}
			}
			last := tk.openSections[len(tk.openSections)-1]
			tk.openSections = tk.openSections[:len(tk.openSections)-1]
			if tagKey != last {
				return nil, &Error{
					Message: fmt.Sprintf("closing tag %q does not match open section %q", tagKey, last),
					Line:    tk.line,
				}
			}
		}

		isStandalone = rSACheck(template, tagType, isStandalone)
		if isStandalone {
			if i := strings.Index(template, "\n"); i >= 0 {
				template = template[i+1:]
			}
			// Partials keep the indentation on their left; other standalone
			// tags strip the trailing spaces of the preceding literal.
			if tagType != TagPartial {
				literal = strings.TrimRight(literal, " ")
			}
		}

		if literal != "" {
			tokens = append(tokens, Token{Type: TagLiteral, Key: literal})
		}
		if tagType != TagComment && tagType != tagSetDelimiterQ {
			tokens = append(tokens, Token{Type: tagType, Key: tagKey})
		}
	}

	if len(tk.openSections) > 0 {
		return nil, &Error{
			Message: fmt.Sprintf("section %q was never closed", tk.openSections[len(tk.openSections)-1]),
			Line:    tk.lastTagLine,
		}
	}
	return tokens, nil
}

// grabLiteral returns the text up to the next left delimiter and the text
// after it. When there is no left delimiter it returns the whole template and
// an empty remainder. The line counter is advanced by the number of newlines
// in the literal, matching the Python reference.
func grabLiteral(template, lDel string, line *int) (string, string) {
	idx := strings.Index(template, lDel)
	if idx < 0 {
		return template, ""
	}
	literal := template[:idx]
	rest := template[idx+len(lDel):]
	*line += strings.Count(literal, "\n")
	return literal, rest
}

// lSACheck performs the preliminary check for whether a tag could be
// standalone based on the literal to its left and the previous standalone
// state.
func lSACheck(literal string, isStandalone bool) bool {
	if strings.Contains(literal, "\n") || isStandalone {
		padding := literal
		if i := strings.LastIndex(literal, "\n"); i >= 0 {
			padding = literal[i+1:]
		}
		return padding == "" || allSpace(padding)
	}
	return false
}

// rSACheck performs the final check for whether a tag is standalone based on
// the text to its right. Variable and unescaped tags are never standalone.
func rSACheck(template string, tagType TagType, isStandalone bool) bool {
	if isStandalone && tagType != TagVariable && tagType != TagNoEscape {
		firstLine := template
		if i := strings.Index(template, "\n"); i >= 0 {
			firstLine = template[:i]
		}
		return firstLine == "" || allSpace(firstLine)
	}
	return false
}

// parseTag parses one tag from text (which starts just after the left
// delimiter) up to the next right delimiter.
func parseTag(text, lDel, rDel string, line int) (Token, string, error) {
	idx := strings.Index(text, rDel)
	if idx < 0 {
		return Token{}, "", &Error{Message: "unclosed tag", Line: line}
	}
	tag := text[:idx]
	remainder := text[idx+len(rDel):]

	if strings.TrimSpace(tag) == "" {
		return Token{}, "", &Error{Message: "empty tag", Line: line}
	}

	tagType := TagVariable
	switch tag[0] {
	case '!':
		tagType = TagComment
	case '#':
		tagType = TagSection
	case '^':
		tagType = TagInvertedSection
	case '/':
		tagType = TagEnd
	case '>':
		tagType = TagPartial
	case '=':
		tagType = tagSetDelimiterQ
	case '{':
		tagType = tagNoEscapeQ
	case '&':
		tagType = TagNoEscape
	}

	// Every non-variable tag drops its leading sigil.
	if tagType != TagVariable {
		tag = tag[1:]
	}

	switch tagType {
	case tagSetDelimiterQ:
		if !strings.HasSuffix(tag, "=") {
			return Token{}, "", &Error{Message: "unclosed set delimiter tag", Line: line}
		}
		tag = tag[:len(tag)-1]
		tagType = TagSetDelimiter
	case tagNoEscapeQ:
		// A triple-brace tag {{{name}}} is only recognized with the default
		// curly delimiters and an extra "}" immediately after the right delim.
		if lDel == "{{" && rDel == "}}" && strings.HasPrefix(remainder, "}") {
			remainder = remainder[1:]
			tagType = TagNoEscape
		} else {
			tagType = tagNoEscapeQ
		}
	}

	return Token{Type: tagType, Key: strings.TrimSpace(tag)}, remainder, nil
}

// Render renders a Mustache template with the given data. Partials ({{>p}})
// are resolved from the string values of data: data["p"] is rendered as a
// Mustache template within the current scope, and a missing or non-string
// partial renders as empty. Use RenderWithPartials to supply explicit partial
// templates.
func Render(template string, data map[string]any) (string, error) {
	return RenderWithPartials(template, data, partialsFromData(data))
}

// RenderWithPartials renders a Mustache template with data and an explicit
// map of partial templates keyed by partial name.
func RenderWithPartials(template string, data map[string]any, partials map[string]string) (string, error) {
	if data == nil {
		data = map[string]any{}
	}
	if partials == nil {
		partials = map[string]string{}
	}
	tokens, err := Tokenize(template)
	if err != nil {
		return "", err
	}
	root := buildTree(tokens)
	var out strings.Builder
	if err := renderNodes(root, []any{data}, partials, "", 0, &out); err != nil {
		return "", err
	}
	return out.String(), nil
}

// partialsFromData collects the string values of data as partial templates.
func partialsFromData(data map[string]any) map[string]string {
	out := map[string]string{}
	for key, value := range data {
		if text, ok := value.(string); ok {
			out[key] = text
		}
	}
	return out
}

// mNode is a node in the rendering tree built from a token stream.
type mNode struct {
	kind     string
	key      string
	children []*mNode
}

// buildTree converts a balanced token stream into a tree so sections can be
// rendered recursively. Inert tokens (comments, delimiter changes, stray "no
// escape?" tags) are dropped.
func buildTree(tokens []Token) []*mNode {
	var root []*mNode
	type frame struct{ children *[]*mNode }
	stack := []*frame{{children: &root}}
	for _, tok := range tokens {
		cur := stack[len(stack)-1].children
		switch tok.Type {
		case TagEnd:
			if len(stack) > 1 {
				stack = stack[:len(stack)-1]
			}
		case TagSection, TagInvertedSection:
			kind := "section"
			if tok.Type == TagInvertedSection {
				kind = "inverted"
			}
			node := &mNode{kind: kind, key: tok.Key}
			*cur = append(*cur, node)
			stack = append(stack, &frame{children: &node.children})
		case TagLiteral:
			*cur = append(*cur, &mNode{kind: "literal", key: tok.Key})
		case TagVariable:
			*cur = append(*cur, &mNode{kind: "variable", key: tok.Key})
		case TagNoEscape:
			*cur = append(*cur, &mNode{kind: "noescape", key: tok.Key})
		case TagPartial:
			*cur = append(*cur, &mNode{kind: "partial", key: tok.Key})
		}
	}
	return root
}

const maxRenderDepth = 64

// renderNodes renders a node list into out. scopes is a stack of data contexts
// with the current one first. padding is the partial indentation applied to
// literal newlines. depth guards against runaway partial recursion.
func renderNodes(nodes []*mNode, scopes []any, partials map[string]string, padding string, depth int, out *strings.Builder) error {
	if depth > maxRenderDepth {
		return &Error{Message: "partial recursion limit exceeded"}
	}
	for _, node := range nodes {
		switch node.kind {
		case "literal":
			if padding == "" {
				out.WriteString(node.key)
			} else {
				out.WriteString(strings.ReplaceAll(node.key, "\n", "\n"+padding))
			}
		case "variable":
			out.WriteString(htmlEscape(stringify(lookup(node.key, scopes))))
		case "noescape":
			out.WriteString(stringify(lookup(node.key, scopes)))
		case "section":
			value := lookup(node.key, scopes)
			if !truthy(value) {
				continue
			}
			if items, ok := toSlice(value); ok {
				for _, item := range items {
					if err := renderNodes(node.children, prependScope(item, scopes), partials, padding, depth, out); err != nil {
						return err
					}
				}
			} else if err := renderNodes(node.children, prependScope(value, scopes), partials, padding, depth, out); err != nil {
				return err
			}
		case "inverted":
			if !truthy(lookup(node.key, scopes)) {
				if err := renderNodes(node.children, scopes, partials, padding, depth, out); err != nil {
					return err
				}
			}
		case "partial":
			text, ok := partials[node.key]
			if !ok {
				continue
			}
			ptokens, err := Tokenize(text)
			if err != nil {
				return err
			}
			pnodes := buildTree(ptokens)
			left := out.String()
			if i := strings.LastIndex(left, "\n"); i >= 0 {
				left = left[i+1:]
			}
			partPadding := padding
			leftIsSpace := left != "" && allSpace(left)
			if leftIsSpace {
				partPadding += left
			}
			var pb strings.Builder
			if err := renderNodes(pnodes, scopes, partials, partPadding, depth+1, &pb); err != nil {
				return err
			}
			partOut := pb.String()
			if leftIsSpace {
				partOut = strings.TrimRight(partOut, " \t")
			}
			out.WriteString(partOut)
		}
	}
	return nil
}

func prependScope(value any, scopes []any) []any {
	out := make([]any, 0, len(scopes)+1)
	out = append(out, value)
	out = append(out, scopes...)
	return out
}

// lookup resolves a dotted key against the scope stack, mirroring chevron's
// _get_key. The special key "." returns the current scope. Missing keys
// resolve to an empty string.
func lookup(key string, scopes []any) any {
	if key == "." {
		if len(scopes) == 0 {
			return ""
		}
		return scopes[0]
	}
	parts := strings.Split(key, ".")
	for _, scope := range scopes {
		current := scope
		ok := true
		for _, part := range parts {
			current, ok = indexValue(current, part)
			if !ok {
				break
			}
		}
		if ok {
			return current
		}
	}
	return ""
}

// indexValue indexes one dot-separated part into a value. Only string-keyed
// maps, slices, and arrays are traversable; anything else is rejected for
// safety, matching the Python reference's refusal to traverse arbitrary
// objects.
func indexValue(current any, part string) (any, bool) {
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

// toSlice reports whether value is a non-string sequence and returns its
// elements. Maps are deliberately excluded: a section over a map pushes it as
// a scope rather than iterating, matching the Mustache spec.
func toSlice(value any) ([]any, bool) {
	if value == nil {
		return nil, false
	}
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Slice, reflect.Array:
		out := make([]any, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			out[i] = rv.Index(i).Interface()
		}
		return out, true
	default:
		return nil, false
	}
}

// truthy mirrors Python truthiness for the values commonly found in prompt
// data: nil, false, zero, empty strings, and empty containers are falsey.
func truthy(value any) bool {
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
	case reflect.Pointer, reflect.Interface:
		return !rv.IsNil()
	default:
		return true
	}
}

// stringify formats a value for interpolation using the same rules as the
// repository's utils.StringifyValue.
func stringify(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprint(v)
	}
}

// htmlEscape escapes the characters `" & < >` with named entities, matching
// the Python reference's _html_escape (the ampersand must be handled first).
func htmlEscape(value string) string {
	value = strings.ReplaceAll(value, "&", "&amp;")
	value = strings.ReplaceAll(value, "\"", "&quot;")
	value = strings.ReplaceAll(value, "<", "&lt;")
	value = strings.ReplaceAll(value, ">", "&gt;")
	return value
}

// allSpace reports whether s is non-empty and consists only of whitespace.
func allSpace(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return true
}

// Schema builds a JSON-schema-shaped description of the variables a Mustache
// template expects, mirroring langchain_core.prompts.string.mustache_schema.
// Top-level and nested variables become string properties; sections become
// nested object properties. All properties are marked required. Malformed
// templates return a deterministic *Error.
func Schema(template string) (schema.Schema, error) {
	tokens, err := Tokenize(template)
	if err != nil {
		return nil, err
	}

	entries := []*schemaField{}
	byKey := map[string]*schemaField{}
	prefix := []string{}
	var stack [][]string

	setField := func(path []string, wasVar bool) {
		key := strings.Join(path, "\x00")
		if existing, ok := byKey[key]; ok {
			// Python dict assignment keeps insertion order but updates value.
			existing.wasVar = wasVar
			return
		}
		f := &schemaField{path: append([]string(nil), path...), wasVar: wasVar}
		byKey[key] = f
		entries = append(entries, f)
	}

	for _, tok := range tokens {
		if tok.Key == "." {
			continue
		}
		switch tok.Type {
		case TagEnd:
			if len(stack) > 0 {
				prefix = stack[len(stack)-1]
				stack = stack[:len(stack)-1]
			}
		case TagSection, TagInvertedSection:
			stack = append(stack, append([]string(nil), prefix...))
			newPrefix := append(append([]string(nil), prefix...), strings.Split(tok.Key, ".")...)
			prefix = newPrefix
			setField(prefix, false)
		case TagVariable, TagNoEscape:
			path := append(append([]string(nil), prefix...), strings.Split(tok.Key, ".")...)
			setField(path, true)
		}
	}

	for i, f := range entries {
		f.isLeaf = f.wasVar && !isPrefixOfAny(f.path, entries, i)
	}

	defs := &schemaDefs{children: map[string]*schemaDefs{}}
	for i := len(entries) - 1; i >= 0; i-- {
		defs.setLeaf(entries[i].path, entries[i].isLeaf)
	}
	return defs.toSchema(), nil
}

// schemaField is an intermediate field path used while building Schema.
type schemaField struct {
	path   []string
	wasVar bool
	isLeaf bool
}

// isPrefixOfAny reports whether path is a strict prefix of any other field
// path, reproducing Python's is_subsequence check in mustache_schema.
func isPrefixOfAny(path []string, entries []*schemaField, self int) bool {
	for i, other := range entries {
		if i == self {
			continue
		}
		if strictPrefix(path, other.path) {
			return true
		}
	}
	return false
}

// strictPrefix mirrors langchain_core.prompts.string.is_subsequence for path
// tuples: child must be non-empty, no longer than parent, and equal to the
// leading elements of parent.
func strictPrefix(child, parent []string) bool {
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

// schemaDefs is a node in the intermediate defs tree used to build Schema.
// A node with str=true is a string leaf; otherwise children holds nested
// object properties.
type schemaDefs struct {
	children map[string]*schemaDefs
	str      bool
}

// ensureObject walks (creating as needed) the object nodes for parts and
// returns the deepest node.
func (d *schemaDefs) ensureObject(parts []string) *schemaDefs {
	if d.children == nil {
		d.children = map[string]*schemaDefs{}
	}
	current := d
	for _, part := range parts {
		child, ok := current.children[part]
		if !ok || child == nil {
			child = &schemaDefs{children: map[string]*schemaDefs{}}
			current.children[part] = child
		}
		current = child
	}
	return current
}

// setLeaf mirrors the Python reference's setdefault call that installs either
// a string leaf or an object at the last path part. It creates the leaf only
// when the key is not already present.
func (d *schemaDefs) setLeaf(path []string, isLeaf bool) {
	if len(path) == 0 {
		return
	}
	parent := d.ensureObject(path[:len(path)-1])
	key := path[len(path)-1]
	if _, exists := parent.children[key]; exists {
		return
	}
	child := &schemaDefs{}
	if isLeaf {
		child.str = true
	} else {
		child.children = map[string]*schemaDefs{}
	}
	parent.children[key] = child
}

// toSchema converts the defs tree into a schema.Schema. Property keys are
// sorted for deterministic output and every property is marked required.
func (d *schemaDefs) toSchema() schema.Schema {
	if d.str {
		return schema.String("")
	}
	keys := make([]string, 0, len(d.children))
	for key := range d.children {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	props := make(map[string]schema.Schema, len(keys))
	for _, key := range keys {
		props[key] = d.children[key].toSchema()
	}
	return schema.Object(props, keys...)
}
