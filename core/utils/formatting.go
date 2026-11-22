package utils

import (
	"fmt"
	"reflect"
	"strings"
	"unicode"
	"unicode/utf8"
)

// StrictFormatter formats templates using keyword-only substitution. It is the
// Go port of LangChain's StrictFormatter: positional placeholders such as {}
// and {0} are rejected, every replacement field must be a named identifier,
// and both Format and ValidateInputVariables report variable mismatches as
// errors instead of panicking. The zero value is ready to use and stateless.
type StrictFormatter struct{}

// smallASpec is a parsed, deliberately small subset of a Python format spec:
// an optional fill, an optional alignment, and an optional integer width.
type smallASpec struct {
	fill  rune
	align rune
	width int
}

// smallAField is a parsed replacement field.
type smallAField struct {
	name string
	conv string
	spec smallASpec
}

// smallAToken is one literal run or one replacement field in a template.
type smallAToken struct {
	literal string
	isField bool
	field   smallAField
}

// Format substitutes named replacement fields in template using kwargs. It
// returns an error (never panics) on positional placeholders, malformed
// fields, unsupported conversions or format specs, and on any placeholder
// whose name is not present in kwargs. Additional, unused keys in kwargs are
// ignored. Supported conversions are !r, !s, and !a; supported format specs
// are an optional fill, one of < > ^ = alignment, and an integer width (for
// example {name:>10} or {name:*^12}). Anything else yields a clear error.
func (StrictFormatter) Format(template string, kwargs map[string]any) (string, error) {
	tokens, err := smallAParseTemplate(template)
	if err != nil {
		return "", err
	}
	if kwargs == nil {
		kwargs = map[string]any{}
	}
	var b strings.Builder
	for _, token := range tokens {
		if !token.isField {
			b.WriteString(token.literal)
			continue
		}
		value, ok := kwargs[token.field.name]
		if !ok {
			return "", fmt.Errorf("format: missing keyword argument %q (unknown variable)", token.field.name)
		}
		rendered, err := smallAConvert(value, token.field.conv)
		if err != nil {
			return "", err
		}
		defaultAlign := '<'
		if token.field.conv == "" && smallAIsNumeric(value) {
			defaultAlign = '>'
		}
		b.WriteString(smallAApplySpec(rendered, token.field.spec, defaultAlign))
	}
	return b.String(), nil
}

// ValidateInputVariables checks that every replacement field in template is
// declared in variables. Placeholders that are not listed produce an error.
// Declared variables that the template does not use are allowed, matching
// LangChain's StrictFormatter. Positional placeholders, unsupported
// conversions, and unsupported format specs also produce errors.
func (StrictFormatter) ValidateInputVariables(template string, variables []string) error {
	tokens, err := smallAParseTemplate(template)
	if err != nil {
		return err
	}
	allowed := make(map[string]bool, len(variables))
	for _, name := range variables {
		allowed[name] = true
	}
	for _, token := range tokens {
		if !token.isField {
			continue
		}
		if !allowed[token.field.name] {
			return fmt.Errorf("format: unknown input variable %q (not declared in variables)", token.field.name)
		}
	}
	return nil
}

// smallAParseTemplate splits a template into literal tokens and parsed
// replacement fields, resolving {{ and }} escapes.
func smallAParseTemplate(template string) ([]smallAToken, error) {
	var tokens []smallAToken
	var literal strings.Builder
	flush := func() {
		if literal.Len() > 0 {
			tokens = append(tokens, smallAToken{literal: literal.String()})
			literal.Reset()
		}
	}
	for i := 0; i < len(template); {
		c := template[i]
		switch c {
		case '{':
			if i+1 < len(template) && template[i+1] == '{' {
				literal.WriteByte('{')
				i += 2
				continue
			}
			rel := strings.IndexByte(template[i+1:], '}')
			if rel < 0 {
				return nil, fmt.Errorf("format: unmatched '{' in template")
			}
			content := template[i+1 : i+1+rel]
			field, err := smallAParseField(content)
			if err != nil {
				return nil, err
			}
			flush()
			tokens = append(tokens, smallAToken{isField: true, field: field})
			i = i + 1 + rel + 1
		case '}':
			if i+1 < len(template) && template[i+1] == '}' {
				literal.WriteByte('}')
				i += 2
				continue
			}
			return nil, fmt.Errorf("format: single '}' in template")
		default:
			literal.WriteByte(c)
			i++
		}
	}
	flush()
	return tokens, nil
}

// smallAParseField parses one replacement field body (without braces) into a
// name, an optional conversion, and a validated format spec.
func smallAParseField(content string) (smallAField, error) {
	var field smallAField
	sep := strings.IndexAny(content, "!:")
	name := content
	rest := ""
	if sep >= 0 {
		name = content[:sep]
		rest = content[sep:]
	}
	if err := smallAValidateName(name); err != nil {
		return field, err
	}
	field.name = name

	if strings.HasPrefix(rest, "!") {
		if len(rest) < 2 {
			return field, fmt.Errorf("format: empty conversion in placeholder {%s}", content)
		}
		conv := rest[1:2]
		switch conv {
		case "r", "s", "a":
			field.conv = conv
		default:
			return field, fmt.Errorf("format: unsupported conversion %q in placeholder {%s}", conv, content)
		}
		rest = rest[2:]
		if rest != "" {
			if !strings.HasPrefix(rest, ":") {
				return field, fmt.Errorf("format: expected ':' after conversion in placeholder {%s}", content)
			}
			rest = rest[1:]
		}
	} else if strings.HasPrefix(rest, ":") {
		rest = rest[1:]
	}

	spec, err := smallAParseSpec(rest)
	if err != nil {
		return field, err
	}
	field.spec = spec
	return field, nil
}

// smallAValidateName enforces keyword-only, identifier-shaped field names.
func smallAValidateName(name string) error {
	if name == "" {
		return fmt.Errorf("format: positional-style placeholder {} is not allowed; use keyword names")
	}
	if smallAAllDigits(name) {
		return fmt.Errorf("format: positional-style placeholder {%s} is not allowed; use keyword names", name)
	}
	for i, r := range name {
		if r == '_' || unicode.IsLetter(r) || (i > 0 && unicode.IsDigit(r)) {
			continue
		}
		return fmt.Errorf("format: invalid field name %q; attribute and index access are unsupported", name)
	}
	return nil
}

// smallAAllDigits reports whether s consists solely of ASCII digits.
func smallAAllDigits(s string) bool {
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

// smallAIsAlign reports whether r is a supported alignment character.
func smallAIsAlign(r rune) bool {
	return r == '<' || r == '>' || r == '^' || r == '='
}

// smallAParseSpec parses the supported subset of a format spec. Any other
// characters (precision, type, sign, grouping, and so on) are rejected.
func smallAParseSpec(spec string) (smallASpec, error) {
	out := smallASpec{fill: ' '}
	if spec == "" {
		return out, nil
	}
	runes := []rune(spec)
	i := 0
	if smallAIsAlign(runes[0]) {
		out.align = runes[0]
		i = 1
	} else if len(runes) >= 2 && smallAIsAlign(runes[1]) {
		out.fill = runes[0]
		out.align = runes[1]
		i = 2
	}
	for ; i < len(runes); i++ {
		if runes[i] < '0' || runes[i] > '9' {
			return smallASpec{}, fmt.Errorf("format: unsupported format spec %q", spec)
		}
		out.width = out.width*10 + int(runes[i]-'0')
	}
	return out, nil
}

// smallAConvert renders a value with an optional conversion.
func smallAConvert(value any, conv string) (string, error) {
	switch conv {
	case "":
		return fmt.Sprint(value), nil
	case "s":
		return fmt.Sprint(value), nil
	case "r":
		return fmt.Sprintf("%#v", value), nil
	case "a":
		return smallAASCIIEscape(fmt.Sprintf("%#v", value)), nil
	default:
		return "", fmt.Errorf("format: unsupported conversion %q", conv)
	}
}

// smallAIsNumeric reports whether value has a numeric Go kind, used to pick
// Python's default right alignment for numbers.
func smallAIsNumeric(value any) bool {
	switch reflect.ValueOf(value).Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64:
		return true
	default:
		return false
	}
}

// smallAApplySpec pads a rendered string to the spec width using the spec
// alignment (or defaultAlign when no alignment was given).
func smallAApplySpec(rendered string, spec smallASpec, defaultAlign rune) string {
	if spec.width <= 0 {
		return rendered
	}
	pad := spec.width - utf8.RuneCountInString(rendered)
	if pad <= 0 {
		return rendered
	}
	fill := spec.fill
	if fill == 0 {
		fill = ' '
	}
	align := spec.align
	if align == 0 {
		align = defaultAlign
	}
	padding := strings.Repeat(string(fill), pad)
	switch align {
	case '>':
		return padding + rendered
	case '^':
		left := pad / 2
		return strings.Repeat(string(fill), left) + rendered + strings.Repeat(string(fill), pad-left)
	case '=':
		if len(rendered) > 0 && (rendered[0] == '+' || rendered[0] == '-') {
			return rendered[:1] + padding + rendered[1:]
		}
		return padding + rendered
	default:
		return rendered + padding
	}
}

// smallAASCIIEscape replaces non-ASCII runes with Go-style \u / \U escapes so
// the !a conversion produces ASCII-only output.
func smallAASCIIEscape(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r < 128 {
			b.WriteRune(r)
		} else if r <= 0xFFFF {
			fmt.Fprintf(&b, "\\u%04x", r)
		} else {
			fmt.Fprintf(&b, "\\U%08x", r)
		}
	}
	return b.String()
}
