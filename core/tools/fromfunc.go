package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/projanvil/langchain-golang/core/schema"
)

// Sentinels for FromFunc failures. Callers may use errors.Is to classify a
// rejection (e.g. distinguishing a bad callable from an unsupported field
// type when surfacing configuration errors to a user).
var (
	// ErrNotAFunc is returned when the value passed to FromFunc is not a
	// reflect.Func (or is nil).
	ErrNotAFunc = errors.New("FromFunc: not a function")
	// ErrInvalidSignature is returned when the function signature does not
	// match one of the accepted shapes (see FromFunc).
	ErrInvalidSignature = errors.New("FromFunc: invalid function signature")
	// ErrUnsupportedType is returned when a struct field or map value type
	// cannot be expressed as a JSON-schema type by the reflector.
	ErrUnsupportedType = errors.New("FromFunc: unsupported type")
)

// maxStructDepth bounds how deeply nested structs may be before the
// reflector gives up. Mirrors the "reasonable depth" guidance from the
// Python signature inspector, which does not attempt to model arbitrary
// recursion.
const maxStructDepth = 3

// FromFunc builds a tool from a Go function by reflecting its input
// struct/map into a schema.Schema, mirroring Python's `@tool` decorator over
// a function signature. fn must have one of these shapes:
//
//	func(context.Context, T) (Result, error)        // T is a struct or map[string]any
//	func(context.Context, T) (any, error)
//	func(context.Context) (Result, error)           // no-arg form -> empty object schema
//	func(context.Context) (any, error)
//
// Struct fields are mapped via their `json` tag (falling back to the field
// name). Supported field types: string, int*/uint*, float*, bool, slices and
// arrays of supported types, map[string]V, and nested structs (up to
// maxStructDepth levels deep). Non-pointer, non-omitempty fields are marked
// required, mirroring Python's default-required behaviour.
//
// Known limitations (these are reflected as-is and will NOT round-trip
// through encoding/json; avoid or wrap them):
//   - []byte fields: the schema reflects them as array-of-integer (per
//     uint8), but encoding/json base64-encodes []byte, so the invoker's
//     JSON round-trip will not populate them. Treat []byte fields as
//     unsupported for now.
//   - Embedded (anonymous) struct fields are not flattened (unlike
//     encoding/json); use explicit top-level fields instead.
//   - time.Time fields produce an empty object schema (only unexported
//     fields reflect); avoid or wrap as a string.
func FromFunc(name, description string, fn any) (Func, error) {
	if name == "" {
		return Func{}, fmt.Errorf("tool name is required")
	}
	if fn == nil {
		return Func{}, fmt.Errorf("%w: nil", ErrNotAFunc)
	}
	v := reflect.ValueOf(fn)
	if v.Kind() != reflect.Func {
		return Func{}, fmt.Errorf("%w: got %T", ErrNotAFunc, fn)
	}
	argType, returnsAny, err := fromFuncSignature(v.Type())
	if err != nil {
		return Func{}, err
	}
	argsSchema, err := reflectSchema(argType, 0)
	if err != nil {
		return Func{}, fmt.Errorf("FromFunc(%s): %w", name, err)
	}
	if err := ValidateArgsSchema(argsSchema); err != nil {
		return Func{}, err
	}
	invoker := buildFromFuncInvoker(v, argType, returnsAny)
	return Func{name: name, description: description, argsSchema: argsSchema, fn: invoker}, nil
}

// fromFuncSignature validates the function shape against the accepted
// FromFunc contracts and returns the reflected argument type (nil for the
// no-arg form) and whether the first return value is the generic `any` type
// (vs tools.Result).
func fromFuncSignature(t reflect.Type) (reflect.Type, bool, error) {
	if t.NumIn() < 1 || !isContextType(t.In(0)) {
		return nil, false, fmt.Errorf(
			"%w: first argument must be context.Context", ErrInvalidSignature)
	}
	var argType reflect.Type
	switch t.NumIn() {
	case 1:
		// no-arg form: just context.Context
	case 2:
		argType = t.In(1)
		if err := validateArgType(argType); err != nil {
			return nil, false, err
		}
	default:
		return nil, false, fmt.Errorf(
			"%w: fn must have 1 or 2 inputs (context.Context, [args]), got %d",
			ErrInvalidSignature, t.NumIn())
	}

	if t.NumOut() != 2 {
		return nil, false, fmt.Errorf(
			"%w: fn must return (Result, error) or (any, error), got %d returns",
			ErrInvalidSignature, t.NumOut())
	}
	if !isErrorType(t.Out(1)) {
		return nil, false, fmt.Errorf(
			"%w: fn's second return must be error, got %v", ErrInvalidSignature, t.Out(1))
	}
	returnsAny, err := isAnyReturn(t.Out(0))
	if err != nil {
		return nil, false, fmt.Errorf("%w: %v", ErrInvalidSignature, err)
	}
	return argType, returnsAny, nil
}

// validateArgType restricts the user-argument slot to structs (json-tagged),
// pointers to structs, or map[string]any. Anything else is rejected so
// callers get an upfront, typed error rather than a confusing runtime
// failure during Invoke.
func validateArgType(t reflect.Type) error {
	deref := t
	for deref.Kind() == reflect.Ptr {
		deref = deref.Elem()
	}
	switch deref.Kind() {
	case reflect.Struct:
		return nil
	case reflect.Map:
		if deref.Key().Kind() != reflect.String ||
			deref.Elem().Kind() != reflect.Interface ||
			deref.Elem().NumMethod() != 0 {
			return fmt.Errorf("%w: arg map must be map[string]any, got %v",
				ErrInvalidSignature, t)
		}
		return nil
	default:
		return fmt.Errorf("%w: arg must be a struct or map[string]any, got %v",
			ErrInvalidSignature, t)
	}
}

// reflectSchema converts a reflect.Type into a schema.Schema. A nil type
// (the no-arg form) yields an empty object schema. Unsupported types
// (chan/func/complex/interface{}/deeply nested structs) return
// ErrUnsupportedType wrapped with context.
func reflectSchema(t reflect.Type, depth int) (schema.Schema, error) {
	if t == nil {
		return schema.Object(map[string]schema.Schema{}), nil
	}
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return schema.String(""), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return schema.Integer(""), nil
	case reflect.Float32, reflect.Float64:
		return schema.Number(""), nil
	case reflect.Bool:
		return schema.Boolean(""), nil
	case reflect.Slice, reflect.Array:
		elem, err := reflectSchema(t.Elem(), depth)
		if err != nil {
			return nil, fmt.Errorf("%w: array element %v", ErrUnsupportedType, err)
		}
		return schema.Schema{"type": "array", "items": elem}, nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("%w: map keys must be strings, got %v",
				ErrUnsupportedType, t.Key())
		}
		out := schema.Schema{"type": "object"}
		// Emit additionalProperties for non-any value types so callers get
		// richer metadata. map[string]any stays open-ended (no constraint).
		if elemKind := t.Elem().Kind(); elemKind != reflect.Interface {
			elem, err := reflectSchema(t.Elem(), depth+1)
			if err != nil {
				return nil, fmt.Errorf("%w: map value %v", ErrUnsupportedType, err)
			}
			out["additionalProperties"] = elem
		}
		return out, nil
	case reflect.Struct:
		return reflectStructSchema(t, depth)
	default:
		return nil, fmt.Errorf("%w: kind %v not representable", ErrUnsupportedType, t.Kind())
	}
}

// reflectStructSchema walks the exported fields of a struct type and emits
// a schema.Object. Fields are named via their `json` tag (falling back to
// the field name). A field is considered required unless it is a pointer
// type or carries the `omitempty` json option.
func reflectStructSchema(t reflect.Type, depth int) (schema.Schema, error) {
	if depth > maxStructDepth {
		return nil, fmt.Errorf("%w: struct %v nested deeper than %d levels",
			ErrUnsupportedType, t, maxStructDepth)
	}
	props := make(map[string]schema.Schema)
	var required []string
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		name, omitempty, skip := parseJSONTag(field.Tag.Get("json"))
		if skip {
			continue
		}
		// parseJSONTag returns "" for a missing or options-only tag
		// (e.g. `json:",omitempty"`); fall back to the Go field name so the
		// property is keyed correctly and matches encoding/json's marshalling.
		if name == "" {
			name = field.Name
		}
		fieldSchema, err := reflectSchema(field.Type, depth+1)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", field.Name, err)
		}
		props[name] = fieldSchema
		// Required policy: pointer fields and omitempty fields are optional.
		// Everything else is required, matching Python's default-required
		// behaviour for non-Optional annotations.
		if field.Type.Kind() == reflect.Ptr || omitempty {
			continue
		}
		required = append(required, name)
	}
	return schema.Object(props, required...), nil
}

// parseJSONTag interprets a struct field's `json` tag, returning the
// property name, whether omitempty was set, and whether the field should be
// skipped entirely (json:"-"). The empty tag falls back to the Go field
// name; the caller supplies that name when invoking this helper.
func parseJSONTag(tag string) (name string, omitempty bool, skip bool) {
	if tag == "" {
		return "", false, false
	}
	parts := strings.Split(tag, ",")
	if parts[0] == "-" {
		return "-", false, true
	}
	name = parts[0]
	for _, opt := range parts[1:] {
		if opt == "omitempty" {
			omitempty = true
		}
	}
	return name, omitempty, false
}

// buildFromFuncInvoker produces the closure stored in the resulting Func.
// It materialises a T from the input map via a JSON round-trip, calls the
// user's function, and coerces an `any` return into Result the same way
// when needed.
func buildFromFuncInvoker(
	v reflect.Value,
	argType reflect.Type,
	returnsAny bool,
) func(context.Context, map[string]any) (Result, error) {
	return func(ctx context.Context, input map[string]any) (Result, error) {
		in := make([]reflect.Value, 0, 2)
		in = append(in, reflect.ValueOf(ctx))
		if argType != nil {
			argVal, err := constructArg(input, argType)
			if err != nil {
				return Result{}, err
			}
			in = append(in, argVal)
		}
		out := v.Call(in)
		errVal, _ := out[1].Interface().(error)
		if errVal != nil {
			return Result{}, errVal
		}
		if returnsAny {
			return coerceAnyToResult(out[0].Interface())
		}
		result, _ := out[0].Interface().(Result)
		return result, nil
	}
}

// constructArg marshals the input map and unmarshals it into a fresh T,
// returning a reflect.Value suitable for passing to Func.Call. Pointer
// argument types are preserved.
func constructArg(input map[string]any, t reflect.Type) (reflect.Value, error) {
	raw, err := json.Marshal(input)
	if err != nil {
		return reflect.Value{}, fmt.Errorf("marshal args: %w", err)
	}
	elemType := t
	if t.Kind() == reflect.Ptr {
		elemType = t.Elem()
	}
	ptr := reflect.New(elemType)
	if err := json.Unmarshal(raw, ptr.Interface()); err != nil {
		return reflect.Value{}, fmt.Errorf("unmarshal args into %v: %w", t, err)
	}
	if t.Kind() == reflect.Ptr {
		return ptr, nil
	}
	return ptr.Elem(), nil
}

// coerceAnyToResult converts an arbitrary return value into a Result. Plain
// scalars (string, []byte, numbers, bools) cannot round-trip through a
// struct - encoding/json rejects a top-level non-object JSON value - so
// they map directly to Content. Composite types (maps, structs, slices)
// take a JSON round-trip so that user-defined json tags land on Result's
// Content/Artifact/Metadata fields.
func coerceAnyToResult(v any) (Result, error) {
	if v == nil {
		return Result{}, nil
	}
	if r, ok := v.(Result); ok {
		return r, nil
	}
	switch x := v.(type) {
	case string:
		return Result{Content: x}, nil
	case []byte:
		return Result{Content: string(x)}, nil
	case int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		float32, float64, bool:
		return Result{Content: fmt.Sprint(x)}, nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return Result{}, fmt.Errorf("marshal result: %w", err)
	}
	var r Result
	if err := json.Unmarshal(raw, &r); err != nil {
		return Result{}, fmt.Errorf("unmarshal result: %w", err)
	}
	return r, nil
}

// isContextType reports whether t is context.Context.
func isContextType(t reflect.Type) bool {
	return t == reflect.TypeOf((*context.Context)(nil)).Elem()
}

// isErrorType reports whether t satisfies the error interface.
func isErrorType(t reflect.Type) bool {
	return t == reflect.TypeOf((*error)(nil)).Elem()
}

// isAnyReturn reports whether t is tools.Result or the empty interface. It
// returns true for `any`, false for Result, and an error for anything else.
func isAnyReturn(t reflect.Type) (bool, error) {
	if t == reflect.TypeOf(Result{}) {
		return false, nil
	}
	if t.Kind() == reflect.Interface && t.NumMethod() == 0 {
		return true, nil
	}
	return false, fmt.Errorf("fn's first return must be tools.Result or any, got %v", t)
}
