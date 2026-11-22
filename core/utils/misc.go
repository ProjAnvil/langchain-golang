package utils

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"reflect"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/projanvil/langchain-golang/core/schema"
)

// XorArgs validates that each group of mutually exclusive arguments has
// exactly one member set. Each group is a map from argument name to a boolean
// indicating whether that argument was supplied. It is the Go-idiomatic
// equivalent of Python's xor_args decorator: Go has no decorators or kwargs,
// so callers resolve argument presence into maps and pass the groups directly.
func XorArgs(groups ...map[string]bool) error {
	invalid := []string{}
	for _, group := range groups {
		count := 0
		for _, set := range group {
			if set {
				count++
			}
		}
		if count != 1 {
			keys := make([]string, 0, len(group))
			for k := range group {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			invalid = append(invalid, strings.Join(keys, ", "))
		}
	}
	if len(invalid) > 0 {
		return fmt.Errorf("exactly one argument in each of the following groups must be defined: %s", strings.Join(invalid, "; "))
	}
	return nil
}

// BuildExtraKwargs folds unknown constructor values into extra kwargs. It is
// the Go port of Python's build_extra_kwargs: any key in values that is not a
// known field is moved into extraKwargs (and removed from values); a key
// present in both maps is an error, as is a known field supplied through
// extraKwargs. Python emits warnings for the moved keys, which Go omits in
// favour of the returned values. Both input maps are mutated, matching Python.
func BuildExtraKwargs(extraKwargs map[string]any, values map[string]any, knownFields map[string]bool) (map[string]any, error) {
	if extraKwargs == nil {
		extraKwargs = map[string]any{}
	}
	for fieldName := range values {
		if _, dup := extraKwargs[fieldName]; dup {
			return nil, fmt.Errorf("found %s supplied twice", fieldName)
		}
		if !knownFields[fieldName] {
			extraKwargs[fieldName] = values[fieldName]
			delete(values, fieldName)
		}
	}
	invalid := []string{}
	for k := range extraKwargs {
		if knownFields[k] {
			invalid = append(invalid, k)
		}
	}
	if len(invalid) > 0 {
		sort.Strings(invalid)
		return nil, fmt.Errorf("parameters %s should be specified explicitly. Instead they were passed in as part of the model_kwargs parameter", strings.Join(invalid, ", "))
	}
	return extraKwargs, nil
}

// largeUtilClock is the process-wide, injectable clock read by Now. Go has no
// way to monkey-patch a datetime class as Python's mock_now context manager
// does, so a replaceable clock function is the idiomatic substitute.
var largeUtilClock = struct {
	sync.RWMutex
	now func() time.Time
}{now: time.Now}

// Now returns the current time from the injectable package clock.
func Now() time.Time {
	largeUtilClock.RLock()
	now := largeUtilClock.now
	largeUtilClock.RUnlock()
	return now()
}

// MockNow installs a fixed clock and returns both the injectable clock
// function and a restore function. It is the Go-idiomatic equivalent of
// Python's mock_now: instead of patching datetime.datetime.now globally, the
// fixed clock is returned for injection into code that accepts a
// func() time.Time dependency, and restore resets the package clock used by
// Now. The restore function is idempotent.
func MockNow(fixed time.Time) (clock func() time.Time, restore func()) {
	clock = func() time.Time { return fixed }
	largeUtilClock.Lock()
	previous := largeUtilClock.now
	largeUtilClock.now = clock
	largeUtilClock.Unlock()

	var once sync.Once
	restore = func() {
		once.Do(func() {
			largeUtilClock.Lock()
			largeUtilClock.now = previous
			largeUtilClock.Unlock()
		})
	}
	return clock, restore
}

// GetPydanticFieldNames returns the set of field names for a struct type,
// including aliases. It is the Go analogue of Python's
// get_pydantic_field_names, which reads a Pydantic model's fields and aliases:
// Go has no Pydantic models, so reflection over exported struct fields,
// their json tag names, and an optional alias tag is used instead.
func GetPydanticFieldNames(model any) (map[string]bool, error) {
	t := largeUtilStructType(model)
	if t == nil {
		return nil, fmt.Errorf("get field names: expected a struct or pointer to struct, got %T", model)
	}
	out := map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		out[field.Name] = true
		if tag := field.Tag.Get("json"); tag != "" {
			name := strings.Split(tag, ",")[0]
			if name != "" && name != "-" {
				out[name] = true
			}
		}
		if alias := field.Tag.Get("alias"); alias != "" {
			out[alias] = true
		}
	}
	return out, nil
}

// CreateSubsetModel builds an object schema containing only the selected
// fields of a Go struct type. It is the Go-idiomatic equivalent of Pydantic's
// create_model / subset-model helpers: Python dynamically generates a new
// model class at runtime, which Go cannot do, so instead this function derives
// a schema.Schema from the injected struct type via reflection (mapping Go
// kinds to JSON-schema types) and delegates the field projection to
// CreateSubsetSchema. Field names are the json tag names (or Go field names).
func CreateSubsetModel(name string, model any, fieldNames []string, descriptions map[string]string, fnDescription string) (schema.Schema, error) {
	t := largeUtilStructType(model)
	if t == nil {
		return nil, fmt.Errorf("create subset model: expected a struct or pointer to struct, got %T", model)
	}
	full := largeUtilStructToObjectSchema(t)
	return CreateSubsetSchema(name, full, fieldNames, descriptions, fnDescription)
}

// largeUtilStructType returns the struct type behind a value or pointer.
func largeUtilStructType(model any) reflect.Type {
	if model == nil {
		return nil
	}
	t := reflect.TypeOf(model)
	for t != nil && t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t == nil || t.Kind() != reflect.Struct {
		return nil
	}
	return t
}

// largeUtilStructToObjectSchema derives a full object schema from a struct.
func largeUtilStructToObjectSchema(t reflect.Type) schema.Schema {
	props := map[string]schema.Schema{}
	required := []string{}
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}
		name, omitempty, skip := largeUtilJSONFieldName(field)
		if skip {
			continue
		}
		prop := largeUtilFieldSchema(field.Type)
		if desc := field.Tag.Get("description"); desc != "" {
			prop["description"] = desc
		}
		props[name] = prop
		if !omitempty {
			required = append(required, name)
		}
	}
	return schema.Object(props, required...)
}

// largeUtilJSONFieldName resolves a struct field's JSON property name.
func largeUtilJSONFieldName(field reflect.StructField) (name string, omitempty bool, skip bool) {
	name = field.Name
	tag := field.Tag.Get("json")
	if tag == "" {
		return name, false, false
	}
	parts := strings.Split(tag, ",")
	if parts[0] == "-" {
		return name, false, true
	}
	if parts[0] != "" {
		name = parts[0]
	}
	for _, part := range parts[1:] {
		if part == "omitempty" {
			omitempty = true
		}
	}
	return name, omitempty, false
}

// largeUtilFieldSchema maps a Go type to a small JSON schema.
func largeUtilFieldSchema(t reflect.Type) schema.Schema {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.Bool:
		return schema.Schema{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return schema.Schema{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return schema.Schema{"type": "number"}
	case reflect.Slice, reflect.Array:
		return schema.Schema{"type": "array", "items": largeUtilFieldSchema(t.Elem())}
	case reflect.Map, reflect.Struct, reflect.Interface:
		return schema.Schema{"type": "object"}
	default:
		return schema.Schema{"type": "string"}
	}
}

// SecretStr wraps a sensitive string so it is masked by default formatting
// and JSON serialization. It is the Go analogue of pydantic.SecretStr: the
// real value is only available through SecretValue.
type SecretStr struct {
	value string
}

// SecretValue returns the underlying sensitive string.
func (s SecretStr) SecretValue() string { return s.value }

// String masks the secret, mirroring pydantic.SecretStr's str() behaviour.
func (s SecretStr) String() string {
	if s.value == "" {
		return ""
	}
	return "**********"
}

// GoString keeps %#v from leaking the secret.
func (s SecretStr) GoString() string { return s.String() }

// MarshalJSON emits the masked value so secrets are not leaked when a
// SecretStr is serialized as part of a larger structure.
func (s SecretStr) MarshalJSON() ([]byte, error) { return json.Marshal(s.String()) }

// ConvertToSecretStr wraps a string (or returns an existing SecretStr) as a
// SecretStr. Other types are rejected with an error instead of panicking. It
// mirrors Python's convert_to_secret_str(SecretStr | str).
func ConvertToSecretStr(value any) (SecretStr, error) {
	switch v := value.(type) {
	case SecretStr:
		return v, nil
	case *SecretStr:
		if v == nil {
			return SecretStr{}, fmt.Errorf("convert to secret str: nil *SecretStr")
		}
		return *v, nil
	case string:
		return SecretStr{value: v}, nil
	default:
		return SecretStr{}, fmt.Errorf("convert to secret str: unsupported type %T", value)
	}
}

// EnvOptions configures the FromEnv / SecretFromEnv factories. HasDefault
// distinguishes an explicitly empty default from "no default"; when no key is
// present and HasDefault is false the factory returns an error.
type EnvOptions struct {
	Default      string
	HasDefault   bool
	ErrorMessage string
}

// FromEnv returns a factory that reads the first present environment variable
// from keys. It is the Go-idiomatic equivalent of Python's from_env: instead of
// returning a callable that raises ValueError, the factory returns
// (string, error), and an options struct replaces keyword-only arguments.
func FromEnv(keys []string, opts ...EnvOptions) func() (string, error) {
	o := largeUtilEnvOptions(opts)
	return func() (string, error) {
		if v, ok := largeUtilLookupEnv(keys); ok {
			return v, nil
		}
		if o.HasDefault {
			return o.Default, nil
		}
		return "", largeUtilMissingEnvError(keys, o.ErrorMessage)
	}
}

// SecretFromEnv is the SecretStr-returning counterpart of FromEnv, mirroring
// Python's secret_from_env.
func SecretFromEnv(keys []string, opts ...EnvOptions) func() (SecretStr, error) {
	o := largeUtilEnvOptions(opts)
	return func() (SecretStr, error) {
		if v, ok := largeUtilLookupEnv(keys); ok {
			return SecretStr{value: v}, nil
		}
		if o.HasDefault {
			return SecretStr{value: o.Default}, nil
		}
		return SecretStr{}, largeUtilMissingEnvError(keys, o.ErrorMessage)
	}
}

func largeUtilEnvOptions(opts []EnvOptions) EnvOptions {
	if len(opts) > 0 {
		return opts[0]
	}
	return EnvOptions{}
}

// largeUtilLookupEnv returns the first present key, including an empty but
// explicitly set value (matching Python's "key in os.environ" check).
func largeUtilLookupEnv(keys []string) (string, bool) {
	for _, k := range keys {
		if v, ok := os.LookupEnv(k); ok {
			return v, true
		}
	}
	return "", false
}

func largeUtilMissingEnvError(keys []string, custom string) error {
	if custom != "" {
		return errors.New(custom)
	}
	joined := strings.Join(keys, ", ")
	return fmt.Errorf("did not find %s, please add an environment variable `%s` which contains it, or pass it as a named parameter", joined, joined)
}

// RaiseForStatusWithText returns an error containing the HTTP response body
// when the status code indicates an error (4xx/5xx). It is the Go equivalent of
// Python's raise_for_status_with_text: instead of calling requests'
// raise_for_status and re-raising ValueError(response.text), it inspects
// StatusCode and reads Body directly. A 2xx/3xx response yields nil.
func RaiseForStatusWithText(resp *http.Response) error {
	if resp == nil {
		return errors.New("raise for status with text: nil HTTP response")
	}
	if resp.StatusCode < 400 {
		return nil
	}
	var text string
	if resp.Body != nil {
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("raise for status with text: read body: %w", err)
		}
		text = string(data)
	}
	if text != "" {
		return errors.New(text)
	}
	return fmt.Errorf("HTTP %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode))
}

// DependencyChecker reports the version of a named dependency and whether it
// is available. It is the injectable seam for GuardImport and
// CheckPackageVersion, allowing tests (and alternative build systems) to
// supply their own dependency registry instead of relying on build info.
type DependencyChecker func(name string) (version string, ok bool)

// GuardImport verifies that a Go module dependency is present and returns its
// version. It is the Go-idiomatic equivalent of Python's guard_import: Go
// resolves imports at build time, so instead of importlib.import_module it
// inspects the embedded module build info. The dependency lookup is
// injectable through the optional DependencyChecker; pass a checker to adapt
// the check to your environment.
func GuardImport(moduleName string, checkers ...DependencyChecker) (string, error) {
	if moduleName == "" {
		return "", fmt.Errorf("guard import: module name is required")
	}
	check := largeUtilSelectChecker(checkers)
	version, ok := check(moduleName)
	if !ok {
		return "", fmt.Errorf("could not import %s go package. Please add it with `go get %s`.", moduleName, moduleName)
	}
	return version, nil
}

// VersionConstraints holds optional upper/lower version bounds, replacing the
// four optional keyword arguments of Python's check_package_version.
type VersionConstraints struct {
	LT  string // version must be < LT
	LTE string // version must be <= LTE
	GT  string // version must be > GT
	GTE string // version must be >= GTE
}

// CheckPackageVersion validates a dependency's version against optional
// bounds. It is the Go-idiomatic equivalent of Python's check_package_version:
// the version is supplied by an injectable DependencyChecker (or read from
// build info) rather than importlib.metadata, and a simplified dotted-numeric
// comparator avoids adding a semver dependency.
func CheckPackageVersion(packageName string, constraints VersionConstraints, checkers ...DependencyChecker) error {
	check := largeUtilSelectChecker(checkers)
	version, ok := check(packageName)
	if !ok {
		return fmt.Errorf("check package version: package %s is not available", packageName)
	}

	if constraints.LT != "" {
		cmp, err := largeUtilCompareVersions(version, constraints.LT)
		if err != nil {
			return fmt.Errorf("check package version: %w", err)
		}
		if cmp >= 0 {
			return fmt.Errorf("expected %s version to be < %s. Received %s.", packageName, constraints.LT, version)
		}
	}
	if constraints.LTE != "" {
		cmp, err := largeUtilCompareVersions(version, constraints.LTE)
		if err != nil {
			return fmt.Errorf("check package version: %w", err)
		}
		if cmp > 0 {
			return fmt.Errorf("expected %s version to be <= %s. Received %s.", packageName, constraints.LTE, version)
		}
	}
	if constraints.GT != "" {
		cmp, err := largeUtilCompareVersions(version, constraints.GT)
		if err != nil {
			return fmt.Errorf("check package version: %w", err)
		}
		if cmp <= 0 {
			return fmt.Errorf("expected %s version to be > %s. Received %s.", packageName, constraints.GT, version)
		}
	}
	if constraints.GTE != "" {
		cmp, err := largeUtilCompareVersions(version, constraints.GTE)
		if err != nil {
			return fmt.Errorf("check package version: %w", err)
		}
		if cmp < 0 {
			return fmt.Errorf("expected %s version to be >= %s. Received %s.", packageName, constraints.GTE, version)
		}
	}
	return nil
}

func largeUtilSelectChecker(checkers []DependencyChecker) DependencyChecker {
	for _, c := range checkers {
		if c != nil {
			return c
		}
	}
	return largeUtilBuildInfoChecker
}

// largeUtilBuildInfoChecker looks a module up in the binary's embedded build
// info; it is the default dependency registry.
func largeUtilBuildInfoChecker(name string) (string, bool) {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "", false
	}
	if info.Main.Path == name {
		return info.Main.Version, true
	}
	for _, dep := range info.Deps {
		if dep.Path == name {
			return dep.Version, true
		}
	}
	return "", false
}

// largeUtilCompareVersions compares dotted-numeric versions, returning -1, 0,
// or 1. A leading "v" and pre-release/build suffixes are ignored, and missing
// segments are treated as zero. This is intentionally simpler than full
// semver to avoid a new module dependency.
func largeUtilCompareVersions(a, b string) (int, error) {
	pa, err := largeUtilParseVersion(a)
	if err != nil {
		return 0, err
	}
	pb, err := largeUtilParseVersion(b)
	if err != nil {
		return 0, err
	}
	n := len(pa)
	if len(pb) > n {
		n = len(pb)
	}
	for i := 0; i < n; i++ {
		x, y := 0, 0
		if i < len(pa) {
			x = pa[i]
		}
		if i < len(pb) {
			y = pb[i]
		}
		if x < y {
			return -1, nil
		}
		if x > y {
			return 1, nil
		}
	}
	return 0, nil
}

func largeUtilParseVersion(v string) ([]int, error) {
	raw := v
	v = strings.TrimSpace(v)
	v = strings.TrimPrefix(v, "v")
	if i := strings.IndexAny(v, "-+"); i >= 0 {
		v = v[:i]
	}
	if v == "" {
		return nil, fmt.Errorf("invalid version %q", raw)
	}
	parts := strings.Split(v, ".")
	out := make([]int, len(parts))
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid version %q: %w", raw, err)
		}
		out[i] = n
	}
	return out, nil
}
