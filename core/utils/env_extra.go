package utils

import (
	"fmt"
	"os"
	"reflect"
	"strings"
)

// EnvVarIsSet reports whether name is present in the environment with a value
// that LangChain treats as "set". Mirroring Python's env_var_is_set, the values
// "", "0", "false" and "False" are considered unset; any other present value,
// including "FALSE" or "no", is considered set. A variable that is not present
// at all is unset.
func EnvVarIsSet(name string) bool {
	value, ok := os.LookupEnv(name)
	if !ok {
		return false
	}
	switch value {
	case "", "0", "false", "False":
		return false
	}
	return true
}

// GetFromDictOrEnv resolves a string value by looking in values under key
// first, then in the environment variables named by envKeys in order. A plain
// string value is used directly; a SecretStr-like wrapper is revealed through
// its SecretValue() (or GetSecretValue(), Reveal(), or String()) method. Empty
// values are treated as missing so the lookup falls through. When no value is
// found, a deterministic error listing every checked key is returned; this
// function never panics.
func GetFromDictOrEnv(values map[string]any, key string, envKeys ...string) (string, error) {
	if raw, ok := values[key]; ok {
		if value, ok := smallBExtractString(raw); ok && value != "" {
			return value, nil
		}
	}
	for _, envKey := range envKeys {
		if value := os.Getenv(envKey); value != "" {
			return value, nil
		}
	}
	return "", smallBMissingKeyError(key, envKeys)
}

// smallBExtractString extracts a usable string from a map value. Reveal-style
// methods are checked before fmt.Stringer so that a masked String() on a
// secret wrapper is never returned in place of the real secret. Typed nil
// values are rejected up front to avoid calling methods on a nil pointer.
func smallBExtractString(value any) (string, bool) {
	if value == nil {
		return "", false
	}
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		if rv.IsNil() {
			return "", false
		}
	}
	switch v := value.(type) {
	case string:
		return v, true
	}
	switch v := value.(type) {
	case interface{ SecretValue() string }:
		return v.SecretValue(), true
	case interface{ GetSecretValue() string }:
		return v.GetSecretValue(), true
	case interface{ Reveal() string }:
		return v.Reveal(), true
	case fmt.Stringer:
		return v.String(), true
	case []byte:
		return string(v), true
	}
	return "", false
}

// smallBMissingKeyError builds a deterministic error that lists the map key
// and every environment variable key that was checked, in a stable order.
func smallBMissingKeyError(key string, envKeys []string) error {
	checked := make([]string, 0, 1+len(envKeys))
	checked = append(checked, fmt.Sprintf("%q (map key)", key))
	for _, envKey := range envKeys {
		checked = append(checked, fmt.Sprintf("%q (env)", envKey))
	}
	return fmt.Errorf("did not find a value for key %q; checked keys: %s", key, strings.Join(checked, ", "))
}
