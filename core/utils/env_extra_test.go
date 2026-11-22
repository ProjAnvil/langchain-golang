package utils

import (
	"strings"
	"testing"
)

// smallBSecretWrapper models a SecretStr-like value. SecretValue returns the
// real secret while String returns a masked value, so the tests can prove that
// extraction prefers the real secret over the masked String representation.
type smallBSecretWrapper struct {
	value string
}

func (s smallBSecretWrapper) SecretValue() string { return s.value }
func (s smallBSecretWrapper) String() string      { return "**********" }
func (s smallBSecretWrapper) GoString() string    { return s.String() }

// smallBStringer models a wrapper that only exposes fmt.Stringer.
type smallBStringer struct {
	value string
}

func (s smallBStringer) String() string { return s.value }

func TestEnvVarIsSetNotPresent(t *testing.T) {
	if EnvVarIsSet("LC_ENV_EXTRA_NOT_PRESENT") {
		t.Fatal("EnvVarIsSet should be false for a variable that is not present")
	}
}

func TestEnvVarIsSetFalsyValues(t *testing.T) {
	key := "LC_ENV_EXTRA_FALSY"
	for _, value := range []string{"", "0", "false", "False"} {
		t.Setenv(key, value)
		if EnvVarIsSet(key) {
			t.Errorf("EnvVarIsSet should be false for value %q", value)
		}
	}
}

func TestEnvVarIsSetTruthyValues(t *testing.T) {
	key := "LC_ENV_EXTRA_TRUTHY"
	for _, value := range []string{"1", "yes", "FALSE", "no", "value", " "} {
		t.Setenv(key, value)
		if !EnvVarIsSet(key) {
			t.Errorf("EnvVarIsSet should be true for value %q", value)
		}
	}
}

func TestGetFromDictOrEnvPlainStringWins(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_PLAIN", "from-env")
	got, err := GetFromDictOrEnv(map[string]any{"k": "from-map"}, "k", "LC_ENV_EXTRA_PLAIN")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "from-map" {
		t.Fatalf("GetFromDictOrEnv = %q, want the map value", got)
	}
}

func TestGetFromDictOrEnvSecretWrapper(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_SECRET", "from-env")
	got, err := GetFromDictOrEnv(map[string]any{"k": smallBSecretWrapper{value: "s3cr3t"}}, "k", "LC_ENV_EXTRA_SECRET")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "s3cr3t" {
		t.Fatalf("GetFromDictOrEnv = %q, want the real secret, not the masked String", got)
	}
}

func TestGetFromDictOrEnvSecretWrapperPointer(t *testing.T) {
	got, err := GetFromDictOrEnv(map[string]any{"k": &smallBSecretWrapper{value: "ptr-secret"}}, "k")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "ptr-secret" {
		t.Fatalf("GetFromDictOrEnv = %q, want %q", got, "ptr-secret")
	}
}

func TestGetFromDictOrEnvStringerWrapper(t *testing.T) {
	got, err := GetFromDictOrEnv(map[string]any{"k": smallBStringer{value: "stringer-value"}}, "k")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "stringer-value" {
		t.Fatalf("GetFromDictOrEnv = %q", got)
	}
}

func TestGetFromDictOrEnvByteArray(t *testing.T) {
	got, err := GetFromDictOrEnv(map[string]any{"k": []byte("byte-value")}, "k")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "byte-value" {
		t.Fatalf("GetFromDictOrEnv = %q", got)
	}
}

func TestGetFromDictOrEnvTypedNilNoPanic(t *testing.T) {
	var wrapper *smallBSecretWrapper
	_, err := GetFromDictOrEnv(map[string]any{"k": wrapper}, "k")
	if err == nil {
		t.Fatal("GetFromDictOrEnv should return an error for a typed nil wrapper")
	}
}

func TestGetFromDictOrEnvNilMapFallsToEnv(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_NILMAP", "env-value")
	got, err := GetFromDictOrEnv(nil, "k", "LC_ENV_EXTRA_NILMAP")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "env-value" {
		t.Fatalf("GetFromDictOrEnv = %q, want the env value", got)
	}
}

func TestGetFromDictOrEnvEnvFallback(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_FALLBACK", "env-value")
	got, err := GetFromDictOrEnv(map[string]any{}, "k", "LC_ENV_EXTRA_MISSING", "LC_ENV_EXTRA_FALLBACK")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "env-value" {
		t.Fatalf("GetFromDictOrEnv = %q, want the env value", got)
	}
}

func TestGetFromDictOrEnvMultipleEnvKeysFirstEmpty(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_EMPTY", "")
	t.Setenv("LC_ENV_EXTRA_SECOND", "second-value")
	got, err := GetFromDictOrEnv(nil, "k", "LC_ENV_EXTRA_EMPTY", "LC_ENV_EXTRA_SECOND")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "second-value" {
		t.Fatalf("GetFromDictOrEnv = %q, want the second env value", got)
	}
}

func TestGetFromDictOrEnvFalsyEnvValueReturned(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_ZERO", "0")
	got, err := GetFromDictOrEnv(nil, "k", "LC_ENV_EXTRA_ZERO")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "0" {
		t.Fatalf("GetFromDictOrEnv = %q, want %q; unlike EnvVarIsSet, this is a valid value", got, "0")
	}
}

func TestGetFromDictOrEnvEmptyMapValueFallsThrough(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_EMPTYMAP", "env-value")
	got, err := GetFromDictOrEnv(map[string]any{"k": ""}, "k", "LC_ENV_EXTRA_EMPTYMAP")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "env-value" {
		t.Fatalf("GetFromDictOrEnv = %q, want the env value", got)
	}
}

func TestGetFromDictOrEnvEmptySecretFallsThrough(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_EMPTYSECRET", "env-value")
	got, err := GetFromDictOrEnv(map[string]any{"k": smallBSecretWrapper{value: ""}}, "k", "LC_ENV_EXTRA_EMPTYSECRET")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "env-value" {
		t.Fatalf("GetFromDictOrEnv = %q, want the env value", got)
	}
}

func TestGetFromDictOrEnvUnsupportedTypeFallsThrough(t *testing.T) {
	t.Setenv("LC_ENV_EXTRA_UNSUPPORTED", "env-value")
	got, err := GetFromDictOrEnv(map[string]any{"k": 42}, "k", "LC_ENV_EXTRA_UNSUPPORTED")
	if err != nil {
		t.Fatalf("GetFromDictOrEnv returned error: %v", err)
	}
	if got != "env-value" {
		t.Fatalf("GetFromDictOrEnv = %q, want the env value", got)
	}
}

func TestGetFromDictOrEnvMissingKeyErrorMessage(t *testing.T) {
	_, err := GetFromDictOrEnv(map[string]any{}, "api_key", "LC_MISSING_1", "LC_MISSING_2")
	if err == nil {
		t.Fatal("GetFromDictOrEnv should return an error when nothing is found")
	}
	message := err.Error()
	for _, key := range []string{"api_key", "LC_MISSING_1", "LC_MISSING_2"} {
		if !strings.Contains(message, key) {
			t.Errorf("error %q does not list checked key %q", message, key)
		}
	}
}

func TestGetFromDictOrEnvErrorIsDeterministic(t *testing.T) {
	_, first := GetFromDictOrEnv(nil, "k", "LC_M_A", "LC_M_B")
	_, second := GetFromDictOrEnv(nil, "k", "LC_M_A", "LC_M_B")
	if first == nil || second == nil {
		t.Fatal("GetFromDictOrEnv should return errors")
	}
	if first.Error() != second.Error() {
		t.Fatalf("errors differ: %q vs %q", first.Error(), second.Error())
	}
}

func TestGetFromDictOrEnvNoEnvKeys(t *testing.T) {
	_, err := GetFromDictOrEnv(map[string]any{}, "solo")
	if err == nil || !strings.Contains(err.Error(), "solo") {
		t.Fatalf("GetFromDictOrEnv error = %v, should mention the map key", err)
	}
}
