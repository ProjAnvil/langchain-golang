package utils

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestLargeUtilXorArgs(t *testing.T) {
	if err := XorArgs(
		map[string]bool{"a": true, "b": false},
		map[string]bool{"x": false, "y": true},
	); err != nil {
		t.Fatalf("valid groups rejected: %v", err)
	}
	if err := XorArgs(); err != nil {
		t.Fatalf("no groups should be valid: %v", err)
	}
	if err := XorArgs(map[string]bool{"a": false, "b": false}); err == nil {
		t.Fatal("expected error for a group with zero set")
	}
	if err := XorArgs(map[string]bool{"a": true, "b": true}); err == nil {
		t.Fatal("expected error for a group with two set")
	}
	err := XorArgs(map[string]bool{"a": false, "b": false})
	if !strings.Contains(err.Error(), "must be defined") {
		t.Fatalf("unexpected error text: %v", err)
	}
}

func TestLargeUtilBuildExtraKwargs(t *testing.T) {
	extra := map[string]any{}
	values := map[string]any{"known": 1, "unknown": 2}
	known := map[string]bool{"known": true}
	got, err := BuildExtraKwargs(extra, values, known)
	if err != nil {
		t.Fatal(err)
	}
	if got["unknown"] != 2 {
		t.Fatalf("extra = %#v", got)
	}
	if _, ok := values["unknown"]; ok {
		t.Fatal("unknown field was not removed from values")
	}
	if values["known"] != 1 {
		t.Fatal("known field was removed from values")
	}

	if _, err := BuildExtraKwargs(
		map[string]any{"a": 1},
		map[string]any{"a": 2},
		map[string]bool{},
	); err == nil {
		t.Fatal("expected duplicate-field error")
	}

	if _, err := BuildExtraKwargs(
		map[string]any{"known": 1},
		map[string]any{"other": 2},
		map[string]bool{"known": true, "other": true},
	); err == nil || !strings.Contains(err.Error(), "should be specified explicitly") {
		t.Fatalf("expected invalid model kwarg error, got %v", err)
	}
}

func TestLargeUtilMockNow(t *testing.T) {
	fixed := time.Date(2011, 2, 3, 10, 11, 0, 0, time.UTC)
	clock, restore := MockNow(fixed)
	defer restore()

	if !Now().Equal(fixed) {
		t.Fatalf("Now() = %v, want %v", Now(), fixed)
	}
	if !clock().Equal(fixed) {
		t.Fatal("injected clock did not return the fixed time")
	}
	restore() // idempotent
	if Now().Equal(fixed) {
		t.Fatal("clock was not restored to real time")
	}
}

func TestLargeUtilGetPydanticFieldNames(t *testing.T) {
	type person struct {
		Name   string `json:"name" alias:"full_name"`
		Age    int    `json:"age,omitempty"`
		Skip   bool   `json:"-"`
		hidden string
	}
	got, err := GetPydanticFieldNames(person{hidden: "secret"})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"Name", "name", "full_name", "Age", "age"} {
		if !got[name] {
			t.Fatalf("missing field name %q in %#v", name, got)
		}
	}
	if got["hidden"] {
		t.Fatal("unexported field should not be included")
	}

	// A pointer is accepted as well.
	got2, err := GetPydanticFieldNames(&person{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, got2) {
		t.Fatalf("pointer and value results differ: %#v vs %#v", got, got2)
	}

	if _, err := GetPydanticFieldNames(42); err == nil {
		t.Fatal("expected error for non-struct")
	}
}

func TestLargeUtilCreateSubsetModel(t *testing.T) {
	type address struct {
		City string `json:"city" description:"The city"`
		Zip  int    `json:"zip,omitempty"`
		Note string `json:"-"`
	}
	got, err := CreateSubsetModel(
		"AddressSubset",
		address{},
		[]string{"city"},
		map[string]string{"city": "City name"},
		"subset description",
	)
	if err != nil {
		t.Fatal(err)
	}
	if got["title"] != "AddressSubset" || got["description"] != "subset description" {
		t.Fatalf("metadata = %#v", got)
	}
	props := SchemaProperties(got)
	if len(props) != 1 {
		t.Fatalf("expected a single property, got %#v", props)
	}
	if props["city"]["type"] != "string" || props["city"]["description"] != "City name" {
		t.Fatalf("city property = %#v", props["city"])
	}
	if !reflect.DeepEqual(SchemaRequired(got), []string{"city"}) {
		t.Fatalf("required = %#v", SchemaRequired(got))
	}

	// A missing field is an error.
	if _, err := CreateSubsetModel("", address{}, []string{"missing"}, nil, ""); err == nil {
		t.Fatal("expected missing field error")
	}
	// A non-struct model is an error.
	if _, err := CreateSubsetModel("", "not a struct", nil, nil, ""); err == nil {
		t.Fatal("expected non-struct error")
	}

	// Verify the derived full schema maps Go kinds to JSON-schema types.
	full, err := CreateSubsetModel("", struct {
		Flag   bool     `json:"flag"`
		Score  float64  `json:"score"`
		Tags   []string `json:"tags"`
		Nested struct{} `json:"nested"`
	}{}, []string{"flag", "score", "tags", "nested"}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	allProps := SchemaProperties(full)
	if allProps["flag"]["type"] != "boolean" || allProps["score"]["type"] != "number" {
		t.Fatalf("scalar type mapping = %#v", allProps)
	}
	if allProps["tags"]["type"] != "array" || allProps["nested"]["type"] != "object" {
		t.Fatalf("composite type mapping = %#v", allProps)
	}
}

func TestLargeUtilConvertToSecretStr(t *testing.T) {
	secret, err := ConvertToSecretStr("hunter2")
	if err != nil {
		t.Fatal(err)
	}
	if secret.SecretValue() != "hunter2" {
		t.Fatalf("secret value = %q", secret.SecretValue())
	}
	if secret.String() != "**********" || fmt.Sprint(secret) != "**********" {
		t.Fatalf("secret was not masked: %q / %q", secret.String(), fmt.Sprint(secret))
	}
	data, err := json.Marshal(secret)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `"**********"` {
		t.Fatalf("secret JSON = %s", data)
	}

	// Idempotent conversion of an existing SecretStr.
	same, err := ConvertToSecretStr(secret)
	if err != nil || same.SecretValue() != "hunter2" {
		t.Fatalf("idempotent conversion = (%#v, %v)", same, err)
	}
	ptr := &secret
	fromPtr, err := ConvertToSecretStr(ptr)
	if err != nil || fromPtr.SecretValue() != "hunter2" {
		t.Fatalf("pointer conversion = (%#v, %v)", fromPtr, err)
	}

	// Empty secrets are masked as an empty string.
	empty, err := ConvertToSecretStr("")
	if err != nil || empty.String() != "" {
		t.Fatalf("empty secret = (%#v, %v)", empty, err)
	}
	if _, err := ConvertToSecretStr(42); err == nil {
		t.Fatal("expected error for unsupported type")
	}
	if _, err := ConvertToSecretStr((*SecretStr)(nil)); err == nil {
		t.Fatal("expected error for nil *SecretStr")
	}
}

func TestLargeUtilFromEnv(t *testing.T) {
	t.Setenv("LC_LARGEUTIL_KEY", "value")
	t.Setenv("LC_LARGEUTIL_B", "b")
	t.Setenv("LC_LARGEUTIL_EMPTY", "")

	if v, err := FromEnv([]string{"LC_LARGEUTIL_KEY"})(); err != nil || v != "value" {
		t.Fatalf("single key = (%q, %v)", v, err)
	}
	if v, err := FromEnv([]string{"LC_LARGEUTIL_MISSING_A", "LC_LARGEUTIL_B"})(); err != nil || v != "b" {
		t.Fatalf("first present key = (%q, %v)", v, err)
	}
	// An empty but explicitly set value is "present".
	if v, err := FromEnv([]string{"LC_LARGEUTIL_EMPTY"})(); err != nil || v != "" {
		t.Fatalf("empty-but-present key = (%q, %v)", v, err)
	}
	// Default used when missing.
	if v, err := FromEnv([]string{"LC_LARGEUTIL_MISSING_A"}, EnvOptions{Default: "fallback", HasDefault: true})(); err != nil || v != "fallback" {
		t.Fatalf("default = (%q, %v)", v, err)
	}
	// Explicit empty default is distinct from "no default".
	if v, err := FromEnv([]string{"LC_LARGEUTIL_MISSING_A"}, EnvOptions{Default: "", HasDefault: true})(); err != nil || v != "" {
		t.Fatalf("empty default = (%q, %v)", v, err)
	}
	// Missing with no default is an error.
	if _, err := FromEnv([]string{"LC_LARGEUTIL_MISSING_A"})(); err == nil {
		t.Fatal("expected missing env error")
	}
	// Custom error message is used verbatim.
	if _, err := FromEnv([]string{"LC_LARGEUTIL_MISSING_A"}, EnvOptions{ErrorMessage: "custom boom"})(); err == nil || err.Error() != "custom boom" {
		t.Fatalf("custom error = %v", err)
	}
}

func TestLargeUtilSecretFromEnv(t *testing.T) {
	t.Setenv("LC_LARGEUTIL_SECRET", "s3cr3t")

	sec, err := SecretFromEnv([]string{"LC_LARGEUTIL_SECRET"})()
	if err != nil {
		t.Fatal(err)
	}
	if sec.SecretValue() != "s3cr3t" || sec.String() != "**********" {
		t.Fatalf("secret = (%q, %q)", sec.SecretValue(), sec.String())
	}
	secDefault, err := SecretFromEnv([]string{"LC_LARGEUTIL_MISSING_B"}, EnvOptions{Default: "d", HasDefault: true})()
	if err != nil || secDefault.SecretValue() != "d" {
		t.Fatalf("default secret = (%q, %v)", secDefault.SecretValue(), err)
	}
	if _, err := SecretFromEnv([]string{"LC_LARGEUTIL_MISSING_B"})(); err == nil {
		t.Fatal("expected missing env error")
	}
}

func TestLargeUtilRaiseForStatusWithText(t *testing.T) {
	if err := RaiseForStatusWithText(&http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader("ok")),
	}); err != nil {
		t.Fatalf("2xx should be nil: %v", err)
	}
	if err := RaiseForStatusWithText(&http.Response{
		StatusCode: 500,
		Body:       io.NopCloser(strings.NewReader("boom: invalid key")),
	}); err == nil || err.Error() != "boom: invalid key" {
		t.Fatalf("500 error text = %v", err)
	}
	if err := RaiseForStatusWithText(&http.Response{
		StatusCode: 404,
		Body:       io.NopCloser(strings.NewReader("")),
	}); err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("empty-body error = %v", err)
	}
	if err := RaiseForStatusWithText(nil); err == nil {
		t.Fatal("expected nil response error")
	}
}

func TestLargeUtilGuardImport(t *testing.T) {
	registry := map[string]string{"github.com/example/widget": "v1.2.3"}
	checker := func(name string) (string, bool) {
		v, ok := registry[name]
		return v, ok
	}

	version, err := GuardImport("github.com/example/widget", checker)
	if err != nil || version != "v1.2.3" {
		t.Fatalf("guard import = (%q, %v)", version, err)
	}
	if _, err := GuardImport("github.com/example/missing", checker); err == nil {
		t.Fatal("expected missing dependency error")
	}
	if _, err := GuardImport("", checker); err == nil {
		t.Fatal("expected empty name error")
	}
}

func TestLargeUtilCheckPackageVersion(t *testing.T) {
	checker := func(name string) (string, bool) { return "1.2.3", true }

	if err := CheckPackageVersion("widget", VersionConstraints{GTE: "1.0.0", LT: "2.0.0"}, checker); err != nil {
		t.Fatalf("valid constraints rejected: %v", err)
	}
	if err := CheckPackageVersion("widget", VersionConstraints{LTE: "1.2.3"}, checker); err != nil {
		t.Fatalf("LTE equality rejected: %v", err)
	}
	if err := CheckPackageVersion("widget", VersionConstraints{LT: "1.0.0"}, checker); err == nil {
		t.Fatal("expected LT violation")
	}
	if err := CheckPackageVersion("widget", VersionConstraints{GTE: "2.0.0"}, checker); err == nil {
		t.Fatal("expected GTE violation")
	}
	if err := CheckPackageVersion("widget", VersionConstraints{GT: "1.2.3"}, checker); err == nil {
		t.Fatal("expected GT violation")
	}
	if err := CheckPackageVersion("widget", VersionConstraints{LTE: "1.2.2"}, checker); err == nil {
		t.Fatal("expected LTE violation")
	}

	// v-prefix and pre-release metadata are ignored by the comparator.
	prefixed := func(name string) (string, bool) { return "v2.0.0-beta.1", true }
	if err := CheckPackageVersion("widget", VersionConstraints{GTE: "2.0.0"}, prefixed); err != nil {
		t.Fatalf("prefixed version rejected: %v", err)
	}

	// Missing package and unparseable versions are errors.
	missing := func(name string) (string, bool) { return "", false }
	if err := CheckPackageVersion("widget", VersionConstraints{}, missing); err == nil {
		t.Fatal("expected missing package error")
	}
	bad := func(name string) (string, bool) { return "notaversion", true }
	if err := CheckPackageVersion("widget", VersionConstraints{GTE: "1.0.0"}, bad); err == nil {
		t.Fatal("expected invalid version error")
	}
}
