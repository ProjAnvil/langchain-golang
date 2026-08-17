package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestModelDataToProfile(t *testing.T) {
	modelData := map[string]any{
		"name":            "Claude Haiku 4.5",
		"status":          "stable",
		"reasoning":       true,
		"tool_call":       true,
		"pdf_inputs":      false,
		"limit":           map[string]any{"context": float64(200000), "output": float64(8192)},
		"modalities":      map[string]any{"input": []any{"text", "image"}, "output": []any{"text"}},
		"unrelated_field": "ignored",
	}

	got := ModelDataToProfile(modelData)

	want := Profile{
		"name":              "Claude Haiku 4.5",
		"status":            "stable",
		"reasoning_output":  true,
		"tool_calling":      true,
		"max_input_tokens":  float64(200000),
		"max_output_tokens": float64(8192),
		"text_inputs":       true,
		"image_inputs":      true,
		"audio_inputs":      false,
		"video_inputs":      false,
		"text_outputs":      true,
		"image_outputs":     false,
		"audio_outputs":     false,
		"video_outputs":     false,
		"pdf_inputs":        false,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ModelDataToProfile() = %#v, want %#v", got, want)
	}
}

func TestModelDataToProfilePDFInputsFromModality(t *testing.T) {
	modelData := map[string]any{
		"modalities": map[string]any{"input": []any{"pdf"}},
	}
	got := ModelDataToProfile(modelData)
	if got["pdf_inputs"] != true {
		t.Errorf("expected pdf_inputs = true when 'pdf' modality present, got %v", got["pdf_inputs"])
	}
}

func TestModelDataToProfileOmitsNilFields(t *testing.T) {
	got := ModelDataToProfile(map[string]any{})
	// Boolean modality fields are always present (default false); everything
	// else that resolves to nil must be omitted entirely.
	for _, key := range []string{"name", "status", "max_input_tokens", "pdf_inputs", "reasoning_output"} {
		if _, ok := got[key]; ok {
			t.Errorf("expected key %q to be omitted, got %v", key, got[key])
		}
	}
	for _, key := range []string{"text_inputs", "image_inputs", "audio_inputs", "video_inputs"} {
		if v, ok := got[key]; !ok || v != false {
			t.Errorf("expected key %q = false, got %v (present=%v)", key, v, ok)
		}
	}
}

func TestApplyOverrides(t *testing.T) {
	base := Profile{"tool_calling": true, "structured_output": false}
	providerAug := Profile{"structured_output": true, "tool_call_streaming": true}
	modelAug := Profile{"structured_output": false} // model-level wins over provider-level

	got := ApplyOverrides(base, providerAug, modelAug)
	want := Profile{"tool_calling": true, "structured_output": false, "tool_call_streaming": true}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ApplyOverrides() = %v, want %v", got, want)
	}

	// base must not be mutated.
	if base["structured_output"] != false || len(base) != 2 {
		t.Errorf("ApplyOverrides mutated base: %v", base)
	}
}

func TestApplyOverridesSkipsNil(t *testing.T) {
	base := Profile{"a": 1}
	got := ApplyOverrides(base, nil, Profile{"a": nil, "b": 2})
	want := Profile{"a": 1, "b": 2}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ApplyOverrides() = %v, want %v", got, want)
	}
}

func TestValidateDataDir(t *testing.T) {
	cwd := t.TempDir()
	inside := filepath.Join(cwd, "data")

	resolved, needsConfirmation, err := ValidateDataDir(inside, cwd)
	if err != nil {
		t.Fatalf("ValidateDataDir() error = %v", err)
	}
	if needsConfirmation {
		t.Errorf("expected no confirmation for path inside cwd")
	}
	if resolved != inside {
		t.Errorf("resolved = %q, want %q", resolved, inside)
	}

	other := t.TempDir()
	_, needsConfirmation, err = ValidateDataDir(other, cwd)
	if err != nil {
		t.Fatalf("ValidateDataDir() error = %v", err)
	}
	if !needsConfirmation {
		t.Errorf("expected confirmation required for path outside cwd")
	}
}

func TestValidateDataDirEmpty(t *testing.T) {
	if _, _, err := ValidateDataDir("", "/tmp"); err == nil {
		t.Errorf("expected error for empty data dir")
	}
}

func TestLoadAugmentationsMissingFile(t *testing.T) {
	dir := t.TempDir()
	providerAug, modelAugs, err := LoadAugmentations(dir)
	if err != nil {
		t.Fatalf("LoadAugmentations() error = %v", err)
	}
	if len(providerAug) != 0 || len(modelAugs) != 0 {
		t.Errorf("expected empty results for missing file, got %v %v", providerAug, modelAugs)
	}
}

func TestBuildProfilesJSONDeterministic(t *testing.T) {
	profiles := Registry{
		"b-model": Profile{"name": "B"},
		"a-model": Profile{"name": "A"},
	}
	out, err := BuildProfilesJSON(profiles)
	if err != nil {
		t.Fatalf("BuildProfilesJSON() error = %v", err)
	}
	var roundTrip Registry
	if err := json.Unmarshal(out, &roundTrip); err != nil {
		t.Fatalf("failed to unmarshal output: %v", err)
	}
	if !reflect.DeepEqual(roundTrip, profiles) {
		t.Errorf("round trip mismatch: %v vs %v", roundTrip, profiles)
	}
	aIdx := bytes.Index(out, []byte(`"a-model"`))
	bIdx := bytes.Index(out, []byte(`"b-model"`))
	if aIdx == -1 || bIdx == -1 || aIdx > bIdx {
		t.Errorf("expected sorted keys in output, got %s", out)
	}
}

func TestRefreshEndToEnd(t *testing.T) {
	apiResponse := map[string]any{
		"anthropic": map[string]any{
			"models": map[string]any{
				"claude-haiku-4-5": map[string]any{
					"name":       "Claude Haiku 4.5",
					"tool_call":  true,
					"limit":      map[string]any{"context": 200000, "output": 8192},
					"modalities": map[string]any{"input": []string{"text", "image"}, "output": []string{"text"}},
				},
			},
		},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(apiResponse)
	}))
	defer server.Close()

	dataDir := t.TempDir()
	augPath := filepath.Join(dataDir, augmentationsFileName)
	if err := os.WriteFile(augPath, []byte(`provider = "anthropic"

[overrides]
tool_call_streaming = true

[overrides."claude-haiku-4-5"]
structured_output = true

[overrides."claude-extra-only"]
structured_output = false
`), 0o644); err != nil {
		t.Fatalf("failed to write augmentations file: %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Stdout:   &stdout,
		Stderr:   &stderr,
		Cwd:      dataDir,
	})
	if err != nil {
		t.Fatalf("Refresh() error = %v, stderr = %s", err, stderr.String())
	}

	outputFile := filepath.Join(dataDir, profilesFileName)
	contents, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("failed to read output file: %v", err)
	}

	var got Registry
	if err := json.Unmarshal(contents, &got); err != nil {
		t.Fatalf("failed to unmarshal output: %v", err)
	}

	haiku, ok := got["claude-haiku-4-5"]
	if !ok {
		t.Fatalf("expected claude-haiku-4-5 in output, got %v", got)
	}
	if haiku["tool_call_streaming"] != true {
		t.Errorf("expected provider override tool_call_streaming=true, got %v", haiku["tool_call_streaming"])
	}
	if haiku["structured_output"] != true {
		t.Errorf("expected model override structured_output=true, got %v", haiku["structured_output"])
	}
	if haiku["tool_calling"] != true {
		t.Errorf("expected tool_calling=true from base data, got %v", haiku["tool_calling"])
	}

	extra, ok := got["claude-extra-only"]
	if !ok {
		t.Fatalf("expected augmentation-only model claude-extra-only in output, got %v", got)
	}
	if extra["structured_output"] != false {
		t.Errorf("expected structured_output=false for augmentation-only model, got %v", extra["structured_output"])
	}
}

func TestRefreshUnknownProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{})
	}))
	defer server.Close()

	dataDir := t.TempDir()
	err := Refresh(RefreshOptions{
		Provider: "nonexistent",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Cwd:      dataDir,
	})
	if err == nil {
		t.Fatalf("expected error for unknown provider")
	}
}

func TestFetchModelsDevDataHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	_, err := fetchModelsDevData(server.Client(), server.URL)
	if err == nil {
		t.Fatalf("expected error for non-2xx status")
	}
	if !strings.Contains(err.Error(), "HTTP error 500") {
		t.Errorf("expected HTTP error message, got %v", err)
	}
}

func TestFetchModelsDevDataInvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("this is not json"))
	}))
	defer server.Close()

	_, err := fetchModelsDevData(server.Client(), server.URL)
	if err == nil {
		t.Fatalf("expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "invalid JSON") {
		t.Errorf("expected invalid JSON message, got %v", err)
	}
}

func TestFetchModelsDevDataConnectError(t *testing.T) {
	// Start a server and immediately close it so connects are refused.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	server.Close()

	_, err := fetchModelsDevData(server.Client(), server.URL)
	if err == nil {
		t.Fatalf("expected connection error")
	}
	if !strings.Contains(err.Error(), "failed to connect") {
		t.Errorf("expected connection error message, got %v", err)
	}
}

type errReadCloser struct{}

func (errReadCloser) Read([]byte) (int, error) { return 0, fmt.Errorf("boom") }
func (errReadCloser) Close() error             { return nil }

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestFetchModelsDevDataReadError(t *testing.T) {
	client := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       errReadCloser{},
			Header:     make(http.Header),
		}, nil
	})}

	_, err := fetchModelsDevData(client, "http://example.invalid/api.json")
	if err == nil {
		t.Fatalf("expected read error")
	}
	if !strings.Contains(err.Error(), "failed to read response") {
		t.Errorf("expected read error message, got %v", err)
	}
}

func TestRefreshInvalidAugmentations(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"anthropic": map[string]any{"models": map[string]any{}},
		})
	}))
	defer server.Close()

	dataDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dataDir, augmentationsFileName), []byte("[unterminated"), 0o644); err != nil {
		t.Fatalf("failed to write augmentations file: %v", err)
	}

	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Cwd:      dataDir,
	})
	if err == nil {
		t.Fatalf("expected error for invalid augmentations TOML")
	}
	if !strings.Contains(err.Error(), "invalid TOML") {
		t.Errorf("expected TOML error message, got %v", err)
	}
}

func TestRefreshUnknownKeysWarning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"anthropic": map[string]any{
				"models": map[string]any{
					"model-x": map[string]any{"name": "Model X"},
				},
			},
		})
	}))
	defer server.Close()

	dataDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dataDir, augmentationsFileName), []byte(`[overrides]
bogus_capability = true
`), 0o644); err != nil {
		t.Fatalf("failed to write augmentations file: %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Stdout:   &stdout,
		Stderr:   &stderr,
		Cwd:      dataDir,
	})
	if err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
	if !strings.Contains(stderr.String(), "bogus_capability") {
		t.Errorf("expected warning about unknown key, got stderr = %q", stderr.String())
	}

	// The unknown key must still be written to the output profiles.
	contents, err := os.ReadFile(filepath.Join(dataDir, profilesFileName))
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}
	if !bytes.Contains(contents, []byte("bogus_capability")) {
		t.Errorf("expected unknown key in output, got %s", contents)
	}
}

func TestRefreshMkdirAllFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"anthropic": map[string]any{"models": map[string]any{}},
		})
	}))
	defer server.Close()

	// A read-only parent lets LoadAugmentations succeed (missing file) but
	// makes MkdirAll fail with a permission error.
	parent := t.TempDir()
	if err := os.Chmod(parent, 0o555); err != nil {
		t.Fatalf("failed to chmod parent: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(parent, 0o755) })
	dataDir := filepath.Join(parent, "data")

	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Cwd:      parent,
	})
	if err == nil {
		t.Fatalf("expected error when data dir cannot be created")
	}
	if !strings.Contains(err.Error(), "failed to create directory") {
		t.Errorf("expected mkdir error message, got %v", err)
	}
}

func TestRefreshOutsideCwdNilConfirmAborts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{})
	}))
	defer server.Close()

	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  t.TempDir(),
		APIURL:   server.URL,
		Cwd:      t.TempDir(),
		Confirm:  nil, // nil Confirm must be treated as declined
	})
	if err == nil {
		t.Fatalf("expected abort error when Confirm is nil")
	}
	if !strings.Contains(err.Error(), "aborted") {
		t.Errorf("expected abort message, got %v", err)
	}
}

func TestRefreshOutsideCwdConfirmTrueProceeds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"anthropic": map[string]any{"models": map[string]any{}},
		})
	}))
	defer server.Close()

	dataDir := t.TempDir()
	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Cwd:      t.TempDir(),
		Confirm:  func() bool { return true },
	})
	if err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(dataDir, profilesFileName)); err != nil {
		t.Errorf("expected profiles file to be written: %v", err)
	}
}

func TestRefreshDefaultStdoutStderr(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"anthropic": map[string]any{"models": map[string]any{}},
		})
	}))
	defer server.Close()

	dataDir := t.TempDir()
	// Nil Stdout/Stderr must default to io.Discard without panicking.
	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Cwd:      dataDir,
	})
	if err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
}

func TestBuildProfilesJSONError(t *testing.T) {
	profiles := Registry{"bad": Profile{"ch": make(chan int)}}
	if _, err := BuildProfilesJSON(profiles); err == nil {
		t.Fatalf("expected error for unmarshalable profile value")
	}
}

func TestWriteProfilesFileAtomicErrors(t *testing.T) {
	dataDir := t.TempDir()

	// ensureSafeOutputPath failure: output path escapes dataDir.
	escaping := filepath.Join(dataDir, "..", "profiles.json")
	if err := writeProfilesFileAtomic(dataDir, escaping, []byte("{}")); err == nil {
		t.Errorf("expected error for output path escaping data dir")
	}

	// CreateTemp failure: dataDir does not exist.
	missing := filepath.Join(dataDir, "does-not-exist")
	if err := writeProfilesFileAtomic(missing, filepath.Join(missing, profilesFileName), []byte("{}")); err == nil {
		t.Errorf("expected error for missing data dir")
	}

	// Rename failure: output path is an existing directory.
	outputDir := filepath.Join(dataDir, "profiles.json")
	if err := os.Mkdir(outputDir, 0o755); err != nil {
		t.Fatalf("failed to create output dir: %v", err)
	}
	if err := writeProfilesFileAtomic(dataDir, outputDir, []byte("{}")); err == nil {
		t.Errorf("expected error when output path is a directory")
	}
}

func TestEnsureSafeOutputPath(t *testing.T) {
	dataDir := t.TempDir()
	outputFile := filepath.Join(dataDir, profilesFileName)

	// Happy path.
	if err := ensureSafeOutputPath(dataDir, outputFile); err != nil {
		t.Fatalf("ensureSafeOutputPath() error = %v", err)
	}

	// Symlinked data directory is rejected.
	realDir := t.TempDir()
	linkDir := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(realDir, linkDir); err != nil {
		t.Fatalf("failed to create symlink: %v", err)
	}
	if err := ensureSafeOutputPath(linkDir, filepath.Join(linkDir, profilesFileName)); err == nil {
		t.Errorf("expected error for symlinked data dir")
	}

	// Symlinked output file is rejected.
	target := filepath.Join(dataDir, "target.json")
	if err := os.WriteFile(target, []byte("{}"), 0o644); err != nil {
		t.Fatalf("failed to write target: %v", err)
	}
	linkFile := filepath.Join(dataDir, "link.json")
	if err := os.Symlink(target, linkFile); err != nil {
		t.Fatalf("failed to create symlink: %v", err)
	}
	if err := ensureSafeOutputPath(dataDir, linkFile); err == nil {
		t.Errorf("expected error for symlinked output file")
	}

	// Output path escaping the data dir is rejected.
	if err := ensureSafeOutputPath(dataDir, filepath.Join(dataDir, "..", "evil.json")); err == nil {
		t.Errorf("expected error for escaping output path")
	}
}

func TestUnknownKeysAcross(t *testing.T) {
	profiles := Registry{
		"a": Profile{"name": "A", "bogus_one": true},
		"b": Profile{"name": "B", "bogus_two": 1, "bogus_one": false},
	}
	got := unknownKeysAcross(profiles)
	want := []string{"bogus_one", "bogus_two"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("unknownKeysAcross() = %v, want %v", got, want)
	}

	if got := unknownKeysAcross(Registry{"ok": Profile{"name": "A"}}); len(got) != 0 {
		t.Errorf("expected no unknown keys, got %v", got)
	}
}

func TestLoadAugmentationsReadError(t *testing.T) {
	// A directory named like the augmentations file cannot be read as a file.
	dataDir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dataDir, augmentationsFileName), 0o755); err != nil {
		t.Fatalf("failed to create dir: %v", err)
	}
	if _, _, err := LoadAugmentations(dataDir); err == nil {
		t.Fatalf("expected read error for directory-as-file")
	}
}

func TestParseAugmentationsOverridesNotTable(t *testing.T) {
	// `overrides = 1` makes [overrides] resolve to a scalar, not a table.
	_, _, err := ParseAugmentations([]byte("overrides = 1"))
	if err == nil {
		t.Fatalf("expected error for non-table overrides")
	}
	if !strings.Contains(err.Error(), "must be a table") {
		t.Errorf("expected table error message, got %v", err)
	}
}

func TestRefreshEmptyDataDir(t *testing.T) {
	err := Refresh(RefreshOptions{Provider: "anthropic", DataDir: ""})
	if err == nil {
		t.Fatalf("expected error for empty data dir")
	}
	if !strings.Contains(err.Error(), "data directory must not be empty") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRefreshFetchError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	server.Close() // connections now refused

	dataDir := t.TempDir()
	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Cwd:      dataDir,
	})
	if err == nil {
		t.Fatalf("expected error when fetch fails")
	}
	if !strings.Contains(err.Error(), "failed to connect") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRefreshDefaultsCwdAndAPIURL(t *testing.T) {
	// A custom transport intercepts the request, so the default APIURL branch
	// is exercised without any network access.
	client := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != DefaultAPIURL {
			t.Errorf("expected request to default URL %q, got %q", DefaultAPIURL, r.URL.String())
		}
		body := `{"anthropic": {"models": {}}}`
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(body)),
			Header:     make(http.Header),
		}, nil
	})}

	// Chdir into the data dir so an empty Cwd resolves via os.Getwd and the
	// data dir counts as "inside the current directory". EvalSymlinks keeps
	// the path consistent with os.Getwd on macOS (/var -> /private/var).
	dataDir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatalf("failed to resolve temp dir: %v", err)
	}
	oldWd, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get working directory: %v", err)
	}
	if err := os.Chdir(dataDir); err != nil {
		t.Fatalf("failed to chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldWd) })

	if err := Refresh(RefreshOptions{
		Provider:   "anthropic",
		DataDir:    dataDir,
		HTTPClient: client,
	}); err != nil {
		t.Fatalf("Refresh() error = %v", err)
	}
}

func TestRefreshSymlinkedDataDirRejected(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"anthropic": map[string]any{"models": map[string]any{}},
		})
	}))
	defer server.Close()

	realDir := t.TempDir()
	linkDir := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(realDir, linkDir); err != nil {
		t.Fatalf("failed to create symlink: %v", err)
	}

	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  linkDir,
		APIURL:   server.URL,
		Cwd:      linkDir,
	})
	if err == nil {
		t.Fatalf("expected error for symlinked data dir")
	}
	if !strings.Contains(err.Error(), "symlink") {
		t.Errorf("expected symlink error message, got %v", err)
	}
}

func TestRefreshOutsideCwdRequiresConfirmation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{})
	}))
	defer server.Close()

	cwd := t.TempDir()
	dataDir := t.TempDir()

	confirmCalled := false
	err := Refresh(RefreshOptions{
		Provider: "anthropic",
		DataDir:  dataDir,
		APIURL:   server.URL,
		Cwd:      cwd,
		Confirm: func() bool {
			confirmCalled = true
			return false
		},
	})
	if err == nil {
		t.Fatalf("expected error when confirmation declined")
	}
	if !confirmCalled {
		t.Errorf("expected Confirm to be called for out-of-cwd data dir")
	}
}
