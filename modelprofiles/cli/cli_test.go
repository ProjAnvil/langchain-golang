package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
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
