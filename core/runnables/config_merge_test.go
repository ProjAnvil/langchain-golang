package runnables

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
)

func TestMergeConfigEmpty(t *testing.T) {
	out := MergeConfig()
	if out.Name != "" || out.RunID != "" || len(out.Tags) != 0 {
		t.Fatalf("unexpected non-zero zero value: %+v", out)
	}
}

func TestMergeConfigTagsDedup(t *testing.T) {
	a := Config{Tags: []string{"b", "a"}}
	b := Config{Tags: []string{"a", "c"}}
	out := MergeConfig(a, b)
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(out.Tags, want) {
		t.Fatalf("tags = %v, want %v", out.Tags, want)
	}
}

func TestMergeConfigMetadataLastWriterWins(t *testing.T) {
	a := Config{Metadata: map[string]any{"x": 1, "y": "a"}}
	b := Config{Metadata: map[string]any{"y": "b", "z": true}}
	out := MergeConfig(a, b)
	if out.Metadata["x"] != 1 {
		t.Errorf("x: got %v, want 1", out.Metadata["x"])
	}
	if out.Metadata["y"] != "b" {
		t.Errorf("y: got %v, want 'b'", out.Metadata["y"])
	}
	if out.Metadata["z"] != true {
		t.Errorf("z: got %v, want true", out.Metadata["z"])
	}
}

func TestMergeConfigMetadataLCVersionsAccumulate(t *testing.T) {
	a := Config{Metadata: map[string]any{
		"lc_versions": map[string]any{"core": "0.1"},
	}}
	b := Config{Metadata: map[string]any{
		"lc_versions": map[string]any{"langchain": "1.0"},
	}}
	out := MergeConfig(a, b)
	versions, ok := out.Metadata["lc_versions"].(map[string]any)
	if !ok {
		t.Fatalf("lc_versions not a map: %T", out.Metadata["lc_versions"])
	}
	if versions["core"] != "0.1" {
		t.Errorf("core: got %v", versions["core"])
	}
	if versions["langchain"] != "1.0" {
		t.Errorf("langchain: got %v", versions["langchain"])
	}
}

func TestMergeConfigLCVersionsPreservedWithoutIncoming(t *testing.T) {
	a := Config{Metadata: map[string]any{
		"lc_versions": map[string]any{"core": "0.1"},
	}}
	b := Config{Metadata: map[string]any{"other": "val"}}
	out := MergeConfig(a, b)
	versions, ok := out.Metadata["lc_versions"].(map[string]any)
	if !ok {
		t.Fatalf("lc_versions not preserved: %T", out.Metadata["lc_versions"])
	}
	if versions["core"] != "0.1" {
		t.Errorf("core: got %v", versions["core"])
	}
}

func TestMergeConfigConfigurableLastWriterWins(t *testing.T) {
	a := Config{Configurable: map[string]any{"model": "a", "k": 1}}
	b := Config{Configurable: map[string]any{"model": "b"}}
	out := MergeConfig(a, b)
	if out.Configurable["model"] != "b" {
		t.Errorf("model: got %v, want 'b'", out.Configurable["model"])
	}
	if out.Configurable["k"] != 1 {
		t.Errorf("k: got %v, want 1", out.Configurable["k"])
	}
}

func TestMergeConfigScalarsLastNonEmptyWins(t *testing.T) {
	a := Config{Name: "first", RunID: "run1"}
	b := Config{Name: "second"}
	out := MergeConfig(a, b)
	if out.Name != "second" {
		t.Errorf("name: got %q, want 'second'", out.Name)
	}
	if out.RunID != "run1" {
		t.Errorf("run_id: got %q, want 'run1'", out.RunID)
	}
}

func TestMergeConfigCallbacksLastNonEmptyWins(t *testing.T) {
	mgr := callbacks.NewManager(callbacks.NewStdOutHandler(nil))
	a := Config{}               // empty callbacks
	b := Config{Callbacks: mgr} // has a handler
	out := MergeConfig(a, b)
	if out.Callbacks.Empty() {
		t.Fatal("expected non-empty callbacks from second config")
	}
}

func TestMergeConfigMutationSafety(t *testing.T) {
	a := Config{
		Tags:         []string{"a"},
		Metadata:     map[string]any{"k": 1},
		Configurable: map[string]any{"m": "v"},
	}
	out := MergeConfig(a)
	// Mutate original — should not affect output.
	a.Tags[0] = "MUTATED"
	a.Metadata["k"] = 999
	a.Configurable["m"] = "MUTATED"
	if out.Tags[0] != "a" {
		t.Errorf("tags affected by mutation: %v", out.Tags)
	}
	// Metadata/Configurable in MergeConfig are rebuilt, so check they're independent.
	if out.Metadata["k"] == 999 {
		t.Error("metadata affected by mutation")
	}
	if out.Configurable["m"] == "MUTATED" {
		t.Error("configurable affected by mutation")
	}
}
