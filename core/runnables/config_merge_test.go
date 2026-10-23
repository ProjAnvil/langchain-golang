package runnables

import (
	"maps"
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

func TestMergeConfigVersionMapsAreIndependent(t *testing.T) {
	for _, tc := range []struct {
		name     string
		base     map[string]any
		incoming map[string]any
		want     map[string]any
	}{
		{name: "nil incoming", incoming: map[string]any{"lc_versions": map[string]any(nil)}},
		{
			name: "nil base", base: map[string]any{"lc_versions": map[string]any(nil)},
			incoming: map[string]any{"other": true},
		},
		{
			name: "both nil", base: map[string]any{"lc_versions": map[string]any(nil)},
			incoming: map[string]any{"lc_versions": map[string]any(nil)},
		},
		{
			name: "base only", base: map[string]any{"lc_versions": map[string]any{"core": "1"}},
			incoming: map[string]any{"other": true},
			want:     map[string]any{"core": "1"},
		},
		{
			name: "incoming only", incoming: map[string]any{"lc_versions": map[string]any{"partner": "2"}},
			want: map[string]any{"partner": "2"},
		},
		{
			name: "overlapping versions", base: map[string]any{"lc_versions": map[string]any{"core": "1"}},
			incoming: map[string]any{"lc_versions": map[string]any{"core": "2", "partner": "3"}},
			want:     map[string]any{"core": "2", "partner": "3"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := MergeConfig(Config{Metadata: tc.base}, Config{Metadata: tc.incoming})
			versions, ok := out.Metadata["lc_versions"].(map[string]any)
			if !ok || versions == nil || !maps.Equal(versions, tc.want) {
				t.Fatalf("lc_versions = %#v, want non-nil map with entries %#v", out.Metadata["lc_versions"], tc.want)
			}
			versions["changed"] = true
			for _, metadata := range []map[string]any{tc.base, tc.incoming} {
				source, _ := metadata["lc_versions"].(map[string]any)
				if _, exists := source["changed"]; exists {
					t.Fatal("merged version map aliases an input map")
				}
			}
		})
	}
}
