package postgres

import (
	"maps"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// newTestSaver builds a Saver with only the serde wired up — enough for the
// encode/decode unit tests, which never touch the pool.
func newTestSaver() *Saver {
	return &Saver{serde: serde.NewJSONSerializer()}
}

func TestIsInline(t *testing.T) {
	cases := []struct {
		name  string
		value any
		want  bool
	}{
		{"nil", nil, true},
		{"string", "s", true},
		{"bool", true, true},
		{"float64", float64(1.5), true},
		{"int", 7, false},
		{"int64", int64(7), false},
		{"map", map[string]any{}, false},
		{"slice-any", []any{1}, false},
		{"slice-string", []string{"a"}, false},
		{"types.Send", types.Send{}, false},
		{"time.Time", time.Time{}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isInline(tc.value); got != tc.want {
				t.Fatalf("isInline(%#v) = %v, want %v", tc.value, got, tc.want)
			}
		})
	}
}

func TestSplitChannelValues(t *testing.T) {
	values := map[string]any{
		"name":   "alice",
		"active": true,
		"score":  1.5,
		"nilv":   nil,
		"count":  7,
		"big":    int64(9),
		"tags":   []string{"a"},
		"meta":   map[string]any{"k": "v"},
		"send":   types.Send{Node: "n"},
	}
	inline, blobs := splitChannelValues(values)

	wantInline := map[string]bool{"name": true, "active": true, "score": true, "nilv": true}
	if len(inline) != len(wantInline) {
		t.Fatalf("inline keys = %v, want exactly %v", maps.Keys(inline), maps.Keys(wantInline))
	}
	for k := range wantInline {
		if _, ok := inline[k]; !ok {
			t.Errorf("inline missing primitive key %q", k)
		}
	}
	if len(blobs) != len(values)-len(wantInline) {
		t.Fatalf("blobs keys = %v, want the non-primitive keys", maps.Keys(blobs))
	}
	for k := range values {
		_, inInline := inline[k]
		_, inBlobs := blobs[k]
		if inInline == inBlobs {
			t.Errorf("key %q: inline=%v blobs=%v, want exactly one", k, inInline, inBlobs)
		}
	}
}

func TestCheckpointProjectionRoundTrip(t *testing.T) {
	s := newTestSaver()
	ts := time.Date(2026, 8, 8, 1, 2, 3, 0, time.UTC)
	cp := checkpoint.Checkpoint{
		V:  1,
		ID: "cp-1",
		TS: ts,
		ChannelValues: map[string]any{
			"name":  "alice",
			"score": 1.5,
			// Non-primitive values go to checkpoint_blobs; the projection
			// keeps only the inline entries.
			"tags": []string{"a"},
		},
		ChannelVersions: map[string]int64{"name": 1, "score": 2, "tags": 3},
		VersionsSeen:    map[string]map[string]int64{"node-a": {"name": 1}},
		Next: []checkpoint.PlannedTask{
			{ID: "task-1", Node: "node-b", Arg: map[string]any{"send": types.Send{Node: "node-c", Arg: map[string]any{"input": "payload"}}}},
		},
	}
	inline, _ := splitChannelValues(cp.ChannelValues)

	encoded, err := s.encodeCheckpoint(cp, inline)
	if err != nil {
		t.Fatalf("encodeCheckpoint: %v", err)
	}
	got, err := s.decodeCheckpoint(encoded)
	if err != nil {
		t.Fatalf("decodeCheckpoint: %v", err)
	}

	want := cp
	want.ChannelValues = inline // blob keys are merged back by assemble, not decode
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip mismatch:\n got: %#v\nwant: %#v", got, want)
	}
}

func TestMigrations(t *testing.T) {
	if len(Migrations) != 10 {
		t.Fatalf("len(Migrations) = %d, want 10 (v0-v9)", len(Migrations))
	}
	for _, v := range []int{6, 7, 8} {
		if !strings.Contains(Migrations[v], "CREATE INDEX CONCURRENTLY") {
			t.Errorf("Migrations[%d] missing CREATE INDEX CONCURRENTLY:\n%s", v, Migrations[v])
		}
	}
	if !strings.Contains(Migrations[2], "version BIGINT NOT NULL") {
		t.Errorf("Migrations[2] missing Go deviation column `version BIGINT NOT NULL`:\n%s", Migrations[2])
	}
	if !strings.Contains(Migrations[9], "task_path") {
		t.Errorf("Migrations[9] missing task_path column:\n%s", Migrations[9])
	}
}

func TestListQuery(t *testing.T) {
	cfg := checkpoint.Config{ThreadID: "t1", CheckpointNS: "ns"}
	const base = `SELECT checkpoint_id, parent_checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = $1 AND checkpoint_ns = $2`

	t.Run("empty options", func(t *testing.T) {
		query, args, err := listQuery(cfg, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("listQuery: %v", err)
		}
		if query != base+` ORDER BY checkpoint_id DESC` {
			t.Errorf("query = %q", query)
		}
		if len(args) != 2 || args[0] != "t1" || args[1] != "ns" {
			t.Errorf("args = %v, want [t1 ns]", args)
		}
	})

	t.Run("before", func(t *testing.T) {
		query, args, err := listQuery(cfg, checkpoint.ListOptions{
			Before: &checkpoint.Config{CheckpointID: "cp9"},
		})
		if err != nil {
			t.Fatalf("listQuery: %v", err)
		}
		if !strings.Contains(query, `AND checkpoint_id < $3`) {
			t.Errorf("query missing before predicate: %q", query)
		}
		if len(args) != 3 || args[2] != "cp9" {
			t.Errorf("args = %v, want [t1 ns cp9]", args)
		}
	})

	t.Run("filter", func(t *testing.T) {
		query, args, err := listQuery(cfg, checkpoint.ListOptions{
			Filter: map[string]any{"source": "loop"},
		})
		if err != nil {
			t.Fatalf("listQuery: %v", err)
		}
		if !strings.Contains(query, `AND metadata @> $3::jsonb`) {
			t.Errorf("query missing filter predicate: %q", query)
		}
		if len(args) != 3 || args[2] != `{"source":"loop"}` {
			t.Errorf("args = %v, want [t1 ns {\"source\":\"loop\"}]", args)
		}
	})

	t.Run("limit", func(t *testing.T) {
		query, args, err := listQuery(cfg, checkpoint.ListOptions{Limit: 5})
		if err != nil {
			t.Fatalf("listQuery: %v", err)
		}
		if !strings.Contains(query, `LIMIT $3`) {
			t.Errorf("query missing limit: %q", query)
		}
		if len(args) != 3 || args[2] != 5 {
			t.Errorf("args = %v, want [t1 ns 5]", args)
		}
	})

	t.Run("combined", func(t *testing.T) {
		query, args, err := listQuery(cfg, checkpoint.ListOptions{
			Before: &checkpoint.Config{CheckpointID: "cp9"},
			Filter: map[string]any{"step": 2},
			Limit:  3,
		})
		if err != nil {
			t.Fatalf("listQuery: %v", err)
		}
		for _, frag := range []string{`AND checkpoint_id < $3`, `AND metadata @> $4::jsonb`, `LIMIT $5`} {
			if !strings.Contains(query, frag) {
				t.Errorf("query missing %q: %q", frag, query)
			}
		}
		if len(args) != 5 {
			t.Errorf("args = %v, want 5 args", args)
		}
	})

	t.Run("unmarshalable filter errors", func(t *testing.T) {
		_, _, err := listQuery(cfg, checkpoint.ListOptions{
			Filter: map[string]any{"bad": func() {}},
		})
		if err == nil {
			t.Fatal("listQuery with a func filter value returned nil error")
		}
	})
}
