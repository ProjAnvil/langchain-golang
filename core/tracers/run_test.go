// Tests for the LangSmith run-tree model (design t18 PR3, section 6.1): the
// dotted-order computation ported verbatim from langchain_core/tracers/core.py
// (_start_trace, lines 125-130) and the Run JSON shape posted to
// {endpoint}/runs/batch.
package tracers

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestDottedOrderSuffixMatchesPythonFormat(t *testing.T) {
	// Python: run.start_time.strftime("%Y%m%dT%H%M%S%fZ") + str(run.id)
	// %Y%m%dT%H%M%S = 20060102T150405, %f = zero-padded 6-digit microseconds.
	start := time.Date(2026, 9, 12, 10, 20, 30, 123456000, time.UTC)
	id := "11111111-2222-3333-4444-555555555555"
	got := dottedOrderSuffix(start, id)
	want := "20260912T102030123456Z" + id
	if got != want {
		t.Fatalf("dotted order suffix:\ngot  %s\nwant %s", got, want)
	}
	// Zero microseconds must stay zero-padded to six digits.
	zero := dottedOrderSuffix(time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC), id)
	if !strings.HasPrefix(zero, "20260102T030405000000Z") {
		t.Fatalf("zero-padded micros: %s", zero)
	}
	// Non-UTC input is converted (Python datetimes are tz-aware UTC here).
	local := time.Date(2026, 9, 12, 12, 20, 30, 123456000, time.FixedZone("X", 2*3600))
	if got, want := dottedOrderSuffix(local, id), "20260912T102030123456Z"+id; got != want {
		t.Fatalf("non-utc conversion:\ngot  %s\nwant %s", got, want)
	}
}

func TestStartTraceDottedOrderChain(t *testing.T) {
	tracer := NewLangChainTracerWithOptions(LangSmithOptions{
		Endpoint: "http://localhost:0", APIKey: "k", Project: "p",
		BatchSize: 64, FlushEvery: time.Hour,
	})
	defer func() { _ = tracer.Close() }()

	root := &Run{ID: "root-id", Name: "root", RunType: RunTypeChain,
		StartTime: time.Date(2026, 9, 12, 10, 20, 30, 123456000, time.UTC)}
	child := &Run{ID: "child-id", Name: "child", RunType: RunTypeChain, ParentRunID: "root-id",
		StartTime: time.Date(2026, 9, 12, 10, 20, 31, 654321000, time.UTC)}
	grand := &Run{ID: "grand-id", Name: "grand", RunType: RunTypeChain, ParentRunID: "child-id",
		StartTime: time.Date(2026, 9, 12, 10, 20, 32, 111000, time.UTC)}

	tracer.startTrace(root)
	tracer.startTrace(child)
	tracer.startTrace(grand)

	if root.TraceID != "root-id" || root.DottedOrder == "" {
		t.Fatalf("root trace: %q %q", root.TraceID, root.DottedOrder)
	}
	wantRootDO := "20260912T102030123456Zroot-id"
	if root.DottedOrder != wantRootDO {
		t.Fatalf("root dotted order %q, want %q", root.DottedOrder, wantRootDO)
	}
	// child = parent + "." + current; trace id inherited from root.
	wantChildDO := wantRootDO + ".20260912T102031654321Zchild-id"
	if child.DottedOrder != wantChildDO {
		t.Fatalf("child dotted order %q, want %q", child.DottedOrder, wantChildDO)
	}
	wantGrandDO := wantChildDO + ".20260912T102032000111Zgrand-id"
	if grand.DottedOrder != wantGrandDO {
		t.Fatalf("grand dotted order %q, want %q", grand.DottedOrder, wantGrandDO)
	}
	for _, run := range []*Run{root, child, grand} {
		if run.TraceID != "root-id" {
			t.Fatalf("trace id for %s = %q, want root-id", run.ID, run.TraceID)
		}
	}
	// Local tree: root -> child -> grand; session id mirrors the trace id.
	if len(root.ChildRuns) != 1 || root.ChildRuns[0] != child || len(child.ChildRuns) != 1 || child.ChildRuns[0] != grand {
		t.Fatalf("child runs: root=%v child=%v", root.ChildRuns, child.ChildRuns)
	}
	if root.SessionID != "root-id" || child.SessionID != "root-id" || grand.SessionID != "root-id" {
		t.Fatalf("session ids: %q %q %q", root.SessionID, child.SessionID, grand.SessionID)
	}
	if root.SessionName != "p" {
		t.Fatalf("session name %q", root.SessionName)
	}
}

func TestStartTraceMissingParentTreatedAsRoot(t *testing.T) {
	// core.py:135-144 — a parent absent from order_map is demoted to a root run.
	tracer := NewLangChainTracerWithOptions(LangSmithOptions{
		Endpoint: "http://localhost:0", APIKey: "k", BatchSize: 64, FlushEvery: time.Hour,
	})
	defer func() { _ = tracer.Close() }()
	orphan := &Run{ID: "orphan", RunType: RunTypeTool, ParentRunID: "nope",
		StartTime: time.Date(2026, 9, 12, 10, 0, 0, 0, time.UTC)}
	tracer.startTrace(orphan)
	if orphan.ParentRunID != "" {
		t.Fatalf("orphan parent %q, want cleared", orphan.ParentRunID)
	}
	if orphan.TraceID != "orphan" || !strings.HasSuffix(orphan.DottedOrder, "Zorphan") {
		t.Fatalf("orphan trace: %q %q", orphan.TraceID, orphan.DottedOrder)
	}
}

func TestRunJSONShape(t *testing.T) {
	run := &Run{
		ID: "id-1", Name: "n", RunType: RunTypeChatModel,
		TraceID: "id-1", DottedOrder: "20260912T102030000000Zid-1",
		Tags:     []string{"a"},
		Metadata: map[string]any{"m": 1},
		Inputs:   "in",
		StartTime: time.Date(2026, 9, 12, 10, 20, 30, 0, time.UTC),
		Extra:     map[string]any{"metadata": map[string]any{"m": 1}},
		SessionID: "id-1", SessionName: "proj",
		ChildRuns: []*Run{{ID: "child"}},
	}
	raw, err := json.Marshal(run)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"id", "name", "run_type", "trace_id", "dotted_order",
		"tags", "inputs", "start_time", "extra", "session_id", "session_name"} {
		if _, ok := decoded[key]; !ok {
			t.Fatalf("payload %s missing key %q: %s", raw, key, raw)
		}
	}
	for _, absent := range []string{"metadata", "child_runs"} {
		if _, ok := decoded[absent]; ok {
			t.Fatalf("payload must not serialize %q: %s", absent, raw)
		}
	}
	if decoded["run_type"] != "chat_model" {
		t.Fatalf("run_type %v", decoded["run_type"])
	}
	// End-time/output/error are omitempty until the run finishes.
	if _, ok := decoded["end_time"]; ok {
		t.Fatalf("unfinished run serialized end_time: %s", raw)
	}
	run.Outputs, run.Error = "out", "boom"
	end := time.Date(2026, 9, 12, 10, 20, 31, 0, time.UTC)
	run.EndTime = &end
	raw, _ = json.Marshal(run)
	decoded = map[string]any{}
	_ = json.Unmarshal(raw, &decoded)
	for _, key := range []string{"outputs", "error", "end_time"} {
		if _, ok := decoded[key]; !ok {
			t.Fatalf("finished run payload missing %q: %s", key, raw)
		}
	}
	if decoded["parent_run_id"] != nil {
		t.Fatalf("empty parent_run_id must be omitted: %s", raw)
	}
}
