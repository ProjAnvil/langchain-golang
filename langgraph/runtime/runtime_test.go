package runtime

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/store"
)

// TestRuntimeSatisfiesContext is the compile-time assertion that Runtime
// implements context.Context (it is also asserted at package level via
// `var _ context.Context = Runtime{}`); this test makes the contract
// executable too.
func TestRuntimeSatisfiesContext(t *testing.T) {
	var ctx context.Context = Runtime{}
	_ = ctx // unused beyond the type check
}

// TestNewRuntimeDelegation verifies Deadline/Done/Err/Value delegate to the
// backing context.Context.
func TestNewRuntimeDelegation(t *testing.T) {
	t.Run("Background", func(t *testing.T) {
		rt := NewRuntime(context.Background())
		if got := rt.Err(); got != nil {
			t.Errorf("Err() = %v, want nil for an active Background ctx", got)
		}
		if dl, ok := rt.Deadline(); ok || !dl.IsZero() {
			t.Errorf("Deadline() = (%v, %v), want zero/false for Background", dl, ok)
		}
		if ch := rt.Done(); ch != nil {
			t.Errorf("Done() = %v, want nil for Background", ch)
		}
	})
	t.Run("Value delegates", func(t *testing.T) {
		type k struct{}
		ctx := context.WithValue(context.Background(), k{}, "hello")
		rt := NewRuntime(ctx)
		if got := rt.Value(k{}); got != "hello" {
			t.Errorf("Value() = %v, want %q", got, "hello")
		}
		// Unrelated keys fall through to the backing ctx (returns nil).
		if got := rt.Value("missing"); got != nil {
			t.Errorf("Value(missing) = %v, want nil", got)
		}
	})
	t.Run("Cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		rt := NewRuntime(ctx)
		cancel()
		select {
		case <-rt.Done():
		default:
			t.Fatalf("Done() channel not closed after cancel")
		}
		if err := rt.Err(); !errors.Is(err, context.Canceled) {
			t.Errorf("Err() = %v, want context.Canceled", err)
		}
	})
	t.Run("Deadline", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
		defer cancel()
		rt := NewRuntime(ctx)
		dl, ok := rt.Deadline()
		if !ok {
			t.Fatalf("Deadline() ok = false, want true")
		}
		if dl.IsZero() {
			t.Fatalf("Deadline() = zero time, want the timeout deadline")
		}
	})
	t.Run("NilCtxNormalized", func(t *testing.T) {
		// NewRuntime(nil) must not panic; methods stay safe to call.
		rt := NewRuntime(nil)
		if err := rt.Err(); err != nil {
			t.Errorf("Err() after nil ctx = %v, want nil", err)
		}
		if got := rt.Value("anything"); got != nil {
			t.Errorf("Value after nil ctx = %v, want nil", got)
		}
	})
}

// TestNewRuntimeDefaults verifies the nil defaults for StreamWriter/Heartbeat
// and the zero values for the remaining fields.
func TestNewRuntimeDefaults(t *testing.T) {
	rt := NewRuntime(context.Background())
	if rt.StreamWriter != nil {
		t.Errorf("NewRuntime StreamWriter is non-nil, want nil (unset sentinel)")
	}
	if rt.Heartbeat != nil {
		t.Errorf("NewRuntime Heartbeat is non-nil, want nil (unset sentinel)")
	}
	if rt.Context != nil {
		t.Errorf("NewRuntime Context = %v, want nil", rt.Context)
	}
	if rt.Store != nil {
		t.Errorf("NewRuntime Store = %v, want nil", rt.Store)
	}
	if rt.ExecutionInfo != nil {
		t.Errorf("NewRuntime ExecutionInfo = %v, want nil", rt.ExecutionInfo)
	}
	// NoOpStreamWriter/Heartbeat are callable without panicking when installed.
	rt = rt.Override(WithRuntimeStreamWriter(NoOpStreamWriter), WithRuntimeHeartbeat(NoOpHeartbeat))
	rt.StreamWriter("payload")
	rt.Heartbeat()
}

// TestMerge verifies per-field "other wins when set" semantics.
func TestMerge(t *testing.T) {
	t.Run("OtherWinsWhenSet", func(t *testing.T) {
		type k struct{}
		selfCtx := context.WithValue(context.Background(), k{}, "self")
		otherCtx := context.WithValue(context.Background(), k{}, "other")
		self := NewRuntime(selfCtx).
			Override(
				WithRuntimeContext("self-ctx"),
				WithRuntimePrevious("self-prev"),
			)
		other := NewRuntime(otherCtx).
			Override(
				WithRuntimeContext("other-ctx"),
				WithRuntimeStreamWriter(func(any) {}),
				WithRuntimePrevious("other-prev"),
			)
		merged := self.Merge(other)
		if merged.Context != "other-ctx" {
			t.Errorf("Context = %v, want other-ctx", merged.Context)
		}
		if merged.Previous != "other-prev" {
			t.Errorf("Previous = %v, want other-prev", merged.Previous)
		}
		if !streamWriterSet(merged.StreamWriter) {
			t.Errorf("StreamWriter = nil, want other's writer")
		}
		// Backing ctx tracks other.
		if got := merged.Value(k{}); got != "other" {
			t.Errorf("merged.Value = %v, want %q (other's ctx)", got, "other")
		}
	})
	t.Run("SelfKeepsWhenOtherUnset", func(t *testing.T) {
		self := NewRuntime(context.Background()).
			Override(
				WithRuntimeContext("self-ctx"),
				WithRuntimePrevious("self-prev"),
			)
		// other has zero-value Context/Previous and no-op writer.
		other := NewRuntime(context.Background())
		merged := self.Merge(other)
		if merged.Context != "self-ctx" {
			t.Errorf("Context = %v, want self-ctx (other was nil)", merged.Context)
		}
		if merged.Previous != "self-prev" {
			t.Errorf("Previous = %v, want self-prev (other was nil)", merged.Previous)
		}
	})
	t.Run("NilPreviousIsKept", func(t *testing.T) {
		// Python: previous=self.previous if other.previous is None else other.previous.
		self := NewRuntime(context.Background()).Override(WithRuntimePrevious("self-prev"))
		other := NewRuntime(context.Background()) // Previous nil
		merged := self.Merge(other)
		if merged.Previous != "self-prev" {
			t.Errorf("Previous = %v, want self-prev when other's is nil", merged.Previous)
		}
	})
}

// TestOverride replaces fields unconditionally.
func TestOverride(t *testing.T) {
	rt := NewRuntime(context.Background()).
		Override(
			WithRuntimeContext(42),
			WithRuntimePrevious("prev"),
		)
	if rt.Context != 42 {
		t.Errorf("Context = %v, want 42", rt.Context)
	}
	if rt.Previous != "prev" {
		t.Errorf("Previous = %v, want prev", rt.Previous)
	}
	// Backing ctx is preserved across Override (Override copies the struct).
	if rt.Err() != nil {
		t.Errorf("Err() after Override = %v, want nil", rt.Err())
	}
}

// TestPatchExecutionInfo mirrors Python's patch_execution_info, including the
// nil-error case.
func TestPatchExecutionInfo(t *testing.T) {
	t.Run("NilReturnsError", func(t *testing.T) {
		rt := NewRuntime(context.Background())
		if _, err := rt.PatchExecutionInfo(WithNodeAttempt(2)); err == nil {
			t.Fatalf("PatchExecutionInfo on nil ExecutionInfo: want error, got nil")
		}
	})
	t.Run("PatchesCopy", func(t *testing.T) {
		first := time.Now()
		info := &ExecutionInfo{
			CheckpointID:         "cp-1",
			TaskID:               "task-1",
			NodeAttempt:          1,
			NodeFirstAttemptTime: &first,
		}
		rt := NewRuntime(context.Background()).Override(WithRuntimeExecutionInfo(info))
		patched, err := rt.PatchExecutionInfo(WithNodeAttempt(2), WithCheckpointID("cp-2"))
		if err != nil {
			t.Fatalf("PatchExecutionInfo error: %v", err)
		}
		if patched.ExecutionInfo == info {
			t.Fatalf("PatchExecutionInfo mutated the original pointer (want a fresh copy)")
		}
		if got, want := patched.ExecutionInfo.NodeAttempt, 2; got != want {
			t.Errorf("NodeAttempt = %d, want %d", got, want)
		}
		if got, want := patched.ExecutionInfo.CheckpointID, "cp-2"; got != want {
			t.Errorf("CheckpointID = %q, want %q", got, want)
		}
		// Untouched fields survive.
		if got, want := patched.ExecutionInfo.TaskID, "task-1"; got != want {
			t.Errorf("TaskID = %q, want %q (untouched)", got, want)
		}
		if patched.ExecutionInfo.NodeFirstAttemptTime != &first {
			// Pointer should be carried through.
			if patched.ExecutionInfo.NodeFirstAttemptTime == nil {
				t.Errorf("NodeFirstAttemptTime = nil, want the original pointer")
			}
		}
	})
}

// TestExecutionInfoOverrides covers the ExecutionInfoOverride constructors not
// exercised elsewhere, verifying each one replaces exactly its own field.
func TestExecutionInfoOverrides(t *testing.T) {
	first := time.Now()
	info := ExecutionInfo{
		CheckpointID:         "cp-0",
		CheckpointNS:         "ns-0",
		TaskID:               "task-0",
		ThreadID:             "thread-0",
		RunID:                "run-0",
		NodeAttempt:          1,
		NodeFirstAttemptTime: nil,
	}
	out := info.Patch(
		WithCheckpointNS("ns-1"),
		WithTaskID("task-1"),
		WithThreadID("thread-1"),
		WithRunID("run-1"),
		WithNodeFirstAttemptTime(&first),
	)
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"CheckpointNS", out.CheckpointNS, "ns-1"},
		{"TaskID", out.TaskID, "task-1"},
		{"ThreadID", out.ThreadID, "thread-1"},
		{"RunID", out.RunID, "run-1"},
		// Untouched fields survive.
		{"CheckpointID", out.CheckpointID, "cp-0"},
		{"NodeAttempt", out.NodeAttempt, 1},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	if out.NodeFirstAttemptTime != &first {
		t.Errorf("NodeFirstAttemptTime = %v, want the patched pointer", out.NodeFirstAttemptTime)
	}
	// Original is not mutated (value receiver).
	if info.CheckpointNS != "ns-0" || info.TaskID != "task-0" || info.ThreadID != "thread-0" || info.RunID != "run-0" {
		t.Errorf("original ExecutionInfo mutated by Patch: %+v", info)
	}
}

// TestZeroRuntimeContextMethods verifies Deadline/Done/Err/Value are safe on a
// zero-value Runtime (nil backing ctx) and return the documented sentinels.
func TestZeroRuntimeContextMethods(t *testing.T) {
	var rt Runtime
	if dl, ok := rt.Deadline(); ok || !dl.IsZero() {
		t.Errorf("zero Runtime Deadline() = (%v, %v), want zero/false", dl, ok)
	}
	if ch := rt.Done(); ch != nil {
		t.Errorf("zero Runtime Done() = %v, want nil", ch)
	}
	if err := rt.Err(); err != nil {
		t.Errorf("zero Runtime Err() = %v, want nil", err)
	}
	if got := rt.Value("key"); got != nil {
		t.Errorf("zero Runtime Value() = %v, want nil", got)
	}
}

// TestMergeRemainingFields covers the Merge branches for Store, Heartbeat,
// ExecutionInfo, ServerInfo, and Control: other wins when set, self is kept
// when other's field is nil.
func TestMergeRemainingFields(t *testing.T) {
	selfStore := store.NewInMemoryStore()
	otherStore := store.NewInMemoryStore()
	selfInfo := &ExecutionInfo{TaskID: "self-task"}
	otherInfo := &ExecutionInfo{TaskID: "other-task"}
	selfServer := &ServerInfo{AssistantID: "self-asst"}
	otherServer := &ServerInfo{AssistantID: "other-asst"}
	selfControl := NewRunControl()
	otherControl := NewRunControl()
	selfHeartbeat := func() {}
	otherHeartbeat := func() {}

	t.Run("OtherWinsWhenSet", func(t *testing.T) {
		self := NewRuntime(context.Background()).Override(
			WithRuntimeStore(selfStore),
			WithRuntimeHeartbeat(selfHeartbeat),
			WithRuntimeExecutionInfo(selfInfo),
			WithRuntimeServerInfo(selfServer),
			WithRuntimeControl(selfControl),
		)
		other := NewRuntime(context.Background()).Override(
			WithRuntimeStore(otherStore),
			WithRuntimeHeartbeat(otherHeartbeat),
			WithRuntimeExecutionInfo(otherInfo),
			WithRuntimeServerInfo(otherServer),
			WithRuntimeControl(otherControl),
		)
		merged := self.Merge(other)
		if merged.Store != Store(otherStore) {
			t.Errorf("Store: want other's store")
		}
		if !heartbeatSet(merged.Heartbeat) {
			t.Errorf("Heartbeat = nil, want other's heartbeat")
		}
		if merged.ExecutionInfo != otherInfo {
			t.Errorf("ExecutionInfo = %v, want other's pointer", merged.ExecutionInfo)
		}
		if merged.ServerInfo != otherServer {
			t.Errorf("ServerInfo = %v, want other's pointer", merged.ServerInfo)
		}
		if merged.Control != otherControl {
			t.Errorf("Control = %v, want other's pointer", merged.Control)
		}
	})
	t.Run("SelfKeepsWhenOtherUnset", func(t *testing.T) {
		self := NewRuntime(context.Background()).Override(
			WithRuntimeStore(selfStore),
			WithRuntimeHeartbeat(selfHeartbeat),
			WithRuntimeExecutionInfo(selfInfo),
			WithRuntimeServerInfo(selfServer),
			WithRuntimeControl(selfControl),
		)
		other := NewRuntime(context.Background()) // all fields nil
		merged := self.Merge(other)
		if merged.Store != Store(selfStore) {
			t.Errorf("Store: want self's store (other was nil)")
		}
		if !heartbeatSet(merged.Heartbeat) {
			t.Errorf("Heartbeat = nil, want self's heartbeat (other was nil)")
		}
		if merged.ExecutionInfo != selfInfo {
			t.Errorf("ExecutionInfo = %v, want self's pointer (other was nil)", merged.ExecutionInfo)
		}
		if merged.ServerInfo != selfServer {
			t.Errorf("ServerInfo = %v, want self's pointer (other was nil)", merged.ServerInfo)
		}
		if merged.Control != selfControl {
			t.Errorf("Control = %v, want self's pointer (other was nil)", merged.Control)
		}
	})
}

// TestWithRuntimeCtx verifies WithRuntimeCtx swaps the backing context.Context
// and ignores a nil replacement.
func TestWithRuntimeCtx(t *testing.T) {
	type k struct{}
	rt := NewRuntime(context.Background()).Override(WithRuntimeContext("static"))
	derived := context.WithValue(context.Background(), k{}, "derived")
	swapped := rt.Override(WithRuntimeCtx(derived))
	if got := swapped.Value(k{}); got != "derived" {
		t.Errorf("Value after WithRuntimeCtx = %v, want %q", got, "derived")
	}
	// Other fields are preserved across the ctx swap.
	if swapped.Context != "static" {
		t.Errorf("Context = %v, want static (preserved across ctx swap)", swapped.Context)
	}
	// The original runtime is untouched (Override copies).
	if got := rt.Value(k{}); got != nil {
		t.Errorf("original Value = %v, want nil (original ctx unchanged)", got)
	}
	// A nil ctx is ignored: the previous backing ctx survives.
	ctx2, cancel := context.WithCancel(context.Background())
	defer cancel()
	rt2 := NewRuntime(ctx2).Override(WithRuntimeCtx(nil))
	cancel()
	if err := rt2.Err(); !errors.Is(err, context.Canceled) {
		t.Errorf("Err after WithRuntimeCtx(nil) = %v, want context.Canceled (nil ctx ignored)", err)
	}
}

// TestContextSchemaValues covers ContextWithValues / ValuesFromContext /
// ValueFromRuntime — the context_schema data flow shared with the agents and
// graph packages.
func TestContextSchemaValues(t *testing.T) {
	t.Run("RoundTrip", func(t *testing.T) {
		values := map[string]any{"user_id": "u-1", "db_conn": "conn"}
		ctx := ContextWithValues(context.Background(), values)
		got := ValuesFromContext(ctx)
		if got["user_id"] != "u-1" || got["db_conn"] != "conn" {
			t.Errorf("ValuesFromContext = %v, want %v", got, values)
		}
	})
	t.Run("NoValuesAttached", func(t *testing.T) {
		if got := ValuesFromContext(context.Background()); got != nil {
			t.Errorf("ValuesFromContext on bare ctx = %v, want nil", got)
		}
	})
	t.Run("NilContext", func(t *testing.T) {
		//nolint:staticcheck // deliberately probing nil-ctx safety
		if got := ValuesFromContext(nil); got != nil {
			t.Errorf("ValuesFromContext(nil) = %v, want nil", got)
		}
	})
	t.Run("ValueFromRuntime", func(t *testing.T) {
		rt := NewRuntime(context.Background()).Override(
			WithRuntimeContext(map[string]any{"user_id": "u-1"}),
		)
		if v, ok := ValueFromRuntime(rt, "user_id"); !ok || v != "u-1" {
			t.Errorf("ValueFromRuntime(user_id) = (%v, %v), want (u-1, true)", v, ok)
		}
		if v, ok := ValueFromRuntime(rt, "missing"); ok || v != nil {
			t.Errorf("ValueFromRuntime(missing) = (%v, %v), want (nil, false)", v, ok)
		}
	})
	t.Run("ValueFromRuntimeNonMapContext", func(t *testing.T) {
		// A non-map Context (including nil) yields (nil, false).
		if v, ok := ValueFromRuntime(NewRuntime(context.Background()), "k"); ok || v != nil {
			t.Errorf("ValueFromRuntime with nil Context = (%v, %v), want (nil, false)", v, ok)
		}
		rt := NewRuntime(context.Background()).Override(WithRuntimeContext("not-a-map"))
		if v, ok := ValueFromRuntime(rt, "k"); ok || v != nil {
			t.Errorf("ValueFromRuntime with string Context = (%v, %v), want (nil, false)", v, ok)
		}
	})
}

// TestOverrideStoreAndServerInfo covers the Override options not exercised
// elsewhere (WithRuntimeStore, WithRuntimeServerInfo).
func TestOverrideStoreAndServerInfo(t *testing.T) {
	st := store.NewInMemoryStore()
	si := &ServerInfo{AssistantID: "asst-1", GraphID: "graph-1", User: "user-1"}
	rt := NewRuntime(context.Background()).Override(
		WithRuntimeStore(st),
		WithRuntimeServerInfo(si),
	)
	if rt.Store != Store(st) {
		t.Errorf("Store = %v, want the installed store", rt.Store)
	}
	if rt.ServerInfo != si {
		t.Errorf("ServerInfo = %v, want the installed pointer", rt.ServerInfo)
	}
}

// TestExecutionInfoPatch mirrors Python's ExecutionInfo.patch (replace).
func TestExecutionInfoPatch(t *testing.T) {
	info := ExecutionInfo{CheckpointID: "a", TaskID: "t", NodeAttempt: 1}
	out := info.Patch(WithCheckpointID("b"), WithNodeAttempt(3))
	if out.CheckpointID != "b" {
		t.Errorf("CheckpointID = %q, want b", out.CheckpointID)
	}
	if out.NodeAttempt != 3 {
		t.Errorf("NodeAttempt = %d, want 3", out.NodeAttempt)
	}
	if out.TaskID != "t" {
		t.Errorf("TaskID = %q, want t (untouched)", out.TaskID)
	}
	// Original is not mutated (value receiver).
	if info.CheckpointID != "a" {
		t.Errorf("original CheckpointID mutated to %q", info.CheckpointID)
	}
}

// TestRunControl covers the drain lifecycle.
func TestRunControl(t *testing.T) {
	t.Run("Fresh", func(t *testing.T) {
		c := NewRunControl()
		if c.DrainRequested() {
			t.Errorf("DrainRequested = true on fresh control")
		}
		if got := c.DrainReason(); got != "" {
			t.Errorf("DrainReason = %q, want empty", got)
		}
	})
	t.Run("RequestThenRead", func(t *testing.T) {
		c := NewRunControl()
		c.RequestDrain("shutdown")
		if !c.DrainRequested() {
			t.Errorf("DrainRequested = false after RequestDrain")
		}
		if got, want := c.DrainReason(), "shutdown"; got != want {
			t.Errorf("DrainReason = %q, want %q", got, want)
		}
	})
	t.Run("DefaultReason", func(t *testing.T) {
		c := NewRunControl()
		c.RequestDrain("")
		if got, want := c.DrainReason(), "shutdown"; got != want {
			t.Errorf("DrainReason after empty reason = %q, want default %q", got, want)
		}
	})
	t.Run("FirstWriterWins", func(t *testing.T) {
		c := NewRunControl()
		c.RequestDrain("shutdown")
		c.RequestDrain("timeout")
		if got, want := c.DrainReason(), "shutdown"; got != want {
			t.Errorf("DrainReason overwritten to %q, want first %q", got, want)
		}
	})
	t.Run("NilSafe", func(t *testing.T) {
		var c *RunControl
		if c.DrainRequested() {
			t.Errorf("nil DrainRequested = true, want false")
		}
		if got := c.DrainReason(); got != "" {
			t.Errorf("nil DrainReason = %q, want empty", got)
		}
		c.RequestDrain("shutdown") // must not panic
	})
}

// TestRuntimeDrainDelegates verifies Runtime.DrainRequested/DrainReason
// delegate to Control and tolerate a nil Control.
func TestRuntimeDrainDelegates(t *testing.T) {
	rt := NewRuntime(context.Background())
	if rt.DrainRequested() {
		t.Errorf("DrainRequested = true without Control")
	}
	if got := rt.DrainReason(); got != "" {
		t.Errorf("DrainReason = %q, want empty without Control", got)
	}
	ctrl := NewRunControl()
	rt = rt.Override(WithRuntimeControl(ctrl))
	ctrl.RequestDrain("shutdown")
	if !rt.DrainRequested() {
		t.Errorf("DrainRequested = false after Control.RequestDrain")
	}
	if got, want := rt.DrainReason(), "shutdown"; got != want {
		t.Errorf("DrainReason = %q, want %q", got, want)
	}
}
