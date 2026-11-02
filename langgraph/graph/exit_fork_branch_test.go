package graph

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// forkBranchGraph builds START -> a -> b over two channels: "log", a
// DeltaChannel with snapshotFrequency=100 (never cadence-snapshotted, so its
// value lives only as per-checkpoint pending writes reconstructed by the
// ancestor walk), and "plain", a plain accumulating reducer channel every
// checkpoint stores in full. Each node appends its turn-tagged token to both
// channels, so every turn contributes an input delta write plus one node
// delta write per superstep.
func forkBranchGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddReducer("plain", channels.AppendSliceReducer)
	node := func(name string) func(runtime.Runtime, map[string]any) (any, error) {
		return func(_ runtime.Runtime, state map[string]any) (any, error) {
			token := fmt.Sprintf("%s%v", name, state["turn"])
			return map[string]any{"log": []string{token}, "plain": []string{token}}, nil
		}
	}
	g.AddNode("a", node("a"))
	g.AddNode("b", node("b"))
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// mustGetLog reads the reconstructed "log" delta channel at cfg.
func mustGetLog(t *testing.T, cg *CompiledGraph, cfg checkpoint.Config) []string {
	t.Helper()
	snap, err := cg.GetState(t.Context(), cfg)
	if err != nil {
		t.Fatalf("GetState(%s) error = %v", cfg.CheckpointID, err)
	}
	got, _ := snap.Values["log"].([]string)
	return got
}

// TestExitDurabilityPinnedForkKeepsNewerBranchClean pins #9269: an
// exit-durability run pinned via Options.CheckpointID at a checkpoint the
// thread already moved past must not store its delta writes on that shared
// checkpoint — the newer branch's ancestor walk would replay them as its own
// state (durable cross-branch contamination). Mirrors Python's
// test_exit_runs_from_a_checkpoint_the_thread_moved_past_leave_its_newer_branch_alone
// (test_delta_channel_exit_mode.py): the pinned run instead force-snapshots
// the delta channels it wrote into its own final checkpoint.
func TestExitDurabilityPinnedForkKeepsNewerBranchClean(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	cg := forkBranchGraph(t, WithCheckpointer(saver)) // turns 1-2 run sync

	invoke := func(turn int, opts Options) (Result, error) {
		token := fmt.Sprintf("in%d", turn)
		return cg.InvokeWithOptions(ctx, map[string]any{
			"turn":  turn,
			"log":   []string{token},
			"plain": []string{token},
		}, opts)
	}

	// Turn 1 runs to completion; its final checkpoint is A.
	if _, err := invoke(1, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	snapA, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 1 error = %v", err)
	}
	cpA := snapA.Config.CheckpointID

	// Turn 2 continues the thread past A: the newer branch B.
	if _, err := invoke(2, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}
	snapB, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 2 error = %v", err)
	}
	cpB := snapB.Config.CheckpointID

	// Turn 3: an exit-durability run PINNED at A while B is the latest.
	// A second pinned run (Python's exit_runs=2 parametrization) repeats the
	// fork from the same superseded A, so both fork finals descend from it.
	for turn := 3; turn <= 4; turn++ {
		want := []string{"in1", "a1", "b1", fmt.Sprintf("in%d", turn), fmt.Sprintf("a%d", turn), fmt.Sprintf("b%d", turn)}
		res, err := invoke(turn, Options{ThreadID: "t", CheckpointID: cpA}.WithRunDurability(DurabilityExit))
		if err != nil {
			t.Fatalf("pinned exit Invoke() (turn %d) error = %v", turn, err)
		}
		if got, _ := res.Values["log"].([]string); !reflect.DeepEqual(got, want) {
			t.Fatalf("pinned run %d live log = %v, want %v", turn, got, want)
		}

		// (a) The newer branch B must NOT read the pinned run's writes: its
		// ancestor walk must not replay anything the forked run stored.
		if got := mustGetLog(t, cg, checkpoint.Config{ThreadID: "t", CheckpointID: cpB}); !reflect.DeepEqual(got, []string{"in1", "a1", "b1", "in2", "a2", "b2"}) {
			t.Fatalf("turn %d: B-branch log = %v, want [in1 a1 b1 in2 a2 b2] (pinned run's writes leaked into the newer branch)", turn, got)
		}

		// (b) The pinned run's own branch still reads its writes — via the
		// forced snapshot embedded in its final checkpoint (the thread's
		// latest), not via writes anchored on A.
		if got := mustGetLog(t, cg, checkpoint.Config{ThreadID: "t"}); !reflect.DeepEqual(got, want) {
			t.Fatalf("turn %d: forked branch log = %v, want %v", turn, got, want)
		}
		tupF, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
		if err != nil || tupF == nil {
			t.Fatalf("turn %d: GetTuple(latest) = (%v, %v)", turn, tupF, err)
		}
		if _, ok := channels.UnwrapDeltaSnapshot(tupF.Checkpoint.ChannelValues["log"]); !ok {
			t.Fatalf("turn %d: final checkpoint log = %#v, want a forced delta snapshot blob (writes anchored on A instead)", turn, tupF.Checkpoint.ChannelValues["log"])
		}
	}

	// (c) Python's cross-check: every checkpoint in the thread's history
	// reconstructs the delta channel to exactly the plain channel's value.
	hist, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	if len(hist) < 6 {
		t.Fatalf("history has %d checkpoints, want at least 6 (two sync turns + the exit finals)", len(hist))
	}
	for _, s := range hist {
		log, _ := s.Values["log"].([]string)
		plain, _ := s.Values["plain"].([]string)
		if !reflect.DeepEqual(log, plain) {
			t.Fatalf("checkpoint %s: log = %v, plain = %v (delta reconstruction diverged from plain state)", s.Config.CheckpointID, log, plain)
		}
	}
}

// exitPinResumeGraph builds in -> a -> q -> r under exit durability: `a` and
// `r` write to both `log` (a DeltaChannel whose cadence never fires, so its
// value lives only as anchored pending writes) and `plain` (the plain
// always-stored oracle channel), and `q` interrupts in-node until its
// interrupt is answered by a resume value.
func exitPinResumeGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddReducer("plain", channels.AppendSliceReducer)
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"a"}, "plain": []string{"a"}}, nil
	})
	g.AddNode("q", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(rt, "q?")
		return nil, nil
	})
	g.AddNode("r", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"r"}, "plain": []string{"r"}}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "q")
	g.AddEdge("q", "r")
	g.AddEdge("r", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityExit))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestExitPinnedResumeDoesNotReplayAnchoredWrites pins the exit-mode time
// travel hazard (#8548's anchored-shape counterpart): an exit run that resumes
// from pause checkpoint P anchors its accumulated writes on P under synthetic
// step-prefixed task ids while P.Next still plans the very tasks that ran. A
// later resume pinned at P must re-execute those tasks (the synthetic ids keep
// them from LOOKING done — the wave4/#9114 invariant) against P's own state,
// not against state seeded with the consuming branch's anchored writes: r's
// write would otherwise apply twice. The fork's checkpoints must also seal the
// channel (forced snapshot) so the abandoned anchored write never replays into
// the fork's stored state either. Mirrors Python, whose loaded pending writes
// never hydrate live state (only reapply, which cannot match a synthetic id,
// feeds them back — _loop.py:833-845) and which force-snapshots the delta
// channels of loaded writes that did not go back to their task
// (_loop.py:708-722) and of every pending-write channel on a replaying run
// (_loop.py:1011-1015, #8548).
func TestExitPinnedResumeDoesNotReplayAnchoredWrites(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	cg := exitPinResumeGraph(t, saver)

	// Turn 1: pause at q. P anchors the input write and a's write (early-path
	// synthetic ids — P's own pre-checkpoint state) and plans q in Next.
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []string{"in"}, "plain": []string{"in"}}, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	snapP, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 1 error = %v", err)
	}
	cpP := snapP.Config.CheckpointID

	// Turn 2: a LIVE resume completes the thread; flushExit anchors r's
	// accumulated write (a later-superstep synthetic id) on P and saves the
	// final checkpoint past it.
	res, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "go"})
	if err != nil {
		t.Fatalf("turn 2 resume Invoke() error = %v", err)
	}
	if got, _ := res.Values["log"].([]string); !reflect.DeepEqual(got, []string{"in", "a", "r"}) {
		t.Fatalf("turn 2 live log = %v, want [in a r]", got)
	}

	// Turn 3: pin P (time travel). Both durability modes of the fork must
	// re-derive the branch from P's own state: q re-executes (wave4: the
	// paused task does not look done) and r's write applies exactly once —
	// live, on read, and against the plain-channel oracle.
	for _, tc := range []struct {
		name string
		opts Options
	}{
		{"exit", Options{ThreadID: "t", CheckpointID: cpP, Resume: "again"}},
		{"sync", Options{ThreadID: "t", CheckpointID: cpP, Resume: "again"}.WithRunDurability(DurabilitySync)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res, err := cg.InvokeWithOptions(ctx, nil, tc.opts)
			if err != nil {
				t.Fatalf("pinned resume Invoke() error = %v", err)
			}
			want := []string{"in", "a", "r"}
			log, _ := res.Values["log"].([]string)
			if !reflect.DeepEqual(log, want) {
				t.Fatalf("pinned resume live log = %v, want %v (the consuming branch's anchored r write replayed into the fork)", log, want)
			}
			plain, _ := res.Values["plain"].([]string)
			if !reflect.DeepEqual(log, plain) {
				t.Fatalf("pinned resume log = %v diverged from plain oracle = %v", log, plain)
			}
			if got := mustGetLog(t, cg, checkpoint.Config{ThreadID: "t"}); !reflect.DeepEqual(got, want) {
				t.Fatalf("pinned resume stored log = %v, want %v (the anchored r write leaked into the fork branch's stored state)", got, want)
			}
		})
	}

	// The plain-channel oracle must hold across the whole mixed-branch
	// history for every checkpoint that does not itself carry anchored
	// exit-delta log writes: such a checkpoint's read legitimately folds its
	// own pending writes in (Python's get_state of the fork base does the
	// same — #8548's "fork base" line includes the pending 'first-out'
	// write), while every FORK checkpoint must be sealed against the
	// abandoned branch's anchored writes.
	hist, err := cg.GetStateHistory(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory() error = %v", err)
	}
	for _, s := range hist {
		anchored := false
		tup, err := saver.GetTuple(ctx, s.Config)
		if err != nil || tup == nil {
			t.Fatalf("GetTuple(%s) = (%v, %v)", s.Config.CheckpointID, tup, err)
		}
		for _, w := range tup.PendingWrites {
			if _, ok := parseExitDeltaTaskID(w.TaskID); ok && w.Channel == "log" {
				anchored = true
				break
			}
		}
		if anchored {
			continue
		}
		log, _ := s.Values["log"].([]string)
		plain, _ := s.Values["plain"].([]string)
		if !reflect.DeepEqual(log, plain) {
			t.Fatalf("checkpoint %s: log = %v, plain = %v (delta reconstruction diverged from plain state)", s.Config.CheckpointID, log, plain)
		}
	}
}

// exitPauseBoundaryGraph builds in -> a -> q1 -> b -> q2 -> r under exit
// durability: `a` and `b` write to `log`/`plain`, and q1/q2 interrupt in-node
// until answered by a resume value — so the second pause checkpoint P2 anchors
// b's early-path write at a step exactly equal to P2's own metadata step.
func exitPauseBoundaryGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddReducer("plain", channels.AppendSliceReducer)
	write := func(token string) func(runtime.Runtime, map[string]any) (any, error) {
		return func(_ runtime.Runtime, _ map[string]any) (any, error) {
			return map[string]any{"log": []string{token}, "plain": []string{token}}, nil
		}
	}
	ask := func(name string) func(runtime.Runtime, map[string]any) (any, error) {
		return func(rt runtime.Runtime, _ map[string]any) (any, error) {
			Interrupt(rt, name+"?")
			return nil, nil
		}
	}
	g.AddNode("a", write("a"))
	g.AddNode("q1", ask("q1"))
	g.AddNode("b", write("b"))
	g.AddNode("q2", ask("q2"))
	g.AddNode("r", write("r"))
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "q1")
	g.AddEdge("q1", "b")
	g.AddEdge("b", "q2")
	g.AddEdge("q2", "r")
	g.AddEdge("r", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityExit))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestExitPinnedResumeKeepsPauseAnchoredWrites guards the boundary of the
// stale-anchored-write exclusion (see TestExitPinnedResumeDoesNotReplay
// AnchoredWrites): putPauseCheckpointExit anchors the run's accumulated
// writes on the pause checkpoint ITSELF (the Go-only early path), and those
// writes' embedded steps are at or before the pause checkpoint's metadata
// step — they are the checkpoint's OWN state, not a consuming branch's work.
// A pinned resume from such a checkpoint must keep seeding them.
func TestExitPinnedResumeKeepsPauseAnchoredWrites(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	cg := exitPauseBoundaryGraph(t, saver)

	// Turn 1: pause at q1.
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []string{"in"}, "plain": []string{"in"}}, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	// Turn 2: answer q1, run b, pause again at q2. P2 anchors b's write
	// (early path) at a step equal to P2's own metadata step.
	res, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "go"})
	if err != nil {
		t.Fatalf("turn 2 resume Invoke() error = %v", err)
	}
	if got, _ := res.Values["log"].([]string); !reflect.DeepEqual(got, []string{"in", "a", "b"}) {
		t.Fatalf("turn 2 live log = %v, want [in a b]", got)
	}
	snapP2, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 2 error = %v", err)
	}
	cpP2 := snapP2.Config.CheckpointID

	// Turn 3: pin P2 (now superseded by the completed fork below is NOT
	// required — P2 is the latest; the pin itself makes the run addressed)
	// and answer q2: the run must continue from P2's full state.
	res, err = cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", CheckpointID: cpP2, Resume: "done"})
	if err != nil {
		t.Fatalf("pinned resume Invoke() error = %v", err)
	}
	want := []string{"in", "a", "b", "r"}
	if got, _ := res.Values["log"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("pinned resume live log = %v, want %v (the pause checkpoint's own anchored b write was dropped)", got, want)
	}
	if got := mustGetLog(t, cg, checkpoint.Config{ThreadID: "t"}); !reflect.DeepEqual(got, want) {
		t.Fatalf("pinned resume stored log = %v, want %v", got, want)
	}
}

// TestExitDurabilityPinnedAtLatestSnapshotsForFutureForks freezes the widened
// #9269 gate: Python force-snapshots whenever a run was loaded by an
// ADDRESSED checkpoint_id (`not _loaded_latest`) — even when that checkpoint
// IS the namespace's latest ("costs one snapshot of each channel the run
// wrote, even when the addressed checkpoint is the latest one"). Anchoring on
// the latest would be safe only until the thread forks from it: a later
// time-travel pin or a concurrent sibling exit run from the same checkpoint
// would replay the anchored writes as its own state.
func TestExitDurabilityPinnedAtLatestSnapshotsForFutureForks(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	cg := forkBranchGraph(t, WithCheckpointer(saver))

	invoke := func(turn int, opts Options) (Result, error) {
		token := fmt.Sprintf("in%d", turn)
		return cg.InvokeWithOptions(ctx, map[string]any{
			"turn":  turn,
			"log":   []string{token},
			"plain": []string{token},
		}, opts)
	}

	// Turn 1 runs to completion; its final checkpoint A is the latest.
	if _, err := invoke(1, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	snapA, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 1 error = %v", err)
	}
	cpA := snapA.Config.CheckpointID

	// Turn 2 pins A explicitly while A is STILL the latest: the run must
	// force-snapshot (addressed load), leaving no anchored writes on A.
	if _, err := invoke(2, Options{ThreadID: "t", CheckpointID: cpA}.WithRunDurability(DurabilityExit)); err != nil {
		t.Fatalf("pinned-at-latest Invoke() error = %v", err)
	}
	tupF2, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tupF2 == nil {
		t.Fatalf("GetTuple(latest) after turn 2 = (%v, %v)", tupF2, err)
	}
	if _, ok := channels.UnwrapDeltaSnapshot(tupF2.Checkpoint.ChannelValues["log"]); !ok {
		t.Fatalf("turn-2 final checkpoint log = %#v, want a forced delta snapshot blob (pinned-at-latest must snapshot, not anchor)", tupF2.Checkpoint.ChannelValues["log"])
	}
	tupA, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t", CheckpointID: cpA})
	if err != nil || tupA == nil {
		t.Fatalf("GetTuple(A) = (%v, %v)", tupA, err)
	}
	for _, w := range tupA.PendingWrites {
		if w.Channel != "log" {
			continue
		}
		for _, tok := range w.Value.([]string) {
			if tok == "in2" || tok == "a2" || tok == "b2" {
				t.Fatalf("A carries the pinned-at-latest run's %q write (anchored instead of snapshotted): pending log writes = %+v", tok, tupA.PendingWrites)
			}
		}
	}

	// Turn 3 forks from A (now superseded by turn 2's final): it must see
	// A's state only — none of turn 2's writes, live or on read.
	res, err := invoke(3, Options{ThreadID: "t", CheckpointID: cpA}.WithRunDurability(DurabilityExit))
	if err != nil {
		t.Fatalf("fork Invoke() error = %v", err)
	}
	if got, _ := res.Values["log"].([]string); !reflect.DeepEqual(got, []string{"in1", "a1", "b1", "in3", "a3", "b3"}) {
		t.Fatalf("fork live log = %v, want [in1 a1 b1 in3 a3 b3] (turn-2 writes replayed into the fork)", got)
	}
	if got := mustGetLog(t, cg, checkpoint.Config{ThreadID: "t"}); !reflect.DeepEqual(got, []string{"in1", "a1", "b1", "in3", "a3", "b3"}) {
		t.Fatalf("forked branch log = %v, want [in1 a1 b1 in3 a3 b3] (turn-2 writes replayed into the fork)", got)
	}
}
