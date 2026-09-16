package fn

import (
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// §6.10 (design t15-design.md): an Entrypoint's internal compiled graph used
// as a StateGraph subgraph node — the composition Python expresses as
// add_node(name, entrypoint_func). The parent state flows in through the
// __start__ channel and the entrypoint's {__end__, __previous__} writes merge
// back as the subgraph node's update. These tests are white-box (package fn)
// because the internal graph is not exported; they lock the interrupt
// propagation, resume forwarding, and — the §8.2 risk — the fn task-replay /
// ReservedFnConsumed alignment across pause/resume cycles.

// fnSubgraphTop builds the composition: top's single node "fn" is the
// entrypoint's internal compiled graph, added via AddSubgraph so the T15
// wrapper (pause propagation + resume forwarding + per-task child namespace)
// applies.
func fnSubgraphTop(t *testing.T, e *Entrypoint[any, string, any], saver *checkpoint.MemorySaver) *graph.CompiledGraph {
	t.Helper()
	top := graph.NewStateGraph()
	top.AddSubgraph("fn", e.graph)
	top.AddEdge(types.START, "fn")
	top.AddEdge("fn", types.END)
	cg, err := top.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}
	return cg
}

// TestEntrypointSubgraphInterruptPropagatesAndResumes (§6.10, propagation
// half): an Interrupt inside the entrypoint function pauses the PARENT run —
// the propagated interrupt carries the nested task NS
// (<fn>:<taskID>/entrypoint:<taskID>) — and a scalar resume forwarded into
// the pinned child run answers it; the entrypoint's __end__ write merges back
// into the parent state.
func TestEntrypointSubgraphInterruptPropagatesAndResumes(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	var runs atomic.Int32
	e, err := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: saver},
		func(rt runtime.Runtime, _ any, _ any, _ bool) (string, error) {
			runs.Add(1)
			v, _ := graph.Interrupt(rt, "fn-q").(string)
			return "answered:" + v, nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}
	cg := fnSubgraphTop(t, e, saver)

	res, err := cg.InvokeWithOptions(ctx, map[string]any{"__start__": "in"}, graph.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result (the entrypoint interrupt propagates)", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want the single entrypoint interrupt", res.Interrupts)
	}
	intr := res.Interrupts[0]
	if intr.Value != "fn-q" || intr.ID != "entrypoint-1" {
		t.Fatalf("interrupt = %+v, want the entrypoint node's own identity", intr)
	}
	fnNS, epSeg, ok := strings.Cut(intr.NS, "/")
	if !ok || !strings.HasPrefix(fnNS, "fn:") || !strings.HasPrefix(epSeg, "entrypoint:") {
		t.Fatalf("interrupt NS = %q, want <fn>:<taskID>/entrypoint:<taskID>", intr.NS)
	}

	res2, err := cg.InvokeWithOptions(ctx, nil, graph.Options{ThreadID: "t1", Resume: "42"})
	if err != nil {
		t.Fatalf("resume error = %v, want completion", err)
	}
	if len(res2.Interrupts) != 0 {
		t.Fatalf("resume Interrupts = %+v, want none", res2.Interrupts)
	}
	if got := res2.Values["__end__"]; got != "answered:42" {
		t.Fatalf("Values[__end__] = %v, want the entrypoint result merged into the parent", res2.Values)
	}
	if runs.Load() != 2 {
		t.Fatalf("entrypoint runs = %d, want 2 (paused once, re-executed once for the resume)", runs.Load())
	}
}

// TestEntrypointSubgraphTaskReplayAcrossPauses (§6.10, replay half — the
// §8.2 ReservedFnConsumed alignment): across a pause -> resume -> pause ->
// resume chain, the entrypoint re-executes as a whole function while its
// tasks' persisted results REPLAY (no re-execution), and a task whose
// execution consumed a resume value re-skips it on replay so the entrypoint's
// NEXT Interrupt aligns with the newly supplied value — not the one already
// consumed. Misalignment here would surface as the second interrupt
// re-receiving the first resume value.
func TestEntrypointSubgraphTaskReplayAcrossPauses(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	var setupCalls, askCalls atomic.Int32
	setup := NewTask[any, string]("setup", func(_ runtime.Runtime, _ any) (string, error) {
		setupCalls.Add(1)
		return "S", nil
	}, TaskOpts{})
	// ask's own body interrupts: its execution consumes one resume value.
	ask := NewTask[any, string]("ask", func(rt runtime.Runtime, _ any) (string, error) {
		askCalls.Add(1)
		v, _ := graph.Interrupt(rt, "task-q").(string)
		return "got:" + v, nil
	}, TaskOpts{})
	e, err := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: saver},
		func(rt runtime.Runtime, _ any, _ any, _ bool) (string, error) {
			s, err := setup.Call(rt, nil).Get(rt)
			if err != nil {
				return "", err
			}
			a, err := ask.Call(rt, nil).Get(rt)
			if err != nil {
				return "", err
			}
			// This direct interrupt must receive the SECOND resume value; the
			// first was consumed inside ask's (now replayed) execution.
			tail, _ := graph.Interrupt(rt, "entrypoint-q").(string)
			return s + a + tail, nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}
	cg := fnSubgraphTop(t, e, saver)

	// Pause 1: ask's Interrupt fires on the empty queue.
	res1, err := cg.InvokeWithOptions(ctx, map[string]any{"__start__": "in"}, graph.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v, want a paused Result", err)
	}
	if len(res1.Interrupts) != 1 || res1.Interrupts[0].Value != "task-q" {
		t.Fatalf("pause 1 Interrupts = %+v, want ask's task interrupt", res1.Interrupts)
	}
	if setupCalls.Load() != 1 || askCalls.Load() != 1 {
		t.Fatalf("after pause 1: setup=%d ask=%d, want 1 each", setupCalls.Load(), askCalls.Load())
	}

	// Resume 1 ("v1"): setup replays; ask re-executes and consumes v1; the
	// direct entrypoint interrupt then pauses again ("entrypoint-q").
	res2, err := cg.InvokeWithOptions(ctx, nil, graph.Options{ThreadID: "t1", Resume: "v1"})
	if err != nil {
		t.Fatalf("resume 1 error = %v, want a re-pause", err)
	}
	if len(res2.Interrupts) != 1 || res2.Interrupts[0].Value != "entrypoint-q" {
		t.Fatalf("pause 2 Interrupts = %+v, want the entrypoint's direct interrupt", res2.Interrupts)
	}
	if setupCalls.Load() != 1 {
		t.Fatalf("setup calls after resume 1 = %d, want 1 (result replayed, not re-executed)", setupCalls.Load())
	}

	// Resume 2 ("v2"): setup AND ask both replay; ask's replay re-skips the
	// consumed v1 (ReservedFnConsumed), so the direct interrupt receives v2.
	res3, err := cg.InvokeWithOptions(ctx, nil, graph.Options{ThreadID: "t1", Resume: "v2"})
	if err != nil {
		t.Fatalf("resume 2 error = %v, want completion", err)
	}
	if len(res3.Interrupts) != 0 {
		t.Fatalf("resume 2 Interrupts = %+v, want none", res3.Interrupts)
	}
	if got := res3.Values["__end__"]; got != "Sgot:v1v2" {
		t.Fatalf("Values[__end__] = %v, want Sgot:v1v2 (v1 answered ask, v2 the direct interrupt)", res3.Values)
	}
	if setupCalls.Load() != 1 || askCalls.Load() != 2 {
		t.Fatalf("after resume 2: setup=%d ask=%d, want setup=1 (replayed every resume) and ask=2 "+
			"(executed during the two pauses; replayed in the final resume)", setupCalls.Load(), askCalls.Load())
	}
}
