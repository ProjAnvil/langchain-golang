package fn

// End-to-end port of Python's functional-API test cases from
// langgraph/libs/langgraph/tests/test_pregel.py (each test cites its source
// lines). Porting policy:
//
//   - Python parametrizes over several checkpointers; the Go ports pin
//     checkpoint.NewMemorySaver() (sqlite/postgres saver contracts are
//     covered by the savers' own tests).
//   - jsonschema assertions (the get_input/output_jsonschema blocks of
//     test_imp_task / test_imp_nested) are NOT ported: Go has no jsonschema
//     concept. For the same reason test_entrypoint_output_schema_with_return_and_save
//     (test_pregel.py:6755-6783) — which asserts nothing but output-schema
//     inference for entrypoint.final — has no Go counterpart at all.
//   - Per-task stream chunks ({"mapper": ...} updates items) are NOT
//     asserted: Go fn tasks run inside the entrypoint node and do not
//     produce individual updates chunks (documented divergence — see
//     Entrypoint.Stream). Result/replay semantics are asserted via
//     Invoke and call counters instead.

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// requireInterrupt fatals unless err is an *InterruptError carrying exactly
// one interrupt, which it returns.
func requireInterrupt(t *testing.T, err error) types.Interrupt {
	t.Helper()
	var ierr *InterruptError
	if !errors.As(err, &ierr) {
		t.Fatalf("Invoke() error = %v (%T), want *InterruptError", err, err)
	}
	if len(ierr.Interrupts) != 1 {
		t.Fatalf("InterruptError.Interrupts = %+v, want exactly one", ierr.Interrupts)
	}
	return ierr.Interrupts[0]
}

// Mirrors test_pregel.py:1269 test_imp_task, with the AwaitAll form of
// collecting the futures (Task 6.1's TestPersistReplaySkipsReexecution covers
// the per-future Get form): concurrent mapper calls produce ordered results,
// the interrupt pauses the run, and the resume replays persisted task
// results without re-executing (mapper_calls stays 2).
func TestFunctionalImpTaskAwaitAll(t *testing.T) {
	var mapperCalls atomic.Int32
	mapper := NewTask[int, string]("mapper", func(_ runtime.Runtime, in int) (string, error) {
		mapperCalls.Add(1)
		time.Sleep(time.Duration(in) * 10 * time.Millisecond) // Python: time.sleep(input / 100)
		return strings.Repeat(strconv.Itoa(in), 2), nil
	}, TaskOpts{})
	e, err := NewEntrypoint[[]int, []string, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, in []int, _ any, _ bool) ([]string, error) {
			futs := make([]*Future[string], len(in))
			for i, v := range in {
				futs[i] = mapper.Call(ctx, v)
			}
			mapped, err := AwaitAll(ctx, futs...)
			if err != nil {
				return nil, err
			}
			answer, _ := graph.Interrupt(ctx, "question").(string)
			for i := range mapped {
				mapped[i] += answer
			}
			return mapped, nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	_, err = e.Invoke(ctx, []int{0, 1}, graph.Options{ThreadID: "1"})
	if intr := requireInterrupt(t, err); intr.Value != "question" {
		t.Fatalf("interrupt value = %v, want %q", intr.Value, "question")
	}
	if got := mapperCalls.Load(); got != 2 {
		t.Fatalf("mapper calls after first Invoke = %d, want 2", got)
	}

	out, err := e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "answer"})
	if err != nil {
		t.Fatalf("resumed Invoke() error = %v", err)
	}
	if want := []string{"00answer", "11answer"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("resumed Invoke() = %v, want %v", out, want)
	}
	if got := mapperCalls.Load(); got != 2 {
		t.Fatalf("mapper calls after resume = %d, want 2 (results replayed, tasks not re-executed)", got)
	}
}

// Mirrors test_pregel.py:1332 test_imp_nested: a task calling another task
// (mapper -> submapper), an interrupt/resume cycle, and a plain compiled
// StateGraph (add_a) invoked from inside the entrypoint for the final step.
// Both counters stay at 2 across the resume: nested results replay like root
// ones (a replayed mapper never re-runs, so its submapper is never re-called).
func TestFunctionalImpNested(t *testing.T) {
	var mapperCalls, submapperCalls atomic.Int32
	submapper := NewTask[int, string]("submapper", func(_ runtime.Runtime, in int) (string, error) {
		submapperCalls.Add(1)
		time.Sleep(time.Duration(in) * 10 * time.Millisecond)
		return strconv.Itoa(in), nil
	}, TaskOpts{})
	mapper := NewTask[int, string]("mapper", func(ctx runtime.Runtime, in int) (string, error) {
		mapperCalls.Add(1)
		sub, err := submapper.Call(ctx, in).Get(ctx)
		if err != nil {
			return "", err
		}
		time.Sleep(time.Duration(in) * 10 * time.Millisecond)
		return strings.Repeat(sub, 2), nil
	}, TaskOpts{})

	// add_a: a plain compiled StateGraph (Python: StateGraph(list[str]) with
	// one mynode appending "a"), invoked from inside the entrypoint.
	addAGraph, err := graph.NewStateGraph().
		AddChannel("items", channels.NewLastValue()).
		AddNode("mynode", func(_ runtime.Runtime, state map[string]any) (any, error) {
			items := state["items"].([]string)
			out := make([]string, len(items))
			for i, it := range items {
				out[i] = it + "a"
			}
			return map[string]any{"items": out}, nil
		}).
		SetEntryPoint("mynode").
		AddEdge("mynode", types.END).
		Compile()
	if err != nil {
		t.Fatalf("add_a Compile() error = %v", err)
	}

	e, err := NewEntrypoint[[]int, []string, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, in []int, _ any, _ bool) ([]string, error) {
			futs := make([]*Future[string], len(in))
			for i, v := range in {
				futs[i] = mapper.Call(ctx, v)
			}
			mapped := make([]string, len(in))
			for i, f := range futs {
				s, err := f.Get(ctx)
				if err != nil {
					return nil, err
				}
				mapped[i] = s
			}
			answer, _ := graph.Interrupt(ctx, "question").(string)
			final := make([]string, len(mapped))
			for i, m := range mapped {
				final[i] = m + answer
			}
			res, err := addAGraph.InvokeWithOptions(ctx, map[string]any{"items": final}, graph.Options{})
			if err != nil {
				return nil, err
			}
			return res.Values["items"].([]string), nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	_, err = e.Invoke(ctx, []int{0, 1}, graph.Options{ThreadID: "1"})
	if intr := requireInterrupt(t, err); intr.Value != "question" {
		t.Fatalf("interrupt value = %v, want %q", intr.Value, "question")
	}
	if got := mapperCalls.Load(); got != 2 {
		t.Fatalf("mapper calls after first Invoke = %d, want 2", got)
	}
	if got := submapperCalls.Load(); got != 2 {
		t.Fatalf("submapper calls after first Invoke = %d, want 2", got)
	}

	out, err := e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "answer"})
	if err != nil {
		t.Fatalf("resumed Invoke() error = %v", err)
	}
	if want := []string{"00answera", "11answera"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("resumed Invoke() = %v, want %v", out, want)
	}
	if got := mapperCalls.Load(); got != 2 {
		t.Fatalf("mapper calls after resume = %d, want 2 (replayed)", got)
	}
	if got := submapperCalls.Load(); got != 2 {
		t.Fatalf("submapper calls after resume = %d, want 2 (replayed via the replayed mapper)", got)
	}
}

// Mirrors test_pregel.py:4985 test_interrupt_functional: foo's future is
// created before the interrupt and awaited after it; the resume value feeds
// bar's input. foo may still be in flight at the pause (its late completion
// is dropped and it re-executes or replays on resume — same result either
// way, so no counter is asserted, matching Python).
func TestFunctionalInterrupt(t *testing.T) {
	foo := NewTask[map[string]any, map[string]any]("foo", func(_ runtime.Runtime, st map[string]any) (map[string]any, error) {
		return map[string]any{"a": st["a"].(string) + "foo"}, nil
	}, TaskOpts{})
	bar := NewTask[map[string]any, map[string]any]("bar", func(_ runtime.Runtime, st map[string]any) (map[string]any, error) {
		return map[string]any{"a": st["a"].(string) + "bar", "b": st["b"]}, nil
	}, TaskOpts{})
	e, err := NewEntrypoint[map[string]any, map[string]any, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, in map[string]any, _ any, _ bool) (map[string]any, error) {
			futFoo := foo.Call(ctx, in)
			value, _ := graph.Interrupt(ctx, "Provide value for bar:").(string)
			fooRes, err := futFoo.Get(ctx)
			if err != nil {
				return nil, err
			}
			barInput := map[string]any{"a": fooRes["a"], "b": value}
			return bar.Call(ctx, barInput).Get(ctx)
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	_, err = e.Invoke(ctx, map[string]any{"a": ""}, graph.Options{ThreadID: "1"})
	if intr := requireInterrupt(t, err); intr.Value != "Provide value for bar:" {
		t.Fatalf("interrupt value = %v, want %q", intr.Value, "Provide value for bar:")
	}

	out, err := e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "bar"})
	if err != nil {
		t.Fatalf("resumed Invoke() error = %v", err)
	}
	if want := map[string]any{"a": "foobar", "b": "bar"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("resumed Invoke() = %v, want %v", out, want)
	}
}

// Mirrors test_pregel.py:5019 test_interrupt_task_functional: the interrupt
// fires INSIDE the bar task (the task's ctx carries the entrypoint node's
// interrupt state; Get re-panics the GraphInterrupt and the run pauses).
// Second segment: the same task interrupts in two consecutive calls; both
// resumes must align in order — on the third run the first bar call's result
// replays from the checkpoint (bar does not re-execute) yet its already-
// consumed resume value must not be re-served to the second bar call
// (checkpoint.ReservedFnConsumed alignment).
func TestFunctionalInterruptTask(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var fooCalls, barCalls atomic.Int32
	foo := NewTask[map[string]any, map[string]any]("foo", func(_ runtime.Runtime, st map[string]any) (map[string]any, error) {
		fooCalls.Add(1)
		return map[string]any{"a": st["a"].(string) + "foo"}, nil
	}, TaskOpts{})
	bar := NewTask[map[string]any, map[string]any]("bar", func(ctx runtime.Runtime, st map[string]any) (map[string]any, error) {
		barCalls.Add(1)
		value, _ := graph.Interrupt(ctx, "Provide value for bar:").(string)
		return map[string]any{"a": st["a"].(string) + value}, nil
	}, TaskOpts{})

	// Segment 1 (Python thread "1"): a single interrupting bar call.
	e1, err := NewEntrypoint[map[string]any, map[string]any, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, in map[string]any, _ any, _ bool) (map[string]any, error) {
			fooRes, err := foo.Call(ctx, in).Get(ctx)
			if err != nil {
				return nil, err
			}
			return bar.Call(ctx, fooRes).Get(ctx)
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	_, err = e1.Invoke(ctx, map[string]any{"a": ""}, graph.Options{ThreadID: "1"})
	if intr := requireInterrupt(t, err); intr.Value != "Provide value for bar:" {
		t.Fatalf("segment 1 interrupt value = %v, want %q", intr.Value, "Provide value for bar:")
	}
	out, err := e1.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "bar"})
	if err != nil {
		t.Fatalf("segment 1 resumed Invoke() error = %v", err)
	}
	if want := map[string]any{"a": "foobar"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("segment 1 resumed Invoke() = %v, want %v", out, want)
	}

	// Segment 2 (Python thread "2"): interrupt the same task twice.
	fooCalls.Store(0)
	barCalls.Store(0)
	e2, err := NewEntrypoint[map[string]any, map[string]any, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, in map[string]any, _ any, _ bool) (map[string]any, error) {
			fooRes, err := foo.Call(ctx, in).Get(ctx)
			if err != nil {
				return nil, err
			}
			barRes, err := bar.Call(ctx, fooRes).Get(ctx)
			if err != nil {
				return nil, err
			}
			return bar.Call(ctx, barRes).Get(ctx)
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	_, err = e2.Invoke(ctx, map[string]any{"a": ""}, graph.Options{ThreadID: "2"})
	if intr := requireInterrupt(t, err); intr.Value != "Provide value for bar:" {
		t.Fatalf("segment 2 first interrupt value = %v, want %q", intr.Value, "Provide value for bar:")
	}
	_, err = e2.Invoke(ctx, nil, graph.Options{ThreadID: "2", Resume: "bar"})
	if intr := requireInterrupt(t, err); intr.Value != "Provide value for bar:" {
		t.Fatalf("segment 2 second interrupt value = %v, want %q", intr.Value, "Provide value for bar:")
	}
	out, err = e2.Invoke(ctx, nil, graph.Options{ThreadID: "2", Resume: "baz"})
	if err != nil {
		t.Fatalf("segment 2 third Invoke() error = %v", err)
	}
	if want := map[string]any{"a": "foobarbaz"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("segment 2 third Invoke() = %v, want %v (resume values matched in order)", out, want)
	}
	// Execution count per run: run 1 runs bar's first call (interrupts);
	// run 2 re-runs the first call (consumes "bar", completes) and runs the
	// second call (interrupts); run 3 REPLAYS the first call's persisted
	// result and re-runs only the second — 4 total. A 5th execution would
	// mean the completed call failed to replay; a wrong final value would
	// mean its consumed resume value leaked to the second call.
	if got := fooCalls.Load(); got != 1 {
		t.Fatalf("segment 2 foo calls = %d, want 1 (replayed on both resumes)", got)
	}
	if got := barCalls.Load(); got != 4 {
		t.Fatalf("segment 2 bar calls = %d, want 4 (only completed executions replay)", got)
	}
}

// Mirrors test_pregel.py:5710 test_multiple_interrupts_functional: a loop of
// task call + interrupt; each resume value must feed the interrupt of its
// own iteration (sequential resume alignment), while the double calls of
// earlier iterations replay without re-executing (counter stays 3).
func TestFunctionalMultipleInterrupts(t *testing.T) {
	var counter atomic.Int32
	double := NewTask[int, int]("double", func(_ runtime.Runtime, x int) (int, error) {
		counter.Add(1)
		return 2 * x, nil
	}, TaskOpts{})
	e, err := NewEntrypoint[map[string]any, map[string]any, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, _ map[string]any, _ any, _ bool) (map[string]any, error) {
			var values []any
			for _, idx := range []int{1, 2, 3} {
				v, err := double.Call(ctx, idx).Get(ctx)
				if err != nil {
					return nil, err
				}
				values = append(values, v, graph.Interrupt(ctx, map[string]any{"a": "boo"}))
			}
			return map[string]any{"values": values}, nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	_, err = e.Invoke(ctx, map[string]any{}, graph.Options{ThreadID: "1"})
	requireInterrupt(t, err)
	_, err = e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "a"})
	requireInterrupt(t, err)
	_, err = e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "b"})
	requireInterrupt(t, err)
	out, err := e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "c"})
	if err != nil {
		t.Fatalf("final Invoke() error = %v", err)
	}
	want := map[string]any{"values": []any{2, "a", 4, "b", 6, "c"}}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("final Invoke() = %v, want %v", out, want)
	}
	if got := counter.Load(); got != 3 {
		t.Fatalf("double calls = %d, want 3 (earlier iterations replayed on each resume)", got)
	}
}

// Mirrors test_pregel.py:5745 test_multiple_interrupts_functional_cache:
// with a cache policy on double, repeated inputs are served from the cache —
// within one thread (mixed with checkpoint replay), across a brand-new
// thread (counter stays 3), and not at all after ClearCache (counter 6).
func TestFunctionalMultipleInterruptsCache(t *testing.T) {
	cache := checkpoint.NewInMemoryCache()
	var counter atomic.Int32
	double := NewTask[int, int]("double", func(_ runtime.Runtime, x int) (int, error) {
		counter.Add(1)
		return 2 * x, nil
	}, TaskOpts{Cache: &graph.CachePolicy{}})
	e, err := NewEntrypoint[map[string]any, map[string]any, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver(), Cache: cache},
		func(ctx runtime.Runtime, _ map[string]any, _ any, _ bool) (map[string]any, error) {
			var values []any
			for _, idx := range []int{1, 1, 2, 2, 3, 3} {
				v, err := double.Call(ctx, idx).Get(ctx)
				if err != nil {
					return nil, err
				}
				values = append(values, v, graph.Interrupt(ctx, map[string]any{"a": "boo"}))
			}
			return map[string]any{"values": values}, nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	want := map[string]any{"values": []any{2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"}}
	// run drives a full six-interrupt conversation on thread and returns the
	// final result.
	run := func(thread string) map[string]any {
		t.Helper()
		_, err := e.Invoke(ctx, map[string]any{}, graph.Options{ThreadID: thread})
		requireInterrupt(t, err)
		resumes := []string{"a", "b", "c", "d", "e"}
		for _, r := range resumes {
			_, err = e.Invoke(ctx, nil, graph.Options{ThreadID: thread, Resume: r})
			requireInterrupt(t, err)
		}
		out, err := e.Invoke(ctx, nil, graph.Options{ThreadID: thread, Resume: "f"})
		if err != nil {
			t.Fatalf("thread %q final Invoke() error = %v", thread, err)
		}
		return out
	}

	if out := run("1"); !reflect.DeepEqual(out, want) {
		t.Fatalf("thread 1 result = %v, want %v", out, want)
	}
	if got := counter.Load(); got != 3 {
		t.Fatalf("double calls after thread 1 = %d, want 3 (repeated inputs served from cache)", got)
	}

	// A brand-new thread re-runs the whole conversation; every double call
	// is a cache hit, so the counter does not move.
	if out := run("2"); !reflect.DeepEqual(out, want) {
		t.Fatalf("thread 2 result = %v, want %v", out, want)
	}
	if got := counter.Load(); got != 3 {
		t.Fatalf("double calls after thread 2 = %d, want 3 (cache shared across threads)", got)
	}

	if err := double.ClearCache(ctx, cache); err != nil {
		t.Fatalf("ClearCache() error = %v", err)
	}
	if out := run("3"); !reflect.DeepEqual(out, want) {
		t.Fatalf("thread 3 result = %v, want %v", out, want)
	}
	if got := counter.Load(); got != 6 {
		t.Fatalf("double calls after ClearCache + thread 3 = %d, want 6 (recomputed)", got)
	}
}

// Mirrors test_pregel.py:5868 test_multiple_tasks_before_interrupt_resume
// (its sibling test_task_before_interrupt_resume, test_pregel.py:5818, is
// covered by Task 6.4's TestPersistChainedPauseRestamp): several tasks run
// before the interrupt-producing ask task; on resume step_a/step_b replay
// (counters stay 1) and the resume value reaches ask's re-executed Interrupt.
func TestFunctionalMultipleTasksBeforeInterruptResume(t *testing.T) {
	var aCalls, bCalls atomic.Int32
	stepA := NewTask[int, int]("step_a", func(_ runtime.Runtime, x int) (int, error) {
		aCalls.Add(1)
		return x + 1, nil
	}, TaskOpts{})
	stepB := NewTask[int, int]("step_b", func(_ runtime.Runtime, x int) (int, error) {
		bCalls.Add(1)
		return x * 2, nil
	}, TaskOpts{})
	ask := NewTask[string, string]("ask", func(ctx runtime.Runtime, q string) (string, error) {
		v, _ := graph.Interrupt(ctx, q).(string)
		return v, nil
	}, TaskOpts{})
	e, err := NewEntrypoint[map[string]any, map[string]any, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, in map[string]any, _ any, _ bool) (map[string]any, error) {
			a, err := stepA.Call(ctx, in["x"].(int)).Get(ctx)
			if err != nil {
				return nil, err
			}
			b, err := stepB.Call(ctx, a).Get(ctx)
			if err != nil {
				return nil, err
			}
			answer, err := ask.Call(ctx, fmt.Sprintf("Result so far is %d. What next?", b)).Get(ctx)
			if err != nil {
				return nil, err
			}
			return map[string]any{"computed": b, "answer": answer}, nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	_, err = e.Invoke(ctx, map[string]any{"x": 5}, graph.Options{ThreadID: "1"})
	if intr := requireInterrupt(t, err); intr.Value != "Result so far is 12. What next?" {
		t.Fatalf("interrupt value = %v, want %q", intr.Value, "Result so far is 12. What next?")
	}

	out, err := e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "continue"})
	if err != nil {
		t.Fatalf("resumed Invoke() error = %v", err)
	}
	if want := map[string]any{"computed": 12, "answer": "continue"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("resumed Invoke() = %v, want %v", out, want)
	}
	if got := aCalls.Load(); got != 1 {
		t.Fatalf("step_a calls = %d, want 1 (replayed on resume)", got)
	}
	if got := bCalls.Load(); got != 1 {
		t.Fatalf("step_b calls = %d, want 1 (replayed on resume)", got)
	}
}

// Mirrors test_pregel.py:6830 test_named_tasks_functional: explicitly named
// tasks — two names wrapping the SAME function (custom_foo/other_foo, the
// Python class-method tasks) and distinct names per closure (custom_bar,
// baz, custom_baz, qux, the Python decorator/partial/callable forms) — chain
// into the same result string. Python asserts per-task stream updates keyed
// by task name; Go tasks do not stream (divergence, see file header), so the
// name-level assertion here is white-box: two tasks wrapping one function
// under different names produce different deterministic task IDs.
func TestFunctionalNamedTasks(t *testing.T) {
	var fooCalls atomic.Int32
	fooFn := func(_ runtime.Runtime, v string) (string, error) {
		fooCalls.Add(1)
		return v + "foo", nil
	}
	foo := NewTask[string, string]("custom_foo", fooFn, TaskOpts{})
	otherFoo := NewTask[string, string]("other_foo", fooFn, TaskOpts{})
	bar := NewTask[string, string]("custom_bar", func(_ runtime.Runtime, v string) (string, error) {
		return v + "|bar", nil
	}, TaskOpts{})
	bazTask := NewTask[string, string]("baz", func(_ runtime.Runtime, v string) (string, error) {
		return v + "|baz", nil
	}, TaskOpts{})
	customBazTask := NewTask[string, string]("custom_baz", func(_ runtime.Runtime, v string) (string, error) {
		return v + "|custom_baz", nil
	}, TaskOpts{})
	quxTask := NewTask[string, string]("qux", func(_ runtime.Runtime, v string) (string, error) {
		return v + "|qux", nil
	}, TaskOpts{})

	e, err := NewEntrypoint[string, string, any](EntrypointOpts{},
		func(ctx runtime.Runtime, in string, _ any, _ bool) (string, error) {
			fooRes, err := foo.Call(ctx, in).Get(ctx)
			if err != nil {
				return "", err
			}
			if _, err := otherFoo.Call(ctx, in).Get(ctx); err != nil { // result intentionally discarded, as in Python
				return "", err
			}
			barRes, err := bar.Call(ctx, fooRes).Get(ctx)
			if err != nil {
				return "", err
			}
			bazRes, err := bazTask.Call(ctx, barRes).Get(ctx)
			if err != nil {
				return "", err
			}
			customBazRes, err := customBazTask.Call(ctx, bazRes).Get(ctx)
			if err != nil {
				return "", err
			}
			return quxTask.Call(ctx, customBazRes).Get(ctx)
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	out, err := e.Invoke(context.Background(), "", graph.Options{})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if want := "foo|bar|baz|custom_baz|qux"; out != want {
		t.Fatalf("Invoke() = %q, want %q", out, want)
	}
	if got := fooCalls.Load(); got != 2 {
		t.Fatalf("shared foo function calls = %d, want 2 (custom_foo and other_foo each invoked it)", got)
	}

	// White-box: the name is what identifies a task in deterministic task
	// IDs — one function under two names yields two distinct IDs, while the
	// same name recomputes the same ID.
	idFoo := graph.FnTaskID("cp", "", 0, "custom_foo", "", 0)
	idOther := graph.FnTaskID("cp", "", 0, "other_foo", "", 0)
	if idFoo == idOther {
		t.Fatalf("FnTaskID(custom_foo) == FnTaskID(other_foo) = %q, want distinct IDs for one function under two names", idFoo)
	}
	if again := graph.FnTaskID("cp", "", 0, "custom_foo", "", 0); again != idFoo {
		t.Fatalf("FnTaskID(custom_foo) recomputed = %q, want stable %q", again, idFoo)
	}
}

// Mirrors test_pregel.py:6611 test_multiple_subgraphs_mixed_state_graph:
// entrypoint "subgraphs" invoked from inside a parent StateGraph's nodes.
// (Documented divergence: Python also allows calling a bare @task directly
// inside a StateGraph node via Pregel config injection; Go has no
// equivalent — a node must go through Entrypoint.Invoke, as here.)
func TestFunctionalSubgraphsMixedStateGraph(t *testing.T) {
	add, err := NewEntrypoint[[]int, int, any](EntrypointOpts{},
		func(_ runtime.Runtime, in []int, _ any, _ bool) (int, error) {
			return in[0] + in[1], nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}
	multiplyTask := NewTask[[]int, int]("multiply_task", func(_ runtime.Runtime, in []int) (int, error) {
		return in[0] * in[1], nil
	}, TaskOpts{})
	multiply, err := NewEntrypoint[[]int, int, any](EntrypointOpts{},
		func(ctx runtime.Runtime, in []int, _ any, _ bool) (int, error) {
			return multiplyTask.Call(ctx, in).Get(ctx)
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()

	// call_same_subgraph: two chained add invocations inside one node; the
	// intermediate results must match Python's (5, then 15).
	var firstAdd int
	parentSame, err := graph.NewStateGraph().
		AddNode("call_same_subgraph", func(nctx runtime.Runtime, state map[string]any) (any, error) {
			a, b := state["a"].(int), state["b"].(int)
			result, err := add.Invoke(nctx, []int{a, b}, graph.Options{})
			if err != nil {
				return nil, err
			}
			firstAdd = result
			another, err := add.Invoke(nctx, []int{result, 10}, graph.Options{})
			if err != nil {
				return nil, err
			}
			return map[string]any{"result": another}, nil
		}).
		SetEntryPoint("call_same_subgraph").
		AddEdge("call_same_subgraph", types.END).
		Compile(graph.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("parentSame Compile() error = %v", err)
	}
	res, err := parentSame.InvokeWithOptions(ctx, map[string]any{"a": 2, "b": 3}, graph.Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("parentSame Invoke() error = %v", err)
	}
	if firstAdd != 5 {
		t.Fatalf("first add.Invoke inside node = %d, want 5 (2+3)", firstAdd)
	}
	if got := res.Values["result"]; got != 15 {
		t.Fatalf("parentSame result = %v, want 15 (5+10)", got)
	}

	// call_multiple_subgraphs: add and multiply (the latter wrapping a task)
	// invoked inside one node.
	parentMulti, err := graph.NewStateGraph().
		AddNode("call_multiple_subgraphs", func(nctx runtime.Runtime, state map[string]any) (any, error) {
			a, b := state["a"].(int), state["b"].(int)
			addRes, err := add.Invoke(nctx, []int{a, b}, graph.Options{})
			if err != nil {
				return nil, err
			}
			mulRes, err := multiply.Invoke(nctx, []int{a, b}, graph.Options{})
			if err != nil {
				return nil, err
			}
			return map[string]any{"add_result": addRes, "multiply_result": mulRes}, nil
		}).
		SetEntryPoint("call_multiple_subgraphs").
		AddEdge("call_multiple_subgraphs", types.END).
		Compile(graph.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("parentMulti Compile() error = %v", err)
	}
	res, err = parentMulti.InvokeWithOptions(ctx, map[string]any{"a": 2, "b": 3}, graph.Options{ThreadID: "2"})
	if err != nil {
		t.Fatalf("parentMulti Invoke() error = %v", err)
	}
	if got := res.Values["add_result"]; got != 5 {
		t.Fatalf("parentMulti add_result = %v, want 5", got)
	}
	if got := res.Values["multiply_result"]; got != 6 {
		t.Fatalf("parentMulti multiply_result = %v, want 6", got)
	}
}

// Mirrors test_pregel.py:8288 test_imp_exception: the entrypoint CATCHES the
// task's error and continues; a second invocation on the same thread is a
// new turn — no replay of the previous turn's results, so every task
// re-executes (counters double). Python's second invocation asserts stream
// chunks; Go asserts via Invoke + counters (see file header).
func TestFunctionalImpException(t *testing.T) {
	var myTaskCalls, excCalls atomic.Int32
	var caught []string
	myTask := NewTask[int, int]("my_task", func(_ runtime.Runtime, n int) (int, error) {
		myTaskCalls.Add(1)
		return n * 2, nil
	}, TaskOpts{})
	taskWithException := NewTask[int, int]("task_with_exception", func(_ runtime.Runtime, _ int) (int, error) {
		excCalls.Add(1)
		return 0, errors.New("This is a test exception")
	}, TaskOpts{})
	e, err := NewEntrypoint[int, string, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, n int, _ any, _ bool) (string, error) {
			if _, err := myTask.Call(ctx, n).Get(ctx); err != nil {
				return "", err
			}
			if _, err := taskWithException.Call(ctx, n).Get(ctx); err != nil {
				caught = append(caught, err.Error()) // caught: the workflow continues
			}
			if _, err := myTask.Call(ctx, n).Get(ctx); err != nil {
				return "", err
			}
			return "done", nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	out, err := e.Invoke(ctx, 1, graph.Options{ThreadID: "1"})
	if err != nil || out != "done" {
		t.Fatalf("first Invoke() = %q, %v; want %q, nil", out, err, "done")
	}
	if got := myTaskCalls.Load(); got != 2 {
		t.Fatalf("my_task calls after first Invoke = %d, want 2", got)
	}
	if got := excCalls.Load(); got != 1 {
		t.Fatalf("task_with_exception calls after first Invoke = %d, want 1", got)
	}
	if len(caught) != 1 || !strings.Contains(caught[0], "This is a test exception") {
		t.Fatalf("caught errors = %v, want exactly the task's exception", caught)
	}

	// Second turn on the same thread: a completed previous turn has no
	// replay gate, so all tasks re-execute and the error is caught again.
	out, err = e.Invoke(ctx, 1, graph.Options{ThreadID: "1"})
	if err != nil || out != "done" {
		t.Fatalf("second Invoke() = %q, %v; want %q, nil", out, err, "done")
	}
	if got := myTaskCalls.Load(); got != 4 {
		t.Fatalf("my_task calls after second Invoke = %d, want 4 (new turn re-executes)", got)
	}
	if got := excCalls.Load(); got != 2 {
		t.Fatalf("task_with_exception calls after second Invoke = %d, want 2 (new turn re-executes)", got)
	}
	if len(caught) != 2 {
		t.Fatalf("caught errors = %v, want the exception caught once per turn", caught)
	}
}

// Go-side supplement (no Python counterpart): across an interrupt -> resume
// cycle the __previous__ value injected at the start of the interrupted
// invocation must survive — on resume the graph ignores fresh input and the
// re-executed entrypoint has to read previous from the PAUSE checkpoint, not
// lose it. Pinned here because every resume run re-executes the function.
func TestFunctionalPreviousSurvivesInterruptResume(t *testing.T) {
	type observation struct {
		prev    []string
		hasPrev bool
	}
	var observed []observation
	e, err := NewEntrypoint[string, []string, []string](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, in string, prev []string, hasPrev bool) ([]string, error) {
			observed = append(observed, observation{append([]string(nil), prev...), hasPrev})
			_ = graph.Interrupt(ctx, "continue?")
			return append(append([]string(nil), prev...), in), nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	// First turn: no previous; interrupt, resume, completes with ["a"].
	_, err = e.Invoke(ctx, "a", graph.Options{ThreadID: "1"})
	requireInterrupt(t, err)
	out, err := e.Invoke(ctx, "", graph.Options{ThreadID: "1", Resume: "go"})
	if err != nil {
		t.Fatalf("first resume Invoke() error = %v", err)
	}
	if want := []string{"a"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("first turn result = %v, want %v", out, want)
	}

	// Second turn: previous ["a"] is injected; the run interrupts and the
	// resumed re-execution must STILL observe ["a"] (hasPrev=true).
	_, err = e.Invoke(ctx, "b", graph.Options{ThreadID: "1"})
	requireInterrupt(t, err)
	out, err = e.Invoke(ctx, "", graph.Options{ThreadID: "1", Resume: "go"})
	if err != nil {
		t.Fatalf("second resume Invoke() error = %v", err)
	}
	if want := []string{"a", "b"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("second turn result = %v, want %v (previous survived the resume)", out, want)
	}
	wantObserved := []observation{
		{nil, false},          // turn 1, fresh run
		{nil, false},          // turn 1, resumed re-execution
		{[]string{"a"}, true}, // turn 2, fresh run: previous injected
		{[]string{"a"}, true}, // turn 2, resumed re-execution: previous restored from the pause checkpoint
	}
	if !reflect.DeepEqual(observed, wantObserved) {
		t.Fatalf("observed (prev, hasPrev) per execution = %+v, want %+v", observed, wantObserved)
	}
}

// Go-side supplement (no Python counterpart): an interrupt raised while
// Streaming surfaces as an updates chunk carrying the "__interrupt__" key,
// passed through unchanged (no node key, no reserved-channel filtering
// applied to it), and the resumed Stream yields the rewritten completion
// chunk {"entrypoint": <value>}.
func TestFunctionalStreamInterruptPassthrough(t *testing.T) {
	e, err := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, _ any, _ any, _ bool) (string, error) {
			v, _ := graph.Interrupt(ctx, "Provide value").(string)
			return "got " + v, nil
		})
	if err != nil {
		t.Fatalf("NewEntrypoint: %v", err)
	}

	ctx := context.Background()
	var chunks []graph.StreamChunk
	for chunk, err := range e.Stream(ctx, "in", graph.Options{ThreadID: "1"}) {
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 1 {
		t.Fatalf("first Stream produced %d chunks, want exactly the interrupt chunk: %+v", len(chunks), chunks)
	}
	payload, ok := chunks[0].Payload.(map[string]any)
	if !ok {
		t.Fatalf("interrupt chunk payload = %T %+v, want map[string]any", chunks[0].Payload, chunks[0].Payload)
	}
	rawInterrupts, ok := payload["__interrupt__"]
	if !ok {
		t.Fatalf("interrupt chunk payload = %+v, want an %q key", payload, "__interrupt__")
	}
	interrupts, ok := rawInterrupts.([]types.Interrupt)
	if !ok || len(interrupts) != 1 || interrupts[0].Value != "Provide value" {
		t.Fatalf("__interrupt__ payload = %+v (%T), want one types.Interrupt with value %q", rawInterrupts, rawInterrupts, "Provide value")
	}
	if _, hasNodeKey := payload["entrypoint"]; hasNodeKey {
		t.Fatalf("interrupt chunk payload = %+v, must not carry the node key (passthrough, not rewritten)", payload)
	}

	chunks = chunks[:0]
	for chunk, err := range e.Stream(ctx, nil, graph.Options{ThreadID: "1", Resume: "bar"}) {
		if err != nil {
			t.Fatalf("resumed Stream() error = %v", err)
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 1 {
		t.Fatalf("resumed Stream produced %d chunks, want 1: %+v", len(chunks), chunks)
	}
	if want := map[string]any{"entrypoint": "got bar"}; !reflect.DeepEqual(chunks[0].Payload, want) {
		t.Fatalf("resumed Stream chunk payload = %+v, want %+v", chunks[0].Payload, want)
	}
}
