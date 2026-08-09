package fn

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

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/graph"
)

// fnWrites returns the pending writes of tup on the given channel.
func fnWrites(tup *checkpoint.Tuple, channel string) []checkpoint.Write {
	var out []checkpoint.Write
	for _, w := range tup.PendingWrites {
		if w.Channel == channel {
			out = append(out, w)
		}
	}
	return out
}

// Mirrors the task body of test_pregel.py:1269 test_imp_task: task results
// persist to the pause checkpoint and replay on resume — the tasks do NOT
// re-execute (the call counter stays put) and their results feed the
// resumed run.
func TestPersistReplaySkipsReexecution(t *testing.T) {
	var calls atomic.Int32
	mapper := NewTask[int, string]("mapper", func(_ runtime.Runtime, in int) (string, error) {
		calls.Add(1)
		return strings.Repeat(strconv.Itoa(in), 2), nil
	}, TaskOpts{})
	e := NewEntrypoint[[]int, []string, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, in []int, _ any, _ bool) ([]string, error) {
			futs := make([]*Future[string], len(in))
			for i, v := range in {
				futs[i] = mapper.Call(ctx, v)
			}
			outs := make([]string, len(in))
			for i, f := range futs {
				s, err := f.Get(ctx)
				if err != nil {
					return nil, err
				}
				outs[i] = s
			}
			answer, _ := graph.Interrupt(ctx, "question").(string)
			for i := range outs {
				outs[i] += answer
			}
			return outs, nil
		})

	ctx := context.Background()
	_, err := e.Invoke(ctx, []int{0, 1}, graph.Options{ThreadID: "1"})
	var ierr *InterruptError
	if !errors.As(err, &ierr) {
		t.Fatalf("first Invoke() error = %v (%T), want *InterruptError", err, err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls after first Invoke = %d, want 2", got)
	}

	out, err := e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "answer"})
	if err != nil {
		t.Fatalf("resumed Invoke() error = %v", err)
	}
	if want := []string{"00answer", "11answer"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("resumed Invoke() = %v, want %v", out, want)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls after resume = %d, want 2 (results replayed, tasks not re-executed)", got)
	}
}

// Mirrors test_pregel.py:5486 test_falsy_return_from_task: a false result
// round-trips through the checkpoint — the result state is carried by the
// channel (__return__ vs __error__), not by truthiness, so replay does not
// misread it as a zero value / miss.
func TestPersistReplayFalsyResult(t *testing.T) {
	var calls atomic.Int32
	falsy := NewTask[any, bool]("falsy", func(_ runtime.Runtime, _ any) (bool, error) {
		calls.Add(1)
		return false, nil
	}, TaskOpts{})
	e := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, _, _ any, _ bool) (string, error) {
			v, err := falsy.Call(ctx, nil).Get(ctx)
			if err != nil {
				return "", err
			}
			_ = graph.Interrupt(ctx, "q")
			return strconv.FormatBool(v), nil
		})

	ctx := context.Background()
	_, err := e.Invoke(ctx, "in", graph.Options{ThreadID: "1"})
	var ierr *InterruptError
	if !errors.As(err, &ierr) {
		t.Fatalf("first Invoke() error = %v (%T), want *InterruptError", err, err)
	}

	out, err := e.Invoke(ctx, "ignored", graph.Options{ThreadID: "1", Resume: "ok"})
	if err != nil {
		t.Fatalf("resumed Invoke() error = %v", err)
	}
	if out != "false" {
		t.Fatalf("resumed Invoke() = %q, want %q (the replayed false, not a zero-value misread)", out, "false")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls after resume = %d, want 1 (falsy result replayed, task not re-executed)", got)
	}
}

// Task failure persists as an __error__ write against the pre-run input
// checkpoint; the next Invoke on the same thread replays and re-raises it
// without re-executing the task (gate 2, Python parity _runner.py:751-754).
// Gate 2 does not look at the input, so even a brand-new input re-raises —
// a documented divergence from Python (doc.go list item 11): Python can
// escape the error state with new input, Go cannot; a new ThreadID is the
// way out.
func TestPersistErrorRethrow(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var calls atomic.Int32
	bad := NewTask[any, string]("bad", func(_ runtime.Runtime, _ any) (string, error) {
		calls.Add(1)
		return "", errors.New("boom")
	}, TaskOpts{})
	e := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, _, _ any, _ bool) (string, error) {
			return bad.Call(ctx, nil).Get(ctx)
		})

	ctx := context.Background()
	_, err := e.Invoke(ctx, "in1", graph.Options{ThreadID: "t"})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("first Invoke() error = %v, want one containing %q", err, "boom")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls after first Invoke = %d, want 1", got)
	}

	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %v, %v; want the thread's latest tuple", tup, err)
	}
	errWrites := fnWrites(tup, checkpoint.ReservedError)
	if len(errWrites) != 1 || errWrites[0].Value != "boom" {
		t.Fatalf("latest tuple __error__ writes = %+v, want exactly one with Value %q", errWrites, "boom")
	}

	// Same thread, same input: replay re-raises, the task does not re-run.
	_, err = e.Invoke(ctx, "in1", graph.Options{ThreadID: "t"})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("second Invoke() error = %v, want replayed %q", err, "boom")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls after second Invoke = %d, want 1 (replayed, not re-executed)", got)
	}

	// Same thread, brand-new input: gate 2 ignores input — still re-raises
	// (documented divergence, doc.go list item 11).
	_, err = e.Invoke(ctx, "completely-new-input", graph.Options{ThreadID: "t"})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("third Invoke() error = %v, want replayed %q even with new input", err, "boom")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls after third Invoke = %d, want 1", got)
	}

	// A fresh thread is unaffected by the poisoned one.
	good := NewTask[any, string]("bad", func(_ runtime.Runtime, _ any) (string, error) {
		return "fine", nil
	}, TaskOpts{})
	e2 := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, _, _ any, _ bool) (string, error) {
			return good.Call(ctx, nil).Get(ctx)
		})
	out, err := e2.Invoke(ctx, "in", graph.Options{ThreadID: "t2"})
	if err != nil || out != "fine" {
		t.Fatalf("fresh-thread Invoke() = %q, %v; want %q, nil", out, err, "fine")
	}
}

// Mirrors test_pregel.py:5818 test_task_before_interrupt_resume: across a
// pause -> resume -> pause chain the whole result table (replay hits
// included) is re-appended to each new pause checkpoint, re-stamped with
// that checkpoint's ID/step (F4 re-stamping), and resume values match the
// sequential interrupts in order (Task 2's RESUME-write mechanism).
func TestPersistChainedPauseRestamp(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var setupCalls atomic.Int32
	setup := NewTask[any, int]("setup", func(_ runtime.Runtime, _ any) (int, error) {
		setupCalls.Add(1)
		return 2, nil
	}, TaskOpts{})
	e := NewEntrypoint[any, []string, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, _, _ any, _ bool) ([]string, error) {
			n, err := setup.Call(ctx, nil).Get(ctx)
			if err != nil {
				return nil, err
			}
			answers := make([]string, 0, n)
			for i := 0; i < n; i++ {
				a, _ := graph.Interrupt(ctx, fmt.Sprintf("q%d", i)).(string)
				answers = append(answers, a)
			}
			return answers, nil
		})

	ctx := context.Background()
	_, err := e.Invoke(ctx, "in", graph.Options{ThreadID: "1"})
	var ierr *InterruptError
	if !errors.As(err, &ierr) {
		t.Fatalf("first Invoke() error = %v (%T), want *InterruptError", err, err)
	}

	_, err = e.Invoke(ctx, "ignored", graph.Options{ThreadID: "1", Resume: "answer1"})
	if !errors.As(err, &ierr) {
		t.Fatalf("second Invoke() error = %v (%T), want *InterruptError", err, err)
	}
	if len(ierr.Interrupts) != 1 || ierr.Interrupts[0].Value != "q1" {
		t.Fatalf("second Invoke() interrupts = %+v, want one interrupt with value %q", ierr.Interrupts, "q1")
	}

	out, err := e.Invoke(ctx, "ignored", graph.Options{ThreadID: "1", Resume: "answer2"})
	if err != nil {
		t.Fatalf("third Invoke() error = %v", err)
	}
	if want := []string{"answer1", "answer2"}; !reflect.DeepEqual(out, want) {
		t.Fatalf("third Invoke() = %v, want %v (resume values matched in order)", out, want)
	}
	if got := setupCalls.Load(); got != 1 {
		t.Fatalf("setup calls = %d, want 1 (replayed across both resumes)", got)
	}

	// Re-stamping: both pause checkpoints (B = first pause, C = second)
	// carry the setup result write, stamped with distinct task IDs, each
	// equal to FnTaskID recomputed from that tuple's own ID/step.
	tups, err := saver.List(ctx, checkpoint.Config{ThreadID: "1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	var ids []string
	for i := range tups {
		tup := &tups[i]
		if len(tup.Checkpoint.Next) == 0 {
			continue // not a pause checkpoint
		}
		writes := fnWrites(tup, checkpoint.ReservedReturn)
		if len(writes) != 1 {
			t.Fatalf("pause checkpoint %s fn writes = %+v, want exactly one __return__ write", tup.Checkpoint.ID, writes)
		}
		want := graph.FnTaskID(tup.Checkpoint.ID, tup.Config.CheckpointNS, tup.Metadata.Step, "setup", "", 0)
		if writes[0].TaskID != want {
			t.Fatalf("pause checkpoint %s fn write taskID = %q, want re-stamped %q", tup.Checkpoint.ID, writes[0].TaskID, want)
		}
		ids = append(ids, writes[0].TaskID)
	}
	if len(ids) != 2 {
		t.Fatalf("found %d pause checkpoints with fn writes, want 2", len(ids))
	}
	if ids[0] == ids[1] {
		t.Fatalf("pause checkpoints share fn write taskID %q, want distinct re-stamped IDs", ids[0])
	}
}

// Spec closure point 1 (the fn layer owns the dispatcher): results of tasks
// completed before the interrupt survive the pause in pending writes; tasks
// still in flight at the pause are abandoned by the run cancel/seal and
// their late completion is dropped — on resume the completed task replays
// and the abandoned one re-executes.
func TestPersistInterruptKeepsBufferedResults(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	var aCalls, bCalls atomic.Int32
	bStarted := make(chan struct{})
	taskA := NewTask[any, string]("a", func(_ runtime.Runtime, _ any) (string, error) {
		aCalls.Add(1)
		return "A", nil
	}, TaskOpts{})
	taskB := NewTask[any, string]("b", func(_ runtime.Runtime, _ any) (string, error) {
		if bCalls.Add(1) == 1 {
			close(bStarted) // first-run start signal; the goroutine is started but in flight
		}
		time.Sleep(200 * time.Millisecond)
		return "B", nil
	}, TaskOpts{})
	e := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, _, _ any, _ bool) (string, error) {
			futA := taskA.Call(ctx, nil)
			vA, err := futA.Get(ctx) // A completes and is buffered
			if err != nil {
				return "", err
			}
			futB := taskB.Call(ctx, nil) // B in flight, not yet awaited
			_ = graph.Interrupt(ctx, "q")
			vB, err := futB.Get(ctx)
			if err != nil {
				return "", err
			}
			return vA + vB, nil
		})

	ctx := context.Background()
	_, err := e.Invoke(ctx, "in", graph.Options{ThreadID: "1"})
	var ierr *InterruptError
	if !errors.As(err, &ierr) {
		t.Fatalf("first Invoke() error = %v (%T), want *InterruptError", err, err)
	}
	if got := aCalls.Load(); got != 1 {
		t.Fatalf("A calls after first Invoke = %d, want 1", got)
	}
	// B's goroutine is scheduled independently of the pause; wait for its
	// start before counting it as in flight.
	select {
	case <-bStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("B never started during the first run")
	}
	if got := bCalls.Load(); got != 1 {
		t.Fatalf("B calls after first Invoke = %d, want 1 (in flight at the pause)", got)
	}

	// A's completed result is persisted; B's late completion is dropped by
	// the seal and never lands.
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "1"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = %v, %v; want the pause tuple", tup, err)
	}
	retWrites := fnWrites(tup, checkpoint.ReservedReturn)
	if len(retWrites) != 1 || retWrites[0].Value != "A" {
		t.Fatalf("pause tuple __return__ writes = %+v, want exactly A's result", retWrites)
	}

	out, err := e.Invoke(ctx, "ignored", graph.Options{ThreadID: "1", Resume: "ok"})
	if err != nil {
		t.Fatalf("resumed Invoke() error = %v", err)
	}
	if out != "AB" {
		t.Fatalf("resumed Invoke() = %q, want %q", out, "AB")
	}
	if got := aCalls.Load(); got != 1 {
		t.Fatalf("A calls after resume = %d, want 1 (replayed from the buffered write)", got)
	}
	if got := bCalls.Load(); got != 2 {
		t.Fatalf("B calls after resume = %d, want 2 (abandoned in-flight run + re-execution)", got)
	}
}

// With no checkpointer nothing persists: an interrupt surfaces as an
// InterruptError but is not resumable (the assertion stops at the first
// Invoke, mirroring the no-checkpointer contract).
func TestPersistDisabledWithoutCheckpointer(t *testing.T) {
	var calls atomic.Int32
	task := NewTask[any, string]("t", func(_ runtime.Runtime, _ any) (string, error) {
		calls.Add(1)
		return "v", nil
	}, TaskOpts{})
	e := NewEntrypoint[any, string, any](EntrypointOpts{},
		func(ctx runtime.Runtime, _, _ any, _ bool) (string, error) {
			v, err := task.Call(ctx, nil).Get(ctx)
			if err != nil {
				return "", err
			}
			_ = graph.Interrupt(ctx, "q")
			return v, nil
		})

	_, err := e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
	var ierr *InterruptError
	if !errors.As(err, &ierr) {
		t.Fatalf("Invoke() error = %v (%T), want *InterruptError", err, err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("task calls = %d, want 1", got)
	}
}
