package fn

import (
	"context"
	"errors"
	"fmt"
	"net"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func TestNewTaskEmptyNamePanics(t *testing.T) {
	defer func() {
		r := recover()
		if r != "fn: task name must be non-empty" {
			t.Fatalf("recover = %v, want the empty-name panic", r)
		}
	}()
	NewTask[int, int]("", func(runtime.Runtime, int) (int, error) { return 0, nil }, TaskOpts{})
}

func TestNewTaskNilFuncPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("NewTask with nil f did not panic")
		}
	}()
	NewTask[int, int]("x", nil, TaskOpts{})
}

func TestCallOutsideEntrypointPanics(t *testing.T) {
	task := NewTask[int, int]("x", func(_ runtime.Runtime, in int) (int, error) { return in, nil }, TaskOpts{})
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("Call on a bare context did not panic")
		}
		if msg := fmt.Sprint(r); !strings.Contains(msg, "Entrypoint") {
			t.Fatalf("panic = %q, want a message mentioning Entrypoint", msg)
		}
	}()
	task.Call(context.Background(), 1)
}

func TestCallStartsImmediately(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	started := make(chan struct{})
	task := NewTask[int, int]("double", func(_ runtime.Runtime, in int) (int, error) {
		close(started)
		return in * 2, nil
	}, TaskOpts{})

	fut := task.Call(ctx, 21)
	select {
	case <-started: // the task is already running; no Get needed
	case <-time.After(time.Second):
		t.Fatal("task did not start without Get")
	}
	v, err := fut.Get(ctx)
	if err != nil || v != 42 {
		t.Fatalf("Get = (%v, %v), want (42, nil)", v, err)
	}
}

func TestConcurrentFutures(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	task := NewTask[int, int]("sleep", func(_ runtime.Runtime, in int) (int, error) {
		// Later calls sleep less: an in-order result under a serial
		// execution would take 20*(5+4+3+2+1) = 300ms.
		time.Sleep(time.Duration(5-in) * 20 * time.Millisecond)
		return in * 10, nil
	}, TaskOpts{})

	start := time.Now()
	futs := make([]*Future[int], 5)
	for i := range futs {
		futs[i] = task.Call(ctx, i)
	}
	vs, err := AwaitAll(ctx, futs...)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("AwaitAll error = %v, want nil", err)
	}
	for i, v := range vs {
		if v != i*10 {
			t.Fatalf("AwaitAll values = %v, want [0 10 20 30 40] in argument order", vs)
		}
	}
	if elapsed > 250*time.Millisecond {
		t.Fatalf("5 calls took %v, want concurrent execution well under the 300ms serial sum", elapsed)
	}
}

func TestCallCounterDeterministic(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	task := NewTask[int, int]("id", func(_ runtime.Runtime, in int) (int, error) { return in, nil }, TaskOpts{})

	for i := 0; i < 3; i++ {
		if v, err := task.Call(ctx, i).Get(ctx); err != nil || v != i {
			t.Fatalf("call %d: Get = (%v, %v), want (%d, nil)", i, v, err, i)
		}
	}
	if got := d.counts[""]; got != 3 {
		t.Fatalf("counts[\"\"] = %d, want 3", got)
	}
	results := d.snapshotResults()
	if len(results) != 3 {
		t.Fatalf("snapshotResults = %d results, want 3", len(results))
	}
	for i, r := range results {
		if r.callIdx != i {
			t.Fatalf("results[%d].callIdx = %d, want %d (callIdx 0,1,2 in call order)", i, r.callIdx, i)
		}
	}
}

func TestNestedTask(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	var aPath, bPath string
	b := NewTask[int, int]("b", func(ctx runtime.Runtime, in int) (int, error) {
		bPath, _ = ctx.Value(callPathKey{}).(string)
		return in + 1, nil
	}, TaskOpts{})
	a := NewTask[int, int]("a", func(ctx runtime.Runtime, in int) (int, error) {
		aPath, _ = ctx.Value(callPathKey{}).(string)
		return b.Call(ctx, in).Get(ctx)
	}, TaskOpts{})

	v, err := a.Call(ctx, 41).Get(ctx)
	if err != nil || v != 42 {
		t.Fatalf("Get = (%v, %v), want (42, nil)", v, err)
	}
	if aPath != "a@0" {
		t.Fatalf("a's call path = %q, want %q (root call: no leading /)", aPath, "a@0")
	}
	if bPath != "a@0/b@0" {
		t.Fatalf("b's call path = %q, want %q", bPath, "a@0/b@0")
	}
	if d.counts[""] != 1 || d.counts["a@0"] != 1 {
		t.Fatalf("counts = %v, want root and a@0 each counted once (b counts independently)", d.counts)
	}
	var bResult *taskResult
	for i, r := range d.snapshotResults() {
		if r.name == "b" {
			bResult = &d.results[i]
		}
	}
	if bResult == nil || bResult.parentPath != "a@0" || bResult.callIdx != 0 {
		t.Fatalf("b's result = %+v, want parentPath \"a@0\" callIdx 0", bResult)
	}
}

func TestRetrySucceedsAfterFailures(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	var calls atomic.Int32
	task := NewTask[int, int]("flaky", func(_ runtime.Runtime, in int) (int, error) {
		if calls.Add(1) < 3 {
			return 0, &net.DNSError{IsTimeout: true} // DefaultRetryOn retries net.Error
		}
		return in * 2, nil
	}, TaskOpts{Retry: &graph.RetryPolicy{InitialInterval: time.Millisecond, NoJitter: true}})

	v, err := task.Call(ctx, 5).Get(ctx)
	if err != nil || v != 10 {
		t.Fatalf("Get = (%v, %v), want (10, nil)", v, err)
	}
	if got := calls.Load(); got != 3 {
		t.Fatalf("calls = %d, want 3 (two failures then success)", got)
	}
}

func TestRetryExhaustedReturnsLastError(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	var calls atomic.Int32
	task := NewTask[int, int]("always", func(_ runtime.Runtime, in int) (int, error) {
		calls.Add(1)
		return 0, &net.DNSError{IsTimeout: true}
	}, TaskOpts{Retry: &graph.RetryPolicy{MaxAttempts: 2, InitialInterval: time.Millisecond, NoJitter: true}})

	_, err := task.Call(ctx, 1).Get(ctx)
	var dnsErr *net.DNSError
	if !errors.As(err, &dnsErr) {
		t.Fatalf("Get error = %v, want the last *net.DNSError", err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls = %d, want 2 (MaxAttempts)", got)
	}
	results := d.snapshotResults()
	if len(results) != 1 || !results[0].isErr {
		t.Fatalf("snapshotResults = %+v, want one recorded error", results)
	}
}

func TestRetryOnFalseNeverRetries(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	var calls atomic.Int32
	task := NewTask[int, int]("once", func(_ runtime.Runtime, in int) (int, error) {
		calls.Add(1)
		return 0, &net.DNSError{IsTimeout: true}
	}, TaskOpts{Retry: &graph.RetryPolicy{
		InitialInterval: time.Millisecond,
		NoJitter:        true,
		RetryOn:         func(error) bool { return false },
	}})

	if _, err := task.Call(ctx, 1).Get(ctx); err == nil {
		t.Fatal("Get error = nil, want the task's error")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1 (RetryOn false)", got)
	}
}

func TestTaskCache(t *testing.T) {
	cache := checkpoint.NewInMemoryCache()
	var calls atomic.Int32
	task := NewTask[int, int]("double", func(_ runtime.Runtime, in int) (int, error) {
		calls.Add(1)
		return in * 2, nil
	}, TaskOpts{Cache: &graph.CachePolicy{}})

	call := func(d *dispatcher) (int, error) {
		return task.Call(contextWithDispatcher(context.Background(), d), 5).Get(context.Background())
	}

	v, err := call(newDispatcher(cache))
	if err != nil || v != 10 {
		t.Fatalf("first call: Get = (%v, %v), want (10, nil)", v, err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1", got)
	}

	// A new dispatcher (the next turn) with the same backend hits the cache.
	d2 := newDispatcher(cache)
	v, err = call(d2)
	if err != nil || v != 10 {
		t.Fatalf("cached call: Get = (%v, %v), want (10, nil)", v, err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1 (second turn served from cache)", got)
	}
	if got := d2.snapshotResults(); len(got) != 1 || got[0].value != 10 {
		t.Fatalf("cache-hit results = %+v, want the hit re-recorded", got)
	}

	if err := task.ClearCache(context.Background(), cache); err != nil {
		t.Fatalf("ClearCache error = %v, want nil", err)
	}
	if _, err := call(newDispatcher(cache)); err != nil {
		t.Fatalf("post-clear call error = %v, want nil", err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("calls = %d, want 2 after ClearCache", got)
	}
}

func TestCacheKeyFuncReceivesInputMap(t *testing.T) {
	cache := checkpoint.NewInMemoryCache()
	var got map[string]any
	task := NewTask[int, int]("k", func(_ runtime.Runtime, in int) (int, error) {
		return in, nil
	}, TaskOpts{Cache: &graph.CachePolicy{KeyFunc: func(m map[string]any) (types.CacheKey, error) {
		got = m
		return types.CacheKey{Key: "k1"}, nil
	}}})

	d := newDispatcher(cache)
	if _, err := task.Call(contextWithDispatcher(context.Background(), d), 7).Get(context.Background()); err != nil {
		t.Fatalf("Get error = %v, want nil", err)
	}
	if want := (map[string]any{"input": 7}); !reflect.DeepEqual(got, want) {
		t.Fatalf("KeyFunc input = %v, want exactly %v", got, want)
	}
}

func TestCacheKeyFuncErrorFailsTask(t *testing.T) {
	cache := checkpoint.NewInMemoryCache()
	keyErr := errors.New("no key")
	task := NewTask[int, int]("k", func(_ runtime.Runtime, in int) (int, error) {
		return in, nil
	}, TaskOpts{Cache: &graph.CachePolicy{KeyFunc: func(map[string]any) (types.CacheKey, error) {
		return types.CacheKey{}, keyErr
	}}})

	d := newDispatcher(cache)
	_, err := task.Call(contextWithDispatcher(context.Background(), d), 7).Get(context.Background())
	if !errors.Is(err, keyErr) || !strings.Contains(err.Error(), `task "k" cache key`) {
		t.Fatalf("Get error = %v, want the wrapped key error", err)
	}
	if got := d.snapshotResults(); len(got) != 1 || !got[0].isErr {
		t.Fatalf("snapshotResults = %+v, want one recorded error", got)
	}
}

func TestTaskTimeout(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	finished := make(chan struct{})
	task := NewTask[int, int]("slow", func(_ runtime.Runtime, in int) (int, error) {
		time.Sleep(500 * time.Millisecond) // does not honor ctx
		close(finished)
		return in, nil
	}, TaskOpts{Timeout: 50 * time.Millisecond})

	start := time.Now()
	_, err := task.Call(ctx, 1).Get(ctx)
	elapsed := time.Since(start)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Get error = %v, want context.DeadlineExceeded", err)
	}
	if elapsed > 400*time.Millisecond {
		t.Fatalf("Get returned after %v, want ~50ms (the timeout, not f's 500ms)", elapsed)
	}
	select {
	case <-finished: // the abandoned goroutine ran to completion without blocking
	case <-time.After(2 * time.Second):
		t.Fatal("abandoned goroutine did not finish")
	}
}

func TestTaskPanicBecomesError(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	task := NewTask[int, int]("x", func(runtime.Runtime, int) (int, error) {
		panic("boom")
	}, TaskOpts{})

	_, err := task.Call(ctx, 1).Get(ctx)
	if err == nil || !strings.Contains(err.Error(), `task "x" panicked: boom`) {
		t.Fatalf("Get error = %v, want it to contain %q", err, `task "x" panicked: boom`)
	}
	if got := d.snapshotResults(); len(got) != 1 || !got[0].isErr {
		t.Fatalf("snapshotResults = %+v, want one recorded error", got)
	}
}

func TestTaskGraphInterruptPassthrough(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	gi := &types.GraphInterrupt{Interrupt: types.Interrupt{Value: "q", ID: "n-1"}}
	task := NewTask[int, int]("intr", func(runtime.Runtime, int) (int, error) {
		panic(gi)
	}, TaskOpts{})

	fut := task.Call(ctx, 1)
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("Get did not re-panic the GraphInterrupt")
		}
		if r != any(gi) {
			t.Fatalf("panic value = %v, want the same *types.GraphInterrupt", r)
		}
		if got := d.snapshotResults(); len(got) != 0 {
			t.Fatalf("snapshotResults = %+v, want empty (interrupts are not recorded)", got)
		}
	}()
	_, _ = fut.Get(context.Background())
}

func TestRunCancelNotRecorded(t *testing.T) {
	d := newDispatcher(nil)
	runCtx, cancel := context.WithCancel(context.Background())
	ctx := contextWithDispatcher(runCtx, d)
	task := NewTask[int, int]("block", func(ctx runtime.Runtime, _ int) (int, error) {
		<-ctx.Done()
		return 0, ctx.Err()
	}, TaskOpts{})

	fut := task.Call(ctx, 1)
	cancel()
	if _, err := fut.Get(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Get error = %v, want context.Canceled", err)
	}
	if got := d.snapshotResults(); len(got) != 0 {
		t.Fatalf("snapshotResults = %+v, want empty (canceled runs record nothing)", got)
	}
}

func TestSealDropsLateResult(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	task := NewTask[int, int]("slow", func(_ runtime.Runtime, in int) (int, error) {
		time.Sleep(100 * time.Millisecond) // does not honor ctx
		return 7, nil
	}, TaskOpts{})

	fut := task.Call(ctx, 0)
	time.Sleep(10 * time.Millisecond)
	d.seal()
	v, err := fut.Get(ctx)
	if err != nil || v != 7 {
		t.Fatalf("Get = (%v, %v), want (7, nil): the task still completes for its caller", v, err)
	}
	if got := d.snapshotResults(); len(got) != 0 {
		t.Fatalf("snapshotResults = %+v, want empty (sealed dispatcher drops late results)", got)
	}
}

// replayDispatcher builds a dispatcher whose replay gate 1 hit (Resume +
// non-empty Next), with cpID "cp1", ns "", step 3 as the replay base.
func replayDispatcher(t *testing.T, writes ...checkpoint.Write) *dispatcher {
	t.Helper()
	d := newDispatcher(nil)
	d.loadReplay(&checkpoint.Tuple{
		Config: checkpoint.Config{ThreadID: "t1"},
		Checkpoint: checkpoint.Checkpoint{
			ID:   "cp1",
			Next: []checkpoint.PlannedTask{{ID: "x", Node: "entrypoint"}},
		},
		Metadata:      checkpoint.Metadata{Source: "loop", Step: 3},
		PendingWrites: writes,
	}, graph.Options{Resume: "v"})
	if d.replay == nil {
		t.Fatal("replay gate did not hit")
	}
	return d
}

func TestCallReplayReturn(t *testing.T) {
	id := graph.FnTaskID("cp1", "", 3, "a", "", 0)
	d := replayDispatcher(t, checkpoint.Write{TaskID: id, Channel: checkpoint.ReservedReturn, Value: 21})
	var calls atomic.Int32
	task := NewTask[int, int]("a", func(_ runtime.Runtime, in int) (int, error) {
		calls.Add(1)
		return in * 2, nil
	}, TaskOpts{})

	v, err := task.Call(contextWithDispatcher(context.Background(), d), 5).Get(context.Background())
	if err != nil || v != 21 {
		t.Fatalf("Get = (%v, %v), want (21, nil) from the replayed write", v, err)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("calls = %d, want 0 (replay must not re-execute)", got)
	}
	// The replayed result is re-buffered so a chained pause re-appends it
	// to the next checkpoint.
	results := d.snapshotResults()
	if len(results) != 1 || results[0].value != 21 || results[0].name != "a" || results[0].callIdx != 0 {
		t.Fatalf("snapshotResults = %+v, want the replayed result re-recorded", results)
	}
}

func TestCallReplayError(t *testing.T) {
	id := graph.FnTaskID("cp1", "", 3, "a", "", 0)
	d := replayDispatcher(t, checkpoint.Write{TaskID: id, Channel: checkpoint.ReservedError, Value: "boom"})
	var calls atomic.Int32
	task := NewTask[int, int]("a", func(_ runtime.Runtime, in int) (int, error) {
		calls.Add(1)
		return in, nil
	}, TaskOpts{})

	_, err := task.Call(contextWithDispatcher(context.Background(), d), 5).Get(context.Background())
	if err == nil || err.Error() != "boom" {
		t.Fatalf("Get error = %v, want boom", err)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("calls = %d, want 0 (error replays must not re-execute)", got)
	}
	results := d.snapshotResults()
	if len(results) != 1 || !results[0].isErr || results[0].errMsg != "boom" {
		t.Fatalf("snapshotResults = %+v, want the replayed error re-recorded", results)
	}
}

func TestCallReplayTypeMismatch(t *testing.T) {
	id := graph.FnTaskID("cp1", "", 3, "a", "", 0)
	d := replayDispatcher(t, checkpoint.Write{TaskID: id, Channel: checkpoint.ReservedReturn, Value: "oops"})
	task := NewTask[int, int]("a", func(_ runtime.Runtime, in int) (int, error) { return in, nil }, TaskOpts{})

	_, err := task.Call(contextWithDispatcher(context.Background(), d), 5).Get(context.Background())
	if err == nil || !strings.Contains(err.Error(), `replayed result of task "a" has type string`) {
		t.Fatalf("Get error = %v, want a descriptive type-mismatch error", err)
	}
}

// TestCallReplayConcurrent hammers the replay path from many goroutines:
// nextCallIdx, replayWrite, and record all run concurrently (verify with
// -race). Every call resolves from the persisted table; f never executes.
func TestCallReplayConcurrent(t *testing.T) {
	const n = 20
	writes := make([]checkpoint.Write, n)
	for i := 0; i < n; i++ {
		writes[i] = checkpoint.Write{
			TaskID:  graph.FnTaskID("cp1", "", 3, "a", "", i),
			Channel: checkpoint.ReservedReturn,
			Value:   i * 2,
		}
	}
	d := replayDispatcher(t, writes...)
	var calls atomic.Int32
	task := NewTask[int, int]("a", func(_ runtime.Runtime, in int) (int, error) {
		calls.Add(1)
		return in, nil
	}, TaskOpts{})
	ctx := contextWithDispatcher(context.Background(), d)

	out := make(chan int, n)
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func() {
			v, err := task.Call(ctx, 0).Get(ctx)
			if err != nil {
				errs <- err
				return
			}
			out <- v
		}()
	}
	seen := make(map[int]bool, n)
	for i := 0; i < n; i++ {
		select {
		case err := <-errs:
			t.Fatalf("Get error = %v, want nil", err)
		case v := <-out:
			if v < 0 || v >= 2*n || v%2 != 0 || seen[v] {
				t.Fatalf("replayed value %v (seen=%v), want each of 0,2,...,%d exactly once", v, seen, 2*(n-1))
			}
			seen[v] = true
		}
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("calls = %d, want 0 (all calls replayed)", got)
	}
	if got := len(d.snapshotResults()); got != n {
		t.Fatalf("snapshotResults = %d results, want %d re-recorded", got, n)
	}
}

// TestTaskConcurrentStress hammers fresh execution from many goroutines:
// the goroutine writes fut's fields before close(done) (happens-before) and
// every done channel is closed exactly once (verify with -race).
func TestTaskConcurrentStress(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	task := NewTask[int, int]("work", func(_ runtime.Runtime, in int) (int, error) {
		time.Sleep(time.Millisecond)
		return in * 2, nil
	}, TaskOpts{})

	const n = 50
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func() {
			for j := 0; j < 4; j++ {
				v, err := task.Call(ctx, j).Get(ctx)
				if err != nil || v != j*2 {
					errs <- fmt.Errorf("Get = (%v, %v), want (%d, nil)", v, err, j*2)
					return
				}
			}
			errs <- nil
		}()
	}
	for i := 0; i < n; i++ {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if got := d.counts[""]; got != n*4 {
		t.Fatalf("counts[\"\"] = %d, want %d", got, n*4)
	}
	if got := len(d.snapshotResults()); got != n*4 {
		t.Fatalf("snapshotResults = %d results, want %d", got, n*4)
	}
}

// A task without a Cache policy treats ClearCache as a no-op (the cache
// backend is never touched), mirroring Python's `_TaskFunction.clear_cache`.
func TestClearCacheNoPolicy(t *testing.T) {
	task := NewTask[int, int]("plain", func(_ runtime.Runtime, in int) (int, error) { return in, nil }, TaskOpts{})
	if err := task.ClearCache(context.Background(), checkpoint.NewInMemoryCache()); err != nil {
		t.Fatalf("ClearCache error = %v, want nil for a cache-less task", err)
	}
}

// A replayed __error__ write whose persisted value is not a string (a serde
// round-trip can decode it differently) still replays as an error, with the
// message rendered via fmt.Sprint.
func TestCallReplayErrorNonStringValue(t *testing.T) {
	id := graph.FnTaskID("cp1", "", 3, "a", "", 0)
	d := replayDispatcher(t, checkpoint.Write{TaskID: id, Channel: checkpoint.ReservedError, Value: 42})
	task := NewTask[int, int]("a", func(_ runtime.Runtime, in int) (int, error) { return in, nil }, TaskOpts{})

	_, err := task.Call(contextWithDispatcher(context.Background(), d), 5).Get(context.Background())
	if err == nil || err.Error() != "42" {
		t.Fatalf("Get error = %v, want %q (fmt.Sprint of the non-string value)", err, "42")
	}
	results := d.snapshotResults()
	if len(results) != 1 || !results[0].isErr || results[0].errMsg != "42" {
		t.Fatalf("snapshotResults = %+v, want the replayed error re-recorded with message %q", results, "42")
	}
}

// Defense in depth: a replay write on a channel other than
// __return__/__error__ (loadReplay normally filters these) fails the call
// with a descriptive error instead of being misread.
func TestCallReplayUnexpectedChannel(t *testing.T) {
	id := graph.FnTaskID("cp1", "", 3, "a", "", 0)
	d := newDispatcher(nil)
	d.replay = map[string]checkpoint.Write{id: {TaskID: id, Channel: "__other__", Value: 1}}
	d.cpID, d.ns, d.step = "cp1", "", 3
	task := NewTask[int, int]("a", func(_ runtime.Runtime, in int) (int, error) { return in, nil }, TaskOpts{})

	_, err := task.Call(contextWithDispatcher(context.Background(), d), 5).Get(context.Background())
	if err == nil || !strings.Contains(err.Error(), `replayed write of task "a" has unexpected channel "__other__"`) {
		t.Fatalf("Get error = %v, want an unexpected-channel error", err)
	}
}

// A KeyFunc-supplied namespace overrides the default __fn_writes/<task>
// namespace on both the lookup and the store, and a KeyFunc-supplied TTL
// overrides the policy TTL.
func TestCacheNamespaceAndTTLOverride(t *testing.T) {
	cache := checkpoint.NewInMemoryCache()
	var calls atomic.Int32
	task := NewTask[int, int]("ns", func(_ runtime.Runtime, in int) (int, error) {
		calls.Add(1)
		return in * 2, nil
	}, TaskOpts{Cache: &graph.CachePolicy{KeyFunc: func(map[string]any) (types.CacheKey, error) {
		return types.CacheKey{Namespace: []string{"custom", "ns"}, Key: "k", TTL: time.Minute}, nil
	}}})

	d := newDispatcher(cache)
	v, err := task.Call(contextWithDispatcher(context.Background(), d), 3).Get(context.Background())
	if err != nil || v != 6 {
		t.Fatalf("first call: Get = (%v, %v), want (6, nil)", v, err)
	}
	// The result is stored under the KeyFunc namespace, not __fn_writes/ns.
	writes, ok, err := cache.Get(context.Background(), "custom/ns", "k")
	if err != nil || !ok {
		t.Fatalf("cache.Get(custom/ns, k) = (%v, %v, %v), want a hit", writes, ok, err)
	}
	if len(writes) != 1 || writes[0].Channel != checkpoint.ReservedReturn || writes[0].Value != 6 {
		t.Fatalf("cached writes = %+v, want one __return__ write with value 6", writes)
	}
	if _, ok, err := cache.Get(context.Background(), fnCacheNS("ns"), "k"); err != nil || ok {
		t.Fatalf("cache.Get(%q, k) ok = %v, want a miss (namespace overridden)", fnCacheNS("ns"), ok)
	}

	// The next turn's lookup uses the same override and hits the cache.
	d2 := newDispatcher(cache)
	v, err = task.Call(contextWithDispatcher(context.Background(), d2), 3).Get(context.Background())
	if err != nil || v != 6 {
		t.Fatalf("cached call: Get = (%v, %v), want (6, nil)", v, err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls = %d, want 1 (second turn served from the overridden namespace)", got)
	}
}

// errCache is a checkpoint.Cache whose Get/Set fail with the configured
// errors (a miss otherwise), for the cache error-path tests.
type errCache struct {
	getErr error
	setErr error
}

func (c *errCache) Get(context.Context, string, string) ([]checkpoint.Write, bool, error) {
	if c.getErr != nil {
		return nil, false, c.getErr
	}
	return nil, false, nil
}

func (c *errCache) Set(context.Context, string, string, []checkpoint.Write, time.Duration) error {
	return c.setErr
}

func (c *errCache) Clear(context.Context, string) error { return nil }

// A cache backend Get failure fails the task with a wrapped error (graph
// node-cache parity) and is recorded as an __error__ outcome.
func TestCacheGetErrorFailsTask(t *testing.T) {
	boom := errors.New("cache down")
	var calls atomic.Int32
	task := NewTask[int, int]("g", func(_ runtime.Runtime, in int) (int, error) {
		calls.Add(1)
		return in, nil
	}, TaskOpts{Cache: &graph.CachePolicy{}})

	d := newDispatcher(&errCache{getErr: boom})
	_, err := task.Call(contextWithDispatcher(context.Background(), d), 7).Get(context.Background())
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), `task "g" cache get`) {
		t.Fatalf("Get error = %v, want the wrapped cache-get error", err)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("calls = %d, want 0 (the lookup failed before execution)", got)
	}
	if got := d.snapshotResults(); len(got) != 1 || !got[0].isErr {
		t.Fatalf("snapshotResults = %+v, want one recorded error", got)
	}
}

// A cache backend Set failure after a successful execution fails the task
// (the caller must not observe a success first) and records the error.
func TestCacheSetErrorFailsTask(t *testing.T) {
	boom := errors.New("cache read-only")
	task := NewTask[int, int]("s", func(_ runtime.Runtime, in int) (int, error) { return in * 2, nil },
		TaskOpts{Cache: &graph.CachePolicy{}})

	d := newDispatcher(&errCache{setErr: boom})
	_, err := task.Call(contextWithDispatcher(context.Background(), d), 7).Get(context.Background())
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), `task "s" cache set`) {
		t.Fatalf("Get error = %v, want the wrapped cache-set error", err)
	}
	if got := d.snapshotResults(); len(got) != 1 || !got[0].isErr {
		t.Fatalf("snapshotResults = %+v, want one recorded error", got)
	}
}

// A cache hit whose persisted value does not match the task's declared
// output type fails the call with a descriptive error — never a silent
// zero-value downgrade.
func TestCachedResultTypeMismatch(t *testing.T) {
	cache := checkpoint.NewInMemoryCache()
	ctx := context.Background()
	if err := cache.Set(ctx, fnCacheNS("m"), "k1",
		[]checkpoint.Write{{Channel: checkpoint.ReservedReturn, Value: "oops"}}, 0); err != nil {
		t.Fatalf("cache.Set error = %v, want nil", err)
	}
	task := NewTask[int, int]("m", func(_ runtime.Runtime, in int) (int, error) { return in, nil },
		TaskOpts{Cache: &graph.CachePolicy{KeyFunc: func(map[string]any) (types.CacheKey, error) {
			return types.CacheKey{Key: "k1"}, nil
		}}})

	d := newDispatcher(cache)
	_, err := task.Call(contextWithDispatcher(ctx, d), 1).Get(ctx)
	if err == nil || !strings.Contains(err.Error(), `cached result of task "m" has type string`) {
		t.Fatalf("Get error = %v, want a descriptive type-mismatch error", err)
	}
	if got := d.snapshotResults(); len(got) != 1 || !got[0].isErr {
		t.Fatalf("snapshotResults = %+v, want one recorded error", got)
	}
}

// Canceling the run while a retryable task sits in its backoff sleep aborts
// the retry loop and surfaces the context error (the backoff select on
// taskCtx.Done), not the task's failure.
func TestRetryBackoffCanceled(t *testing.T) {
	d := newDispatcher(nil)
	runCtx, cancel := context.WithCancel(context.Background())
	ctx := contextWithDispatcher(runCtx, d)
	var retrying atomic.Bool
	retryingCh := make(chan struct{})
	task := NewTask[int, int]("flaky", func(_ runtime.Runtime, in int) (int, error) {
		return 0, &net.DNSError{IsTimeout: true}
	}, TaskOpts{Retry: &graph.RetryPolicy{
		MaxAttempts:     5,
		InitialInterval: time.Hour, // long enough that only cancellation ends the wait
		NoJitter:        true,
		RetryOn: func(error) bool {
			if retrying.CompareAndSwap(false, true) {
				close(retryingCh)
			}
			return true
		},
	}})

	fut := task.Call(ctx, 1)
	select {
	case <-retryingCh: // first attempt failed; the goroutine is entering the backoff wait
	case <-time.After(2 * time.Second):
		t.Fatal("task did not reach the retry backoff")
	}
	cancel()
	if _, err := fut.Get(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Get error = %v, want context.Canceled", err)
	}
	if got := d.snapshotResults(); len(got) != 0 {
		t.Fatalf("snapshotResults = %+v, want empty (canceled retries record nothing)", got)
	}
}

// A task with a Timeout that finishes well within it resolves normally (the
// attempt-outcome branch of the timeout select).
func TestTaskTimeoutFastSuccess(t *testing.T) {
	d := newDispatcher(nil)
	ctx := contextWithDispatcher(context.Background(), d)
	task := NewTask[int, int]("fast", func(_ runtime.Runtime, in int) (int, error) {
		return in * 2, nil
	}, TaskOpts{Timeout: time.Minute})

	v, err := task.Call(ctx, 21).Get(ctx)
	if err != nil || v != 42 {
		t.Fatalf("Get = (%v, %v), want (42, nil)", v, err)
	}
}

// Canceling the parent context while a timed attempt is still in flight
// surfaces the PARENT's cancellation, not context.DeadlineExceeded.
func TestTaskTimeoutParentCancel(t *testing.T) {
	d := newDispatcher(nil)
	runCtx, cancel := context.WithCancel(context.Background())
	ctx := contextWithDispatcher(runCtx, d)
	started := make(chan struct{})
	release := make(chan struct{})
	defer close(release) // let the abandoned attempt goroutine exit with the test
	task := NewTask[int, int]("slow", func(_ runtime.Runtime, in int) (int, error) {
		close(started)
		<-release // does not honor ctx; only the test releases it
		return in, nil
	}, TaskOpts{Timeout: time.Hour})

	fut := task.Call(ctx, 1)
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("task did not start")
	}
	cancel()
	if _, err := fut.Get(context.Background()); !errors.Is(err, context.Canceled) {
		t.Fatalf("Get error = %v, want context.Canceled (parent cancellation, not the task's timeout)", err)
	}
	if got := d.snapshotResults(); len(got) != 0 {
		t.Fatalf("snapshotResults = %+v, want empty (canceled runs record nothing)", got)
	}
}
