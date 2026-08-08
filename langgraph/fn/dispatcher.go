package fn

import (
	"context"
	"sync"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
)

// dispatcher drives Task.Call execution and replay for one entrypoint run.
// The fn wrapper layer (Invoke/Stream) owns it — it is NOT on the node
// stack, so an interrupt panic unwinding the node never loses buffered
// results. It reaches Task.Call through the context, the same injection
// pattern as graph.Interrupt (graph/graph.go:1127-1128).
type dispatcher struct {
	mu       sync.Mutex
	counts   map[string]int              // parentPath -> next call index (per-run replay counter)
	replay   map[string]checkpoint.Write // deterministic task ID -> persisted result write; nil on a fresh run
	consumed map[string]int              // deterministic task ID -> resume values the persisted execution consumed (nil on a fresh run)
	cpID     string                      // replay base: the checkpoint the run resumed from
	ns       string                      // checkpoint namespace (always "" — fn runs are root-namespace)
	step     int                         // replay base: that checkpoint's Metadata.Step
	cache    checkpoint.Cache            // EntrypointOpts.Cache; nil disables task caching
	results  []taskResult                // every outcome completed this run (execution, replay, cache hit)
	sealed   bool                        // set at run end (after cancel); record() drops everything once sealed
}

type taskResult struct {
	name       string
	parentPath string
	callIdx    int
	value      any    // set when isErr is false (falsy values included — the channel carries the state, not truthiness)
	errMsg     string // set when isErr is true
	isErr      bool
	consumed   int // resume values the execution consumed via Interrupt (0 for replayed non-interrupting tasks)
}

type dispatcherKey struct{} // context key for *dispatcher
type callPathKey struct{}   // context key for the caller's call path (string)

// newDispatcher returns a dispatcher for one entrypoint run. cache is the
// EntrypointOpts.Cache backend; nil disables task caching.
func newDispatcher(cache checkpoint.Cache) *dispatcher {
	return &dispatcher{
		counts: make(map[string]int),
		cache:  cache,
	}
}

// contextWithDispatcher injects d into ctx so Task.Call can reach it.
func contextWithDispatcher(ctx context.Context, d *dispatcher) context.Context {
	return context.WithValue(ctx, dispatcherKey{}, d)
}

// dispatcherFromContext returns the dispatcher injected into ctx, or nil
// when ctx carries none (Task.Call panics on nil — the Go analogue of
// Python's runtime error for tasks called outside an entrypoint).
func dispatcherFromContext(ctx context.Context) *dispatcher {
	d, _ := ctx.Value(dispatcherKey{}).(*dispatcher)
	return d
}

// nextCallIdx returns the next call index for parentPath (0, 1, 2, ...),
// counting each parent path independently. Every entrypoint re-run replays
// the numbering from zero, so the call order across re-runs must be
// deterministic (same constraint as Python's determinism section).
func (d *dispatcher) nextCallIdx(parentPath string) int {
	d.mu.Lock()
	defer d.mu.Unlock()
	idx := d.counts[parentPath]
	d.counts[parentPath] = idx + 1
	return idx
}

// record appends an outcome to the run's result table. Once the dispatcher
// is sealed (run teardown), late completions are dropped: a task that
// finished after the run returned must not enter the persisted table.
func (d *dispatcher) record(r taskResult) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.sealed {
		return
	}
	d.results = append(d.results, r)
}

// seal marks the run ended; record drops everything from here on. The fn
// layer calls it after canceling the run-scoped context and before
// snapshotResults (the F4 teardown order: cancel -> seal -> snapshot ->
// PutWrites).
func (d *dispatcher) seal() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.sealed = true
}

// snapshotResults returns a copy of the whole result table.
func (d *dispatcher) snapshotResults() []taskResult {
	d.mu.Lock()
	defer d.mu.Unlock()
	out := make([]taskResult, len(d.results))
	copy(out, d.results)
	return out
}

// replayWrite returns the persisted result write for a deterministic task
// ID, or ok=false on a fresh run / replay miss. Task.Call uses it to fill a
// Future from the checkpoint instead of re-executing the task.
func (d *dispatcher) replayWrite(taskID string) (checkpoint.Write, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	w, ok := d.replay[taskID]
	return w, ok
}

// replayConsumed returns how many resume values the persisted execution of
// taskID consumed via Interrupt (0 on a fresh run / replay miss / a task
// that never interrupted). Task.Call advances the node's shared resume queue
// by this count when it replays the result (see graph.ReplayInterruptConsumption).
func (d *dispatcher) replayConsumed(taskID string) int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.consumed[taskID]
}

// loadReplay builds the replay table from the checkpoint the run resumed
// from, when — and only when — one of the two gates holds:
//
//  1. opts.Resume != nil and the tuple's Checkpoint.Next is non-empty (the
//     tuple is a pause checkpoint: an interrupt resume); or
//  2. the tuple's Metadata.Source is "input" and its pending writes contain
//     __return__/__error__ writes (the previous invocation failed before
//     committing its first superstep — an entrypoint error retry, mirroring
//     Python's invoke(None, config) re-run re-raise).
//
// A fresh run (a "loop"-source tuple without Resume, or no fn writes) does
// NOT replay, so a new turn can never hit the previous turn's results. On a
// gate hit, cpID/ns/step are taken from the tuple as the replay base; the
// replay table keeps only __return__/__error__ writes (last write wins) and
// the consumed table the __fn_consumed__ counts (see ReservedFnConsumed for
// why replayed tasks must re-skip their consumed resume values).
func (d *dispatcher) loadReplay(tup *checkpoint.Tuple, opts graph.Options) {
	hit := opts.Resume != nil && len(tup.Checkpoint.Next) > 0 // gate 1
	if !hit && tup.Metadata.Source == "input" {               // gate 2
		for _, w := range tup.PendingWrites {
			if w.Channel == checkpoint.ReservedReturn || w.Channel == checkpoint.ReservedError {
				hit = true
				break
			}
		}
	}
	if !hit {
		return
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	d.replay = make(map[string]checkpoint.Write)
	d.consumed = make(map[string]int)
	for _, w := range tup.PendingWrites {
		switch w.Channel {
		case checkpoint.ReservedReturn, checkpoint.ReservedError:
			d.replay[w.TaskID] = w // last write wins
		case checkpoint.ReservedFnConsumed:
			// Savers round-tripping through JSON serde decode the count as
			// float64; accept both (memory saver keeps the int as-is).
			switch v := w.Value.(type) {
			case int:
				d.consumed[w.TaskID] = v
			case float64:
				d.consumed[w.TaskID] = int(v)
			}
		}
	}
	d.cpID = tup.Checkpoint.ID
	d.ns = tup.Config.CheckpointNS
	d.step = tup.Metadata.Step
}
