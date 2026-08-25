package savertest_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/savertest"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// This file meta-tests the contract suite: savertest.Run must FAIL when the
// Saver under test violates the contract. Each negative case wraps a
// conformant MemorySaver in a faultySaver injecting exactly one violation,
// runs the full suite against it, and requires the run to fail.
//
// Those nested suite runs fail by design, and a failed subtest would
// normally fail the whole `go test` run. TestMain therefore runs the suite
// twice: phase 1 executes the negative cases (expecting failures), phase 2
// re-runs with them skipped and supplies the process exit code. The phase-1
// failures appear in the test output; they are expected.

var (
	// expectFailures gates the negative cases: they run in phase 1 only.
	expectFailures atomic.Bool
	// negativePhaseRan records that the negative cases executed in phase 1
	// (false when they were filtered out via -run).
	negativePhaseRan atomic.Bool
	// detectionMisses collects the names of negative cases whose suite run
	// unexpectedly PASSED — a blind spot in the contract suite.
	detectionMu     sync.Mutex
	detectionMisses []string
)

func recordDetectionMiss(name string) {
	detectionMu.Lock()
	defer detectionMu.Unlock()
	detectionMisses = append(detectionMisses, name)
}

func TestMain(m *testing.M) {
	// Phase 1: everything runs, including the negative cases whose nested
	// suite runs fail by design. The phase must fail; a pass means the
	// negative cases did not execute at all.
	expectFailures.Store(true)
	code := m.Run()
	if !negativePhaseRan.Load() {
		// Negative cases were filtered out (e.g. -run=...); the single
		// ordinary run is the correct result.
		os.Exit(code)
	}
	if code == 0 {
		fmt.Fprintln(os.Stderr, "savertest: expect-failures phase unexpectedly passed")
		os.Exit(1)
	}
	detectionMu.Lock()
	misses := slices.Clone(detectionMisses)
	detectionMu.Unlock()
	if len(misses) > 0 {
		for _, name := range misses {
			fmt.Fprintf(os.Stderr, "savertest: suite did not detect non-conformant saver %q\n", name)
		}
		os.Exit(1)
	}
	// Phase 2: rerun with the intentionally failing negative cases skipped;
	// this run's result is the package's real result.
	expectFailures.Store(false)
	os.Exit(m.Run())
}

// TestRunRejectsNonConformantSavers runs the contract suite against one
// faulty saver per contract clause and requires every run to fail. A
// passing nested run means the suite has a blind spot; it is reported as a
// detection miss and fails the overall test run (see TestMain).
func TestRunRejectsNonConformantSavers(t *testing.T) {
	if !expectFailures.Load() {
		t.Skip("negative cases run only in the expect-failures phase (see TestMain)")
	}
	negativePhaseRan.Store(true)
	for _, tc := range rejectionCases {
		t.Run(tc.name, func(t *testing.T) {
			conformant := t.Run("suite", func(t *testing.T) {
				savertest.Run(t, tc.newSaver)
			})
			if conformant {
				recordDetectionMiss(tc.name)
			}
		})
	}
}

// errFault is the error injected by faultySaver.
var errFault = errors.New("savertest: injected fault")

// Hook types for faultySaver; next delegates to the wrapped saver.
type (
	putHook          func(call int, cfg checkpoint.Config, next func() (checkpoint.Config, error)) (checkpoint.Config, error)
	getTupleHook     func(cfg checkpoint.Config, next func() (*checkpoint.Tuple, error)) (*checkpoint.Tuple, error)
	listHook         func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error)
	putWritesHook    func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error
	deleteThreadHook func(next func() error) error
)

// faultySaver wraps a conformant Saver and lets a test inject one contract
// violation per method. The call arguments are 1-based per saver instance.
type faultySaver struct {
	inner checkpoint.Saver

	putCalls       atomic.Int64
	onPut          putHook
	onGetTuple     getTupleHook
	onList         listHook
	putWritesCalls atomic.Int64
	onPutWrites    putWritesHook
	onDeleteThread deleteThreadHook
}

func (f *faultySaver) Put(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64) (checkpoint.Config, error) {
	call := f.putCalls.Add(1)
	next := func() (checkpoint.Config, error) { return f.inner.Put(ctx, cfg, cp, md, newVersions) }
	if f.onPut != nil {
		return f.onPut(int(call), cfg, next)
	}
	return next()
}

func (f *faultySaver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	next := func() (*checkpoint.Tuple, error) { return f.inner.GetTuple(ctx, cfg) }
	if f.onGetTuple != nil {
		return f.onGetTuple(cfg, next)
	}
	return next()
}

func (f *faultySaver) List(ctx context.Context, cfg checkpoint.Config, opts checkpoint.ListOptions) ([]checkpoint.Tuple, error) {
	next := func(c checkpoint.Config, o checkpoint.ListOptions) ([]checkpoint.Tuple, error) {
		return f.inner.List(ctx, c, o)
	}
	if f.onList != nil {
		return f.onList(cfg, opts, next)
	}
	return next(cfg, opts)
}

func (f *faultySaver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	call := f.putWritesCalls.Add(1)
	next := func(w []checkpoint.Write, id, path string) error { return f.inner.PutWrites(ctx, cfg, w, id, path) }
	if f.onPutWrites != nil {
		return f.onPutWrites(int(call), writes, taskID, taskPath, next)
	}
	return next(writes, taskID, taskPath)
}

func (f *faultySaver) DeleteThread(ctx context.Context, threadID string) error {
	next := func() error { return f.inner.DeleteThread(ctx, threadID) }
	if f.onDeleteThread != nil {
		return f.onDeleteThread(next)
	}
	return next()
}

func (f *faultySaver) DeleteForRuns(ctx context.Context, runIDs []string) error {
	return f.inner.DeleteForRuns(ctx, runIDs)
}

func (f *faultySaver) CopyThread(ctx context.Context, srcThreadID, dstThreadID string) error {
	return f.inner.CopyThread(ctx, srcThreadID, dstThreadID)
}

func (f *faultySaver) Prune(ctx context.Context, threadIDs []string, strategy checkpoint.PruneStrategy) error {
	return f.inner.Prune(ctx, threadIDs, strategy)
}

// rejectionCase pairs a violation name with the hook that injects it.
type rejectionCase struct {
	name string
	hook func(f *faultySaver)
}

// newSaver returns a fresh faultySaver (empty storage) carrying the case's
// violation, as savertest.Run's factory contract requires.
func (tc rejectionCase) newSaver(t *testing.T) checkpoint.Saver {
	t.Helper()
	f := &faultySaver{inner: checkpoint.NewMemorySaver()}
	tc.hook(f)
	return f
}

func putCase(name string, h putHook) rejectionCase {
	return rejectionCase{name, func(f *faultySaver) { f.onPut = h }}
}

func getTupleCase(name string, h getTupleHook) rejectionCase {
	return rejectionCase{name, func(f *faultySaver) { f.onGetTuple = h }}
}

func listCase(name string, h listHook) rejectionCase {
	return rejectionCase{name, func(f *faultySaver) { f.onList = h }}
}

func putWritesCase(name string, h putWritesHook) rejectionCase {
	return rejectionCase{name, func(f *faultySaver) { f.onPutWrites = h }}
}

// mutatingGetTuple adapts a tuple mutation into a getTupleHook: the fetched
// tuple is shallow-copied (MemorySaver already clones its maps and slices
// per read), mutated, and returned. Nil tuples and errors pass through.
func mutatingGetTuple(mut func(cfg checkpoint.Config, tup *checkpoint.Tuple)) getTupleHook {
	return func(cfg checkpoint.Config, next func() (*checkpoint.Tuple, error)) (*checkpoint.Tuple, error) {
		tup, err := next()
		if err != nil || tup == nil {
			return tup, err
		}
		dup := *tup
		mut(cfg, &dup)
		return &dup, nil
	}
}

// rejectionCases enumerates the contract violations the suite must detect.
// Each breaks exactly one clause, so the suite's failure pins that clause.
var rejectionCases = []rejectionCase{
	// --- Put failures and results ---------------------------------------
	putCase("put_error", func(call int, cfg checkpoint.Config, next func() (checkpoint.Config, error)) (checkpoint.Config, error) {
		return checkpoint.Config{}, errFault
	}),
	putCase("put_returns_wrong_config", func(call int, cfg checkpoint.Config, next func() (checkpoint.Config, error)) (checkpoint.Config, error) {
		out, err := next()
		out.CheckpointID += "-bogus"
		return out, err
	}),
	putCase("put_with_parent_fails", func(call int, cfg checkpoint.Config, next func() (checkpoint.Config, error)) (checkpoint.Config, error) {
		if cfg.CheckpointID != "" {
			return checkpoint.Config{}, errFault
		}
		return next()
	}),
	putCase("put_thread_t2_fails", func(call int, cfg checkpoint.Config, next func() (checkpoint.Config, error)) (checkpoint.Config, error) {
		if cfg.ThreadID == "t2" {
			return checkpoint.Config{}, errFault
		}
		return next()
	}),

	// --- GetTuple failures and corrupted reads --------------------------
	getTupleCase("gettuple_error", func(cfg checkpoint.Config, next func() (*checkpoint.Tuple, error)) (*checkpoint.Tuple, error) {
		return nil, errFault
	}),
	getTupleCase("gettuple_drops_existing", func(cfg checkpoint.Config, next func() (*checkpoint.Tuple, error)) (*checkpoint.Tuple, error) {
		if _, err := next(); err != nil {
			return nil, err
		}
		return nil, nil
	}),
	getTupleCase("gettuple_corrupts_checkpoint", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		tup.Checkpoint.TS = tup.Checkpoint.TS.Add(time.Hour)
	})),
	getTupleCase("gettuple_corrupts_metadata", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		tup.Metadata.Step++
	})),
	getTupleCase("gettuple_fabricates_parent", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		if tup.ParentConfig == nil {
			tup.ParentConfig = &checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointID: "ghost"}
		}
	})),
	getTupleCase("gettuple_wrong_tuple_config", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		tup.Config.CheckpointID = "bogus"
	})),
	getTupleCase("gettuple_drops_by_id", func(cfg checkpoint.Config, next func() (*checkpoint.Tuple, error)) (*checkpoint.Tuple, error) {
		tup, err := next()
		if err != nil {
			return nil, err
		}
		if cfg.CheckpointID != "" {
			return nil, nil
		}
		return tup, nil
	}),
	getTupleCase("gettuple_corrupts_by_id", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		if cfg.CheckpointID != "" {
			tup.Checkpoint.TS = tup.Checkpoint.TS.Add(time.Hour)
		}
	})),
	getTupleCase("gettuple_fabricates_missing", func(cfg checkpoint.Config, next func() (*checkpoint.Tuple, error)) (*checkpoint.Tuple, error) {
		tup, err := next()
		if err != nil || tup != nil {
			return tup, err
		}
		return &checkpoint.Tuple{Config: cfg, Checkpoint: checkpoint.Checkpoint{ID: "ghost"}}, nil
	}),
	getTupleCase("gettuple_corrupts_int_value", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		if _, ok := tup.Checkpoint.ChannelValues["n"]; ok {
			tup.Checkpoint.ChannelValues["n"] = 42.5
		}
	})),
	getTupleCase("gettuple_corrupts_int64_value", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		if _, ok := tup.Checkpoint.ChannelValues["big"]; ok {
			tup.Checkpoint.ChannelValues["big"] = int64(1)
		}
	})),
	getTupleCase("gettuple_drops_parent", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		tup.ParentConfig = nil
	})),
	getTupleCase("gettuple_drops_thread_t2", func(cfg checkpoint.Config, next func() (*checkpoint.Tuple, error)) (*checkpoint.Tuple, error) {
		tup, err := next()
		if err != nil {
			return nil, err
		}
		if cfg.ThreadID == "t2" {
			return nil, nil
		}
		return tup, nil
	}),
	getTupleCase("gettuple_strips_t2_writes", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		if cfg.ThreadID == "t2" {
			tup.PendingWrites = nil
		}
	})),
	getTupleCase("gettuple_appends_pending_write", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		tup.PendingWrites = append(slices.Clone(tup.PendingWrites), checkpoint.Write{TaskID: "ghost", Channel: "ghost", Value: 1})
	})),
	getTupleCase("gettuple_corrupts_state_key_write", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		for i, w := range tup.PendingWrites {
			if w.Channel == "state_key" {
				tup.PendingWrites[i].Value = "corrupted"
			}
		}
	})),
	getTupleCase("gettuple_corrupts_tasks_write", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		for i, w := range tup.PendingWrites {
			if w.Channel == checkpoint.ReservedTasks {
				tup.PendingWrites[i].Value = types.Send{Node: "corrupted"}
			}
		}
	})),
	getTupleCase("gettuple_corrupts_interrupt_write", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		for i, w := range tup.PendingWrites {
			if w.Channel == checkpoint.ReservedInterrupt {
				tup.PendingWrites[i].Value = types.Interrupt{Value: "corrupted", ID: "i1"}
			}
		}
	})),
	getTupleCase("gettuple_corrupts_first_write_value", mutatingGetTuple(func(cfg checkpoint.Config, tup *checkpoint.Tuple) {
		if len(tup.PendingWrites) > 0 {
			tup.PendingWrites[0].Value = "corrupted"
		}
	})),

	// --- List failures and ordering/filter violations -------------------
	listCase("list_error", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		return nil, errFault
	}),
	listCase("list_drops_oldest", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		tups, err := next(cfg, opts)
		if len(tups) > 0 {
			tups = tups[:len(tups)-1]
		}
		return tups, err
	}),
	listCase("list_reverses_order", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		tups, err := next(cfg, opts)
		slices.Reverse(tups)
		return tups, err
	}),
	listCase("list_before_errors", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		if opts.Before != nil {
			return nil, errFault
		}
		return next(cfg, opts)
	}),
	listCase("list_ignores_before", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		opts.Before = nil
		return next(cfg, opts)
	}),
	listCase("list_limit_errors", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		if opts.Limit > 0 {
			return nil, errFault
		}
		return next(cfg, opts)
	}),
	listCase("list_ignores_limit", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		opts.Limit = 0
		return next(cfg, opts)
	}),
	listCase("list_filter_returns_empty", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		if len(opts.Filter) > 0 {
			return nil, nil
		}
		return next(cfg, opts)
	}),
	listCase("list_step_filter_returns_empty", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		if _, ok := opts.Filter["step"]; ok {
			return nil, nil
		}
		return next(cfg, opts)
	}),
	listCase("list_empty_filter_thread1_returns_empty", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		if opts.Filter != nil && len(opts.Filter) == 0 && cfg.ThreadID == "thread-1" {
			return nil, nil
		}
		return next(cfg, opts)
	}),
	listCase("list_empty_filter_thread2_root_returns_empty", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		if opts.Filter != nil && len(opts.Filter) == 0 && cfg.ThreadID == "thread-2" && cfg.CheckpointNS == "" {
			return nil, nil
		}
		return next(cfg, opts)
	}),
	listCase("list_empty_filter_inner_ns_returns_empty", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		if opts.Filter != nil && len(opts.Filter) == 0 && cfg.CheckpointNS == "inner" {
			return nil, nil
		}
		return next(cfg, opts)
	}),
	listCase("list_ignores_filter", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		opts.Filter = nil
		return next(cfg, opts)
	}),
	listCase("list_leaks_inner_namespace", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		tups, err := next(cfg, opts)
		if err != nil || opts.Filter != nil || cfg.CheckpointNS != "" {
			return tups, err
		}
		inner, err := next(checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: "inner"}, opts)
		if err != nil {
			return nil, err
		}
		return append(tups, inner...), nil
	}),
	listCase("list_fabricates_when_empty", func(cfg checkpoint.Config, opts checkpoint.ListOptions, next func(checkpoint.Config, checkpoint.ListOptions) ([]checkpoint.Tuple, error)) ([]checkpoint.Tuple, error) {
		tups, err := next(cfg, opts)
		if err != nil || len(tups) > 0 {
			return tups, err
		}
		return []checkpoint.Tuple{{Config: cfg, Checkpoint: checkpoint.Checkpoint{ID: "ghost"}}}, nil
	}),

	// --- PutWrites failures and stamping violations ---------------------
	putWritesCase("putwrites_error", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		return errFault
	}),
	putWritesCase("putwrites_second_call_fails", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		if call == 2 {
			return errFault
		}
		return next(writes, taskID, taskPath)
	}),
	putWritesCase("putwrites_third_call_fails", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		if call == 3 {
			return errFault
		}
		return next(writes, taskID, taskPath)
	}),
	putWritesCase("putwrites_fourth_call_fails", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		if call == 4 {
			return errFault
		}
		return next(writes, taskID, taskPath)
	}),
	putWritesCase("putwrites_missing_checkpoint_succeeds", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		return nil
	}),
	putWritesCase("putwrites_empty_task_id", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		return next(writes, "", taskPath)
	}),
	putWritesCase("putwrites_wrong_task_id", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		return next(writes, "wrong-task", taskPath)
	}),
	putWritesCase("putwrites_drops_last_write", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		if len(writes) > 0 {
			writes = writes[:len(writes)-1]
		}
		return next(writes, taskID, taskPath)
	}),
	putWritesCase("putwrites_reorders_writes", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		reversed := slices.Clone(writes)
		slices.Reverse(reversed)
		return next(reversed, taskID, taskPath)
	}),
	putWritesCase("putwrites_strips_task_path", func(call int, writes []checkpoint.Write, taskID, taskPath string, next func([]checkpoint.Write, string, string) error) error {
		return next(writes, taskID, "")
	}),

	// --- DeleteThread failures ------------------------------------------
	{
		name: "deletethread_error",
		hook: func(f *faultySaver) {
			f.onDeleteThread = func(next func() error) error { return errFault }
		},
	},
}
