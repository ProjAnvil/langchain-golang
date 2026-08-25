package savertest

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// putChain puts n checkpoints on (tid, ns), each Put carrying the previous
// checkpoint's Config so parent links form a chain — mirroring the
// conformance _setup_thread helper (test_prune.py:18-31). Metadata gets
// Step i and — when runIDs has exactly n entries — RunID runIDs[i]; the
// channel values carry "step": i so copied content is distinguishable per
// checkpoint (test_copy_thread.py:36 uses the same shape). Returns the
// configs Put returned, in put order.
func putChain(t *testing.T, s checkpoint.Saver, tid, ns string, n int, runIDs []string) []checkpoint.Config {
	t.Helper()
	ctx := context.Background()
	out := make([]checkpoint.Config, 0, n)
	var cfg checkpoint.Config
	for i := 0; i < n; i++ {
		md := checkpoint.Metadata{Source: "loop", Step: i}
		if len(runIDs) == n {
			md.RunID = runIDs[i]
		}
		cp := sampleCheckpoint(checkpoint.NewID(i + 1))
		cp.ChannelValues["step"] = i
		next, err := s.Put(ctx,
			checkpoint.Config{ThreadID: tid, CheckpointNS: ns, CheckpointID: cfg.CheckpointID},
			cp, md, nil)
		if err != nil {
			t.Fatalf("Put %d on %s/%q: %v", i, tid, ns, err)
		}
		cfg = next
		out = append(out, next)
	}
	return out
}

// listIDs returns the thread/namespace's checkpoint IDs, newest first.
func listIDs(t *testing.T, s checkpoint.Saver, cfg checkpoint.Config) []string {
	t.Helper()
	tups, err := s.List(context.Background(), cfg, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List(%+v): %v", cfg, err)
	}
	return tupleIDs(tups)
}

// testDeleteForRuns mirrors the checkpoint-conformance spec
// test_delete_for_runs.py (all seven scenarios).
func testDeleteForRuns(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()

	runIDsOf := func(t *testing.T, s checkpoint.Saver, cfg checkpoint.Config) map[string]bool {
		t.Helper()
		tups, err := s.List(ctx, cfg, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("List: %v", err)
		}
		out := map[string]bool{}
		for _, tup := range tups {
			out[tup.Metadata.RunID] = true
		}
		return out
	}

	// test_delete_for_runs_single: one run_id removed, the other kept.
	t.Run("single", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "t1", "", 2, []string{"run1", "run2"})
		if got := runIDsOf(t, s, checkpoint.Config{ThreadID: "t1"}); !got["run1"] || !got["run2"] {
			t.Fatalf("pre-delete runs = %v, want run1+run2", got)
		}
		if err := s.DeleteForRuns(ctx, []string{"run1"}); err != nil {
			t.Fatalf("DeleteForRuns: %v", err)
		}
		got := runIDsOf(t, s, checkpoint.Config{ThreadID: "t1"})
		if got["run1"] || !got["run2"] {
			t.Fatalf("post-delete runs = %v, want only run2", got)
		}
	})

	// test_delete_for_runs_multiple: a list of run_ids is removed.
	t.Run("multiple", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "t1", "", 3, []string{"run1", "run2", "run3"})
		if err := s.DeleteForRuns(ctx, []string{"run1", "run2"}); err != nil {
			t.Fatalf("DeleteForRuns: %v", err)
		}
		got := runIDsOf(t, s, checkpoint.Config{ThreadID: "t1"})
		if got["run1"] || got["run2"] || !got["run3"] {
			t.Fatalf("post-delete runs = %v, want only run3", got)
		}
	})

	// test_delete_for_runs_preserves_other_runs: unrelated runs untouched.
	t.Run("preserves_other_runs", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "t1", "", 2, []string{"run-keep", "run-delete"})
		if err := s.DeleteForRuns(ctx, []string{"run-delete"}); err != nil {
			t.Fatalf("DeleteForRuns: %v", err)
		}
		if got := runIDsOf(t, s, checkpoint.Config{ThreadID: "t1"}); !got["run-keep"] {
			t.Fatalf("post-delete runs = %v, want run-keep", got)
		}
	})

	// test_delete_for_runs_removes_writes: the run's checkpoint AND its
	// writes are gone.
	t.Run("removes_writes", func(t *testing.T) {
		s := newSaver(t)
		cfgs := putChain(t, s, "t1", "", 1, []string{"run1"})
		if err := s.PutWrites(ctx, cfgs[0], []checkpoint.Write{{Channel: "ch", Value: "val"}}, "task-1", ""); err != nil {
			t.Fatalf("PutWrites: %v", err)
		}
		if err := s.DeleteForRuns(ctx, []string{"run1"}); err != nil {
			t.Fatalf("DeleteForRuns: %v", err)
		}
		tup, err := s.GetTuple(ctx, cfgs[0])
		if err != nil || tup != nil {
			t.Fatalf("GetTuple after DeleteForRuns = %v, %v; want nil, nil", tup, err)
		}
	})

	// test_delete_for_runs_empty_list_noop / _nonexistent_noop.
	t.Run("empty_and_nonexistent_noop", func(t *testing.T) {
		s := newSaver(t)
		if err := s.DeleteForRuns(ctx, nil); err != nil {
			t.Fatalf("DeleteForRuns(nil): %v", err)
		}
		if err := s.DeleteForRuns(ctx, []string{"no-such-run"}); err != nil {
			t.Fatalf("DeleteForRuns(nonexistent): %v", err)
		}
	})

	// test_delete_for_runs_across_namespaces: all namespaces cleaned.
	t.Run("across_namespaces", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "t1", "", 1, []string{"run1"})
		putChain(t, s, "t1", "child:1", 1, []string{"run1"})
		if err := s.DeleteForRuns(ctx, []string{"run1"}); err != nil {
			t.Fatalf("DeleteForRuns: %v", err)
		}
		for _, ns := range []string{"", "child:1"} {
			if ids := listIDs(t, s, checkpoint.Config{ThreadID: "t1", CheckpointNS: ns}); len(ids) != 0 {
				t.Fatalf("ns %q still has checkpoints %v", ns, ids)
			}
		}
	})
}

// testCopyThread mirrors the checkpoint-conformance spec test_copy_thread.py
// (all eight scenarios).
func testCopyThread(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()

	// test_copy_thread_basic / _all_checkpoints / _preserves_ordering:
	// checkpoints appear on the target, content-identical, in the same order.
	t.Run("basic_all_ordering", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "src", "", 3, nil)
		if err := s.CopyThread(ctx, "src", "dst"); err != nil {
			t.Fatalf("CopyThread: %v", err)
		}
		srcIDs := listIDs(t, s, checkpoint.Config{ThreadID: "src"})
		dstIDs := listIDs(t, s, checkpoint.Config{ThreadID: "dst"})
		if !reflect.DeepEqual(srcIDs, dstIDs) {
			t.Fatalf("dst IDs = %v, want %v (same checkpoints, same newest-first order)", dstIDs, srcIDs)
		}
		// Content matches per checkpoint (channel_values incl. "step").
		for _, id := range srcIDs {
			st, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "src", CheckpointID: id})
			if err != nil || st == nil {
				t.Fatalf("src GetTuple %q: %v, %v", id, st, err)
			}
			dt, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "dst", CheckpointID: id})
			if err != nil || dt == nil {
				t.Fatalf("dst GetTuple %q: %v, %v", id, dt, err)
			}
			if !reflect.DeepEqual(st.Checkpoint.ChannelValues, dt.Checkpoint.ChannelValues) {
				t.Fatalf("checkpoint %q channel_values mismatch:\nsrc %+v\ndst %+v", id, st.Checkpoint.ChannelValues, dt.Checkpoint.ChannelValues)
			}
		}
	})

	// test_copy_thread_preserves_metadata.
	t.Run("preserves_metadata", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "src", "", 2, nil)
		if err := s.CopyThread(ctx, "src", "dst"); err != nil {
			t.Fatalf("CopyThread: %v", err)
		}
		srcTups, err := s.List(ctx, checkpoint.Config{ThreadID: "src"}, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("List src: %v", err)
		}
		dstTups, err := s.List(ctx, checkpoint.Config{ThreadID: "dst"}, checkpoint.ListOptions{})
		if err != nil {
			t.Fatalf("List dst: %v", err)
		}
		if len(srcTups) != len(dstTups) {
			t.Fatalf("dst has %d tuples, want %d", len(dstTups), len(srcTups))
		}
		for i := range srcTups {
			if !reflect.DeepEqual(srcTups[i].Metadata, dstTups[i].Metadata) {
				t.Fatalf("metadata[%d] mismatch: src %+v dst %+v", i, srcTups[i].Metadata, dstTups[i].Metadata)
			}
			// Parent links survive the copy (full parent chain, Python's
			// DeltaChannel warning on copy_thread, base/__init__.py:361-371).
			if (srcTups[i].ParentConfig == nil) != (dstTups[i].ParentConfig == nil) {
				t.Fatalf("parent link mismatch at %d: src %+v dst %+v", i, srcTups[i].ParentConfig, dstTups[i].ParentConfig)
			}
			if srcTups[i].ParentConfig != nil && srcTups[i].ParentConfig.CheckpointID != dstTups[i].ParentConfig.CheckpointID {
				t.Fatalf("parent ID mismatch at %d: src %q dst %q", i, srcTups[i].ParentConfig.CheckpointID, dstTups[i].ParentConfig.CheckpointID)
			}
		}
	})

	// test_copy_thread_preserves_namespaces: root + child namespaces copied.
	t.Run("preserves_namespaces", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "src", "", 1, nil)
		putChain(t, s, "src", "child:1", 1, nil)
		if err := s.CopyThread(ctx, "src", "dst"); err != nil {
			t.Fatalf("CopyThread: %v", err)
		}
		for _, ns := range []string{"", "child:1"} {
			if ids := listIDs(t, s, checkpoint.Config{ThreadID: "dst", CheckpointNS: ns}); len(ids) != 1 {
				t.Fatalf("dst ns %q has %d checkpoints, want 1", ns, len(ids))
			}
		}
	})

	// test_copy_thread_preserves_writes.
	t.Run("preserves_writes", func(t *testing.T) {
		s := newSaver(t)
		cfgs := putChain(t, s, "src", "", 1, nil)
		if err := s.PutWrites(ctx, cfgs[0], []checkpoint.Write{{Channel: "ch", Value: "write_val"}}, "task-1", ""); err != nil {
			t.Fatalf("PutWrites: %v", err)
		}
		if err := s.CopyThread(ctx, "src", "dst"); err != nil {
			t.Fatalf("CopyThread: %v", err)
		}
		tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "dst"})
		if err != nil || tup == nil {
			t.Fatalf("dst GetTuple: %v, %v", tup, err)
		}
		if len(tup.PendingWrites) != 1 || tup.PendingWrites[0].Channel != "ch" || tup.PendingWrites[0].Value != "write_val" {
			t.Fatalf("dst PendingWrites = %+v, want one ch=write_val write", tup.PendingWrites)
		}
	})

	// test_copy_thread_source_unchanged.
	t.Run("source_unchanged", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "src", "", 2, nil)
		before := listIDs(t, s, checkpoint.Config{ThreadID: "src"})
		if err := s.CopyThread(ctx, "src", "dst"); err != nil {
			t.Fatalf("CopyThread: %v", err)
		}
		if after := listIDs(t, s, checkpoint.Config{ThreadID: "src"}); !reflect.DeepEqual(before, after) {
			t.Fatalf("src changed: %v -> %v", before, after)
		}
	})

	// test_copy_thread_nonexistent_source: no error, destination stays empty.
	t.Run("nonexistent_source", func(t *testing.T) {
		s := newSaver(t)
		if err := s.CopyThread(ctx, "nope", "dst"); err != nil {
			t.Fatalf("CopyThread from nonexistent source: %v", err)
		}
		if ids := listIDs(t, s, checkpoint.Config{ThreadID: "dst"}); len(ids) != 0 {
			t.Fatalf("dst has %v, want empty", ids)
		}
	})
}

// testPrune mirrors the checkpoint-conformance spec test_prune.py (all eight
// scenarios), plus the Go-only unknown-strategy error.
func testPrune(t *testing.T, newSaver func(t *testing.T) checkpoint.Saver) {
	t.Helper()
	ctx := context.Background()

	// test_prune_keep_latest_single_thread: only the latest survives.
	t.Run("keep_latest_single_thread", func(t *testing.T) {
		s := newSaver(t)
		cfgs := putChain(t, s, "t1", "", 4, nil)
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err != nil {
			t.Fatalf("Prune: %v", err)
		}
		ids := listIDs(t, s, checkpoint.Config{ThreadID: "t1"})
		if len(ids) != 1 || ids[0] != cfgs[3].CheckpointID {
			t.Fatalf("after prune: %v, want only latest %q", ids, cfgs[3].CheckpointID)
		}
	})

	// test_prune_keep_latest_multiple_threads: each thread keeps its latest.
	t.Run("keep_latest_multiple_threads", func(t *testing.T) {
		s := newSaver(t)
		c1 := putChain(t, s, "t1", "", 3, nil)
		c2 := putChain(t, s, "t2", "", 2, nil)
		if err := s.Prune(ctx, []string{"t1", "t2"}, checkpoint.PruneKeepLatest); err != nil {
			t.Fatalf("Prune: %v", err)
		}
		if ids := listIDs(t, s, checkpoint.Config{ThreadID: "t1"}); len(ids) != 1 || ids[0] != c1[2].CheckpointID {
			t.Fatalf("t1 after prune: %v, want [%q]", ids, c1[2].CheckpointID)
		}
		if ids := listIDs(t, s, checkpoint.Config{ThreadID: "t2"}); len(ids) != 1 || ids[0] != c2[1].CheckpointID {
			t.Fatalf("t2 after prune: %v, want [%q]", ids, c2[1].CheckpointID)
		}
	})

	// test_prune_keep_latest_across_namespaces: latest per namespace kept.
	t.Run("keep_latest_across_namespaces", func(t *testing.T) {
		s := newSaver(t)
		root := putChain(t, s, "t1", "", 3, nil)
		child := putChain(t, s, "t1", "child:1", 2, nil)
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err != nil {
			t.Fatalf("Prune: %v", err)
		}
		if ids := listIDs(t, s, checkpoint.Config{ThreadID: "t1"}); len(ids) != 1 || ids[0] != root[2].CheckpointID {
			t.Fatalf("root ns after prune: %v, want [%q]", ids, root[2].CheckpointID)
		}
		if ids := listIDs(t, s, checkpoint.Config{ThreadID: "t1", CheckpointNS: "child:1"}); len(ids) != 1 || ids[0] != child[1].CheckpointID {
			t.Fatalf("child ns after prune: %v, want [%q]", ids, child[1].CheckpointID)
		}
	})

	// test_prune_keep_latest_preserves_writes.
	t.Run("keep_latest_preserves_writes", func(t *testing.T) {
		s := newSaver(t)
		cfgs := putChain(t, s, "t1", "", 3, nil)
		if err := s.PutWrites(ctx, cfgs[2], []checkpoint.Write{{Channel: "ch", Value: "val"}}, "task-1", ""); err != nil {
			t.Fatalf("PutWrites: %v", err)
		}
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err != nil {
			t.Fatalf("Prune: %v", err)
		}
		tup, err := s.GetTuple(ctx, checkpoint.Config{ThreadID: "t1"})
		if err != nil || tup == nil {
			t.Fatalf("GetTuple latest: %v, %v", tup, err)
		}
		if len(tup.PendingWrites) != 1 || tup.PendingWrites[0].Channel != "ch" || tup.PendingWrites[0].Value != "val" {
			t.Fatalf("PendingWrites = %+v, want one ch=val write on the latest", tup.PendingWrites)
		}
	})

	// test_prune_delete_all: the delete strategy removes everything.
	t.Run("delete_all", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "t1", "", 3, nil)
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneDeleteAll); err != nil {
			t.Fatalf("Prune(delete): %v", err)
		}
		if ids := listIDs(t, s, checkpoint.Config{ThreadID: "t1"}); len(ids) != 0 {
			t.Fatalf("after delete-all: %v, want empty", ids)
		}
	})

	// test_prune_preserves_other_threads.
	t.Run("preserves_other_threads", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "t1", "", 3, nil)
		putChain(t, s, "t2", "", 2, nil)
		pre := listIDs(t, s, checkpoint.Config{ThreadID: "t2"})
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneKeepLatest); err != nil {
			t.Fatalf("Prune: %v", err)
		}
		if post := listIDs(t, s, checkpoint.Config{ThreadID: "t2"}); !reflect.DeepEqual(pre, post) {
			t.Fatalf("t2 changed: %v -> %v", pre, post)
		}
	})

	// test_prune_empty_list_noop / _nonexistent_noop.
	t.Run("empty_and_nonexistent_noop", func(t *testing.T) {
		s := newSaver(t)
		if err := s.Prune(ctx, nil, checkpoint.PruneKeepLatest); err != nil {
			t.Fatalf("Prune(nil): %v", err)
		}
		if err := s.Prune(ctx, []string{"no-such-thread"}, checkpoint.PruneKeepLatest); err != nil {
			t.Fatalf("Prune(nonexistent): %v", err)
		}
	})

	// Go-only: an unknown strategy is an error (Python type-checks the
	// Literal; Go must validate at runtime).
	t.Run("unknown_strategy_errors", func(t *testing.T) {
		s := newSaver(t)
		putChain(t, s, "t1", "", 1, nil)
		if err := s.Prune(ctx, []string{"t1"}, checkpoint.PruneStrategy("bogus")); err == nil {
			t.Fatal("Prune(bogus strategy) = nil, want error")
		}
		if ids := listIDs(t, s, checkpoint.Config{ThreadID: "t1"}); len(ids) != 1 {
			t.Fatalf("thread damaged by failed prune: %v", ids)
		}
	})
}
