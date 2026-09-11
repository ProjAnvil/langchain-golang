package graph

import (
	"context"
	"fmt"
	"maps"
	"reflect"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// checkpointSink dispatches checkpoint and per-task writes according to the
// configured Durability mode. It encapsulates the sync/async/exit branching
// that Python spreads across PregelLoop._put_checkpoint / put_writes /
// _suppress_interrupt (_loop.py).
//
// The mode passed to newCheckpointSink is the EFFECTIVE mode for the run: the
// per-run Options.Durability override when set (Python's invoke/stream
// durability argument), else the compiled WithDurability value.
//
// Sync mode: direct saver calls (no goroutine).
// Async mode: a single sequential worker goroutine processes requests in FIFO
// order from a buffered channel. Per-request panic recovery ensures the worker
// survives saver panics. The enqueue path uses a select-with-workerDone guard
// so it never blocks if the worker exits unexpectedly.
// Exit mode: writes are accumulated during the loop and flushed at exit via
// flushExit (implemented in Task 8).
type checkpointSink struct {
	saver checkpoint.Saver
	mode  Durability

	// async mode: single sequential writer goroutine
	bgCtx      context.Context
	writeCh    chan sinkRequest
	workerDone chan struct{}
	workerErr  error // first error (single writer = no race after close)

	// exit mode: accumulator
	exitDeltaWrites    []exitDeltaWrite
	hasPersistedParent bool
	initialCfg         checkpoint.Config
	// exit mode: current superstep (updated by the invoke loop)
	currentStep int
	// exitPauseFlushed records that a pause checkpoint (and its pending
	// writes) were already persisted synchronously by putPauseCheckpoint,
	// so the deferred flushExit must not write anything else: its final
	// checkpoint would reuse the pause checkpoint's ID with an empty Next,
	// clobbering the planned resume tasks.
	exitPauseFlushed bool

	// flush context (set by invoke loop, read by flush/flushExit)
	flushCtx  context.Context
	flushOpts Options
	flushRS   *runState
	flushCfg  checkpoint.Config
	flushMd   checkpoint.Metadata
}

type sinkRequestKind int

const (
	reqPut sinkRequestKind = iota
	reqPutWrites
)

type sinkRequest struct {
	kind        sinkRequestKind
	cfg         checkpoint.Config
	cp          checkpoint.Checkpoint
	md          checkpoint.Metadata
	newVersions map[string]int64
	writes      []checkpoint.Write
	taskID      string
}

type exitDeltaWrite struct {
	step    int
	taskID  string
	channel string
	value   any
}

func (r sinkRequest) execute(saver checkpoint.Saver, ctx context.Context) error {
	switch r.kind {
	case reqPut:
		_, err := saver.Put(ctx, r.cfg, r.cp, r.md, r.newVersions)
		return err
	case reqPutWrites:
		return saver.PutWrites(ctx, r.cfg, r.writes, r.taskID, "")
	default:
		return fmt.Errorf("checkpointSink: unknown request kind %d", r.kind)
	}
}

// newCheckpointSink creates a sink for the given durability mode.
// tup is the loaded checkpoint tuple (nil for fresh threads).
func newCheckpointSink(saver checkpoint.Saver, mode Durability, ctx context.Context, tup *checkpoint.Tuple) *checkpointSink {
	s := &checkpointSink{saver: saver, mode: mode}
	if mode == DurabilityAsync && saver != nil {
		s.bgCtx = context.WithoutCancel(ctx)
		s.writeCh = make(chan sinkRequest, 32)
		s.workerDone = make(chan struct{})
		go s.worker()
	}
	if mode == DurabilityExit && tup != nil {
		s.hasPersistedParent = true
		s.initialCfg = tup.Config
	}
	return s
}

// worker processes requests sequentially from writeCh. Per-request recover
// ensures a single panic does not kill the worker — it continues processing
// remaining requests. The worker exits when writeCh is closed (by flush).
func (s *checkpointSink) worker() {
	defer close(s.workerDone)
	for req := range s.writeCh {
		func() {
			defer func() {
				if r := recover(); r != nil {
					if s.workerErr == nil {
						s.workerErr = fmt.Errorf("checkpointSink worker panic: %v", r)
					}
				}
			}()
			if err := req.execute(s.saver, s.bgCtx); err != nil && s.workerErr == nil {
				s.workerErr = err
			}
		}()
	}
}

// putCheckpoint persists (or defers) a checkpoint. Returns the config with the
// checkpoint ID immediately (the ID is generated locally, not by the saver).
func (s *checkpointSink) putCheckpoint(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64) (checkpoint.Config, error) {
	resultCfg := checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS, CheckpointID: cp.ID}
	switch s.mode {
	case DurabilitySync:
		_, err := s.saver.Put(ctx, cfg, cp, md, newVersions)
		return resultCfg, err
	case DurabilityAsync:
		req := sinkRequest{
			kind:        reqPut,
			cfg:         cfg,
			cp:          cloneForSink(cp),
			md:          md,
			newVersions: maps.Clone(newVersions),
		}
		select {
		case s.writeCh <- req:
		case <-s.workerDone:
			return resultCfg, s.workerErr
		}
		return resultCfg, nil
	case DurabilityExit:
		return resultCfg, nil
	}
	return resultCfg, nil
}

// putWrites persists (or accumulates) per-task writes.
func (s *checkpointSink) putWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID string) error {
	switch s.mode {
	case DurabilitySync:
		return s.saver.PutWrites(ctx, cfg, writes, taskID, "")
	case DurabilityAsync:
		req := sinkRequest{kind: reqPutWrites, cfg: cfg, writes: writes, taskID: taskID}
		select {
		case s.writeCh <- req:
		case <-s.workerDone:
			return s.workerErr
		}
		return nil
	case DurabilityExit:
		s.accumulateExitWrites(writes, taskID)
		return nil
	}
	return nil
}

// putPauseCheckpoint persists a PAUSE checkpoint (interrupt_before /
// interrupt_after / in-node interrupt), which — unlike a regular loop
// checkpoint — must be durable before the run returns, or the thread is not
// resumable at all:
//
//   - sync: identical to putCheckpoint (direct saver call).
//   - async: identical to putCheckpoint (enqueued); FIFO order with the
//     putPauseWrites calls that follow guarantees the checkpoint lands before
//     its pending writes.
//   - exit: writes the checkpoint (plus the run's accumulated per-task delta
//     writes, anchored on the checkpoint itself) DIRECTLY to the saver,
//     bypassing the exit deferral, and marks the sink so the deferred
//     flushExit becomes a no-op. This mirrors Python, where
//     _suppress_interrupt force-calls _put_checkpoint + _put_pending_writes
//     even under durability="exit" (_loop.py:1319-1329).
func (s *checkpointSink) putPauseCheckpoint(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64) (checkpoint.Config, error) {
	if s.mode == DurabilityExit {
		return s.putPauseCheckpointExit(ctx, cfg, cp, md, newVersions)
	}
	return s.putCheckpoint(ctx, cfg, cp, md, newVersions)
}

// putPauseCheckpointExit is the exit-mode branch of putPauseCheckpoint. It
// persists the pause checkpoint synchronously and directly, then anchors the
// run's accumulated per-task delta writes (which exit mode normally defers to
// flushExit) on that checkpoint, under their step-prefixed synthetic task IDs
// — the same reconstruction the ancestor-write walk performs for flushExit's
// writes, which also collects the starting tuple's pending writes
// (snapshot.go's reconstructDeltaChannels).
//
// The pause checkpoint's ParentConfig is rewritten to the last PERSISTED
// checkpoint (initialCfg, when one exists): the in-memory parent chain (this
// run's input and loop checkpoints) was never persisted under exit mode, so
// linking to it would strand the ancestor walk. Unlike flushExit, no stub
// anchor is created: writes land on the pause checkpoint itself, which avoids
// minting any ID after the pause checkpoint's (a stub minted in a later
// millisecond would sort AFTER the pause checkpoint and shadow it as the
// namespace's latest).
func (s *checkpointSink) putPauseCheckpointExit(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64) (checkpoint.Config, error) {
	parentCfg := checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS}
	if s.hasPersistedParent {
		parentCfg.CheckpointID = s.initialCfg.CheckpointID
	}
	resultCfg, err := s.saver.Put(ctx, parentCfg, cp, md, newVersions)
	if err != nil {
		return checkpoint.Config{}, err
	}

	// Delta channels whose cadence fired at this save embed a snapshot blob in
	// the pause checkpoint itself (saveCheckpoint already advanced the
	// counters and computed the set); accumulated writes for them are
	// redundant — the blob seeds reconstruction and the writes would
	// double-apply. Everything else must be materialized now, since the
	// deferred flush below is suppressed.
	for _, w := range s.exitDeltaWrites {
		if _, ok := channels.UnwrapDeltaSnapshot(cp.ChannelValues[w.channel]); ok {
			continue
		}
		synthTID := exitDeltaTaskID(w.step, w.taskID)
		if err := s.saver.PutWrites(ctx, resultCfg, []checkpoint.Write{{Channel: w.channel, Value: w.value}}, synthTID, ""); err != nil {
			return checkpoint.Config{}, err
		}
	}

	// The pause state is now complete and durable. Record it in the sink:
	// exitPauseFlushed suppresses the deferred flushExit (whose final
	// checkpoint would overwrite the pause checkpoint's ID with an empty
	// Next), and hasPersistedParent/initialCfg keep the sink self-consistent
	// were any further write to occur.
	s.exitPauseFlushed = true
	s.hasPersistedParent = true
	s.initialCfg = resultCfg
	return resultCfg, nil
}

// putPauseWrites persists the pending writes of a pause (interrupt records,
// consumed-resume prefixes, and completed-sibling writes of the interrupted
// superstep) against the pause checkpoint identified by cfg. Like
// putPauseCheckpoint it must not be deferred: sync persists directly, async
// enqueues on the same FIFO channel as the pause checkpoint (which therefore
// lands first), and exit writes directly to the saver. A nil/empty writes
// slice is a no-op so callers need not pre-check.
func (s *checkpointSink) putPauseWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID string) error {
	if len(writes) == 0 {
		return nil
	}
	if s.mode == DurabilityExit {
		return s.saver.PutWrites(ctx, cfg, writes, taskID, "")
	}
	return s.putWrites(ctx, cfg, writes, taskID)
}

// setFlushContext stores the context needed by flushExit. Called by the invoke
// loop so that flush() remains parameterless (amendment C7).
func (s *checkpointSink) setFlushContext(ctx context.Context, opts Options, rs *runState, currentCfg checkpoint.Config, md checkpoint.Metadata) {
	s.flushCtx = ctx
	s.flushOpts = opts
	s.flushRS = rs
	s.flushCfg = currentCfg
	s.flushMd = md
}

// flush waits for all pending writes to complete and surfaces any error.
func (s *checkpointSink) flush() error {
	if s.mode == DurabilityExit {
		return s.flushExit()
	}
	if s.writeCh != nil {
		close(s.writeCh)
		<-s.workerDone
		s.writeCh = nil // prevent double-flush; safe now that worker has exited
	}
	return s.workerErr
}

// cloneForSink creates a defensive shallow copy of checkpoint data for the
// background goroutine. Slice values get new backing arrays; map values get
// new maps. Does NOT modify the shared channels.cloneValue.
func cloneForSink(cp checkpoint.Checkpoint) checkpoint.Checkpoint {
	cp.ChannelValues = cloneMapShallow(cp.ChannelValues)
	cp.ChannelVersions = maps.Clone(cp.ChannelVersions)
	return cp
}

// cloneMapShallow creates a new map with shallow-copied values (new backing
// arrays for slices, new maps for maps).
func cloneMapShallow(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = cloneAnyShallow(v)
	}
	return out
}

// cloneAnyShallow returns a shallow defensive copy of v: new backing array for
// slices, new map for maps. Other types are returned as-is (value types are
// safe to share).
func cloneAnyShallow(v any) any {
	if v == nil {
		return nil
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Slice:
		out := reflect.MakeSlice(rv.Type(), rv.Len(), rv.Len()) // Len not Cap — new backing array
		reflect.Copy(out, rv)
		return out.Interface()
	case reflect.Map:
		out := reflect.MakeMapWithSize(rv.Type(), rv.Len())
		for _, k := range rv.MapKeys() {
			out.SetMapIndex(k, rv.MapIndex(k))
		}
		return out.Interface()
	default:
		return v
	}
}

// --- exit mode (stubs — implemented in Task 8) ---

func (s *checkpointSink) accumulateExitWrites(writes []checkpoint.Write, taskID string) {
	for _, w := range writes {
		s.exitDeltaWrites = append(s.exitDeltaWrites, exitDeltaWrite{
			step:    s.currentStep,
			taskID:  taskID,
			channel: w.Channel,
			value:   w.Value,
		})
	}
}

func (s *checkpointSink) flushExit() error {
	if s.saver == nil || s.flushRS == nil {
		return nil
	}
	// A pause already persisted its definitive state synchronously
	// (putPauseCheckpointExit): there is nothing left to flush, and the final
	// checkpoint this method would write reuses the pause checkpoint's ID
	// with an empty Next — clobbering the resume plan.
	if s.exitPauseFlushed {
		return nil
	}

	// 1. Compute channelsToSnapshot from current counters
	newCounters := advanceDeltaCounters(s.flushRS.channels, s.flushRS.deltaCounters, s.flushRS.updatedChannels)
	channelsToSnapshot := channels.DeltaChannelsToSnapshot(s.flushRS.channels, newCounters)
	for k := range s.flushRS.deltaOverwriteChs {
		channelsToSnapshot[k] = true
	}

	// 2. Filter exit delta writes — exclude channels that will snapshot
	pending := make([]exitDeltaWrite, 0)
	for _, w := range s.exitDeltaWrites {
		if !channelsToSnapshot[w.channel] {
			pending = append(pending, w)
		}
	}

	// 3. Anchor selection (mirrors Python _loop.py:1253-1281).
	// When a parent checkpoint already exists, accumulated delta writes must
	// anchor on that persisted parent (initialCfg) — NOT on the final
	// checkpoint (flushCfg), which is only persisted in step 5 below.
	// PutWrites against the not-yet-persisted flushCfg errors ("no checkpoint"),
	// aborting flushExit before the final checkpoint is saved.
	anchorCfg := s.flushCfg
	if s.hasPersistedParent {
		anchorCfg = s.initialCfg
	} else if len(pending) > 0 {
		stubCp := checkpoint.Checkpoint{
			V:               1,
			ID:              checkpoint.NewID(0),
			ChannelValues:   map[string]any{},
			ChannelVersions: map[string]int64{},
			VersionsSeen:    map[string]map[string]int64{},
		}
		stubCfg := checkpoint.Config{ThreadID: s.flushOpts.ThreadID, CheckpointNS: s.flushOpts.checkpointNS}
		if _, err := s.saver.Put(s.flushCtx, stubCfg, stubCp, checkpoint.Metadata{Source: "input", Step: -2}, nil); err != nil {
			return err
		}
		anchorCfg = checkpoint.Config{ThreadID: s.flushOpts.ThreadID, CheckpointNS: s.flushOpts.checkpointNS, CheckpointID: stubCp.ID}
	}

	// 4. Persist accumulated delta writes with step-prefixed task IDs
	for _, w := range pending {
		synthTID := exitDeltaTaskID(w.step, w.taskID)
		if err := s.saver.PutWrites(s.flushCtx, anchorCfg, []checkpoint.Write{{Channel: w.channel, Value: w.value}}, synthTID, ""); err != nil {
			return err
		}
	}

	// 5. Save final checkpoint from current rs state
	for k := range channelsToSnapshot {
		newCounters[k] = [2]int{0, 0}
	}
	s.flushRS.deltaCounters = newCounters
	s.flushMd.CountersSinceDeltaSnapshot = nonZeroCounters(newCounters)

	finalCp := checkpoint.Checkpoint{
		V:               1,
		ID:              s.flushCfg.CheckpointID,
		ChannelValues:   s.flushRS.checkpointValues(channelsToSnapshot),
		ChannelVersions: maps.Clone(s.flushRS.versions),
		VersionsSeen:    cloneSeen(s.flushRS.seen),
	}
	parentCfg := checkpoint.Config{ThreadID: s.flushOpts.ThreadID, CheckpointNS: s.flushOpts.checkpointNS}
	if s.hasPersistedParent {
		parentCfg.CheckpointID = s.initialCfg.CheckpointID
	} else if anchorCfg.CheckpointID != "" && anchorCfg.CheckpointID != s.flushCfg.CheckpointID {
		parentCfg.CheckpointID = anchorCfg.CheckpointID
	}
	if _, err := s.saver.Put(s.flushCtx, parentCfg, finalCp, s.flushMd, nil); err != nil {
		return err
	}

	return nil
}
