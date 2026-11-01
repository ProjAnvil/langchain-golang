package graph

import (
	"context"
	"errors"
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
// #9235: a DeltaChannel is rebuilt from its writes along the parent chain, so
// a checkpoint saved without one of them reads back short for good. The sink
// therefore never saves past a failure: the first saver error (a checkpoint
// save or a delta-channel write) poisons the sink — every later save and
// write of the run returns the recorded error without touching the saver —
// and a failed delta write additionally "repairs" the checkpoint it anchors
// on by re-putting it with full snapshot blobs of every available delta
// channel, so the half-written checkpoint carries the complete value instead
// of relying on the write that never landed. Python ties each save to the
// delta-write futures submitted before it and fails the save when one fails;
// Go's per-task writes must anchor on the checkpoint they are persisted
// after (every saver requires the checkpoint row to exist), so writes cannot
// precede their checkpoint's save — the repair re-put is the Go form of the
// same invariant.
//
// The mode passed to newCheckpointSink is the EFFECTIVE mode for the run: the
// per-run Options.Durability override when set (Python's invoke/stream
// durability argument), else the compiled WithDurability value.
//
// Sync mode: direct saver calls (no goroutine).
// Async mode: a single sequential worker goroutine processes requests in FIFO
// order from a buffered channel; the worker STOPS at the first saver error
// (nothing later in the run is saved past it) after repairing the checkpoint
// a failed delta write anchored on. Per-request panic recovery ensures a
// panic is recorded as the worker error. The enqueue path uses a
// select-with-workerDone guard so it never blocks if the worker exited.
// Exit mode: writes are accumulated during the loop and flushed at exit via
// flushExit (implemented in Task 8).
type checkpointSink struct {
	saver checkpoint.Saver
	mode  Durability

	// poisoned is the first saver failure of the run (#9235): once set, every
	// later checkpoint save and write of the run fails with it instead of
	// persisting — later checkpoints must never be saved past a failed write
	// or on top of a checkpoint that was never saved (Python: a save fails if
	// a write or the previous save failed).
	poisoned error

	// async mode: single sequential writer goroutine
	bgCtx      context.Context
	writeCh    chan sinkRequest
	workerDone chan struct{}
	workerErr  error // first error (single writer = no race after close)

	// lastPut records the most recent successful checkpoint put (sync mode),
	// held so a failed delta write anchored on it can repair it (#9235).
	lastPut *putRecord

	// exit mode: accumulator
	exitDeltaWrites    []exitDeltaWrite
	hasPersistedParent bool
	initialCfg         checkpoint.Config
	// initialValues is the loaded checkpoint's ChannelValues (nil without a
	// persisted parent): the values the flush's write anchor (initialCfg)
	// already stores. A channel stored there must not receive anchored writes
	// — the ancestor walk stops at the stored value, stranding them — so it
	// snapshots into the final checkpoint instead (see flushExit).
	initialValues map[string]any
	exitAnchorID  string
	// exit mode: current superstep (updated by the invoke loop via
	// beginSuperstep; input writes accumulate before the first superstep
	// under step 0, mirroring Python's _first appending input writes at the
	// entry step)
	currentStep int
	// exit mode: the run's FIRST superstep — the anchor superstep whose
	// writes keep their real dispatch paths so they interleave with the
	// writes a resume loaded from the anchor checkpoint exactly as live
	// execution applied them (Python #9114's _exit_first_step).
	exitFirstStep int
	exitFirstSet  bool
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
	taskPath    string
	// delta marks a write batch containing writes to a DeltaChannel (#9235:
	// only delta writes gate the checkpoint they anchor on — plain channels
	// carry their value in ChannelValues, so a lost plain write cannot short
	// a read). Carried per request so the worker can repair without proto
	// knowledge.
	delta bool
	// repair is the forced delta-snapshot payload for the checkpoint being
	// put (nil when the run has no delta channels): map of channel name to
	// SnapshotBlob, computed by saveCheckpoint from live channel state at
	// put time. If one of this checkpoint's delta writes later fails, the
	// checkpoint is re-put with these blobs overlaid (#9235).
	repair map[string]any
}

// putRecord captures one successful checkpoint put so it can be repaired
// (re-put with forced delta snapshots) when a delta write anchored on it
// fails (#9235).
type putRecord struct {
	cfg         checkpoint.Config
	cp          checkpoint.Checkpoint
	md          checkpoint.Metadata
	newVersions map[string]int64
	repair      map[string]any
}

type exitDeltaWrite struct {
	step     int
	taskID   string
	taskPath string
	channel  string
	value    any
}

func (r sinkRequest) execute(saver checkpoint.Saver, ctx context.Context) error {
	switch r.kind {
	case reqPut:
		_, err := saver.Put(ctx, r.cfg, r.cp, r.md, r.newVersions)
		return err
	case reqPutWrites:
		return saver.PutWrites(ctx, r.cfg, r.writes, r.taskID, r.taskPath)
	default:
		return fmt.Errorf("checkpointSink: unknown request kind %d", r.kind)
	}
}

// repairPut re-puts rec's checkpoint with rec.repair's forced delta snapshots
// overlaid into ChannelValues (#9235). Savers treat a re-put of an existing
// checkpoint ID as a row replace (MemorySaver keeps the checkpoint's pending
// writes; sqlite/postgres/redis store them beside the row), so the repair
// only changes the checkpoint blob: a delta channel whose write failed is
// materialized as a snapshot, and the checkpoint never reads short. A nil
// repair payload (no delta channels) makes the call a no-op.
func (s *checkpointSink) repairPut(ctx context.Context, rec *putRecord) error {
	if rec == nil || len(rec.repair) == 0 {
		return nil
	}
	cp := rec.cp
	cp.ChannelValues = maps.Clone(cp.ChannelValues)
	if cp.ChannelValues == nil {
		cp.ChannelValues = map[string]any{}
	}
	maps.Copy(cp.ChannelValues, rec.repair)
	_, err := s.saver.Put(ctx, rec.cfg, cp, rec.md, rec.newVersions)
	return err
}

// poison records the run's first saver failure: every later save and write of
// the run stops (#9235). The first error wins so the caller surfaces the root
// cause, not a follow-on failure.
func (s *checkpointSink) poison(err error) {
	if s.poisoned == nil {
		s.poisoned = err
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
	if mode == DurabilityExit {
		if tup != nil {
			s.hasPersistedParent = true
			s.initialCfg = tup.Config
			s.initialValues = tup.Checkpoint.ChannelValues
		} else if saver != nil {
			// Reserve before loop checkpoint IDs so a later flush cannot
			// create an anchor that shadows the final checkpoint as latest.
			s.exitAnchorID = checkpoint.NewID(int(checkpointSeq.Add(1)))
		}
	}
	return s
}

// worker processes requests sequentially from writeCh. It stops at the first
// saver error (#9235): after repairing the checkpoint a failed delta write
// anchored on, the worker returns, so nothing later in the run is saved past
// the failure — later checkpoints would otherwise be saved without their
// writes or on top of a checkpoint that was never saved. The worker exits
// when writeCh is closed (by flush); pending requests left in the channel are
// dropped. Per-request recover records a panic as the worker error.
func (s *checkpointSink) worker() {
	defer close(s.workerDone)
	var lastPut *putRecord
	for req := range s.writeCh {
		err := func() (err error) {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("checkpointSink worker panic: %v", r)
				}
			}()
			return req.execute(s.saver, s.bgCtx)
		}()
		if err == nil {
			if req.kind == reqPut {
				// Copy the request: req is reused per iteration.
				r := req
				lastPut = &putRecord{cfg: r.cfg, cp: r.cp, md: r.md, newVersions: r.newVersions, repair: r.repair}
			}
			continue
		}
		s.workerErr = err
		if req.kind == reqPutWrites && req.delta && lastPut != nil && lastPut.cp.ID == req.cfg.CheckpointID {
			if rerr := s.repairPut(s.bgCtx, lastPut); rerr != nil {
				s.workerErr = errors.Join(err, fmt.Errorf("repairing checkpoint %q after a failed write: %w", lastPut.cp.ID, rerr))
			}
		}
		return
	}
}

// putCheckpoint persists (or defers) a checkpoint. Returns the config with the
// checkpoint ID immediately (the ID is generated locally, not by the saver).
// repair is the checkpoint's forced delta-snapshot payload (see sinkRequest),
// used only when one of its delta writes later fails (#9235).
func (s *checkpointSink) putCheckpoint(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64, repair map[string]any) (checkpoint.Config, error) {
	resultCfg := checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS, CheckpointID: cp.ID}
	switch s.mode {
	case DurabilitySync:
		if s.poisoned != nil {
			return resultCfg, s.poisoned
		}
		if _, err := s.saver.Put(ctx, cfg, cp, md, newVersions); err != nil {
			s.poison(err) // #9235: nothing later in the run is saved
			return resultCfg, err
		}
		s.lastPut = &putRecord{cfg: cfg, cp: cloneForSink(cp), md: md, newVersions: maps.Clone(newVersions), repair: repair}
		return resultCfg, nil
	case DurabilityAsync:
		req := sinkRequest{
			kind:        reqPut,
			cfg:         cfg,
			cp:          cloneForSink(cp),
			md:          md,
			newVersions: maps.Clone(newVersions),
			repair:      repair,
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

// putWrites persists (or accumulates) per-task writes, stamped with the
// task's dispatch path so savers replay them in live order (Python #8544).
// A "" taskPath (graph input, update_state writes, synthetic boundary
// interrupts) is pathless and sorts before every task's writes. delta marks
// a batch containing DeltaChannel writes: on failure the sink repairs the
// anchoring checkpoint with full snapshots and poisons the run (#9235).
func (s *checkpointSink) putWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string, delta bool) error {
	switch s.mode {
	case DurabilitySync:
		if s.poisoned != nil {
			return s.poisoned
		}
		if err := s.saver.PutWrites(ctx, cfg, writes, taskID, taskPath); err != nil {
			s.poison(err)
			if delta && s.lastPut != nil && s.lastPut.cp.ID == cfg.CheckpointID {
				if rerr := s.repairPut(ctx, s.lastPut); rerr != nil {
					return errors.Join(err, fmt.Errorf("repairing checkpoint %q after a failed write: %w", cfg.CheckpointID, rerr))
				}
			}
			return err
		}
		return nil
	case DurabilityAsync:
		req := sinkRequest{kind: reqPutWrites, cfg: cfg, writes: writes, taskID: taskID, taskPath: taskPath, delta: delta}
		select {
		case s.writeCh <- req:
		case <-s.workerDone:
			return s.workerErr
		}
		return nil
	case DurabilityExit:
		s.accumulateExitWrites(writes, taskID, taskPath)
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
//
// repair is the pause checkpoint's forced delta-snapshot payload (#9235): a
// failed accumulated delta write repairs the pause checkpoint the same way a
// failed task write repairs a loop checkpoint.
func (s *checkpointSink) putPauseCheckpoint(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64, repair map[string]any) (checkpoint.Config, error) {
	if s.mode == DurabilityExit {
		return s.putPauseCheckpointExit(ctx, cfg, cp, md, newVersions, repair)
	}
	return s.putCheckpoint(ctx, cfg, cp, md, newVersions, repair)
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
//
// #9235: a failed accumulated write repairs the pause checkpoint with full
// delta snapshots (a pause checkpoint saved without one of its writes would
// read short for good, exactly like a loop checkpoint), and any failure
// poisons the sink so the deferred flushExit writes nothing more.
func (s *checkpointSink) putPauseCheckpointExit(ctx context.Context, cfg checkpoint.Config, cp checkpoint.Checkpoint, md checkpoint.Metadata, newVersions map[string]int64, repair map[string]any) (checkpoint.Config, error) {
	if s.poisoned != nil {
		return checkpoint.Config{}, s.poisoned
	}
	parentCfg := checkpoint.Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS}
	if s.hasPersistedParent {
		parentCfg.CheckpointID = s.initialCfg.CheckpointID
	}
	resultCfg, err := s.saver.Put(ctx, parentCfg, cp, md, newVersions)
	if err != nil {
		s.poison(err)
		return checkpoint.Config{}, err
	}
	lastPut := &putRecord{cfg: parentCfg, cp: cloneForSink(cp), md: md, newVersions: maps.Clone(newVersions), repair: repair}

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
		tid, path := s.exitPauseWriteKey(w)
		if err := s.saver.PutWrites(ctx, resultCfg, []checkpoint.Write{{Channel: w.channel, Value: w.value}}, tid, path); err != nil {
			s.poison(err)
			if rerr := s.repairPut(ctx, lastPut); rerr != nil {
				return checkpoint.Config{}, errors.Join(err, fmt.Errorf("repairing pause checkpoint %q after a failed write: %w", resultCfg.CheckpointID, rerr))
			}
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
	s.initialValues = cp.ChannelValues
	return resultCfg, nil
}

// putPauseWrites persists the pending writes of a pause (interrupt records,
// consumed-resume prefixes, and completed-sibling writes of the interrupted
// superstep) against the pause checkpoint identified by cfg, stamped with
// the task's dispatch path (live replay order, Python #8544). Like
// putPauseCheckpoint it must not be deferred: sync persists directly, async
// enqueues on the same FIFO channel as the pause checkpoint (which therefore
// lands first), and exit writes directly to the saver. A nil/empty writes
// slice is a no-op so callers need not pre-check. delta marks a batch with
// DeltaChannel writes (repair/poison gating, #9235).
func (s *checkpointSink) putPauseWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string, delta bool) error {
	if len(writes) == 0 {
		return nil
	}
	if s.mode == DurabilityExit {
		if s.poisoned != nil {
			return s.poisoned
		}
		if err := s.saver.PutWrites(ctx, cfg, writes, taskID, taskPath); err != nil {
			s.poison(err)
			return err
		}
		return nil
	}
	return s.putWrites(ctx, cfg, writes, taskID, taskPath, delta)
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

// beginSuperstep records the superstep the executor is entering, latching
// the run's FIRST superstep as the anchor superstep (Python #9114's
// _exit_first_step): its accumulated writes keep their real dispatch paths
// so they interleave with the writes a resume loaded from the anchor
// checkpoint; every later superstep's writes sort after all real task ids
// and paths, in step order. No-op outside exit durability.
func (s *checkpointSink) beginSuperstep(step int) {
	if s.mode != DurabilityExit {
		return
	}
	if !s.exitFirstSet {
		s.exitFirstSet = true
		s.exitFirstStep = step
	}
	s.currentStep = step
}

// exitWriteKey derives the (task ID, task path) pair an accumulated exit
// write is stored under when the run COMPLETES and flushExit anchors it on
// the run's starting checkpoint — Python #9114's _put_exit_delta_writes
// grouping (_loop.py):
//
//   - input writes (NullTaskID) replay as the anchor checkpoint's own state:
//     pathless, sorting before every task write;
//   - the anchor superstep's writes keep their real dispatch paths — they
//     interleave with the writes loaded from the anchor checkpoint exactly
//     as live execution applied them — under step-prefixed synthetic ids
//     (synthetic, never the real id, so a run whose final checkpoint fails
//     to save does not leave the resumed task looking done);
//   - every later superstep sorts after all real task ids and paths, in
//     step order, so replay preserves chronology whether a saver orders by
//     task path (#8544) or only by task id.
func (s *checkpointSink) exitWriteKey(w exitDeltaWrite) (taskID, taskPath string) {
	switch {
	case w.taskID == checkpoint.NullTaskID:
		return exitDeltaTaskID(w.step, w.taskID), ""
	case w.step == s.exitFirstStep:
		return exitDeltaTaskID(w.step, w.taskID), w.taskPath
	default:
		return exitDeltaLateTaskID(w.step, w.taskID), exitDeltaLatePath(w.step, w.taskPath)
	}
}

// exitPauseWriteKey is the pause-path variant used by
// putPauseCheckpointExit, which anchors the accumulator on the PAUSE
// checkpoint — the same checkpoint that receives the pause superstep's own
// direct writes (real ids, real dispatch paths). Every accumulated write is
// from a superstep strictly BEFORE the pause superstep, so beyond the
// input write (pathless) they all sort before every real dispatch path, in
// step order: a "!"-prefixed step-qualified path, the mirror image of
// exitDeltaLatePath's sort-after "~~" prefix. (Python does not need this:
// its _suppress_interrupt anchors accumulated writes on the run's START
// checkpoint, not the pause checkpoint — a Go divergence kept from the
// v0.11.1 exit-anchor checkpoint-ID reservation work.)
func (s *checkpointSink) exitPauseWriteKey(w exitDeltaWrite) (taskID, taskPath string) {
	if w.taskID == checkpoint.NullTaskID {
		return exitDeltaTaskID(w.step, w.taskID), ""
	}
	return exitDeltaTaskID(w.step, w.taskID), exitDeltaEarlyPath(w.step, w.taskPath)
}

func (s *checkpointSink) accumulateExitWrites(writes []checkpoint.Write, taskID, taskPath string) {
	for _, w := range writes {
		s.exitDeltaWrites = append(s.exitDeltaWrites, exitDeltaWrite{
			step:     s.currentStep,
			taskID:   taskID,
			taskPath: taskPath,
			channel:  w.Channel,
			value:    w.Value,
		})
	}
}

func (s *checkpointSink) flushExit() error {
	if s.saver == nil || s.flushRS == nil {
		return nil
	}
	// #9235: a saver failure already poisoned the run (a direct put in the
	// pause path, or nothing — exit mode defers everything else to here).
	// Nothing more may be saved: the final checkpoint would land past the
	// failed write or on top of a checkpoint that was never saved.
	if s.poisoned != nil {
		return s.poisoned
	}
	// A pause already persisted its definitive state synchronously
	// (putPauseCheckpointExit): there is nothing left to flush, and the final
	// checkpoint this method would write reuses the pause checkpoint's ID
	// with an empty Next — clobbering the resume plan.
	if s.exitPauseFlushed {
		return nil
	}

	// 1. Compute channelsToSnapshot from the counters the (deferred)
	// saveCheckpoint calls already advanced — flushExit must NOT advance
	// again: exit mode persists exactly one final checkpoint, and its
	// counters must match what a sync run of the same graph produces.
	newCounters := maps.Clone(s.flushRS.deltaCounters)
	if newCounters == nil {
		// No saveCheckpoint advanced the counters yet (e.g. a flush context
		// assembled without a loop save); start from empty so the snapshot
		// resets below never write into a nil map.
		newCounters = map[string][2]int{}
	}
	channelsToSnapshot := channels.DeltaChannelsToSnapshot(s.flushRS.channels, newCounters)
	for k := range s.flushRS.deltaOverwriteChs {
		channelsToSnapshot[k] = true
	}
	// #9269: a checkpoint an explicit Options.CheckpointID addressed may
	// already have children — the thread grew a newer branch past it. Writes
	// anchored on that shared checkpoint replay into EVERY branch's ancestor
	// walk, so the newer branch would read this run's delta writes as its
	// own state. A run loaded by an addressed checkpoint_id therefore
	// force-snapshots the delta channels it wrote into its final checkpoint
	// instead of anchoring them on the pinned parent — exactly Python's
	// `_has_persisted_parent and not self._loaded_latest`
	// (_loop.py:_put_exit_delta_writes): `_loaded_latest` is set ONLY on the
	// no-checkpoint-id load path, so an addressed run snapshots even when
	// the pinned checkpoint IS the latest ("costs one snapshot of each
	// channel the run wrote, even when the addressed checkpoint is the
	// latest one") — anchoring on the latest is safe only until the thread
	// forks from it (a later time-travel pin, a concurrent sibling exit
	// run), and those future children replay anchored writes the same way.
	// Go's CheckpointID axis covers Python's subgraph replay_state axis too:
	// a re-entered subgraph loads its recorded parent position via
	// opts.CheckpointID (subgraph.go), the same addressed load. No I/O, no
	// error path: the earlier superseded-probe (latest via GetTuple) was
	// both a narrower gate than Python's and racy.
	if s.hasPersistedParent && s.flushOpts.CheckpointID != "" {
		for _, w := range s.exitDeltaWrites {
			if d, ok := channels.AsDelta(s.flushRS.channels[w.channel]); ok && d.IsAvailable() {
				channelsToSnapshot[w.channel] = true
			}
		}
	}
	// The unaddressed sibling of that gate: the persisted parent this flush
	// anchors on may itself store a value for a channel (a blob an earlier
	// run materialized — e.g. the previous turn's force-snapshot final). The
	// ancestor walk stops at a stored value, so writes anchored beside it
	// would strand: the channel snapshots into the final checkpoint instead
	// (the write's effect is folded into the live state the blob captures).
	// Same shape as #9269's addressed-load gate; both are Python
	// _put_exit_delta_writes' "anchoring is unsafe once the anchor has
	// readers" family. Delta channels only: a plain channel is never omitted
	// from ChannelValues, so it has nothing to strand.
	if s.hasPersistedParent {
		for _, w := range s.exitDeltaWrites {
			if _, stored := s.initialValues[w.channel]; !stored {
				continue
			}
			if d, ok := channels.AsDelta(s.flushRS.channels[w.channel]); ok && d.IsAvailable() {
				channelsToSnapshot[w.channel] = true
			}
		}
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
			ID:              s.exitAnchorID,
			ChannelValues:   map[string]any{},
			ChannelVersions: map[string]int64{},
			VersionsSeen:    map[string]map[string]int64{},
		}
		stubCfg := checkpoint.Config{ThreadID: s.flushOpts.ThreadID, CheckpointNS: s.flushOpts.checkpointNS}
		if _, err := s.saver.Put(s.flushCtx, stubCfg, stubCp, checkpoint.Metadata{Source: "input", Step: -2}, nil); err != nil {
			s.poison(err)
			return err
		}
		anchorCfg = checkpoint.Config{ThreadID: s.flushOpts.ThreadID, CheckpointNS: s.flushOpts.checkpointNS, CheckpointID: stubCp.ID}
	}

	// 4. Persist accumulated delta writes under their exit write keys
	// (real paths for the anchor superstep, sort-after keys for later
	// supersteps — see exitWriteKey). A failure stops the flush BEFORE the
	// final checkpoint is saved (#9235): the thread stays at its last
	// checkpoint that has all its writes.
	for _, w := range pending {
		tid, path := s.exitWriteKey(w)
		if err := s.saver.PutWrites(s.flushCtx, anchorCfg, []checkpoint.Write{{Channel: w.channel, Value: w.value}}, tid, path); err != nil {
			s.poison(err)
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
