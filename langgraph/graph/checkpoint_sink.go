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

	// 3. Anchor selection (mirrors Python _loop.py:1253-1281)
	anchorCfg := s.flushCfg
	if !s.hasPersistedParent && len(pending) > 0 {
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
