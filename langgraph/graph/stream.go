package graph

import (
	"context"
	"fmt"
	"iter"
	"sync/atomic"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Stream API for CompiledGraph: Python-parity stream modes (S1-S3 of the M3
// plan) delivered as a Go 1.23 iterator over an emission layer inside the
// run loop.
//
// Stream coexists with InvokeStream/NodeEventSink: InvokeStream is the
// event-ified node-lifecycle path used by langchain/agents' StreamEvents;
// Stream is the general langgraph stream-modes surface (values/updates/
// debug/messages/custom). Neither replaces the other.

// StreamMode selects which chunk families Stream yields, mirroring Python's
// `stream_mode` argument.
type StreamMode string

const (
	// StreamValues yields the full graph state after the input batch and
	// after each superstep commit that changed at least one channel.
	StreamValues StreamMode = "values"
	// StreamUpdates yields each task's state writes as they are collected.
	StreamUpdates StreamMode = "updates"
	// StreamDebug yields task dispatch/completion and checkpoint events.
	StreamDebug StreamMode = "debug"
	// StreamMessages yields per-token LLM message chunks (MessageChunk
	// payloads). The executor installs a callbacks.Manager into each node's
	// context; node code opts in by pulling it with
	// callbacks.ManagerFromContext and fanning it into its model configs
	// (runnables.WithCallbacks) — see stream_messages.go.
	StreamMessages StreamMode = "messages"
	// StreamCustom yields node-emitted custom payloads: whatever the node
	// passes to the StreamWriter it obtained via StreamWriterFromContext.
	StreamCustom StreamMode = "custom"
)

// StreamChunk is one streamed unit. Unlike Python, which reshapes output by
// mode count (bare payload vs tuples), Go always carries Mode and Namespace
// explicitly — the type system makes Python's shape-shifting unnecessary
// (documented simplification, S3).
type StreamChunk struct {
	// Namespace is "" for the root graph; for subgraph chunks it is the
	// subgraph node path joined by "/" (derived from the node path threaded
	// through invokeSubgraph, NOT from checkpoint config, so it works without
	// a checkpointer; when checkpointing is active it coincides with the
	// checkpoint namespace).
	Namespace string
	// Mode identifies which stream mode produced this chunk.
	Mode StreamMode
	// Payload is mode-dependent:
	//
	//   - values: map[string]any, the full state snapshot; on pause it
	//     includes the "__interrupt__" key ([]types.Interrupt).
	//   - updates: map[string]any{nodeName: map[string]any{channel: value}}
	//     per task; the interrupt chunk is
	//     map[string]any{"__interrupt__": []types.Interrupt{...}}.
	//   - debug: map[string]any{"step": int, "timestamp": string,
	//     "type": "task"|"task_result"|"checkpoint", "payload": ...} (see
	//     debugPayload).
	//   - messages: MessageChunk — a streamed message chunk plus the node's
	//     langgraph_node/langgraph_step/langgraph_checkpoint_ns metadata.
	//   - custom: whatever payload the node passed to its StreamWriter.
	Payload any
}

// StreamOptions configures a Stream call. The embedded Options keep their
// Invoke semantics (ThreadID/CheckpointID/Resume) unchanged.
type StreamOptions struct {
	Options
	// Modes selects the emitted stream modes. Required, non-empty; order is
	// irrelevant (emission follows run chronology, not this list).
	Modes []StreamMode
	// Subgraphs includes subgraph chunks (carrying their Namespace) when
	// true; when false, chunks emitted inside subgraph runs are dropped.
	Subgraphs bool
}

// Stream runs the graph like InvokeWithOptions and yields chunks as they are
// emitted. The iterator is single-use; the run executes on a goroutine and
// chunks are delivered in emission order. A run failure is yielded as the
// final (zero-chunk, error) pair after all prior chunks.
//
// Breaking out of the iteration early cancels the run: the iterator does not
// return until the run goroutine has exited, so no goroutine leaks. (The run
// observes cancellation at the next superstep boundary at the latest; nodes
// that honor their context abort sooner.)
//
// Chronology note: updates chunks are emitted post-superstep in deterministic
// task order (Go applies writes only after all tasks of a superstep
// complete), so in multi-mode streams they bunch after node-time events
// (messages/custom chunks) instead of interleaving as in Python's
// as-they-finish timing — a documented divergence.
func (g *CompiledGraph) Stream(ctx context.Context, input map[string]any, opts StreamOptions) iter.Seq2[StreamChunk, error] {
	return func(yield func(StreamChunk, error) bool) {
		if len(opts.Modes) == 0 {
			yield(StreamChunk{}, fmt.Errorf("graph: StreamOptions.Modes must be non-empty"))
			return
		}
		for _, mode := range opts.Modes {
			switch mode {
			case StreamValues, StreamUpdates, StreamDebug, StreamMessages, StreamCustom:
			default:
				yield(StreamChunk{}, fmt.Errorf("graph: unknown StreamMode %q", mode))
				return
			}
		}

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		chunks := make(chan StreamChunk)
		done := make(chan struct{})
		var runErr error
		emitter := newStreamEmitter(ctx, opts.Modes, opts.Subgraphs, chunks)
		go func() {
			// LIFO: the closed flag is stored before close(chunks), which runs
			// before close(done), and both run after runErr is assigned, so the
			// consumer safely reads runErr once the channel closes. The flag
			// makes emit a no-op once the delivery channel is gone.
			defer close(done)
			defer close(chunks)
			defer emitter.closed.Store(true)
			_, err := g.run(contextWithEmitter(ctx, emitter), input, opts.Options, nil)
			runErr = topLevelParentCommandError(err)
		}()

		for c := range chunks {
			if !yield(c, nil) {
				// Early break: cancel the run and wait for its goroutine to
				// exit so no goroutine outlives the iterator.
				cancel()
				<-done
				return
			}
		}
		if runErr != nil {
			yield(StreamChunk{}, runErr)
		}
	}
}

// streamEmitter is the per-run emission layer: the run loop calls its hooks
// at the input batch, task dispatch/collection, superstep commit, pause, and
// checkpoint save points. A nil *streamEmitter is valid and every hook is a
// no-op, keeping the non-streaming Invoke/InvokeStream paths at zero added
// overhead.
type streamEmitter struct {
	ctx       context.Context
	send      chan<- StreamChunk
	modes     map[StreamMode]bool
	subgraphs bool
	// closed is set just before the run's delivery channel closes; shared
	// with child emitters so an emit after the run ends (e.g. a StreamWriter
	// invoked post-return) is a no-op instead of panicking on the closed
	// channel.
	closed *atomic.Bool
	// ns is this run's emission namespace: "" for the root graph, the node
	// path ("a", "a/b") for subgraph runs.
	ns string
}

func newStreamEmitter(ctx context.Context, modes []StreamMode, subgraphs bool, send chan<- StreamChunk) *streamEmitter {
	m := make(map[StreamMode]bool, len(modes))
	for _, mode := range modes {
		m[mode] = true
	}
	return &streamEmitter{ctx: ctx, send: send, modes: m, subgraphs: subgraphs, closed: &atomic.Bool{}}
}

// streamEmitterKey is the context-value key under which the active emitter
// travels, so invokeSubgraph can derive (or strip) the child run's emitter.
type streamEmitterKey struct{}

// contextWithEmitter installs em (possibly nil, which strips any inherited
// emitter) into ctx.
func contextWithEmitter(ctx context.Context, em *streamEmitter) context.Context {
	return context.WithValue(ctx, streamEmitterKey{}, em)
}

// emitterFromContext returns the emitter installed by Stream (or a parent
// run's invokeSubgraph), or nil when streaming is not active.
func emitterFromContext(ctx context.Context) *streamEmitter {
	em, _ := ctx.Value(streamEmitterKey{}).(*streamEmitter)
	return em
}

// child returns the emitter for a subgraph run: same delivery channel and
// modes, namespace extended with the subgraph node name.
func (e *streamEmitter) child(name string) *streamEmitter {
	return &streamEmitter{
		ctx:       e.ctx,
		send:      e.send,
		modes:     e.modes,
		subgraphs: e.subgraphs,
		closed:    e.closed,
		ns:        joinCheckpointNS(e.ns, name),
	}
}

// emit delivers one chunk for mode, unless the mode is inactive, the run has
// been cancelled (early iterator break), or the run has ended.
func (e *streamEmitter) emit(mode StreamMode, payload any) {
	if e == nil || !e.modes[mode] || e.closed.Load() {
		return
	}
	// A send racing the run goroutine's close would panic; drop the chunk
	// instead — post-run emission is a no-op by contract.
	defer func() { _ = recover() }()
	select {
	case e.send <- StreamChunk{Namespace: e.ns, Mode: mode, Payload: payload}:
	case <-e.ctx.Done():
	}
}

// nodeContext derives the per-task context for one node invocation,
// installing the affordances of the active stream modes: a callbacks.Manager
// (discoverable via callbacks.ManagerFromContext) bridging model events to
// messages chunks when StreamMessages is active, and a StreamWriter
// (discoverable via StreamWriterFromContext) when StreamCustom is active.
// step is the superstep the task runs in (rs.step+1 at dispatch). When
// neither mode is active — including every non-streaming run, where the
// emitter itself is nil — ctx is returned unchanged (zero overhead).
//
// Manager interplay: the installed manager OVERWRITES any manager the caller
// installed into the run's context (e.g. via callbacks.ContextWithManager
// before Stream), because the bridge must own the handler set to attach the
// node/step/namespace metadata. Such a caller-installed manager is still
// visible to non-node code and to modes that do not install one; to compose
// user handlers with messages mode, add them inside the node — derive from
// the installed manager (callbacks.ManagerFromContext) and wrap it, e.g.
// callbacks.NewManager(append([]callbacks.Handler{myHandler}, manager)...).
func (e *streamEmitter) nodeContext(ctx context.Context, node string, step int) context.Context {
	if e == nil {
		return ctx
	}
	if e.modes[StreamMessages] {
		ctx = callbacks.ContextWithManager(ctx, callbacks.NewManager(newMessagesBridge(e, node, step)))
	}
	if e.modes[StreamCustom] {
		ctx = contextWithStreamWriter(ctx, func(payload any) {
			e.emit(StreamCustom, payload)
		})
	}
	return ctx
}

// stripStreamCarriers derives the child-run context for a subgraph the stream
// did not request (StreamOptions.Subgraphs false): the emitter is removed and
// the per-node carriers the PARENT's nodeContext installed for the subgraph
// node — the messages-bridge manager and the StreamWriter — are shadowed with
// inert values, so inner nodes see no live carrier and any messages/custom
// emissions they attempt are dropped rather than delivered under the root
// namespace. The zero manager satisfies ManagerFromContext's ok while having
// no handlers (Emit drops events); the nil writer fails the nil-check nodes
// must perform. Must not be used on the Subgraphs:true path, where
// nodeContext re-installs live carriers with the child namespace.
func stripStreamCarriers(ctx context.Context) context.Context {
	ctx = contextWithEmitter(ctx, nil)
	ctx = callbacks.ContextWithManager(ctx, callbacks.Manager{})
	ctx = contextWithStreamWriter(ctx, nil)
	return ctx
}

// emitValues delivers a values chunk with the given state snapshot.
func (e *streamEmitter) emitValues(state map[string]any) {
	e.emit(StreamValues, state)
}

// emitPause delivers the pause pair: the updates interrupt chunk followed by
// the values chunk with "__interrupt__" merged into the state snapshot.
func (e *streamEmitter) emitPause(state map[string]any, interrupts []types.Interrupt) {
	e.emit(StreamUpdates, map[string]any{checkpoint.ReservedInterrupt: interrupts})
	merged := make(map[string]any, len(state)+1)
	for k, v := range state {
		merged[k] = v
	}
	merged[checkpoint.ReservedInterrupt] = interrupts
	e.emit(StreamValues, merged)
}

// emitUpdate delivers one task's updates chunk ({node: {channel: value}}).
// Tasks that wrote nothing produce no chunk.
func (e *streamEmitter) emitUpdate(node string, update map[string]any) {
	if len(update) == 0 {
		return
	}
	e.emit(StreamUpdates, map[string]any{node: update})
}

// emitDebug delivers a debug chunk wrapping typ/payload with the step number
// and an RFC3339Nano timestamp, mirroring Python's `pregel/debug.py`
// envelope.
func (e *streamEmitter) emitDebug(step int, typ string, payload map[string]any) {
	e.emit(StreamDebug, map[string]any{
		"step":      step,
		"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
		"type":      typ,
		"payload":   payload,
	})
}

// debugTask emits the dispatch event for one task. `triggers` is approximate
// in this edge-driven executor — Python records the triggering channel/branch
// names, which this executor does not track; the predecessor node name would
// be used where known, and since it is never known here, the node name
// itself stands in (documented approximation). `id` is the task's planned ID
// when the task comes from a resumed checkpoint, else "" (planned IDs are
// minted only at checkpoint-save time). The task `input` field is the full
// pre-superstep state snapshot (or the Send arg for Send tasks); Python passes
// the task's actual input — a documented approximation.
func (e *streamEmitter) debugTask(step int, t task, state map[string]any) {
	if e == nil || !e.modes[StreamDebug] {
		return
	}
	input := state
	if t.arg != nil {
		input = t.arg
	}
	e.emitDebug(step, "task", map[string]any{
		"id":       t.id,
		"name":     t.node,
		"input":    input,
		"triggers": []string{t.node},
	})
}

// debugTaskResult emits the completion event for one task outcome. result is
// the task's state update; error is the task's error string (nil on success);
// interrupts lists the interrupts the task raised.
func (e *streamEmitter) debugTaskResult(step int, t task, result map[string]any, taskErr error, interrupts []types.Interrupt) {
	if e == nil || !e.modes[StreamDebug] {
		return
	}
	var errStr any
	if taskErr != nil {
		errStr = taskErr.Error()
	}
	var interruptsAny any
	if len(interrupts) > 0 {
		interruptsAny = interrupts
	}
	e.emitDebug(step, "task_result", map[string]any{
		"id":         t.id,
		"name":       t.node,
		"error":      errStr,
		"result":     result,
		"interrupts": interruptsAny,
	})
}

// debugCheckpoint emits the checkpoint event after a save. Divergence from
// Python: Go's checkpoint.Checkpoint carries no per-task `tasks` detail, so
// the payload omits Python's "tasks" key (documented omission).
func (e *streamEmitter) debugCheckpoint(md checkpoint.Metadata, cfg checkpoint.Config, parent checkpoint.Config, values map[string]any, next []checkpoint.PlannedTask) {
	if e == nil || !e.modes[StreamDebug] {
		return
	}
	var parentCfg any
	if parent.CheckpointID != "" {
		parentCfg = parent
	}
	e.emitDebug(md.Step, "checkpoint", map[string]any{
		"config":        cfg,
		"parent_config": parentCfg,
		"values":        values,
		"metadata":      md,
		"next":          next,
	})
}
