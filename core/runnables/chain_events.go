// This file implements the chain-event instrumentation shared by the
// core/runnables combinators (design t18 PR1). When a caller's Config carries
// a non-empty callbacks manager, every instrumented combinator emits
// chain_start / chain_end (or chain_error) for its own run, mints child run
// IDs via childOptions, and — for Stream surfaces — wraps the returned stream
// so yielded chunks surface as chain_stream events. All instrumentation is
// gated on Config.Callbacks.Empty(): with no callback manager configured,
// helpers return a nil *chainRun whose methods are no-ops and behavior is
// byte-for-byte the pre-instrumentation path.
package runnables

import (
	"context"
	crand "crypto/rand"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
)

// NewRunID returns a fresh RFC 4122 version 4 UUID string, the counterpart of
// Python's uuid.uuid4() used for run-tree identifiers. The module carries no
// UUID dependency: 16 crypto/rand bytes are versioned and formatted directly
// (same dependency-free approach as langgraph/checkpoint IDs). crypto/rand
// does not fail on supported platforms; the time-derived fallback keeps
// instrumentation from panicking in the impossible error case.
func NewRunID() string {
	var b [16]byte
	if _, err := crand.Read(b[:]); err != nil {
		now := time.Now().UnixNano()
		for i := 0; i < 16; i++ {
			b[i] = byte(now >> (uint(i%8) * 8))
		}
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10xx
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// chainRun tracks one combinator-level run for chain-event instrumentation.
// A nil *chainRun means instrumentation is disabled; every method tolerates a
// nil receiver, so call sites need no extra conditional.
type chainRun struct {
	manager  callbacks.Manager
	runID    string
	parentID string
	name     string
	tags     []string
	metadata map[string]any
}

// startChainRun begins a chain run for one combinator invocation: it emits
// chain_start (with an input snapshot) when callbacks are configured and
// derives the run ID from Config.RunID (minting via NewRunID when unset, so a
// caller-supplied WithRunID becomes the run's own ID). The returned context
// carries the manager (callbacks.ContextWithManager) so nested code — e.g. a
// future DispatchCustomEvent — can discover it; installing the manager changes
// no behavior by itself. Without callbacks the original ctx is returned with a
// nil run.
func startChainRun(ctx context.Context, opts []Option, defaultName string, input any) (context.Context, *chainRun) {
	return startChainRunWithConfig(ctx, NewConfig(opts...), defaultName, input)
}

// startChainRunWithConfig is startChainRun for callers that already built the
// Config (avoiding a second NewConfig).
func startChainRunWithConfig(ctx context.Context, cfg Config, defaultName string, input any) (context.Context, *chainRun) {
	if cfg.Callbacks.Empty() {
		return ctx, nil
	}
	runID := cfg.RunID
	if runID == "" {
		runID = NewRunID()
	}
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	run := &chainRun{
		manager:  cfg.Callbacks,
		runID:    runID,
		parentID: cfg.ParentID,
		name:     name,
		tags:     cfg.Tags,
		metadata: cfg.Metadata,
	}
	run.emit(ctx, callbacks.Event{Kind: callbacks.EventChainStart, Input: input})
	return callbacks.ContextWithManager(ctx, cfg.Callbacks), run
}

// child derives the child invocation options for one nested runnable call: the
// run's ID becomes the child's parent and childOptions mints the child's own
// run ID (callbacks-gated). With instrumentation disabled it is plain
// childOptions, preserving the pre-instrumentation config exactly.
func (c *chainRun) child(name string, opts ...Option) []Option {
	if c == nil {
		return childOptions(name, opts...)
	}
	merged := make([]Option, 0, len(opts)+1)
	merged = append(merged, opts...)
	merged = append(merged, WithRunID(c.runID))
	return childOptions(name, merged...)
}

// fnChildOptions derives the options Func hands to its wrapped function.
// Python's Runnable._call_with_config pops the incoming run_id (it belongs to
// the Func's own run) and replaces the function's callbacks with
// run_manager.get_child(), so runnables the function invokes — most importantly
// a chat model called as model.Invoke(ctx, in, opts...) — start their OWN runs
// as children of the Func run and never reuse its ID. The Go analog: the
// function's config parents to the Func run and carries a freshly minted RunID
// when callbacks are active (Go models read RunID from the config instead of
// having a child callback manager mint it). Name, Tags, Metadata, Configurable,
// and MaxConcurrency pass through unchanged — like Python's patch_config,
// run_name propagates only when the caller set one. With a nil run (callbacks
// disabled) nothing is minted; only the consumed RunID shifts to ParentID.
func fnChildOptions(run *chainRun, opts []Option) []Option {
	cfg := NewConfig(opts...).Clone()
	if run != nil {
		cfg.ParentID = run.runID
	} else if cfg.RunID != "" {
		cfg.ParentID = cfg.RunID
	}
	cfg.RunID = ""
	if !cfg.Callbacks.Empty() {
		cfg.RunID = NewRunID()
	}
	return []Option{configOption(cfg)}
}

// end emits chain_end with the run's output. No-op on a nil run.
func (c *chainRun) end(ctx context.Context, output any) {
	if c == nil {
		return
	}
	c.emit(ctx, callbacks.Event{Kind: callbacks.EventChainEnd, Output: output})
}

// fail emits chain_error for err. No-op on a nil run or nil error.
func (c *chainRun) fail(ctx context.Context, err error) {
	if c == nil || err == nil {
		return
	}
	c.emit(ctx, callbacks.Event{Kind: callbacks.EventChainError, Error: err.Error()})
}

// chunk emits chain_stream for one streamed value. No-op on a nil run.
func (c *chainRun) chunk(ctx context.Context, value any) {
	if c == nil {
		return
	}
	c.emit(ctx, callbacks.Event{Kind: callbacks.EventChainStream, Chunk: value})
}

// emit fills the run identity fields and dispatches to the manager. Handler
// errors are swallowed: instrumentation must never fail the business call.
// Tags and metadata are passed by reference (not cloned); handlers must not
// mutate them.
func (c *chainRun) emit(ctx context.Context, event callbacks.Event) {
	event.Name = c.name
	event.RunID = c.runID
	event.ParentID = c.parentID
	event.Tags = c.tags
	event.Metadata = c.metadata
	_ = c.manager.Emit(ctx, event)
}

// chainStream wraps a Stream so the wrapping combinator's run observes every
// pulled chunk as chain_stream, stream completion as chain_end (output nil —
// chunk aggregation is the consumer's job), and the first stream error as
// chain_error. An abandoned stream (Close before exhaustion) emits no end
// event, mirroring how an unread Python async generator never reaches
// on_chain_end. A nil run makes every emission a no-op; constructors return
// the inner stream unwrapped in that case to keep the disabled path
// allocation-free.
type chainStream[O any] struct {
	inner Stream[O]
	run   *chainRun
	done  bool
}

func (s *chainStream[O]) Next(ctx context.Context) (O, bool, error) {
	var zero O
	if s.done {
		return zero, false, nil
	}
	value, ok, err := s.inner.Next(ctx)
	switch {
	case err != nil:
		s.done = true
		s.run.fail(ctx, err)
		return zero, false, err
	case !ok:
		s.done = true
		s.run.end(ctx, nil)
		return zero, false, nil
	default:
		s.run.chunk(ctx, value)
		return value, true, nil
	}
}

// Close abandons the stream early: it suppresses any further events for this
// run, including the chain_end a fully-drained stream would emit. This is a
// documented divergence carried over from Python (an unread async generator
// never reaches on_chain_end): tracer-side, an early-Closed run stays open —
// the LangSmith tracer posts its create but never a patch, and its entry
// lingers in the tracer's local run map for the process lifetime (Python's
// RunTree is likewise never ended; only the local map retention is Go-specific).
func (s *chainStream[O]) Close() error {
	s.done = true
	return s.inner.Close()
}

// wrapChainStream wraps stream with run's chain_stream/chain_end/chain_error
// emissions, returning stream unchanged when instrumentation is disabled.
func wrapChainStream[O any](stream Stream[O], run *chainRun) Stream[O] {
	if run == nil {
		return stream
	}
	return &chainStream[O]{inner: stream, run: run}
}
