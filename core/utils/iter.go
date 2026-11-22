package utils

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
)

// largeUtilDefaultTeeBuffer is the per-consumer channel capacity used when a
// tee is created without an explicit buffer size. It bounds how far the most
// advanced consumer may run ahead of the slowest one (backpressure), while
// still being large enough that finite in-memory sources can be consumed one
// consumer at a time without stalling.
const largeUtilDefaultTeeBuffer = 4096

// largeUtilTeeSource is the normalized source consumed by the fan-out hub.
// pull returns the next item (ok=false ends the stream) and close releases the
// underlying resource. Expressing the source as a pull function lets the same
// hub serve both Iterator[T] values and plain channels.
type largeUtilTeeSource[T any] struct {
	pull  func(ctx context.Context) (T, bool, error)
	close func() error
}

// largeUtilIteratorSource adapts an Iterator[T] into a tee source.
func largeUtilIteratorSource[T any](source Iterator[T]) largeUtilTeeSource[T] {
	return largeUtilTeeSource[T]{
		pull:  source.Next,
		close: source.Close,
	}
}

// largeUtilChannelSource adapts a receive-only channel into a tee source. The
// channel is never closed by the hub; cancellation is observed through ctx.
func largeUtilChannelSource[T any](ch <-chan T) largeUtilTeeSource[T] {
	return largeUtilTeeSource[T]{
		pull: func(ctx context.Context) (T, bool, error) {
			var zero T
			select {
			case v, ok := <-ch:
				if !ok {
					return zero, false, nil
				}
				return v, true, nil
			case <-ctx.Done():
				return zero, false, ctx.Err()
			}
		},
		close: func() error { return nil },
	}
}

// largeUtilTeeChild is a single output lane of a tee. The hub is the only
// sender and the only closer of ch. done is closed by the consumer to signal
// that it has stopped; the hub then stops delivering to this lane.
type largeUtilTeeChild[T any] struct {
	ch     chan T
	done   chan struct{}
	active bool // guarded by the hub mutex
}

// largeUtilTeeHub is the shared goroutine that pulls from one source and fans
// each item out to every active child. Bounded child channels provide natural
// backpressure: the hub cannot read a new item until every active child has
// room, so a fast consumer cannot outrun the slowest one unboundedly.
type largeUtilTeeHub[T any] struct {
	src      largeUtilTeeSource[T]
	ctx      context.Context
	cancel   context.CancelFunc
	mu       sync.Mutex
	children []*largeUtilTeeChild[T]
	finished chan struct{}
	errMu    sync.RWMutex
	err      error
}

// largeUtilNewTeeHub validates the arguments, creates the child lanes, and
// starts the fan-out goroutine.
func largeUtilNewTeeHub[T any](parent context.Context, src largeUtilTeeSource[T], n int, bufferSize int) (*largeUtilTeeHub[T], error) {
	if n <= 0 {
		return nil, fmt.Errorf("tee: number of consumers must be positive, got %d", n)
	}
	if bufferSize < 0 {
		return nil, fmt.Errorf("tee: buffer size must not be negative, got %d", bufferSize)
	}
	if bufferSize == 0 {
		bufferSize = largeUtilDefaultTeeBuffer
	}
	if parent == nil {
		parent = context.Background()
	}

	ctx, cancel := context.WithCancel(parent)
	h := &largeUtilTeeHub[T]{
		src:      src,
		ctx:      ctx,
		cancel:   cancel,
		children: make([]*largeUtilTeeChild[T], n),
		finished: make(chan struct{}),
	}
	for i := 0; i < n; i++ {
		h.children[i] = &largeUtilTeeChild[T]{
			ch:     make(chan T, bufferSize),
			done:   make(chan struct{}),
			active: true,
		}
	}
	go h.run()
	return h, nil
}

// run pulls items from the source and delivers each one to every active child.
// Defer order is intentional: the terminal error is recorded first, child
// channels are closed next, the source is closed, and finished is signalled
// last so waiters observe a fully quiesced hub.
func (h *largeUtilTeeHub[T]) run() {
	defer close(h.finished)
	defer func() { _ = h.src.close() }()
	defer h.closeChannels()
	var terminal error
	defer func() { h.setErr(terminal) }()

	for {
		if err := h.ctx.Err(); err != nil {
			terminal = err
			return
		}
		if !h.hasActive() {
			return
		}

		v, ok, err := h.src.pull(h.ctx)
		if err != nil {
			terminal = err
			return
		}
		if !ok {
			return
		}

		h.mu.Lock()
		active := make([]*largeUtilTeeChild[T], 0, len(h.children))
		for _, c := range h.children {
			if c.active {
				active = append(active, c)
			}
		}
		h.mu.Unlock()
		if len(active) == 0 {
			return
		}

		for _, c := range active {
			select {
			case c.ch <- v:
			case <-c.done:
				h.deactivate(c)
			case <-h.ctx.Done():
				terminal = h.ctx.Err()
				return
			}
		}
	}
}

// closeChannels closes every child channel exactly once (run is the sole
// sender and closer). Buffered items remain readable after the close.
func (h *largeUtilTeeHub[T]) closeChannels() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, c := range h.children {
		close(c.ch)
	}
}

func (h *largeUtilTeeHub[T]) hasActive() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, c := range h.children {
		if c.active {
			return true
		}
	}
	return false
}

// deactivate marks a child as finished. When the last child is gone it cancels
// the hub context, which unblocks a source pull that may still be in flight.
func (h *largeUtilTeeHub[T]) deactivate(c *largeUtilTeeChild[T]) {
	h.mu.Lock()
	wasActive := c.active
	c.active = false
	allDone := true
	for _, child := range h.children {
		if child.active {
			allDone = false
			break
		}
	}
	h.mu.Unlock()
	if wasActive && allDone {
		h.cancel()
	}
}

func (h *largeUtilTeeHub[T]) setErr(err error) {
	h.errMu.Lock()
	h.err = err
	h.errMu.Unlock()
}

func (h *largeUtilTeeHub[T]) getErr() error {
	h.errMu.RLock()
	defer h.errMu.RUnlock()
	return h.err
}

// largeUtilTeeBufferArg extracts an optional buffer size, defaulting to zero
// (which the hub maps to largeUtilDefaultTeeBuffer).
func largeUtilTeeBufferArg(bufferSize []int) int {
	if len(bufferSize) > 0 {
		return bufferSize[0]
	}
	return 0
}

// TeePeer is one child iterator produced by a Tee. It implements Iterator[T]
// and may be consumed by a single goroutine while other peers are consumed
// concurrently; peers share the source but never lose or duplicate items.
type TeePeer[T any] struct {
	hub    *largeUtilTeeHub[T]
	child  *largeUtilTeeChild[T]
	closed atomic.Bool
	once   sync.Once
}

// Next returns the next item for this peer. It honours the per-call context for
// cancellation and returns (zero, false, err) once the source is exhausted or
// the tee is canceled.
func (p *TeePeer[T]) Next(ctx context.Context) (T, bool, error) {
	var zero T
	if p.closed.Load() {
		return zero, false, nil
	}
	select {
	case <-p.child.done:
		return zero, false, nil
	case v, ok := <-p.child.ch:
		if !ok {
			return zero, false, p.hub.getErr()
		}
		return v, true, nil
	case <-ctx.Done():
		return zero, false, ctx.Err()
	}
}

// Close stops this peer. The hub skips it for future items; when every peer is
// closed the source is released. Close is idempotent.
func (p *TeePeer[T]) Close() error {
	p.once.Do(func() {
		p.closed.Store(true)
		close(p.child.done)
		p.hub.deactivate(p.child)
	})
	return nil
}

// Tee splits one Iterator[T] into n independent iterators that each yield the
// same items in the same order. Consumers may advance at different speeds; the
// tee buffers items for slower consumers up to the configured channel capacity
// (backpressure). A single shared goroutine pulls from the source, and a
// sync.Mutex guards the active-child set so the tee is safe for concurrent use
// from multiple goroutines.
type Tee[T any] struct {
	hub   *largeUtilTeeHub[T]
	peers []*TeePeer[T]
}

// Safetee is an alias for Tee, mirroring Python's safetee = Tee. It exists so
// callers can use the historical name without a distinct implementation.
type Safetee[T any] = Tee[T]

// NewTee splits source into n tee peers. The optional bufferSize sets the
// maximum number of in-flight items per consumer; a value <= 0 selects a large
// default. ctx governs the lifetime of the source-reading goroutine; cancel it
// (or call Close) to stop the tee.
func NewTee[T any](ctx context.Context, source Iterator[T], n int, bufferSize ...int) (*Tee[T], error) {
	if source == nil {
		return nil, fmt.Errorf("tee: source iterator is required")
	}
	hub, err := largeUtilNewTeeHub(ctx, largeUtilIteratorSource(source), n, largeUtilTeeBufferArg(bufferSize))
	if err != nil {
		return nil, err
	}
	return largeUtilNewTee(hub), nil
}

// NewSafetee is a named constructor for the Safetee alias. It is identical to
// NewTee and returns *Tee[T], which is assignable to *Safetee[T].
func NewSafetee[T any](ctx context.Context, source Iterator[T], n int, bufferSize ...int) (*Tee[T], error) {
	return NewTee(ctx, source, n, bufferSize...)
}

func largeUtilNewTee[T any](hub *largeUtilTeeHub[T]) *Tee[T] {
	tee := &Tee[T]{hub: hub, peers: make([]*TeePeer[T], len(hub.children))}
	for i, child := range hub.children {
		tee.peers[i] = &TeePeer[T]{hub: hub, child: child}
	}
	return tee
}

// Len returns the number of child iterators.
func (t *Tee[T]) Len() int { return len(t.peers) }

// Peers returns a defensive copy of the child iterators.
func (t *Tee[T]) Peers() []*TeePeer[T] {
	return append([]*TeePeer[T](nil), t.peers...)
}

// At returns the child iterator at index i.
func (t *Tee[T]) At(i int) (*TeePeer[T], error) {
	if i < 0 || i >= len(t.peers) {
		return nil, fmt.Errorf("tee: index %d out of range [0,%d)", i, len(t.peers))
	}
	return t.peers[i], nil
}

// Close stops all peers, cancels the source reader, and waits for a clean
// shutdown. It is idempotent.
func (t *Tee[T]) Close() error {
	for _, p := range t.peers {
		_ = p.Close()
	}
	t.hub.cancel()
	<-t.hub.finished
	return nil
}

// Done is closed once the source-reading goroutine has fully exited.
func (t *Tee[T]) Done() <-chan struct{} { return t.hub.finished }

// Err returns the terminal cause of the tee: nil on normal exhaustion, or the
// source/context error that stopped it.
func (t *Tee[T]) Err() error { return t.hub.getErr() }

// largeUtilBatchIterator lazily groups source items into batches.
type largeUtilBatchIterator[T any] struct {
	source     Iterator[T]
	size       int // >0: fixed batches; 0: a single batch containing everything
	sourceDone bool
	done       bool
}

// BatchIterate returns a lazy Iterator over batches of size items. size > 0
// yields fixed-size batches (the final one may be partial). size == 0 yields a
// single batch containing every item, mirroring Python's batch_iterate(None).
// size < 0 is rejected with an error, mirroring the Python ValueError for a
// non-positive size.
func BatchIterate[T any](size int, source Iterator[T]) (Iterator[[]T], error) {
	if size < 0 {
		return nil, fmt.Errorf("batch iterate: batch size must not be negative, got %d", size)
	}
	if source == nil {
		return nil, fmt.Errorf("batch iterate: source iterator is required")
	}
	return &largeUtilBatchIterator[T]{source: source, size: size}, nil
}

// Next returns the next batch.
func (b *largeUtilBatchIterator[T]) Next(ctx context.Context) ([]T, bool, error) {
	if b.done {
		return nil, false, nil
	}
	batch := []T{}
	for !b.sourceDone {
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		if b.size > 0 && len(batch) >= b.size {
			break
		}
		value, ok, err := b.source.Next(ctx)
		if err != nil {
			return nil, false, err
		}
		if !ok {
			b.sourceDone = true
			break
		}
		batch = append(batch, value)
	}
	if len(batch) == 0 {
		b.done = true
		return nil, false, nil
	}
	if b.size == 0 {
		b.done = true
	}
	return batch, true, nil
}

// Close marks the batch iterator exhausted and releases the source.
func (b *largeUtilBatchIterator[T]) Close() error {
	b.done = true
	return b.source.Close()
}
