package utils

import (
	"context"
	"fmt"
)

// AsyncTee is the channel-based equivalent of Python's atee. It splits one
// source into n output channels that each receive the same items in the same
// order. A single goroutine reads the source and fans items out; bounded
// channels provide backpressure and the constructor context drives clean
// shutdown. When the source is exhausted or the context is canceled, every
// output channel is closed, so consumers can range over them to completion.
type AsyncTee[T any] struct {
	hub     *largeUtilTeeHub[T]
	outputs []<-chan T
}

// NewAsyncTee splits an Iterator[T] source into n output channels. The
// optional bufferSize bounds how far the fastest consumer may outrun the
// slowest one; a value <= 0 selects a large default. ctx cancels the tee.
func NewAsyncTee[T any](ctx context.Context, source Iterator[T], n int, bufferSize ...int) (*AsyncTee[T], error) {
	if source == nil {
		return nil, fmt.Errorf("async tee: source iterator is required")
	}
	return largeUtilNewAsyncTee(ctx, largeUtilIteratorSource(source), n, largeUtilTeeBufferArg(bufferSize))
}

// NewAsyncTeeFromChannel splits a receive-only channel into n output channels.
// The input channel is owned by the caller and is never closed by the tee.
func NewAsyncTeeFromChannel[T any](ctx context.Context, source <-chan T, n int, bufferSize ...int) (*AsyncTee[T], error) {
	if source == nil {
		return nil, fmt.Errorf("async tee: source channel is required")
	}
	return largeUtilNewAsyncTee(ctx, largeUtilChannelSource(source), n, largeUtilTeeBufferArg(bufferSize))
}

func largeUtilNewAsyncTee[T any](ctx context.Context, src largeUtilTeeSource[T], n int, bufferSize int) (*AsyncTee[T], error) {
	hub, err := largeUtilNewTeeHub(ctx, src, n, bufferSize)
	if err != nil {
		return nil, err
	}
	outputs := make([]<-chan T, n)
	for i := 0; i < n; i++ {
		outputs[i] = hub.children[i].ch
	}
	return &AsyncTee[T]{hub: hub, outputs: outputs}, nil
}

// Channels returns a defensive copy of the output channels.
func (a *AsyncTee[T]) Channels() []<-chan T {
	return append([]<-chan T(nil), a.outputs...)
}

// At returns the output channel at index i.
func (a *AsyncTee[T]) At(i int) (<-chan T, error) {
	if i < 0 || i >= len(a.outputs) {
		return nil, fmt.Errorf("async tee: index %d out of range [0,%d)", i, len(a.outputs))
	}
	return a.outputs[i], nil
}

// Len returns the number of output channels.
func (a *AsyncTee[T]) Len() int { return len(a.outputs) }

// Close cancels the tee and waits for the fan-out goroutine to exit. All
// output channels are closed before it returns, giving a clean shutdown.
// Close is idempotent with respect to cancellation; the wait is always safe
// because the source adapters honour the context.
func (a *AsyncTee[T]) Close() error {
	a.hub.cancel()
	<-a.hub.finished
	return nil
}

// Done is closed once the fan-out goroutine has fully exited.
func (a *AsyncTee[T]) Done() <-chan struct{} { return a.hub.finished }

// Err returns the terminal cause of the tee: nil on normal exhaustion, or the
// source/context error that stopped it.
func (a *AsyncTee[T]) Err() error { return a.hub.getErr() }
