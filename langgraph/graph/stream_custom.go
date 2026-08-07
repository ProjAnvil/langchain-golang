package graph

import "context"

// StreamWriter lets a node emit an arbitrary custom payload into an active
// Stream stream (StreamCustom mode), mirroring Python's
// `langgraph.config.get_stream_writer`. Payloads flow straight to the chunk
// stream with the emitting node's namespace.
type StreamWriter func(payload any)

// streamWriterKey is the context-value key under which the executor installs
// the per-node StreamWriter while StreamCustom is active.
type streamWriterKey struct{}

// contextWithStreamWriter installs w into ctx.
func contextWithStreamWriter(ctx context.Context, w StreamWriter) context.Context {
	return context.WithValue(ctx, streamWriterKey{}, w)
}

// StreamWriterFromContext returns the StreamWriter installed in ctx by the
// executor, or nil when StreamCustom mode is not active (including the plain
// Invoke/InvokeWithOptions paths) — nodes must nil-check before writing,
// which keeps the inactive path at zero overhead.
func StreamWriterFromContext(ctx context.Context) StreamWriter {
	writer, _ := ctx.Value(streamWriterKey{}).(StreamWriter)
	return writer
}
