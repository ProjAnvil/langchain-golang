package callbacks

import "context"

// internalCallKey is an unexported context key marking a model call as
// internal to middleware. Being unexported, callers cannot forge the marker
// from outside the package — user-supplied context values can never hide a
// real model call from stream projections.
type internalCallKey struct{}

// WithInternalCall marks ctx as carrying a middleware-internal model call.
// Events emitted under the returned context (chat-model stream/end events)
// are excluded from messages-mode stream projections and from StreamEvents'
// chat-model events, keeping middleware bookkeeping calls (e.g. internal
// summarization or classification calls) out of what consumers see as the
// conversation. Python langchain 1.4's internal_call_metadata /
// InternalCallTransformer (#39252) — the Go port uses a context token
// instead of metadata dicts (see DIVERGENCES.md).
func WithInternalCall(ctx context.Context) context.Context {
	return context.WithValue(ctx, internalCallKey{}, true)
}

// internalCallSourceKey carries the lc_source value of an internal call
// (Python tags the internal run's metadata with lc_source, e.g.
// "summarization"; #41579).
type internalCallSourceKey struct{}

// WithInternalCallSource marks ctx with the feature name behind an internal
// model call (the lc_source metadata upstream attaches to the tagged run).
func WithInternalCallSource(ctx context.Context, source string) context.Context {
	return context.WithValue(WithInternalCall(ctx), internalCallSourceKey{}, source)
}

// InternalCallSource returns the lc_source of the internal call ctx carries,
// or "" when unset.
func InternalCallSource(ctx context.Context) string {
	source, _ := ctx.Value(internalCallSourceKey{}).(string)
	return source
}

// IsInternalCall reports whether ctx carries the middleware-internal model
// call marker (see WithInternalCall).
func IsInternalCall(ctx context.Context) bool {
	return ctx.Value(internalCallKey{}) != nil
}
