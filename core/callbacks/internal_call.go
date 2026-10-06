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

// IsInternalCall reports whether ctx carries the middleware-internal model
// call marker (see WithInternalCall).
func IsInternalCall(ctx context.Context) bool {
	return ctx.Value(internalCallKey{}) != nil
}
