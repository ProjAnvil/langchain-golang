package openai

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"sync/atomic"
	"time"
)

// defaultStreamChunkTimeout mirrors Python's 120s fallback
// (chat_models/base.py:921).
const defaultStreamChunkTimeout = 120 * time.Second

// streamChunkTimeoutEnvVar overrides the default per-chunk stream timeout
// (seconds, float).
const streamChunkTimeoutEnvVar = "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S"

// StreamChunkTimeoutError is returned when no streaming chunk arrives within
// the configured stream_chunk_timeout, mirroring Python's
// langchain_openai.chat_models._client_utils.StreamChunkTimeoutError. It
// matches context.DeadlineExceeded / os.ErrDeadlineExceeded via errors.Is and
// satisfies the Timeout() bool convention (net.Error style), so existing
// timeout handlers keep catching it — the Go analog of Python subclassing
// TimeoutError. The structured fields mirror the WARNING log's attributes so
// diagnostic code doesn't need to parse the message.
type StreamChunkTimeoutError struct {
	TimeoutS       float64 // seconds (Python: timeout_s)
	Model          string
	ChunksReceived int
}

// Error returns the self-describing message naming the knob and the env var.
func (e *StreamChunkTimeoutError) Error() string {
	context := ""
	if e.Model != "" {
		context = fmt.Sprintf("model=%s, ", e.Model)
	}
	return fmt.Sprintf(
		"No streaming chunk received for %.1fs (%schunks_received=%d). The "+
			"connection may be alive at the TCP layer but is not producing "+
			"content. Tune or disable via the stream_chunk_timeout option "+
			"(WithStreamChunkTimeout; set to 0 to disable) or the %s env var.",
		e.TimeoutS, context, e.ChunksReceived, streamChunkTimeoutEnvVar)
}

// Timeout reports the error as a timeout (net.Error-style convention).
func (e *StreamChunkTimeoutError) Timeout() bool { return true }

// Is matches the deadline-exceeded sentinels so generic timeout handlers
// catch it (Python subclasses TimeoutError).
func (e *StreamChunkTimeoutError) Is(target error) bool {
	return target == context.DeadlineExceeded || target == os.ErrDeadlineExceeded
}

// resolveStreamChunkTimeout mirrors Python's _float_env + the
// stream_chunk_timeout field validator: garbage or negative env values fall
// back to the 120s default with a WARNING; "0" disables.
func resolveStreamChunkTimeout() time.Duration {
	raw, ok := os.LookupEnv(streamChunkTimeoutEnvVar)
	if !ok || raw == "" {
		return defaultStreamChunkTimeout
	}
	seconds, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		slog.Warn("openai: invalid "+streamChunkTimeoutEnvVar+" value; falling back to default",
			slog.String("value", raw),
			slog.String("default", defaultStreamChunkTimeout.String()))
		return defaultStreamChunkTimeout
	}
	if seconds < 0 {
		slog.Warn("openai: negative "+streamChunkTimeoutEnvVar+" value; falling back to default (pass 0 to disable)",
			slog.Float64("value_s", seconds),
			slog.String("default", defaultStreamChunkTimeout.String()))
		return defaultStreamChunkTimeout
	}
	return time.Duration(seconds * float64(time.Second))
}

// chunkTimeoutGuard implements the per-chunk deadline, mirroring Python's
// _astream_with_chunk_timeout (_client_utils.py:617-683): arm starts a timer
// that, on expiry, records the fire and cancels the stream's request context
// — unblocking the body read and releasing the HTTP connection promptly, the
// analog of Python's aclose()-on-early-exit. Only atomics cross goroutines
// (context.CancelFunc is concurrency-safe), so the stream structs stay
// single-goroutine.
type chunkTimeoutGuard struct {
	fired atomic.Bool
}

// arm starts the per-chunk timer; it returns nil (no bound) when timeout <= 0
// (0 disables, mirroring Python's None/0 off switch). The caller stops the
// returned timer when the chunk wait ends.
func (g *chunkTimeoutGuard) arm(timeout time.Duration, cancel context.CancelFunc) *time.Timer {
	if timeout <= 0 {
		return nil
	}
	// Consume any previous fire so an already-done stream reports (false, nil)
	// instead of re-raising on subsequent Next calls.
	g.fired.Store(false)
	// Race note: if the timer fires concurrently with an on-time chunk,
	// cancel() has already killed the request context, so the next Next
	// returns a raw transport error (context canceled) instead of
	// StreamChunkTimeoutError. Semantically defensible — the chunk did arrive
	// late from the caller's wall-clock perspective; Python's
	// asyncio.wait_for has the same edge.
	return time.AfterFunc(timeout, func() {
		g.fired.Store(true)
		cancel()
	})
}

// timeoutError builds and logs the StreamChunkTimeoutError after a fire,
// emitting the structured WARNING (source=stream_chunk_timeout) that Python
// logs from _astream_with_chunk_timeout.
func (g *chunkTimeoutGuard) timeoutError(timeout time.Duration, model string, chunksReceived int) *StreamChunkTimeoutError {
	slog.Warn("openai stream_chunk_timeout fired",
		slog.String("source", "stream_chunk_timeout"),
		slog.Float64("timeout_s", timeout.Seconds()),
		slog.String("model_name", model),
		slog.Int("chunks_received", chunksReceived))
	return &StreamChunkTimeoutError{
		TimeoutS:       timeout.Seconds(),
		Model:          model,
		ChunksReceived: chunksReceived,
	}
}
