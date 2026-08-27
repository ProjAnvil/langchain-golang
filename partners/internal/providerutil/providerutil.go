// Package providerutil holds callback-emission and SSE-scanning helpers
// shared by the partner provider packages (openai, anthropic, ollama,
// chroma). Before this package existed each provider carried a private copy
// of the same emit/clone helpers and the copies drifted — most visibly when
// two providers kept bufio.Scanner's 64KB default line limit while a third
// raised it, so large tool-call argument payloads failed on some providers
// and not others. Provider-specific translation stays in the provider; the
// mechanical event plumbing lives here exactly once.
package providerutil

import (
	"bufio"
	"context"
	"io"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/streamevents"
)

// maxSSELineBytes bounds a single SSE line. Tool-call arguments, image
// payloads, and OpenAI Responses-API terminal events (which re-embed the
// whole response) can exceed bufio.Scanner's 64KB default by orders of
// magnitude, so every provider shares one generously sized buffer.
const maxSSELineBytes = 16 << 20 // 16 MiB

// NewSSEScanner returns a bufio.Scanner over an SSE (or NDJSON) response
// body configured to accept lines up to maxSSELineBytes instead of the
// 64KB default, which rejects large tool-call argument deltas and base64
// payloads mid-stream.
func NewSSEScanner(body io.Reader) *bufio.Scanner {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), maxSSELineBytes)
	return scanner
}

// Emit publishes a chat-model lifecycle event (start/end/error) on cfg's
// callback manager. It is a no-op when no callbacks are registered.
func Emit(
	ctx context.Context,
	cfg runnables.Config,
	kind callbacks.EventKind,
	input any,
	output any,
	err error,
) error {
	if cfg.Callbacks.Empty() {
		return nil
	}
	event := callbacks.Event{
		Kind:     kind,
		Name:     cfg.Name,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     append([]string(nil), cfg.Tags...),
		Metadata: CloneMetadata(cfg.Metadata),
		Input:    input,
		Output:   output,
	}
	if err != nil {
		event.Error = err.Error()
	}
	return cfg.Callbacks.Emit(ctx, event)
}

// EmitStream publishes one streamed chat-model chunk on cfg's callback
// manager. It is a no-op when no callbacks are registered.
func EmitStream(ctx context.Context, cfg runnables.Config, chunk messages.Message) error {
	if cfg.Callbacks.Empty() {
		return nil
	}
	return cfg.Callbacks.Emit(ctx, callbacks.Event{
		Kind:     callbacks.EventChatModelStream,
		Name:     cfg.Name,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     append([]string(nil), cfg.Tags...),
		Metadata: CloneMetadata(cfg.Metadata),
		Chunk:    chunk,
	})
}

// EmitProtocolEvent publishes a single v3 protocol (stream-events) chunk on
// cfg's callback manager without any start-once bookkeeping.
func EmitProtocolEvent(ctx context.Context, cfg runnables.Config, event streamevents.Event) error {
	if cfg.Callbacks.Empty() {
		return nil
	}
	return cfg.Callbacks.Emit(ctx, callbacks.Event{
		Kind:     callbacks.EventChatModelProtocol,
		Name:     cfg.Name,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     append([]string(nil), cfg.Tags...),
		Metadata: CloneMetadata(cfg.Metadata),
		Chunk:    event,
	})
}

// EmitProtocol publishes a v3 protocol (stream-events) chunk on cfg's
// callback manager, lazily emitting the leading message-start event exactly
// once per stream; started tracks whether that start event has been sent.
func EmitProtocol(ctx context.Context, cfg runnables.Config, started *bool, event streamevents.Event) error {
	if cfg.Callbacks.Empty() {
		return nil
	}
	if !*started {
		*started = true
		if err := EmitProtocolEvent(ctx, cfg, streamevents.Event{Event: streamevents.EventMessageStart}); err != nil {
			return err
		}
	}
	return EmitProtocolEvent(ctx, cfg, event)
}

// CloneMetadata shallow-copies a metadata map so callback events never alias
// a caller-mutable map. Nil maps stay nil.
func CloneMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return nil
	}
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		out[key] = value
	}
	return out
}
