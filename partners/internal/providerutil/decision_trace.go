// Decision-run event emission for partner decision-model adapters, porting
// the BaseDecisionModel run semantics of Python langchain-core 1.8 (#41765)
// onto the same cfg.Callbacks path every provider event uses: start and
// end/error events carry the provider identity (ls_provider /
// ls_model_name / ls_model_type "chat") and, on success, the request's
// token usage. Emission is a no-op without registered callbacks, and an
// emit failure is logged and swallowed — tracing never fails a
// classification.

package providerutil

import (
	"context"
	"log/slog"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// DecisionStart emits the chat-model start event for one classification
// run, with the provider-identity metadata layered on top of the caller's
// config metadata (base.py _traced_config). It is a no-op when no
// callbacks are registered.
func DecisionStart(ctx context.Context, cfg runnables.Config, provider, model string, input any) {
	if cfg.Callbacks.Empty() {
		return
	}
	metadata := CloneMetadata(cfg.Metadata)
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadata["ls_provider"] = provider
	metadata["ls_model_name"] = model
	metadata["ls_model_type"] = "chat"
	err := cfg.Callbacks.Emit(ctx, callbacks.Event{
		Kind:     callbacks.EventChatModelStart,
		Name:     model,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     append([]string(nil), cfg.Tags...),
		Metadata: metadata,
		Input:    input,
	})
	if err != nil {
		slog.Debug("decision-run start event rejected", "provider", provider, "error", err)
	}
}

// DecisionEnd emits the terminal event for one classification run: an
// error event on failure, or the end event carrying the typed response
// plus its token usage (base.py _record_usage writes usage_metadata onto
// the active run). Failures are logged and swallowed like the start event.
func DecisionEnd(ctx context.Context, cfg runnables.Config, provider, model string, output any, usage *messages.UsageMetadata, err error) {
	if cfg.Callbacks.Empty() {
		return
	}
	metadata := CloneMetadata(cfg.Metadata)
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadata["ls_provider"] = provider
	metadata["ls_model_name"] = model
	metadata["ls_model_type"] = "chat"
	if usage != nil {
		metadata["usage_metadata"] = map[string]any{
			"input_tokens":  usage.InputTokens,
			"output_tokens": usage.OutputTokens,
			"total_tokens":  usage.TotalTokens,
		}
	}
	kind := callbacks.EventChatModelEnd
	if err != nil {
		kind = callbacks.EventChatModelError
	}
	event := callbacks.Event{
		Kind:     kind,
		Name:     model,
		RunID:    cfg.RunID,
		ParentID: cfg.ParentID,
		Tags:     append([]string(nil), cfg.Tags...),
		Metadata: metadata,
		Output:   output,
	}
	if err != nil {
		event.Error = err.Error()
	}
	if emitErr := cfg.Callbacks.Emit(ctx, event); emitErr != nil {
		slog.Debug("decision-run end event rejected", "provider", provider, "error", emitErr)
	}
}
