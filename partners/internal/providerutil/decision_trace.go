// Decision-run tracing for partner decision-model adapters, porting the
// BaseDecisionModel run semantics of Python langchain-core 1.8 (#41765):
// every classification surfaces as a chat-model run carrying the provider
// identity (ls_provider / ls_model_name / ls_model_type) and, on success,
// the request's token usage. Tracing failures never fail an otherwise
// successful classification.

package providerutil

import (
	"context"
	"log/slog"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
)

// decisionRunMetadata builds the provider-identity metadata every decision
// run carries (base.py _traced_config: ls_provider, ls_model_name, and the
// fixed ls_model_type "chat").
func decisionRunMetadata(provider, model string) map[string]any {
	return map[string]any{
		"ls_provider":   provider,
		"ls_model_name": model,
		"ls_model_type": "chat",
	}
}

// EmitDecisionStart emits the chat-model start event for one classification
// run. It is a no-op outside a callbacks context (no manager installed), and
// an emit failure is logged and swallowed — tracing must not break
// classification.
func EmitDecisionStart(ctx context.Context, provider, model string, input any) {
	manager, ok := callbacks.ManagerFromContext(ctx)
	if !ok {
		return
	}
	err := manager.Emit(ctx, callbacks.Event{
		Kind:     callbacks.EventChatModelStart,
		Name:     model,
		Metadata: decisionRunMetadata(provider, model),
		Input:    input,
	})
	if err != nil {
		slog.Debug("decision-run start event rejected", "provider", provider, "error", err)
	}
}

// EmitDecisionEnd emits the terminal event for one classification run: an
// error event on failure, or the end event carrying the typed response plus
// its token usage (base.py _record_usage writes usage_metadata onto the
// active run). Failures are logged and swallowed like the start event.
func EmitDecisionEnd(ctx context.Context, provider, model string, output any, usage *messages.UsageMetadata, err error) {
	manager, ok := callbacks.ManagerFromContext(ctx)
	if !ok {
		return
	}
	metadata := decisionRunMetadata(provider, model)
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
	emitErr := manager.Emit(ctx, callbacks.Event{
		Kind:     kind,
		Name:     model,
		Metadata: metadata,
		Output:   output,
		Error:    errorString(err),
	})
	if emitErr != nil {
		slog.Debug("decision-run end event rejected", "provider", provider, "error", emitErr)
	}
}

// errorString renders err for the event's Error field ("" when nil).
func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
