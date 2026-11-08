package middleware

import (
	"context"
	"errors"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/outputparser"
)

// aiMessageCarrier matches agents.StructuredOutputValidationError (and any
// similar error) without importing the agents package — agents imports THIS
// package, so a direct import would cycle (see types.go's TracePolicy note).
type aiMessageCarrier interface {
	error
	RefusalMessage() messages.Message
}

// ModelRefusalMiddleware raises *lcerrors.ModelRefusalError for model
// responses containing a standard refusal content block, mirroring Python's
// ModelRefusalMiddleware (langchain 1.4.9, #41582). Add it to CreateAgent to
// treat a refusal as an error instead of a successful final answer; model
// calls without it keep returning refusal-carrying messages. List it after
// ModelFallbackMiddleware so the fallback can catch the error. Streaming
// content may already have been emitted before the refusal is known; the
// middleware neither infers refusals from text nor retries blocked requests.
type ModelRefusalMiddleware struct{}

// NewModelRefusalMiddleware creates the middleware.
func NewModelRefusalMiddleware() *ModelRefusalMiddleware { return &ModelRefusalMiddleware{} }

// WrapModelCall checks the completed model response for a refusal block.
func (m *ModelRefusalMiddleware) WrapModelCall(ctx context.Context, request ModelRequest, handler ModelHandler) (ModelResponse, error) {
	response, err := handler(ctx, request)
	if err != nil {
		// A structured-output validation failure may still carry the
		// refusing AI message (Python: wrap_model_call inspects
		// exc.ai_message before re-raising).
		var carrier aiMessageCarrier
		if errors.As(err, &carrier) {
			if refusalErr := outputparser.RaiseIfRefused(carrier.RefusalMessage()); refusalErr != nil {
				return ModelResponse{}, refusalErr
			}
		}
		return ModelResponse{}, err
	}
	for _, msg := range response.Result {
		if refusalErr := outputparser.RaiseIfRefused(msg); refusalErr != nil {
			return ModelResponse{}, refusalErr
		}
	}
	return response, nil
}

// TracePolicy implements TracePolicyProvider with omit-payload inputs
// (langchain #41689: trace_policy = TracePolicy(process_inputs=omit_payload)),
// matching the ModelRouter precedent (typesafe model_router.go:169). Python's
// omit_payload returns an empty dict; Go uses a non-empty marker so the
// scrubbed payload stays visibly intentional (see DIVERGENCES.md).
func (m *ModelRefusalMiddleware) TracePolicy() TracePolicyConfig {
	return TracePolicyConfig{
		ProcessInputs: func(any) any { return map[string]any{"omitted": true} },
	}
}
