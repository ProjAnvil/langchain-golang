// Package decisions ports Python langchain-core's langchain_core.decisions
// (1.8.0, upstream #41765): the provider-neutral vocabulary for decision
// models — classifiers that answer typed questions about text, messages, or
// structured state.
//
// A DecisionModel evaluates shared state against one or more questions and
// returns typed answers: a probability for a Noul, a selected label for a
// Choice, an expected level for a Score, and a location for a Box. Answers
// keep their probability distributions and confidence so applications choose
// their own thresholds. Provider implementations (langchain-typesafe,
// langchain-openai's decisions API) adapt onto the DecisionModel interface;
// langchain's ModelRouterMiddleware and AutoModeMiddleware are driven by any
// implementation.
package decisions

import (
	"context"

	"github.com/projanvil/langchain-golang/core/runnables"
)

// DecisionModel answers typed questions about state. It is the Go
// equivalent of Python's BaseDecisionModel: implementations call their
// provider, map the reply onto the typed answers, and report token usage
// when the provider does. Implementations must be safe for concurrent use.
type DecisionModel interface {
	// Invoke evaluates state against the request's questions and returns
	// the typed answers keyed by the same question IDs, with request
	// metadata. A provider failure is returned as an error and propagates
	// to the caller — nothing silently falls back. The options mirror the
	// chat-model signature (runnables.Option): callbacks, run IDs, tags,
	// and metadata flow through the provider's event emission, so a
	// classification run composes with tracing exactly like a model call.
	Invoke(ctx context.Context, request Request, opts ...runnables.Option) (Response, error)
}
