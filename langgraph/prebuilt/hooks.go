package prebuilt

import (
	"context"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
)

// ModelHookFunc is the shape of the deprecated Python prebuilt's
// pre_model_hook / post_model_hook callables (chat_agent_executor.py): it
// receives the current graph state and returns a state update. Go adds the
// context.Context every agents middleware hook already receives, so a hook
// can call graph.Interrupt for human-in-the-loop pauses.
type ModelHookFunc func(ctx context.Context, state map[string]any) (map[string]any, error)

// preModelHookMiddleware adapts a ModelHookFunc to agents.BeforeModelHook so
// WithPreModelHook can ride CreateAgent's existing hook machinery instead of
// a parallel prebuilt implementation.
type preModelHookMiddleware struct {
	hook ModelHookFunc
}

func (m preModelHookMiddleware) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	return m.hook(ctx, state)
}

// postModelHookMiddleware adapts a ModelHookFunc to agents.AfterModelHook.
type postModelHookMiddleware struct {
	hook ModelHookFunc
}

func (m postModelHookMiddleware) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	return m.hook(ctx, state)
}

// WithPreModelHook mirrors the deprecated Python prebuilt create_react_agent's
// `pre_model_hook` parameter: hook runs before each model call and returns a
// state update. Useful for managing long message histories (trimming,
// summarization) and human-in-the-loop gates.
//
// Mapping note: the hook is implemented as an agents BeforeModelHook
// middleware, which is Python's recommended replacement for pre_model_hook.
// Under that mapping a "messages" update reshapes only the local view feeding
// the next model call (Python prebuilt's `llm_input_messages` behavior) and
// is not persisted; every other key in the update IS persisted. To persist a
// rewritten history, return it under a custom state key or write via a
// middleware that owns the rewrite. A `jump_to` value of "model", "tools", or
// "end" short-circuits routing exactly as in agents middleware.
func WithPreModelHook(hook ModelHookFunc) agents.AgentOption {
	return agents.WithAgentMiddleware(preModelHookMiddleware{hook: hook})
}

// WithPostModelHook mirrors the deprecated Python prebuilt create_react_agent's
// `post_model_hook` parameter: hook runs after each model call and returns a
// state update (appending messages, rewriting routing via "jump_to", etc.).
// It is implemented as an agents.AfterModelHook middleware, Python's
// recommended replacement for post_model_hook.
func WithPostModelHook(hook ModelHookFunc) agents.AgentOption {
	return agents.WithAgentMiddleware(postModelHookMiddleware{hook: hook})
}

// WithDynamicModel mirrors the deprecated Python prebuilt create_react_agent's
// callable-model overload (`model: Callable[[AgentState, Runtime],
// LanguageModel]`): resolver is consulted inside the model node on every
// model call — after BeforeModel/pre_model_hook state updates are applied —
// and its non-nil return becomes the model for that call. Returning nil falls
// back to the static model passed to CreateReactAgent. See
// agents.WithAgentDynamicModel for the full contract.
func WithDynamicModel(resolver func(state map[string]any, rt runtime.Runtime) language.ChatModel) agents.AgentOption {
	return agents.WithAgentDynamicModel(resolver)
}
