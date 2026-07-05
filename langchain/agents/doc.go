// Package agents implements a scoped Go port of Python's
// langchain.agents.create_agent: a model<->tools loop wired on top of the
// internal agent runtime graph, with composable middleware hooks around the
// model call and each tool call.
//
// CreateAgent is the single entry point and is at full parameter parity with
// Python's create_agent for the in-scope parameter set (16 of 17). The
// positional model argument together with the WithAgent* option families
// covers:
//
//   - model: the positional model arg, or a bare "provider:model" string
//     resolved at build time via WithAgentModel (e.g.
//     WithAgentModel("openai:gpt-4o"));
//   - tools: the toolList slice, accepting any core/tools.Tool including
//     FromFunc callables reflected from ordinary Go funcs;
//   - prompts: WithAgentSystemPrompt / WithAgentSystemPromptTemplate;
//   - middleware: WithAgentMiddleware (Before/After-model,
//     Before/After-agent, and Wrap-model/Wrap-tool-call hooks; see the Hook
//     interfaces in create_agent.go);
//   - structured output: WithAgentResponseFormat accepts a ToolStrategy,
//     ProviderStrategy, or AutoStrategy (AutoStrategy resolves to one of the
//     former at build time);
//   - state schema: WithAgentStateFields, paired with the WithContextValues /
//     ContextValue context helpers for per-run context (see context_schema.go)
//     and WithAgentContextSchema to declare its shape;
//   - persistence: WithAgentCheckpointer, WithAgentStore, WithAgentCache;
//   - control flow: WithAgentInterruptBefore / WithAgentInterruptAfter,
//     WithAgentRecursionLimit;
//   - diagnostics: WithAgentDebug, WithAgentName.
//
// The one Python parameter NOT ported is transformers (per-callable output
// transformations such as PII redaction of streaming deltas). It depends on
// langgraph stream modes this port does not expose; the equivalent streaming
// PII redaction is delivered instead via the WrapModelStreamHook delta-layer
// middleware (see langchain/agents/middleware).
//
// See create_agent.go for detailed scope notes (interrupts, the jump_to
// convention, before/after-agent wiring, structured-output best-effort
// binding) and migration_plan/core-v1-migration-todo.md (P5 langchain/agents)
// for the authoritative in/out-of-scope list.
package agents
