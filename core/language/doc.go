// Package language defines the chat-model abstraction the agent runtime and
// partner packages build on.
//
// The core entry point is the ChatModel interface: a
// Runnable[[]Message, Message] that also exposes BindTools (model-side tool
// binding) and Capabilities (feature flags such as ToolCalling and
// StructuredOutput). FakeChatModel is an in-process, deterministic
// implementation used by tests and examples; configure its canned behaviour
// via the WithResponses, WithStreamChunks, and WithCapabilities options.
//
// StructuredCaller is implemented by models that can produce output conforming
// to a JSON schema natively (provider response_format / tool-based). The
// generic helper InvokeStructured(ctx, m, input, sch) returns a message whose
// text is JSON conforming to sch: it prefers m's native StructuredCaller path
// when available, and otherwise falls back to Invoke plus a best-effort
// JSON-decode and required-key validation (returning ErrSchemaViolation on
// failure). Partner implementations such as partners/openai implement
// StructuredCaller so the native path is used.
package language
