// Package openai is the OpenAI partner: a real (not adapter-only)
// implementation of language.ChatModel backed by the OpenAI Responses API
// (/responses endpoint).
//
// ChatModel implements:
//
//   - Invoke and Stream over the Responses API, including server-sent-event
//     streaming and tool calling (BindTools);
//   - language.StructuredCaller via its InvokeStructured method, which
//     delegates to WithStructuredOutput so the JSON schema is enforced
//     upstream by the API rather than by client-side validation.
//
// The package self-registers its ProviderFactory into langchain/chatmodels
// under the provider name "openai" via an init() (see init.go), so
// chatmodels.Resolve(ChatModelSpec{Provider: "openai", ...}) produces an
// openai.ChatModel once this package is imported, and the factory reads
// OPENAI_API_KEY and OPENAI_BASE_URL from the environment when set. By
// contrast the anthropic, ollama, and chroma partner packages remain
// adapter-only (Python metadata plus thin wrappers, with no Go Responses-API
// target), so they do not register a Go factory.
package openai
