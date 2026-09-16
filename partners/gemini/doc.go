// Package gemini adapts langchain-golang chat calls to Google's Gemini API
// via the official google.golang.org/genai SDK (the Go counterpart of the
// google-genai SDK Python's langchain-google-genai builds on). ChatModel
// implements core/language.ChatModel (Invoke/Batch/Stream),
// core/language.StructuredCaller (InvokeStructured via the native
// response_json_schema method — Python with_structured_output's default),
// core/language.ToolBinder (tool_choice mapped onto
// toolConfig.functionCallingConfig modes), and reports its _llm_type
// ("chat-google-generative-ai") for provider-aware middleware.
//
// Importing this package self-registers the "gemini" factory into
// langchain/chatmodels, so WithAgentModel("gemini:<model>") resolves
// end-to-end. GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_GEMINI_BASE_URL are
// read from the environment (the factory reads the keys explicitly; the SDK
// also falls back to the same variables when no key is configured).
//
// This adapter targets the Gemini API backend (generativelanguage
// .googleapis.com) with API-key auth only; the Vertex AI backend is not
// covered (Python's use_vertexai path).
package gemini
