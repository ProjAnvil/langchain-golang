// Package ollama adapts langchain-golang chat calls to the Ollama /api/chat
// endpoint. ChatModel implements core/language.ChatModel (Invoke/Batch/Stream),
// core/language.StructuredCaller (InvokeStructured via the native json_schema
// format — Python's with_structured_output default), and reports its _llm_type
// ("ollama-chat") for provider-aware middleware.
//
// Importing this package self-registers the "ollama" factory into
// langchain/chatmodels, so WithAgentModel("ollama:<model>") resolves
// end-to-end. OLLAMA_HOST overrides the default http://localhost:11434 base URL.
package ollama
