// Package anthropic adapts langchain-golang chat calls to Anthropic's Messages
// API. ChatModel implements core/language.ChatModel (Invoke/Batch/Stream),
// core/language.StructuredCaller (InvokeStructured via the function_calling
// method — Python's with_structured_output default), and reports its
// _llm_type ("anthropic-chat") for provider-aware middleware.
//
// Importing this package self-registers the "anthropic" factory into
// langchain/chatmodels, so WithAgentModel("anthropic:<model>") resolves
// end-to-end. ANTHROPIC_API_KEY / ANTHROPIC_API_URL / ANTHROPIC_BASE_URL are
// read from the environment by the self-registered factory.
package anthropic
