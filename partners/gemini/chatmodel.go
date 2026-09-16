package gemini

import (
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/structuredoutput"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/partners/internal/providerutil"
	"google.golang.org/genai"
)

// ChatModel adapts LangChain chat calls to the Gemini API via the official
// google.golang.org/genai SDK, the Go counterpart of the google-genai SDK
// Python's langchain-google-genai builds on.
type ChatModel struct {
	config           modelconfig.Config
	client           *genai.Client
	clientErr        error
	boundTools       []tools.Tool
	structuredOutput *structuredoutput.JSONSchema
	toolConfig       *genai.ToolConfig
}

// Compile-time guard: ChatModel (value receiver) satisfies
// language.StructuredCaller so the agent's ProviderStrategy native path
// (agents.invokeModel → language.InvokeStructured) can use it.
var _ language.StructuredCaller = ChatModel{}

// Compile-time guard: ChatModel satisfies the optional language.ToolBinder
// capability so agent code can thread bind_tools tool_choice
// (agents.bindModelTools → BindToolsWithOptions).
var _ language.ToolBinder = ChatModel{}

// NewChatModel creates a Gemini chat model adapter. Configuration follows the
// repo convention (modelconfig options): WithAPIKey / WithBaseURL / WithModel
// (default "gemini-2.5-flash") / WithHTTPClient / WithTemperature /
// WithMaxTokens / WithTimeout. When no API key is configured the genai SDK
// falls back to GEMINI_API_KEY then GOOGLE_API_KEY from the environment
// (GOOGLE_API_KEY wins when both are set, matching Python's SDK); a missing
// key surfaces as an error on the first Invoke/Stream.
func NewChatModel(opts ...modelconfig.Option) ChatModel {
	cfg := modelconfig.New(opts...)
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	// The client is built eagerly so a missing API key or unusable config is
	// captured once (surfaced on first use) and the underlying HTTP
	// connections stay pooled across Invoke/Stream calls.
	client, err := newClient(cfg)
	return ChatModel{config: cfg, client: client, clientErr: err}
}

// Invoke calls the Gemini API generateContent endpoint.
func (m ChatModel) Invoke(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (messages.Message, error) {
	cfg := runnables.NewConfig(opts...)
	if err := emit(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return messages.Message{}, err
	}

	response, err := m.generateContent(ctx, input)
	if err != nil {
		_ = emit(ctx, cfg, callbacks.EventChatModelError, nil, nil, err)
		return messages.Message{}, err
	}

	message, err := responseToMessage(response)
	if err != nil {
		_ = emit(ctx, cfg, callbacks.EventChatModelError, nil, nil, err)
		return messages.Message{}, err
	}
	if err := emit(ctx, cfg, callbacks.EventChatModelEnd, nil, message, nil); err != nil {
		return messages.Message{}, err
	}
	return message, nil
}

// Batch invokes the model for each input while preserving order.
func (m ChatModel) Batch(
	ctx context.Context,
	inputs [][]messages.Message,
	opts ...runnables.Option,
) ([]messages.Message, error) {
	runnable := runnables.NewFunc(m.Invoke, m.InputSchema(), m.OutputSchema())
	return runnable.Batch(ctx, inputs, opts...)
}

// Stream calls the Gemini API streamGenerateContent endpoint (SSE) and
// yields AI chunks. Usage metadata lands on the terminal chunk: the Gemini
// API reports cumulative counts on every chunk, and re-emitting them per
// chunk would overcount for aggregating consumers, so intermediate chunks
// carry no usage.
func (m ChatModel) Stream(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (runnables.Stream[messages.Message], error) {
	cfg := runnables.NewConfig(opts...)
	if err := emit(ctx, cfg, callbacks.EventChatModelStart, input, nil, nil); err != nil {
		return nil, err
	}
	stream, err := m.createStream(ctx, input, cfg)
	if err != nil {
		_ = emit(ctx, cfg, callbacks.EventChatModelError, nil, nil, err)
		return nil, err
	}
	return stream, nil
}

// InputSchema returns the chat input schema.
func (m ChatModel) InputSchema() schema.Schema {
	return schema.Schema{
		"type":        "array",
		"description": "chat messages",
	}
}

// OutputSchema returns the chat output schema.
func (m ChatModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"role":    schema.String("message role"),
		"content": schema.String("message content"),
	})
}

// BindTools returns a copy of the model with function tools bound, declaring
// them as a single Tool with N FunctionDeclarations (Python's
// _convert_to_genai_tools shape). It is equivalent to BindToolsWithOptions(
// tools, language.BindToolsOptions{}).
func (m ChatModel) BindTools(boundTools []tools.Tool) (language.ChatModel, error) {
	next := m
	next.boundTools = slices.Clone(boundTools)
	return next, nil
}

// BindToolsWithOptions implements language.ToolBinder, mapping the core
// ToolChoice modes onto toolConfig.functionCallingConfig — the exact mapping
// of Python's _tool_choice_to_tool_config (_function_utils.py:700-770):
//
//	"auto"         -> mode AUTO
//	"none"         -> mode NONE
//	"any"          -> mode ANY + allowedFunctionNames = all bound tool names
//	any other name -> mode ANY + allowedFunctionNames = [name]
//
// A zero-value ToolChoice keeps any constructor-level WithToolConfig value.
// ParallelToolCalls is ignored: the Gemini API has no parallel-tool-calls
// payload field (allowedFunctionNames is the closest control and is already
// fully determined by the choice mode).
func (m ChatModel) BindToolsWithOptions(
	boundTools []tools.Tool,
	opts language.BindToolsOptions,
) (language.ChatModel, error) {
	next := m
	next.boundTools = slices.Clone(boundTools)
	if opts.ToolChoice != "" {
		next.toolConfig = toolConfigFromCore(opts.ToolChoice, boundTools)
	}
	return next, nil
}

// toolConfigFromCore translates a language.ToolChoice into the Gemini
// ToolConfig. "any" pins allowedFunctionNames to every bound tool name the
// way Python does (so ANY ranges over exactly the bound set).
func toolConfigFromCore(choice language.ToolChoice, boundTools []tools.Tool) *genai.ToolConfig {
	config := &genai.FunctionCallingConfig{}
	switch choice {
	case language.ToolChoiceAuto:
		config.Mode = genai.FunctionCallingConfigModeAuto
	case language.ToolChoiceNone:
		config.Mode = genai.FunctionCallingConfigModeNone
	case language.ToolChoiceAny:
		config.Mode = genai.FunctionCallingConfigModeAny
		names := make([]string, 0, len(boundTools))
		for _, tool := range boundTools {
			names = append(names, tool.Name())
		}
		config.AllowedFunctionNames = names
	default:
		// Any other string names the specific tool to force.
		config.Mode = genai.FunctionCallingConfigModeAny
		config.AllowedFunctionNames = []string{string(choice)}
	}
	return &genai.ToolConfig{FunctionCallingConfig: config}
}

// WithToolConfig returns a copy of the model with a raw genai ToolConfig set
// (e.g. &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
// Mode: genai.FunctionCallingConfigModeAny}}), mirroring Python's
// ChatGoogleGenerativeAI(tool_config=...). It is overwritten by a subsequent
// BindToolsWithOptions carrying a ToolChoice.
func (m ChatModel) WithToolConfig(toolConfig *genai.ToolConfig) ChatModel {
	next := m
	next.toolConfig = toolConfig
	return next
}

// WithStructuredOutput returns a copy of the model configured for Gemini's
// native structured output: generationConfig.responseMimeType
// "application/json" + responseJsonSchema (the standard JSON Schema is passed
// through untouched — Python's current behavior, chat_models.py:3765-3770
// "Regardless, we use `response_json_schema` in the request"). The name and
// strict flags exist for the structuredoutput.BindJSON signature; the Gemini
// API has no schema name or strictness toggle, so both are ignored.
func (m ChatModel) WithStructuredOutput(
	name string,
	outputSchema schema.Schema,
	strict bool,
) ChatModel {
	next := m
	cfg := structuredoutput.NewJSONSchema(name, outputSchema, strict)
	next.structuredOutput = &cfg
	return next
}

// InvokeStructured implements language.StructuredCaller via Gemini's native
// response_json_schema method — Python's with_structured_output default
// ("json_schema", chat_models.py:2545-2581). The returned message's text is
// the model's JSON conforming to sch.
func (m ChatModel) InvokeStructured(
	ctx context.Context,
	input []messages.Message,
	sch schema.Schema,
) (messages.Message, error) {
	name := "response_format"
	if title, ok := sch["title"].(string); ok && title != "" {
		name = title
	}
	return m.WithStructuredOutput(name, sch, true).Invoke(ctx, input)
}

// Capabilities returns the adapter capability declaration. Every flag is
// exercised by the implementation: tool calling/choice serialize through
// tools/toolConfig, structured output through generationConfig
// .responseJsonSchema, image/audio/video/file inputs through inline_data /
// file_data parts (URLs included — the API fetches file_data URIs
// server-side), usage metadata on invoke and on the terminal stream chunk,
// and streaming over the SSE endpoint. JSONMode stays false: plain-JSON mode
// is not exposed as its own toggle (structured output already sets the JSON
// mime type internally).
func (m ChatModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{
		ToolCalling:      true,
		ToolChoice:       true,
		StructuredOutput: true,
		ImageInputs:      true,
		ImageURLs:        true,
		AudioInputs:      true,
		PDFInputs:        true,
		VideoInputs:      true,
		UsageMetadata:    true,
		Streaming:        true,
	}
}

// LLMType reports the model's Python "_llm_type" identifier
// ("chat-google-generative-ai"), mirroring Python's BaseChatModel._llm_type
// attribute (chat_models.py:3315). Used by middleware (e.g.
// SummarizationMiddleware) to tune provider-specific behavior.
func (m ChatModel) LLMType() string { return "chat-google-generative-ai" }

func (m ChatModel) ensureClient() (*genai.Client, error) {
	if m.client != nil {
		return m.client, nil
	}
	if m.clientErr != nil {
		return nil, m.clientErr
	}
	return nil, fmt.Errorf("gemini: client is not configured")
}

func (m ChatModel) generateContent(
	ctx context.Context,
	input []messages.Message,
) (*genai.GenerateContentResponse, error) {
	client, err := m.ensureClient()
	if err != nil {
		return nil, err
	}
	systemInstruction, contents, err := buildContents(input)
	if err != nil {
		return nil, err
	}
	config, err := m.buildGenerateContentConfig(systemInstruction)
	if err != nil {
		return nil, err
	}
	response, err := client.Models.GenerateContent(ctx, m.config.Model, contents, config)
	if err != nil {
		return nil, fmt.Errorf("gemini %s: generate content: %w", m.config.Model, err)
	}
	return response, nil
}

// buildGenerateContentConfig assembles the request's generationConfig /
// tools / toolConfig / systemInstruction block. Sampling knobs mirror Python
// BaseChatGoogleGenerativeAI's passthrough fields (temperature, max_output_
// tokens; top_p/top_k not exposed by modelconfig today).
func (m ChatModel) buildGenerateContentConfig(systemInstruction *genai.Content) (*genai.GenerateContentConfig, error) {
	config := &genai.GenerateContentConfig{SystemInstruction: systemInstruction}
	if m.config.Temperature != nil {
		config.Temperature = new(float32(*m.config.Temperature))
	}
	if m.config.MaxTokens != nil {
		config.MaxOutputTokens = int32(*m.config.MaxTokens)
	}
	if m.structuredOutput != nil {
		config.ResponseMIMEType = "application/json"
		config.ResponseJsonSchema = map[string]any(m.structuredOutput.Schema)
	}
	if len(m.boundTools) > 0 {
		declarations := make([]*genai.FunctionDeclaration, 0, len(m.boundTools))
		for _, tool := range m.boundTools {
			declaration := &genai.FunctionDeclaration{
				Name:        tool.Name(),
				Description: tool.Description(),
			}
			if argsSchema := tool.ArgsSchema(); len(argsSchema) > 0 {
				parameters, err := toGenAISchema(argsSchema)
				if err != nil {
					return nil, fmt.Errorf("gemini: tool %q schema: %w", tool.Name(), err)
				}
				declaration.Parameters = parameters
			}
			declarations = append(declarations, declaration)
		}
		config.Tools = []*genai.Tool{{FunctionDeclarations: declarations}}
	}
	if m.toolConfig != nil {
		config.ToolConfig = m.toolConfig
	}
	return config, nil
}

func emit(
	ctx context.Context,
	cfg runnables.Config,
	kind callbacks.EventKind,
	input any,
	output any,
	err error,
) error {
	return providerutil.Emit(ctx, cfg, kind, input, output, err)
}
