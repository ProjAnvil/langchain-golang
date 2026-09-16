package gemini

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"google.golang.org/genai"
)

// buildContents converts the LangChain message history into the Gemini API's
// (systemInstruction, contents) shape, mirroring Python's _parse_chat_history
// (langchain_google_genai/chat_models.py:1430-1651):
//
//   - System messages accumulate into a single systemInstruction Content;
//   - Human messages become role "user" Contents (text + multimodal parts);
//   - AI messages become role "model" Contents; their tool calls serialize as
//     functionCall parts;
//   - Tool messages become functionResponse parts grouped under one role
//     "user" Content following the AI message they answer, with the function
//     name resolved from the preceding AI message's tool calls (Python's
//     _get_ai_message_tool_messages_parts, chat_models.py:1085-1110);
//   - A model Content that converts to no parts keeps one empty text part
//     (the API requires at least one part per Content).
func buildContents(input []messages.Message) (*genai.Content, []*genai.Content, error) {
	var systemInstruction *genai.Content
	var contents []*genai.Content

	// Pending functionResponse parts flushed as one user Content when the run
	// of Tool messages ends, mirroring Python's grouping.
	var pendingToolParts []*genai.Part
	flushToolParts := func() {
		if len(pendingToolParts) == 0 {
			return
		}
		contents = append(contents, &genai.Content{Role: "user", Parts: pendingToolParts})
		pendingToolParts = nil
	}

	// toolCallNames resolves tool call IDs to function names for the
	// functionResponse parts (the API keys responses by name, not id).
	toolCallNames := make(map[string]string)

	for _, message := range input {
		switch message.Role {
		case messages.RoleSystem:
			parts, err := contentParts(message)
			if err != nil {
				return nil, nil, err
			}
			if systemInstruction == nil {
				systemInstruction = &genai.Content{Role: "user", Parts: parts}
			} else {
				systemInstruction.Parts = append(systemInstruction.Parts, parts...)
			}
		case messages.RoleHuman:
			flushToolParts()
			parts, err := contentParts(message)
			if err != nil {
				return nil, nil, err
			}
			contents = append(contents, &genai.Content{Role: "user", Parts: parts})
		case messages.RoleAI:
			flushToolParts()
			parts, err := contentParts(message)
			if err != nil {
				return nil, nil, err
			}
			for _, call := range message.ToolCalls {
				if call.ID != "" && call.Name != "" {
					toolCallNames[call.ID] = call.Name
				}
				parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{
					ID:   call.ID,
					Name: call.Name,
					Args: call.Args,
				}})
			}
			if len(parts) == 0 {
				// The API rejects a Content without parts; an empty model turn
				// still needs a placeholder (chat_models.py:1590-1604).
				parts = []*genai.Part{{Text: ""}}
			}
			contents = append(contents, &genai.Content{Role: "model", Parts: parts})
		case messages.RoleTool:
			parts, err := toolResponseParts(message, toolCallNames)
			if err != nil {
				return nil, nil, err
			}
			pendingToolParts = append(pendingToolParts, parts...)
		default:
			return nil, nil, fmt.Errorf("gemini: unexpected message role %q", message.Role)
		}
	}
	flushToolParts()
	return systemInstruction, contents, nil
}

// contentParts serializes a message's text and content blocks into Gemini
// parts. Multimodal mapping (Python _convert_to_parts): image/audio/video/file
// blocks with inline bytes become inlineData parts (the SDK base64-encodes
// the bytes on the wire); blocks carrying a URL become fileData parts, which
// the Gemini API fetches server-side. An empty result is allowed here; the
// model role applies its own one-part placeholder.
func contentParts(message messages.Message) ([]*genai.Part, error) {
	var parts []*genai.Part
	if message.Content != "" {
		parts = append(parts, &genai.Part{Text: message.Content})
	}
	for _, block := range message.ContentBlocks {
		part, err := blockToPart(block)
		if err != nil {
			return nil, err
		}
		if part != nil {
			parts = append(parts, part)
		}
	}
	return parts, nil
}

func blockToPart(block messages.ContentBlock) (*genai.Part, error) {
	switch b := block.(type) {
	case messages.TextBlock:
		return &genai.Part{Text: b.Text}, nil
	case messages.ImageBlock:
		return dataPart("image", b.MimeType, b.Base64, b.URL)
	case messages.AudioBlock:
		return dataPart("audio", b.MimeType, b.Base64, b.URL)
	case messages.VideoBlock:
		return dataPart("video", b.MimeType, b.Base64, b.URL)
	case messages.FileBlock:
		return dataPart("file", b.MimeType, b.Base64, b.URL)
	case messages.ToolCallBlock:
		return &genai.Part{FunctionCall: &genai.FunctionCall{
			ID:   b.ID,
			Name: b.Name,
			Args: b.Args,
		}}, nil
	case messages.ReasoningBlock:
		return &genai.Part{Text: b.Reasoning, Thought: true}, nil
	case messages.ToolCallChunkBlock, messages.InvalidToolCallBlock:
		// Streaming-only shapes never appear on requests.
		return nil, nil
	default:
		return nil, fmt.Errorf("gemini: unsupported content block %T", block)
	}
}

// dataPart maps a data block (image/audio/video/file) onto inline_data or
// file_data depending on whether bytes or a URL is present. MimeType defaults
// follow the block kind so inline_data always carries a usable media type.
func dataPart(kind, mimeType, base64Data, url string) (*genai.Part, error) {
	if base64Data != "" {
		data, err := base64.StdEncoding.DecodeString(base64Data)
		if err != nil {
			return nil, fmt.Errorf("gemini: decode %s base64 data: %w", kind, err)
		}
		if mimeType == "" {
			mimeType = defaultMimeType(kind)
		}
		return &genai.Part{InlineData: &genai.Blob{Data: data, MIMEType: mimeType}}, nil
	}
	if url != "" {
		return &genai.Part{FileData: &genai.FileData{FileURI: url, MIMEType: mimeType}}, nil
	}
	return nil, fmt.Errorf("gemini: %s block carries neither base64 data nor a url", kind)
}

func defaultMimeType(kind string) string {
	switch kind {
	case "image":
		return "image/png"
	case "audio":
		return "audio/wav"
	case "video":
		return "video/mp4"
	default:
		return "application/octet-stream"
	}
}

// toolResponseParts converts a tool-result message into functionResponse
// parts, mirroring Python's _convert_tool_message_to_parts
// (chat_models.py:1049-1090): the content is JSON-parsed; a dict response is
// passed through as-is, anything else is wrapped as {"output": response}. The
// function name comes from the originating AI tool call (resolved via
// toolCallNames), falling back to the tool call ID.
func toolResponseParts(message messages.Message, toolCallNames map[string]string) ([]*genai.Part, error) {
	var parts []*genai.Part
	// Media blocks in tool results travel as their own parts alongside the
	// functionResponse (Python routes data blocks through _convert_to_parts).
	for _, block := range message.ContentBlocks {
		part, err := blockToPart(block)
		if err != nil {
			return nil, err
		}
		if part != nil {
			parts = append(parts, part)
		}
	}

	name := toolCallNames[message.ToolCallID]
	if name == "" {
		name = message.ToolCallID
	}
	var response map[string]any
	if message.Content != "" {
		var decoded map[string]any
		if err := json.Unmarshal([]byte(message.Content), &decoded); err == nil {
			response = decoded
		} else {
			response = map[string]any{"output": message.Content}
		}
	}
	if response == nil {
		response = map[string]any{}
	}
	parts = append(parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
		Name:     name,
		Response: response,
	}})
	return parts, nil
}

// responseToMessage converts a GenerateContentResponse into the core AI
// message, mirroring Python's _response_to_result /
// _parse_response_candidate (chat_models.py:1748-2026): text parts join into
// the content, functionCall parts become tool calls (the Gemini API returns
// no stable call ID, so one is synthesized when missing — Python's
// uuid4-based fallback, chat_models.py:1946-1948), thought parts become
// reasoning blocks, and usage maps through _response_to_result's formula
// (chat_models.py:2042-2070).
func responseToMessage(response *genai.GenerateContentResponse) (messages.Message, error) {
	if len(response.Candidates) == 0 {
		detail := ""
		if response.PromptFeedback != nil && response.PromptFeedback.BlockReason != "" {
			detail = fmt.Sprintf(" (prompt blocked: %s)", response.PromptFeedback.BlockReason)
		}
		return messages.Message{}, fmt.Errorf("gemini: response carries no candidates%s", detail)
	}

	candidate := response.Candidates[0]
	var textParts []string
	var toolCalls []messages.ToolCall
	var blocks []messages.ContentBlock
	callIndex := 0
	if candidate.Content != nil {
		for _, part := range candidate.Content.Parts {
			if part == nil {
				continue
			}
			switch {
			case part.FunctionCall != nil:
				call := messages.ToolCall{
					ID:   orSyntheticCallID(part.FunctionCall.ID, callIndex),
					Name: part.FunctionCall.Name,
					Args: part.FunctionCall.Args,
				}
				callIndex++
				toolCalls = append(toolCalls, call)
				blocks = append(blocks, messages.ToolCallBlock{
					ID:   call.ID,
					Name: call.Name,
					Args: call.Args,
				})
			case part.InlineData != nil && part.InlineData.MIMEType != "":
				encoded := base64.StdEncoding.EncodeToString(part.InlineData.Data)
				switch {
				case strings.HasPrefix(part.InlineData.MIMEType, "audio/"):
					blocks = append(blocks, messages.AudioBlock{
						MimeType: part.InlineData.MIMEType,
						Base64:   encoded,
					})
				case strings.HasPrefix(part.InlineData.MIMEType, "video/"):
					blocks = append(blocks, messages.VideoBlock{
						MimeType: part.InlineData.MIMEType,
						Base64:   encoded,
					})
				default:
					blocks = append(blocks, messages.ImageBlock{
						MimeType: part.InlineData.MIMEType,
						Base64:   encoded,
					})
				}
			case part.Text != "":
				if part.Thought {
					blocks = append(blocks, messages.ReasoningBlock{Reasoning: part.Text})
					continue
				}
				textParts = append(textParts, part.Text)
				blocks = append(blocks, messages.TextBlock{Text: part.Text})
			}
		}
	}

	message := messages.AI(strings.Join(textParts, ""))
	message.ID = response.ResponseID
	message.ToolCalls = toolCalls
	message.ContentBlocks = blocks
	message.ResponseMetadata = map[string]any{
		"model":          response.ModelVersion,
		"model_provider": providerName,
		"finish_reason":  string(candidate.FinishReason),
	}
	if candidate.FinishMessage != "" {
		message.ResponseMetadata["finish_message"] = candidate.FinishMessage
	}
	message.UsageMetadata = usageToMetadata(response.UsageMetadata)
	return message, nil
}

// usageToMetadata applies Python's _response_to_result token accounting
// (chat_models.py:2042-2070): input adds the server-side tool-use prompt
// tokens, output adds thought tokens, and cache/thought counts land in the
// input/output token details.
func usageToMetadata(usage *genai.GenerateContentResponseUsageMetadata) messages.UsageMetadata {
	if usage == nil {
		return messages.UsageMetadata{}
	}
	input := int(usage.PromptTokenCount) + int(usage.ToolUsePromptTokenCount)
	thoughts := int(usage.ThoughtsTokenCount)
	output := int(usage.CandidatesTokenCount) + thoughts
	meta := messages.UsageMetadata{
		InputTokens:  input,
		OutputTokens: output,
		TotalTokens:  int(usage.TotalTokenCount),
	}
	if usage.CachedContentTokenCount > 0 {
		if meta.InputTokenDetails == nil {
			meta.InputTokenDetails = &messages.InputTokenDetails{}
		}
		meta.InputTokenDetails.CacheReadInputTokens = int(usage.CachedContentTokenCount)
	}
	if thoughts > 0 {
		meta.OutputTokenDetails = &messages.OutputTokenDetails{
			ReasoningOutputTokens: thoughts,
		}
	}
	return meta
}

// orSyntheticCallID returns the API-provided function-call ID or synthesizes
// a stable one ("call_<index>") when the API omitted it — the Gemini API
// predates function-call IDs, so every adapter must invent them (Python
// generates a uuid4, chat_models.py:1946-1948; Go uses a per-message index so
// IDs stay deterministic and testable while still unique within the message).
func orSyntheticCallID(id string, index int) string {
	if id != "" {
		return id
	}
	return fmt.Sprintf("call_%d", index)
}
