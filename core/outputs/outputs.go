package outputs

import "github.com/projanvil/langchain-golang/core/messages"

// Generation is a single text generation.
type Generation struct {
	Text           string         `json:"text"`
	GenerationInfo map[string]any `json:"generation_info,omitempty"`
	Type           string         `json:"type"`
}

func NewGeneration(text string, info map[string]any) Generation {
	return Generation{Text: text, GenerationInfo: cloneMap(info), Type: "Generation"}
}

// GenerationChunk can be concatenated with another chunk.
type GenerationChunk = Generation

func MergeGenerationChunks(chunks ...Generation) Generation {
	out := NewGeneration("", nil)
	for _, chunk := range chunks {
		out.Text += chunk.Text
		out.GenerationInfo = MergeMaps(out.GenerationInfo, chunk.GenerationInfo)
	}
	out.Type = "Generation"
	return out
}

// ChatGeneration is a generation containing a chat message.
type ChatGeneration struct {
	Generation
	Message messages.Message `json:"message"`
	Type    string           `json:"type"`
}

func NewChatGeneration(message messages.Message, info map[string]any) ChatGeneration {
	return ChatGeneration{
		Generation: NewGeneration(messageText(message), info),
		Message:    message,
		Type:       "ChatGeneration",
	}
}

// LLMResult is the batched model output shape used by callbacks and tracers.
type LLMResult struct {
	Generations [][]Generation `json:"generations"`
	LLMOutput   map[string]any `json:"llm_output,omitempty"`
	Run         []RunInfo      `json:"run,omitempty"`
	ModelName   string         `json:"model_name,omitempty"`
}

type ChatResult struct {
	Generations []ChatGeneration `json:"generations"`
	LLMOutput   map[string]any   `json:"llm_output,omitempty"`
}

type RunInfo struct {
	RunID string `json:"run_id"`
}

func messageText(message messages.Message) string {
	if message.Content != "" {
		return message.Content
	}
	text := ""
	for _, block := range message.ContentBlocks {
		if block["type"] == "text" || block["type"] == nil {
			if value, ok := block["text"].(string); ok {
				text += value
			}
		}
	}
	return text
}

func MergeMaps(base map[string]any, overlays ...map[string]any) map[string]any {
	out := cloneMap(base)
	for _, overlay := range overlays {
		for key, value := range overlay {
			if existing, ok := out[key].(map[string]any); ok {
				if next, ok := value.(map[string]any); ok {
					out[key] = MergeMaps(existing, next)
					continue
				}
			}
			out[key] = value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func cloneMap(values map[string]any) map[string]any {
	if values == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}
