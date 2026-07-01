package agents

import (
	"regexp"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
)

var fallbackModelsWithStructuredOutput = []*regexp.Regexp{
	regexp.MustCompile(`(^|[/:.])gpt-4\.1($|[-/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-4o($|[-/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-5($|[-/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-5\.1($|[-/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-5\.2(-\d{4}-\d{2}-\d{2})?($|[/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-5\.2-(chat|codex)($|[-/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-5\.3($|[-/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-5\.4(-\d{4}-\d{2}-\d{2})?($|[/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-5\.4-(mini|nano)($|[-/:])`),
	regexp.MustCompile(`(^|[/:.])gpt-5\.5($|[-/:])`),
	regexp.MustCompile(`(^|[/:.])claude-(fable|mythos)-5(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])`),
	regexp.MustCompile(`(^|[/:.])claude-haiku-4-5(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])`),
	regexp.MustCompile(`(^|[/:.])claude-opus-4-(5|6|7|8)(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])`),
	regexp.MustCompile(`(^|[/:.])claude-sonnet-4-(5|6)(?:-\d{8})?(?:-v\d(?::\d)?)?($|[/:])`),
	regexp.MustCompile(`(^|[/:.])grok-4($|[-.:/])`),
	regexp.MustCompile(`(^|[/:.])grok-build($|[-/:])`),
}

type ModelInfo struct {
	ModelName string
	Model     string
	ModelID   string
	Profile   map[string]any
}

func FetchLastAIAndToolMessages(batch []messages.Message) (*messages.Message, []messages.Message) {
	for i := len(batch) - 1; i >= 0; i-- {
		if batch[i].Role != messages.RoleAI {
			continue
		}

		toolMessages := make([]messages.Message, 0)
		for _, message := range batch[i+1:] {
			if message.Role == messages.RoleTool {
				toolMessages = append(toolMessages, message)
			}
		}

		aiMessage := batch[i]
		return &aiMessage, toolMessages
	}

	return nil, nil
}

func SupportsProviderStrategy(model any, tools []any) bool {
	modelName := ""

	switch typed := model.(type) {
	case string:
		modelName = typed
	case ModelInfo:
		modelName = firstNonEmpty(typed.ModelName, typed.Model, typed.ModelID)
		if profileSupportsStructuredOutput(typed.Profile) && !blocksStructuredOutputWithTools(modelName, tools) {
			return true
		}
	case *ModelInfo:
		if typed == nil {
			return false
		}
		modelName = firstNonEmpty(typed.ModelName, typed.Model, typed.ModelID)
		if profileSupportsStructuredOutput(typed.Profile) && !blocksStructuredOutputWithTools(modelName, tools) {
			return true
		}
	}

	if modelName == "" {
		return false
	}

	modelLower := strings.ToLower(modelName)
	for _, pattern := range fallbackModelsWithStructuredOutput {
		if pattern.MatchString(modelLower) {
			return true
		}
	}
	return false
}

func profileSupportsStructuredOutput(profile map[string]any) bool {
	value, ok := profile["structured_output"]
	if !ok {
		return false
	}
	supported, ok := value.(bool)
	return ok && supported
}

func blocksStructuredOutputWithTools(modelName string, tools []any) bool {
	if len(tools) == 0 {
		return false
	}
	modelLower := strings.ToLower(modelName)
	return strings.Contains(modelLower, "gemini") && !strings.Contains(modelLower, "gemini-3")
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
