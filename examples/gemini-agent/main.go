// Command gemini-agent runs a tool-calling agent on Google Gemini via
// partners/gemini (the official google.golang.org/genai SDK under the hood).
//
// Required env:
//
//	GEMINI_API_KEY   Gemini API key (GOOGLE_API_KEY is honored as a fallback,
//	                 matching the SDK's own precedence)
//
// When no key is configured the example prints an explanation and exits 0,
// so it can be shipped in keyless CI.
//
// Usage:
//
//	go run ./examples/gemini-agent
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/partners/gemini"
)

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		// partners/gemini falls back to GOOGLE_API_KEY when no explicit key
		// is configured (GOOGLE_API_KEY wins when both are set, matching the
		// SDK), so honor the same fallback before declaring the key missing.
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		fmt.Println("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
		fmt.Println("Get a key at https://aistudio.google.com/apikey, then:")
		fmt.Println("  GEMINI_API_KEY=... go run ./examples/gemini-agent")
		return
	}

	// One local tool so the run exercises a genuine tool-call round trip.
	wordcount, err := coretools.NewSimple("count_words", "counts the words in the input text",
		func(_ context.Context, text string) (coretools.Result, error) {
			n := len(strings.Fields(text))
			return coretools.Result{Content: fmt.Sprintf("%d words", n)}, nil
		})
	if err != nil {
		fmt.Println("build count_words tool:", err)
		return
	}

	model := gemini.NewChatModel(
		modelconfig.WithAPIKey(apiKey),
		modelconfig.WithModel("gemini-2.5-flash"),
	)

	agent, err := agents.CreateAgent(model, []coretools.Tool{wordcount},
		agents.WithAgentSystemPrompt("You are a concise assistant. Use the count_words tool for any word-count question."),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}

	out, err := agent.Invoke(ctx, []messages.Message{
		messages.Human("How many words are in 'the quick brown fox jumps over the lazy dog'?"),
	})
	if err != nil {
		fmt.Println("invoke:", err)
		return
	}
	for _, m := range out {
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  %-8s tool_call: %s(%v)\n", m.Role, m.ToolCalls[0].Name, m.ToolCalls[0].Args)
			continue
		}
		fmt.Printf("  %-8s %s\n", m.Role, messages.Text(m))
	}
}
