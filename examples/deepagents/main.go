// Command deepagents runs a Deep Agent — a tool-calling agent with a state
// filesystem, an ephemeral subagent, and context summarization — assembled
// by deepagents.CreateDeepAgent.
//
// The chat model comes from partners/openaicompat (the OpenAI-compatible
// provider registry); groq is used here, keyed off GROQ_API_KEY like the
// registry's own env handling. Set GROQ_API_BASE to override the endpoint
// (e.g. a local OpenAI-compatible server).
//
// Required env:
//
//	GROQ_API_KEY   Groq API key (the compat factory reads it itself; the
//	               example only checks it to print a friendly skip note)
//
// When no key is configured the example prints an explanation and exits 0,
// so it can ship in keyless CI.
//
// Usage:
//
//	go run ./examples/deepagents
package main

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/deepagents"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	deepmw "github.com/projanvil/langchain-golang/deepagents/middleware"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"

	// Activate the OpenAI-compatible providers (groq, mistralai, deepseek,
	// xai, openrouter, fireworks, perplexity) in the chatmodels registry.
	_ "github.com/projanvil/langchain-golang/partners/openaicompat"
)

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		fmt.Println("GROQ_API_KEY is not set; skipping the live deepagents example.")
		fmt.Println("Get a key at https://console.groq.com/keys, then:")
		fmt.Println("  GROQ_API_KEY=... go run ./examples/deepagents")
		return
	}

	model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
		Provider: "groq",
		Model:    "openai/gpt-oss-20b",
	})
	if err != nil {
		fmt.Println("resolve model:", err)
		return
	}

	// A Deep Agent over the default state backend: files live in agent
	// state (no disk), the seven filesystem tools and the task tool are
	// bound automatically, and one declarative subagent (a nil Model
	// inherits the parent's) is available for delegation alongside the
	// auto-added general-purpose subagent.
	agent, err := deepagents.CreateDeepAgent(model,
		deepagents.WithSystemPrompt("You are a concise research assistant. "+
			"Write files under /notes/ and delegate deep dives to the researcher subagent."),
		deepagents.WithSubAgents([]any{deepmw.SubAgentSpec{
			Name:        "researcher",
			Description: "Researches a topic and reports a two-sentence summary. Use for any question needing investigation.",
		}}),
	)
	if err != nil {
		fmt.Println("create deep agent:", err)
		return
	}

	values, err := agent.InvokeWithState(ctx, []messages.Message{messages.Human(
		"Ask the researcher subagent what makes the Go programming language distinctive, " +
			"then write its two-sentence answer to /notes/go.md and tell me the file is saved.")})
	if err != nil {
		fmt.Println("invoke:", err)
		return
	}

	fmt.Println("--- transcript ---")
	msgs, _ := values["messages"].([]messages.Message)
	for _, msg := range msgs {
		switch msg.Role {
		case messages.RoleTool:
			fmt.Printf("tool  %s: %s\n", toolLabel(msg), firstLine(msg.Content))
		case messages.RoleAI:
			if len(msg.ToolCalls) > 0 {
				names := make([]string, 0, len(msg.ToolCalls))
				for _, call := range msg.ToolCalls {
					names = append(names, call.Name)
				}
				fmt.Printf("ai    (calls: %s)\n", strings.Join(names, ", "))
			}
			if msg.Content != "" {
				fmt.Printf("ai    %s\n", firstLine(msg.Content))
			}
		default:
			fmt.Printf("%-5s %s\n", msg.Role, firstLine(msg.Content))
		}
	}

	fmt.Println("--- files ---")
	files, _ := values[backends.FilesStateKey].(map[string]backends.FileMeta)
	paths := make([]string, 0, len(files))
	for path := range files {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		fmt.Println("(none)")
	}
	for _, path := range paths {
		fmt.Printf("%s: %s\n", path, firstLine(files[path].Content))
	}
}

// firstLine collapses a (possibly multi-line) tool or file content to its
// first line for a compact transcript.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i] + " …"
	}
	return s
}

// toolLabel renders a tool message's tool name when the adapter recorded it.
func toolLabel(msg messages.Message) string {
	if msg.Name != "" {
		return msg.Name
	}
	return msg.ToolCallID
}
