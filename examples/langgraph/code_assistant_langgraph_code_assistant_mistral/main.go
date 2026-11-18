// Command code_assistant_langgraph_code_assistant_mistral is the Go port of
// examples/code_assistant/langgraph_code_assistant_mistral.ipynb.
//
// The notebook uses a Mistral chat model with structured output to produce a
// {prefix, imports, code} answer and wires generate -> check_code into a
// LangGraph retry loop, streaming values events with InMemorySaver backed
// threads. This port reproduces the graph and its streamed values view with an
// offline, function-backed ChatModel double. The notebook's `exec` checks are
// replaced by safe in-memory string inspection: generated code is NEVER run.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	checkpoint "github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// scriptedModel is a deterministic offline ChatModel double. Its answer
// function inspects the input messages and returns a fixed response; BindTools
// returns the same instance so no state is lost across a graph's re-binds.
type scriptedModel struct {
	answer func(ctx context.Context, input []messages.Message) (messages.Message, error)
}

func (m *scriptedModel) Invoke(ctx context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	return m.answer(ctx, input)
}

func (m *scriptedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		msg, err := m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = msg
	}
	return out, nil
}

func (m *scriptedModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	msg, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{msg}), nil
}

func (m *scriptedModel) InputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *scriptedModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *scriptedModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}
func (m *scriptedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true, StructuredOutput: true}
}

// solution is the Go equivalent of the notebook's `code` Pydantic model.
type solution struct {
	Prefix  string
	Imports string
	Code    string
}

func (s solution) String() string {
	return fmt.Sprintf("%s\nImports: %s\nCode: %s", s.Prefix, s.Imports, s.Code)
}

// offlineAnswer models `llm.with_structured_output(code)`. The first attempt
// for a question returns a bad import; later attempts return the clean answer.
func offlineAnswer(_ context.Context, input []messages.Message) (messages.Message, error) {
	attempts := 0
	question := ""
	for _, m := range input {
		if m.Role == messages.RoleAI && strings.Contains(m.Content, "Imports: ") {
			attempts++
		}
		if m.Role == messages.RoleHuman && question == "" {
			question = messages.Text(m)
		}
	}
	imports := "from langchain_core.runnables import RunnablePassthrough"
	if attempts == 0 {
		imports = "import nonexistent_mistral_helper"
	}
	content := "PREFIX: Offline solution for the requested program.\n" +
		"IMPORTS: " + imports + "\n" +
		"CODE: print('result for: " + truncate(question, 28) + "')"
	return messages.AI(content), nil
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

// parseSolution parses the offline model's structured text response.
func parseSolution(content string) solution {
	var out solution
	for _, line := range strings.Split(content, "\n") {
		switch {
		case strings.HasPrefix(line, "PREFIX: "):
			out.Prefix = strings.TrimPrefix(line, "PREFIX: ")
		case strings.HasPrefix(line, "IMPORTS: "):
			out.Imports = strings.TrimPrefix(line, "IMPORTS: ")
		case strings.HasPrefix(line, "CODE: "):
			out.Code = strings.TrimPrefix(line, "CODE: ")
		}
	}
	return out
}

// checkCodeSafe replaces `exec(imports)` / `exec(imports + code)`. It only
// inspects strings against an allowlist and never evaluates generated code.
func checkCodeSafe(s solution) string {
	for _, line := range strings.Split(s.Imports, "\n") {
		line = strings.TrimSpace(line)
		if line != "" && (strings.Contains(line, "nonexistent") || strings.Contains(line, "subprocess")) {
			return "failed import test: " + line
		}
	}
	if !strings.Contains(s.Code, "print(") {
		return "failed code execution test: expected a print statement"
	}
	return ""
}

func msgsOf(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

func newGraph(model *scriptedModel) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("generate", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---GENERATING CODE SOLUTION---")
		msgs := msgsOf(state)
		input := append([]messages.Message{messages.System("You are a coding assistant. Structure the answer as prefix, imports, code.")}, msgs...)
		resp, err := model.Invoke(context.Background(), input)
		if err != nil {
			return nil, err
		}
		sol := parseSolution(messages.Text(resp))
		// The messages reducer appends, so only the new AI message is returned.
		updates := []messages.Message{messages.AI("Here is my attempt to solve the problem: " + sol.String())}
		iterations, _ := state["iterations"].(int)
		return map[string]any{"generation": sol, "messages": updates, "iterations": iterations + 1}, nil
	})
	g.AddNode("check_code", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---CHECKING CODE---")
		sol, _ := state["generation"].(solution)
		if problem := checkCodeSafe(sol); problem != "" {
			fmt.Println("---CODE BLOCK CHECK: FAILED---")
			return map[string]any{
				"messages": []messages.Message{messages.Human(problem + ". Reflect on this error and return the FULL SOLUTION using the code tool.")},
				"error":    "yes",
			}, nil
		}
		fmt.Println("---NO CODE TEST FAILURES---")
		return map[string]any{"error": "no"}, nil
	})
	g.AddEdge(types.START, "generate")
	g.AddEdge("generate", "check_code")
	g.AddConditionalEdges("check_code", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		errorFlag, _ := state["error"].(string)
		iterations, _ := state["iterations"].(int)
		if errorFlag == "no" || iterations >= 3 {
			fmt.Println("---DECISION: FINISH---")
			return graphpkg.To(types.END), nil
		}
		fmt.Println("---DECISION: RE-TRY SOLUTION---")
		return graphpkg.To("generate"), nil
	})
	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile mistral code assistant graph: %v", err)
	}
	return cg
}

// printEvent is the Go form of the notebook's `_print_event`: show only the
// most recent message of each streamed state snapshot, once per message.
func printEvent(state map[string]any, printed map[string]bool) {
	msgs := msgsOf(state)
	if len(msgs) == 0 {
		return
	}
	last := msgs[len(msgs)-1]
	key := fmt.Sprintf("%s:%s", last.ID, messages.Text(last))
	if printed[key] {
		return
	}
	printed[key] = true
	text := messages.Text(last)
	if len(text) > 300 {
		text = text[:300] + " ... (truncated)"
	}
	fmt.Printf("  %-5s %s\n", last.Role, text)
}

func runExample(ctx context.Context, cg *graphpkg.CompiledGraph, thread, question string) {
	fmt.Printf("\n===== thread %s =====\nquestion: %s\n", thread, truncate(question, 70))
	printed := map[string]bool{}
	for chunk, err := range cg.Stream(ctx, map[string]any{
		"messages":   []messages.Message{messages.Human(question)},
		"iterations": 0,
	}, graphpkg.StreamOptions{
		Options: graphpkg.Options{ThreadID: thread},
		Modes:   []graphpkg.StreamMode{graphpkg.StreamValues},
	}) {
		if err != nil {
			log.Fatalf("stream %s: %v", thread, err)
		}
		state, ok := chunk.Payload.(map[string]any)
		if !ok {
			continue
		}
		printEvent(state, printed)
	}
}

func main() {
	ctx := context.Background()
	cg := newGraph(&scriptedModel{answer: offlineAnswer})

	questions := []string{
		"Write a Python program that prints 'Hello, World!' to the console.",
		"Write a program that checks if a given string is a palindrome.",
		"Write a program that prints the numbers from 1 to 100: FizzBuzz.",
	}
	for i, question := range questions {
		runExample(ctx, cg, fmt.Sprintf("mistral-thread-%d", i), question)
	}
}
