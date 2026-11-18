// Command usaco_usaco is the Go port of examples/usaco/usaco.ipynb.
//
// The notebook downloads a USACO dataset, asks an Anthropic model to solve a
// programming problem with a `writePython` tool, and evaluates submissions by
// running generated Python in a subprocess against test cases. It shows two
// graphs: a basic solver <-> evaluate loop and a recall-augmented
// draft -> retrieve -> solve -> evaluate loop with an interrupt/resume
// walkthrough. This port keeps both graph shapes and the tool-call protocol
// but replaces execution with a safe in-memory emulator: generated code is
// NEVER run, no subprocess is spawned, and no shell command is executed.
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

// scriptedModel is a deterministic offline ChatModel double.
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
	return language.ChatModelCapabilities{ToolCalling: true}
}

type testCase struct {
	Inputs  string
	Outputs string
}

// problem is the offline stand-in for one row of the downloaded dataset.
type problem struct {
	Description  string
	TestCases    []testCase
	ProblemLevel string
}

var sampleProblem = problem{
	Description:  "Read a greeting and print it back.",
	ProblemLevel: "Bronze",
	TestCases: []testCase{
		{Inputs: "greeting", Outputs: "hello, world!"},
		{Inputs: "greeting", Outputs: "hello, world!"},
		{Inputs: "greeting", Outputs: "hello, world!"},
	},
}

// exampleCorpus is the offline stand-in for the BM25 retriever's training
// problems. The retrieve node returns the first entry deterministically.
var exampleCorpus = []string{
	"<problem>Print a fixed line to standard output.</problem>\n<solution>Use a single print statement with the exact expected text.</solution>",
	"<problem>Read two numbers and print their sum.</problem>\n<solution>Parse the inputs, add them, and print the result.</solution>",
}

// emulateRun is the safe replacement for check_correctness/subprocess. It
// never executes code: it reads a `# OUTPUT:` marker comment and treats that
// as the program's stdout. This lets the retry graph be exercised offline.
func emulateRun(code, _ string) string {
	for _, line := range strings.Split(code, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "# OUTPUT:") {
			return strings.TrimSpace(strings.TrimPrefix(trimmed, "# OUTPUT:"))
		}
	}
	return ""
}

// writePythonCall builds the tool call the notebook's solver emits.
func writePythonCall(id, code string) messages.Message {
	return messages.Message{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
		ID:   id,
		Name: "writePython",
		Args: map[string]any{
			"reasoning":  "Offline deterministic solution.",
			"pseudocode": "Read input, then print the expected line.",
			"code":       code,
		},
	}}}
}

func codeFromMessage(m messages.Message) string {
	if len(m.ToolCalls) == 0 {
		return ""
	}
	code, _ := m.ToolCalls[0].Args["code"].(string)
	return code
}

func msgsOf(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

func countToolCalls(msgs []messages.Message) int {
	count := 0
	for _, m := range msgs {
		if m.Role == messages.RoleAI && len(m.ToolCalls) > 0 {
			count++
		}
	}
	return count
}

// evaluator mirrors the notebook's evaluate node without running code.
func evaluateNode(_ runtime.Runtime, state map[string]any) (any, error) {
	fmt.Println("---EVALUATE---")
	msgs := msgsOf(state)
	if len(msgs) == 0 || len(msgs[len(msgs)-1].ToolCalls) == 0 {
		return map[string]any{
			"messages": []messages.Message{messages.Human("No code submitted. Please try again using the correct Python code.")},
			"status":   "in_progress",
		}, nil
	}
	last := msgs[len(msgs)-1]
	code := codeFromMessage(last)
	cases, _ := state["testCases"].([]testCase)
	succeeded := 0
	results := make([]string, 0, len(cases))
	for i, tc := range cases {
		if emulateRun(code, tc.Inputs) == tc.Outputs {
			succeeded++
			results = append(results, fmt.Sprintf("<test id=%d>\npassed\n</test>", i))
		} else {
			results = append(results, fmt.Sprintf("<test id=%d>\nwrong answer. Expected '%s', got '%s'\n</test>", i, tc.Outputs, emulateRun(code, tc.Inputs)))
		}
	}
	if len(cases) > 0 && succeeded == len(cases) {
		fmt.Println("  all tests passed")
		return map[string]any{"status": "success"}, nil
	}
	response := fmt.Sprintf("Incorrect submission. Pass rate: %d/%d\nResults:\n%s", succeeded, len(cases), strings.Join(results, "\n"))
	toolMsg := messages.Tool(last.ToolCalls[0].ID, response+"\nMake all fixes using the writePython tool.")
	toolMsg.Name = "writePython"
	fmt.Printf("  pass rate %d/%d\n", succeeded, len(cases))
	return map[string]any{
		"messages": []messages.Message{toolMsg},
		"status":   "in_progress",
	}, nil
}

func controlEdge(retryNode string) graphpkg.ConditionalEdge {
	return func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		if status, _ := state["status"].(string); status == "success" {
			return graphpkg.To(types.END), nil
		}
		return graphpkg.To(retryNode), nil
	}
}

// buildBasicGraph mirrors the notebook's first graph: solver <-> evaluate.
func buildBasicGraph() *graphpkg.CompiledGraph {
	model := &scriptedModel{answer: func(_ context.Context, input []messages.Message) (messages.Message, error) {
		id := "call_solve"
		if countToolCalls(input) == 0 {
			return writePythonCall(id, "# OUTPUT: goodbye\nprint('goodbye')"), nil
		}
		return writePythonCall(id, "# OUTPUT: hello, world!\nprint('hello, world!')"), nil
	}}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("solver", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---SOLVER---")
		resp, err := model.Invoke(context.Background(), msgsOf(state))
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": []messages.Message{resp}}, nil
	})
	g.AddNode("evaluate", evaluateNode)
	g.AddEdge(types.START, "solver")
	g.AddEdge("solver", "evaluate")
	g.AddConditionalEdges("evaluate", controlEdge("solver"))
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile basic usaco graph: %v", err)
	}
	return cg
}

// buildRecallGraph mirrors the notebook's second graph with an offline BM25
// substitute and a MemorySaver checkpointer.
func buildRecallGraph() *graphpkg.CompiledGraph {
	draftModel := &scriptedModel{answer: func(_ context.Context, _ []messages.Message) (messages.Message, error) {
		return writePythonCall("call_draft", "# OUTPUT: goodbye\nprint('draft attempt')"), nil
	}}
	solveModel := &scriptedModel{answer: func(_ context.Context, input []messages.Message) (messages.Message, error) {
		if countToolCalls(input) == 0 {
			return writePythonCall("call_solve", "# OUTPUT: goodbye\nprint('first attempt')"), nil
		}
		return writePythonCall("call_solve", "# OUTPUT: hello, world!\nprint('hello, world!')"), nil
	}}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("draft", func(rt runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---DRAFT---")
		resp, err := draftModel.Invoke(rt, msgsOf(state))
		if err != nil {
			return nil, err
		}
		return map[string]any{"candidate": resp}, nil
	})
	g.AddNode("retrieve", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---RETRIEVE---")
		candidate, _ := state["candidate"].(messages.Message)
		if len(candidate.ToolCalls) == 0 {
			return nil, fmt.Errorf("draft did not produce a code tool call")
		}
		fmt.Printf("  retrieved %d offline example(s)\n", len(exampleCorpus))
		return map[string]any{"examples": strings.Join(exampleCorpus, "\n\n")}, nil
	})
	g.AddNode("solve", func(rt runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---SOLVE---")
		resp, err := solveModel.Invoke(rt, msgsOf(state))
		if err != nil {
			return nil, err
		}
		return map[string]any{"messages": []messages.Message{resp}}, nil
	})
	g.AddNode("evaluate", evaluateNode)
	g.AddEdge(types.START, "draft")
	g.AddEdge("draft", "retrieve")
	g.AddEdge("retrieve", "solve")
	g.AddEdge("solve", "evaluate")
	g.AddConditionalEdges("evaluate", controlEdge("solve"))
	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile recall usaco graph: %v", err)
	}
	return cg
}

func initialState() map[string]any {
	return map[string]any{
		"messages":     []messages.Message{messages.Human(sampleProblem.Description)},
		"testCases":    sampleProblem.TestCases,
		"status":       "in_progress",
		"problemLevel": sampleProblem.ProblemLevel,
	}
}

func runBasic(ctx context.Context) {
	fmt.Println("===== BASIC SOLVER/EVALUATE GRAPH =====")
	result, err := buildBasicGraph().Invoke(ctx, initialState())
	if err != nil {
		log.Fatalf("invoke basic usaco graph: %v", err)
	}
	fmt.Printf("status: %v\n", result.Values["status"])
}

func runRecall(ctx context.Context) {
	fmt.Println("\n===== RECALL-AUGMENTED GRAPH =====")
	result, err := buildRecallGraph().InvokeWithOptions(ctx, initialState(), graphpkg.Options{ThreadID: "question-recall"})
	if err != nil {
		log.Fatalf("invoke recall usaco graph: %v", err)
	}
	fmt.Printf("status: %v\n", result.Values["status"])
	fmt.Printf("examples retrieved: %v\n", result.Values["examples"] != "")
}

func main() {
	ctx := context.Background()
	runBasic(ctx)
	runRecall(ctx)
}
