// Command code_assistant_langgraph_code_assistant is the Go port of
// examples/code_assistant/langgraph_code_assistant.ipynb.
//
// The notebook crawls the LCEL documentation with RecursiveUrlLoader, asks a
// hosted model for a structured {prefix, imports, code} solution, and runs
// `exec` on the generated Python to decide whether to retry. This port keeps
// the generate -> check_code -> (reflect | generate) -> finish graph but
// substitutes both sides: an offline scripted ChatModel produces deterministic
// code solutions from a tiny in-memory documentation corpus, and the "exec"
// checks are safe in-memory string inspections. NO generated code is executed.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	checkpoint "github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// codeSolution mirrors the notebook's `code` Pydantic model.
type codeSolution struct {
	Prefix  string
	Imports string
	Code    string
}

func (s codeSolution) String() string {
	return fmt.Sprintf("%s\nImports: %s\nCode: %s", s.Prefix, s.Imports, s.Code)
}

// offlineCorpus is the deterministic stand-in for the crawled LCEL docs. The
// real notebook builds this by fetching python.langchain.com; here it is a
// small set of fixed strings so the program is fully offline.
var offlineCorpus = []string{
	"LCEL composes runnables with the pipe operator: prompt | model | parser.",
	"A RAG chain retrieves documents and passes them into a prompt template.",
	"RunnablePassthrough forwards its input unchanged to the next step.",
}

// generateCode is the offline replacement for `code_gen_chain.invoke`. It
// returns a deliberately broken import on the first attempt and a clean
// solution afterwards, so the retry loop is exercised deterministically.
func generateCode(question string, iterations int, hadError bool) codeSolution {
	if iterations == 0 {
		return codeSolution{
			Prefix:  "First draft: a minimal LCEL RAG chain (with a placeholder import).",
			Imports: "import nonexistent_lcel_helper\nfrom langchain_core.runnables import RunnablePassthrough",
			Code:    "chain = RunnablePassthrough() | retriever | prompt | model\nprint(chain)",
		}
	}
	if hadError {
		return codeSolution{
			Prefix:  "Revised solution: drop the placeholder import and keep only documented symbols.",
			Imports: "from langchain_core.runnables import RunnablePassthrough",
			Code:    "chain = RunnablePassthrough() | retriever | prompt | model\nprint(chain)",
		}
	}
	return codeSolution{
		Prefix:  "Solution assembled from the offline LCEL corpus.",
		Imports: "from langchain_core.runnables import RunnablePassthrough",
		Code:    "chain = RunnablePassthrough() | retriever | prompt | model\nprint(chain)",
	}
}

// checkCodeSafe is the offline replacement for `exec(imports)` and
// `exec(imports + "\n" + code)`. It never evaluates generated text: it only
// checks against a fixed allowlist of documented imports and required tokens.
func checkCodeSafe(s codeSolution) (string, bool) {
	for _, line := range strings.Split(s.Imports, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.Contains(line, "nonexistent") || strings.Contains(line, "os.system") {
			return "import test failed: unknown or unsafe symbol in " + line, false
		}
	}
	if !strings.Contains(s.Code, "chain") || !strings.Contains(s.Code, "print") {
		return "code execution test failed: expected a chain assignment and a print call", false
	}
	return "", true
}

func msgsOf(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

func intOf(state map[string]any, key string) int {
	v, _ := state[key].(int)
	return v
}

func generateNode() graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---GENERATING CODE SOLUTION---")
		msgs := msgsOf(state)
		iterations := intOf(state, "iterations")
		errorFlag, _ := state["error"].(string)

		// The messages reducer appends, so nodes return only NEW messages.
		updates := []messages.Message{}
		if errorFlag == "yes" {
			updates = append(updates, messages.Human(
				"Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:"))
		}
		question := ""
		if len(msgs) > 0 {
			question = messages.Text(msgs[0])
		}
		solution := generateCode(question, iterations, errorFlag == "yes")
		updates = append(updates, messages.AI(solution.String()))
		fmt.Printf("  corpus size (offline): %d documents\n", len(offlineCorpus))
		return map[string]any{
			"generation": solution,
			"messages":   updates,
			"iterations": iterations + 1,
		}, nil
	}
}

func checkCodeNode(_ runtime.Runtime, state map[string]any) (any, error) {
	fmt.Println("---CHECKING CODE---")
	solution, _ := state["generation"].(codeSolution)
	errText, ok := checkCodeSafe(solution)
	if !ok {
		fmt.Println("---CODE TEST: FAILED---")
		return map[string]any{
			"messages": []messages.Message{messages.Human("Your solution failed the offline check: " + errText)},
			"error":    "yes",
		}, nil
	}
	fmt.Println("---NO CODE TEST FAILURES---")
	return map[string]any{"error": "no"}, nil
}

func reflectNode(_ runtime.Runtime, state map[string]any) (any, error) {
	fmt.Println("---REFLECTING ON ERROR---")
	return map[string]any{
		"messages": []messages.Message{messages.AI(
			"Reflection: the failing import was not in the offline corpus; remove it and keep the documented symbols.")},
	}, nil
}

// newGraph builds the notebook's graph. reflectMode selects the branch the
// notebook controls with its `flag` variable ("reflect" vs "do not reflect").
func newGraph(reflectMode bool) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("generate", generateNode())
	g.AddNode("check_code", checkCodeNode)
	g.AddNode("reflect", reflectNode)
	g.AddEdge(types.START, "generate")
	g.AddEdge("generate", "check_code")
	g.AddConditionalEdges("check_code", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		errorFlag, _ := state["error"].(string)
		iterations := intOf(state, "iterations")
		if errorFlag == "no" || iterations >= 3 {
			fmt.Println("---DECISION: FINISH---")
			return graphpkg.To(types.END), nil
		}
		fmt.Println("---DECISION: RE-TRY SOLUTION---")
		if reflectMode {
			return graphpkg.To("reflect"), nil
		}
		return graphpkg.To("generate"), nil
	})
	g.AddEdge("reflect", "generate")
	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile code assistant graph: %v", err)
	}
	return cg
}

func run(label string, cg *graphpkg.CompiledGraph, reflectMode bool) {
	ctx := context.Background()
	question := "How can I directly pass a string to a runnable and use it to construct the input needed for my prompt?"
	fmt.Printf("\n===== %s =====\n", label)
	result, err := cg.InvokeWithOptions(ctx, map[string]any{
		"messages":   []messages.Message{messages.Human(question)},
		"iterations": 0,
		"error":      "",
	}, graphpkg.Options{ThreadID: label})
	if err != nil {
		log.Fatalf("invoke %s: %v", label, err)
	}
	solution, _ := result.Values["generation"].(codeSolution)
	fmt.Printf("iterations: %d, error: %v, reflect: %v\n", intOf(result.Values, "iterations"), result.Values["error"], reflectMode)
	fmt.Println("final generation:")
	fmt.Println(solution.String())
}

func main() {
	run("no-reflect", newGraph(false), false)
	run("reflect", newGraph(true), true)
}
