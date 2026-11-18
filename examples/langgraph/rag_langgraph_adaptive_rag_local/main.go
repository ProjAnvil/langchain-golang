// Command rag_langgraph_adaptive_rag_local is the Go port of
// langgraph/examples/rag/langgraph_adaptive_rag_local.ipynb.
//
// The notebook uses the same Adaptive RAG graph as the hosted variant but
// swaps OpenAI embeddings/ChatOpenAI and Nomic local embeddings/ChatOllama
// for a local model stack. The Go port keeps the identical graph topology and
// replaces those models with deterministic offline functions; retrieval uses
// a built-in in-memory corpus.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/vectorstores"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// corpus is a deterministic stand-in for the three web pages loaded in the
// notebook. Keeping it in-process removes the network dependency.
var corpus = []struct {
	id, source, content string
}{
	{
		id:     "agent-memory",
		source: "blog/2023-06-23-agent",
		content: "LLM agents use different types of memory. Short-term memory is the " +
			"context window. Long-term memory stores information across sessions " +
			"using an external vector store. Autonomous agents combine planning, " +
			"memory and tool use to solve tasks.",
	},
	{
		id:     "prompt-engineering",
		source: "blog/2023-03-15-prompt-engineering",
		content: "Prompt engineering techniques include few-shot prompting, chain of " +
			"thought prompting and self-consistency. Chain of thought prompting asks " +
			"the model to reason step by step before answering. Prompt design " +
			"improves reasoning quality.",
	},
	{
		id:     "adversarial-attacks",
		source: "blog/2023-10-25-adv-attack-llm",
		content: "Adversarial attacks on LLMs include token manipulation, jailbreak " +
			"prompting, gradient based attacks and prompt injection. Defenses include " +
			"input filtering and red teaming.",
	},
}

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true, "was": true,
	"of": true, "to": true, "in": true, "on": true, "for": true, "and": true,
	"or": true, "what": true, "how": true, "does": true, "do": true, "about": true,
	"with": true, "different": true, "types": true, "type": true,
}

// tokenSet lowercases text, keeps alpha-numeric words and drops stop words.
func tokenSet(s string) map[string]bool {
	set := map[string]bool{}
	for _, tok := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	}) {
		if len(tok) > 1 && !stopWords[tok] {
			set[tok] = true
		}
	}
	return set
}

func overlap(a, b map[string]bool) int {
	n := 0
	for k := range a {
		if b[k] {
			n++
		}
	}
	return n
}

// newRetriever builds the deterministic in-memory retriever. NewFake hashes
// tokens into vector buckets, so cosine similarity acts like keyword overlap.
func newRetriever() retrievers.Retriever {
	store := vectorstores.NewInMemory(embeddings.NewFake(64))
	docs := make([]documents.Document, 0, len(corpus))
	for _, item := range corpus {
		docs = append(docs, documents.New(item.content,
			map[string]any{"source": item.source}).WithID(item.id))
	}
	if _, err := store.AddDocuments(context.Background(), docs); err != nil {
		log.Fatalf("add documents: %v", err)
	}
	r, err := retrievers.AsRetriever(store, retrievers.WithSearchKwargs(map[string]any{"k": 3}))
	if err != nil {
		log.Fatalf("as retriever: %v", err)
	}
	return r
}

func asDocs(v any) []documents.Document {
	docs, _ := v.([]documents.Document)
	return docs
}

func docText(docs []documents.Document) string {
	parts := make([]string, 0, len(docs))
	for _, d := range docs {
		parts = append(parts, d.PageContent)
	}
	return strings.Join(parts, "\n")
}

func inDomain(q string) bool {
	t := tokenSet(q)
	for _, kw := range []string{"agent", "memory", "prompt", "chain", "thought", "adversarial", "attack", "jailbreak", "llm"} {
		if t[kw] {
			return true
		}
	}
	return false
}

func retrieveNode(r retrievers.Retriever) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		q, _ := state["question"].(string)
		fmt.Println("---RETRIEVE---")
		docs, err := r.GetRelevantDocuments(context.Background(), q)
		if err != nil {
			return nil, err
		}
		for i, d := range docs {
			fmt.Printf("  %d. %s\n", i+1, d.Source())
		}
		return map[string]any{"documents": docs, "question": q}, nil
	}
}

func gradeDocumentsNode(_ runtime.Runtime, state map[string]any) (any, error) {
	q, _ := state["question"].(string)
	docs := asDocs(state["documents"])
	fmt.Println("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
	qt := tokenSet(q)
	kept := make([]documents.Document, 0, len(docs))
	for _, d := range docs {
		if overlap(qt, tokenSet(d.PageContent)) > 0 {
			fmt.Println("---GRADE: DOCUMENT RELEVANT---")
			kept = append(kept, d)
		} else {
			fmt.Println("---GRADE: DOCUMENT NOT RELEVANT---")
		}
	}
	return map[string]any{"documents": kept, "question": q}, nil
}

func generateNode(_ runtime.Runtime, state map[string]any) (any, error) {
	q, _ := state["question"].(string)
	docs := asDocs(state["documents"])
	fmt.Println("---GENERATE---")
	contextText := docText(docs)
	answer := fmt.Sprintf("Regarding %q: %s", q, summarize(contextText))
	return map[string]any{"documents": docs, "question": q, "generation": answer}, nil
}

func summarize(text string) string {
	flat := strings.Join(strings.Fields(text), " ")
	if len(flat) > 160 {
		return flat[:160] + "..."
	}
	if flat == "" {
		return "no supporting context was found."
	}
	return flat
}

func transformQueryNode(_ runtime.Runtime, state map[string]any) (any, error) {
	q, _ := state["question"].(string)
	docs := asDocs(state["documents"])
	fmt.Println("---TRANSFORM QUERY---")
	better := "Explain " + q + " with details about LLM agents, memory, prompt engineering and adversarial attacks."
	return map[string]any{"documents": docs, "question": better}, nil
}

func webSearchNode(_ runtime.Runtime, state map[string]any) (any, error) {
	q, _ := state["question"].(string)
	fmt.Println("---WEB SEARCH---")
	doc := documents.New("Web search result: external sources discuss "+q+".",
		map[string]any{"source": "web_search"})
	return map[string]any{"documents": []documents.Document{doc}, "question": q}, nil
}

// routeQuestion replaces the structured-output router at START.
func routeQuestion(_ runtime.Runtime, state map[string]any) ([]any, error) {
	q, _ := state["question"].(string)
	fmt.Println("---ROUTE QUESTION---")
	if inDomain(q) {
		fmt.Println("---ROUTE QUESTION TO RAG (vectorstore)---")
		return graphpkg.To("retrieve"), nil
	}
	fmt.Println("---ROUTE QUESTION TO WEB SEARCH---")
	return graphpkg.To("web_search"), nil
}

func decideToGenerate(_ runtime.Runtime, state map[string]any) ([]any, error) {
	docs := asDocs(state["documents"])
	fmt.Println("---ASSESS GRADED DOCUMENTS---")
	if len(docs) == 0 {
		fmt.Println("---DECISION: ALL DOCUMENTS NOT RELEVANT, TRANSFORM QUERY---")
		return graphpkg.To("transform_query"), nil
	}
	fmt.Println("---DECISION: GENERATE---")
	return graphpkg.To("generate"), nil
}

// gradeGeneration checks hallucination then usefulness, mirroring the two
// structured graders in the notebook. Our generated text always contains
// question and context tokens, so the useful branch terminates the loop.
func gradeGeneration(_ runtime.Runtime, state map[string]any) ([]any, error) {
	docs := asDocs(state["documents"])
	gen, _ := state["generation"].(string)
	q, _ := state["question"].(string)
	fmt.Println("---CHECK HALLUCINATIONS---")
	if overlap(tokenSet(gen), tokenSet(docText(docs))) == 0 {
		fmt.Println("---DECISION: GENERATION NOT GROUNDED, RE-TRY---")
		return graphpkg.To("generate"), nil
	}
	fmt.Println("---GENERATION GROUNDED IN DOCUMENTS---")
	if overlap(tokenSet(gen), tokenSet(q)) == 0 {
		fmt.Println("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
		return graphpkg.To("transform_query"), nil
	}
	fmt.Println("---DECISION: GENERATION ADDRESSES QUESTION (useful)---")
	return graphpkg.To(types.END), nil
}

func buildGraph(r retrievers.Retriever) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddNode("web_search", webSearchNode)
	g.AddNode("retrieve", retrieveNode(r))
	g.AddNode("grade_documents", gradeDocumentsNode)
	g.AddNode("generate", generateNode)
	g.AddNode("transform_query", transformQueryNode)

	g.SetConditionalEntryPoint(routeQuestion)
	g.AddEdge("web_search", "generate")
	g.AddEdge("retrieve", "grade_documents")
	g.AddConditionalEdges("grade_documents", decideToGenerate)
	g.AddEdge("transform_query", "retrieve")
	g.AddConditionalEdges("generate", gradeGeneration)

	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}
	return cg
}

func run(ctx context.Context, cg *graphpkg.CompiledGraph, question string) {
	fmt.Printf("\n=== QUESTION: %s ===\n", question)
	result, err := cg.Invoke(ctx, map[string]any{"question": question})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	generation, _ := result.Values["generation"].(string)
	fmt.Printf("FINAL GENERATION: %s\n", generation)
}

func main() {
	ctx := context.Background()
	cg := buildGraph(newRetriever())
	run(ctx, cg, "What are the types of agent memory?")
	run(ctx, cg, "Who will the Bears draft first in the 2024 NFL draft?")
}
