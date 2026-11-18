// Command rag_langgraph_self_rag_pinecone_movies is the Go port of
// langgraph/examples/rag/langgraph_self_rag_pinecone_movies.ipynb.
//
// The notebook runs the Self-RAG graph over a hosted Pinecone "sample-movies"
// index. The Go port keeps the identical graph topology but replaces the
// Pinecone store with a deterministic in-memory corpus of movie summaries and
// the OpenAI/LangSmith model calls with offline keyword-overlap functions.
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

// corpus is a deterministic stand-in for the Pinecone sample-movies index.
var corpus = []struct {
	id, title, content string
}{
	{
		id: "avatar", title: "Avatar",
		content: "Avatar is a science fiction film directed by James Cameron. A marine " +
			"explores the moon Pandora and encounters alien creatures and a native " +
			"species called the Na'vi.",
	},
	{
		id: "terminator", title: "The Terminator",
		content: "The Terminator is a science fiction film directed by James Cameron " +
			"about a cyborg assassin sent back in time and a fight against machines.",
	},
	{
		id: "aquaman", title: "Aquaman",
		content: "Aquaman is a superhero film starring Jason Momoa as Arthur Curry, " +
			"the king of Atlantis who protects the underwater realm.",
	},
	{
		id: "dune", title: "Dune",
		content: "Dune is a science fiction film featuring Jason Momoa as Duncan Idaho, " +
			"set on the desert planet Arrakis.",
	},
	{
		id: "casino-royale", title: "Casino Royale",
		content: "Casino Royale is a spy film starring Daniel Craig as James Bond, " +
			"the secret agent working for the British intelligence service.",
	},
	{
		id: "skyfall", title: "Skyfall",
		content: "Skyfall is a spy film starring Daniel Craig as James Bond facing " +
			"a former agent in a mission about aliens of the past and cyber threats.",
	},
	{
		id: "edge-of-tomorrow", title: "Edge of Tomorrow",
		content: "Edge of Tomorrow is an action film about soldiers fighting an alien " +
			"invasion, with time loops helping a pilot replay each battle.",
	},
}

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true, "was": true,
	"of": true, "to": true, "in": true, "on": true, "for": true, "and": true,
	"or": true, "what": true, "how": true, "does": true, "do": true, "about": true,
	"with": true, "which": true, "movie": true, "movies": true, "that": true,
}

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

func newRetriever() retrievers.Retriever {
	store := vectorstores.NewInMemory(embeddings.NewFake(64))
	docs := make([]documents.Document, 0, len(corpus))
	for _, item := range corpus {
		docs = append(docs, documents.New(item.title+" "+item.content,
			map[string]any{"source": item.title, "title": item.title}).WithID(item.id))
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

func retrieveNode(r retrievers.Retriever) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		q, _ := state["question"].(string)
		fmt.Println("---RETRIEVE---")
		docs, err := r.GetRelevantDocuments(context.Background(), q)
		if err != nil {
			return nil, err
		}
		for i, d := range docs {
			fmt.Printf("  %d. %v\n", i+1, d.Metadata["title"])
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
	answer := fmt.Sprintf("Regarding %q: %s", q, summarize(docText(docs)))
	return map[string]any{"documents": docs, "question": q, "generation": answer}, nil
}

func transformQueryNode(_ runtime.Runtime, state map[string]any) (any, error) {
	q, _ := state["question"].(string)
	docs := asDocs(state["documents"])
	fmt.Println("---TRANSFORM QUERY---")
	better := "Find movies and film summaries relevant to " + q + " including cast and director details."
	return map[string]any{"documents": docs, "question": better}, nil
}

func decideToGenerate(_ runtime.Runtime, state map[string]any) ([]any, error) {
	fmt.Println("---ASSESS GRADED DOCUMENTS---")
	if len(asDocs(state["documents"])) == 0 {
		fmt.Println("---DECISION: ALL DOCUMENTS NOT RELEVANT, TRANSFORM QUERY---")
		return graphpkg.To("transform_query"), nil
	}
	fmt.Println("---DECISION: GENERATE---")
	return graphpkg.To("generate"), nil
}

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
	g.AddNode("retrieve", retrieveNode(r))
	g.AddNode("grade_documents", gradeDocumentsNode)
	g.AddNode("generate", generateNode)
	g.AddNode("transform_query", transformQueryNode)

	g.AddEdge(types.START, "retrieve")
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
	run(ctx, cg, "Movies that star Daniel Craig")
	run(ctx, cg, "Which movies are about aliens?")
}
