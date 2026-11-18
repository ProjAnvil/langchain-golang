// Command rag_langgraph_agentic_rag is the Go port of
// langgraph/examples/rag/langgraph_agentic_rag.ipynb.
//
// Agentic RAG uses an agent that decides whether to call a retriever tool.
// Retrieved documents are graded: relevant ones go to generation, irrelevant
// ones trigger a query rewrite that returns to the agent. The Go port keeps
// the message-based state and graph topology, with a scripted offline agent
// and a deterministic in-memory retriever replacing the chat model,
// ToolNode, Chroma store and LangSmith Hub prompt.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/vectorstores"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// corpus is a deterministic stand-in for the three blog posts in the notebook.
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

// retrievalQuery drops stop words so the token-hashing embeddings are driven
// by topical terms rather than common filler words.
func retrievalQuery(q string) string {
	tokens := []string{}
	for t := range tokenSet(q) {
		tokens = append(tokens, t)
	}
	return strings.Join(tokens, " ")
}

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

func asMessages(v any) []messages.Message {
	msgs, _ := v.([]messages.Message)
	return msgs
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

// firstQuestion returns the initial user question from the message history.
func firstQuestion(msgs []messages.Message) string {
	for _, m := range msgs {
		if m.Role == messages.RoleHuman {
			return strings.TrimPrefix(m.Content, "Improved question: ")
		}
	}
	return ""
}

// agent is a scripted ChatModel: it emits a retriever tool call the first time
// it sees a human message, and a final answer once a tool result is present.
func agentNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs := asMessages(state["messages"])
	last := msgs[len(msgs)-1]
	fmt.Println("---CALL AGENT---")
	if last.Role == messages.RoleTool {
		q := firstQuestion(msgs)
		answer := fmt.Sprintf("Based on the retrieved documents about %q: %s", q, summarize(last.Content))
		return map[string]any{"messages": []messages.Message{messages.AI(answer)}}, nil
	}
	q := firstQuestion(msgs)
	call := messages.Message{
		Role: messages.RoleAI,
		ToolCalls: []messages.ToolCall{{
			ID:   "call_1",
			Name: "retrieve_blog_posts",
			Args: map[string]any{"query": q},
		}},
	}
	return map[string]any{"messages": []messages.Message{call}}, nil
}

// toolsCondition is the Go form of langgraph.prebuilt.tools_condition.
func toolsCondition(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs := asMessages(state["messages"])
	last := msgs[len(msgs)-1]
	if last.Role == messages.RoleAI && len(last.ToolCalls) > 0 {
		return graphpkg.To("retrieve"), nil
	}
	return graphpkg.To(types.END), nil
}

// retrieveNode emulates the ToolNode for the retriever tool.
func retrieveNode(r retrievers.Retriever) graphpkg.NodeFunc {
	return func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs := asMessages(state["messages"])
		last := msgs[len(msgs)-1]
		query, _ := last.ToolCalls[0].Args["query"].(string)
		fmt.Println("---RETRIEVE (retrieve_blog_posts)---")
		docs, err := r.GetRelevantDocuments(context.Background(), retrievalQuery(query))
		if err != nil {
			return nil, err
		}
		for i, d := range docs {
			fmt.Printf("  %d. %s\n", i+1, d.Source())
		}
		result := messages.Tool(last.ToolCalls[0].ID, docText(docs))
		return map[string]any{"messages": []messages.Message{result}}, nil
	}
}

// gradeDocuments routes after retrieval based on keyword overlap between the
// question and the tool result, replacing the structured-output grader.
func gradeDocuments(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs := asMessages(state["messages"])
	q := firstQuestion(msgs)
	docs := msgs[len(msgs)-1].Content
	fmt.Println("---CHECK RELEVANCE---")
	if overlap(tokenSet(q), tokenSet(docs)) > 0 {
		fmt.Println("---DECISION: DOCS RELEVANT, GENERATE---")
		return graphpkg.To("generate"), nil
	}
	fmt.Println("---DECISION: DOCS NOT RELEVANT, REWRITE---")
	return graphpkg.To("rewrite"), nil
}

func generateNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs := asMessages(state["messages"])
	q := firstQuestion(msgs)
	docs := msgs[len(msgs)-1].Content
	fmt.Println("---GENERATE---")
	answer := fmt.Sprintf("Regarding %q: %s", q, summarize(docs))
	return map[string]any{"messages": []messages.Message{messages.AI(answer)}}, nil
}

func rewriteNode(_ runtime.Runtime, state map[string]any) (any, error) {
	msgs := asMessages(state["messages"])
	q := firstQuestion(msgs)
	fmt.Println("---TRANSFORM QUERY---")
	better := "Improved question: " + q + " about LLM agents, memory and prompt engineering."
	return map[string]any{"messages": []messages.Message{messages.Human(better)}}, nil
}

func buildGraph(r retrievers.Retriever) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("agent", agentNode)
	g.AddNode("retrieve", retrieveNode(r))
	g.AddNode("rewrite", rewriteNode)
	g.AddNode("generate", generateNode)

	g.AddEdge(types.START, "agent")
	g.AddConditionalEdges("agent", toolsCondition)
	g.AddConditionalEdges("retrieve", gradeDocuments)
	g.AddEdge("generate", types.END)
	g.AddEdge("rewrite", "agent")

	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile graph: %v", err)
	}
	return cg
}

func main() {
	ctx := context.Background()
	cg := buildGraph(newRetriever())
	question := "What does Lilian Weng say about the types of agent memory?"
	result, err := cg.Invoke(ctx, map[string]any{
		"messages": []messages.Message{messages.Human(question)},
	})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	msgs := asMessages(result.Values["messages"])
	fmt.Println("\n=== FINAL MESSAGE TRACE ===")
	for _, m := range msgs {
		body := summarize(m.Content)
		if body == "no supporting context was found." && len(m.ToolCalls) > 0 {
			body = "(tool-call request)"
		}
		fmt.Printf("[%s] %s\n", m.Role, body)
		if len(m.ToolCalls) > 0 {
			fmt.Printf("  tool call: %s\n", m.ToolCalls[0].Name)
		}
	}
}
