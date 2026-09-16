// Command rag-fullchain wires the whole RAG stack end to end:
//
//	HTML loader -> recursive text splitter -> embeddings -> vector store
//	-> retrieval -> (optional) Cohere rerank -> tool-calling agent answer
//
// Degradation ladder (each step works offline when its env var is absent):
//
//	OPENAI_API_KEY  set -> real OpenAI embeddings;   unset -> fake embeddings
//	PGVECTOR_DSN    set -> pgvector collection;      unset -> in-memory store
//	COHERE_API_KEY  set -> Cohere rerank compressor; unset -> raw retrieval
//
// The agent model is an offline scripted double that calls the search tool
// and then answers, so the example always runs to completion.
//
// Required env (all optional):
//
//	OPENAI_API_KEY   OpenAI embeddings (default model text-embedding-3-small)
//	PGVECTOR_DSN     Postgres URL with the pgvector extension installed
//	COHERE_API_KEY   Cohere rerank (default model rerank-v3.5)
//
// Usage:
//
//	go run ./examples/rag-fullchain
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/documentloaders"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/core/vectorstores"
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/partners/cohere"
	"github.com/projanvil/langchain-golang/partners/openai"
	"github.com/projanvil/langchain-golang/partners/pgvector"
	"github.com/projanvil/langchain-golang/textsplitters"
)

const supportHTML = `<html><head><title>Acme Support</title></head><body>
<h1>Acme Cloud support</h1>
<p>Billing: invoices are issued on the first business day of each month. Refund
requests for monthly plans must be filed within 14 days of the charge and are
processed within five business days.</p>
<p>Uptime SLA: the Pro plan carries a 99.95% monthly uptime guarantee. Service
credits are issued automatically when uptime falls below the guarantee; no
ticket is required.</p>
<p>Data residency: EU customers can pin storage to the Frankfurt region at
tenant creation time. Migration of an existing tenant requires a support
ticket and takes up to two weeks.</p>
</body></html>`

// scriptedModel: offline ChatModel double, self-returning from BindTools
// (see examples/quickstart for the rationale).
type scriptedModel struct {
	mu          sync.Mutex
	responses   []messages.Message
	idx         int
	invocations [][]messages.Message
}

func (m *scriptedModel) Invoke(_ context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.invocations = append(m.invocations, append([]messages.Message(nil), input...))
	if m.idx >= len(m.responses) {
		return messages.Message{}, fmt.Errorf("scriptedModel: no more responses (call %d)", m.idx+1)
	}
	resp := m.responses[m.idx]
	m.idx++
	return resp, nil
}

func (m *scriptedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		out[i], err = m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *scriptedModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	resp, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{resp}), nil
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

func main() {
	ctx := context.Background()
	query := "how long does a refund take and when are invoices issued?"

	// --- 1. Load: HTML document -> one Document of readable body text. ---
	loader := documentloaders.NewHTMLLoader(strings.NewReader(supportHTML))
	docs, err := documentloaders.Load(ctx, loader)
	if err != nil {
		fmt.Println("load html:", err)
		return
	}
	fmt.Printf("loaded %d document(s) (title=%v)\n", len(docs), docs[0].Metadata["title"])

	// --- 2. Split: recursive character splitter into chunk-sized pieces. ---
	splitter, err := textsplitters.NewRecursiveCharacter(nil, false, textsplitters.Config{
		ChunkSize:    160,
		ChunkOverlap: 24,
	})
	if err != nil {
		fmt.Println("new splitter:", err)
		return
	}
	chunks := splitter.SplitDocuments(docs)
	fmt.Printf("split into %d chunk(s)\n", len(chunks))

	// --- 3. Embed: OpenAI when a key is present, fake embeddings otherwise. ---
	var embedder embeddings.Embeddings
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		embedder = openai.NewEmbeddings() // key + base URL default; model text-embedding-3-small
		fmt.Println("embeddings: OpenAI (text-embedding-3-small)")
	} else {
		embedder = embeddings.NewFake(64)
		fmt.Println("embeddings: deterministic fake (set OPENAI_API_KEY for real embeddings)")
	}

	// --- 4. Store: pgvector when a DSN is configured, in-memory otherwise.
	// To switch this example onto Postgres permanently, export PGVECTOR_DSN
	// (a URL to a database with the pgvector extension) — the code path below
	// is the switching point. ---
	var store vectorstores.VectorStore
	if dsn := os.Getenv("PGVECTOR_DSN"); dsn != "" {
		pgStore, err := pgvector.New(ctx, "rag_fullchain_demo",
			pgvector.WithDSN(dsn),
			pgvector.WithEmbedder(embedder))
		if err != nil {
			fmt.Println("pgvector new:", err)
			return
		}
		defer pgStore.Close()
		store = pgStore
		fmt.Println("vector store: pgvector collection rag_fullchain_demo")
	} else {
		store = vectorstores.NewInMemory(embedder)
		fmt.Println("vector store: in-memory (set PGVECTOR_DSN for pgvector)")
	}

	// --- 5. Index. ---
	if _, err := store.AddDocuments(ctx, chunks); err != nil {
		fmt.Println("add documents:", err)
		return
	}

	// --- 6. Retrieve (k=3) ---.
	var base retrievers.Retriever
	base, err = retrievers.AsRetriever(store, retrievers.WithSearchKwargs(map[string]any{"k": 3}))
	if err != nil {
		fmt.Println("as retriever:", err)
		return
	}

	// --- 7. Rerank: Cohere compressor when a key is present. ---
	var finalRetriever retrievers.Retriever
	finalRetriever = base
	if key := os.Getenv("COHERE_API_KEY"); key != "" {
		reranker := cohere.New(cohere.WithAPIKey(key), cohere.WithTopN(3))
		finalRetriever, err = retrievers.NewContextualCompressionRetriever(base, reranker)
		if err != nil {
			fmt.Println("compression retriever:", err)
			return
		}
		fmt.Println("rerank: Cohere rerank-v3.5")
	} else {
		fmt.Println("rerank: skipped (set COHERE_API_KEY for Cohere rerank)")
	}

	hits, err := finalRetriever.GetRelevantDocuments(ctx, query)
	if err != nil {
		fmt.Println("retrieve:", err)
		return
	}
	fmt.Printf("retrieved %d chunk(s):\n", len(hits))
	for i, doc := range hits {
		preview := doc.PageContent
		if len(preview) > 90 {
			preview = preview[:90] + "..."
		}
		fmt.Printf("  %d. %s\n", i+1, strings.ReplaceAll(preview, "\n", " "))
	}

	// --- 8. Answer: an agent whose search_docs tool wraps the retriever.
	// Swap the scripted model for a partner chat model (e.g.
	// openai.NewChatModel) to get genuinely generated answers. ---
	search, err := coretools.NewFunc("search_docs", "search the Acme support knowledge base",
		schema.Object(map[string]schema.Schema{
			"query": schema.String("the question to search for"),
		}, "query"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			q, _ := input["query"].(string)
			docs, err := finalRetriever.GetRelevantDocuments(ctx, q)
			if err != nil {
				return coretools.Result{}, err
			}
			parts := make([]string, 0, len(docs))
			for _, doc := range docs {
				parts = append(parts, doc.PageContent)
			}
			return coretools.Result{Content: strings.Join(parts, "\n---\n")}, nil
		})
	if err != nil {
		fmt.Println("build search tool:", err)
		return
	}

	model := &scriptedModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "search_docs", Args: map[string]any{"query": query}},
			},
		},
		messages.AI("Refunds are processed within five business days of a request, and " +
			"invoices are issued on the first business day of each month."),
	}}

	agent, err := agents.CreateAgent(model, []coretools.Tool{search},
		agents.WithAgentSystemPrompt("Answer Acme Cloud support questions using the search_docs tool."),
	)
	if err != nil {
		fmt.Println("create agent:", err)
		return
	}
	out, err := agent.Invoke(ctx, []messages.Message{messages.Human(query)})
	if err != nil {
		fmt.Println("invoke agent:", err)
		return
	}
	fmt.Println("agent answer:")
	fmt.Printf("  %s\n", messages.Text(out[len(out)-1]))
}
