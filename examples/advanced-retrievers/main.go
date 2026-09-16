// Command advanced-retrievers demonstrates the M1 retriever stack on an
// in-memory store:
//
//   - a dense retriever (vector similarity over fake embeddings),
//   - a keyword retriever (exact-term scoring, standing in for BM25),
//   - EnsembleRetriever fusing both with reciprocal rank fusion (RRF),
//   - MultiQueryRetriever expanding the question into paraphrases with a
//     chat model (a fake model here; any partner chat model drops in) and
//     merging the per-variant hits, deduplicated.
//
// The printed output shows each layer's ranking, so the RRF reordering and
// the multi-query merge are visible.
//
// Usage:
//
//	go run ./examples/advanced-retrievers
package main

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// keywordRetriever is a minimal sparse retriever: documents score one point
// per query term they contain (a stand-in for BM25; swap in a real
// keyword index in production). It ranks by score, ties by insertion order.
type keywordRetriever struct {
	docs []documents.Document
}

func (r keywordRetriever) GetRelevantDocuments(_ context.Context, query string) ([]documents.Document, error) {
	type scored struct {
		doc   documents.Document
		score int
	}
	var hits []scored
	for _, doc := range r.docs {
		content := strings.ToLower(doc.PageContent)
		score := 0
		for _, term := range strings.Fields(strings.ToLower(query)) {
			if strings.Contains(content, term) {
				score++
			}
		}
		if score > 0 {
			hits = append(hits, scored{doc, score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool { return hits[i].score > hits[j].score })
	out := make([]documents.Document, 0, len(hits))
	for i, h := range hits {
		if i == 3 { // keep the top-3, like the dense side's k
			break
		}
		doc := h.doc.Clone()
		if doc.Metadata == nil {
			doc.Metadata = map[string]any{}
		}
		doc.Metadata["rank"] = i + 1
		doc.Metadata["member"] = "keyword"
		out = append(out, doc)
	}
	return out, nil
}

func show(label string, docs []documents.Document) {
	fmt.Println(label)
	if len(docs) == 0 {
		fmt.Println("  (no hits)")
		return
	}
	for i, doc := range docs {
		score := ""
		if s, ok := doc.Metadata["score"].(float64); ok {
			score = fmt.Sprintf(" rrf=%.4f", s)
		}
		fmt.Printf("  %d. [%s]%s %s\n", i+1, docID(doc), score, preview(doc.PageContent))
	}
}

func docID(doc documents.Document) string {
	if id, ok := doc.Metadata["id"].(string); ok {
		return id
	}
	return "doc"
}

func preview(text string) string {
	if len(text) > 64 {
		return text[:64] + "..."
	}
	return text
}

func main() {
	ctx := context.Background()

	corpus := []documents.Document{
		documents.New("pgvector stores embeddings in Postgres for similarity search", map[string]any{"id": "d1"}),
		documents.New("hybrid search combines dense vectors with keyword matching", map[string]any{"id": "d2"}),
		documents.New("reciprocal rank fusion merges multiple ranked lists", map[string]any{"id": "d3"}),
		documents.New("keyword BM25 indexing ranks documents by term frequency", map[string]any{"id": "d4"}),
		documents.New("multi-query retrieval paraphrases the question for better recall", map[string]any{"id": "d5"}),
		documents.New("embedding models map text into dense vector spaces", map[string]any{"id": "d6"}),
	}

	// Dense side: in-memory store over deterministic fake embeddings. Swap
	// embeddings.NewFake for a partner embedder (and vectorstores.NewInMemory
	// for pgvector) without touching anything below.
	store := vectorstores.NewInMemory(embeddings.NewFake(32))
	if _, err := store.AddDocuments(ctx, corpus); err != nil {
		fmt.Println("add documents:", err)
		return
	}
	dense, err := retrievers.AsRetriever(store, retrievers.WithSearchKwargs(map[string]any{"k": 3}))
	if err != nil {
		fmt.Println("as retriever:", err)
		return
	}

	// Sparse side: the keyword retriever over the same corpus.
	keyword := keywordRetriever{docs: corpus}

	query := "how does hybrid search combine keyword and vector results"

	denseHits, err := dense.GetRelevantDocuments(ctx, query)
	if err != nil {
		fmt.Println("dense retrieve:", err)
		return
	}
	keywordHits, err := keyword.GetRelevantDocuments(ctx, query)
	if err != nil {
		fmt.Println("keyword retrieve:", err)
		return
	}

	// Ensemble: reciprocal rank fusion, score(d) = sum(weight / (60 + rank)).
	ensemble, err := retrievers.NewEnsembleRetriever([]retrievers.Retriever{dense, keyword})
	if err != nil {
		fmt.Println("new ensemble retriever:", err)
		return
	}
	fusedHits, err := ensemble.GetRelevantDocuments(ctx, query)
	if err != nil {
		fmt.Println("ensemble retrieve:", err)
		return
	}

	// Multi-query: a chat model expands the question into variants. A fake
	// model keeps this offline; any partner chat model satisfies QueryModel.
	variantModel := language.NewFakeChatModel(
		language.WithResponses(messages.AI(
			"hybrid search combining keyword matching and vector similarity\n" +
				"mixing sparse BM25 retrieval with dense embedding search\n" +
				"blending keyword and semantic vector ranking")),
	)
	multi, err := retrievers.NewMultiQueryRetriever(ensemble,
		retrievers.WithQueryModel(variantModel),
		retrievers.WithIncludeOriginalQuery(true),
	)
	if err != nil {
		fmt.Println("new multi-query retriever:", err)
		return
	}
	multiHits, err := multi.GetRelevantDocuments(ctx, query)
	if err != nil {
		fmt.Println("multi-query retrieve:", err)
		return
	}

	show("--- dense (vector similarity) ---", denseHits)
	show("--- keyword (term matching) ---", keywordHits)
	show("--- ensemble (RRF fusion) ---", fusedHits)
	show("--- multi-query over ensemble (deduplicated merge) ---", multiHits)
}
