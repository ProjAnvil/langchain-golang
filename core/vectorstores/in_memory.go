package vectorstores

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

// InMemory is a deterministic vector store for tests and local examples.
type InMemory struct {
	mu         sync.RWMutex
	embedder   embeddings.Embeddings
	nextID     int
	documents  map[string]documents.Document
	vectors    map[string][]float64
	idSequence []string
}

// NewInMemory creates an in-memory vector store.
func NewInMemory(embedder embeddings.Embeddings) *InMemory {
	return &InMemory{
		embedder:  embedder,
		documents: map[string]documents.Document{},
		vectors:   map[string][]float64{},
	}
}

// AddTexts embeds and stores raw text values.
func (s *InMemory) AddTexts(
	ctx context.Context,
	texts []string,
	metadatas []map[string]any,
	ids []string,
) ([]string, error) {
	docs := make([]documents.Document, len(texts))
	for i, text := range texts {
		doc := documents.New(text, metadataAt(metadatas, i))
		if i < len(ids) {
			doc.ID = ids[i]
		}
		docs[i] = doc
	}
	return s.AddDocuments(ctx, docs)
}

// AddDocuments embeds and stores documents.
func (s *InMemory) AddDocuments(
	ctx context.Context,
	docs []documents.Document,
) ([]string, error) {
	if s.embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}

	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
	}
	vectors, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, err
	}
	if len(vectors) != len(docs) {
		return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(vectors), len(docs))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	ids := make([]string, len(docs))
	for i, doc := range docs {
		id := doc.ID
		if id == "" {
			s.nextID++
			id = fmt.Sprintf("doc-%d", s.nextID)
		}
		stored := doc.Clone().WithID(id)
		if _, exists := s.documents[id]; !exists {
			s.idSequence = append(s.idSequence, id)
		}
		s.documents[id] = stored
		s.vectors[id] = append([]float64(nil), vectors[i]...)
		ids[i] = id
	}

	return ids, nil
}

// Delete removes IDs from the store. Missing IDs are ignored.
func (s *InMemory) Delete(_ context.Context, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range ids {
		delete(s.documents, id)
		delete(s.vectors, id)
	}
	s.idSequence = compactIDs(s.idSequence, s.documents)
	return nil
}

// GetByIDs returns documents for found IDs. Missing IDs are skipped.
func (s *InMemory) GetByIDs(_ context.Context, ids []string) ([]documents.Document, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]documents.Document, 0, len(ids))
	for _, id := range ids {
		doc, ok := s.documents[id]
		if ok {
			out = append(out, doc.Clone())
		}
	}
	return out, nil
}

// SimilaritySearch returns the top k documents.
func (s *InMemory) SimilaritySearch(
	ctx context.Context,
	query string,
	k int,
) ([]documents.Document, error) {
	results, err := s.SimilaritySearchWithScore(ctx, query, k)
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, len(results))
	for i, result := range results {
		docs[i] = result.Document
	}
	return docs, nil
}

// SimilaritySearchWithScore returns the top k documents and scores.
func (s *InMemory) SimilaritySearchWithScore(
	ctx context.Context,
	query string,
	k int,
) ([]SearchResult, error) {
	return s.SimilaritySearchWithScoreFilter(ctx, query, k, nil)
}

// SimilaritySearchWithScoreFilter returns top documents matching a filter.
func (s *InMemory) SimilaritySearchWithScoreFilter(
	ctx context.Context,
	query string,
	k int,
	filter Filter,
) ([]SearchResult, error) {
	if k <= 0 {
		return nil, nil
	}
	if s.embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}
	queryVector, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return s.similaritySearchWithScoreByVectorLocked(queryVector, k, filter), nil
}

// SimilaritySearchByVector returns top documents for an embedding vector.
func (s *InMemory) SimilaritySearchByVector(
	ctx context.Context,
	vector []float64,
	k int,
	filter Filter,
) ([]documents.Document, error) {
	results, err := s.SimilaritySearchWithScoreByVector(ctx, vector, k, filter)
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, len(results))
	for i, result := range results {
		docs[i] = result.Document
	}
	return docs, nil
}

// SimilaritySearchWithScoreByVector returns top scored documents for an
// embedding vector.
func (s *InMemory) SimilaritySearchWithScoreByVector(
	ctx context.Context,
	vector []float64,
	k int,
	filter Filter,
) ([]SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if k <= 0 {
		return nil, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.similaritySearchWithScoreByVectorLocked(vector, k, filter), nil
}

// MaxMarginalRelevanceSearch returns documents selected for similarity and
// diversity.
func (s *InMemory) MaxMarginalRelevanceSearch(
	ctx context.Context,
	query string,
	k int,
	fetchK int,
	lambdaMult float64,
	filter Filter,
) ([]documents.Document, error) {
	if s.embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}
	queryVector, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	return s.MaxMarginalRelevanceSearchByVector(ctx, queryVector, k, fetchK, lambdaMult, filter)
}

// MaxMarginalRelevanceSearchByVector is the vector form of MMR search.
func (s *InMemory) MaxMarginalRelevanceSearchByVector(
	ctx context.Context,
	vector []float64,
	k int,
	fetchK int,
	lambdaMult float64,
	filter Filter,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if fetchK <= 0 {
		fetchK = 20
	}
	if k <= 0 {
		k = 4
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	prefetch := s.similaritySearchWithScoreByVectorLocked(vector, fetchK, filter)
	embeds := make([][]float64, len(prefetch))
	for i, hit := range prefetch {
		embeds[i] = append([]float64(nil), s.vectors[hit.Document.ID]...)
	}
	indices := MaximalMarginalRelevance(vector, embeds, lambdaMult, k)
	docs := make([]documents.Document, 0, len(indices))
	for _, index := range indices {
		docs = append(docs, prefetch[index].Document.Clone())
	}
	return docs, nil
}

func (s *InMemory) similaritySearchWithScoreByVectorLocked(
	queryVector []float64,
	k int,
	filter Filter,
) []SearchResult {
	results := make([]SearchResult, 0, len(s.documents))
	for _, id := range s.idSequence {
		doc, ok := s.documents[id]
		if !ok {
			continue
		}
		doc = doc.Clone()
		if filter != nil && !filter(doc) {
			continue
		}
		results = append(results, SearchResult{
			Document: doc,
			Score:    cosine(queryVector, s.vectors[id]),
		})
	}
	sort.SliceStable(results, func(i int, j int) bool {
		return results[i].Score > results[j].Score
	})
	if len(results) > k {
		results = results[:k]
	}
	return results
}

func compactIDs(ids []string, docs map[string]documents.Document) []string {
	out := ids[:0]
	for _, id := range ids {
		if _, ok := docs[id]; ok {
			out = append(out, id)
		}
	}
	return out
}

func metadataAt(metadatas []map[string]any, index int) map[string]any {
	if index < len(metadatas) {
		return metadatas[index]
	}
	return nil
}

func cosine(a []float64, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot float64
	var normA float64
	var normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
