package vectorstores

import (
	"context"
	"math"

	"github.com/projanvil/langchain-golang/core/documents"
)

// SearchResult is a document and its similarity score.
type SearchResult struct {
	Document documents.Document
	Score    float64
}

// Filter decides whether a document should be considered by a search.
type Filter func(documents.Document) bool

// VectorStore stores embedded documents and performs similarity search.
type VectorStore interface {
	AddDocuments(ctx context.Context, docs []documents.Document) ([]string, error)
	Delete(ctx context.Context, ids []string) error
	GetByIDs(ctx context.Context, ids []string) ([]documents.Document, error)
	SimilaritySearch(ctx context.Context, query string, k int) ([]documents.Document, error)
	SimilaritySearchWithScore(ctx context.Context, query string, k int) ([]SearchResult, error)
}

// TextAdder is implemented by stores that can add raw text values directly.
type TextAdder interface {
	AddTexts(ctx context.Context, texts []string, metadatas []map[string]any, ids []string) ([]string, error)
}

// CosineRelevanceScore converts cosine distance to a [0, 1] relevance score.
func CosineRelevanceScore(distance float64) float64 {
	return 1.0 - distance
}

// EuclideanRelevanceScore converts euclidean distance for normalized embeddings
// to a [0, 1] relevance score.
func EuclideanRelevanceScore(distance float64) float64 {
	return 1.0 - distance/1.4142135623730951
}

// MaxInnerProductRelevanceScore converts max-inner-product distance to a
// relevance score matching LangChain's default helper.
func MaxInnerProductRelevanceScore(distance float64) float64 {
	if distance > 0 {
		return 1.0 - distance
	}
	return -distance
}

// MaximalMarginalRelevance returns indices selected for query similarity and
// diversity. lambdaMult=1 favors query similarity; 0 favors diversity.
func MaximalMarginalRelevance(query []float64, embeddings [][]float64, lambdaMult float64, k int) []int {
	if k <= 0 || len(embeddings) == 0 {
		return nil
	}
	if lambdaMult < 0 {
		lambdaMult = 0
	}
	if lambdaMult > 1 {
		lambdaMult = 1
	}
	limit := k
	if len(embeddings) < limit {
		limit = len(embeddings)
	}
	queryScores := make([]float64, len(embeddings))
	best := 0
	for i, embedding := range embeddings {
		queryScores[i] = cosine(query, embedding)
		if i == 0 || queryScores[i] > queryScores[best] {
			best = i
		}
	}
	selected := []int{best}
	selectedSet := map[int]bool{best: true}
	for len(selected) < limit {
		bestScore := math.Inf(-1)
		bestIndex := -1
		for i, embedding := range embeddings {
			if selectedSet[i] {
				continue
			}
			redundant := 0.0
			for _, selectedIndex := range selected {
				score := cosine(embedding, embeddings[selectedIndex])
				if score > redundant {
					redundant = score
				}
			}
			score := lambdaMult*queryScores[i] - (1-lambdaMult)*redundant
			if bestIndex == -1 || score > bestScore {
				bestScore = score
				bestIndex = i
			}
		}
		if bestIndex == -1 {
			break
		}
		selected = append(selected, bestIndex)
		selectedSet[bestIndex] = true
	}
	return selected
}
