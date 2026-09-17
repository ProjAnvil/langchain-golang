package pgvector

import (
	"context"
	"errors"
	"regexp"
	"testing"

	"github.com/pashagolub/pgxmock/v4"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// A failing embedder must abort option-based searches before any SQL runs.
func TestSimilaritySearchWithOptions_EmbeddingFailure(t *testing.T) {
	store, _ := newTestStore(t, "langchain")
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	if _, err := store.SimilaritySearchWithOptions(ctx, "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
		t.Fatal("expected embedding failure to abort the search")
	}
}

func TestSimilaritySearchWithOptions_QueryFailure(t *testing.T) {
	store, pool := newTestStore(t, "langchain")

	pool.ExpectQuery(regexp.QuoteMeta(`SELECT id, content, metadata, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance`)).
		WithArgs(pgxmock.AnyArg(), pgxmock.AnyArg()).
		WillReturnError(errors.New("postgres went away"))

	if _, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
		t.Fatal("expected query failure to propagate")
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestSimilaritySearchWithOptions_ScanFailure(t *testing.T) {
	store, pool := newTestStore(t, "langchain")

	pool.ExpectQuery(regexp.QuoteMeta(`SELECT id, content, metadata, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance LIMIT $2`)).
		WithArgs(pgxmock.AnyArg(), pgxmock.AnyArg()).
		WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "distance"}).
			AddRow(struct{}{}, "alpha", nil, 0.1))

	if _, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
		t.Fatal("expected row scan failure to propagate")
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestMMRSearchWithOptions_RejectsInvalidFilter(t *testing.T) {
	store, _ := newTestStore(t, "langchain")
	if _, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
		Filter: map[string]any{"group": map[string]any{"$bogus": "a"}},
	}); err == nil {
		t.Fatal("expected invalid filter error")
	}
}

// K and FetchK must fall back to the langchain defaults (4 and 20) when unset.
func TestMMRSearchWithOptions_DefaultsKAndFetchK(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	embedder := embeddings.NewFake(8)
	queryVector, _ := embedder.EmbedQuery(t.Context(), "alpha")

	pool.ExpectQuery(regexp.QuoteMeta(
		`SELECT id, content, metadata, embedding, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance LIMIT $2`,
	)).WithArgs(formatVector(queryVector), 20).
		WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "embedding", "distance"}))

	docs, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{})
	if err != nil {
		t.Fatalf("MMRSearchWithOptions: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs: got %d want 0", len(docs))
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestMMRSearchWithOptions_Failures(t *testing.T) {
	t.Run("embedding_failure", func(t *testing.T) {
		store, _ := newTestStore(t, "langchain")
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := store.MMRSearchWithOptions(ctx, "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
			t.Fatal("expected embedding failure")
		}
	})

	t.Run("unrenderable_filter", func(t *testing.T) {
		store, _ := newTestStore(t, "langchain")
		_, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
			Filter: map[string]any{"group": map[string]any{vectorstores.FilterIn: []any{}}},
		})
		if err == nil {
			t.Fatal("expected filter rendering failure")
		}
	})

	t.Run("query_failure", func(t *testing.T) {
		store, pool := newTestStore(t, "langchain")
		pool.ExpectQuery(regexp.QuoteMeta(
			`SELECT id, content, metadata, embedding, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance`,
		)).WithArgs(pgxmock.AnyArg(), pgxmock.AnyArg()).
			WillReturnError(errors.New("postgres went away"))
		if _, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
			t.Fatal("expected query failure")
		}
	})

	t.Run("scan_failure", func(t *testing.T) {
		store, pool := newTestStore(t, "langchain")
		pool.ExpectQuery(regexp.QuoteMeta(
			`SELECT id, content, metadata, embedding, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance LIMIT $2`,
		)).WithArgs(pgxmock.AnyArg(), pgxmock.AnyArg()).
			WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "embedding", "distance"}).
				AddRow(struct{}{}, "alpha", nil, "[1]", 0.1))
		if _, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
			t.Fatal("expected scan failure")
		}
	})

	t.Run("malformed_stored_vector", func(t *testing.T) {
		store, pool := newTestStore(t, "langchain")
		pool.ExpectQuery(regexp.QuoteMeta(
			`SELECT id, content, metadata, embedding, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance LIMIT $2`,
		)).WithArgs(pgxmock.AnyArg(), pgxmock.AnyArg()).
			WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "embedding", "distance"}).
				AddRow("one", "alpha", nil, "[not-a-float]", 0.1))
		if _, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
			t.Fatal("expected stored vector parse failure")
		}
	})

	t.Run("rows_error", func(t *testing.T) {
		store, pool := newTestStore(t, "langchain")
		pool.ExpectQuery(regexp.QuoteMeta(
			`SELECT id, content, metadata, embedding, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance LIMIT $2`,
		)).WithArgs(pgxmock.AnyArg(), pgxmock.AnyArg()).
			WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "embedding", "distance"}).
				AddRow("one", "alpha", nil, "[1]", 0.1).
				CloseError(errors.New("connection reset")))
		if _, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{K: 2}); err == nil {
			t.Fatal("expected rows error")
		}
	})
}
