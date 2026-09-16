package pgvector

import (
	"fmt"
	"regexp"
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/pashagolub/pgxmock/v4"
	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// newMockPool returns a pgxmock pool with the schema DDL expectations every
// New call performs (extension, table, index).
func newMockPool(t *testing.T) pgxmock.PgxPoolIface {
	t.Helper()
	pool, err := pgxmock.NewPool()
	if err != nil {
		t.Fatalf("pgxmock.NewPool: %v", err)
	}
	pool.ExpectExec("CREATE EXTENSION IF NOT EXISTS vector").
		WillReturnResult(pgconn.NewCommandTag("CREATE EXTENSION"))
	return pool
}

func expectTable(pool pgxmock.PgxPoolIface, table string, dim int, indexSQL string) {
	pool.ExpectExec(regexp.QuoteMeta(
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS "%s" (id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata JSONB, embedding vector(%d))`, table, dim),
	)).WillReturnResult(pgconn.NewCommandTag("CREATE TABLE"))
	pool.ExpectExec(regexp.QuoteMeta(indexSQL)).
		WillReturnResult(pgconn.NewCommandTag("CREATE INDEX"))
}

func expectDefaultSchema(t *testing.T, table string) pgxmock.PgxPoolIface {
	t.Helper()
	pool := newMockPool(t)
	expectTable(pool, table, 8, fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS "%s_embedding_idx" ON "%s" USING hnsw (embedding vector_cosine_ops)`, table, table,
	))
	return pool
}

// newTestStore builds a store over a pgxmock pool with the default cosine
// HNSW schema and a deterministic 8-dimension fake embedder.
func newTestStore(t *testing.T, table string) (*Store, pgxmock.PgxPoolIface) {
	t.Helper()
	pool := expectDefaultSchema(t, table)
	store, err := New(t.Context(), table,
		WithPool(pool),
		WithEmbedder(embeddings.NewFake(8)),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return store, pool
}

func TestNewCreatesSchema(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	_ = store
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestNewOptions(t *testing.T) {
	t.Run("l2 metric selects l2 ops class and operator", func(t *testing.T) {
		pool := newMockPool(t)
		expectTable(pool, "docs", 4, `CREATE INDEX IF NOT EXISTS "docs_embedding_idx" ON "docs" USING hnsw (embedding vector_l2_ops)`)
		store, err := New(t.Context(), "docs",
			WithPool(pool),
			WithEmbedder(embeddings.NewFake(4)),
			WithDistanceMetric(DistanceL2),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if store.metric != DistanceL2 {
			t.Fatalf("metric: got %q want l2", store.metric)
		}
		if got := store.distanceOperator(); got != "<->" {
			t.Fatalf("distance operator: got %q want <->", got)
		}
	})

	t.Run("ivfflat index kind", func(t *testing.T) {
		pool := newMockPool(t)
		expectTable(pool, "docs", 4, `CREATE INDEX IF NOT EXISTS "docs_embedding_idx" ON "docs" USING ivfflat (embedding vector_ip_ops) WITH (lists = 100)`)
		_, err := New(t.Context(), "docs",
			WithPool(pool),
			WithEmbedder(embeddings.NewFake(4)),
			WithDistanceMetric(DistanceIP),
			WithIndexKind(IndexIVFFlat),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
	})

	t.Run("explicit dimension overrides embedder dimension", func(t *testing.T) {
		pool := newMockPool(t)
		expectTable(pool, "docs", 16, `CREATE INDEX IF NOT EXISTS "docs_embedding_idx" ON "docs" USING hnsw (embedding vector_cosine_ops)`)
		_, err := New(t.Context(), "docs",
			WithPool(pool),
			WithEmbedder(embeddings.NewFake(4)),
			WithDimension(16),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
	})

	t.Run("missing embedder", func(t *testing.T) {
		if _, err := New(t.Context(), "docs", WithPool(newMockPool(t))); err == nil {
			t.Fatal("expected error for missing embedder")
		}
	})

	t.Run("missing pool and dsn", func(t *testing.T) {
		_, err := New(t.Context(), "docs", WithEmbedder(embeddings.NewFake(4)))
		if err == nil {
			t.Fatal("expected error for missing dsn")
		}
	})

	t.Run("unknown dimension without Dimensions embedder", func(t *testing.T) {
		_, err := New(t.Context(), "docs",
			WithPool(newMockPool(t)),
			WithEmbedder(embeddings.Static{}),
		)
		if err == nil {
			t.Fatal("expected error for unknown vector dimension")
		}
	})

	t.Run("invalid distance metric", func(t *testing.T) {
		_, err := New(t.Context(), "docs",
			WithPool(newMockPool(t)),
			WithEmbedder(embeddings.NewFake(4)),
			WithDistanceMetric("manhattan"),
		)
		if err == nil {
			t.Fatal("expected error for invalid distance metric")
		}
	})

	t.Run("invalid index kind", func(t *testing.T) {
		_, err := New(t.Context(), "docs",
			WithPool(newMockPool(t)),
			WithEmbedder(embeddings.NewFake(4)),
			WithIndexKind("boom"),
		)
		if err == nil {
			t.Fatal("expected error for invalid index kind")
		}
	})

	t.Run("empty collection name", func(t *testing.T) {
		if _, err := New(t.Context(), "  ", WithPool(newMockPool(t)), WithEmbedder(embeddings.NewFake(4))); err == nil {
			t.Fatal("expected error for empty collection name")
		}
	})
}

func TestFormatAndParseVector(t *testing.T) {
	formatted := formatVector([]float64{1, 0.5, -0.25})
	if formatted != "[1,0.5,-0.25]" {
		t.Fatalf("formatVector: got %q", formatted)
	}
	parsed, err := parseVector("[1,0.5,-0.25]")
	if err != nil {
		t.Fatalf("parseVector: %v", err)
	}
	if len(parsed) != 3 || parsed[0] != 1 || parsed[1] != 0.5 || parsed[2] != -0.25 {
		t.Fatalf("parseVector: got %v", parsed)
	}
	if v, err := parseVector(""); err != nil || v != nil {
		t.Fatalf("parseVector empty: got %v err %v", v, err)
	}
}

func TestAddDocumentsInsertsRows(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	embedder := embeddings.NewFake(8)
	// AddDocuments binds the embedder's document vectors, not a query vector.
	docs0, err := embedder.EmbedDocuments(t.Context(), []string{"alpha one"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	docs1, err := embedder.EmbedDocuments(t.Context(), []string{"alpha two"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}

	insertSQL := regexp.QuoteMeta(`INSERT INTO "langchain" (id, content, metadata, embedding) VALUES ($1, $2, $3::jsonb, $4::vector) ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding`)
	pool.ExpectExec(insertSQL).WithArgs(
		"one", "alpha one", `{"group":"a","page":1}`, formatVector(docs0[0]),
	).WillReturnResult(pgconn.NewCommandTag("INSERT 0 1"))
	pool.ExpectExec(insertSQL).WithArgs(
		"two", "alpha two", "null", formatVector(docs1[0]),
	).WillReturnResult(pgconn.NewCommandTag("INSERT 0 1"))

	ids, err := store.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha one", map[string]any{"group": "a", "page": 1}).WithID("one"),
		documents.New("alpha two", nil).WithID("two"),
	})
	if err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}
	if len(ids) != 2 || ids[0] != "one" || ids[1] != "two" {
		t.Fatalf("ids: got %v", ids)
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestAddDocumentsGeneratesMissingIDs(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	pool.ExpectExec(regexp.QuoteMeta(`INSERT INTO "langchain" (id, content, metadata, embedding)`)).
		WithArgs(
			pgxmock.AnyArg(), "alpha", "null", pgxmock.AnyArg(),
		).WillReturnResult(pgconn.NewCommandTag("INSERT 0 1"))

	ids, err := store.AddDocuments(t.Context(), []documents.Document{documents.New("alpha", nil)})
	if err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}
	if ids[0] == "" {
		t.Fatal("expected a generated id")
	}
}

func TestDeleteAndDeleteWithFilter(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	pool.ExpectExec(regexp.QuoteMeta(`DELETE FROM "langchain" WHERE id = ANY($1)`)).
		WithArgs([]string{"one", "two"}).
		WillReturnResult(pgconn.NewCommandTag("DELETE 2"))
	if err := store.Delete(t.Context(), []string{"one", "two"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	pool.ExpectExec(regexp.QuoteMeta(`DELETE FROM "langchain" WHERE (metadata->>'group' = $1)`)).
		WithArgs("b").
		WillReturnResult(pgconn.NewCommandTag("DELETE 1"))
	if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err != nil {
		t.Fatalf("DeleteWithFilter: %v", err)
	}

	if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": map[string]any{"$bogus": "x"}}); err == nil {
		t.Fatal("expected error for invalid filter")
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestGetByIDs(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	pool.ExpectQuery(regexp.QuoteMeta(
		`SELECT id, content, metadata FROM "langchain" WHERE id = ANY($1) ORDER BY array_position($1::text[], id)`,
	)).WithArgs([]string{"one", "missing"}).
		WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata"}).
			AddRow("one", "alpha one", []byte(`{"group":"a"}`)))

	docs, err := store.GetByIDs(t.Context(), []string{"one", "missing"})
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs: got %d want 1", len(docs))
	}
	if docs[0].ID != "one" || docs[0].PageContent != "alpha one" || docs[0].Metadata["group"] != "a" {
		t.Fatalf("doc: %#v", docs[0])
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestSimilaritySearchWithScore(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	embedder := embeddings.NewFake(8)
	vec, _ := embedder.EmbedQuery(t.Context(), "alpha")

	pool.ExpectQuery(regexp.QuoteMeta(
		`SELECT id, content, metadata, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance LIMIT $2`,
	)).WithArgs(formatVector(vec), 2).
		WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "distance"}).
			AddRow("one", "alpha one", []byte(`{"group":"a"}`), 0.25).
			AddRow("two", "alpha two", nil, 0.75))

	// SimilaritySearch delegates to SimilaritySearchWithScore, so it issues
	// the same query a second time.
	pool.ExpectQuery(regexp.QuoteMeta(
		`SELECT id, content, metadata, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance LIMIT $2`,
	)).WithArgs(formatVector(vec), 2).
		WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "distance"}).
			AddRow("one", "alpha one", []byte(`{"group":"a"}`), 0.25).
			AddRow("two", "alpha two", nil, 0.75))

	results, err := store.SimilaritySearchWithScore(t.Context(), "alpha", 2)
	if err != nil {
		t.Fatalf("SimilaritySearchWithScore: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results: got %d want 2", len(results))
	}
	if results[0].Document.ID != "one" || results[0].Score != 0.25 {
		t.Fatalf("first result: %#v", results[0])
	}

	docs, err := store.SimilaritySearch(t.Context(), "alpha", 2)
	if err != nil || len(docs) != 2 {
		t.Fatalf("SimilaritySearch: docs=%d err=%v", len(docs), err)
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestBuildFilterClause(t *testing.T) {
	tests := []struct {
		name   string
		filter map[string]any
		want   string
		args   []any
	}{
		{
			name:   "empty filter",
			filter: nil,
			want:   "",
		},
		{
			name:   "shorthand equality string",
			filter: map[string]any{"group": "a"},
			want:   `(metadata->>'group' = $1)`,
			args:   []any{"a"},
		},
		{
			name:   "eq numeric",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterEq: 2}},
			want:   `((metadata->>'page')::numeric = $1)`,
			args:   []any{2},
		},
		{
			name:   "eq bool",
			filter: map[string]any{"active": map[string]any{vectorstores.FilterEq: true}},
			want:   `(metadata->>'active' = $1)`,
			args:   []any{"true"},
		},
		{
			name:   "eq nil matches jsonb null",
			filter: map[string]any{"gone": map[string]any{vectorstores.FilterEq: nil}},
			want:   `(metadata->'gone' = $1::jsonb)`,
			args:   []any{"null"},
		},
		{
			name:   "ne string",
			filter: map[string]any{"group": map[string]any{vectorstores.FilterNe: "a"}},
			want:   `(metadata->>'group' <> $1)`,
			args:   []any{"a"},
		},
		{
			name:   "gt numeric",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterGt: 2}},
			want:   `((metadata->>'page')::numeric > $1)`,
			args:   []any{2},
		},
		{
			name:   "gte numeric",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterGte: 2.5}},
			want:   `((metadata->>'page')::numeric >= $1)`,
			args:   []any{2.5},
		},
		{
			name:   "lt string",
			filter: map[string]any{"name": map[string]any{vectorstores.FilterLt: "m"}},
			want:   `(metadata->>'name' < $1)`,
			args:   []any{"m"},
		},
		{
			name:   "lte numeric",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterLte: 10}},
			want:   `((metadata->>'page')::numeric <= $1)`,
			args:   []any{10},
		},
		{
			name:   "in strings",
			filter: map[string]any{"group": map[string]any{vectorstores.FilterIn: []any{"a", "b"}}},
			want:   `((metadata->>'group' = $1 OR metadata->>'group' = $2))`,
			args:   []any{"a", "b"},
		},
		{
			name: "in mixed types",
			filter: map[string]any{"tag": map[string]any{
				vectorstores.FilterIn: []any{"x", 3},
			}},
			want: `((metadata->>'tag' = $1 OR (metadata->>'tag')::numeric = $2))`,
			args: []any{"x", 3},
		},
		{
			name:   "nin requires field presence",
			filter: map[string]any{"group": map[string]any{vectorstores.FilterNin: []any{"a"}}},
			want:   `(metadata ? $1 AND NOT (metadata->>'group' = $2))`,
			args:   []any{"group", "a"},
		},
		{
			name:   "between numeric",
			filter: map[string]any{"page": map[string]any{vectorstores.FilterBetween: []any{2, 5}}},
			want:   `((metadata->>'page')::numeric >= $1 AND (metadata->>'page')::numeric <= $2)`,
			args:   []any{2, 5},
		},
		{
			name:   "between strings",
			filter: map[string]any{"name": map[string]any{vectorstores.FilterBetween: []any{"a", "c"}}},
			want:   `(metadata->>'name' >= $1 AND metadata->>'name' <= $2)`,
			args:   []any{"a", "c"},
		},
		{
			name:   "exists true",
			filter: map[string]any{"group": map[string]any{vectorstores.FilterExists: true}},
			want:   `(metadata ? $1)`,
			args:   []any{"group"},
		},
		{
			name:   "exists false",
			filter: map[string]any{"group": map[string]any{vectorstores.FilterExists: false}},
			want:   `(NOT (metadata ? $1))`,
			args:   []any{"group"},
		},
		{
			name:   "like",
			filter: map[string]any{"name": map[string]any{vectorstores.FilterLike: "%al%"}},
			want:   `(metadata->>'name' LIKE $1)`,
			args:   []any{"%al%"},
		},
		{
			name: "multiple fields and deterministically",
			filter: map[string]any{
				"page":  map[string]any{vectorstores.FilterGt: 2},
				"group": "a",
			},
			want: `(metadata->>'group' = $1) AND ((metadata->>'page')::numeric > $2)`,
			args: []any{"a", 2},
		},
		{
			name:   "field names with quotes are escaped",
			filter: map[string]any{"na'me": "a"},
			want:   `(metadata->>'na''me' = $1)`,
			args:   []any{"a"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args := []any{}
			got, err := buildFilterClause(tt.filter, &args)
			if err != nil {
				t.Fatalf("buildFilterClause: %v", err)
			}
			if got != tt.want {
				t.Fatalf("clause:\n got %q\nwant %q", got, tt.want)
			}
			if len(tt.args) == 0 && len(args) != 0 {
				t.Fatalf("args: got %v want none", args)
			}
			for i := range tt.args {
				if fmt.Sprint(args[i]) != fmt.Sprint(tt.args[i]) {
					t.Fatalf("args: got %v want %v", args, tt.args)
				}
			}
		})
	}

	if _, err := buildFilterClause(map[string]any{"bad": map[string]any{"$nope": 1}}, new([]any)); err == nil {
		t.Fatal("expected error for invalid filter")
	}
}

func TestSimilaritySearchWithOptions(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	embedder := embeddings.NewFake(8)
	vec, _ := embedder.EmbedQuery(t.Context(), "alpha")

	pool.ExpectQuery(regexp.QuoteMeta(
		`SELECT id, content, metadata, embedding <=> $1::vector AS distance FROM "langchain" WHERE (metadata->>'group' = $2) ORDER BY distance LIMIT $3`,
	)).WithArgs(formatVector(vec), "a", 5).
		WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "distance"}).
			AddRow("one", "alpha one", []byte(`{"group":"a"}`), 0.1).
			AddRow("two", "alpha two", []byte(`{"group":"a"}`), 0.9))

	docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
		K:              5,
		Filter:         map[string]any{"group": "a"},
		ScoreThreshold: 0.5,
	})
	if err != nil {
		t.Fatalf("SimilaritySearchWithOptions: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != "one" {
		t.Fatalf("docs (score threshold applied client-side): %#v", docs)
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestSimilaritySearchWithOptionsDefaultsAndValidation(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	embedder := embeddings.NewFake(8)
	vec, _ := embedder.EmbedQuery(t.Context(), "alpha")

	pool.ExpectQuery(regexp.QuoteMeta(
		`SELECT id, content, metadata, embedding <=> $1::vector AS distance FROM "langchain" ORDER BY distance LIMIT $2`,
	)).WithArgs(formatVector(vec), 4). // zero K defaults to 4
						WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "distance"}))

	if _, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{}); err != nil {
		t.Fatalf("search: %v", err)
	}

	if _, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
		Filter: map[string]any{"group": map[string]any{"$bogus": "a"}},
	}); err == nil {
		t.Fatal("expected error for invalid filter")
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestMMRSearchWithOptions(t *testing.T) {
	store, pool := newTestStore(t, "langchain")
	embedder := embeddings.NewFake(8)
	queryVector, _ := embedder.EmbedQuery(t.Context(), "alpha")

	// Candidate layout (lambda_mult = 0.5): "one" and "two" share the same
	// redundant vector v (0.9*q + 0.436*e2), while "three" is a moderately
	// similar but non-overlapping vector w (0.6*q + 0.8*e3). The first MMR
	// pick is "one" (highest query similarity, tie broken by row order); for
	// the second pick "two" scores 0.45 - 0.5*1 = -0.05 while "three" scores
	// 0.3 - 0.5*0.54 = 0.03, so diversity wins and "two" is skipped.
	redundant := combineVectors(queryVector, 0.9, firstZeroDimension(queryVector, 0), 0.436)
	diverse := combineVectors(queryVector, 0.6, firstZeroDimension(queryVector, 1), 0.8)

	pool.ExpectQuery(regexp.QuoteMeta(
		`SELECT id, content, metadata, embedding, embedding <=> $1::vector AS distance FROM "langchain" WHERE (metadata->>'group' = $2) ORDER BY distance LIMIT $3`,
	)).WithArgs(formatVector(queryVector), "a", 3).
		WillReturnRows(pgxmock.NewRows([]string{"id", "content", "metadata", "embedding", "distance"}).
			AddRow("one", "alpha one", []byte(`{"group":"a"}`), formatVector(redundant), 0.1).
			AddRow("two", "alpha two", []byte(`{"group":"a"}`), formatVector(redundant), 0.1).
			AddRow("three", "alpha three", []byte(`{"group":"a"}`), formatVector(diverse), 1.0))

	docs, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
		K:      2,
		FetchK: 3,
		Filter: map[string]any{"group": "a"},
	})
	if err != nil {
		t.Fatalf("MMRSearchWithOptions: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2", len(docs))
	}
	// The first pick maximizes query similarity; the second maximizes
	// diversity, so the redundant duplicate must not be selected.
	if docs[0].ID != "one" || docs[1].ID != "three" {
		t.Fatalf("docs: got [%s %s] want [one three]", docs[0].ID, docs[1].ID)
	}
	if err := pool.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestRelevanceScorePerMetric(t *testing.T) {
	cosine := &Store{metric: DistanceCosine}
	if got := cosine.relevanceScore(0.25); got != 0.75 {
		t.Fatalf("cosine relevance: got %v", got)
	}
	ip := &Store{metric: DistanceIP}
	if got := ip.relevanceScore(-1.5); got != 1.5 {
		t.Fatalf("ip relevance: got %v", got)
	}
	l2 := &Store{metric: DistanceL2}
	if got := l2.relevanceScore(0); got != 1 {
		t.Fatalf("l2 relevance: got %v", got)
	}
}

// firstZeroDimension returns the index of the nth dimension where v is zero
// (the fake embedder produces single-token unit vectors, so all-but-one
// dimensions are zero). These dimensions are orthogonal to v.
func firstZeroDimension(v []float64, nth int) int {
	seen := 0
	for i, value := range v {
		if value == 0 {
			if seen == nth {
				return i
			}
			seen++
		}
	}
	return len(v) - 1
}

// combineVectors returns coefficientA*query + coefficientB*e where e is the
// unit vector at dimension dim, yielding a unit vector when the coefficients
// satisfy a^2 + b^2 = 1.
func combineVectors(query []float64, coefficientA float64, dim int, coefficientB float64) []float64 {
	out := make([]float64, len(query))
	for i, value := range query {
		out[i] = coefficientA * value
	}
	out[dim] += coefficientB
	return out
}
