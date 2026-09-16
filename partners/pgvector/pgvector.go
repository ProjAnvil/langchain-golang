// Package pgvector provides a pgvector-backed vector store built on pgx v5.
//
// Python parity: this partner mirrors langchain-postgres PGVector
// (langchain_postgres/vectorstores.py) — the declarative filter operator set,
// distance-to-relevance conversions, and score-threshold behavior follow that
// library — but the physical schema is a simplified one-table-per-collection
// layout instead of Python's langchain_pg_collection/langchain_pg_embedding
// registry pair (see the Divergence note on New).
package pgvector

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// DistanceMetric selects the pgvector distance metric. It also decides the
// index opclass and the SQL distance operator used for searches.
type DistanceMetric string

const (
	// DistanceCosine is cosine distance (`<=>`, opclass vector_cosine_ops),
	// the langchain-postgres default.
	DistanceCosine DistanceMetric = "cosine"
	// DistanceL2 is euclidean distance (`<->`, opclass vector_l2_ops).
	DistanceL2 DistanceMetric = "l2"
	// DistanceIP is negative inner product (`<#>`, opclass vector_ip_ops);
	// smaller distances mean higher inner product.
	DistanceIP DistanceMetric = "ip"
)

// IndexKind selects the ANN index algorithm created for the embedding column.
type IndexKind string

const (
	// IndexHNSW creates an hnsw index (default).
	IndexHNSW IndexKind = "hnsw"
	// IndexIVFFlat creates an ivfflat index with the default 100 lists.
	IndexIVFFlat IndexKind = "ivfflat"
)

const ivfflatLists = 100

// Pool is the subset of *pgxpool.Pool the store relies on. Keeping it minimal
// lets tests inject a pgxmock pool (github.com/pashagolub/pgxmock/v4), which
// satisfies this interface, so all SQL construction is verified offline.
type Pool interface {
	Exec(ctx context.Context, sql string, arguments ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
	Close()
}

var (
	_ Pool = (*pgxpool.Pool)(nil)

	_ vectorstores.VectorStore    = (*Store)(nil)
	_ vectorstores.TextAdder      = (*Store)(nil)
	_ vectorstores.OptionSearcher = (*Store)(nil)
)

// Store is a pgvector-backed vector store: one collection maps to one table.
type Store struct {
	pool      Pool
	ownsPool  bool
	dsn       string
	embedder  embeddings.Embeddings
	table     string
	metric    DistanceMetric
	indexKind IndexKind
	dimension int
}

// Option configures a pgvector store.
type Option func(*Store)

// WithDSN sets the Postgres connection string (URL or DSN form). Required
// unless WithPool injects a pool.
func WithDSN(dsn string) Option {
	return func(s *Store) { s.dsn = dsn }
}

// WithURL is an alias for WithDSN that reads better for URL-form connection
// strings (postgres://user:pass@host/db).
func WithURL(url string) Option {
	return WithDSN(url)
}

// WithPool injects an existing pool (for tests, or to share one pgxpool
// across collections). The caller keeps ownership: Store.Close does not close
// an injected pool.
func WithPool(pool Pool) Option {
	return func(s *Store) { s.pool = pool }
}

// WithEmbedder sets the embedding callback used to vectorize documents and
// queries (the chroma-style embedder integration: EmbedDocuments on writes,
// EmbedQuery on searches). Required.
func WithEmbedder(embedder embeddings.Embeddings) Option {
	return func(s *Store) { s.embedder = embedder }
}

// WithDistanceMetric selects the distance metric. It defaults to cosine.
func WithDistanceMetric(metric DistanceMetric) Option {
	return func(s *Store) { s.metric = metric }
}

// WithIndexKind selects the ANN index kind. It defaults to hnsw.
func WithIndexKind(kind IndexKind) Option {
	return func(s *Store) { s.indexKind = kind }
}

// WithDimension sets the embedding vector dimension. When omitted, the
// dimension is derived from an embedder exposing Dimensions() int.
func WithDimension(dim int) Option {
	return func(s *Store) { s.dimension = dim }
}

// New connects (or adopts an injected pool), resolves the vector dimension,
// and runs the idempotent schema DDL:
//
//	CREATE EXTENSION IF NOT EXISTS vector
//	CREATE TABLE IF NOT EXISTS "<collection>" (id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata JSONB, embedding vector(<dim>))
//	CREATE INDEX IF NOT EXISTS "<collection>_embedding_idx" ON "<collection>" USING hnsw|ivfflat (embedding vector_cosine_ops|...)
//
// Divergence from langchain-postgres (Python): Python keeps a single
// langchain_pg_embedding table plus a langchain_pg_collection registry so
// several collections share one physical table (uuid primary key, cmetadata
// column). This port keeps one table per collection with self-describing
// column names (id/content/metadata) because Go callers manage their own
// schema lifecycle and a table-per-collection maps 1:1 onto the SQL generated
// for filters and searches. The CREATE EXTENSION failure is tolerated
// (matching Python's create_vector_extension best-effort behavior): the
// extension is usually preinstalled, and a genuinely missing one surfaces in
// the table DDL that follows.
func New(ctx context.Context, collectionName string, opts ...Option) (*Store, error) {
	if strings.TrimSpace(collectionName) == "" {
		return nil, fmt.Errorf("collection name is required")
	}

	store := &Store{
		table:     collectionName,
		metric:    DistanceCosine,
		indexKind: IndexHNSW,
	}
	for _, opt := range opts {
		opt(store)
	}
	if store.embedder == nil {
		return nil, fmt.Errorf("embedder is required: pass WithEmbedder")
	}
	if !slices.Contains([]DistanceMetric{DistanceCosine, DistanceL2, DistanceIP}, store.metric) {
		return nil, fmt.Errorf(
			"invalid distance metric %q: expected one of cosine, l2, ip", store.metric,
		)
	}
	if !slices.Contains([]IndexKind{IndexHNSW, IndexIVFFlat}, store.indexKind) {
		return nil, fmt.Errorf(
			"invalid index kind %q: expected one of hnsw, ivfflat", store.indexKind,
		)
	}
	if store.dimension <= 0 {
		if dimmer, ok := store.embedder.(interface{ Dimensions() int }); ok {
			store.dimension = dimmer.Dimensions()
		}
	}
	if store.dimension <= 0 {
		return nil, fmt.Errorf(
			"vector dimension is required: pass WithDimension or use an embedder exposing Dimensions()",
		)
	}

	if store.pool == nil {
		if strings.TrimSpace(store.dsn) == "" {
			return nil, fmt.Errorf("a connection string is required: pass WithDSN or WithPool")
		}
		pool, err := pgxpool.New(ctx, store.dsn)
		if err != nil {
			return nil, fmt.Errorf("connect postgres: %w", err)
		}
		store.pool = pool
		store.ownsPool = true
	}

	if err := store.initSchema(ctx); err != nil {
		if store.ownsPool {
			store.pool.Close()
		}
		return nil, err
	}
	return store, nil
}

// initSchema runs the idempotent DDL for the collection table and its ANN
// index.
func (s *Store) initSchema(ctx context.Context) error {
	// Best-effort: fails without privileges, which is fine when the
	// extension already exists (see the New doc comment).
	_, _ = s.pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")

	if _, err := s.pool.Exec(ctx, fmt.Sprintf(
		`CREATE TABLE IF NOT EXISTS %s (id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata JSONB, embedding vector(%d))`,
		s.quotedTable(), s.dimension,
	)); err != nil {
		return fmt.Errorf("create table %q: %w", s.table, err)
	}

	if _, err := s.pool.Exec(ctx, fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS %s ON %s USING %s (embedding %s)%s`,
		s.quotedIndexName(), s.quotedTable(), s.indexKind, s.opclass(), s.indexOptions(),
	)); err != nil {
		return fmt.Errorf("create index on %q: %w", s.table, err)
	}
	return nil
}

func (s *Store) quotedTable() string {
	return `"` + strings.ReplaceAll(s.table, `"`, `""`) + `"`
}

func (s *Store) quotedIndexName() string {
	return `"` + strings.ReplaceAll(s.table+"_embedding_idx", `"`, `""`) + `"`
}

func (s *Store) opclass() string {
	switch s.metric {
	case DistanceL2:
		return "vector_l2_ops"
	case DistanceIP:
		return "vector_ip_ops"
	default:
		return "vector_cosine_ops"
	}
}

func (s *Store) indexOptions() string {
	if s.indexKind == IndexIVFFlat {
		return fmt.Sprintf(" WITH (lists = %d)", ivfflatLists)
	}
	return ""
}

// distanceOperator returns the pgvector operator for the configured metric
// (`<=>` cosine, `<->` euclidean, `<#>` negative inner product).
func (s *Store) distanceOperator() string {
	switch s.metric {
	case DistanceL2:
		return "<->"
	case DistanceIP:
		return "<#>"
	default:
		return "<=>"
	}
}

// relevanceScore converts a pgvector distance into a [0, 1] relevance score
// (1 == most similar) using the same helpers langchain-pgvector selects per
// distance strategy.
func (s *Store) relevanceScore(distance float64) float64 {
	switch s.metric {
	case DistanceL2:
		return vectorstores.EuclideanRelevanceScore(distance)
	case DistanceIP:
		return vectorstores.MaxInnerProductRelevanceScore(distance)
	default:
		return vectorstores.CosineRelevanceScore(distance)
	}
}

// Close releases the pool when the store created it from a DSN. Injected
// pools (WithPool) stay owned by the caller.
func (s *Store) Close() {
	if s.ownsPool {
		s.pool.Close()
	}
}

// AddTexts implements vectorstores.TextAdder.
func (s *Store) AddTexts(
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

// AddDocuments embeds and upserts documents into the collection table.
// Divergence note: langchain-postgres rejects duplicate ids with a unique
// constraint error; this port upserts (ON CONFLICT DO UPDATE), which keeps
// retries idempotent and matches the chroma partner behavior.
func (s *Store) AddDocuments(
	ctx context.Context,
	docs []documents.Document,
) ([]string, error) {
	if len(docs) == 0 {
		return nil, nil
	}
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

	sql := fmt.Sprintf(
		`INSERT INTO %s (id, content, metadata, embedding) VALUES ($1, $2, $3::jsonb, $4::vector) `+
			`ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding`,
		s.quotedTable(),
	)

	ids := make([]string, len(docs))
	for i, doc := range docs {
		id := doc.ID
		if id == "" {
			id = newID()
		}
		// Empty metadata is stored as JSON null, matching langchain-postgres
		// (jsonb NULL); `metadata->>'x'` then yields SQL NULL for filters.
		metadataJSON := "null"
		if len(doc.Metadata) > 0 {
			encoded, err := json.Marshal(doc.Metadata)
			if err != nil {
				return nil, fmt.Errorf("marshal metadata for doc %q: %w", id, err)
			}
			metadataJSON = string(encoded)
		}
		if _, err := s.pool.Exec(ctx, sql, id, doc.PageContent, metadataJSON, formatVector(vectors[i])); err != nil {
			return nil, fmt.Errorf("insert doc %q: %w", id, err)
		}
		ids[i] = id
	}
	return ids, nil
}

// Delete removes documents by id. Missing ids are ignored.
func (s *Store) Delete(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	_, err := s.pool.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE id = ANY($1)`, s.quotedTable()),
		ids,
	)
	return err
}

// DeleteWithFilter removes documents matching the declarative filter DSL
// (same translation as filtered searches).
func (s *Store) DeleteWithFilter(ctx context.Context, filter map[string]any) error {
	args := make([]any, 0, len(filter))
	where, err := buildFilterClause(filter, &args)
	if err != nil {
		return err
	}
	if where == "" {
		return fmt.Errorf("refusing to delete with an empty filter")
	}
	_, err = s.pool.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE %s`, s.quotedTable(), where),
		args...,
	)
	return err
}

// GetByIDs returns found documents in request order. Missing ids are skipped.
func (s *Store) GetByIDs(ctx context.Context, ids []string) ([]documents.Document, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	rows, err := s.pool.Query(ctx, fmt.Sprintf(
		`SELECT id, content, metadata FROM %s WHERE id = ANY($1) ORDER BY array_position($1::text[], id)`,
		s.quotedTable(),
	), ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	docs := make([]documents.Document, 0, len(ids))
	for rows.Next() {
		var (
			id      string
			content string
			rawMeta []byte
		)
		if err := rows.Scan(&id, &content, &rawMeta); err != nil {
			return nil, err
		}
		docs = append(docs, documents.Document{
			ID:          id,
			PageContent: content,
			Metadata:    decodeMetadata(rawMeta),
		})
	}
	return docs, rows.Err()
}

// SimilaritySearch returns the top k documents for query.
func (s *Store) SimilaritySearch(
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

// SimilaritySearchWithScore returns the top k documents with their raw
// pgvector distances (lower == more similar, matching langchain-postgres
// similarity_search_with_score semantics; use relevanceScore-aware searches
// for [0, 1] scores).
func (s *Store) SimilaritySearchWithScore(
	ctx context.Context,
	query string,
	k int,
) ([]vectorstores.SearchResult, error) {
	if k <= 0 {
		k = 4
	}
	vector, err := s.embedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	args := []any{formatVector(vector)}
	sql := fmt.Sprintf(
		`SELECT id, content, metadata, embedding %s $1::vector AS distance FROM %s ORDER BY distance`,
		s.distanceOperator(), s.quotedTable(),
	)
	return s.queryResults(ctx, sql, args, k)
}

// queryResults appends the LIMIT placeholder after the given args and maps
// rows to results. The SQL must not embed its own LIMIT clause.
func (s *Store) queryResults(
	ctx context.Context,
	sql string,
	args []any,
	limit int,
) ([]vectorstores.SearchResult, error) {
	args = append(args, limit)
	sql = fmt.Sprintf("%s LIMIT $%d", sql, len(args))
	rows, err := s.pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	results := make([]vectorstores.SearchResult, 0, limit)
	for rows.Next() {
		var (
			id       string
			content  string
			rawMeta  []byte
			distance float64
		)
		if err := rows.Scan(&id, &content, &rawMeta, &distance); err != nil {
			return nil, err
		}
		results = append(results, vectorstores.SearchResult{
			Document: documents.Document{
				ID:          id,
				PageContent: content,
				Metadata:    decodeMetadata(rawMeta),
			},
			Score: distance,
		})
	}
	return results, rows.Err()
}

// embedQuery runs the embedder and validates the dimension matches the table.
func (s *Store) embedQuery(ctx context.Context, query string) ([]float64, error) {
	if s.embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}
	vector, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	if len(vector) != s.dimension {
		return nil, fmt.Errorf("query embedding dimension mismatch: got %d want %d", len(vector), s.dimension)
	}
	return vector, nil
}

// searchWithFilterSQL builds the filtered search statement after embedding
// the query; see searchWithFilterSQLByVector for the SQL shape.
func (s *Store) searchWithFilterSQL(
	ctx context.Context,
	query string,
	filter map[string]any,
	withEmbedding bool,
) (string, []any, error) {
	queryVector, err := s.embedQuery(ctx, query)
	if err != nil {
		return "", nil, err
	}
	return s.searchWithFilterSQLByVector(queryVector, filter, withEmbedding)
}

// searchWithFilterSQLByVector builds the filtered search statement from an
// already-embedded query vector; the query vector is always $1, filter
// operands follow, and the LIMIT placeholder is appended by queryResults.
// When withEmbedding is set the stored embedding column is selected too
// (used by MMR re-ranking).
func (s *Store) searchWithFilterSQLByVector(
	queryVector []float64,
	filter map[string]any,
	withEmbedding bool,
) (string, []any, error) {
	args := []any{formatVector(queryVector)}
	where, err := buildFilterClause(filter, &args)
	if err != nil {
		return "", nil, err
	}
	embeddingColumn := ""
	if withEmbedding {
		embeddingColumn = "embedding, "
	}
	sql := fmt.Sprintf(
		`SELECT id, content, metadata, %sembedding %s $1::vector AS distance FROM %s`,
		embeddingColumn, s.distanceOperator(), s.quotedTable(),
	)
	if where != "" {
		sql += " WHERE " + where
	}
	sql += " ORDER BY distance"
	return sql, args, nil
}

// decodeMetadata parses a jsonb column payload; SQL NULL and JSON null both
// yield nil metadata.
func decodeMetadata(raw []byte) map[string]any {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var metadata map[string]any
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return nil
	}
	return metadata
}

// formatVector renders a float64 vector in pgvector literal form
// ("[1,0.5,-0.25]"); values are bound as text and cast with ::vector, which
// avoids a dedicated pgvector codec dependency.
func formatVector(vector []float64) string {
	var builder strings.Builder
	builder.WriteByte('[')
	for i, value := range vector {
		if i > 0 {
			builder.WriteByte(',')
		}
		builder.WriteString(strconv.FormatFloat(value, 'g', -1, 64))
	}
	builder.WriteByte(']')
	return builder.String()
}

// parseVector parses a pgvector text representation back into floats (NULL
// yields nil, nil).
func parseVector(text string) ([]float64, error) {
	if text == "" {
		return nil, nil
	}
	trimmed := strings.Trim(text, "[]")
	if trimmed == "" {
		return []float64{}, nil
	}
	parts := strings.Split(trimmed, ",")
	out := make([]float64, 0, len(parts))
	for _, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("parse pgvector %q: %w", text, err)
		}
		out = append(out, value)
	}
	return out, nil
}

func metadataAt(metadatas []map[string]any, index int) map[string]any {
	if index < len(metadatas) {
		return metadatas[index]
	}
	return nil
}

func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("doc-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}
