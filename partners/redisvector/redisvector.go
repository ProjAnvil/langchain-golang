// Package redisvector provides a RediSearch-backed vector store built on
// go-redis v9. The directory and package are named redisvector to avoid
// clashing with the nested langgraph/checkpoint/redis module and with the
// go-redis package itself.
//
// Python parity: this partner mirrors langchain-redis (RedisVL) in storage
// shape — documents with content, metadata, and a FLOAT32 embedding indexed
// for KNN search — with two deliberate divergences documented on New: JSON
// document storage (so arbitrary metadata round-trips losslessly) and filter
// fields must be declared up front (RediSearch filters operate on schema
// attributes; JSONPath expressions in queries are not fully supported).
package redisvector

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
	"github.com/redis/go-redis/v9"
)

// DistanceMetric selects the RediSearch vector distance metric.
type DistanceMetric string

const (
	// DistanceCosine maps to DISTANCE_METRIC COSINE (the langchain-redis
	// default); scores are cosine distances in [0, 2].
	DistanceCosine DistanceMetric = "cosine"
	// DistanceL2 maps to DISTANCE_METRIC L2 (euclidean).
	DistanceL2 DistanceMetric = "l2"
	// DistanceIP maps to DISTANCE_METRIC IP (inner product; smaller score
	// means higher similarity, like pgvector's negative inner product).
	DistanceIP DistanceMetric = "ip"
)

// MetadataFieldType selects how a metadata field is indexed by RediSearch.
// Numbers need NUMERIC for range predicates; strings and booleans need TAG
// for exact-match predicates (Redis indexes JSON booleans as TAG values).
type MetadataFieldType string

const (
	// MetadataTag indexes the field as TAG for exact matches on strings and
	// booleans.
	MetadataTag MetadataFieldType = "tag"
	// MetadataNumeric indexes the field as NUMERIC for range predicates.
	MetadataNumeric MetadataFieldType = "numeric"
)

const (
	defaultAddr      = "localhost:6379"
	scoreAlias       = "__embedding_score"
	vectorParamName  = "BLOB"
	indexPathPrefix  = "idx:"
	annIndexKind     = "HNSW"
	annAlgorithmArgs = 6 // TYPE FLOAT32 + DIM d + DISTANCE_METRIC m
)

// Client is the subset of *redis.Client the store relies on. Keeping it
// minimal lets offline tests inject a recording fake (miniredis implements
// neither RediSearch FT.* nor RedisJSON), so the emitted command sequences
// are asserted without a server.
type Client interface {
	Do(ctx context.Context, args ...any) *redis.Cmd
}

var (
	_ Client = (*redis.Client)(nil)

	_ vectorstores.VectorStore    = (*Store)(nil)
	_ vectorstores.TextAdder      = (*Store)(nil)
	_ vectorstores.OptionSearcher = (*Store)(nil)
)

// Store is a RediSearch-backed vector store: documents are JSON values under
// "<prefix>:<id>" keys, indexed by an "idx:<prefix>" FT.CREATE schema with a
// FLOAT32 HNSW embedding attribute.
type Store struct {
	client     Client
	ownsClient bool
	addr       string
	password   string
	username   string
	db         int
	embedder   embeddings.Embeddings
	keyPrefix  string
	indexName  string
	metric     DistanceMetric
	dimension  int

	// metadataDeclarations are collected by WithMetadataField and resolved
	// into metadataFields (with conflict detection) inside New.
	metadataDeclarations []metadataFieldDecl
	metadataFields       map[string]MetadataFieldType
}

// Option configures a redisvector store.
type Option func(*Store)

// WithAddr sets the Redis Stack address (host:port). Required unless
// WithClient injects a client.
func WithAddr(addr string) Option {
	return func(s *Store) { s.addr = addr }
}

// WithPassword sets the Redis AUTH password.
func WithPassword(password string) Option {
	return func(s *Store) { s.password = password }
}

// WithUsername sets the Redis ACL username (Redis 6+ ACL).
func WithUsername(username string) Option {
	return func(s *Store) { s.username = username }
}

// WithDB selects the Redis logical database index.
func WithDB(db int) Option {
	return func(s *Store) { s.db = db }
}

// WithClient injects an existing go-redis client (for tests, or to share one
// connection pool). The caller keeps ownership: Store.Close does not close an
// injected client.
func WithClient(client Client) Option {
	return func(s *Store) { s.client = client }
}

// WithEmbedder sets the embedding callback used to vectorize documents and
// queries (EmbedDocuments on writes, EmbedQuery on searches). Required.
func WithEmbedder(embedder embeddings.Embeddings) Option {
	return func(s *Store) { s.embedder = embedder }
}

// WithDimension sets the embedding vector dimension. When omitted, the
// dimension is derived from an embedder exposing Dimensions() int.
func WithDimension(dim int) Option {
	return func(s *Store) { s.dimension = dim }
}

// WithDistanceMetric selects the vector distance metric. It defaults to
// cosine.
func WithDistanceMetric(metric DistanceMetric) Option {
	return func(s *Store) { s.metric = metric }
}

// WithMetadataField declares a metadata field to index for filtering:
//
//	WithMetadataField("group", MetadataTag)     // strings/booleans
//	WithMetadataField("page", MetadataNumeric)  // numbers
//
// RediSearch filters address schema attributes, and FT.SEARCH does not fully
// support JSONPath expressions in queries, so filtering an undeclared field
// fails loudly instead of silently returning everything. Declared fields are
// created with INDEXMISSING so $exists/$ne/$nin can distinguish absent keys.
// Field names must match [A-Za-z0-9_] (they become attribute names).
// Declaring the same field again with a different type is an error.
func WithMetadataField(name string, typ MetadataFieldType) Option {
	return func(s *Store) {
		s.metadataDeclarations = append(s.metadataDeclarations, metadataFieldDecl{name: name, typ: typ})
	}
}

type metadataFieldDecl struct {
	name string
	typ  MetadataFieldType
}

// New connects (or adopts an injected client), resolves the vector dimension,
// and creates the RediSearch index when it does not exist yet:
//
//	FT.CREATE idx:<prefix> ON JSON PREFIX 1 "<prefix>:"
//	  SCHEMA $.content AS content TEXT,
//	          $.embedding AS embedding VECTOR HNSW 6
//	            TYPE FLOAT32 DIM <dim> DISTANCE_METRIC <COSINE|L2|IP>,
//	          [$.metadata.<field> AS <field> TAG|NUMERIC INDEXMISSING, ...]
//
// Divergences from langchain-redis (Python): Python stores HASH documents and
// generates its schema from the data via RedisVL, exposing a separate filter
// expression language. This port stores JSON documents so metadata round-trips
// losslessly, keeps a fixed HNSW embedding attribute, and translates the
// shared declarative DSL (see options.go) for fields declared with
// WithMetadataField.
func New(ctx context.Context, prefix string, opts ...Option) (*Store, error) {
	if strings.TrimSpace(prefix) == "" {
		return nil, fmt.Errorf("index prefix is required")
	}

	store := &Store{
		addr:           defaultAddr,
		metric:         DistanceCosine,
		metadataFields: map[string]MetadataFieldType{},
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
	for _, declaration := range store.metadataDeclarations {
		if !isValidAttributeName(declaration.name) {
			return nil, fmt.Errorf(
				"invalid metadata field name %q: must match [A-Za-z0-9_]", declaration.name,
			)
		}
		if declaration.typ != MetadataTag && declaration.typ != MetadataNumeric {
			return nil, fmt.Errorf(
				"invalid type %q for metadata field %q: expected tag or numeric",
				declaration.typ, declaration.name,
			)
		}
		if existing, ok := store.metadataFields[declaration.name]; ok && existing != declaration.typ {
			return nil, fmt.Errorf(
				"conflicting declarations for metadata field %q: %s then %s",
				declaration.name, existing, declaration.typ,
			)
		}
		store.metadataFields[declaration.name] = declaration.typ
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

	if store.client == nil {
		if strings.TrimSpace(store.addr) == "" {
			return nil, fmt.Errorf("a server address is required: pass WithAddr or WithClient")
		}
		store.client = redis.NewClient(&redis.Options{
			Addr:     store.addr,
			Username: store.username,
			Password: store.password,
			DB:       store.db,
		})
		store.ownsClient = true
	}

	store.keyPrefix = prefix + ":"
	store.indexName = indexPathPrefix + prefix

	if err := store.ensureIndex(ctx); err != nil {
		if store.ownsClient {
			_ = store.client.Do(ctx, "QUIT")
		}
		return nil, err
	}
	return store, nil
}

// ensureIndex skips creation when FT.INFO reports the index already exists.
func (s *Store) ensureIndex(ctx context.Context) error {
	if err := s.client.Do(ctx, "FT.INFO", s.indexName).Err(); err == nil {
		return nil
	}

	args := []any{
		"FT.CREATE", s.indexName, "ON", "JSON", "PREFIX", "1", s.keyPrefix, "SCHEMA",
		"$.content", "AS", "content", "TEXT",
		"$.embedding", "AS", "embedding", "VECTOR", annIndexKind, strconv.Itoa(annAlgorithmArgs),
		"TYPE", "FLOAT32", "DIM", s.dimension, "DISTANCE_METRIC", s.metric.rediSearchMetric(),
	}
	for _, name := range slices.Sorted(maps.Keys(s.metadataFields)) {
		path := "$.metadata." + name
		switch s.metadataFields[name] {
		case MetadataNumeric:
			args = append(args, path, "AS", name, "NUMERIC", "INDEXMISSING")
		default:
			args = append(args, path, "AS", name, "TAG", "INDEXMISSING")
		}
	}
	return s.client.Do(ctx, args...).Err()
}

// rediSearchMetric maps the public metric to its FT.CREATE spelling.
func (m DistanceMetric) rediSearchMetric() string {
	switch m {
	case DistanceL2:
		return "L2"
	case DistanceIP:
		return "IP"
	default:
		return "COSINE"
	}
}

// relevanceScore converts a RediSearch distance into a [0, 1] relevance score
// using the shared langchain helpers (cosine: 1-d; L2 assumes normalized
// embeddings; IP treats the score as negative inner product, like pgvector).
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

// Close releases the client when the store created it. Injected clients
// (WithClient) stay owned by the caller.
func (s *Store) Close() {
	if s.ownsClient {
		if closer, ok := s.client.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
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

// AddDocuments embeds and stores documents as JSON values under
// "<prefix>:<id>" keys.
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

	ids := make([]string, len(docs))
	for i, doc := range docs {
		id := doc.ID
		if id == "" {
			id = newID()
		}
		payload, err := json.Marshal(storedDocument{
			Content:   doc.PageContent,
			Metadata:  cloneMetadata(doc.Metadata),
			Embedding: vectors[i],
		})
		if err != nil {
			return nil, fmt.Errorf("encode doc %q: %w", id, err)
		}
		if err := s.client.Do(ctx, "JSON.SET", s.keyPrefix+id, "$", string(payload)).Err(); err != nil {
			return nil, fmt.Errorf("store doc %q: %w", id, err)
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
	keys := make([]any, 0, len(ids)+1)
	keys = append(keys, "DEL")
	for _, id := range ids {
		keys = append(keys, s.keyPrefix+id)
	}
	return s.client.Do(ctx, keys...).Err()
}

// GetByIDs returns found documents. Missing ids are skipped.
func (s *Store) GetByIDs(ctx context.Context, ids []string) ([]documents.Document, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	docs := make([]documents.Document, 0, len(ids))
	for _, id := range ids {
		reply := s.client.Do(ctx, "JSON.GET", s.keyPrefix+id)
		if err := reply.Err(); err != nil {
			if isRedisNil(err) {
				continue
			}
			return nil, err
		}
		raw, err := reply.Text()
		if err != nil {
			return nil, err
		}
		var payload storedDocument
		if err := json.Unmarshal([]byte(raw), &payload); err != nil {
			return nil, fmt.Errorf("decode doc %q: %w", id, err)
		}
		docs = append(docs, documents.Document{
			ID:          id,
			PageContent: payload.Content,
			Metadata:    payload.Metadata,
		})
	}
	return docs, nil
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
// RediSearch distances (lower == more similar).
func (s *Store) SimilaritySearchWithScore(
	ctx context.Context,
	query string,
	k int,
) ([]vectorstores.SearchResult, error) {
	if k <= 0 {
		k = 4
	}
	hits, err := s.knnSearch(ctx, query, nil, k)
	if err != nil {
		return nil, err
	}
	results := make([]vectorstores.SearchResult, 0, len(hits))
	for _, hit := range hits {
		results = append(results, vectorstores.SearchResult{Document: hit.doc, Score: hit.score})
	}
	return results, nil
}

// searchHit is one FT.SEARCH result with its KNN distance and stored
// embedding (used by MMR re-ranking).
type searchHit struct {
	doc       documents.Document
	score     float64
	embedding []float64
}

// knnSearch runs the KNN FT.SEARCH. filter may be nil; count is the KNN k
// (and LIMIT) to request.
func (s *Store) knnSearch(
	ctx context.Context,
	query string,
	filter map[string]any,
	count int,
) ([]searchHit, error) {
	queryVector, err := s.embedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	filterClause, err := s.buildFilterQuery(filter)
	if err != nil {
		return nil, err
	}
	knnExpr := fmt.Sprintf("[KNN %d @embedding $%s AS %s]", count, vectorParamName, scoreAlias)
	knn := "*=>" + knnExpr
	if filterClause != "" {
		knn = "(" + filterClause + ")=>" + knnExpr
	}

	reply := s.client.Do(ctx,
		"FT.SEARCH", s.indexName, knn,
		"SORTBY", scoreAlias,
		"LIMIT", 0, count,
		"PARAMS", 2, vectorParamName, encodeBlob(queryVector),
		"DIALECT", 2,
	)
	if err := reply.Err(); err != nil {
		return nil, err
	}
	raw, err := reply.Result()
	if err != nil {
		return nil, err
	}
	return parseSearchReply(raw, s.keyPrefix)
}

// embedQuery runs the embedder and validates the dimension.
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

// storedDocument is the JSON shape stored per document key.
type storedDocument struct {
	Content   string         `json:"content"`
	Metadata  map[string]any `json:"metadata"`
	Embedding []float64      `json:"embedding"`
}

// parseSearchReply maps an FT.SEARCH reply to hits. Two reply shapes occur:
//
//   - RESP2 (flat array): [total, id1, fields1, id2, fields2, ...] where
//     fieldsN is a flat [key, value, ...] array (absent with NOCONTENT).
//   - RESP3 (go-redis negotiates it by default): a map with "total_results"
//     and "results", the latter a list of maps carrying "id" plus field
//     values under "extra_attributes" and/or "values".
//
// Document ids are the full Redis keys; keyPrefix is stripped to recover the
// caller-facing id.
func parseSearchReply(raw any, keyPrefix string) ([]searchHit, error) {
	var hits []searchHit
	addHit := func(id string, fields map[string]any) error {
		if id == "" {
			return nil
		}
		doc := documents.Document{ID: strings.TrimPrefix(id, keyPrefix)}
		var payload storedDocument
		if body, ok := fields["$"].(string); ok {
			if err := json.Unmarshal([]byte(body), &payload); err != nil {
				return fmt.Errorf("decode search result %q: %w", id, err)
			}
		}
		doc.PageContent = payload.Content
		doc.Metadata = payload.Metadata
		hit := searchHit{doc: doc, embedding: payload.Embedding}
		if score, ok := fields[scoreAlias].(string); ok {
			parsed, err := strconv.ParseFloat(score, 64)
			if err != nil {
				return fmt.Errorf("parse score for %q: %w", id, err)
			}
			hit.score = parsed
		}
		hits = append(hits, hit)
		return nil
	}

	if envelope, ok := raw.(map[any]any); ok {
		results, ok := envelope["results"].([]any)
		if !ok {
			return nil, nil
		}
		for _, resultAny := range results {
			result, ok := resultAny.(map[any]any)
			if !ok {
				continue
			}
			id, _ := result["id"].(string)
			fields := map[string]any{}
			if extra, ok := result["extra_attributes"].(map[any]any); ok {
				for keyAny, value := range extra {
					if key, ok := keyAny.(string); ok {
						fields[key] = value
					}
				}
			}
			if values, ok := result["values"].([]any); ok {
				for j := 0; j+1 < len(values); j += 2 {
					if key, ok := values[j].(string); ok {
						fields[key] = values[j+1]
					}
				}
			}
			if err := addHit(id, fields); err != nil {
				return nil, err
			}
		}
		return hits, nil
	}

	top, ok := raw.([]any)
	if !ok || len(top) < 2 {
		return nil, nil
	}
	if documentMap, ok := top[1].(map[any]any); ok {
		for idAny, fieldsAny := range documentMap {
			id, ok := idAny.(string)
			if !ok {
				continue
			}
			fields := map[string]any{}
			if fieldMap, ok := fieldsAny.(map[any]any); ok {
				for keyAny, value := range fieldMap {
					if key, ok := keyAny.(string); ok {
						fields[key] = value
					}
				}
			}
			if err := addHit(id, fields); err != nil {
				return nil, err
			}
		}
		return hits, nil
	}

	for i := 1; i < len(top); {
		id, ok := top[i].(string)
		if !ok {
			i++
			continue
		}
		fields := map[string]any{}
		if i+1 < len(top) {
			if flat, ok := top[i+1].([]any); ok {
				for j := 0; j+1 < len(flat); j += 2 {
					if key, ok := flat[j].(string); ok {
						fields[key] = flat[j+1]
					}
				}
				i += 2
			} else {
				i++
			}
		} else {
			i++
		}
		if err := addHit(id, fields); err != nil {
			return nil, err
		}
	}
	return hits, nil
}

// encodeBlob serializes a vector into the little-endian FLOAT32 byte blob
// RediSearch expects as a KNN query parameter.
func encodeBlob(vector []float64) []byte {
	blob := make([]byte, 4*len(vector))
	for i, value := range vector {
		binary.LittleEndian.PutUint32(blob[i*4:], math.Float32bits(float32(value)))
	}
	return blob
}

// isRedisNil reports whether err is redis.Nil (missing key).
func isRedisNil(err error) bool {
	return errors.Is(err, redis.Nil)
}

func cloneMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		out[key] = value
	}
	return out
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

func isValidAttributeName(name string) bool {
	if name == "" {
		return false
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_':
		default:
			return false
		}
	}
	return true
}
