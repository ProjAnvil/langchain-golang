package redisvector

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
	"github.com/redis/go-redis/v9"
)

// fakeReply is one queued reply for fakeClient.
type fakeReply struct {
	value any
	err   error
}

// fakeClient is a recording stand-in for *redis.Client: miniredis does not
// implement RediSearch FT.* or RedisJSON commands, so offline tests assert the
// exact command sequences the store issues and replay canned replies.
type fakeClient struct {
	mu      sync.Mutex
	calls   [][]any
	replies []fakeReply
}

func (f *fakeClient) Do(_ context.Context, args ...any) *redis.Cmd {
	f.mu.Lock()
	defer f.mu.Unlock()
	call := make([]any, len(args))
	copy(call, args)
	f.calls = append(f.calls, call)
	var reply fakeReply
	if len(f.replies) > 0 {
		reply, f.replies = f.replies[0], f.replies[1:]
	}
	return redis.NewCmdResult(reply.value, reply.err)
}

func (f *fakeClient) call(i int) []any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[i]
}

func (f *fakeClient) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

// lastCall returns the most recent command.
func (f *fakeClient) lastCall() []any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[len(f.calls)-1]
}

// indexAbsentReply is the reply sequence for New against a server where the
// index does not exist yet (FT.INFO fails, FT.CREATE succeeds).
var indexAbsentReply = []fakeReply{
	{err: errors.New("ERR unknown index name")},
	{value: "OK"},
}

func newFakeStore(t *testing.T, replies ...fakeReply) (*Store, *fakeClient) {
	t.Helper()
	client := &fakeClient{replies: replies}
	store, err := New(t.Context(), "docs",
		WithClient(client),
		WithEmbedder(embeddings.NewFake(8)),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return store, client
}

func withDefaultSchema(fields ...fakeReply) []fakeReply {
	return append(append([]fakeReply{}, indexAbsentReply...), fields...)
}

func TestNewCreatesIndex(t *testing.T) {
	store, client := newFakeStore(t, indexAbsentReply...)

	if store.indexName != "idx:docs" {
		t.Fatalf("index name: got %q want idx:docs", store.indexName)
	}
	if store.keyPrefix != "docs:" {
		t.Fatalf("key prefix: got %q want docs:", store.keyPrefix)
	}

	want := []any{
		"FT.CREATE", "idx:docs", "ON", "JSON", "PREFIX", "1", "docs:", "SCHEMA",
		"$.content", "AS", "content", "TEXT",
		"$.embedding", "AS", "embedding", "VECTOR", "HNSW", "6",
		"TYPE", "FLOAT32", "DIM", 8, "DISTANCE_METRIC", "COSINE",
	}
	if got := client.call(1); !equalArgs(got, want) {
		t.Fatalf("FT.CREATE args:\n got %v\nwant %v", got, want)
	}
}

func TestNewSkipsCreateWhenIndexExists(t *testing.T) {
	client := &fakeClient{replies: []fakeReply{{value: map[string]any{}}}}
	_, err := New(t.Context(), "docs",
		WithClient(client),
		WithEmbedder(embeddings.NewFake(8)),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if client.callCount() != 1 {
		t.Fatalf("expected only FT.INFO, got %d calls", client.callCount())
	}
	if got := client.call(0)[0]; got != "FT.INFO" {
		t.Fatalf("first command: got %v want FT.INFO", got)
	}
}

func TestNewOptions(t *testing.T) {
	newStore := func(t *testing.T, client *fakeClient, opts ...Option) (*Store, error) {
		return New(t.Context(), "docs",
			append([]Option{WithClient(client), WithEmbedder(embeddings.NewFake(4))}, opts...)...,
		)
	}

	t.Run("declared metadata fields join the schema", func(t *testing.T) {
		client := &fakeClient{replies: indexAbsentReply}
		if _, err := newStore(t, client,
			WithMetadataField("group", MetadataTag),
			WithMetadataField("page", MetadataNumeric),
			WithDistanceMetric(DistanceL2),
		); err != nil {
			t.Fatalf("New: %v", err)
		}
		want := []any{
			"FT.CREATE", "idx:docs", "ON", "JSON", "PREFIX", "1", "docs:", "SCHEMA",
			"$.content", "AS", "content", "TEXT",
			"$.embedding", "AS", "embedding", "VECTOR", "HNSW", "6",
			"TYPE", "FLOAT32", "DIM", 4, "DISTANCE_METRIC", "L2",
			"$.metadata.group", "AS", "group", "TAG", "INDEXMISSING",
			"$.metadata.page", "AS", "page", "NUMERIC", "INDEXMISSING",
		}
		if got := client.call(1); !equalArgs(got, want) {
			t.Fatalf("FT.CREATE args:\n got %v\nwant %v", got, want)
		}
	})

	t.Run("missing embedder", func(t *testing.T) {
		if _, err := New(t.Context(), "docs", WithClient(&fakeClient{})); err == nil {
			t.Fatal("expected error for missing embedder")
		}
	})

	t.Run("unreachable server fails at New", func(t *testing.T) {
		// The address defaults to localhost:6379; ensureIndex probes FT.INFO
		// eagerly, so an unreachable server surfaces from New.
		if _, err := New(t.Context(), "docs",
			WithAddr("localhost:1"),
			WithEmbedder(embeddings.NewFake(4)),
		); err == nil {
			t.Fatal("expected error for unreachable server")
		}
	})

	t.Run("unknown dimension without Dimensions embedder", func(t *testing.T) {
		_, err := New(t.Context(), "docs",
			WithClient(&fakeClient{}),
			WithEmbedder(embeddings.Static{}),
		)
		if err == nil {
			t.Fatal("expected error for unknown vector dimension")
		}
	})

	t.Run("invalid distance metric", func(t *testing.T) {
		if _, err := newStore(t, &fakeClient{}, WithDistanceMetric("manhattan")); err == nil {
			t.Fatal("expected error for invalid distance metric")
		}
	})

	t.Run("invalid metadata field name", func(t *testing.T) {
		if _, err := newStore(t, &fakeClient{}, WithMetadataField("bad field!", MetadataTag)); err == nil {
			t.Fatal("expected error for invalid metadata field name")
		}
	})

	t.Run("invalid metadata field type", func(t *testing.T) {
		if _, err := newStore(t, &fakeClient{}, WithMetadataField("group", "geo")); err == nil {
			t.Fatal("expected error for invalid metadata field type")
		}
	})

	t.Run("empty prefix", func(t *testing.T) {
		if _, err := New(t.Context(), "  ", WithClient(&fakeClient{}), WithEmbedder(embeddings.NewFake(4))); err == nil {
			t.Fatal("expected error for empty prefix")
		}
	})

	t.Run("duplicate metadata field", func(t *testing.T) {
		if _, err := newStore(t, &fakeClient{},
			WithMetadataField("group", MetadataTag),
			WithMetadataField("group", MetadataNumeric),
		); err == nil {
			t.Fatal("expected error for conflicting duplicate field")
		}
	})
}

func TestEncodeBlob(t *testing.T) {
	blob := encodeBlob([]float64{1, -0.5})
	if len(blob) != 8 {
		t.Fatalf("blob length: got %d want 8", len(blob))
	}
	if got := math.Float32frombits(binary.LittleEndian.Uint32(blob[4:])); got != -0.5 {
		t.Fatalf("second float32: got %v want -0.5", got)
	}
}

func TestAddDocumentsWritesJSON(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(fakeReply{value: "OK"}, fakeReply{value: "OK"})...)

	ids, err := store.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha one", map[string]any{"group": "a", "page": 1}).WithID("one"),
		documents.New("alpha two", nil).WithID("two"),
	})
	if err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}
	if len(ids) != 2 || ids[0] != "one" || ids[1] != "two" {
		t.Fatalf("ids: %v", ids)
	}

	var payload storedDocument
	got := client.call(2)
	if got[0] != "JSON.SET" || got[1] != "docs:one" || got[2] != "$" {
		t.Fatalf("JSON.SET args: %v", got)
	}
	if err := json.Unmarshal([]byte(client.call(2)[3].(string)), &payload); err != nil {
		t.Fatalf("stored json: %v", err)
	}
	if payload.Content != "alpha one" {
		t.Fatalf("content: %q", payload.Content)
	}
	if payload.Metadata["group"] != "a" {
		t.Fatalf("metadata: %#v", payload.Metadata)
	}
	if len(payload.Embedding) != 8 {
		t.Fatalf("embedding dim: %d", len(payload.Embedding))
	}
	// Documents without metadata still store an empty object so jsonpath
	// lookups behave uniformly.
	var payloadTwo storedDocument
	if err := json.Unmarshal([]byte(client.call(3)[3].(string)), &payloadTwo); err != nil {
		t.Fatalf("stored json two: %v", err)
	}
	if payloadTwo.Metadata == nil {
		t.Fatalf("metadata should be an object, got nil")
	}
}

func TestGetByIDsAndDelete(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(
		fakeReply{value: `{"content":"alpha one","metadata":{"group":"a"},"embedding":[0.1,0,0,0,0,0,0,0]}`},
		fakeReply{err: redis.Nil}, // missing key
		fakeReply{value: int64(1)},
	)...)

	docs, err := store.GetByIDs(t.Context(), []string{"one", "missing"})
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != "one" || docs[0].PageContent != "alpha one" || docs[0].Metadata["group"] != "a" {
		t.Fatalf("docs: %#v", docs)
	}
	if got := client.call(2); got[0] != "JSON.GET" || got[1] != "docs:one" {
		t.Fatalf("JSON.GET args: %v", got)
	}

	if err := store.Delete(t.Context(), []string{"one"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if got := client.lastCall(); got[0] != "DEL" || got[1] != "docs:one" {
		t.Fatalf("DEL args: %v", got)
	}
}

func searchReplyRESP2(scoreOne, scoreTwo string) fakeReply {
	fields := func(score, content string) []any {
		body, _ := json.Marshal(storedDocument{
			Content:   content,
			Metadata:  map[string]any{"group": "a"},
			Embedding: []float64{0, 0, 0, 0, 0, 0, 0, 0},
		})
		return []any{"__embedding_score", score, "$", string(body)}
	}
	// RESP2 FT.SEARCH reply: [total, id1, fields1, id2, fields2, ...].
	return fakeReply{value: []any{int64(2),
		"docs:one", fields(scoreOne, "alpha one"),
		"docs:two", fields(scoreTwo, "alpha two"),
	}}
}

func TestSimilaritySearchWithScore(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(searchReplyRESP2("0.25", "0.75"))...)

	results, err := store.SimilaritySearchWithScore(t.Context(), "alpha", 2)
	if err != nil {
		t.Fatalf("SimilaritySearchWithScore: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results: got %d want 2", len(results))
	}
	if results[0].Document.ID != "one" || results[0].Document.PageContent != "alpha one" {
		t.Fatalf("first result: %#v", results[0])
	}
	if results[0].Score != 0.25 {
		t.Fatalf("score: got %v want 0.25", results[0].Score)
	}
	if results[0].Document.Metadata["group"] != "a" {
		t.Fatalf("metadata: %#v", results[0].Document.Metadata)
	}

	embedder := embeddings.NewFake(8)
	vec, _ := embedder.EmbedQuery(t.Context(), "alpha")
	want := []any{
		"FT.SEARCH", "idx:docs",
		"*=>[KNN 2 @embedding $BLOB AS __embedding_score]",
		"SORTBY", "__embedding_score", "LIMIT", 0, 2,
		"PARAMS", 2, "BLOB", encodeBlob(vec),
		"DIALECT", 2,
	}
	if got := client.lastCall(); !equalArgs(got, want) {
		t.Fatalf("FT.SEARCH args:\n got %v\nwant %v", got, want)
	}
}

func TestParseSearchReplyRESP3(t *testing.T) {
	// RESP3 envelope as negotiated by go-redis: a map with total_results and
	// results, each result carrying id plus extra_attributes.
	body, _ := json.Marshal(storedDocument{Content: "alpha one", Metadata: map[string]any{}, Embedding: []float64{0, 0, 0, 0, 0, 0, 0, 0}})
	reply := map[any]any{
		"total_results": int64(1),
		"results": []any{map[any]any{
			"id":               "docs:one",
			"extra_attributes": map[any]any{"__embedding_score": "0.5", "$": string(body)},
		}},
	}
	hits, err := parseSearchReply(reply, "docs:")
	if err != nil {
		t.Fatalf("parseSearchReply: %v", err)
	}
	if len(hits) != 1 || hits[0].doc.ID != "one" || hits[0].score != 0.5 {
		t.Fatalf("hits: %#v", hits)
	}
}

func TestBuildFilterQuery(t *testing.T) {
	store := &Store{
		metadataFields: map[string]MetadataFieldType{
			"group": MetadataTag,
			"page":  MetadataNumeric,
			"alive": MetadataTag,
		},
	}

	tests := []struct {
		name   string
		filter map[string]any
		want   string
	}{
		{"empty", nil, ""},
		{"tag eq", map[string]any{"group": "a"}, `@group:{a}`},
		{"tag eq escapes specials", map[string]any{"group": map[string]any{vectorstores.FilterEq: "a b"}}, `@group:{a\ b}`},
		{"tag ne", map[string]any{"group": map[string]any{vectorstores.FilterNe: "a"}}, `-@group:{a} -ismissing(@group)`},
		{"tag in", map[string]any{"group": map[string]any{vectorstores.FilterIn: []any{"a", "b"}}}, `@group:{a|b}`},
		{"tag nin", map[string]any{"group": map[string]any{vectorstores.FilterNin: []any{"a", "b"}}}, `-@group:{a|b} -ismissing(@group)`},
		{"tag exists true", map[string]any{"group": map[string]any{vectorstores.FilterExists: true}}, `-ismissing(@group)`},
		{"tag exists false", map[string]any{"group": map[string]any{vectorstores.FilterExists: false}}, `ismissing(@group)`},
		{"numeric eq", map[string]any{"page": map[string]any{vectorstores.FilterEq: 2}}, `@page:[2 2]`},
		{"numeric eq float", map[string]any{"page": map[string]any{vectorstores.FilterEq: 2.5}}, `@page:[2.5 2.5]`},
		{"numeric ne", map[string]any{"page": map[string]any{vectorstores.FilterNe: 2}}, `-@page:[2 2] -ismissing(@page)`},
		{"numeric gt", map[string]any{"page": map[string]any{vectorstores.FilterGt: 2}}, `@page:[(2 +inf]`},
		{"numeric gte", map[string]any{"page": map[string]any{vectorstores.FilterGte: 2}}, `@page:[2 +inf]`},
		{"numeric lt", map[string]any{"page": map[string]any{vectorstores.FilterLt: 10}}, `@page:[-inf (10]`},
		{"numeric lte", map[string]any{"page": map[string]any{vectorstores.FilterLte: 10}}, `@page:[-inf 10]`},
		{"numeric between", map[string]any{"page": map[string]any{vectorstores.FilterBetween: []any{2, 5}}}, `@page:[2 5]`},
		{"numeric in", map[string]any{"page": map[string]any{vectorstores.FilterIn: []any{2, 5}}}, `(@page:[2 2]|@page:[5 5])`},
		{"numeric nin", map[string]any{"page": map[string]any{vectorstores.FilterNin: []any{2, 5}}}, `-(@page:[2 2]|@page:[5 5]) -ismissing(@page)`},
		{"bool eq on tag", map[string]any{"alive": map[string]any{vectorstores.FilterEq: true}}, `@alive:{true}`},
		{
			"multiple fields and deterministically",
			map[string]any{"group": "a", "page": map[string]any{vectorstores.FilterGt: 2}},
			`(@group:{a}) (@page:[(2 +inf])`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := store.buildFilterQuery(tt.filter)
			if err != nil {
				t.Fatalf("buildFilterQuery: %v", err)
			}
			if got != tt.want {
				t.Fatalf("query:\n got %q\nwant %q", got, tt.want)
			}
		})
	}

	unsupported := []struct {
		name   string
		filter map[string]any
	}{
		{"like is not supported", map[string]any{"group": map[string]any{vectorstores.FilterLike: "%al%"}}},
		{"string range on tag is not supported", map[string]any{"group": map[string]any{vectorstores.FilterGt: "a"}}},
		{"string between on tag is not supported", map[string]any{"group": map[string]any{vectorstores.FilterBetween: []any{"a", "c"}}}},
		{"undeclared field is not supported", map[string]any{"unknown": "a"}},
		{"string value on numeric field", map[string]any{"page": "two"}},
		{"numeric value on tag field", map[string]any{"group": map[string]any{vectorstores.FilterEq: 3}}},
		{"invalid operator", map[string]any{"group": map[string]any{"$bogus": "a"}}},
	}
	for _, tt := range unsupported {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := store.buildFilterQuery(tt.filter); err == nil {
				t.Fatalf("buildFilterQuery(%#v): expected error", tt.filter)
			}
		})
	}
}

func TestSimilaritySearchWithOptions(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(
		searchReplyRESP2("0.1", "0.9"),
		searchReplyRESP2("0.1", "0.9"),
	)...)
	store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}

	docs, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
		K:              4,
		Filter:         map[string]any{"group": "a"},
		ScoreThreshold: 0.5,
	})
	if err != nil {
		t.Fatalf("SimilaritySearchWithOptions: %v", err)
	}
	// distances 0.1 -> relevance 0.9 keeps, 0.9 -> 0.1 dropped
	if len(docs) != 1 || docs[0].ID != "one" {
		t.Fatalf("docs: %#v", docs)
	}

	query, _, err := parseCommandArg[string](client.lastCall()[2])
	if err != nil {
		t.Fatalf("query arg: %v", err)
	}
	wantQuery := "(@group:{a})=>[KNN 4 @embedding $BLOB AS __embedding_score]"
	if query != wantQuery {
		t.Fatalf("query:\n got %q\nwant %q", query, wantQuery)
	}
	if got := slices.Index(client.lastCall(), "__embedding_score"); got < 0 {
		t.Fatalf("missing score alias: %v", client.lastCall())
	}

	if _, err := store.SimilaritySearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{
		Filter: map[string]any{"nope": "a"},
	}); err == nil {
		t.Fatal("expected error for undeclared filter field")
	}
}

func TestMMRSearchWithOptions(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(searchReplyRESP2("0.1", "0.9"))...)

	docs, err := store.MMRSearchWithOptions(t.Context(), "alpha", vectorstores.SearchOptions{K: 2, FetchK: 2})
	if err != nil {
		t.Fatalf("MMRSearchWithOptions: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2", len(docs))
	}

	query, _, _ := parseCommandArg[string](client.lastCall()[2])
	wantQuery := "*=>[KNN 2 @embedding $BLOB AS __embedding_score]"
	if query != wantQuery {
		t.Fatalf("KNN count should use fetch_k:\n got %q\nwant %q", query, wantQuery)
	}
	limit := client.lastCall()[slices.Index(client.lastCall(), "LIMIT")+2]
	if fmt.Sprint(limit) != "2" {
		t.Fatalf("LIMIT should be fetch_k, got %v", limit)
	}
}

func TestDeleteWithFilter(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(
		fakeReply{value: []any{int64(2)}},                         // count query (LIMIT 0 0, NOCONTENT)
		fakeReply{value: []any{int64(2), "docs:one", "docs:two"}}, // id fetch (NOCONTENT, flat ids)
		fakeReply{value: int64(2)},                                // DEL
	)...)
	store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}

	if err := store.DeleteWithFilter(t.Context(), map[string]any{"group": "b"}); err != nil {
		t.Fatalf("DeleteWithFilter: %v", err)
	}
	countCmd := client.call(2)
	if countCmd[0] != "FT.SEARCH" || countCmd[2] != "@group:{b}" {
		t.Fatalf("count search args: %v", countCmd)
	}
	if fmt.Sprint(countCmd[3]) != "NOCONTENT" {
		t.Fatalf("count search should be NOCONTENT: %v", countCmd)
	}
	del := client.lastCall()
	if del[0] != "DEL" || del[1] != "docs:one" || del[2] != "docs:two" {
		t.Fatalf("DEL args: %v", del)
	}

	if err := store.DeleteWithFilter(t.Context(), nil); err == nil {
		t.Fatal("expected error for empty filter")
	}
}

func TestRelevanceScorePerMetric(t *testing.T) {
	cosine := &Store{metric: DistanceCosine}
	if got := cosine.relevanceScore(0.25); got != 0.75 {
		t.Fatalf("cosine relevance: got %v", got)
	}
}

func equalArgs(got, want []any) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if fmt.Sprint(got[i]) != fmt.Sprint(want[i]) {
			return false
		}
	}
	return true
}

// parseCommandArg retrieves a typed argument from a recorded command.
func parseCommandArg[T any](arg any) (T, bool, error) {
	typed, ok := arg.(T)
	return typed, ok, nil
}
