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
	ip := &Store{metric: DistanceIP}
	if got := ip.relevanceScore(-1.5); got != 1.5 {
		t.Fatalf("ip relevance: got %v", got)
	}
	l2 := &Store{metric: DistanceL2}
	if got := l2.relevanceScore(0); got != 1 {
		t.Fatalf("l2 relevance: got %v", got)
	}
}

func TestRediSearchMetricMapping(t *testing.T) {
	for metric, want := range map[DistanceMetric]string{
		DistanceCosine: "COSINE",
		DistanceL2:     "L2",
		DistanceIP:     "IP",
	} {
		if got := metric.rediSearchMetric(); got != want {
			t.Fatalf("metric %s: got %q want %q", metric, got, want)
		}
	}
}

// closableFakeClient extends fakeClient with a Close method so Close
// ownership can be asserted.
type closableFakeClient struct {
	fakeClient
	closed bool
}

func (c *closableFakeClient) Close() error {
	c.closed = true
	return nil
}

func TestNewConnectionOptions(t *testing.T) {
	store := &Store{}
	WithPassword("secret")(store)
	WithUsername("alice")(store)
	WithDB(3)(store)
	WithDimension(16)(store)
	if store.password != "secret" || store.username != "alice" || store.db != 3 || store.dimension != 16 {
		t.Fatalf(
			"connection options: password=%q username=%q db=%d dimension=%d",
			store.password, store.username, store.db, store.dimension,
		)
	}
}

func TestNewRejectsBlankAddressWithoutClient(t *testing.T) {
	_, err := New(t.Context(), "docs",
		WithAddr("  "),
		WithEmbedder(embeddings.NewFake(4)),
	)
	if err == nil {
		t.Fatal("expected error for blank address without an injected client")
	}
}

func TestStoreClose(t *testing.T) {
	t.Run("owned_client_is_closed", func(t *testing.T) {
		client := &closableFakeClient{}
		store := &Store{client: client, ownsClient: true}
		store.Close()
		if !client.closed {
			t.Fatal("owned client must be closed")
		}
	})

	t.Run("injected_client_stays_open", func(t *testing.T) {
		client := &closableFakeClient{}
		store := &Store{client: client}
		store.Close()
		if client.closed {
			t.Fatal("injected client must not be closed")
		}
	})
}

func TestAddTextsStoresDocuments(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(
		fakeReply{value: "OK"},
		fakeReply{value: "OK"},
	)...)

	// A metadatas slice shorter than texts exercises metadataAt's out-of-range
	// branch, and the missing id exercises newID.
	ids, err := store.AddTexts(t.Context(),
		[]string{"alpha one", "alpha two"},
		[]map[string]any{{"group": "a"}},
		[]string{"one"},
	)
	if err != nil {
		t.Fatalf("AddTexts: %v", err)
	}
	if len(ids) != 2 || ids[0] != "one" || ids[1] == "" {
		t.Fatalf("ids: %v", ids)
	}

	first := client.call(2)
	if first[0] != "JSON.SET" || first[1] != "docs:one" {
		t.Fatalf("first JSON.SET args: %v", first)
	}
	var payload storedDocument
	if err := json.Unmarshal([]byte(client.call(2)[3].(string)), &payload); err != nil {
		t.Fatalf("stored json: %v", err)
	}
	if payload.Metadata["group"] != "a" {
		t.Fatalf("metadata: %#v", payload.Metadata)
	}
	second := client.call(3)
	if second[0] != "JSON.SET" || second[1] != "docs:"+ids[1] {
		t.Fatalf("second JSON.SET args: %v", second)
	}
	var payloadTwo storedDocument
	if err := json.Unmarshal([]byte(second[3].(string)), &payloadTwo); err != nil {
		t.Fatalf("stored json two: %v", err)
	}
	if payloadTwo.Metadata == nil {
		t.Fatal("metadata should be an object, got nil")
	}
}

// nanEmbedder yields NaN vectors so the JSON encode path fails (json.Marshal
// rejects NaN).
type nanEmbedder struct{}

func (nanEmbedder) EmbedDocuments(context.Context, []string) ([][]float64, error) {
	return [][]float64{{math.NaN()}}, nil
}

func (nanEmbedder) EmbedQuery(context.Context, string) ([]float64, error) {
	return []float64{math.NaN()}, nil
}

// shortEmbedder returns fewer vectors than input texts to drive the
// AddDocuments count-mismatch guard.
type shortEmbedder struct{}

func (shortEmbedder) EmbedDocuments(context.Context, []string) ([][]float64, error) {
	return [][]float64{{0.1}}, nil
}

func (shortEmbedder) EmbedQuery(context.Context, string) ([]float64, error) {
	return []float64{0.1}, nil
}

func TestAddDocumentsFailures(t *testing.T) {
	t.Run("no_documents", func(t *testing.T) {
		store := &Store{}
		ids, err := store.AddDocuments(t.Context(), nil)
		if err != nil || ids != nil {
			t.Fatalf("AddDocuments with no docs: ids=%v err=%v", ids, err)
		}
	})

	t.Run("missing_embedder", func(t *testing.T) {
		store := &Store{}
		if _, err := store.AddDocuments(t.Context(), []documents.Document{documents.New("a", nil)}); err == nil {
			t.Fatal("expected error for missing embedder")
		}
	})

	t.Run("embedding_failure", func(t *testing.T) {
		store, _ := newFakeStore(t)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := store.AddDocuments(ctx, []documents.Document{documents.New("a", nil)}); err == nil {
			t.Fatal("expected embedding failure")
		}
	})

	t.Run("embedding_count_mismatch", func(t *testing.T) {
		store := &Store{embedder: shortEmbedder{}}
		if _, err := store.AddDocuments(t.Context(), []documents.Document{
			documents.New("a", nil), documents.New("b", nil),
		}); err == nil {
			t.Fatal("expected embedding count mismatch error")
		}
	})

	t.Run("encode_failure", func(t *testing.T) {
		store := &Store{
			client:    &fakeClient{},
			embedder:  nanEmbedder{},
			dimension: 1,
		}
		if _, err := store.AddDocuments(t.Context(), []documents.Document{
			documents.New("a", nil).WithID("one"),
		}); err == nil {
			t.Fatal("expected JSON encode failure for NaN embedding")
		}
	})

	t.Run("store_failure", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(fakeReply{err: errors.New("redis is full")})...)
		if _, err := store.AddDocuments(t.Context(), []documents.Document{
			documents.New("a", nil).WithID("one"),
		}); err == nil {
			t.Fatal("expected JSON.SET failure")
		}
	})
}

func TestDeleteWithNoIDs(t *testing.T) {
	store, client := newFakeStore(t, indexAbsentReply...)
	if err := store.Delete(t.Context(), nil); err != nil {
		t.Fatalf("Delete with no ids: %v", err)
	}
	if client.callCount() != 2 { // FT.INFO, FT.CREATE from New only
		t.Fatalf("DEL must not run, got %d calls", client.callCount())
	}
}

func TestGetByIDsEdgeCases(t *testing.T) {
	t.Run("no_ids", func(t *testing.T) {
		store, _ := newFakeStore(t)
		docs, err := store.GetByIDs(t.Context(), nil)
		if err != nil || docs != nil {
			t.Fatalf("GetByIDs with no ids: docs=%v err=%v", docs, err)
		}
	})

	t.Run("non_nil_error_aborts", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(fakeReply{err: errors.New("connection lost")})...)
		if _, err := store.GetByIDs(t.Context(), []string{"one"}); err == nil {
			t.Fatal("expected non-nil error to abort")
		}
	})

	t.Run("non_string_reply", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(fakeReply{value: int64(1)})...)
		if _, err := store.GetByIDs(t.Context(), []string{"one"}); err == nil {
			t.Fatal("expected text conversion failure")
		}
	})

	t.Run("malformed_document_json", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(fakeReply{value: "not-json"})...)
		if _, err := store.GetByIDs(t.Context(), []string{"one"}); err == nil {
			t.Fatal("expected document decode failure")
		}
	})
}

func TestSimilaritySearchReturnsDocuments(t *testing.T) {
	store, _ := newFakeStore(t, withDefaultSchema(searchReplyRESP2("0.25", "0.75"))...)

	docs, err := store.SimilaritySearch(t.Context(), "alpha", 2)
	if err != nil {
		t.Fatalf("SimilaritySearch: %v", err)
	}
	if len(docs) != 2 || docs[0].ID != "one" || docs[0].PageContent != "alpha one" {
		t.Fatalf("docs: %#v", docs)
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := store.SimilaritySearch(ctx, "alpha", 2); err == nil {
		t.Fatal("expected search failure to propagate")
	}
}

// A zero K falls back to the langchain default of 4.
func TestSimilaritySearchWithScoreDefaultsKToFour(t *testing.T) {
	store, client := newFakeStore(t, withDefaultSchema(searchReplyRESP2("0.25", "0.75"))...)

	if _, err := store.SimilaritySearchWithScore(t.Context(), "alpha", 0); err != nil {
		t.Fatalf("SimilaritySearchWithScore: %v", err)
	}
	query := client.lastCall()[2].(string)
	if query != "*=>[KNN 4 @embedding $BLOB AS __embedding_score]" {
		t.Fatalf("KNN expression for zero k: got %q", query)
	}
}

func TestKnnSearchByVectorFailures(t *testing.T) {
	t.Run("filter_rendering_failure", func(t *testing.T) {
		store, _ := newFakeStore(t)
		store.metadataFields = map[string]MetadataFieldType{"group": MetadataTag}
		if _, err := store.knnSearchByVector(t.Context(), []float64{0.1},
			map[string]any{"undeclared": "a"}, 4); err == nil {
			t.Fatal("expected filter rendering failure")
		}
	})

	t.Run("search_error", func(t *testing.T) {
		store, _ := newFakeStore(t, withDefaultSchema(fakeReply{err: errors.New("search blew up")})...)
		if _, err := store.knnSearchByVector(t.Context(), make([]float64, 8), nil, 4); err == nil {
			t.Fatal("expected search failure")
		}
	})
}

func TestEmbedQueryFailures(t *testing.T) {
	t.Run("missing_embedder", func(t *testing.T) {
		store := &Store{}
		if _, err := store.embedQuery(t.Context(), "alpha"); err == nil {
			t.Fatal("expected error for missing embedder")
		}
	})

	t.Run("dimension_mismatch", func(t *testing.T) {
		client := &fakeClient{replies: indexAbsentReply}
		store, err := New(t.Context(), "docs",
			WithClient(client),
			WithEmbedder(embeddings.NewFake(8)),
			WithDimension(4),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, err := store.SimilaritySearchWithScore(t.Context(), "alpha", 2); err == nil {
			t.Fatal("expected query embedding dimension mismatch")
		}
	})
}

func TestParseSearchReplyEdgeCases(t *testing.T) {
	validBody, _ := json.Marshal(storedDocument{Content: "alpha", Metadata: map[string]any{}, Embedding: []float64{0, 0, 0, 0, 0, 0, 0, 0}})

	t.Run("empty_id_is_skipped", func(t *testing.T) {
		hits, err := parseSearchReply(map[any]any{
			"results": []any{map[any]any{"id": ""}},
		}, "docs:")
		if err != nil || len(hits) != 0 {
			t.Fatalf("hits: %v err %v", hits, err)
		}
	})

	t.Run("malformed_document_body", func(t *testing.T) {
		if _, err := parseSearchReply(map[any]any{
			"results": []any{map[any]any{
				"id":               "docs:one",
				"extra_attributes": map[any]any{"$": "not-json"},
			}},
		}, "docs:"); err == nil {
			t.Fatal("expected document decode failure")
		}
	})

	t.Run("malformed_score", func(t *testing.T) {
		if _, err := parseSearchReply(map[any]any{
			"results": []any{map[any]any{
				"id":               "docs:one",
				"extra_attributes": map[any]any{scoreAlias: "fast"},
			}},
		}, "docs:"); err == nil {
			t.Fatal("expected score parse failure")
		}
	})

	t.Run("resp3_without_results_key", func(t *testing.T) {
		hits, err := parseSearchReply(map[any]any{"total_results": int64(0)}, "docs:")
		if err != nil || hits != nil {
			t.Fatalf("hits: %v err %v", hits, err)
		}
	})

	t.Run("resp3_non_map_result_entry", func(t *testing.T) {
		hits, err := parseSearchReply(map[any]any{
			"results": []any{int64(42), map[any]any{
				"id":     "docs:one",
				"values": []any{scoreAlias, "0.5", "$", string(validBody)},
			}},
		}, "docs:")
		if err != nil {
			t.Fatalf("parseSearchReply: %v", err)
		}
		if len(hits) != 1 || hits[0].doc.ID != "one" || hits[0].score != 0.5 || hits[0].doc.PageContent != "alpha" {
			t.Fatalf("hits: %#v", hits)
		}
	})

	t.Run("resp2_array_too_short", func(t *testing.T) {
		hits, err := parseSearchReply([]any{int64(1)}, "docs:")
		if err != nil || hits != nil {
			t.Fatalf("hits: %v err %v", hits, err)
		}
	})

	t.Run("resp2_map_form", func(t *testing.T) {
		hits, err := parseSearchReply([]any{int64(1), map[any]any{
			"docs:one": map[any]any{scoreAlias: "0.5", "$": string(validBody)},
			int64(42):  map[any]any{scoreAlias: "0.9"},
		}}, "docs:")
		if err != nil {
			t.Fatalf("parseSearchReply: %v", err)
		}
		if len(hits) != 1 || hits[0].doc.ID != "one" || hits[0].score != 0.5 {
			t.Fatalf("hits: %#v", hits)
		}
	})

	t.Run("resp2_map_form_decode_failure", func(t *testing.T) {
		if _, err := parseSearchReply([]any{int64(1), map[any]any{
			"docs:one": map[any]any{"$": "not-json"},
		}}, "docs:"); err == nil {
			t.Fatal("expected decode failure")
		}
	})

	t.Run("resp2_flat_with_non_string_id", func(t *testing.T) {
		hits, err := parseSearchReply([]any{int64(2),
			int64(42),
			"docs:one", []any{scoreAlias, "0.5", "$", string(validBody)},
		}, "docs:")
		if err != nil {
			t.Fatalf("parseSearchReply: %v", err)
		}
		if len(hits) != 1 || hits[0].doc.ID != "one" {
			t.Fatalf("hits: %#v", hits)
		}
	})

	t.Run("resp2_flat_id_without_fields", func(t *testing.T) {
		hits, err := parseSearchReply([]any{int64(2), "docs:one"}, "docs:")
		if err != nil {
			t.Fatalf("parseSearchReply: %v", err)
		}
		if len(hits) != 1 || hits[0].doc.ID != "one" || hits[0].doc.PageContent != "" {
			t.Fatalf("hits: %#v", hits)
		}
	})

	t.Run("resp2_flat_id_with_non_array_fields", func(t *testing.T) {
		hits, err := parseSearchReply([]any{int64(1), "docs:one", int64(7)}, "docs:")
		if err != nil {
			t.Fatalf("parseSearchReply: %v", err)
		}
		if len(hits) != 1 || hits[0].doc.ID != "one" {
			t.Fatalf("hits: %#v", hits)
		}
	})

	t.Run("unexpected_reply_type", func(t *testing.T) {
		hits, err := parseSearchReply("OK", "docs:")
		if err != nil || hits != nil {
			t.Fatalf("hits: %v err %v", hits, err)
		}
	})
}

func TestMetadataHelpers(t *testing.T) {
	if got := cloneMetadata(nil); got == nil || len(got) != 0 {
		t.Fatalf("cloneMetadata(nil): got %#v want empty map", got)
	}
	original := map[string]any{"group": "a"}
	cloned := cloneMetadata(original)
	if cloned["group"] != "a" {
		t.Fatalf("cloneMetadata: got %#v", cloned)
	}
	cloned["group"] = "b"
	if original["group"] != "a" {
		t.Fatal("cloneMetadata must deep-copy the map")
	}

	metadatas := []map[string]any{{"group": "a"}}
	if got := metadataAt(metadatas, 0); got == nil || got["group"] != "a" {
		t.Fatalf("metadataAt in range: got %v", got)
	}
	if got := metadataAt(metadatas, 1); got != nil {
		t.Fatalf("metadataAt out of range: got %v", got)
	}

	if id := newID(); len(id) != 32 {
		t.Fatalf("newID: got %q want 32 hex chars", id)
	}
}

func TestIsValidAttributeName(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  bool
	}{
		{"lowercase", "group", true},
		{"mixed_with_digits", "Page_2", true},
		{"empty", "", false},
		{"contains_space", "bad field", false},
		{"contains_special", "bad!", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isValidAttributeName(tt.value); got != tt.want {
				t.Fatalf("isValidAttributeName(%q): got %v want %v", tt.value, got, tt.want)
			}
		})
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
