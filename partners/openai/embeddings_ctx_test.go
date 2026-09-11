package openai

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/tiktoken-go/tokenizer"
)

// Covers the len-safe embeddings path mirroring Python
// OpenAIEmbeddings._get_len_safe_embeddings (embeddings/base.py:575-641):
// texts longer than embedding_ctx_length are tokenized
// (embeddings/base.py:552-560), split into ctx-length token chunks, embedded
// per chunk, and merged by token-weighted average + L2 normalization
// (_process_batched_chunked_embeddings, embeddings/base.py:26-83).

// ctxServer fakes /embeddings: it records each request's input entries (as
// decoded JSON values: strings or token-ID arrays) and answers with the next
// scripted per-chunk vectors, one data item per input entry.
type ctxServer struct {
	t        *testing.T
	vectors  [][]float64 // scripted per-chunk vectors, consumed in order
	mu       sync.Mutex
	requests [][]any
}

func newCtxServer(t *testing.T, vectors [][]float64) *ctxServer {
	t.Helper()
	return &ctxServer{t: t, vectors: vectors}
}

func (s *ctxServer) snapshot() [][]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([][]any, len(s.requests))
	copy(out, s.requests)
	return out
}

func (s *ctxServer) handler(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Model string `json:"model"`
		Input []any  `json:"input"`
	}
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.t.Errorf("decode request: %v", err)
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.requests = append(s.requests, request.Input)
	var response [][]float64
	if len(s.vectors) >= len(request.Input) {
		response = s.vectors[:len(request.Input)]
		s.vectors = s.vectors[len(request.Input):]
	}
	s.mu.Unlock()
	payload := embeddingResponsePayload{Object: "list", Model: request.Model}
	if len(response) != len(request.Input) {
		s.t.Errorf("scripted vectors exhausted: %d inputs, %d vectors", len(request.Input), len(response))
	}
	for i := range request.Input {
		vector := []float64{0, 0}
		if i < len(response) {
			vector = response[i]
		}
		payload.Data = append(payload.Data, embeddingDataPayload{
			Object:    "embedding",
			Index:     i,
			Embedding: vector,
		})
	}
	_ = json.NewEncoder(w).Encode(payload)
}

func newCtxTestServer(t *testing.T, vectors [][]float64) (*httptest.Server, *ctxServer) {
	t.Helper()
	fake := newCtxServer(t, vectors)
	server := httptest.NewServer(http.HandlerFunc(fake.handler))
	t.Cleanup(server.Close)
	return server, fake
}

// requireCl100kIDs returns the cl100k_base token IDs for text.
func requireCl100kIDs(t *testing.T, text string) []uint {
	t.Helper()
	codec, err := getCodec(tokenizer.Cl100kBase)
	if err != nil {
		t.Fatalf("load cl100k codec: %v", err)
	}
	ids, _, err := codec.Encode(text)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return ids
}

// weightedNormalized mirrors Python's merge math (embeddings/base.py:68-81)
// for use as an independent expectation in tests.
func weightedNormalized(vectors [][]float64, weights []int) []float64 {
	total := 0
	for _, w := range weights {
		total += w
	}
	average := make([]float64, len(vectors[0]))
	for d := range average {
		for k, v := range vectors {
			average[d] += v[d] * float64(weights[k])
		}
		average[d] /= float64(total)
	}
	magnitude := 0.0
	for _, val := range average {
		magnitude += val * val
	}
	magnitude = math.Sqrt(magnitude)
	out := make([]float64, len(average))
	for d, val := range average {
		out[d] = val / magnitude
	}
	return out
}

func floatSliceEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-9 {
			return false
		}
	}
	return true
}

// intify converts a decoded token-array entry to ints.
func intify(t *testing.T, entry any) []int {
	t.Helper()
	raw, ok := entry.([]any)
	if !ok {
		t.Fatalf("entry %#v is not a token array", entry)
	}
	out := make([]int, len(raw))
	for i, v := range raw {
		f, ok := v.(float64)
		if !ok {
			t.Fatalf("token %#v is not a number", v)
		}
		out[i] = int(f)
	}
	return out
}

// A text over the default 8191-token ctx splits into token-ID chunks
// (embeddings/base.py:557-560); the result is the token-weighted, normalized
// average of the per-chunk embeddings (embeddings/base.py:65-81).
func TestEmbeddingsSplitLongTextIntoTokenChunks(t *testing.T) {
	text := strings.Repeat(" a", 20000) // cl100k: exactly 20000 tokens
	ids := requireCl100kIDs(t, text)
	chunkVectors := [][]float64{{2, 0}, {0, 2}, {4, 0}}
	server, fake := newCtxTestServer(t, chunkVectors)

	model := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	vectors, err := model.EmbedDocuments(context.Background(), []string{text})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}

	requests := fake.snapshot()
	if len(requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(requests))
	}
	input := requests[0]
	if len(input) != 3 {
		t.Fatalf("inputs = %d, want 3 chunks", len(input))
	}
	wantLens := []int{8191, 8191, 3618}
	for i, want := range wantLens {
		got := intify(t, input[i])
		if len(got) != want {
			t.Fatalf("chunk %d tokens = %d, want %d", i, len(got), want)
		}
	}
	// The chunk token IDs must be the exact slices of the encoded text.
	for i, want := range [][]uint{ids[:8191], ids[8191 : 2*8191], ids[2*8191:]} {
		got := intify(t, input[i])
		for j := range want {
			if got[j] != int(want[j]) {
				t.Fatalf("chunk %d token %d = %d, want %d", i, j, got[j], want[j])
			}
		}
	}

	if len(vectors) != 1 {
		t.Fatalf("vectors = %d, want 1", len(vectors))
	}
	want := weightedNormalized(chunkVectors, []int{8191, 8191, 3618})
	if !floatSliceEqual(vectors[0], want) {
		t.Fatalf("merged vector = %v, want %v", vectors[0], want)
	}
}

// Exactly embedding_ctx_length tokens is a single chunk: no split, the raw
// string is sent, and the embedding is returned untouched (single-chunk
// passthrough, embeddings/base.py:60-63).
func TestEmbeddingsBoundaryExactlyCtxLengthNotSplit(t *testing.T) {
	text := strings.Repeat(" a", 8191) // exactly 8191 tokens
	server, fake := newCtxTestServer(t, [][]float64{{3, 4}})

	model := NewEmbeddings(modelconfig.WithBaseURL(server.URL))
	vectors, err := model.EmbedDocuments(context.Background(), []string{text})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}

	requests := fake.snapshot()
	if len(requests) != 1 || len(requests[0]) != 1 {
		t.Fatalf("requests = %v, want a single single-input request", requests)
	}
	if entry, ok := requests[0][0].(string); !ok || entry != text {
		t.Fatalf("input = %#v, want the original string", requests[0][0])
	}
	if !floatSliceEqual(vectors[0], []float64{3, 4}) {
		t.Fatalf("vector = %v, want [3 4] untouched (no normalization)", vectors[0])
	}
}

// WithEmbeddingCtxLength overrides the chunk size; 0 falls back to the 8191
// default (embeddings/base.py:232).
func TestEmbeddingsCtxLengthOption(t *testing.T) {
	text := strings.Repeat(" a", 250) // 250 tokens
	chunkVectors := [][]float64{{1, 0}, {0, 1}, {1, 1}}
	server, fake := newCtxTestServer(t, chunkVectors)

	model := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		WithEmbeddingCtxLength(100),
	)
	vectors, err := model.EmbedDocuments(context.Background(), []string{text})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	requests := fake.snapshot()
	if len(requests) != 1 || len(requests[0]) != 3 {
		t.Fatalf("requests = %v, want one request with 3 chunks", requests)
	}
	for i, want := range []int{100, 100, 50} {
		if got := intify(t, requests[0][i]); len(got) != want {
			t.Fatalf("chunk %d tokens = %d, want %d", i, len(got), want)
		}
	}
	want := weightedNormalized(chunkVectors, []int{100, 100, 50})
	if !floatSliceEqual(vectors[0], want) {
		t.Fatalf("merged vector = %v, want %v", vectors[0], want)
	}

	// 0 keeps the 8191 default: the 250-token text stays a single string.
	server2, fake2 := newCtxTestServer(t, [][]float64{{9, 9}})
	model2 := NewEmbeddings(
		modelconfig.WithBaseURL(server2.URL),
		WithEmbeddingCtxLength(0),
	)
	vectors2, err := model2.EmbedDocuments(context.Background(), []string{text})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	requests2 := fake2.snapshot()
	if len(requests2) != 1 || len(requests2[0]) != 1 {
		t.Fatalf("requests = %v, want one single-input request", requests2)
	}
	if entry, ok := requests2[0][0].(string); !ok || entry != text {
		t.Fatalf("input = %#v, want the original string", requests2[0][0])
	}
	if !floatSliceEqual(vectors2[0], []float64{9, 9}) {
		t.Fatalf("vector = %v, want [9 9]", vectors2[0])
	}
}

// A batch mixing an over-limit text's token chunks with short texts is sent
// as a homogeneous token-array request (Python always sends token arrays when
// checking ctx length, embeddings/base.py:619); a batch with only short texts
// keeps raw strings. Results map back to the right texts.
func TestEmbeddingsMixedBatchUsesTokenArrays(t *testing.T) {
	long := strings.Repeat(" a", 150) // 150 tokens -> 2 chunks at ctx 100
	server, fake := newCtxTestServer(t, [][]float64{
		{5, 0}, // "alpha" (single chunk, returned as-is)
		{2, 0}, // long chunk 1
		{0, 2}, // long chunk 2
		{7, 0}, // "omega" (single chunk)
	})

	model := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		WithEmbeddingCtxLength(100),
		WithEmbeddingChunkSize(3),
	)
	vectors, err := model.EmbedDocuments(context.Background(), []string{"alpha", long, "omega"})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}

	requests := fake.snapshot()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2 (chunk_size 3 over 4 chunks)", len(requests))
	}
	// Batch 1: alpha's 1 token + long's 100 + 50 — all token arrays.
	if len(requests[0]) != 3 {
		t.Fatalf("batch 1 inputs = %d, want 3", len(requests[0]))
	}
	for i, want := range []int{1, 100, 50} {
		if got := intify(t, requests[0][i]); len(got) != want {
			t.Fatalf("batch 1 chunk %d tokens = %d, want %d", i, len(got), want)
		}
	}
	// Batch 2: only "omega", sent as the raw string.
	if len(requests[1]) != 1 {
		t.Fatalf("batch 2 inputs = %d, want 1", len(requests[1]))
	}
	if entry, ok := requests[1][0].(string); !ok || entry != "omega" {
		t.Fatalf("batch 2 input = %#v, want \"omega\"", requests[1][0])
	}

	if len(vectors) != 3 {
		t.Fatalf("vectors = %d, want 3", len(vectors))
	}
	if !floatSliceEqual(vectors[0], []float64{5, 0}) {
		t.Fatalf("alpha vector = %v, want [5 0] (single chunk as-is)", vectors[0])
	}
	if !floatSliceEqual(vectors[2], []float64{7, 0}) {
		t.Fatalf("omega vector = %v, want [7 0] (single chunk as-is)", vectors[2])
	}
	wantLong := weightedNormalized([][]float64{{2, 0}, {0, 2}}, []int{100, 50})
	if !floatSliceEqual(vectors[1], wantLong) {
		t.Fatalf("long vector = %v, want %v", vectors[1], wantLong)
	}
}

// EmbedQuery goes through the same check (Python embed_query delegates to
// embed_documents([text])[0], embeddings/base.py:794-805).
func TestEmbeddingsQuerySplitsLongText(t *testing.T) {
	text := strings.Repeat(" a", 250)
	chunkVectors := [][]float64{{3, 0}, {0, 3}}
	server, fake := newCtxTestServer(t, chunkVectors)

	model := NewEmbeddings(
		modelconfig.WithBaseURL(server.URL),
		WithEmbeddingCtxLength(200),
	)
	vector, err := model.EmbedQuery(context.Background(), text)
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	requests := fake.snapshot()
	if len(requests) != 1 || len(requests[0]) != 2 {
		t.Fatalf("requests = %v, want one request with 2 chunks", requests)
	}
	want := weightedNormalized(chunkVectors, []int{200, 50})
	if !floatSliceEqual(vector, want) {
		t.Fatalf("query vector = %v, want %v", vector, want)
	}
	if norm := math.Sqrt(vector[0]*vector[0] + vector[1]*vector[1]); math.Abs(norm-1) > 1e-9 {
		t.Fatalf("merged vector norm = %v, want 1", norm)
	}
}

// mergeChunkedEmbeddings unit tests: token-weighted average then L2
// normalization (embeddings/base.py:68-81); a single chunk is returned
// unchanged (embeddings/base.py:60-63).
func TestMergeChunkedEmbeddings(t *testing.T) {
	single := mergeChunkedEmbeddings([][]float64{{3, 4}}, []int{17})
	if !floatSliceEqual(single, []float64{3, 4}) {
		t.Fatalf("single chunk = %v, want [3 4] untouched", single)
	}

	vectors := [][]float64{{3, 0}, {0, 4}}
	weights := []int{1, 3}
	got := mergeChunkedEmbeddings(vectors, weights)
	// average = [3*1/4, 4*3/4] = [0.75, 3]; magnitude = sqrt(9.5625).
	average := []float64{0.75, 3}
	magnitude := math.Sqrt(0.75*0.75 + 3*3)
	want := []float64{average[0] / magnitude, average[1] / magnitude}
	if !floatSliceEqual(got, want) {
		t.Fatalf("merge = %v, want %v", got, want)
	}
}

// planEmbeddingBatches unit tests mirroring the batching loop of
// _get_len_safe_embeddings (embeddings/base.py:604-627): at most chunkSize
// chunks AND at most 300000 tokens per request
// (MAX_TOKENS_PER_REQUEST, embeddings/base.py:22); a single over-limit chunk
// is still sent on its own (embeddings/base.py:613-615).
func TestPlanEmbeddingBatches(t *testing.T) {
	cases := []struct {
		name       string
		counts     []int
		chunkSize  int
		wantRanges [][2]int
	}{
		{"chunk size cap", []int{1, 1, 1}, 2, [][2]int{{0, 2}, {2, 3}}},
		{"fits in one", []int{5, 5}, 1000, [][2]int{{0, 2}}},
		{"token cap splits", []int{200000, 200000}, 1000, [][2]int{{0, 1}, {1, 2}}},
		{"token cap boundary inclusive", []int{150000, 150000, 150000}, 10, [][2]int{{0, 2}, {2, 3}}},
		{"single chunk over token cap sent anyway", []int{300001}, 1000, [][2]int{{0, 1}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := planEmbeddingBatches(tc.counts, tc.chunkSize)
			if len(got) != len(tc.wantRanges) {
				t.Fatalf("ranges = %v, want %v", got, tc.wantRanges)
			}
			for i, want := range tc.wantRanges {
				if got[i] != want {
					t.Fatalf("range %d = %v, want %v", i, got[i], want)
				}
			}
		})
	}
}
