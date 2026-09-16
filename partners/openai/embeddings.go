package openai

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/tiktoken-go/tokenizer"
)

// Embeddings adapts LangChain embedding calls to OpenAI's embeddings endpoint.
type Embeddings struct {
	config modelconfig.Config
}

const (
	embeddingDimensionsKey     = "openai_embedding_dimensions"
	embeddingEncodingFormatKey = "openai_embedding_encoding_format"
)

// WithEmbeddingDimensions sets the optional OpenAI embeddings dimensions
// request parameter.
func WithEmbeddingDimensions(dimensions int) modelconfig.Option {
	return modelconfig.WithExtra(embeddingDimensionsKey, dimensions)
}

// WithEmbeddingEncodingFormat sets the optional OpenAI embeddings
// encoding_format request parameter.
func WithEmbeddingEncodingFormat(format string) modelconfig.Option {
	return modelconfig.WithExtra(embeddingEncodingFormatKey, format)
}

const embeddingChunkSizeKey = "openai_embedding_chunk_size"

// WithEmbeddingChunkSize sets the maximum number of texts embedded per API
// request, mirroring Python OpenAIEmbeddings(chunk_size=...) (default 1000,
// embeddings/base.py:260). Non-positive values fall back to the default.
func WithEmbeddingChunkSize(size int) modelconfig.Option {
	return modelconfig.WithExtra(embeddingChunkSizeKey, size)
}

// chunkSize returns the configured batch size (default 1000).
func (e Embeddings) chunkSize() int {
	if size, ok := e.config.Extra[embeddingChunkSizeKey].(int); ok && size > 0 {
		return size
	}
	return 1000
}

const embeddingCtxLengthKey = "openai_embedding_ctx_length"

// defaultEmbeddingCtxLength mirrors Python OpenAIEmbeddings'
// embedding_ctx_length (embeddings/base.py:232).
const defaultEmbeddingCtxLength = 8191

// WithEmbeddingCtxLength sets the maximum number of tokens embedded at once;
// longer texts are split into token chunks and merged, mirroring Python
// OpenAIEmbeddings(embedding_ctx_length=...) (embeddings/base.py:232, default
// 8191). 0 and negative values fall back to the default.
func WithEmbeddingCtxLength(length int) modelconfig.Option {
	return modelconfig.WithExtra(embeddingCtxLengthKey, length)
}

// embeddingCtxLength returns the configured per-request token budget
// (default 8191).
func (e Embeddings) embeddingCtxLength() int {
	if length, ok := e.config.Extra[embeddingCtxLengthKey].(int); ok && length > 0 {
		return length
	}
	return defaultEmbeddingCtxLength
}

// maxTokensPerEmbeddingRequest mirrors Python MAX_TOKENS_PER_REQUEST
// (embeddings/base.py:22): the total token budget per embeddings request when
// batching token chunks.
const maxTokensPerEmbeddingRequest = 300000

// NewEmbeddings creates an OpenAI embeddings adapter.
func NewEmbeddings(opts ...modelconfig.Option) Embeddings {
	cfg := modelconfig.New(opts...)
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.Model == "" {
		cfg.Model = "text-embedding-3-small"
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	return Embeddings{config: cfg}
}

// embeddingChunk is one embeddable slice of an input text. Texts within the
// context budget stay whole strings (asTokens=false); over-budget texts are
// split into embedding_ctx_length token chunks (asTokens=true), mirroring
// Python _tokenize (embeddings/base.py:552-560).
type embeddingChunk struct {
	textIndex int
	// ids are the chunk's token IDs; for whole-text chunks they are the
	// full text's tokens and provide the merge weight (len(ids), Python's
	// num_tokens_in_batch, embeddings/base.py:46).
	ids []uint
	// asTokens reports whether the chunk must be sent as a token-ID array.
	asTokens bool
	// text is the original text for whole-text chunks (valid when !asTokens).
	text string
}

// planEmbeddingBatches returns [start,end) ranges over token chunks so each
// request carries at most chunkSize chunks and at most
// maxTokensPerEmbeddingRequest tokens, mirroring Python
// _get_len_safe_embeddings' batching loop (embeddings/base.py:604-627). A
// single chunk whose token count alone exceeds the budget is still planned as
// its own batch (embeddings/base.py:613-615, "Single chunk exceeds limit -
// handle it anyway").
func planEmbeddingBatches(tokenCounts []int, chunkSize int) [][2]int {
	var ranges [][2]int
	for i := 0; i < len(tokenCounts); {
		batchTokenCount := 0
		batchEnd := i
		limit := i + chunkSize
		if limit > len(tokenCounts) {
			limit = len(tokenCounts)
		}
		for j := i; j < limit; j++ {
			if batchTokenCount+tokenCounts[j] > maxTokensPerEmbeddingRequest {
				if batchEnd == i {
					batchEnd = j + 1
				}
				break
			}
			batchTokenCount += tokenCounts[j]
			batchEnd = j + 1
		}
		ranges = append(ranges, [2]int{i, batchEnd})
		i = batchEnd
	}
	return ranges
}

// mergeChunkedEmbeddings combines one text's per-chunk embeddings into its
// final vector, mirroring Python _process_batched_chunked_embeddings
// (embeddings/base.py:26-83): a single chunk is returned unchanged (:60-63),
// otherwise the chunks are averaged weighted by their token counts (:68-76)
// and the average is L2-normalized (:78-81). A zero magnitude leaves the
// (degenerate all-zero) average as-is instead of dividing by zero.
func mergeChunkedEmbeddings(vectors [][]float64, weights []int) []float64 {
	if len(vectors) == 0 {
		return nil
	}
	if len(vectors) == 1 {
		return append([]float64(nil), vectors[0]...)
	}
	dims := len(vectors[0])
	for _, v := range vectors[1:] {
		if len(v) < dims {
			// Python's zip(*_result) silently truncates to the shortest
			// embedding (embeddings/base.py:75).
			dims = len(v)
		}
	}
	totalWeight := 0
	for _, w := range weights {
		totalWeight += w
	}
	if totalWeight == 0 {
		totalWeight = 1
	}
	average := make([]float64, dims)
	for d := range dims {
		for k, v := range vectors {
			average[d] += v[d] * float64(weights[k])
		}
		average[d] /= float64(totalWeight)
	}
	magnitude := 0.0
	for _, val := range average {
		magnitude += val * val
	}
	magnitude = math.Sqrt(magnitude)
	if magnitude == 0 {
		return average
	}
	out := make([]float64, dims)
	for d, val := range average {
		out[d] = val / magnitude
	}
	return out
}

// EmbedDocuments embeds all documents. Each text longer than the embedding
// context budget (default 8191 tokens, WithEmbeddingCtxLength) is tokenized,
// split into budget-sized token chunks, embedded chunk by chunk, and merged
// by token-weighted average + L2 normalization, mirroring Python
// OpenAIEmbeddings.embed_documents -> _get_len_safe_embeddings
// (embeddings/base.py:575-641). Whole-text chunks are batched exactly like
// the previous string-only implementation (chunkSize texts per request), so
// the short-text wire behavior is unchanged.
func (e Embeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	post := func(ctx context.Context, payload any) (embeddingResponsePayload, error) {
		return e.createEmbeddings(ctx, payload)
	}
	return e.lenSafeEmbedDocuments(ctx, texts, post)
}

// EmbedQuery embeds a single query. Like Python's embed_query
// (embeddings/base.py:794-805, a thin delegate to embed_documents([text])[0])
// it goes through the same context-length-safe path, so over-budget queries
// are chunked and merged.
func (e Embeddings) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	vectors, err := e.EmbedDocuments(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}
	return vectors[0], nil
}

// lenSafeEmbedDocuments is the shared len-safe embedding pipeline used by
// Embeddings and AzureEmbeddings (Python's AzureOpenAIEmbeddings inherits the
// same embed_documents). post sends one embeddings request.
func (e Embeddings) lenSafeEmbedDocuments(
	ctx context.Context,
	texts []string,
	post func(ctx context.Context, payload any) (embeddingResponsePayload, error),
) ([][]float64, error) {
	codec, err := e.tokenizer()
	if err != nil {
		return nil, err
	}
	ctxLength := e.embeddingCtxLength()

	// Tokenize and split (Python _tokenize, embeddings/base.py:552-568):
	// whole texts stay strings, over-budget texts become token chunks.
	chunks := make([]embeddingChunk, 0, len(texts))
	for i, text := range texts {
		ids, _, err := codec.Encode(text)
		if err != nil {
			return nil, fmt.Errorf("openai embeddings: tokenize text %d: %w", i, err)
		}
		if len(ids) <= ctxLength {
			chunks = append(chunks, embeddingChunk{textIndex: i, ids: ids, text: text})
			continue
		}
		for start := 0; start < len(ids); start += ctxLength {
			end := start + ctxLength
			if end > len(ids) {
				end = len(ids)
			}
			chunks = append(chunks, embeddingChunk{textIndex: i, ids: ids[start:end], asTokens: true})
		}
	}

	tokenCounts := make([]int, len(chunks))
	for i, chunk := range chunks {
		tokenCounts[i] = len(chunk.ids)
	}

	perTextVectors := make([][][]float64, len(texts))
	perTextWeights := make([][]int, len(texts))
	for _, batch := range planEmbeddingBatches(tokenCounts, e.chunkSize()) {
		response, err := e.postChunkBatch(ctx, chunks[batch[0]:batch[1]], post)
		if err != nil {
			return nil, err
		}
		batchChunks := chunks[batch[0]:batch[1]]
		slices.SortStableFunc(response.Data, func(a, b embeddingDataPayload) int {
			return cmp.Compare(a.Index, b.Index)
		})
		if len(response.Data) != len(batchChunks) {
			return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(response.Data), len(batchChunks))
		}
		for i, item := range response.Data {
			if item.Index < 0 || item.Index >= len(batchChunks) {
				return nil, fmt.Errorf("embedding index out of range: %d", item.Index)
			}
			chunk := batchChunks[i]
			perTextVectors[chunk.textIndex] = append(perTextVectors[chunk.textIndex], item.Embedding)
			perTextWeights[chunk.textIndex] = append(perTextWeights[chunk.textIndex], len(chunk.ids))
		}
	}

	vectors := make([][]float64, len(texts))
	for i := range texts {
		vectors[i] = mergeChunkedEmbeddings(perTextVectors[i], perTextWeights[i])
	}
	return vectors, nil
}

// postChunkBatch sends one batch of chunks. Batches that only contain
// whole-text chunks keep the string input shape; any batch carrying token
// chunks is sent as a homogeneous token-array request (whole-text chunks are
// tokenized too), matching the input shapes Python sends when checking
// context length (embeddings/base.py:619) and keeping the API's mixed-shape
// restriction satisfied.
func (e Embeddings) postChunkBatch(
	ctx context.Context,
	batch []embeddingChunk,
	post func(ctx context.Context, payload any) (embeddingResponsePayload, error),
) (embeddingResponsePayload, error) {
	anyTokens := false
	for _, chunk := range batch {
		if chunk.asTokens {
			anyTokens = true
			break
		}
	}
	if !anyTokens {
		inputs := make([]string, len(batch))
		for i, chunk := range batch {
			inputs[i] = chunk.text
		}
		return post(ctx, embeddingRequestPayload{Model: e.config.Model, Input: inputs})
	}
	inputs := make([][]uint, len(batch))
	for i, chunk := range batch {
		inputs[i] = chunk.ids
	}
	return post(ctx, embeddingTokenRequestPayload{Model: e.config.Model, TokenInput: inputs})
}

// tokenizer returns the codec for the configured model, mirroring Python
// _tokenize's model resolution (embeddings/base.py:486-490):
// tiktoken_model_name overrides the model name, and unknown models fall back
// to cl100k_base.
func (e Embeddings) tokenizer() (tokenizer.Codec, error) {
	model := e.config.Model
	if e.config.TiktokenModelName != "" {
		model = e.config.TiktokenModelName
	}
	codec, err := getCodec(encodingForModel(model))
	if err != nil {
		return nil, fmt.Errorf("openai embeddings: load tokenizer for model %q: %w", model, err)
	}
	return codec, nil
}

// createEmbeddings posts one embeddings request (string- or token-shaped)
// and decodes the response.
func (e Embeddings) createEmbeddings(
	ctx context.Context,
	payload any,
) (embeddingResponsePayload, error) {
	ctx, cancel := context.WithTimeout(ctx, e.config.Timeout)
	defer cancel()
	switch typed := payload.(type) {
	case embeddingRequestPayload:
		if dimensions, ok := e.config.Extra[embeddingDimensionsKey].(int); ok && dimensions > 0 {
			typed.Dimensions = &dimensions
			payload = typed
		}
		if format, ok := e.config.Extra[embeddingEncodingFormatKey].(string); ok && format != "" {
			typed.EncodingFormat = format
			payload = typed
		}
	case embeddingTokenRequestPayload:
		if dimensions, ok := e.config.Extra[embeddingDimensionsKey].(int); ok && dimensions > 0 {
			typed.Dimensions = &dimensions
			payload = typed
		}
		if format, ok := e.config.Extra[embeddingEncodingFormatKey].(string); ok && format != "" {
			typed.EncodingFormat = format
			payload = typed
		}
	}
	return postJSON[embeddingResponsePayload](ctx, e.config, "/embeddings", payload)
}

type embeddingRequestPayload struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	Dimensions     *int     `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

// embeddingTokenRequestPayload is the token-array variant of the embeddings
// request: input is a list of token-ID arrays, the shape Python sends for
// chunked texts (embeddings/base.py:619). Its JSON still uses the "input" key.
type embeddingTokenRequestPayload struct {
	Model          string   `json:"model"`
	TokenInput     [][]uint `json:"input"`
	Dimensions     *int     `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

type embeddingResponsePayload struct {
	Object string                 `json:"object"`
	Model  string                 `json:"model"`
	Data   []embeddingDataPayload `json:"data"`
	Usage  embeddingUsagePayload  `json:"usage"`
}

type embeddingDataPayload struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type embeddingUsagePayload struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
