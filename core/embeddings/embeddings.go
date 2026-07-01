package embeddings

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"hash/fnv"
	"math"
	mathrand "math/rand"
	"strings"
	"sync"
)

// Embeddings converts text into dense vectors.
type Embeddings interface {
	EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error)
	EmbedQuery(ctx context.Context, text string) ([]float64, error)
}

// FakeEmbeddings is a deterministic local embedding model for unit tests.
type FakeEmbeddings struct {
	dimensions int
}

// NewFake creates deterministic embeddings with the requested dimensionality.
func NewFake(dimensions int) FakeEmbeddings {
	if dimensions <= 0 {
		dimensions = 8
	}
	return FakeEmbeddings{dimensions: dimensions}
}

// Dimensions returns the vector size.
func (e FakeEmbeddings) Dimensions() int {
	return e.dimensions
}

// EmbedDocuments embeds all documents.
func (e FakeEmbeddings) EmbedDocuments(
	ctx context.Context,
	texts []string,
) ([][]float64, error) {
	vectors := make([][]float64, len(texts))
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vectors[i] = e.embed(text)
	}
	return vectors, nil
}

// EmbedQuery embeds one query.
func (e FakeEmbeddings) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return e.embed(text), nil
}

func (e FakeEmbeddings) embed(text string) []float64 {
	vector := make([]float64, e.dimensions)
	for _, token := range strings.Fields(strings.ToLower(text)) {
		hash := fnv.New64a()
		_, _ = hash.Write([]byte(token))
		index := int(hash.Sum64() % uint64(e.dimensions))
		vector[index]++
	}
	normalize(vector)
	return vector
}

// DeterministicFakeEmbedding creates deterministic normally-distributed vectors
// using a seed derived from each input text, matching Python's testing fake in
// spirit without requiring numpy.
type DeterministicFakeEmbedding struct {
	size int
}

func NewDeterministicFake(size int) DeterministicFakeEmbedding {
	if size <= 0 {
		size = 8
	}
	return DeterministicFakeEmbedding{size: size}
}

func (e DeterministicFakeEmbedding) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	vectors := make([][]float64, len(texts))
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vectors[i] = seededNormalVector(text, e.size)
	}
	return vectors, nil
}

func (e DeterministicFakeEmbedding) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return seededNormalVector(text, e.size), nil
}

// RandomFakeEmbedding creates normally-distributed vectors for each call,
// matching Python's FakeEmbeddings test model behavior.
type RandomFakeEmbedding struct {
	size int
	mu   sync.Mutex
	rng  *mathrand.Rand
}

func NewRandomFake(size int) RandomFakeEmbedding {
	if size <= 0 {
		size = 8
	}
	return RandomFakeEmbedding{
		size: size,
		rng:  mathrand.New(mathrand.NewSource(randomSeed())),
	}
}

func (e *RandomFakeEmbedding) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	vectors := make([][]float64, len(texts))
	for i := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vectors[i] = e.nextNormalVector()
	}
	return vectors, nil
}

func (e *RandomFakeEmbedding) EmbedQuery(ctx context.Context, _ string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return e.nextNormalVector(), nil
}

func (e *RandomFakeEmbedding) nextNormalVector() []float64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.rng == nil {
		e.rng = mathrand.New(mathrand.NewSource(randomSeed()))
	}
	vector := make([]float64, e.size)
	for i := range vector {
		vector[i] = e.rng.NormFloat64()
	}
	return vector
}

// Static returns configured vectors in order and is useful for error-free
// deterministic tests where exact vectors matter.
type Static struct {
	DocumentVectors [][]float64
	QueryVector     []float64
}

func (e Static) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	vectors := make([][]float64, len(texts))
	for i := range texts {
		if i < len(e.DocumentVectors) {
			vectors[i] = append([]float64(nil), e.DocumentVectors[i]...)
		}
	}
	return vectors, nil
}

func (e Static) EmbedQuery(ctx context.Context, _ string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return append([]float64(nil), e.QueryVector...), nil
}

func seededNormalVector(text string, size int) []float64 {
	sum := sha256.Sum256([]byte(text))
	seed := int64(binary.BigEndian.Uint64(sum[:8]))
	rng := mathrand.New(mathrand.NewSource(seed))
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = rng.NormFloat64()
	}
	return vector
}

func randomSeed() int64 {
	var bytes [8]byte
	if _, err := rand.Read(bytes[:]); err == nil {
		return int64(binary.BigEndian.Uint64(bytes[:]))
	}
	return 1
}

func normalize(vector []float64) {
	var sum float64
	for _, value := range vector {
		sum += value * value
	}
	if sum == 0 {
		return
	}
	norm := math.Sqrt(sum)
	for i := range vector {
		vector[i] /= norm
	}
}
