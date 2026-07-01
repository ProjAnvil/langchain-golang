// Package chroma provides a Chroma vector store adapter.
package chroma

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/httpclient"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/retry"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

const (
	providerName    = "chroma"
	defaultBaseURL  = "http://localhost:8000"
	defaultTenant   = "default_tenant"
	defaultDatabase = "default_database"
)

// Store is a Chroma-backed vector store.
type Store struct {
	cfg            modelconfig.Config
	embedder       embeddings.Embeddings
	collectionName string
	collectionID   string
	metadata       map[string]any
	configuration  map[string]any
	tenant         string
	database       string
}

// Option configures a Chroma store.
type Option func(*Store)

// WithBaseURL sets the Chroma server base URL.
func WithBaseURL(baseURL string) Option {
	return func(s *Store) {
		s.cfg.BaseURL = baseURL
	}
}

// WithHTTPClient sets the HTTP client used for Chroma requests.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Store) {
		s.cfg.HTTPClient = client
	}
}

// WithHeader adds a header to every Chroma request.
func WithHeader(name, value string) Option {
	return func(s *Store) {
		if s.cfg.Headers == nil {
			s.cfg.Headers = map[string]string{}
		}
		s.cfg.Headers[name] = value
	}
}

// WithTenant sets the Chroma tenant name. It defaults to "default_tenant".
func WithTenant(tenant string) Option {
	return func(s *Store) {
		s.tenant = tenant
	}
}

// WithDatabase sets the Chroma database name. It defaults to
// "default_database".
func WithDatabase(database string) Option {
	return func(s *Store) {
		s.database = database
	}
}

// WithCollectionMetadata sets metadata when creating the collection.
func WithCollectionMetadata(metadata map[string]any) Option {
	return func(s *Store) {
		s.metadata = cloneMetadata(metadata)
	}
}

// WithCollectionConfiguration sets Chroma collection configuration, such as
// HNSW/SPANN distance metric options.
func WithCollectionConfiguration(configuration map[string]any) Option {
	return func(s *Store) {
		s.configuration = cloneMetadata(configuration)
	}
}

// WithMaxRetries sets the number of retries for retryable Chroma failures.
func WithMaxRetries(maxRetries int) Option {
	return func(s *Store) {
		s.cfg.MaxRetries = maxRetries
	}
}

// WithRetryDelay sets the initial retry delay.
func WithRetryDelay(delay time.Duration) Option {
	return func(s *Store) {
		s.cfg.RetryDelay = delay
	}
}

// New creates or reuses a Chroma collection.
func New(
	ctx context.Context,
	collectionName string,
	embedder embeddings.Embeddings,
	opts ...Option,
) (*Store, error) {
	if strings.TrimSpace(collectionName) == "" {
		return nil, fmt.Errorf("collection name is required")
	}
	if embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}

	store := &Store{
		cfg: modelconfig.New(
			modelconfig.WithBaseURL(defaultBaseURL),
		),
		embedder:       embedder,
		collectionName: collectionName,
		tenant:         defaultTenant,
		database:       defaultDatabase,
	}
	for _, opt := range opts {
		opt(store)
	}
	if store.cfg.Headers == nil {
		store.cfg.Headers = map[string]string{}
	}
	if strings.TrimSpace(store.cfg.BaseURL) == "" {
		store.cfg.BaseURL = defaultBaseURL
	}
	if strings.TrimSpace(store.tenant) == "" {
		store.tenant = defaultTenant
	}
	if strings.TrimSpace(store.database) == "" {
		store.database = defaultDatabase
	}

	collection, err := store.createCollection(ctx)
	if err != nil {
		return nil, err
	}
	store.collectionID = collection.ID
	if store.collectionID == "" {
		store.collectionID = collectionName
	}
	return store, nil
}

// AddDocuments embeds and adds documents to the Chroma collection.
func (s *Store) AddDocuments(ctx context.Context, docs []documents.Document) ([]string, error) {
	return s.UpsertDocuments(ctx, docs)
}

// AddTexts embeds and upserts raw text values.
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
	return s.UpsertDocuments(ctx, docs)
}

// UpsertDocuments embeds and upserts documents into the Chroma collection.
func (s *Store) UpsertDocuments(ctx context.Context, docs []documents.Document) ([]string, error) {
	if len(docs) == 0 {
		return nil, nil
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
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		id := doc.ID
		if id == "" {
			id = newID()
		}
		ids[i] = id
		metadatas[i] = cloneMetadata(doc.Metadata)
	}

	payload := writeRequest{
		IDs:        ids,
		Documents:  texts,
		Metadatas:  metadatas,
		Embeddings: vectors,
	}
	var out map[string]any
	if err := s.doJSON(ctx, http.MethodPost, s.collectionEndpoint("upsert"), payload, &out); err != nil {
		return nil, err
	}
	return ids, nil
}

// UpdateDocument updates one existing document.
func (s *Store) UpdateDocument(ctx context.Context, id string, doc documents.Document) error {
	return s.UpdateDocuments(ctx, []string{id}, []documents.Document{doc})
}

// UpdateDocuments embeds and updates existing Chroma records.
func (s *Store) UpdateDocuments(ctx context.Context, ids []string, docs []documents.Document) error {
	if len(ids) != len(docs) {
		return fmt.Errorf("ids/documents length mismatch: got %d ids and %d documents", len(ids), len(docs))
	}
	if len(docs) == 0 {
		return nil
	}
	texts := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
		metadatas[i] = cloneMetadata(doc.Metadata)
	}
	vectors, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return err
	}
	if len(vectors) != len(docs) {
		return fmt.Errorf("embedding count mismatch: got %d want %d", len(vectors), len(docs))
	}
	var out map[string]any
	return s.doJSON(ctx, http.MethodPost, s.collectionEndpoint("update"), writeRequest{
		IDs:        ids,
		Documents:  texts,
		Metadatas:  metadatas,
		Embeddings: vectors,
	}, &out)
}

// Delete removes IDs from the Chroma collection. Missing IDs are ignored by
// Chroma.
func (s *Store) Delete(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	var out map[string]any
	return s.doJSON(ctx, http.MethodPost, s.collectionEndpoint("delete"), deleteRequest{IDs: ids}, &out)
}

// GetByIDs returns found documents for the requested IDs.
func (s *Store) GetByIDs(ctx context.Context, ids []string) ([]documents.Document, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	return s.Get(ctx, GetOptions{
		IDs:     ids,
		Include: []string{"documents", "metadatas"},
	})
}

// GetOptions configures a Chroma get request.
type GetOptions struct {
	IDs           []string
	Where         map[string]any
	WhereDocument map[string]any
	Limit         *int
	Offset        *int
	Include       []string
}

// Get returns documents from Chroma using IDs and/or Chroma filters.
func (s *Store) Get(ctx context.Context, opts GetOptions) ([]documents.Document, error) {
	var resp getResponse
	err := s.doJSON(ctx, http.MethodPost, s.collectionEndpoint("get"), getRequest{
		IDs:           opts.IDs,
		Where:         opts.Where,
		WhereDocument: opts.WhereDocument,
		Limit:         opts.Limit,
		Offset:        opts.Offset,
		Include:       defaultInclude(opts.Include, "documents", "metadatas"),
	}, &resp)
	if err != nil {
		return nil, err
	}
	return docsFromFlatResponse(resp.IDs, resp.Documents, resp.Metadatas), nil
}

// SimilaritySearch returns the top k documents for query.
func (s *Store) SimilaritySearch(ctx context.Context, query string, k int) ([]documents.Document, error) {
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

// SimilaritySearchWithScore returns the top k documents and Chroma distances.
func (s *Store) SimilaritySearchWithScore(
	ctx context.Context,
	query string,
	k int,
) ([]vectorstores.SearchResult, error) {
	return s.SimilaritySearchWithScoreOptions(ctx, query, QueryOptions{K: k})
}

// QueryOptions configures Chroma similarity and MMR queries.
type QueryOptions struct {
	K             int
	Where         map[string]any
	WhereDocument map[string]any
	Include       []string
}

// SimilaritySearchOptions runs text similarity search with Chroma filters.
func (s *Store) SimilaritySearchOptions(
	ctx context.Context,
	query string,
	opts QueryOptions,
) ([]documents.Document, error) {
	results, err := s.SimilaritySearchWithScoreOptions(ctx, query, opts)
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, len(results))
	for i, result := range results {
		docs[i] = result.Document
	}
	return docs, nil
}

// SimilaritySearchWithScoreOptions runs text similarity search with Chroma
// filters and returns Chroma distances. Lower scores are more similar.
func (s *Store) SimilaritySearchWithScoreOptions(
	ctx context.Context,
	query string,
	opts QueryOptions,
) ([]vectorstores.SearchResult, error) {
	if opts.K <= 0 {
		opts.K = 4
	}
	queryVector, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	return s.SimilaritySearchByVectorWithScore(ctx, queryVector, opts)
}

// SimilaritySearchByVector returns the top documents for an embedding vector.
func (s *Store) SimilaritySearchByVector(
	ctx context.Context,
	embedding []float64,
	opts QueryOptions,
) ([]documents.Document, error) {
	results, err := s.SimilaritySearchByVectorWithScore(ctx, embedding, opts)
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, len(results))
	for i, result := range results {
		docs[i] = result.Document
	}
	return docs, nil
}

// SimilaritySearchByVectorWithScore returns Chroma distances for an embedding
// vector. Lower scores are more similar.
func (s *Store) SimilaritySearchByVectorWithScore(
	ctx context.Context,
	embedding []float64,
	opts QueryOptions,
) ([]vectorstores.SearchResult, error) {
	if opts.K <= 0 {
		opts.K = 4
	}
	var resp queryResponse
	err := s.doJSON(ctx, http.MethodPost, s.collectionEndpoint("query"), queryRequest{
		QueryEmbeddings: [][]float64{append([]float64(nil), embedding...)},
		NResults:        opts.K,
		Where:           opts.Where,
		WhereDocument:   opts.WhereDocument,
		Include:         defaultInclude(opts.Include, "documents", "metadatas", "distances"),
	}, &resp)
	if err != nil {
		return nil, err
	}
	if len(resp.IDs) == 0 {
		return nil, nil
	}
	return searchResultsFromQuery(resp), nil
}

// DocumentVector is a document paired with its stored embedding.
type DocumentVector struct {
	Document documents.Document
	Vector   []float64
}

// SimilaritySearchWithVectors returns documents and stored embedding vectors.
func (s *Store) SimilaritySearchWithVectors(
	ctx context.Context,
	query string,
	opts QueryOptions,
) ([]DocumentVector, error) {
	queryVector, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	return s.SimilaritySearchByVectorWithVectors(ctx, queryVector, opts)
}

// SimilaritySearchByVectorWithVectors returns documents and stored embedding
// vectors for an embedding query.
func (s *Store) SimilaritySearchByVectorWithVectors(
	ctx context.Context,
	embedding []float64,
	opts QueryOptions,
) ([]DocumentVector, error) {
	if opts.K <= 0 {
		opts.K = 4
	}
	var resp queryResponse
	err := s.doJSON(ctx, http.MethodPost, s.collectionEndpoint("query"), queryRequest{
		QueryEmbeddings: [][]float64{append([]float64(nil), embedding...)},
		NResults:        opts.K,
		Where:           opts.Where,
		WhereDocument:   opts.WhereDocument,
		Include:         defaultInclude(opts.Include, "documents", "metadatas", "embeddings"),
	}, &resp)
	if err != nil {
		return nil, err
	}
	return documentVectorsFromQuery(resp), nil
}

// MMROptions configures maximal marginal relevance search.
type MMROptions struct {
	K             int
	FetchK        int
	LambdaMult    float64
	Where         map[string]any
	WhereDocument map[string]any
}

// MaxMarginalRelevanceSearch returns documents selected for query relevance
// and diversity.
func (s *Store) MaxMarginalRelevanceSearch(
	ctx context.Context,
	query string,
	opts MMROptions,
) ([]documents.Document, error) {
	queryVector, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	return s.MaxMarginalRelevanceSearchByVector(ctx, queryVector, opts)
}

// MaxMarginalRelevanceSearchByVector returns documents selected for embedding
// relevance and diversity.
func (s *Store) MaxMarginalRelevanceSearchByVector(
	ctx context.Context,
	embedding []float64,
	opts MMROptions,
) ([]documents.Document, error) {
	if opts.K <= 0 {
		opts.K = 4
	}
	if opts.FetchK <= 0 {
		opts.FetchK = 20
	}
	if math.IsNaN(opts.LambdaMult) {
		opts.LambdaMult = 0.5
	}
	candidates, err := s.SimilaritySearchByVectorWithVectors(ctx, embedding, QueryOptions{
		K:             opts.FetchK,
		Where:         opts.Where,
		WhereDocument: opts.WhereDocument,
		Include:       []string{"documents", "metadatas", "embeddings", "distances"},
	})
	if err != nil {
		return nil, err
	}
	vectors := make([][]float64, len(candidates))
	for i, candidate := range candidates {
		vectors[i] = candidate.Vector
	}
	selected := maximalMarginalRelevance(embedding, vectors, opts.LambdaMult, opts.K)
	docs := make([]documents.Document, 0, len(selected))
	for _, index := range selected {
		if index >= 0 && index < len(candidates) {
			docs = append(docs, candidates[index].Document)
		}
	}
	return docs, nil
}

// DeleteCollection deletes the underlying Chroma collection and marks this
// store unusable until ResetCollection is called.
func (s *Store) DeleteCollection(ctx context.Context) error {
	endpoint := fmt.Sprintf("%s/%s", s.collectionsEndpoint(), s.collectionID)
	if err := s.doJSON(ctx, http.MethodDelete, endpoint, nil, nil); err != nil {
		return err
	}
	s.collectionID = ""
	return nil
}

// ResetCollection deletes and recreates the collection.
func (s *Store) ResetCollection(ctx context.Context) error {
	if s.collectionID != "" {
		if err := s.DeleteCollection(ctx); err != nil {
			return err
		}
	}
	collection, err := s.createCollection(ctx)
	if err != nil {
		return err
	}
	s.collectionID = collection.ID
	if s.collectionID == "" {
		s.collectionID = s.collectionName
	}
	return nil
}

// Fork asks Chroma to fork the collection and returns a Store bound to the fork.
func (s *Store) Fork(ctx context.Context, newName string) (*Store, error) {
	var resp collectionResponse
	err := s.doJSON(ctx, http.MethodPost, s.collectionEndpoint("fork"), forkRequest{
		NewName: newName,
	}, &resp)
	if err != nil {
		return nil, err
	}
	fork := *s
	fork.collectionName = newName
	fork.collectionID = resp.ID
	if fork.collectionID == "" {
		fork.collectionID = newName
	}
	return &fork, nil
}

func (s *Store) createCollection(ctx context.Context) (collectionResponse, error) {
	var resp collectionResponse
	err := s.doJSON(ctx, http.MethodPost, s.collectionsEndpoint(), createCollectionRequest{
		Name:          s.collectionName,
		GetOrCreate:   true,
		Metadata:      s.metadata,
		Configuration: s.configuration,
	}, &resp)
	if err != nil {
		return collectionResponse{}, err
	}
	return resp, nil
}

func (s *Store) collectionsEndpoint() string {
	return fmt.Sprintf(
		"/api/v2/tenants/%s/databases/%s/collections",
		s.tenant,
		s.database,
	)
}

func (s *Store) collectionEndpoint(action string) string {
	return fmt.Sprintf(
		"/api/v2/tenants/%s/databases/%s/collections/%s/%s",
		s.tenant,
		s.database,
		s.collectionID,
		action,
	)
}

func (s *Store) doJSON(ctx context.Context, method string, endpoint string, payload any, out any) error {
	var body []byte
	if payload != nil {
		var err error
		body, err = json.Marshal(payload)
		if err != nil {
			return err
		}
	}

	client := s.cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	url := strings.TrimRight(s.cfg.BaseURL, "/") + endpoint

	return retry.Do(ctx, retry.Policy{
		MaxAttempts:       s.cfg.MaxRetries + 1,
		Delay:             s.cfg.RetryDelay,
		BackoffMultiplier: s.cfg.RetryBackoffMultiplier,
		MaxDelay:          s.cfg.RetryMaxDelay,
		ShouldRetry:       httpclient.IsRetryable,
	}, func() error {
		req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		for name, value := range s.cfg.Headers {
			req.Header.Set(name, value)
		}

		resp, err := client.Do(req)
		if err != nil {
			return lcerrors.WrapTransport(err)
		}
		defer resp.Body.Close()
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return lcerrors.WrapTransport(err)
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return lcerrors.NewProviderError(providerName, endpoint, resp.StatusCode, string(respBody), httpclient.RetryAfter(resp))
		}
		if out == nil || len(respBody) == 0 {
			return nil
		}
		if err := json.Unmarshal(respBody, out); err != nil {
			return fmt.Errorf("decode chroma %s response: %w", endpoint, err)
		}
		return nil
	})
}

type createCollectionRequest struct {
	Name          string         `json:"name"`
	GetOrCreate   bool           `json:"get_or_create"`
	Metadata      map[string]any `json:"metadata,omitempty"`
	Configuration map[string]any `json:"configuration,omitempty"`
}

type collectionResponse struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type forkRequest struct {
	NewName string `json:"new_name"`
}

type writeRequest struct {
	IDs        []string         `json:"ids"`
	Documents  []string         `json:"documents"`
	Metadatas  []map[string]any `json:"metadatas"`
	Embeddings [][]float64      `json:"embeddings"`
}

type deleteRequest struct {
	IDs []string `json:"ids"`
}

type getRequest struct {
	IDs           []string       `json:"ids,omitempty"`
	Where         map[string]any `json:"where,omitempty"`
	WhereDocument map[string]any `json:"where_document,omitempty"`
	Limit         *int           `json:"limit,omitempty"`
	Offset        *int           `json:"offset,omitempty"`
	Include       []string       `json:"include,omitempty"`
}

type getResponse struct {
	IDs       []string         `json:"ids"`
	Documents []string         `json:"documents"`
	Metadatas []map[string]any `json:"metadatas"`
}

type queryRequest struct {
	QueryEmbeddings [][]float64    `json:"query_embeddings"`
	NResults        int            `json:"n_results"`
	Where           map[string]any `json:"where,omitempty"`
	WhereDocument   map[string]any `json:"where_document,omitempty"`
	Include         []string       `json:"include"`
}

type queryResponse struct {
	IDs        [][]string         `json:"ids"`
	Documents  [][]string         `json:"documents"`
	Metadatas  [][]map[string]any `json:"metadatas"`
	Distances  [][]float64        `json:"distances"`
	Embeddings [][][]float64      `json:"embeddings"`
}

func docsFromFlatResponse(
	ids []string,
	contents []string,
	metadatas []map[string]any,
) []documents.Document {
	docs := make([]documents.Document, 0, len(ids))
	for i, id := range ids {
		doc := documents.Document{ID: id}
		if i < len(contents) {
			doc.PageContent = contents[i]
		}
		if i < len(metadatas) {
			doc.Metadata = cloneMetadata(metadatas[i])
		}
		docs = append(docs, doc)
	}
	return docs
}

func searchResultsFromQuery(resp queryResponse) []vectorstores.SearchResult {
	if len(resp.IDs) == 0 {
		return nil
	}
	ids := resp.IDs[0]
	results := make([]vectorstores.SearchResult, 0, len(ids))
	for i, id := range ids {
		doc := documents.Document{ID: id}
		if len(resp.Documents) > 0 && i < len(resp.Documents[0]) {
			doc.PageContent = resp.Documents[0][i]
		}
		if len(resp.Metadatas) > 0 && i < len(resp.Metadatas[0]) {
			doc.Metadata = cloneMetadata(resp.Metadatas[0][i])
		}
		result := vectorstores.SearchResult{Document: doc}
		if len(resp.Distances) > 0 && i < len(resp.Distances[0]) {
			result.Score = resp.Distances[0][i]
		}
		results = append(results, result)
	}
	return results
}

func documentVectorsFromQuery(resp queryResponse) []DocumentVector {
	if len(resp.IDs) == 0 {
		return nil
	}
	ids := resp.IDs[0]
	results := make([]DocumentVector, 0, len(ids))
	for i, id := range ids {
		doc := documents.Document{ID: id}
		if len(resp.Documents) > 0 && i < len(resp.Documents[0]) {
			doc.PageContent = resp.Documents[0][i]
		}
		if len(resp.Metadatas) > 0 && i < len(resp.Metadatas[0]) {
			doc.Metadata = cloneMetadata(resp.Metadatas[0][i])
		}
		item := DocumentVector{Document: doc}
		if len(resp.Embeddings) > 0 && i < len(resp.Embeddings[0]) {
			item.Vector = append([]float64(nil), resp.Embeddings[0][i]...)
		}
		results = append(results, item)
	}
	return results
}

func maximalMarginalRelevance(
	query []float64,
	embeddings [][]float64,
	lambdaMult float64,
	k int,
) []int {
	if k <= 0 || len(embeddings) == 0 {
		return nil
	}
	similarityToQuery := make([]float64, len(embeddings))
	bestIndex := 0
	bestScore := math.Inf(-1)
	for i, embedding := range embeddings {
		score := cosineSimilarity(query, embedding)
		similarityToQuery[i] = score
		if score > bestScore {
			bestScore = score
			bestIndex = i
		}
	}

	selected := []int{bestIndex}
	for len(selected) < minInt(k, len(embeddings)) {
		nextIndex := -1
		nextScore := math.Inf(-1)
		for i := range embeddings {
			if containsInt(selected, i) {
				continue
			}
			var redundantScore float64
			for _, selectedIndex := range selected {
				redundantScore = math.Max(redundantScore, cosineSimilarity(embeddings[i], embeddings[selectedIndex]))
			}
			score := lambdaMult*similarityToQuery[i] - (1-lambdaMult)*redundantScore
			if score > nextScore {
				nextScore = score
				nextIndex = i
			}
		}
		if nextIndex < 0 {
			break
		}
		selected = append(selected, nextIndex)
	}
	return selected
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot float64
	var normA float64
	var normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func defaultInclude(include []string, values ...string) []string {
	if len(include) > 0 {
		return append([]string(nil), include...)
	}
	return append([]string(nil), values...)
}

func metadataAt(metadatas []map[string]any, index int) map[string]any {
	if index >= len(metadatas) {
		return nil
	}
	return metadatas[index]
}

func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("doc-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

func containsInt(values []int, needle int) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func cloneMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return nil
	}
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		out[key] = value
	}
	return out
}
