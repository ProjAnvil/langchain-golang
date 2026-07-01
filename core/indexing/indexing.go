package indexing

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/projanvil/langchain-golang/core/documentloaders"
	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// Record stores indexing metadata for one document key.
type Record struct {
	GroupID   string
	UpdatedAt time.Time
}

// RecordManager tracks indexed document keys.
type RecordManager interface {
	GetTime(ctx context.Context) (time.Time, error)
	Update(ctx context.Context, keys []string, groupIDs []string, timeAtLeast time.Time) error
	Exists(ctx context.Context, keys []string) ([]bool, error)
	DeleteKeys(ctx context.Context, keys []string) error
	ListKeys(ctx context.Context, groupIDs []string, before time.Time, limit int) ([]string, error)
}

// InMemoryRecordManager is a deterministic in-memory record manager for tests
// and local use.
type InMemoryRecordManager struct {
	mu        sync.RWMutex
	Namespace string
	records   map[string]Record
}

// NewInMemoryRecordManager creates an in-memory record manager.
func NewInMemoryRecordManager(namespace string) *InMemoryRecordManager {
	return &InMemoryRecordManager{
		Namespace: namespace,
		records:   map[string]Record{},
	}
}

// GetTime returns the current local time.
func (m *InMemoryRecordManager) GetTime(context.Context) (time.Time, error) {
	return time.Now(), nil
}

// Update upserts record keys with optional group IDs.
func (m *InMemoryRecordManager) Update(ctx context.Context, keys []string, groupIDs []string, timeAtLeast time.Time) error {
	if len(groupIDs) > 0 && len(keys) != len(groupIDs) {
		return fmt.Errorf("length of keys must match length of groupIDs")
	}
	now, err := m.GetTime(ctx)
	if err != nil {
		return err
	}
	if !timeAtLeast.IsZero() && now.Before(timeAtLeast) {
		return fmt.Errorf("timeAtLeast must be in the past")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for i, key := range keys {
		groupID := ""
		if len(groupIDs) > 0 {
			groupID = groupIDs[i]
		}
		m.records[m.namespaced(key)] = Record{GroupID: groupID, UpdatedAt: now}
	}
	return nil
}

// Exists reports which keys have records.
func (m *InMemoryRecordManager) Exists(_ context.Context, keys []string) ([]bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]bool, len(keys))
	for i, key := range keys {
		_, out[i] = m.records[m.namespaced(key)]
	}
	return out, nil
}

// DeleteKeys deletes records by key.
func (m *InMemoryRecordManager) DeleteKeys(_ context.Context, keys []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, key := range keys {
		delete(m.records, m.namespaced(key))
	}
	return nil
}

// ListKeys lists record keys, optionally filtered by group ID and updated time.
func (m *InMemoryRecordManager) ListKeys(_ context.Context, groupIDs []string, before time.Time, limit int) ([]string, error) {
	groupSet := map[string]bool{}
	for _, groupID := range groupIDs {
		groupSet[groupID] = true
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := []string{}
	for namespacedKey, record := range m.records {
		if len(groupSet) > 0 && !groupSet[record.GroupID] {
			continue
		}
		if !before.IsZero() && !record.UpdatedAt.Before(before) {
			continue
		}
		out = append(out, m.unnamespace(namespacedKey))
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, nil
}

func (m *InMemoryRecordManager) namespaced(key string) string {
	if m.Namespace == "" {
		return key
	}
	return m.Namespace + ":" + key
}

func (m *InMemoryRecordManager) unnamespace(key string) string {
	prefix := m.Namespace + ":"
	if m.Namespace != "" && len(key) >= len(prefix) && key[:len(prefix)] == prefix {
		return key[len(prefix):]
	}
	return key
}

// IndexingResult summarizes an indexing run.
type IndexingResult struct {
	NumAdded   int
	NumUpdated int
	NumDeleted int
	NumSkipped int
}

// CleanupMode controls stale-record cleanup during indexing.
type CleanupMode string

const (
	// CleanupNone leaves stale records untouched.
	CleanupNone CleanupMode = ""
	// CleanupIncremental deletes stale records for touched source groups after
	// each batch.
	CleanupIncremental CleanupMode = "incremental"
	// CleanupFull deletes stale records for touched source groups after all
	// batches are indexed.
	CleanupFull CleanupMode = "full"
	// CleanupScopedFull deletes stale records only for source groups seen
	// during this indexing run after all batches are indexed.
	CleanupScopedFull CleanupMode = "scoped_full"
)

// Options configures indexing.
type Options struct {
	BatchSize        int
	CleanupBatchSize int
	ForceUpdate      bool
	SourceIDKey      string
	Cleanup          CleanupMode
}

// IndexDocuments indexes documents into a vector store while skipping records
// already present in the record manager.
func IndexDocuments(
	ctx context.Context,
	docs []documents.Document,
	recordManager RecordManager,
	vectorStore vectorstores.VectorStore,
	options Options,
) (IndexingResult, error) {
	batchSize, cleanupBatchSize, err := validateIndexingInputs(recordManager, vectorStore, options)
	if err != nil {
		return IndexingResult{}, err
	}
	indexStart, err := recordManager.GetTime(ctx)
	if err != nil {
		return IndexingResult{}, err
	}
	state := indexState{
		touchedGroups: map[string]bool{},
		indexStart:    indexStart,
	}
	for start := 0; start < len(docs); start += batchSize {
		end := start + batchSize
		if end > len(docs) {
			end = len(docs)
		}
		if err := indexBatch(ctx, docs[start:end], recordManager, vectorStore, options, cleanupBatchSize, &state); err != nil {
			return state.result, err
		}
	}
	if err := finishCleanup(ctx, recordManager, vectorStore, options, cleanupBatchSize, &state); err != nil {
		return state.result, err
	}
	return state.result, nil
}

// IndexDocumentIterator indexes documents from an iterator without loading the
// full input into memory. The iterator is closed before the function returns.
func IndexDocumentIterator(
	ctx context.Context,
	iter documentloaders.DocumentIterator,
	recordManager RecordManager,
	vectorStore vectorstores.VectorStore,
	options Options,
) (IndexingResult, error) {
	if iter == nil {
		return IndexingResult{}, fmt.Errorf("document iterator is required")
	}
	defer iter.Close()
	batchSize, cleanupBatchSize, err := validateIndexingInputs(recordManager, vectorStore, options)
	if err != nil {
		return IndexingResult{}, err
	}
	indexStart, err := recordManager.GetTime(ctx)
	if err != nil {
		return IndexingResult{}, err
	}
	state := indexState{
		touchedGroups: map[string]bool{},
		indexStart:    indexStart,
	}
	batch := make([]documents.Document, 0, batchSize)
	for {
		if err := ctx.Err(); err != nil {
			return state.result, err
		}
		doc, ok, err := iter.Next(ctx)
		if err != nil {
			return state.result, err
		}
		if !ok {
			break
		}
		batch = append(batch, doc)
		if len(batch) < batchSize {
			continue
		}
		if err := indexBatch(ctx, batch, recordManager, vectorStore, options, cleanupBatchSize, &state); err != nil {
			return state.result, err
		}
		batch = batch[:0]
	}
	if len(batch) > 0 {
		if err := indexBatch(ctx, batch, recordManager, vectorStore, options, cleanupBatchSize, &state); err != nil {
			return state.result, err
		}
	}
	if err := finishCleanup(ctx, recordManager, vectorStore, options, cleanupBatchSize, &state); err != nil {
		return state.result, err
	}
	return state.result, nil
}

type indexState struct {
	result        IndexingResult
	touchedGroups map[string]bool
	indexStart    time.Time
}

func validateIndexingInputs(
	recordManager RecordManager,
	vectorStore vectorstores.VectorStore,
	options Options,
) (int, int, error) {
	if recordManager == nil {
		return 0, 0, fmt.Errorf("record manager is required")
	}
	if vectorStore == nil {
		return 0, 0, fmt.Errorf("vector store is required")
	}
	switch options.Cleanup {
	case CleanupNone, CleanupIncremental, CleanupFull, CleanupScopedFull:
	default:
		return 0, 0, fmt.Errorf("cleanup should be one of %q, %q, %q, or %q; got %q", CleanupNone, CleanupIncremental, CleanupFull, CleanupScopedFull, options.Cleanup)
	}
	if (options.Cleanup == CleanupIncremental || options.Cleanup == CleanupScopedFull) && options.SourceIDKey == "" {
		return 0, 0, fmt.Errorf("%s cleanup requires SourceIDKey", options.Cleanup)
	}
	batchSize := options.BatchSize
	if batchSize <= 0 {
		batchSize = 100
	}
	cleanupBatchSize := options.CleanupBatchSize
	if cleanupBatchSize <= 0 {
		cleanupBatchSize = 1000
	}
	return batchSize, cleanupBatchSize, nil
}

func indexBatch(
	ctx context.Context,
	batch []documents.Document,
	recordManager RecordManager,
	vectorStore vectorstores.VectorStore,
	options Options,
	cleanupBatchSize int,
	state *indexState,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	batchDocs := make([]documents.Document, 0, len(batch))
	keys := make([]string, 0, len(batch))
	groupIDs := make([]string, 0, len(batch))
	seenKeys := map[string]bool{}
	for i, doc := range batch {
		key, err := HashDocument(doc)
		if err != nil {
			return err
		}
		if seenKeys[key] {
			state.result.NumSkipped++
			continue
		}
		seenKeys[key] = true
		keys = append(keys, key)
		if options.SourceIDKey != "" {
			source, ok := doc.Metadata[options.SourceIDKey].(string)
			if !ok || source == "" {
				return fmt.Errorf("document metadata must include non-empty string source id %q when SourceIDKey is set", options.SourceIDKey)
			}
			groupIDs = append(groupIDs, source)
			state.touchedGroups[source] = true
		}
		batchDocs = append(batchDocs, batch[i])
	}
	if len(keys) == 0 {
		return nil
	}
	exists, err := recordManager.Exists(ctx, keys)
	if err != nil {
		return err
	}
	toAdd := []documents.Document{}
	for i, doc := range batchDocs {
		if exists[i] && !options.ForceUpdate {
			state.result.NumSkipped++
			continue
		}
		stored := doc.Clone().WithID(keys[i])
		toAdd = append(toAdd, stored)
		if exists[i] {
			state.result.NumUpdated++
		} else {
			state.result.NumAdded++
		}
	}
	if len(toAdd) > 0 {
		if _, err := vectorStore.AddDocuments(ctx, toAdd); err != nil {
			return err
		}
	}
	if err := recordManager.Update(ctx, keys, groupIDs, state.indexStart); err != nil {
		return err
	}
	if options.Cleanup == CleanupIncremental {
		deleted, err := cleanupKeys(ctx, recordManager, vectorStore, uniqueStrings(groupIDs), state.indexStart, cleanupBatchSize)
		if err != nil {
			return err
		}
		state.result.NumDeleted += deleted
	}
	return nil
}

func finishCleanup(
	ctx context.Context,
	recordManager RecordManager,
	vectorStore vectorstores.VectorStore,
	options Options,
	cleanupBatchSize int,
	state *indexState,
) error {
	if options.Cleanup == CleanupFull || options.Cleanup == CleanupScopedFull {
		groups := []string(nil)
		if options.Cleanup == CleanupScopedFull {
			groups = mapKeys(state.touchedGroups)
			if len(groups) == 0 {
				return nil
			}
		}
		deleted, err := cleanupKeys(ctx, recordManager, vectorStore, groups, state.indexStart, cleanupBatchSize)
		if err != nil {
			return err
		}
		state.result.NumDeleted += deleted
	}
	return nil
}

func cleanupKeys(
	ctx context.Context,
	recordManager RecordManager,
	vectorStore vectorstores.VectorStore,
	groupIDs []string,
	before time.Time,
	limit int,
) (int, error) {
	if limit <= 0 {
		limit = 100
	}
	deleted := 0
	for {
		if err := ctx.Err(); err != nil {
			return deleted, err
		}
		keys, err := recordManager.ListKeys(ctx, groupIDs, before, limit)
		if err != nil {
			return deleted, err
		}
		if len(keys) == 0 {
			return deleted, nil
		}
		if err := vectorStore.Delete(ctx, keys); err != nil {
			return deleted, err
		}
		if err := recordManager.DeleteKeys(ctx, keys); err != nil {
			return deleted, err
		}
		deleted += len(keys)
	}
}

func uniqueStrings(values []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, value := range values {
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func mapKeys(values map[string]bool) []string {
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	return out
}

// HashDocument returns a stable SHA-256 hash for page content and metadata.
func HashDocument(doc documents.Document) (string, error) {
	payload := struct {
		PageContent string         `json:"page_content"`
		Metadata    map[string]any `json:"metadata,omitempty"`
	}{
		PageContent: doc.PageContent,
		Metadata:    doc.Metadata,
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}
