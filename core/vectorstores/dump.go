package vectorstores

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

// inMemoryDumpRecord mirrors Python's per-record keys
// (vectorstores/in_memory.py:214-219). Python's dumpd leaves these plain
// dicts untouched, so the dumped file is a flat {"<id>": {record}} object.
type inMemoryDumpRecord struct {
	ID       string         `json:"id"`
	Vector   []float64      `json:"vector"`
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata"`
}

// Dump writes the store to path as indented JSON using Python's record shape
// (in_memory.py:537: json.dump(dumpd(self.store), f, indent=2)). Parent
// directories are created as needed (Python mkdir(parents=True, exist_ok=True)).
func (s *InMemory) Dump(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	records := make(map[string]inMemoryDumpRecord, len(s.documents))
	for _, id := range s.idSequence {
		doc, ok := s.documents[id]
		if !ok {
			continue
		}
		metadata := map[string]any{}
		for key, value := range doc.Metadata {
			metadata[key] = value
		}
		records[id] = inMemoryDumpRecord{
			ID:       id,
			Vector:   append([]float64(nil), s.vectors[id]...),
			Text:     doc.PageContent,
			Metadata: metadata,
		}
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("vectorstores: dump: create parent dirs: %w", err)
		}
	}
	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return fmt.Errorf("vectorstores: dump: encode: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("vectorstores: dump: write %s: %w", path, err)
	}
	return nil
}

// LoadInMemory reads a file produced by Dump (or by Python's
// InMemoryVectorStore.dump) and returns a store that uses embedder for future
// embedding calls (Python in_memory.py:517 classmethod load). Records are
// restored with their IDs, vectors, text, and metadata; insertion order is
// by sorted ID for determinism (Python relies on dict order, which Go maps
// do not preserve).
func LoadInMemory(path string, embedder embeddings.Embeddings) (*InMemory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("vectorstores: load: read %s: %w", path, err)
	}
	var records map[string]inMemoryDumpRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, fmt.Errorf("vectorstores: load: decode %s: %w", path, err)
	}
	store := NewInMemory(embedder)
	ids := make([]string, 0, len(records))
	for id := range records {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		record := records[id]
		docID := record.ID
		if docID == "" {
			docID = id
		}
		store.documents[docID] = documents.New(record.Text, record.Metadata).WithID(docID)
		store.vectors[docID] = append([]float64(nil), record.Vector...)
		store.idSequence = append(store.idSequence, docID)
	}
	// Keep generated IDs (doc-N) from colliding with restored records.
	store.nextID = len(store.idSequence)
	return store, nil
}
