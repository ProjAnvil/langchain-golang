package indexing

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestInMemoryDocumentIndexUpsertGeneratesIDs(t *testing.T) {
	ctx := context.Background()
	idx := NewInMemoryDocumentIndex(4)
	docs := []documents.Document{
		documents.New("alpha", nil),
		documents.New("beta", nil),
	}

	resp, err := idx.Upsert(ctx, docs)
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if len(resp.Succeeded) != 2 || len(resp.Failed) != 0 {
		t.Fatalf("upsert response = %+v, want 2 succeeded and 0 failed", resp)
	}
	first, second := resp.Succeeded[0], resp.Succeeded[1]
	if first == "" || second == "" {
		t.Fatalf("generated IDs empty: %q %q", first, second)
	}
	if first == second {
		t.Fatalf("generated IDs not unique: %q", first)
	}

	got, err := idx.Get(ctx, resp.Succeeded)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("get returned %d docs, want 2", len(got))
	}
	if got[0].ID != first || got[1].ID != second {
		t.Fatalf("get IDs = %q %q, want %q %q", got[0].ID, got[1].ID, first, second)
	}
}

func TestInMemoryDocumentIndexUpsertOverwrites(t *testing.T) {
	ctx := context.Background()
	idx := NewInMemoryDocumentIndex(4)
	const id = "fixed-id"

	resp, err := idx.Upsert(ctx, []documents.Document{
		documents.New("original", nil).WithID(id),
	})
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if len(resp.Succeeded) != 1 || resp.Succeeded[0] != id {
		t.Fatalf("first upsert response = %+v, want succeeded [%q]", resp, id)
	}

	resp, err = idx.Upsert(ctx, []documents.Document{
		documents.New("updated", nil).WithID(id),
	})
	if err != nil {
		t.Fatalf("overwrite upsert: %v", err)
	}
	if len(resp.Succeeded) != 1 || resp.Succeeded[0] != id {
		t.Fatalf("overwrite response = %+v, want succeeded [%q]", resp, id)
	}

	got, err := idx.Get(ctx, []string{id})
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("get returned %d docs, want 1", len(got))
	}
	if got[0].PageContent != "updated" {
		t.Fatalf("stored page content = %q, want %q", got[0].PageContent, "updated")
	}
}

func TestInMemoryDocumentIndexDeleteReportsCounts(t *testing.T) {
	ctx := context.Background()
	idx := NewInMemoryDocumentIndex(4)
	_, err := idx.Upsert(ctx, []documents.Document{
		documents.New("alpha", nil).WithID("id-1"),
		documents.New("beta", nil).WithID("id-2"),
	})
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}

	resp, err := idx.Delete(ctx, []string{"id-1", "missing", "id-2"})
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if resp.NumDeleted != 2 || resp.NumFailed != 0 {
		t.Fatalf("delete response = %+v, want NumDeleted=2 NumFailed=0", resp)
	}
	if len(resp.Succeeded) != 2 || len(resp.Failed) != 0 {
		t.Fatalf("delete response = %+v, want 2 succeeded and 0 failed", resp)
	}

	got, err := idx.Get(ctx, []string{"id-1", "id-2"})
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("get returned %d docs after delete, want 0", len(got))
	}
}

func TestInMemoryDocumentIndexDeleteNilErrors(t *testing.T) {
	ctx := context.Background()
	idx := NewInMemoryDocumentIndex(4)

	_, err := idx.Delete(ctx, nil)
	if err == nil {
		t.Fatalf("delete with nil ids returned nil error, want error")
	}
}

func TestInMemoryDocumentIndexGetReturnsPresentInOrder(t *testing.T) {
	ctx := context.Background()
	idx := NewInMemoryDocumentIndex(4)
	_, err := idx.Upsert(ctx, []documents.Document{
		documents.New("alpha", nil).WithID("id-a"),
		documents.New("charlie", nil).WithID("id-c"),
	})
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}

	got, err := idx.Get(ctx, []string{"id-c", "missing", "id-a"})
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("get returned %d docs, want 2", len(got))
	}
	if got[0].ID != "id-c" || got[1].ID != "id-a" {
		t.Fatalf("get IDs = %q %q, want %q %q (input order)", got[0].ID, got[1].ID, "id-c", "id-a")
	}
}

func TestInMemoryDocumentIndexGetRelevantDocumentsRanksByCount(t *testing.T) {
	ctx := context.Background()
	idx := NewInMemoryDocumentIndex(2)
	_, err := idx.Upsert(ctx, []documents.Document{
		documents.New("apple apple apple", nil).WithID("three"),
		documents.New("apple", nil).WithID("one"),
		documents.New("apple apple", nil).WithID("two"),
		documents.New("banana", nil).WithID("zero"),
	})
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}

	got, err := idx.GetRelevantDocuments(ctx, "apple")
	if err != nil {
		t.Fatalf("get relevant: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d docs, want top_k = 2", len(got))
	}
	if got[0].ID != "three" || got[1].ID != "two" {
		t.Fatalf("ranked IDs = %q %q, want %q %q", got[0].ID, got[1].ID, "three", "two")
	}
}

func TestNewInMemoryDocumentIndexDefaultsTopK(t *testing.T) {
	ctx := context.Background()
	idx := NewInMemoryDocumentIndex(0)
	docs := []documents.Document{
		documents.New("apple one", nil),
		documents.New("apple two", nil),
		documents.New("apple three", nil),
		documents.New("apple four", nil),
		documents.New("apple five", nil),
	}
	if _, err := idx.Upsert(ctx, docs); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	got, err := idx.GetRelevantDocuments(ctx, "apple")
	if err != nil {
		t.Fatalf("get relevant: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("got %d docs, want default top_k = 4", len(got))
	}
}

func TestInMemoryDocumentIndexGetRelevantDocumentsCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	idx := NewInMemoryDocumentIndex(4)

	if _, err := idx.GetRelevantDocuments(ctx, "apple"); err == nil {
		t.Fatal("expected context error")
	}
}
