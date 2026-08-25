package standardtests

import (
	"context"
	"sort"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/indexing"
)

// IndexerFactory creates a fresh, empty document index for standard tests.
type IndexerFactory func(t testing.TB) indexing.DocumentIndex

// indexerTestUUID is the fixed uuid.UUID(int=7) value the Python suite uses
// for documents with explicit IDs.
const indexerTestUUID = "00000000-0000-0000-0000-000000000007"

// RunDocumentIndexerConformance verifies the read-write behavior expected from
// every DocumentIndex implementation. It mirrors DocumentIndexerTestSuite in
// libs/standard-tests/langchain_tests/integration_tests/indexer.py; because Go
// has no sync/async API split (a declared divergence), the sync and async
// Python suites collapse into this single runner. Python's
// test_upsert_documents_has_no_ids is a signature-introspection check that the
// Go DocumentIndex interface enforces at compile time, so it has no runtime
// equivalent here.
func RunDocumentIndexerConformance(t *testing.T, factory IndexerFactory) {
	t.Helper()
	ctx := context.Background()

	t.Run("upsert no ids", func(t *testing.T) { // test_upsert_no_ids
		index := factory(t)
		resp, err := index.Upsert(ctx, []documents.Document{
			documents.New("foo", map[string]any{"id": 1}),
			documents.New("bar", map[string]any{"id": 2}),
		})
		requireNoErr(t, "upsert", err)
		requireLen(t, "failed", resp.Failed, 0)
		requireLen(t, "succeeded", resp.Succeeded, 2)
		ids := append([]string(nil), resp.Succeeded...)
		sort.Strings(ids)

		got, err := index.Get(ctx, ids)
		requireNoErr(t, "get", err)
		requireLen(t, "documents", got, 2)
		idSet := map[string]bool{}
		for _, id := range ids {
			idSet[id] = true
		}
		for _, doc := range got {
			if !idSet[doc.ID] {
				t.Fatalf("document ID %q not in upsert response %v", doc.ID, ids)
			}
		}
		requireIndexedDocument(t, got, "foo", 1)
		requireIndexedDocument(t, got, "bar", 2)
	})

	t.Run("upsert some ids", func(t *testing.T) { // test_upsert_some_ids
		index := factory(t)
		resp, err := index.Upsert(ctx, []documents.Document{
			documents.New("foo", map[string]any{"id": 1}).WithID(indexerTestUUID),
			documents.New("bar", map[string]any{"id": 2}),
		})
		requireNoErr(t, "upsert", err)
		requireLen(t, "failed", resp.Failed, 0)
		requireLen(t, "succeeded", resp.Succeeded, 2)
		otherID := ""
		foundFixed := false
		for _, id := range resp.Succeeded {
			if id == indexerTestUUID {
				foundFixed = true
			} else {
				otherID = id
			}
		}
		if !foundFixed || otherID == "" {
			t.Fatalf("succeeded: got %v, want %q plus one generated id", resp.Succeeded, indexerTestUUID)
		}

		got, err := index.Get(ctx, resp.Succeeded)
		requireNoErr(t, "get", err)
		requireLen(t, "documents", got, 2)
		requireIndexedDocumentByID(t, got, indexerTestUUID, "foo", 1)
		requireIndexedDocumentByID(t, got, otherID, "bar", 2)
	})

	t.Run("upsert overwrites", func(t *testing.T) { // test_upsert_overwrites
		index := factory(t)
		resp, err := index.Upsert(ctx, []documents.Document{
			documents.New("foo", map[string]any{"bar": 1}).WithID(indexerTestUUID),
		})
		requireNoErr(t, "upsert", err)
		requireLen(t, "failed", resp.Failed, 0)
		requireDeepEqual(t, "succeeded", resp.Succeeded, []string{indexerTestUUID})

		got, err := index.Get(ctx, resp.Succeeded)
		requireNoErr(t, "get", err)
		requireLen(t, "documents", got, 1)
		requireEqual(t, "page content", got[0].PageContent, "foo")
		requireEqual(t, "metadata bar", got[0].Metadata["bar"], 1)

		_, err = index.Upsert(ctx, []documents.Document{
			documents.New("foo2", map[string]any{"meow": 2}).WithID(indexerTestUUID),
		})
		requireNoErr(t, "overwrite upsert", err)
		got, err = index.Get(ctx, []string{indexerTestUUID})
		requireNoErr(t, "get after overwrite", err)
		requireLen(t, "documents after overwrite", got, 1)
		requireEqual(t, "overwritten page content", got[0].PageContent, "foo2")
		requireEqual(t, "overwritten metadata meow", got[0].Metadata["meow"], 2)
		requireEqual(t, "overwritten metadata key count", len(got[0].Metadata), 1)
	})

	t.Run("delete missing docs", func(t *testing.T) { // test_delete_missing_docs
		index := factory(t)
		got, err := index.Get(ctx, []string{"1"})
		requireNoErr(t, "get", err)
		requireLen(t, "documents", got, 0)

		resp, err := index.Delete(ctx, []string{"1"})
		requireNoErr(t, "delete", err)
		// Deleting a missing ID is **not** a failure.
		requireEqual(t, "num deleted", resp.NumDeleted, 0)
		requireEqual(t, "num failed", resp.NumFailed, 0)
		requireLen(t, "succeeded", resp.Succeeded, 0)
		requireLen(t, "failed", resp.Failed, 0)
	})

	t.Run("delete semantics", func(t *testing.T) { // test_delete_semantics
		index := factory(t)
		resp, err := index.Upsert(ctx, []documents.Document{
			documents.New("foo", map[string]any{}).WithID(indexerTestUUID),
		})
		requireNoErr(t, "upsert", err)
		requireDeepEqual(t, "succeeded", resp.Succeeded, []string{indexerTestUUID})
		requireLen(t, "failed", resp.Failed, 0)

		delResp, err := index.Delete(ctx, []string{"missing_id", indexerTestUUID})
		requireNoErr(t, "delete", err)
		requireEqual(t, "num deleted", delResp.NumDeleted, 1)
		requireEqual(t, "num failed", delResp.NumFailed, 0)
		requireDeepEqual(t, "delete succeeded", delResp.Succeeded, []string{indexerTestUUID})
		requireLen(t, "delete failed", delResp.Failed, 0)
	})

	t.Run("bulk delete", func(t *testing.T) { // test_bulk_delete
		index := factory(t)
		_, err := index.Upsert(ctx, []documents.Document{
			documents.New("foo", map[string]any{"id": 1}).WithID("1"),
			documents.New("bar", map[string]any{"id": 2}).WithID("2"),
			documents.New("baz", map[string]any{"id": 3}).WithID("3"),
		})
		requireNoErr(t, "upsert", err)
		_, err = index.Delete(ctx, []string{"1", "2"})
		requireNoErr(t, "delete", err)

		got, err := index.Get(ctx, []string{"1", "2", "3"})
		requireNoErr(t, "get", err)
		requireLen(t, "documents", got, 1)
		requireEqual(t, "remaining id", got[0].ID, "3")
		requireEqual(t, "remaining page content", got[0].PageContent, "baz")
		requireEqual(t, "remaining metadata id", got[0].Metadata["id"], 3)
	})

	t.Run("delete no args", func(t *testing.T) { // test_delete_no_args
		index := factory(t)
		// Python requires delete() with no IDs to raise ValueError; the Go
		// equivalent is a non-nil error on a nil ids slice.
		if _, err := index.Delete(ctx, nil); err == nil {
			t.Fatalf("delete with nil ids: expected an error, got nil")
		}
	})

	t.Run("delete missing content", func(t *testing.T) { // test_delete_missing_content
		index := factory(t)
		_, err := index.Delete(ctx, []string{"1"})
		requireNoErr(t, "delete single missing", err)
		_, err = index.Delete(ctx, []string{"1", "2", "3"})
		requireNoErr(t, "delete multiple missing", err)
	})

	t.Run("get with missing ids", func(t *testing.T) { // test_get_with_missing_ids
		index := factory(t)
		resp, err := index.Upsert(ctx, []documents.Document{
			documents.New("foo", map[string]any{"id": 1}).WithID("1"),
			documents.New("bar", map[string]any{"id": 2}).WithID("2"),
		})
		requireNoErr(t, "upsert", err)
		requireDeepEqual(t, "succeeded", resp.Succeeded, []string{"1", "2"})
		requireLen(t, "failed", resp.Failed, 0)

		got, err := index.Get(ctx, []string{"1", "2", "3", "4"})
		requireNoErr(t, "get", err)
		requireLen(t, "documents", got, 2)
		sort.Slice(got, func(i, j int) bool { return got[i].ID < got[j].ID })
		requireEqual(t, "first id", got[0].ID, "1")
		requireEqual(t, "first page content", got[0].PageContent, "foo")
		requireEqual(t, "first metadata id", got[0].Metadata["id"], 1)
		requireEqual(t, "second id", got[1].ID, "2")
		requireEqual(t, "second page content", got[1].PageContent, "bar")
		requireEqual(t, "second metadata id", got[1].Metadata["id"], 2)
	})

	t.Run("get missing", func(t *testing.T) { // test_get_missing
		index := factory(t)
		got, err := index.Get(ctx, []string{"1", "2", "3"})
		requireNoErr(t, "get", err)
		requireLen(t, "documents", got, 0)
	})
}

// requireIndexedDocument finds a stored document by page content and checks
// that it has a non-empty ID and the expected numeric metadata id.
func requireIndexedDocument(t *testing.T, docs []documents.Document, pageContent string, metadataID int) {
	t.Helper()
	for _, doc := range docs {
		if doc.PageContent != pageContent {
			continue
		}
		if doc.ID == "" {
			t.Fatalf("document %q has no ID", pageContent)
		}
		requireEqual(t, "metadata id of "+pageContent, doc.Metadata["id"], any(metadataID))
		return
	}
	t.Fatalf("document %q not found in %#v", pageContent, docs)
}

// requireIndexedDocumentByID finds a stored document by ID and checks its
// page content and numeric metadata id.
func requireIndexedDocumentByID(t *testing.T, docs []documents.Document, id string, pageContent string, metadataID int) {
	t.Helper()
	for _, doc := range docs {
		if doc.ID != id {
			continue
		}
		requireEqual(t, "page content of "+id, doc.PageContent, pageContent)
		requireEqual(t, "metadata id of "+id, doc.Metadata["id"], any(metadataID))
		return
	}
	t.Fatalf("document %q not found in %#v", id, docs)
}
