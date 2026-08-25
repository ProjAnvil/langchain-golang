package indexing

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// indexDestination is the resolved write/delete target of an indexing run:
// either a vectorstores.VectorStore or a DocumentIndex. It mirrors Python's
// `index(..., vector_store: VectorStore | DocumentIndex)` dispatch
// (indexing/api.py:296-308, 424-450).
type indexDestination interface {
	addDocuments(ctx context.Context, docs []documents.Document, kwargs map[string]any) error
	deleteIDs(ctx context.Context, ids []string) error
}

// KwargAdder is implemented by VectorStores that accept extra add kwargs,
// receiving Options.UpsertKwargs the way Python passes upsert_kwargs to
// add_documents (indexing/api.py:535-540). Python's batch_size kwarg is not
// forwarded: the Go VectorStore.AddDocuments signature has no batching knob.
type KwargAdder interface {
	AddDocumentsWithKwargs(ctx context.Context, docs []documents.Document, kwargs map[string]any) ([]string, error)
}

// KwargUpserter is implemented by DocumentIndex destinations that accept
// extra upsert kwargs, receiving Options.UpsertKwargs the way Python passes
// upsert_kwargs to upsert (indexing/api.py:542-545).
type KwargUpserter interface {
	UpsertWithKwargs(ctx context.Context, items []documents.Document, kwargs map[string]any) (UpsertResponse, error)
}

// resolveDestination mirrors Python's isinstance dispatch (VectorStore first,
// then DocumentIndex; indexing/api.py:427-450).
func resolveDestination(raw any) (indexDestination, error) {
	switch typed := raw.(type) {
	case nil:
		return nil, fmt.Errorf("vector store is required")
	case vectorstores.VectorStore:
		return vectorStoreDestination{store: typed}, nil
	case DocumentIndex:
		return documentIndexDestination{index: typed}, nil
	default:
		return nil, fmt.Errorf(
			"indexing: destination should be either a vectorstores.VectorStore or an indexing.DocumentIndex; got %T",
			raw)
	}
}

type vectorStoreDestination struct {
	store vectorstores.VectorStore
}

func (d vectorStoreDestination) addDocuments(ctx context.Context, docs []documents.Document, kwargs map[string]any) error {
	if len(kwargs) > 0 {
		adder, ok := d.store.(KwargAdder)
		if !ok {
			return fmt.Errorf(
				"indexing: UpsertKwargs provided but vector store %T does not implement indexing.KwargAdder",
				d.store)
		}
		_, err := adder.AddDocumentsWithKwargs(ctx, docs, kwargs)
		return err
	}
	_, err := d.store.AddDocuments(ctx, docs)
	return err
}

func (d vectorStoreDestination) deleteIDs(ctx context.Context, ids []string) error {
	return d.store.Delete(ctx, ids)
}

type documentIndexDestination struct {
	index DocumentIndex
}

func (d documentIndexDestination) addDocuments(ctx context.Context, docs []documents.Document, kwargs map[string]any) error {
	var response UpsertResponse
	var err error
	if len(kwargs) > 0 {
		upserter, ok := d.index.(KwargUpserter)
		if !ok {
			return fmt.Errorf(
				"indexing: UpsertKwargs provided but document index %T does not implement indexing.KwargUpserter",
				d.index)
		}
		response, err = upserter.UpsertWithKwargs(ctx, docs, kwargs)
	} else {
		response, err = d.index.Upsert(ctx, docs)
	}
	if err != nil {
		return err
	}
	// Python's index() is pessimistic about upsert failures and surfaces them
	// via IndexingException on the delete path; for upserts, failed IDs abort
	// the run the same way a VectorStore AddDocuments error does.
	if len(response.Failed) > 0 {
		return fmt.Errorf("indexing: the upsert operation to DocumentIndex failed for %d document(s)", len(response.Failed))
	}
	return nil
}

func (d documentIndexDestination) deleteIDs(ctx context.Context, ids []string) error {
	response, err := d.index.Delete(ctx, ids)
	if err != nil {
		return err
	}
	// Mirrors Python's _delete (indexing/api.py:267-271): a DocumentIndex
	// delete with num_failed > 0 raises IndexingException.
	if response.NumFailed > 0 {
		return fmt.Errorf("indexing: the delete operation to DocumentIndex failed (%d failures)", response.NumFailed)
	}
	return nil
}
