package vectormock

import "context"

type Store interface {
	AddDocuments(ctx context.Context, docs []Document, emb Embedder) ([]string, error)
}
