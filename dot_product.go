package vectormock

import (
	"context"
)

// DotProduct is an embedder that uses the dot product to calculate the
// similarity between vectors.
type DotProduct struct {
	queryVector []float32
	docs        map[string]Document
	docVectors  map[string][]float32
}

// NewDotProduct will return a new DotProduct embedder with the given dimension.
func NewDotProduct(dim int) *DotProduct {
	return &DotProduct{
		queryVector: newNormalizedVector(dim),
	}
}

// AddDocuments will add the given documents to the embedder, assigning each
// a vector such that similarity score = 0.5 * ( 1 + vector * queryVector).
func (emb *DotProduct) MockDocuments(doc ...Document) error {
	for _, d := range doc {
		emb.docs[d.PageContent] = d
	}

	return nil
}

// existingVectors returns all the vectors that have been added to the embedder.
// The query vector is included in the list to maintian orthogonality.
func (emb *DotProduct) existingVectors() [][]float32 {
	vectors := make([][]float32, 0, len(emb.docs)+1)
	for _, vec := range emb.docVectors {
		vectors = append(vectors, vec)
	}

	return append(vectors, emb.queryVector)
}

// EmbedDocuments will return the embedded vectors for the given texts. If the
// text does not exist in the document set, a zero vector will be returned.
func (emb *DotProduct) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range vectors {
		// If the text does not exist in the document set, return a zero vector.
		doc, ok := emb.docs[texts[i]]
		if !ok {
			vectors[i] = make([]float32, len(emb.queryVector))
		}

		// If the vector exists, use it.
		existing, ok := emb.docVectors[texts[i]]
		if ok {
			vectors[i] = existing

			continue
		}

		// If it does not exist, make a linearly independent vector.
		newVectorBasis := newOrthogonalVector(len(emb.queryVector), emb.existingVectors()...)

		// Update the newVector to be scaled by the score.
		newVector := newScoreVector(doc.Score, emb.queryVector, newVectorBasis)

		vectors[i] = newVector
		emb.docVectors[texts[i]] = newVector
	}

	return vectors, nil
}

// EmbedQuery returns the query vector.
func (emb *DotProduct) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return emb.queryVector, nil
}
