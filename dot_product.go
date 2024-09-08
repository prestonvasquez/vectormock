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
	normFn      DotProductNormFn
}

// DotProductNormFn is a function that will return a new vector
type DotProductNormFn func(S float32, qvector, basis []float32) []float32

// DefaultDotProductNormFn will return a new vector such that v1 * v2 = 2S - 1.
func DefaultDotProductNormFn(S float32, qvector, basis []float32) []float32 {
	var sum float32

	// Populate v2 upto dim-1.
	for i := 0; i < len(qvector)-1; i++ {
		sum += qvector[i] * basis[i]
	}

	// Calculate v_{2, dim} such that v1 * v2 = 2S - 1:
	basis[len(basis)-1] = (2*S - 1 - sum) / qvector[len(qvector)-1]

	// If the vectors are linearly independent, regenerate the dim-1 elements
	// of v2.
	if !linearlyIndependent(qvector, basis) {
		return DefaultDotProductNormFn(S, qvector, basis)
	}

	return basis
}

// NewDotProduct will return a new DotProduct embedder with the given dimension.
// The default norm function is vk * v = 2S - 1.
func NewDotProduct(dim int) *DotProduct {
	return &DotProduct{
		queryVector: newNormalizedVector(dim),
		docs:        make(map[string]Document),
		docVectors:  make(map[string][]float32),
		normFn:      DefaultDotProductNormFn,
	}
}

// NewDotProductWithNormFn will return a new DotProduct embedder with the given
// dimension and norm function.
func NewDotProductWithNormFn(dim int, fn DotProductNormFn) *DotProduct {
	return &DotProduct{
		queryVector: newNormalizedVector(dim),
		docs:        make(map[string]Document),
		docVectors:  make(map[string][]float32),
		normFn:      fn,
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
		newVector := emb.normFn(doc.Score, emb.queryVector, newVectorBasis)

		vectors[i] = newVector
		emb.docVectors[texts[i]] = newVector
	}

	return vectors, nil
}

// EmbedQuery returns the query vector.
func (emb *DotProduct) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return emb.queryVector, nil
}
