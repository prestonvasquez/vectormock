package vectormock

import (
	"crypto/rand"
	"fmt"
	"math/big"
)

// newNormalizedFloat32 will generate a random float32 in [-1, 1].
func newNormalizedFloat32() (float32, error) {
	max := big.NewInt(1 << 24)

	n, err := rand.Int(rand.Reader, max)
	if err != nil {
		return 0.0, fmt.Errorf("failed to normalize float32")
	}

	return 2.0*(float32(n.Int64())/float32(1<<24)) - 1.0, nil
}

// newNormalizedVector will generate a random vector of float32s in [-1, 1].
func newNormalizedVector(dim int) []float32 {
	vector := make([]float32, dim)
	for i := range vector {
		vector[i], _ = newNormalizedFloat32()
	}

	return vector
}

// dotProduct will return the dot product between two slices of f32.
func dotProduct(v1, v2 []float32) (sum float32) {
	for i := range v1 {
		sum += v1[i] * v2[i]
	}

	return
}

// Use Gram Schmidt to return a vector orthogonal to the basis, so long as
// the vectors in the basis are linearly independent.
func newOrthogonalVector(dim int, basis ...[]float32) []float32 {
	candidate := newNormalizedVector(dim)

	for _, b := range basis {
		dp := dotProduct(candidate, b)
		basisNorm := dotProduct(b, b)

		for i := range candidate {
			candidate[i] -= (dp / basisNorm) * b[i]
		}
	}

	return candidate
}

// Make n linearly independent vectors of size dim.
func newLinearlyIndependentVectors(n int, dim int) [][]float32 {
	vectors := [][]float32{}

	for i := 0; i < n; i++ {
		v := newOrthogonalVector(dim, vectors...)

		vectors = append(vectors, v)
	}

	return vectors
}

// linearlyIndependent true if the vectors are linearly independent
func linearlyIndependent(v1, v2 []float32) bool {
	var ratio float32

	for i := range v1 {
		if v1[i] != 0 {
			r := v2[i] / v1[i]

			if ratio == 0 {
				ratio = r

				continue
			}

			if r == ratio {
				continue
			}

			return true
		}

		if v2[i] != 0 {
			return true
		}
	}

	return false
}

// Update the basis vector such that qvector * basis = 2S - 1.
func newScoreVector(S float32, qvector []float32, basis []float32) []float32 {
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
		return newScoreVector(S, qvector, basis)
	}

	return basis
}
