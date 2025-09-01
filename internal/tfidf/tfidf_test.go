package tfidf

import (
	"math"
	"testing"
)

// TestCosineSimilarity covers basic similarity cases:
// - identical vectors → similarity = 1
// - orthogonal vectors (no shared terms) → similarity = 0
// - partial overlap → expected cosine value
func TestCosineSimilarity(t *testing.T) {
	v1 := map[string]float64{"a": 1, "b": 2, "c": 3}
	v2 := map[string]float64{"a": 1, "b": 2, "c": 3}
	v3 := map[string]float64{"d": 1, "e": 2, "f": 3}
	v4 := map[string]float64{"a": 1, "c": 3}

	// Identical vectors
	if sim := CosineSimilarity(v1, v2); math.Abs(sim-1.0) > 1e-9 {
		t.Errorf("Expected similarity of 1.0, got %f", sim)
	}

	// Orthogonal vectors (no overlap)
	if sim := CosineSimilarity(v1, v3); sim != 0.0 {
		t.Errorf("Expected similarity of 0.0, got %f", sim)
	}

	// Partial overlap: manual cosine similarity check
	expectedSim := (1*1 + 3*3) / (math.Sqrt(1*1+2*2+3*3) * math.Sqrt(1*1+3*3))
	if sim := CosineSimilarity(v1, v4); math.Abs(sim-expectedSim) > 1e-9 {
		t.Errorf("Expected similarity of %f, got %f", expectedSim, sim)
	}
}

// TestCorpusVectorize checks TF-IDF vectorization:
// - Common terms get lower weights (IDF down-weighting)
// - Rare terms get higher weights
// - Formula matches expected values with smoothing
func TestCorpusVectorize(t *testing.T) {
	docs := [][]string{
		{"sun", "is", "hot"},
		{"moon", "is", "cold"},
	}
	corpus := NewCorpus(docs)

	vec := corpus.Vectorize([]string{"sun", "is", "hot"})
	if len(vec) != 3 {
		t.Errorf("Expected vector of length 3, got %d", len(vec))
	}

	// Check IDF weighting: "is" appears in both docs, "sun" only in one.
	idfIs := math.Log(1.0 + 2.0/(1.0+2.0))  // log(1 + 2/3)
	idfSun := math.Log(1.0 + 2.0/(1.0+1.0)) // log(1 + 1) = log(2)

	// "is" should be down-weighted compared to "sun"
	if vec["is"] >= vec["sun"] {
		t.Errorf("Expected score for 'is' (%f) to be less than for 'sun' (%f)", vec["is"], vec["sun"])
	}

	// Verify smoothed TF-IDF values
	if math.Abs(vec["is"]-(1.0/3.0)*idfIs) > 1e-9 {
		t.Errorf("Wrong TF-IDF score for 'is'")
	}
	if math.Abs(vec["sun"]-(1.0/3.0)*idfSun) > 1e-9 {
		t.Errorf("Wrong TF-IDF score for 'sun'")
	}
}
