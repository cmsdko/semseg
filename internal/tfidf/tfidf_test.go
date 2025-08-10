package tfidf

import (
	"math"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	v1 := map[string]float64{"a": 1, "b": 2, "c": 3}
	v2 := map[string]float64{"a": 1, "b": 2, "c": 3}
	v3 := map[string]float64{"d": 1, "e": 2, "f": 3}
	v4 := map[string]float64{"a": 1, "c": 3}

	// Similarity with self should be 1
	if sim := CosineSimilarity(v1, v2); math.Abs(sim-1.0) > 1e-9 {
		t.Errorf("Expected similarity of 1.0, got %f", sim)
	}

	// Similarity with orthogonal vector should be 0
	if sim := CosineSimilarity(v1, v3); sim != 0.0 {
		t.Errorf("Expected similarity of 0.0, got %f", sim)
	}

	// General case
	expectedSim := (1*1 + 3*3) / (math.Sqrt(1*1+2*2+3*3) * math.Sqrt(1*1+3*3))
	if sim := CosineSimilarity(v1, v4); math.Abs(sim-expectedSim) > 1e-9 {
		t.Errorf("Expected similarity of %f, got %f", expectedSim, sim)
	}
}

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
	// "is" is common, should have lower score than "sun" or "hot"
	idfIs := math.Log(2.0 / (1.0 + 2.0))
	idfSun := math.Log(2.0 / (1.0 + 1.0))

	if vec["is"] >= vec["sun"] {
		t.Errorf("Expected score for 'is' (%f) to be less than for 'sun' (%f)", vec["is"], vec["sun"])
	}
	if math.Abs(vec["is"]-(1.0/3.0)*idfIs) > 1e-9 {
		t.Errorf("Wrong TF-IDF score for 'is'")
	}
	if math.Abs(vec["sun"]-(1.0/3.0)*idfSun) > 1e-9 {
		t.Errorf("Wrong TF-IDF score for 'sun'")
	}
}
