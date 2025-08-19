package tfidf

import "math"

type corpus struct {
	docFrequencies map[string]int
	numDocs        int
}

func NewCorpus(documents [][]string) *corpus {
	docFrequencies := make(map[string]int)
	for _, doc := range documents {
		seenWords := make(map[string]bool)
		for _, word := range doc {
			if !seenWords[word] {
				docFrequencies[word]++
				seenWords[word] = true
			}
		}
	}
	return &corpus{
		docFrequencies: docFrequencies,
		numDocs:        len(documents),
	}
}

func (c *corpus) Vectorize(tokens []string) map[string]float64 {
	if len(tokens) == 0 {
		return make(map[string]float64)
	}

	// Calculate Term Frequency (TF) - normalized by document length
	tf := make(map[string]float64)
	for _, token := range tokens {
		tf[token]++
	}
	numTokens := float64(len(tokens))
	for token, count := range tf {
		tf[token] = count / numTokens
	}

	// Calculate TF-IDF vector
	vector := make(map[string]float64)
	for token, termFreq := range tf {
		// Modified IDF formula with smoothing:
		// - Add 1 to numerator to prevent log(0) when all docs contain the term
		// - Add 1 to denominator to prevent division by 0 for unseen terms
		// - This smoothing makes the algorithm more robust for small corpora
		idf := math.Log(1 + (float64(c.numDocs) / (1 + float64(c.docFrequencies[token]))))
		vector[token] = termFreq * idf
	}
	return vector
}

func CosineSimilarity(v1, v2 map[string]float64) float64 {
	dotProduct, normA, normB := 0.0, 0.0, 0.0
	allWords := make(map[string]bool)
	for word := range v1 {
		allWords[word] = true
	}
	for word := range v2 {
		allWords[word] = true
	}
	for word := range allWords {
		x, y := v1[word], v2[word]
		dotProduct += x * y
		normA += x * x
		normB += y * y
	}
	if normA == 0 || normB == 0 {
		return 0.0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
