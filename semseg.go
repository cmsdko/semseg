// Package semseg provides tools for semantically splitting text into meaningful chunks.
package semseg

import (
	"errors"
	"strings"

	"github.com/cmsdko/semseg/internal/text"
	"github.com/cmsdko/semseg/internal/tfidf"
)

// Chunk represents a single segment of the original text.
type Chunk struct {
	// Text is the combined string of all sentences in the chunk.
	Text string
	// Sentences are the original sentences that make up this chunk.
	Sentences []string
	// NumTokens is the number of tokens in this chunk, calculated by the simple tokenizer.
	NumTokens int
}

// Options configures the segmentation process.
type Options struct {
	// MaxTokens is a hard limit for the number of tokens in a chunk.
	// The library guarantees that no chunk will exceed this size, unless a single
	// sentence is larger than the limit, in which case it will be in its own chunk.
	// This field is required.
	MaxTokens int

	// MinSplitSimilarity (range 0.0 to 1.0) is the cosine similarity threshold.
	// A gap between sentences with similarity below this value is considered a potential split point.
	// If set to 0 (default), the algorithm will dynamically find local minima (recommended).
	MinSplitSimilarity float64

	// DepthThreshold (range 0.0 to 1.0) is used when MinSplitSimilarity is 0.
	// It defines the minimum "depth" of a similarity dip to be considered a valid split point.
	// This helps filter out minor, insignificant fluctuations.
	// Default: 0.1. To force zero (i.e., accept any local minimum), set to 0 explicitly.
	// If set to a negative value, the library will use the default (0.1).
	DepthThreshold float64
}

// Segment splits a given text into semantic chunks based on the provided options.
func Segment(textStr string, opts Options) ([]Chunk, error) {
	if err := validateOptions(opts); err != nil {
		return nil, err
	}
	setDefaultOptions(&opts)

	sentences := text.SplitSentences(textStr)
	if len(sentences) == 0 {
		return []Chunk{}, nil
	}
	if len(sentences) == 1 {
		tokens := text.Tokenize(sentences[0])
		return []Chunk{makeChunk(sentences, len(tokens))}, nil
	}

	tokenizedSentences := make([][]string, len(sentences))
	tokenCounts := make([]int, len(sentences))
	for i, s := range sentences {
		tokens := text.Tokenize(s)
		tokenizedSentences[i] = tokens
		tokenCounts[i] = len(tokens)
	}

	corpus := tfidf.NewCorpus(tokenizedSentences)
	vectors := make([]map[string]float64, len(sentences))
	for i, ts := range tokenizedSentences {
		vectors[i] = corpus.Vectorize(ts)
	}

	scores := calculateCohesion(vectors)
	boundaryIndices := findBoundaries(scores, opts)

	return buildChunks(sentences, tokenCounts, boundaryIndices, opts.MaxTokens), nil
}

func validateOptions(opts Options) error {
	if opts.MaxTokens <= 0 {
		return errors.New("MaxTokens must be a positive number")
	}
	return nil
}

func setDefaultOptions(opts *Options) {
	// Only apply default when user requested "use default" via negative value.
	// 0 is a valid explicit setting and must not be overridden.
	if opts.MinSplitSimilarity == 0 && opts.DepthThreshold < 0 {
		opts.DepthThreshold = 0.1
	}
}

// calculateCohesion computes the cosine similarity between adjacent sentence vectors.
func calculateCohesion(vectors []map[string]float64) []float64 {
	scores := make([]float64, len(vectors)-1)
	for i := 0; i < len(vectors)-1; i++ {
		scores[i] = tfidf.CosineSimilarity(vectors[i], vectors[i+1])
	}
	return scores
}

// findBoundaries identifies the indices where splits should occur.
func findBoundaries(scores []float64, opts Options) map[int]bool {
	boundaries := make(map[int]bool)
	if len(scores) == 0 {
		return boundaries
	}

	for i := 0; i < len(scores); i++ {
		// Fixed threshold method
		if opts.MinSplitSimilarity > 0 {
			if scores[i] < opts.MinSplitSimilarity {
				boundaries[i] = true
			}
			continue
		}

		// Local minima method (default)
		if i > 0 && i < len(scores)-1 {
			isLocalMinimum := scores[i] < scores[i-1] && scores[i] < scores[i+1]
			if isLocalMinimum {
				depth := (scores[i-1]+scores[i+1])/2 - scores[i]
				if depth >= opts.DepthThreshold {
					boundaries[i] = true
				}
			}
		}
	}
	return boundaries
}

// buildChunks constructs the final chunks, strictly respecting MaxTokens.
func buildChunks(
	sentences []string,
	tokenCounts []int,
	boundaryIndices map[int]bool,
	maxTokens int,
) []Chunk {
	var chunks []Chunk
	currentChunkSentences := []string{}
	currentChunkTokens := 0

	for i, sentence := range sentences {
		sentenceTokens := tokenCounts[i]

		if sentenceTokens > maxTokens {
			if len(currentChunkSentences) > 0 {
				chunks = append(chunks, makeChunk(currentChunkSentences, currentChunkTokens))
			}
			chunks = append(chunks, makeChunk([]string{sentence}, sentenceTokens))
			currentChunkSentences = []string{}
			currentChunkTokens = 0
			continue
		}

		isSemanticBoundary := i > 0 && boundaryIndices[i-1]
		tokenLimitExceeded := currentChunkTokens+sentenceTokens > maxTokens

		if len(currentChunkSentences) > 0 && (isSemanticBoundary || tokenLimitExceeded) {
			chunks = append(chunks, makeChunk(currentChunkSentences, currentChunkTokens))
			currentChunkSentences = []string{}
			currentChunkTokens = 0
		}

		currentChunkSentences = append(currentChunkSentences, sentence)
		currentChunkTokens += sentenceTokens
	}

	if len(currentChunkSentences) > 0 {
		chunks = append(chunks, makeChunk(currentChunkSentences, currentChunkTokens))
	}

	return chunks
}

func makeChunk(sentences []string, numTokens int) Chunk {
	return Chunk{
		Text:      strings.Join(sentences, " "),
		Sentences: sentences,
		NumTokens: numTokens,
	}
}
