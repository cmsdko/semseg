// ./semseg.go
// Package semseg provides tools for semantically splitting text into meaningful chunks.
package semseg

import (
	"errors"
	"math"
	"strings"

	"github.com/cmsdko/semseg/internal/lang"
	"github.com/cmsdko/semseg/internal/text"
	"github.com/cmsdko/semseg/internal/tfidf"
)

// Constants for LanguageDetectionMode.
const (
	// LangDetectModeFirstSentence detects the language from the first sentence only. This is the default mode.
	LangDetectModeFirstSentence = "first_sentence"
	// LangDetectModeFirstTenSentences detects the language from the first 10 sentences.
	LangDetectModeFirstTenSentences = "first_ten_sentences"
	// LangDetectModePerSentence detects the language for each sentence individually.
	LangDetectModePerSentence = "per_sentence"
	// LangDetectModeFullText detects the language from the entire input text.
	LangDetectModeFullText = "full_text"
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

	// Language forces the segmenter to use a specific language (e.g., "english", "russian"),
	// skipping the language auto-detection step. If set, this option overrides LanguageDetectionMode.
	// If empty (default), the language will be auto-detected based on LanguageDetectionMode.
	Language string

	// LanguageDetectionMode specifies the strategy for automatic language detection when Language is not set.
	// Use one of the LangDetectMode* constants (e.g., LangDetectModeFirstSentence).
	// If empty, it defaults to LangDetectModeFirstSentence.
	LanguageDetectionMode string

	// LanguageDetectionTokens enables early detection by the first N tokens.
	// If > 0 and Language is empty and mode is not per-sentence, language is detected
	// from the first N tokens before sentence splitting.
	LanguageDetectionTokens int

	// PreNormalizeAbbreviations controls whether dotted abbreviations/acronyms are normalized
	// (dots removed) before sentence splitting. If nil (default), it is treated as true.
	PreNormalizeAbbreviations *bool

	// EnableStopWordRemoval controls whether common "noise words" are removed before similarity calculation.
	// If nil (default), it is treated as true.
	EnableStopWordRemoval *bool

	// EnableStemming controls whether token stemming (reducing words to their root form) is applied.
	// If nil (default), it is treated as true.
	EnableStemming *bool
}

// Segment splits a given text into semantic chunks based on the provided options.
func Segment(textStr string, opts Options) ([]Chunk, error) {
	if err := validateOptions(opts); err != nil {
		return nil, err
	}
	setDefaultOptions(&opts)

	// --- Early language selection (explicit or by first N tokens) before any normalization/splitting ---
	var globalDetectedLang string
	if opts.Language != "" {
		globalDetectedLang = opts.Language
	} else if opts.LanguageDetectionTokens > 0 && opts.LanguageDetectionMode != LangDetectModePerSentence {
		toks := text.Tokenize(textStr)
		n := opts.LanguageDetectionTokens
		if n > len(toks) {
			n = len(toks)
		}
		// Reuse string-based detector for simplicity.
		globalDetectedLang = lang.DetectLanguage(strings.Join(toks[:n], " "))
	}

	// --- Optional abbreviation normalization before sentence splitting ---
	if *opts.PreNormalizeAbbreviations {
		// Use already chosen language if available, otherwise empty which falls back inside.
		textStr = lang.NormalizeAbbreviations(textStr, globalDetectedLang)
	}

	// Split into sentences after normalization.
	sentences := text.SplitSentences(textStr)
	if len(sentences) == 0 {
		return []Chunk{}, nil
	}
	if len(sentences) == 1 {
		tokens := text.Tokenize(sentences[0])
		return []Chunk{makeChunk(sentences, len(tokens))}, nil
	}

	// If no language yet and mode is not per-sentence, follow the original strategy.
	if globalDetectedLang == "" && opts.LanguageDetectionMode != LangDetectModePerSentence {
		switch opts.LanguageDetectionMode {
		case LangDetectModeFirstSentence:
			globalDetectedLang = lang.DetectLanguage(sentences[0])
		case LangDetectModeFirstTenSentences:
			end := 10
			if len(sentences) < 10 {
				end = len(sentences)
			}
			textForDetection := strings.Join(sentences[:end], " ")
			globalDetectedLang = lang.DetectLanguage(textForDetection)
		case LangDetectModeFullText:
			globalDetectedLang = lang.DetectLanguage(textStr)
		default:
			globalDetectedLang = lang.DetectLanguage(sentences[0])
		}
	}

	tokenizedSentences := make([][]string, len(sentences))
	tokenCounts := make([]int, len(sentences))
	for i, s := range sentences {
		originalTokens := text.Tokenize(s)
		tokenCounts[i] = len(originalTokens)

		var detectedLang string
		if opts.LanguageDetectionMode == LangDetectModePerSentence && opts.Language == "" {
			detectedLang = lang.DetectLanguage(s)
		} else {
			detectedLang = globalDetectedLang
		}

		sentenceForSimilarity := s
		if *opts.EnableStopWordRemoval {
			sentenceForSimilarity = lang.RemoveStopWords(sentenceForSimilarity, detectedLang)
		}

		tokensForSimilarity := text.Tokenize(sentenceForSimilarity)

		if *opts.EnableStemming {
			tokensForSimilarity = lang.StemTokens(tokensForSimilarity, detectedLang)
		}

		tokenizedSentences[i] = tokensForSimilarity
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
	// Default for DepthThreshold when MinSplitSimilarity is 0
	if opts.MinSplitSimilarity == 0 && opts.DepthThreshold < 0 {
		opts.DepthThreshold = 0.1
	}

	// Default for LanguageDetectionMode
	if opts.Language == "" && opts.LanguageDetectionMode == "" {
		opts.LanguageDetectionMode = LangDetectModeFirstSentence
	}

	// Default boolean flags
	if opts.EnableStopWordRemoval == nil {
		t := true
		opts.EnableStopWordRemoval = &t
	}
	if opts.EnableStemming == nil {
		t := true
		opts.EnableStemming = &t
	}
	if opts.PreNormalizeAbbreviations == nil {
		t := true
		opts.PreNormalizeAbbreviations = &t
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

// findBoundaries identifies sentence boundaries based on similarity scores and options
func findBoundaries(sentences []string, scores []float64, opts Options) []bool {
	boundaries := make([]bool, len(scores))

	for i := 1; i < len(scores)-1; i++ {
		// Check for local minimum
		if scores[i] <= scores[i-1] && scores[i] <= scores[i+1] {
			// Apply thresholds
			if (opts.MinSplitSimilarity == 0 || scores[i] <= opts.MinSplitSimilarity) &&
				(opts.DepthThreshold == 0 || (math.Min(scores[i-1], scores[i+1])-scores[i] >= opts.DepthThreshold)) {
				boundaries[i] = true
			}
		}
	}

	// Fallback: if no local minima found and thresholds disabled, cut at global minimum
	if opts.MinSplitSimilarity == 0 && opts.DepthThreshold == 0 && len(scores) > 0 {
		found := false
		for _, b := range boundaries {
			if b {
				found = true
				break
			}
		}
		if !found {
			minIdx := 0
			minVal := scores[0]
			for i := 1; i < len(scores); i++ {
				// use <= to make behavior stable on flat minima
				if scores[i] <= minVal {
					minVal = scores[i]
					minIdx = i
				}
			}
			boundaries[minIdx] = true
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
