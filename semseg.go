// file: ./semseg.go

// Package semseg provides tools for semantically splitting text into meaningful chunks.
package semseg

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

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

// Constants for Ollama worker pool
const (
	OllamaMaxWorkersEnvVar = "OLLAMA_MAX_WORKERS"
	DefaultOllamaWorkers   = 4
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
	// If nil (default), it is treated as true. This option is ignored if using Ollama embeddings.
	EnableStopWordRemoval *bool

	// EnableStemming controls whether token stemming (reducing words to their root form) is applied.
	// If nil (default), it is treated as true. This option is ignored if using Ollama embeddings.
	EnableStemming *bool

	// HTTPClient allows providing a custom http.Client for Ollama requests.
	// If nil, a new client with a default 60-second timeout will be created for each call.
	// Reusing a single client across multiple calls is highly recommended for performance
	// as it enables TCP connection reuse.
	HTTPClient *http.Client
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

	tokenCounts := make([]int, len(sentences))
	for i, s := range sentences {
		tokenCounts[i] = len(text.Tokenize(s))
	}

	var scores []float64

	// Check environment variables for Ollama
	ollamaURL := os.Getenv("OLLAMA_URL")
	ollamaModel := os.Getenv("OLLAMA_MODEL")

	if ollamaURL != "" && ollamaModel != "" {
		// --- PATH WITH OLLAMA EMBEDDINGS ---
		// Use the http.Client from options if provided; otherwise, create a default one.
		// Reusing a client is significantly more performant.
		client := opts.HTTPClient
		if client == nil {
			client = &http.Client{Timeout: 60 * time.Second}
		}
		vectors, err := getOllamaEmbeddings(sentences, ollamaURL, ollamaModel, client)
		if err != nil {
			return nil, fmt.Errorf("failed to get ollama embeddings: %w", err)
		}
		scores = calculateCohesionDense(vectors)
	} else {
		// --- EXISTING PATH WITH TF-IDF ---
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
		for i, s := range sentences {
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
		scores = calculateCohesion(vectors)
	}

	boundaryIndices := findBoundaries(scores, opts)
	return buildChunks(sentences, tokenCounts, boundaryIndices, opts.MaxTokens), nil
}

// Structures for Ollama request and response
type ollamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaResponse struct {
	Embedding []float64 `json:"embedding"`
	Error     string    `json:"error,omitempty"`
}

// Structures for worker pool
type ollamaJob struct {
	index    int
	sentence string
}

type ollamaResult struct {
	index     int
	embedding []float64
	err       error
}

// getOllamaEmbeddings fetches embeddings for all sentences concurrently using a worker pool.
func getOllamaEmbeddings(sentences []string, ollamaURL, ollamaModel string, client *http.Client) ([][]float64, error) {
	numSentences := len(sentences)
	if numSentences == 0 {
		return [][]float64{}, nil
	}

	// Determine the number of workers from environment variable or default.
	numWorkersStr := os.Getenv(OllamaMaxWorkersEnvVar)
	numWorkers, err := strconv.Atoi(numWorkersStr)
	if err != nil || numWorkers <= 0 {
		numWorkers = DefaultOllamaWorkers
	}
	if numWorkers > numSentences {
		numWorkers = numSentences
	}

	jobs := make(chan ollamaJob, numSentences)
	results := make(chan ollamaResult, numSentences)
	url := strings.TrimSuffix(ollamaURL, "/") + "/api/embeddings"

	var wg sync.WaitGroup
	// Start workers.
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go ollamaWorker(&wg, client, jobs, results, url, ollamaModel)
	}

	// Send jobs.
	for i, sentence := range sentences {
		jobs <- ollamaJob{index: i, sentence: sentence}
	}
	close(jobs)

	// Wait for all workers to finish.
	wg.Wait()
	close(results)

	// Collect results.
	vectors := make([][]float64, numSentences)
	for result := range results {
		if result.err != nil {
			// Fail fast on the first error.
			return nil, result.err
		}
		vectors[result.index] = result.embedding
	}

	return vectors, nil
}

// ollamaWorker is a worker function that processes embedding requests from the jobs channel.
func ollamaWorker(wg *sync.WaitGroup, client *http.Client, jobs <-chan ollamaJob, results chan<- ollamaResult, url, model string) {
	defer wg.Done()
	for job := range jobs {
		reqBody, err := json.Marshal(ollamaRequest{Model: model, Prompt: job.sentence})
		if err != nil {
			results <- ollamaResult{index: job.index, err: fmt.Errorf("failed to marshal ollama request for sentence %d: %w", job.index, err)}
			continue
		}

		req, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
		if err != nil {
			results <- ollamaResult{index: job.index, err: fmt.Errorf("failed to create http request for sentence %d: %w", job.index, err)}
			continue
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			results <- ollamaResult{index: job.index, err: fmt.Errorf("failed to call ollama api for sentence %d: %w", job.index, err)}
			continue
		}

		if resp.StatusCode != http.StatusOK {
			results <- ollamaResult{index: job.index, err: fmt.Errorf("ollama api returned non-200 status for sentence %d: %s", job.index, resp.Status)}
			resp.Body.Close()
			continue
		}

		var ollamaResp ollamaResponse
		if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
			results <- ollamaResult{index: job.index, err: fmt.Errorf("failed to decode ollama response for sentence %d: %w", job.index, err)}
			resp.Body.Close()
			continue
		}
		resp.Body.Close()

		if ollamaResp.Error != "" {
			results <- ollamaResult{index: job.index, err: fmt.Errorf("ollama api returned error for sentence %d: %s", job.index, ollamaResp.Error)}
			continue
		}

		results <- ollamaResult{index: job.index, embedding: ollamaResp.Embedding}
	}
}

// cosineSimilarityDense calculates the cosine similarity between two dense vectors.
func cosineSimilarityDense(v1, v2 []float64) float64 {
	if len(v1) != len(v2) || len(v1) == 0 {
		return 0.0
	}

	var dot, normA, normB float64
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
		normA += v1[i] * v1[i]
		normB += v2[i] * v2[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// calculateCohesionDense calculates cohesion scores based on dense vectors.
func calculateCohesionDense(vectors [][]float64) []float64 {
	if len(vectors) < 2 {
		return []float64{}
	}
	scores := make([]float64, len(vectors)-1)
	for i := 0; i < len(vectors)-1; i++ {
		scores[i] = cosineSimilarityDense(vectors[i], vectors[i+1])
	}
	return scores
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
	if len(vectors) < 2 {
		return []float64{}
	}
	scores := make([]float64, len(vectors)-1)
	for i := 0; i < len(vectors)-1; i++ {
		scores[i] = tfidf.CosineSimilarity(vectors[i], vectors[i+1])
	}
	return scores
}

// findBoundaries identifies boundaries from similarity scores and options.
// Returns a map where a key of `i` means a split after sentence `i`.
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

		// Local minima detection method
		if i > 0 && i < len(scores)-1 {
			isLocalMinimum := scores[i] < scores[i-1] && scores[i] < scores[i+1]
			if isLocalMinimum {
				// Calculate the "depth" of the dip
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

		// Handle oversized single sentences
		if sentenceTokens > maxTokens {
			// First, finalize the current chunk if it has content
			if len(currentChunkSentences) > 0 {
				chunks = append(chunks, makeChunk(currentChunkSentences, currentChunkTokens))
			}
			// Add the oversized sentence as its own chunk
			chunks = append(chunks, makeChunk([]string{sentence}, sentenceTokens))
			// Reset for the next chunk
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

	// Add the last remaining chunk if it exists
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
