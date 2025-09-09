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

// ... (LanguageDetectionMode constants remain the same) ...
const (
	LangDetectModeFirstSentence     = "first_sentence"
	LangDetectModeFirstTenSentences = "first_ten_sentences"
	LangDetectModePerSentence       = "per_sentence"
	LangDetectModeFullText          = "full_text"
)

// Constants for EmbeddingCacheMode.
const (
	// CacheModeDisable disables the semantic cache completely. All embedding requests go to the provider. This is the default.
	CacheModeDisable = "disable"
	// CacheModeForce enables the semantic cache in a blocking mode. It checks the cache first; on a miss,
	// it calls the embedding provider and then synchronously updates the cache before returning.
	CacheModeForce = "force"
	// CacheModeAdaptive starts with caching disabled for lookups but asynchronously populates the cache
	// in the background. Once the cache contains enough similar items (see AdaptiveCacheActivationThreshold),
	// it automatically switches to 'force' mode for all subsequent requests.
	CacheModeAdaptive = "adaptive"
)

// Constants for Ollama worker pool
const (
	OllamaMaxWorkersEnvVar = "CHUNKER_OLLAMA_MAX_WORKERS"
	DefaultOllamaWorkers   = 4
)

// ... (Chunk struct remains the same) ...
type Chunk struct {
	Text      string
	Sentences []string
	NumTokens int
}

// Options configures the segmentation process.
type Options struct {
	// ... (MaxTokens, MinSplitSimilarity, etc. remain the same) ...
	MaxTokens                 int
	MinSplitSimilarity        float64
	DepthThreshold            float64
	Language                  string
	LanguageDetectionMode     string
	LanguageDetectionTokens   int
	PreNormalizeAbbreviations *bool
	EnableStopWordRemoval     *bool
	EnableStemming            *bool
	TfidfMinNgramSize         int
	TfidfMaxNgramSize         int
	HTTPClient                *http.Client

	// --- Semantic Caching for Dense Embeddings ---

	// EmbeddingCacheMode specifies the caching strategy: "disable", "force", or "adaptive".
	// Default: "disable".
	EmbeddingCacheMode string

	// EmbeddingCache is an instance of a cache that stores mappings from a sentence's
	// TF-IDF n-gram vector to its dense embedding. This allows reusing embeddings for
	// semantically similar sentences, reducing API calls to heavy models.
	// A default in-memory cache can be created with NewInMemoryCache() or NewAdaptiveCacheManager().
	EmbeddingCache EmbeddingCache

	// CacheSimilarityThreshold (range 0.0 to 1.0) is the cosine similarity
	// threshold used to determine a cache hit. Default: 0.9.
	CacheSimilarityThreshold float64

	// AdaptiveCacheActivationThreshold is the number of items in the cache that must have at least one
	// semantically similar neighbor (defined by CacheSimilarityThreshold) before an 'adaptive' cache
	// switches to 'force' mode. Only used when EmbeddingCacheMode is "adaptive". Default: 100.
	AdaptiveCacheActivationThreshold int
}

// Segment splits a given text into semantic chunks based on the provided options.
// It acts as an orchestrator, handling preprocessing and then dispatching to either
// the Ollama or TF-IDF implementation to get similarity scores.
func Segment(textStr string, opts Options) ([]Chunk, error) {
	if err := validateOptions(opts); err != nil {
		return nil, err
	}
	setDefaultOptions(&opts)

	// --- 1. Early language selection (explicit or by first N tokens) before any normalization/splitting ---
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

	// --- 2. Optional abbreviation normalization before sentence splitting ---
	if *opts.PreNormalizeAbbreviations {
		textStr = lang.NormalizeAbbreviations(textStr, globalDetectedLang)
	}

	// --- 3. Split into sentences and handle edge cases ---
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

	// --- 4. Calculate cohesion scores using the appropriate method (Ollama or TF-IDF) ---
	var scores []float64
	var err error

	ollamaURL := os.Getenv("CHUNKER_OLLAMA_URL")
	ollamaModel := os.Getenv("CHUNKER_OLLAMA_MODEL")

	if ollamaURL != "" && ollamaModel != "" {
		// PATH A: Use modern embeddings via Ollama for higher accuracy.
		scores, err = segmentWithOllama(sentences, ollamaURL, ollamaModel, opts)
		if err != nil {
			return nil, err // Propagate errors from Ollama API calls.
		}
	} else {
		// PATH B: Use the lightweight, built-in TF-IDF method.
		scores = segmentWithTFIDF(textStr, sentences, opts, globalDetectedLang)
	}

	// --- 5. Find split boundaries and build the final chunks ---
	boundaryIndices := findBoundaries(scores, opts)
	return buildChunks(sentences, tokenCounts, boundaryIndices, opts.MaxTokens), nil
}

// segmentWithOllama handles the logic for vectorizing sentences using an Ollama model
// and calculating cohesion scores between them.
func segmentWithOllama(sentences []string, ollamaURL, ollamaModel string, opts Options) ([]float64, error) {
	client := opts.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}

	vectors, err := getOllamaEmbeddings(sentences, ollamaURL, ollamaModel, client, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to get ollama embeddings: %w", err)
	}

	return calculateCohesionDense(vectors), nil
}

// ... (segmentWithTFIDF remains the same) ...
func segmentWithTFIDF(textStr string, sentences []string, opts Options, globalDetectedLang string) []float64 {
	// If language wasn't detected early, detect it now based on the specified mode.
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
			globalDetectedLang = lang.DetectLanguage(sentences[0]) // Fallback to default
		}
	}

	// Pre-process and tokenize each sentence based on options.
	tokenizedSentences := make([][]string, len(sentences))
	for i, s := range sentences {
		var detectedLang string
		if opts.LanguageDetectionMode == LangDetectModePerSentence && opts.Language == "" {
			detectedLang = lang.DetectLanguage(s)
		} else {
			detectedLang = globalDetectedLang
		}

		var tokens []string
		if opts.TfidfMinNgramSize > 0 && opts.TfidfMaxNgramSize >= opts.TfidfMinNgramSize {
			// N-gram mode: stemming and stop words are not applied.
			tokens = text.GenerateCharNgrams(s, opts.TfidfMinNgramSize, opts.TfidfMaxNgramSize)
		} else {
			// Standard word tokenization mode with optional preprocessing.
			sentenceForSimilarity := s
			if *opts.EnableStopWordRemoval {
				sentenceForSimilarity = lang.RemoveStopWords(sentenceForSimilarity, detectedLang)
			}
			tokens = text.Tokenize(sentenceForSimilarity)
			if *opts.EnableStemming {
				tokens = lang.StemTokens(tokens, detectedLang)
			}
		}
		tokenizedSentences[i] = tokens
	}

	// Vectorize sentences using TF-IDF and calculate similarity scores.
	corpus := tfidf.NewCorpus(tokenizedSentences)
	vectors := make([]map[string]float64, len(sentences))
	for i, ts := range tokenizedSentences {
		vectors[i] = corpus.Vectorize(ts)
	}

	return calculateCohesion(vectors)
}

// ... (ollama structs remain the same) ...
type ollamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaResponse struct {
	Embedding []float64 `json:"embedding"`
	Error     string    `json:"error,omitempty"`
}

type ollamaJob struct {
	index    int
	sentence string
}

type ollamaResult struct {
	index     int
	embedding []float64
	err       error
}

// getOllamaEmbeddings fetches embeddings for all sentences, dispatching to the correct caching strategy.
func getOllamaEmbeddings(sentences []string, ollamaURL, ollamaModel string, client *http.Client, opts Options) ([][]float64, error) {
	if len(sentences) == 0 {
		return [][]float64{}, nil
	}

	switch opts.EmbeddingCacheMode {
	case CacheModeForce:
		return getOllamaEmbeddingsWithCache(sentences, ollamaURL, ollamaModel, client, opts)
	case CacheModeAdaptive:
		return getOllamaEmbeddingsAdaptive(sentences, ollamaURL, ollamaModel, client, opts)
	default: // CacheModeDisable or empty
		return getOllamaEmbeddingsDirect(sentences, ollamaURL, ollamaModel, client)
	}
}

// getOllamaEmbeddingsWithCache is the 'force' mode implementation.
func getOllamaEmbeddingsWithCache(sentences []string, ollamaURL, ollamaModel string, client *http.Client, opts Options) ([][]float64, error) {
	numSentences := len(sentences)
	vectors := make([][]float64, numSentences)

	// 1. Pre-calculate all TF-IDF n-gram vectors (cache keys).
	ngramSentences := make([][]string, numSentences)
	for i, s := range sentences {
		ngramSentences[i] = text.GenerateCharNgrams(s, 3, 5)
	}
	corpus := tfidf.NewCorpus(ngramSentences)
	keyVectors := make([]map[string]float64, numSentences)
	for i, ns := range ngramSentences {
		keyVectors[i] = corpus.Vectorize(ns)
	}

	// 2. Identify cache hits and misses.
	jobsToRun := make([]ollamaJob, 0)
	for i, key := range keyVectors {
		embedding, found := opts.EmbeddingCache.Find(key, opts.CacheSimilarityThreshold)
		if found {
			vectors[i] = embedding
		} else {
			jobsToRun = append(jobsToRun, ollamaJob{index: i, sentence: sentences[i]})
		}
	}

	if len(jobsToRun) == 0 {
		return vectors, nil
	}

	// 3. Run Ollama workers for cache misses.
	results, err := runOllamaWorkers(jobsToRun, ollamaURL, ollamaModel, client)
	if err != nil {
		return nil, err
	}

	// 4. Collect results and update the cache.
	for _, result := range results {
		vectors[result.index] = result.embedding
		// Передаем threshold, который используется для инкрементального анализа
		opts.EmbeddingCache.Set(keyVectors[result.index], result.embedding, opts.CacheSimilarityThreshold)
	}
	return vectors, nil
}

// getOllamaEmbeddingsAdaptive handles the 'adaptive' mode logic.
func getOllamaEmbeddingsAdaptive(sentences []string, ollamaURL, ollamaModel string, client *http.Client, opts Options) ([][]float64, error) {
	manager, ok := opts.EmbeddingCache.(AdaptiveCacheManager)
	if !ok {
		return nil, errors.New("adaptive cache mode requires an EmbeddingCache that implements AdaptiveCacheManager")
	}

	manager.Start(opts.CacheSimilarityThreshold, opts.AdaptiveCacheActivationThreshold)

	if manager.IsActivated() {
		// Once activated, it behaves identically to 'force' mode.
		return getOllamaEmbeddingsWithCache(sentences, ollamaURL, ollamaModel, client, opts)
	}

	// --- Pre-activation: Get embeddings directly and queue for async caching ---
	// 1. Get all embeddings directly from Ollama.
	vectors, err := getOllamaEmbeddingsDirect(sentences, ollamaURL, ollamaModel, client)
	if err != nil {
		return nil, err
	}

	// 2. Asynchronously populate the cache.
	// This part does not block the return to the user.
	go func() {
		ngramSentences := make([][]string, len(sentences))
		for i, s := range sentences {
			ngramSentences[i] = text.GenerateCharNgrams(s, 3, 5)
		}
		corpus := tfidf.NewCorpus(ngramSentences)
		for i, ns := range ngramSentences {
			keyVector := corpus.Vectorize(ns)
			manager.QueueSet(keyVector, vectors[i])
		}
	}()

	return vectors, nil
}

// getOllamaEmbeddingsDirect is the 'disable' mode implementation (no caching).
func getOllamaEmbeddingsDirect(sentences []string, ollamaURL, ollamaModel string, client *http.Client) ([][]float64, error) {
	jobsToRun := make([]ollamaJob, len(sentences))
	for i, s := range sentences {
		jobsToRun[i] = ollamaJob{index: i, sentence: s}
	}

	results, err := runOllamaWorkers(jobsToRun, ollamaURL, ollamaModel, client)
	if err != nil {
		return nil, err
	}

	vectors := make([][]float64, len(sentences))
	for _, result := range results {
		vectors[result.index] = result.embedding
	}

	return vectors, nil
}

// runOllamaWorkers manages the worker pool for fetching embeddings.
func runOllamaWorkers(jobsToRun []ollamaJob, ollamaURL, ollamaModel string, client *http.Client) ([]ollamaResult, error) {
	numJobs := len(jobsToRun)
	if numJobs == 0 {
		return []ollamaResult{}, nil
	}

	numWorkersStr := os.Getenv(OllamaMaxWorkersEnvVar)
	numWorkers, err := strconv.Atoi(numWorkersStr)
	if err != nil || numWorkers <= 0 {
		numWorkers = DefaultOllamaWorkers
	}
	if numWorkers > numJobs {
		numWorkers = numJobs
	}

	jobs := make(chan ollamaJob, numJobs)
	resultsChan := make(chan ollamaResult, numJobs)
	url := strings.TrimSuffix(ollamaURL, "/") + "/api/embeddings"

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go ollamaWorker(&wg, client, jobs, resultsChan, url, ollamaModel)
	}

	for _, job := range jobsToRun {
		jobs <- job
	}
	close(jobs)

	wg.Wait()
	close(resultsChan)

	results := make([]ollamaResult, 0, numJobs)
	for result := range resultsChan {
		if result.err != nil {
			return nil, result.err // Fail fast
		}
		results = append(results, result)
	}
	return results, nil
}

// ... (ollamaWorker, cosineSimilarityDense, etc. remain the same) ...
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
	if opts.EmbeddingCacheMode != CacheModeDisable && opts.EmbeddingCacheMode != "" && opts.EmbeddingCache == nil {
		return errors.New("EmbeddingCache must be provided when a cache mode is enabled")
	}
	return nil
}

func setDefaultOptions(opts *Options) {
	if opts.EmbeddingCacheMode == "" {
		opts.EmbeddingCacheMode = CacheModeDisable
	}

	if opts.MinSplitSimilarity == 0 && opts.DepthThreshold < 0 {
		opts.DepthThreshold = 0.1
	}

	if opts.Language == "" && opts.LanguageDetectionMode == "" {
		opts.LanguageDetectionMode = LangDetectModeFirstSentence
	}

	if opts.EmbeddingCacheMode != CacheModeDisable && opts.CacheSimilarityThreshold == 0 {
		opts.CacheSimilarityThreshold = 0.9
	}

	if opts.EmbeddingCacheMode == CacheModeAdaptive && opts.AdaptiveCacheActivationThreshold == 0 {
		opts.AdaptiveCacheActivationThreshold = 100
	}

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

// ... (calculateCohesion, findBoundaries, buildChunks, makeChunk remain the same) ...
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
