// file: example/main.go
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/cmsdko/semseg"
)

const (
	defaultOllamaTimeoutSeconds = 60
)

// APIRequest represents the JSON structure expected by the /segment endpoint.
type APIRequest struct {
	Text                      string  `json:"text"`
	MaxTokens                 int     `json:"max_tokens"`
	MinSplitSimilarity        float64 `json:"min_split_similarity,omitempty"`
	DepthThreshold            float64 `json:"depth_threshold,omitempty"`
	Language                  string  `json:"language,omitempty"`
	LanguageDetectionMode     string  `json:"language_detection_mode,omitempty"`
	LanguageDetectionTokens   int     `json:"language_detection_tokens,omitempty"`
	PreNormalizeAbbreviations *bool   `json:"pre_normalize_abbreviations,omitempty"`
	EnableStopWordRemoval     *bool   `json:"enable_stop_word_removal,omitempty"`
	EnableStemming            *bool   `json:"enable_stemming,omitempty"`
	TfidfMinNgramSize         int     `json:"tfidf_min_ngram_size,omitempty"`
	TfidfMaxNgramSize         int     `json:"tfidf_max_ngram_size,omitempty"`

	// New fields for controlling the semantic cache
	EmbeddingCacheMode               string  `json:"embedding_cache_mode,omitempty"`
	CacheSimilarityThreshold         float64 `json:"cache_similarity_threshold,omitempty"`
	AdaptiveCacheActivationThreshold int     `json:"adaptive_cache_activation_threshold,omitempty"`
}

// ResponseOptions reflects the settings that were actually used for segmentation.
type ResponseOptions struct {
	MaxTokens                 int     `json:"max_tokens"`
	MinSplitSimilarity        float64 `json:"min_split_similarity"`
	DepthThreshold            float64 `json:"depth_threshold"`
	Language                  string  `json:"language"`
	LanguageDetectionMode     string  `json:"language_detection_mode"`
	LanguageDetectionTokens   int     `json:"language_detection_tokens"`
	PreNormalizeAbbreviations bool    `json:"pre_normalize_abbreviations"`
	EnableStopWordRemoval     bool    `json:"enable_stop_word_removal"`
	EnableStemming            bool    `json:"enable_stemming"`
	TfidfMinNgramSize         int     `json:"tfidf_min_ngram_size"`
	TfidfMaxNgramSize         int     `json:"tfidf_max_ngram_size"`

	// New fields for reflecting cache settings
	EmbeddingCacheMode               string  `json:"embedding_cache_mode"`
	CacheSimilarityThreshold         float64 `json:"cache_similarity_threshold"`
	AdaptiveCacheActivationThreshold int     `json:"adaptive_cache_activation_threshold"`
}

// ... (Stats, APIResponse, APIError structs remain the same) ...
type Stats struct {
	TotalChunks      int     `json:"total_chunks"`
	TotalTokens      int     `json:"total_tokens"`
	ProcessingTimeMS float64 `json:"processing_time_ms"`
	ChunksPerSecond  float64 `json:"chunks_per_second"`
	TokensPerSecond  float64 `json:"tokens_per_second"`
}

type APIResponse struct {
	OptionsUsed ResponseOptions `json:"options_used"`
	Chunks      []semseg.Chunk  `json:"chunks"`
	Stats       Stats           `json:"stats"`
}

type APIError struct {
	Error string `json:"error"`
}

// APIHandler holds dependencies like the shared http.Client and the embedding cache.
type APIHandler struct {
	ollamaClient   *http.Client
	embeddingCache semseg.EmbeddingCache
}

// NewAPIHandler creates a new handler with its dependencies initialized.
func NewAPIHandler() *APIHandler {
	timeoutStr := os.Getenv("OLLAMA_TIMEOUT_SECONDS")
	timeoutSec, err := strconv.Atoi(timeoutStr)
	if err != nil || timeoutSec <= 0 {
		timeoutSec = defaultOllamaTimeoutSeconds
	}

	log.Printf("Initializing Ollama client with a %d second timeout", timeoutSec)
	client := &http.Client{
		Timeout: time.Duration(timeoutSec) * time.Second,
	}

	// Create a single, shared in-memory cache wrapped by the adaptive manager.
	// This ensures cache state persists across all API requests.
	log.Println("Initializing a shared adaptive embedding cache for the handler.")
	baseCache := semseg.NewInMemoryCache()
	adaptiveManager := semseg.NewAdaptiveCacheManager(baseCache)

	return &APIHandler{
		ollamaClient:   client,
		embeddingCache: adaptiveManager,
	}
}

// handleSegment handles HTTP POST requests to the /segment endpoint.
func (h *APIHandler) handleSegment(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonError(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1MB limit

	var req APIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.MaxTokens <= 0 {
		jsonError(w, "max_tokens must be > 0", http.StatusBadRequest)
		return
	}

	opts := semseg.Options{
		MaxTokens:                 req.MaxTokens,
		MinSplitSimilarity:        req.MinSplitSimilarity,
		DepthThreshold:            req.DepthThreshold,
		Language:                  req.Language,
		LanguageDetectionMode:     req.LanguageDetectionMode,
		LanguageDetectionTokens:   req.LanguageDetectionTokens,
		PreNormalizeAbbreviations: req.PreNormalizeAbbreviations,
		EnableStopWordRemoval:     req.EnableStopWordRemoval,
		EnableStemming:            req.EnableStemming,
		TfidfMinNgramSize:         req.TfidfMinNgramSize,
		TfidfMaxNgramSize:         req.TfidfMaxNgramSize,
		HTTPClient:                h.ollamaClient,

		// Pass cache settings from the request to the library
		EmbeddingCacheMode:               req.EmbeddingCacheMode,
		EmbeddingCache:                   h.embeddingCache, // Use the shared cache instance
		CacheSimilarityThreshold:         req.CacheSimilarityThreshold,
		AdaptiveCacheActivationThreshold: req.AdaptiveCacheActivationThreshold,
	}

	responseOpts := buildResponseOptions(req)

	startTime := time.Now()
	chunks, err := semseg.Segment(req.Text, opts)
	duration := time.Since(startTime)

	if err != nil {
		jsonError(w, "Internal server error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	stats := calculateStats(chunks, duration)
	response := APIResponse{
		OptionsUsed: responseOpts,
		Chunks:      chunks,
		Stats:       stats,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
}

func buildResponseOptions(req APIRequest) ResponseOptions {
	opts := ResponseOptions{
		MaxTokens:                        req.MaxTokens,
		MinSplitSimilarity:               req.MinSplitSimilarity,
		DepthThreshold:                   req.DepthThreshold,
		Language:                         req.Language,
		LanguageDetectionMode:            req.LanguageDetectionMode,
		LanguageDetectionTokens:          req.LanguageDetectionTokens,
		TfidfMinNgramSize:                req.TfidfMinNgramSize,
		TfidfMaxNgramSize:                req.TfidfMaxNgramSize,
		EmbeddingCacheMode:               req.EmbeddingCacheMode,
		CacheSimilarityThreshold:         req.CacheSimilarityThreshold,
		AdaptiveCacheActivationThreshold: req.AdaptiveCacheActivationThreshold,
	}

	// Apply defaults for clear user feedback
	if opts.EmbeddingCacheMode == "" {
		opts.EmbeddingCacheMode = semseg.CacheModeDisable
	}
	if opts.MinSplitSimilarity == 0 && opts.DepthThreshold <= 0 {
		opts.DepthThreshold = 0.1
	}
	if opts.Language == "" && opts.LanguageDetectionMode == "" {
		opts.LanguageDetectionMode = semseg.LangDetectModeFirstSentence
	}
	if opts.EmbeddingCacheMode != semseg.CacheModeDisable && opts.CacheSimilarityThreshold == 0 {
		opts.CacheSimilarityThreshold = 0.9
	}
	if opts.EmbeddingCacheMode == semseg.CacheModeAdaptive && opts.AdaptiveCacheActivationThreshold == 0 {
		opts.AdaptiveCacheActivationThreshold = 100
	}

	opts.PreNormalizeAbbreviations = req.PreNormalizeAbbreviations == nil || *req.PreNormalizeAbbreviations
	opts.EnableStopWordRemoval = req.EnableStopWordRemoval == nil || *req.EnableStopWordRemoval
	opts.EnableStemming = req.EnableStemming == nil || *req.EnableStemming
	return opts
}

// ... (calculateStats, jsonError, main functions remain the same) ...
func calculateStats(chunks []semseg.Chunk, duration time.Duration) Stats {
	totalTokens := 0
	for _, chunk := range chunks {
		totalTokens += chunk.NumTokens
	}
	totalChunks := len(chunks)
	durationSec := duration.Seconds()

	var chunksPerSec, tokensPerSec float64
	if durationSec > 0 {
		chunksPerSec = float64(totalChunks) / durationSec
		tokensPerSec = float64(totalTokens) / durationSec
	}

	return Stats{
		TotalChunks:      totalChunks,
		TotalTokens:      totalTokens,
		ProcessingTimeMS: float64(duration.Microseconds()) / 1000.0,
		ChunksPerSecond:  chunksPerSec,
		TokensPerSecond:  tokensPerSec,
	}
}

func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(APIError{Error: message})
}

func main() {
	apiHandler := NewAPIHandler()
	mux := http.NewServeMux()
	mux.HandleFunc("/segment", apiHandler.handleSegment)
	srv := &http.Server{
		Addr:              ":8080",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	log.Println("Starting server on :8080...")
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
