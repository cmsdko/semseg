package main

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/cmsdko/semseg"
)

// APIRequest represents the JSON structure expected by the /segment endpoint.
// It contains the text to be segmented and all optional configuration parameters.
type APIRequest struct {
	// Text is the input string for semantic segmentation. This field is required.
	Text string `json:"text"`

	// MaxTokens is the hard limit on the number of tokens in a chunk. This field is required and must be > 0.
	MaxTokens int `json:"max_tokens"`

	// MinSplitSimilarity is the cosine similarity threshold (0.0 to 1.0) for splitting.
	// If 0 (the default), the local minima search method is used.
	MinSplitSimilarity float64 `json:"min_split_similarity,omitempty"`

	// DepthThreshold is used when MinSplitSimilarity=0. It defines the minimum "depth"
	// of the similarity dip for a split point. Defaults to 0.1.
	DepthThreshold float64 `json:"depth_threshold,omitempty"`

	// Language forces a specific language (e.g., "english"), skipping auto-detection.
	Language string `json:"language,omitempty"`

	// LanguageDetectionMode is the strategy for language auto-detection if Language is not set.
	// Possible values: "first_sentence", "first_ten_sentences", "per_sentence", "full_text".
	LanguageDetectionMode string `json:"language_detection_mode,omitempty"`

	// LanguageDetectionTokens enables early language detection based on the first N tokens.
	// If > 0, the language is detected before splitting into sentences.
	LanguageDetectionTokens int `json:"language_detection_tokens,omitempty"`

	// PreNormalizeAbbreviations normalizes abbreviations with periods (e.g., U.S.A. -> USA) before sentence splitting.
	// Defaults to true.
	PreNormalizeAbbreviations *bool `json:"pre_normalize_abbreviations,omitempty"`

	// EnableStopWordRemoval controls the removal of stop words. Defaults to true.
	EnableStopWordRemoval *bool `json:"enable_stop_word_removal,omitempty"`

	// EnableStemming controls stemming (reducing words to their root form). Defaults to true.
	EnableStemming *bool `json:"enable_stemming,omitempty"`
}

// ResponseOptions reflects the settings that were actually used for segmentation,
// including the default values applied by the server.
type ResponseOptions struct {
	MaxTokens                   int     `json:"max_tokens"`
	MinSplitSimilarity          float64 `json:"min_split_similarity"`
	DepthThreshold              float64 `json:"depth_threshold"`
	Language                    string  `json:"language"`
	LanguageDetectionMode       string  `json:"language_detection_mode"`
	LanguageDetectionTokens     int     `json:"language_detection_tokens"`
	PreNormalizeAbbreviations   bool    `json:"pre_normalize_abbreviations"`
	EnableStopWordRemoval       bool    `json:"enable_stop_word_removal"`
	EnableStemming              bool    `json:"enable_stemming"`
}

// Stats contains performance statistics for a single request.
type Stats struct {
	TotalChunks      int     `json:"total_chunks"`
	TotalTokens      int     `json:"total_tokens"`
	ProcessingTimeMS float64 `json:"processing_time_ms"`
	ChunksPerSecond  float64 `json:"chunks_per_second"`
	TokensPerSecond  float64 `json:"tokens_per_second"`
}

// APIResponse is the complete response structure returned by the /segment endpoint.
type APIResponse struct {
	OptionsUsed ResponseOptions `json:"options_used"`
	Chunks      []semseg.Chunk  `json:"chunks"`
	Stats       Stats           `json:"stats"`
}

// handleSegment handles HTTP POST requests to the /segment endpoint.
// It accepts a JSON payload with text and parameters, and returns the segmented chunks,
// the settings used, and performance statistics.
func handleSegment(w http.ResponseWriter, r *http.Request) {
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

	// 1. Create options for the semseg library based on the request
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
	}

	// 2. Create the response options struct, applying default value logic
	// so the user can see exactly which settings were used.
	responseOpts := ResponseOptions{
		MaxTokens:               req.MaxTokens,
		MinSplitSimilarity:      req.MinSplitSimilarity,
		DepthThreshold:          req.DepthThreshold,
		Language:                req.Language,
		LanguageDetectionMode:   req.LanguageDetectionMode,
		LanguageDetectionTokens: req.LanguageDetectionTokens,
	}
	// Apply default values
	if responseOpts.MinSplitSimilarity == 0 && responseOpts.DepthThreshold <= 0 {
		responseOpts.DepthThreshold = 0.1 // Default from the library
	}
	if responseOpts.Language == "" && responseOpts.LanguageDetectionMode == "" {
		responseOpts.LanguageDetectionMode = semseg.LangDetectModeFirstSentence
	}
	responseOpts.PreNormalizeAbbreviations = req.PreNormalizeAbbreviations == nil || *req.PreNormalizeAbbreviations
	responseOpts.EnableStopWordRemoval = req.EnableStopWordRemoval == nil || *req.EnableStopWordRemoval
	responseOpts.EnableStemming = req.EnableStemming == nil || *req.EnableStemming

	// 3. Perform segmentation and measure the time
	startTime := time.Now()
	chunks, err := semseg.Segment(req.Text, opts)
	duration := time.Since(startTime)

	if err != nil {
		jsonError(w, "Internal server error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// 4. Calculate statistics
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

	stats := Stats{
		TotalChunks:      totalChunks,
		TotalTokens:      totalTokens,
		ProcessingTimeMS: float64(duration.Microseconds()) / 1000.0,
		ChunksPerSecond:  chunksPerSec,
		TokensPerSecond:  tokensPerSec,
	}

	// 5. Form and send the complete response
	response := APIResponse{
		OptionsUsed: responseOpts,
		Chunks:      chunks,
		Stats:       stats,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
}

type APIError struct {
	Error string `json:"error"`
}

func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(APIError{Error: message})
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/segment", handleSegment)

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
