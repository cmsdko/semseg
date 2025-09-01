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
// It contains the text to be segmented and all optional configuration parameters.
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

// APIError is the structure for JSON error responses.
type APIError struct {
	Error string `json:"error"`
}

// APIHandler holds dependencies like the shared http.Client.
// This allows us to inject dependencies and makes handlers testable.
type APIHandler struct {
	ollamaClient *http.Client
}

// NewAPIHandler creates a new handler with its dependencies initialized.
func NewAPIHandler() *APIHandler {
	timeoutStr := os.Getenv("OLLAMA_TIMEOUT_SECONDS")
	timeoutSec, err := strconv.Atoi(timeoutStr)
	if err != nil || timeoutSec <= 0 {
		timeoutSec = defaultOllamaTimeoutSeconds
	}

	log.Printf("Initializing Ollama client with a %d second timeout", timeoutSec)

	// Create a single, reusable http.Client
	client := &http.Client{
		Timeout: time.Duration(timeoutSec) * time.Second,
		// It's good practice to customize the transport for production apps,
		// e.g., to set MaxIdleConns, MaxIdleConnsPerHost, etc.
		// For this example, the default transport is sufficient.
	}

	return &APIHandler{
		ollamaClient: client,
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

	// 1. Create options for the semseg library, injecting the shared HTTP client.
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
		HTTPClient:                h.ollamaClient,
	}

	// 2. Create the response options struct to reflect applied defaults.
	responseOpts := buildResponseOptions(req)

	// 3. Perform segmentation and measure the time.
	startTime := time.Now()
	chunks, err := semseg.Segment(req.Text, opts)
	duration := time.Since(startTime)

	if err != nil {
		jsonError(w, "Internal server error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// 4. Calculate statistics.
	stats := calculateStats(chunks, duration)

	// 5. Form and send the complete response.
	response := APIResponse{
		OptionsUsed: responseOpts,
		Chunks:      chunks,
		Stats:       stats,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
}

// buildResponseOptions populates the options structure for the API response,
// applying the same default logic as the library to show the user what was used.
func buildResponseOptions(req APIRequest) ResponseOptions {
	responseOpts := ResponseOptions{
		MaxTokens:               req.MaxTokens,
		MinSplitSimilarity:      req.MinSplitSimilarity,
		DepthThreshold:          req.DepthThreshold,
		Language:                req.Language,
		LanguageDetectionMode:   req.LanguageDetectionMode,
		LanguageDetectionTokens: req.LanguageDetectionTokens,
	}
	// Apply default values for clear user feedback
	if responseOpts.MinSplitSimilarity == 0 && responseOpts.DepthThreshold <= 0 {
		responseOpts.DepthThreshold = 0.1 // Default from the library
	}
	if responseOpts.Language == "" && responseOpts.LanguageDetectionMode == "" {
		responseOpts.LanguageDetectionMode = semseg.LangDetectModeFirstSentence
	}
	responseOpts.PreNormalizeAbbreviations = req.PreNormalizeAbbreviations == nil || *req.PreNormalizeAbbreviations
	responseOpts.EnableStopWordRemoval = req.EnableStopWordRemoval == nil || *req.EnableStopWordRemoval
	responseOpts.EnableStemming = req.EnableStemming == nil || *req.EnableStemming
	return responseOpts
}

// calculateStats computes performance metrics for the segmentation task.
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

// jsonError writes a standard JSON error message to the response.
func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(APIError{Error: message})
}

func main() {
	// Initialize the handler which contains our shared dependencies.
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
