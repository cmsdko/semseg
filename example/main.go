package main

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/cmsdko/semseg"
)

// APIRequest represents the JSON payload structure expected by the /segment endpoint.
// It contains the text to be segmented along with optional configuration parameters
// for controlling the segmentation behavior.
type APIRequest struct {
	// Text is the input string that will be semantically segmented into chunks.
	// This field is required and should contain the full text to process.
	Text string `json:"text"`

	// MaxTokens defines the maximum number of tokens allowed per chunk.
	// This is a hard limit - no chunk will exceed this size unless a single
	// sentence is larger than the limit (in which case it gets its own chunk).
	// This field is required and must be greater than 0.
	MaxTokens int `json:"max_tokens"`

	// MinSplitSimilarity is an optional threshold (0.0 to 1.0) for splitting.
	// If the cosine similarity between adjacent sentences falls below this value,
	// a split will be considered at that point. If set to 0 (default), the algorithm
	// will use the more sophisticated local minima detection method instead.
	MinSplitSimilarity float64 `json:"min_split_similarity,omitempty"`

	// DepthThreshold is used by the default boundary detection method when
	// MinSplitSimilarity is 0. It defines the minimum "depth" of a similarity
	// dip to be considered a valid split point. This prevents minor fluctuations
	// from causing unnecessary splits. Default is 0.1 if not specified.
	DepthThreshold float64 `json:"depth_threshold,omitempty"`
}

// handleSegment processes HTTP requests to the /segment endpoint.
// It expects a POST request with JSON payload containing text and segmentation parameters,
// and returns the segmented chunks as JSON response.
func handleSegment(w http.ResponseWriter, r *http.Request) {
	// Only accept POST requests - segmentation requires request body data
	if r.Method != http.MethodPost {
		jsonError(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// Limit request body size to 1MB to prevent memory exhaustion attacks
	// This creates a wrapper around the request body that enforces the size limit
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	// Parse the JSON request body into our APIRequest struct
	var req APIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Validate required fields - MaxTokens must be positive
	// The semseg library will also validate this, but we check early for better error messages
	if req.MaxTokens <= 0 {
		jsonError(w, "max_tokens must be > 0", http.StatusBadRequest)
		return
	}

	// Configure the segmentation options based on the request parameters
	// The library handles default values internally for optional parameters
	opts := semseg.Options{
		MaxTokens:          req.MaxTokens,
		MinSplitSimilarity: req.MinSplitSimilarity, // 0 = use local minima detection
		DepthThreshold:     req.DepthThreshold,     // < 0 = use default 0.1
	}

	// Perform the actual text segmentation using the semseg library
	chunks, err := semseg.Segment(req.Text, opts)
	if err != nil {
		// Internal errors indicate issues with the segmentation algorithm
		jsonError(w, "Internal server error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Return successful response with the segmented chunks as JSON
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(chunks)
}

// APIError represents the structure of error responses returned by the API.
// All error responses follow this consistent format for easier client handling.
type APIError struct {
	Error string `json:"error"`
}

// jsonError is a helper function that sends a standardized JSON error response.
// It sets the appropriate HTTP status code and Content-Type header before
// encoding the error message as JSON.
func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(APIError{Error: message})
}

// main initializes and starts the HTTP server with proper configuration
// for production use, including timeouts and security settings.
func main() {
	// Create a new HTTP router and register our segmentation endpoint
	mux := http.NewServeMux()
	mux.HandleFunc("/segment", handleSegment)

	// Configure the HTTP server with security-focused timeout settings
	srv := &http.Server{
		Addr:    ":8080", // Listen on port 8080 for incoming connections
		Handler: mux,     // Use our configured router

		// ReadHeaderTimeout prevents clients from sending headers too slowly
		// This helps prevent Slowloris-style attacks
		ReadHeaderTimeout: 5 * time.Second,

		// ReadTimeout is the maximum time allowed to read the entire request,
		// including the body. This prevents clients from holding connections open
		ReadTimeout: 10 * time.Second,

		// WriteTimeout is the maximum time allowed to write the response.
		// This prevents slow clients from keeping connections open indefinitely
		WriteTimeout: 10 * time.Second,

		// MaxHeaderBytes limits the size of request headers to prevent
		// memory exhaustion from maliciously large headers (1MB limit)
		MaxHeaderBytes: 1 << 20,
	}

	// Start the HTTP server and log any startup issues
	log.Println("Starting server on :8080...")
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
