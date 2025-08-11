package main

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	// This import will now fetch the library from GitHub
	"github.com/cmsdko/semseg"
)

type APIRequest struct {
	Text               string  `json:"text"`
	MaxTokens          int     `json:"max_tokens"`
	MinSplitSimilarity float64 `json:"min_split_similarity"` // optional
	DepthThreshold     float64 `json:"depth_threshold"`      // optional
}

func handleSegment(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonError(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req APIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "Bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.MaxTokens <= 0 {
		jsonError(w, "max_tokens must be > 0", http.StatusBadRequest)
		return
	}

	// DepthThreshold: оставляем как есть; если хочешь «дефолт 0.1», передай -1 (мы так сделали в lib).
	opts := semseg.Options{
		MaxTokens:          req.MaxTokens,
		MinSplitSimilarity: req.MinSplitSimilarity, // 0 => локальные минимумы
		DepthThreshold:     req.DepthThreshold,     // <0 => дефолт 0.1, 0 => принимать любые локальные минимумы
	}

	chunks, err := semseg.Segment(req.Text, opts)
	if err != nil {
		jsonError(w, "Internal server error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(chunks)
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
