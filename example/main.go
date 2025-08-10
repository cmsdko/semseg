package main

import (
	"encoding/json"
	"log"
	"net/http"

	// This import will now fetch the library from GitHub
	"github.com/cmsdko/semseg"
)

type APIRequest struct {
	Text      string `json:"text"`
	MaxTokens int    `json:"max_tokens"`
}

type APIError struct {
	Error string `json:"error"`
}

func handleSegment(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	var req APIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	opts := semseg.Options{MaxTokens: req.MaxTokens}
	chunks, err := semseg.Segment(req.Text, opts)
	if err != nil {
		jsonError(w, "Internal server error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(chunks)
}

func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(APIError{Error: message})
}

func main() {
	http.HandleFunc("/segment", handleSegment)
	log.Println("Starting server on :8080...")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Could not start server: %s\n", err)
	}
}
