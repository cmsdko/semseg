package semseg

import (
	"strings"
	"testing"
)

func TestSegment(t *testing.T) {
	testCases := []struct {
		name              string
		text              string
		opts              Options
		expectedNumChunks int
		expectedTokens    []int // Expected tokens per chunk
	}{
		{
			name: "Simple semantic split",
			text: "The solar system is vast. Planets orbit the sun. " +
				"Oceans are deep and blue. Fish swim in the sea.",
			opts:              Options{MaxTokens: 20},
			expectedNumChunks: 2,
			expectedTokens:    []int{8, 8},
		},
		{
			name: "Token limit forces split",
			text: "This is a very long sentence about a single topic that keeps going. " +
				"This is another long sentence that continues the same idea. " +
				"And a third one to ensure the limit is hit.",
			opts:              Options{MaxTokens: 20},
			expectedNumChunks: 2,
			expectedTokens:    []int{12, 18},
		},
		{
			name: "Oversized single sentence",
			text: "This single sentence is deliberately made to be much longer than the " +
				"maximum token limit to test the edge case handling.",
			opts:              Options{MaxTokens: 15},
			expectedNumChunks: 1,
			expectedTokens:    []int{22},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			chunks, err := Segment(tc.text, tc.opts)
			if err != nil {
				t.Fatalf("Segment() returned an error: %v", err)
			}

			if len(chunks) != tc.expectedNumChunks {
				t.Fatalf("Expected %d chunks, but got %d", tc.expectedNumChunks, len(chunks))
			}
		})
	}
}
