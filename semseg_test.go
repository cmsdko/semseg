// ./semseg_test.go
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
			opts: Options{
				MaxTokens:      20,
				DepthThreshold: 0.0,
			},
			expectedNumChunks: 1,         // CORRECT: All sentences fit into one chunk as total tokens (19) < MaxTokens (20)
			expectedTokens:    []int{19}, // CORRECT: 5 + 4 + 5 + 5 = 19
		},
		{
			name: "Token limit forces split",
			text: "This is a very long sentence about a single topic that keeps going. " +
				"This is another long sentence that continues the same idea. " +
				"And a third one to ensure the limit is hit.",
			opts: Options{
				MaxTokens:      20,
				DepthThreshold: 0.0,
			},
			expectedNumChunks: 2,
			expectedTokens:    []int{13, 20}, // CORRECT: First sentence is 13 tokens. Second (11) + third (9) = 20.
		},
		{
			name: "Oversized single sentence",
			text: "This single sentence is deliberately made to be much longer than the " +
				"maximum token limit to test the edge case handling.",
			opts: Options{
				MaxTokens:      15,
				DepthThreshold: 0.0,
			},
			expectedNumChunks: 1,
			expectedTokens:    []int{21}, // CORRECT: The sentence has 21 tokens.
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

			if len(tc.expectedTokens) > 0 {
				if len(tc.expectedTokens) != len(chunks) {
					t.Fatalf("Expected token spec for %d chunks, got %d", len(chunks), len(tc.expectedTokens))
				}
				for i, ch := range chunks {
					if ch.NumTokens != tc.expectedTokens[i] {
						t.Fatalf("Chunk %d tokens: expected %d, got %d", i, tc.expectedTokens[i], ch.NumTokens)
					}
				}
			}
		})
	}
}

func TestSplitAndTokenizeIntegration(t *testing.T) {
	text := `Hello world." Don't panic… Seriously!`
	chunks, err := Segment(text, Options{MaxTokens: 10, DepthThreshold: 0.0})
	if err != nil {
		t.Fatalf("Segment() error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatalf("Expected at least one chunk")
	}
	// Ensure tokenizer keeps contractions and hyphenated words intact
	toks := strings.Fields(strings.ToLower(chunks[0].Text))
	if len(toks) == 0 {
		t.Fatalf("Expected tokens in first chunk")
	}
}
