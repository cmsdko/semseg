// file: internal/text/text_test.go

package text

import (
	"reflect"
	"testing"
)

// TestSplitSentences verifies that sentence boundaries are correctly detected.
// Covers punctuation (., !, ?), multiple sentences, and proper trimming.
func TestSplitSentences(t *testing.T) {
	text := "Hello world. This is a test! Is it working? Yes."
	expected := []string{"Hello world.", "This is a test!", "Is it working?", "Yes."}
	result := SplitSentences(text)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// TestTokenize verifies tokenization rules.
// - Lowercasing
// - Removal of punctuation (, !)
// - Preservation of hyphenated alphanumerics ("world-123")
func TestTokenize(t *testing.T) {
	text := "Hello, world-123!"
	expected := []string{"hello", "world-123"}
	result := Tokenize(text)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// TestGenerateCharNgrams verifies character n-gram generation.
func TestGenerateCharNgrams(t *testing.T) {
	testCases := []struct {
		name     string
		text     string
		minN     int
		maxN     int
		expected []string
	}{
		{
			name:     "Simple ASCII",
			text:     "word",
			minN:     3,
			maxN:     3,
			expected: []string{"wor", "ord"},
		},
		{
			name:     "ASCII with range",
			text:     "text",
			minN:     2,
			maxN:     3,
			expected: []string{"te", "ex", "xt", "tex", "ext"},
		},
		{
			name:     "Cyrillic text",
			text:     "слово",
			minN:     4,
			maxN:     4,
			expected: []string{"слов", "лово"},
		},
		{
			name:     "Text with punctuation and spaces",
			text:     "Hi, world!",
			minN:     3,
			maxN:     3,
			expected: []string{"hiw", "iwo", "wor", "orl", "rld"},
		},
		{
			name:     "Text shorter than minN",
			text:     "go",
			minN:     3,
			maxN:     4,
			expected: []string{}, // non-nil empty slice
		},
		{
			name:     "Invalid range",
			text:     "test",
			minN:     4,
			maxN:     3,
			expected: []string{}, // non-nil empty slice
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := GenerateCharNgrams(tc.text, tc.minN, tc.maxN)
			if !reflect.DeepEqual(result, tc.expected) {
				// --- IMPROVED ERROR MESSAGE HERE ---
				// This provides more diagnostic information in case of failure,
				// specifically checking if one slice is nil while the other is not,
				// which is a common pitfall with reflect.DeepEqual.
				t.Errorf("Expected %v (is nil: %t), but got %v (is nil: %t)",
					tc.expected, tc.expected == nil, result, result == nil)
			}
		})
	}
}
