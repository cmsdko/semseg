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
