package text

import (
	"reflect"
	"testing"
)

func TestSplitSentences(t *testing.T) {
	text := "Hello world. This is a test! Is it working? Yes."
	expected := []string{"Hello world.", "This is a test!", "Is it working?", "Yes."}
	result := SplitSentences(text)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestTokenize(t *testing.T) {
	text := "Hello, world-123!"
	expected := []string{"hello", "world-123"}
	result := Tokenize(text)
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}
