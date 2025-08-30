// ./internal/lang/lang_test.go
package lang

import (
	"reflect"
	"testing"
)

func TestDetectLanguage(t *testing.T) {
	testCases := []struct {
		name     string
		sentence string
		expected string
	}{
		{"English", "This is a sample sentence for language detection.", "english"},
		{"French", "Ceci est une phrase d'exemple pour la détection de la langue.", "french"},
		{"Russian", "Это пример предложения для определения языка.", "russian"},
		{"Arabic", "ما هذا إلا مثال بسيط لتحديد اللغة", "arabic"},
		{"Ambiguous", "Hotel in Berlin", LangUnknown},
		{"Vietnamese", "Đi đâu đó", "vietnamese"},
		{"Too short", "Please go.", LangUnknown},
		{"Empty", "", LangUnknown},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			lang := DetectLanguage(tc.sentence)
			if lang != tc.expected {
				t.Errorf("Expected language '%s', but got '%s'", tc.expected, lang)
			}
		})
	}
}

func TestRemoveStopWords(t *testing.T) {
	testCases := []struct {
		name     string
		sentence string
		lang     string
		expected string
	}{
		{
			"English sentence",
			"This is a sample sentence, and we want to remove some words from it.",
			"english",
			"sample sentence want remove words",
		},
		{
			"Russian sentence",
			"Это просто пример предложения, и из него нужно удалить некоторые слова.",
			"russian",
			"просто пример предложения него нужно удалить некоторые слова",
		},
		{
			"Unknown language",
			"This is a sentence.",
			LangUnknown,
			"This is a sentence.", // Should return original
		},
		{
			"Unsupported language",
			"Sentence in unsupported language.",
			"polish",
			"Sentence in unsupported language.", // Should return original
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cleaned := RemoveStopWords(tc.sentence, tc.lang)
			if cleaned != tc.expected {
				t.Errorf("Expected cleaned sentence '%s', but got '%s'", tc.expected, cleaned)
			}
		})
	}
}

// ADDED: Test for the new StemTokens function
func TestStemTokens(t *testing.T) {
	testCases := []struct {
		name     string
		lang     string
		tokens   []string
		expected []string
	}{
		{
			name:     "English stemming",
			lang:     "english",
			tokens:   []string{"running", "nationalization", "cats", "beautifully"},
			expected: []string{"runn", "nation", "cat", "beautifully"}, // 'running' -> 'runn' because 'ing' is removed
		},
		{
			name:     "Russian stemming",
			lang:     "russian",
			tokens:   []string{"машинами", "хороший", "бегала"},
			expected: []string{"машин", "хорош", "бегал"},
		},
		{
			name:     "German stemming with prefix",
			lang:     "german",
			tokens:   []string{"gemacht", "kinder"},
			expected: []string{"macht", "kind"},
		},
		{
			name:     "Language with no stemming rules",
			lang:     "vietnamese",
			tokens:   []string{"chạy", "nhảy"},
			expected: []string{"chạy", "nhảy"}, // Should be unchanged
		},
		{
			name:     "Word shorter than min_len",
			lang:     "english",
			tokens:   []string{"is", "on", "running"},
			expected: []string{"is", "on", "runn"}, // "is" and "on" are too short
		},
		{
			name:     "Empty input",
			lang:     "english",
			tokens:   []string{},
			expected: []string{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			stemmed := StemTokens(tc.tokens, tc.lang)
			if !reflect.DeepEqual(stemmed, tc.expected) {
				t.Errorf("Expected stemmed tokens %v, but got %v", tc.expected, stemmed)
			}
		})
	}
}
