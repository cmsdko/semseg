package lang

import (
	"reflect"
	"testing"
)

// TestDetectLanguage validates language detection across supported cases.
// Includes positive examples, ambiguous input, too-short sentences, and empty string.
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
		{"Ambiguous (tie)", "Hotel in Berlin", LangUnknown}, // could match multiple languages
		{"Vietnamese", "Đi đâu đó", "vietnamese"},
		{"Too short (below threshold)", "Please go.", LangUnknown},
		{"Empty input", "", LangUnknown},
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

// TestRemoveStopWords checks stopword removal for supported and unsupported languages.
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
			"просто пример предложения нужно удалить некоторые слова",
		},
		{
			"Unknown language constant",
			"This is a sentence.",
			LangUnknown,
			"This is a sentence.", // Should return original unchanged
		},
		{
			"Unsupported language string",
			"Sentence in unsupported language.",
			"polish",
			"Sentence in unsupported language.", // Should return original unchanged
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

// TestStemTokens verifies stemming behavior for different languages and edge cases.
// Uses lightweight affix-based rules defined in stopwords.json.
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
			expected: []string{"runn", "nation", "cat", "beautiful"}, // suffixes stripped
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
			name:     "Language without stemming rules",
			lang:     "vietnamese",
			tokens:   []string{"chạy", "nhảy"},
			expected: []string{"chạy", "nhảy"}, // unchanged
		},
		{
			name:     "Words shorter than MinLen",
			lang:     "english",
			tokens:   []string{"is", "on", "running"},
			expected: []string{"is", "on", "runn"}, // short words remain intact
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
