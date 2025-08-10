package text

import (
	"regexp"
	"strings"
)

var sentenceEndRegex = regexp.MustCompile(`([.!?])(\s+|$)`)

// This regex uses Unicode properties:
// \p{L} - any letter from any language
// \p{N} - any number from any language
var tokenizeCleanRegex = regexp.MustCompile(`[^\p{L}\p{N}\s-]`)

func SplitSentences(text string) []string {
	delimited := sentenceEndRegex.ReplaceAllString(text, "$1|")
	sentencesRaw := strings.Split(delimited, "|")
	var sentences []string
	for _, s := range sentencesRaw {
		trimmed := strings.TrimSpace(s)
		if trimmed != "" {
			sentences = append(sentences, trimmed)
		}
	}
	return sentences
}

func Tokenize(text string) []string {
	lower := strings.ToLower(text)
	cleaned := tokenizeCleanRegex.ReplaceAllString(lower, "")
	return strings.Fields(cleaned)
}
