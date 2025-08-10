package text

import (
	"regexp"
	"strings"
)

var sentenceEndRegex = regexp.MustCompile(`([.!?])(\s+|$)`)
var tokenizeCleanRegex = regexp.MustCompile(`[^a-z0-9\s-]`)

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
