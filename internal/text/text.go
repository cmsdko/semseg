package text

import (
	"regexp"
	"strings"
)

// End of sentence: ., !, ?, … followed by optional closing quotes/brackets then space or end.
// Handles cases like: 'Hello world."', «Привет мир!», 你好。
var sentenceEndRegex = regexp.MustCompile(`([.!?…])([”"»']*)\s+|([.!?…])([”"»']*)$`)

// This regex uses Unicode properties:
// \p{L} - any letter from any language
// \p{N} - any number from any language
// Allow hyphen and apostrophe inside terms.
var tokenizeCleanRegex = regexp.MustCompile(`[^\p{L}\p{N}\s\-']`)

func SplitSentences(text string) []string {
	// Insert delimiter at sentence boundaries, preserving terminal marks
	delimited := sentenceEndRegex.ReplaceAllString(text, "$1$2$3$4|")
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
	parts := strings.Fields(cleaned)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		// Trim apostrophes/dashes only at edges to keep don't, l'état, world-123
		p = strings.Trim(p, "'")
		p = strings.Trim(p, "-")
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
