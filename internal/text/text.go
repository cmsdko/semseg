package text

import (
	"regexp"
	"strings"
)

// Sentence boundary detection regex:
// - Matches sentence-ending punctuation: . ! ? …
// - Followed by optional closing quotes/brackets: " " » '
// - Then either whitespace or end of string
// - Handles multilingual punctuation and quote styles
var sentenceEndRegex = regexp.MustCompile(`([.!?…])([”"»']*)\s+|([.!?…])([”"»']*)$`)

// Token cleaning regex using Unicode character classes:
// - \p{L}: Unicode letters from any language (Latin, Cyrillic, Chinese, etc.)
// - \p{N}: Unicode numbers from any language
// - Preserves hyphens and apostrophes for compound words and contractions
// - Removes all other punctuation and special characters
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
