package text

import (
	"regexp"
	"strings"
)

// sentenceEndRegex detects sentence boundaries.
// - Matches terminal punctuation: . ! ? …
// - Allows trailing closing quotes/brackets: ” " » '
// - Followed by whitespace or end of string
// - Supports multilingual punctuation styles
var sentenceEndRegex = regexp.MustCompile(`([.!?…])([”"»']*)\s+|([.!?…])([”"»']*)$`)

// tokenizeCleanRegex removes unwanted characters from tokens.
// - Keeps Unicode letters (\p{L}) and numbers (\p{N})
// - Preserves internal hyphens and apostrophes
// - Strips other punctuation and symbols
var tokenizeCleanRegex = regexp.MustCompile(`[^\p{L}\p{N}\s\-']`)

// Decimal dot protection.
// Before sentence splitting, protect number patterns like "3.14"
// so they are not mistaken for sentence boundaries.
var (
	reDecimalDot    = regexp.MustCompile(`(\d)\.(\d)`)
	decimalDotToken = "\uE001DECIMAL_DOT\uE001"
)

// SplitSentences splits text into sentences based on punctuation rules.
// - Protects decimal numbers (3.14) before splitting
// - Restores them after splitting
// - Trims whitespace around sentences
func SplitSentences(text string) []string {
	// Protect decimal dots so they are not treated as boundaries.
	protected := reDecimalDot.ReplaceAllString(text, `$1`+decimalDotToken+`$2`)

	// Insert delimiter at sentence boundaries, keeping punctuation.
	delimited := sentenceEndRegex.ReplaceAllString(protected, "$1$2$3$4|")
	sentencesRaw := strings.Split(delimited, "|")
	var sentences []string
	for _, s := range sentencesRaw {
		trimmed := strings.TrimSpace(s)
		if trimmed != "" {
			// Restore decimal dots.
			trimmed = strings.ReplaceAll(trimmed, decimalDotToken, ".")
			sentences = append(sentences, trimmed)
		}
	}
	return sentences
}

// Tokenize normalizes text into a canonical token stream.
// - Converts to lowercase
// - Keeps letters and numbers from any script
// - Preserves internal hyphens/apostrophes (e.g. don't, l'état, world-123)
// - Trims apostrophes/hyphens only at token edges
// This is the single source of truth for tokenization used by lang.* and semseg.*.
func Tokenize(text string) []string {
	lower := strings.ToLower(text)
	cleaned := tokenizeCleanRegex.ReplaceAllString(lower, "")
	parts := strings.Fields(cleaned)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		// Trim apostrophes and dashes only at edges.
		p = strings.Trim(p, "'")
		p = strings.Trim(p, "-")
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
