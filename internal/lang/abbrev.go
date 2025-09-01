package lang

import (
	"regexp"
	"strings"
)

// Abbreviation/acronym normalization prior to sentence splitting.
//
// What it does:
// - Removes dots from language-specific dotted contractions (from JSON), e.g. "e.g." -> "eg", "т.е." -> "те".
// - Removes dots from ALL-caps dotted acronyms in Latin/Cyrillic scripts, e.g. "U.S.A." -> "USA", "П.Т.О." -> "ПТО".
// - Preserves ellipses ("...") by temporarily masking them.
//
// What it does NOT do:
// - It does not touch numeric decimals (e.g. "3.14") or version/IP patterns; decimal protection is handled in text.SplitSentences.
// - It does not normalize lowercase/TitleCase abbreviations unless they are explicitly listed in contractions JSON.
//
// Notes:
// - Language-specific replacements are applied only if `langCode` is known and present in `contractionsByLang`.
// - Generic dotted-acronym removal is purely regex-based and language-agnostic for Latin/Cyrillic uppercase.
// - Keep this step lightweight: the goal is to avoid false sentence splits on dotted abbreviations, not to fully normalize text.

var (
	reDottedAcronymLatin    = regexp.MustCompile(`\b(?:[A-Z]\.){2,}(?:[A-Z])?\b`)
	reDottedAcronymCyrillic = regexp.MustCompile(`\b(?:[А-ЯЁ]\.){2,}(?:[А-ЯЁ])?\b`)

	ellipsisToken = "\uE000ELLIPSIS\uE000"
	reEllipsis    = regexp.MustCompile(`\.{3,}`)
)

// NormalizeAbbreviations removes dots from known contractions and dotted acronyms.
// Ellipses are preserved by masking them before replacements and restoring afterwards.
func NormalizeAbbreviations(s, langCode string) string {
	if s == "" {
		return s
	}

	// Preserve ellipses so they are not altered by replacements below.
	s = reEllipsis.ReplaceAllString(s, ellipsisToken)

	// Language-specific dotted contractions (from JSON).
	// Applied only when langCode is known and the list is non-empty.
	if list, ok := contractionsByLang[langCode]; ok && len(list) > 0 {
		repl := make([]string, 0, len(list)*2)
		for _, c := range list {
			if strings.Contains(c, ".") {
				repl = append(repl, c, strings.ReplaceAll(c, ".", ""))
			}
		}
		if len(repl) > 0 {
			r := strings.NewReplacer(repl...)
			s = r.Replace(s)
		}
	}

	// Generic dotted acronyms (Latin/Cyrillic, uppercase letters only).
	s = reDottedAcronymLatin.ReplaceAllStringFunc(s, func(m string) string {
		return strings.ReplaceAll(m, ".", "")
	})
	s = reDottedAcronymCyrillic.ReplaceAllStringFunc(s, func(m string) string {
		return strings.ReplaceAll(m, ".", "")
	})

	// Restore ellipses.
	s = strings.ReplaceAll(s, ellipsisToken, "...")

	return s
}
