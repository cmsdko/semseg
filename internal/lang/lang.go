package lang

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"
	"unicode"

	"github.com/cmsdko/semseg/internal/text"
)

// --- DATA STRUCTURES ---

// StemmingRules defines minimal, affix-based stemming configuration for a language.
// The goal is lightweight normalization, not linguistic accuracy.
type StemmingRules struct {
	Prefixes []string `json:"prefixes"` // checked longest-first
	Suffixes []string `json:"suffixes"` // checked longest-first
	MinLen   int      `json:"min_len"`  // do not stem if resulting token would be shorter than this
	OneShot  bool     `json:"one_shot"` // if true, stop after the first successful prefix/suffix removal
}

// LanguageData groups all language resources loaded from JSON.
type LanguageData struct {
	Stopwords    []string      `json:"stopwords"`
	Stemming     StemmingRules `json:"stemming"`
	Contractions []string      `json:"contractions"` // dotted contractions for abbreviation normalization
}

// --- EMBEDDED DATA ---

//go:embed data/stopwords.json
var stopWordsJSON []byte

// --- CONSTANTS AND GLOBAL STATE ---

const (
	// LangUnknown is returned when language detection is inconclusive.
	LangUnknown = "unknown"

	// ConfidenceThreshold is the minimal count of stopword "hits" for a language
	// to be considered at all. Below this, the input is treated as unknown.
	// Small, short, or mixed-language inputs often won't reach this threshold.
	ConfidenceThreshold = 2
)

// Script constants used to narrow language candidates quickly via Unicode script checks.
const (
	scriptLatin      = "Latin"
	scriptCyrillic   = "Cyrillic"
	scriptArabic     = "Arabic"
	scriptGreek      = "Greek"
	scriptDevanagari = "Devanagari"
	scriptHebrew     = "Hebrew"
	scriptHan        = "Han"      // Chinese characters
	scriptHiragana   = "Hiragana" // Japanese
	scriptKatakana   = "Katakana" // Japanese
	scriptHangul     = "Hangul"   // Korean
)

var (
	// invertedIndexMask maps a token to a bitmask of languages that list it as a stopword.
	// Bit positions are assigned per language in languageMasks.
	invertedIndexMask map[string]uint64

	// stopWordsByLang stores stopword sets per language (for fast membership checks).
	stopWordsByLang map[string]map[string]struct{}

	// stemmingRulesByLang stores per-language stemming rules.
	stemmingRulesByLang map[string]StemmingRules

	// languageMasks assigns each supported language a unique bit in a 64-bit mask.
	// This implementation intentionally caps at 64 languages for simplicity/perf.
	languageMasks map[string]uint64

	// contractionsByLang stores dotted contractions per language (e.g., "e.g.", "т.е.").
	contractionsByLang map[string][]string

	// langsByScript lists languages mapped to a detected script (heuristic; built from stopwords).
	// Example: "Cyrillic" -> ["russian", "ukrainian"]
	langsByScript map[string][]string

	// allLangsList is a stable list of all loaded languages (fallback when script is unknown).
	allLangsList []string
)

// --- INITIALIZATION (runs once at startup) ---

func init() {
	// Load language resources from the embedded JSON blob.
	var rawData map[string]LanguageData
	if err := json.Unmarshal(stopWordsJSON, &rawData); err != nil {
		panic(fmt.Sprintf("semseg: invalid embedded stopwords.json: %v", err))
	}

	// Build a stable, sorted list of languages that actually have stopwords.
	var languageOrder []string
	for lang, data := range rawData {
		if len(data.Stopwords) > 0 {
			languageOrder = append(languageOrder, lang)
		}
	}
	sort.Strings(languageOrder)
	allLangsList = languageOrder

	// Hard cap: uint64 bitmask allows at most 64 languages.
	if len(languageOrder) > 64 {
		log.Fatalf("FATAL: Cannot support more than 64 languages due to uint64 bitmask limit. Found %d.", len(languageOrder))
	}

	// Assign bit positions for each language.
	languageMasks = make(map[string]uint64)
	for i, lang := range languageOrder {
		languageMasks[lang] = 1 << uint(i)
	}

	// Prepare core structures.
	invertedIndexMask = make(map[string]uint64)
	stopWordsByLang = make(map[string]map[string]struct{})
	stemmingRulesByLang = make(map[string]StemmingRules)
	contractionsByLang = make(map[string][]string)
	langsByScript = make(map[string][]string)

	// Heuristically determine the primary script used by each language from its stopwords.
	for lang, data := range rawData {
		if _, exists := languageMasks[lang]; !exists {
			continue // skip languages without stopwords or beyond the 64-cap
		}

		determinedScript := scriptLatin // default
	wordLoop:
		for _, word := range data.Stopwords {
			for _, r := range word {
				switch {
				case unicode.Is(unicode.Cyrillic, r):
					determinedScript = scriptCyrillic
					break wordLoop
				case unicode.Is(unicode.Arabic, r):
					determinedScript = scriptArabic
					break wordLoop
				case unicode.Is(unicode.Greek, r):
					determinedScript = scriptGreek
					break wordLoop
				case unicode.Is(unicode.Devanagari, r):
					determinedScript = scriptDevanagari
					break wordLoop
				case unicode.Is(unicode.Hebrew, r):
					determinedScript = scriptHebrew
					break wordLoop
				case unicode.Is(unicode.Han, r):
					determinedScript = scriptHan
					break wordLoop
				case unicode.Is(unicode.Katakana, r):
					determinedScript = scriptKatakana
					break wordLoop
				case unicode.Is(unicode.Hiragana, r):
					determinedScript = scriptHiragana
					break wordLoop
				case unicode.Is(unicode.Hangul, r):
					determinedScript = scriptHangul
					break wordLoop
				}
			}
		}
		langsByScript[determinedScript] = append(langsByScript[determinedScript], lang)
	}

	// Build stopword sets, inverted index, stemming rules, and contractions.
	for lang, data := range rawData {
		langMask, ok := languageMasks[lang]
		if !ok {
			continue
		}

		// Stopwords → set + inverted index for language mask aggregation.
		wordSet := make(map[string]struct{}, len(data.Stopwords))
		for _, word := range data.Stopwords {
			wordSet[word] = struct{}{}
			invertedIndexMask[word] |= langMask
		}
		stopWordsByLang[lang] = wordSet

		// Stemming rules: sort affixes by length (longest-first) for more stable stripping.
		rules := data.Stemming
		sort.Slice(rules.Prefixes, func(i, j int) bool { return len(rules.Prefixes[i]) > len(rules.Prefixes[j]) })
		sort.Slice(rules.Suffixes, func(i, j int) bool { return len(rules.Suffixes[i]) > len(rules.Suffixes[j]) })
		stemmingRulesByLang[lang] = rules

		// Dotted contractions (used by abbreviation normalization).
		if len(data.Contractions) > 0 {
			contractionsByLang[lang] = append([]string(nil), data.Contractions...)
		}
	}
}

// --- CORE FUNCTIONS ---

// DetectLanguage returns the most likely language for a sentence based on stopword hits.
// Steps:
// 1) Narrow candidates by Unicode script (heuristic).
// 2) Tokenize via text.Tokenize (shared tokenizer across the library).
// 3) Count stopword matches per candidate language using an inverted index + bitmasks.
// 4) If the best score < ConfidenceThreshold or there is a tie for best, return "unknown".
func DetectLanguage(sentence string) string {
	// 1) Narrow by script to reduce comparisons.
	candidateLangs := getCandidateLangs(sentence)

	// 2) Tokenize with the canonical tokenizer.
	tokens := text.Tokenize(sentence)
	if len(tokens) == 0 {
		return LangUnknown
	}

	// 3) Score candidates by stopword occurrences.
	scores := make(map[string]int)
	for _, token := range tokens {
		if mask, found := invertedIndexMask[token]; found {
			for lang := range languageMasks {
				if (mask&languageMasks[lang]) != 0 && isCandidate(lang, candidateLangs) {
					scores[lang]++
				}
			}
		}
	}

	// No matches at all → unknown.
	if len(scores) == 0 {
		return LangUnknown
	}

	// 4) Pick the best score with a minimal confidence threshold and tie handling.
	bestLang := LangUnknown
	maxScore := ConfidenceThreshold - 1
	isTie := false

	for lang, score := range scores {
		if score > maxScore {
			maxScore = score
			bestLang = lang
			isTie = false
		} else if score == maxScore && maxScore > 0 {
			isTie = true
		}
	}

	if isTie {
		return LangUnknown
	}
	return bestLang
}

// RemoveStopWords removes known stopwords for the specified language.
// If the language is unknown/unsupported, the original sentence is returned.
func RemoveStopWords(sentence string, language string) string {
	stopWords, ok := stopWordsByLang[language]
	if !ok || language == LangUnknown {
		return sentence
	}

	tokens := text.Tokenize(sentence)
	resultTokens := make([]string, 0, len(tokens))

	for _, token := range tokens {
		if _, isStopWord := stopWords[token]; !isStopWord {
			resultTokens = append(resultTokens, token)
		}
	}
	return strings.Join(resultTokens, " ")
}

// StemTokens applies lightweight stemming to tokens for the given language.
// Rules are affix-based and may over-stem in edge cases; this is by design for speed/simplicity.
func StemTokens(tokens []string, language string) []string {
	rules, ok := stemmingRulesByLang[language]
	if !ok || (len(rules.Prefixes) == 0 && len(rules.Suffixes) == 0) {
		return tokens
	}

	stemmedTokens := make([]string, len(tokens))
	for i, token := range tokens {
		stemmedTokens[i] = stemWord(token, rules)
	}
	return stemmedTokens
}

// stemWord strips a single word using the language's prefix/suffix rules.
// The function honors MinLen and OneShot to avoid over-aggressive stripping.
func stemWord(word string, rules StemmingRules) string {
	// Guard: do not stem if the word is too short.
	if len(word) < rules.MinLen {
		return word
	}

	stemmed := word

	// Try prefix removal.
	for _, prefix := range rules.Prefixes {
		if strings.HasPrefix(stemmed, prefix) {
			stemmed = strings.TrimPrefix(stemmed, prefix)
			if rules.OneShot {
				break
			}
		}
	}

	// Re-check length after prefix removal.
	if len(stemmed) < rules.MinLen {
		return stemmed
	}

	// Try suffix removal.
	for _, suffix := range rules.Suffixes {
		if strings.HasSuffix(stemmed, suffix) {
			stemmed = strings.TrimSuffix(stemmed, suffix)
			if rules.OneShot {
				break
			}
		}
	}

	return stemmed
}

// --- HELPER FUNCTIONS ---

// getCandidateLangs returns candidate languages by detecting the Unicode script
// used in the input string. If no script is detected, prefer Latin-script languages;
// as a last resort, fall back to all loaded languages.
func getCandidateLangs(s string) []string {
	for _, r := range s {
		var script string
		switch {
		case unicode.Is(unicode.Cyrillic, r):
			script = scriptCyrillic
		case unicode.Is(unicode.Arabic, r):
			script = scriptArabic
		case unicode.Is(unicode.Greek, r):
			script = scriptGreek
		case unicode.Is(unicode.Devanagari, r):
			script = scriptDevanagari
		case unicode.Is(unicode.Hebrew, r):
			script = scriptHebrew
		case unicode.Is(unicode.Han, r):
			script = scriptHan
		case unicode.Is(unicode.Katakana, r):
			script = scriptKatakana
		case unicode.Is(unicode.Hiragana, r):
			script = scriptHiragana
		case unicode.Is(unicode.Hangul, r):
			script = scriptHangul
		}

		if script != "" {
			if candidates, ok := langsByScript[script]; ok {
				return candidates
			}
		}
	}

	// Default: Latin-script languages if available.
	if candidates, ok := langsByScript[scriptLatin]; ok && len(candidates) > 0 {
		return candidates
	}

	// Fallback: all languages.
	return allLangsList
}

// isCandidate returns true if lang exists in the candidates slice.
func isCandidate(lang string, candidates []string) bool {
	for _, c := range candidates {
		if c == lang {
			return true
		}
	}
	return false
}
