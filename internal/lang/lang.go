// ./internal/lang/lang.go
package lang

import (
	_ "embed"
	"encoding/json"
	"log"
	"sort"
	"strings"
	"unicode"

	"github.com/cmsdko/semseg/internal/text"
)

// --- DATA STRUCTURES ---

// StemmingRules defines the stemming configuration for a language.
type StemmingRules struct {
	Prefixes []string `json:"prefixes"`
	Suffixes []string `json:"suffixes"`
	MinLen   int      `json:"min_len"`
	OneShot  bool     `json:"one_shot"`
}

// LanguageData holds all data for a single language from the JSON file.
type LanguageData struct {
	Stopwords []string      `json:"stopwords"`
	Stemming  StemmingRules `json:"stemming"`
}

// --- EMBEDDED DATA ---

//go:embed ../data/stopwords.json
var stopWordsJSON []byte

// --- CONSTANTS AND GLOBAL VARIABLES ---

const (
	LangUnknown         = "unknown"
	ConfidenceThreshold = 2
)

// Script constants to avoid typos in the code.
const (
	scriptLatin      = "Latin"
	scriptCyrillic   = "Cyrillic"
	scriptArabic     = "Arabic"
	scriptGreek      = "Greek"
	scriptDevanagari = "Devanagari"
	scriptHebrew     = "Hebrew"
	scriptHan        = "Han"
	scriptHiragana   = "Hiragana"
	scriptKatakana   = "Katakana"
	scriptHangul     = "Hangul"
)

var (
	invertedIndexMask   map[string]uint64
	stopWordsByLang     map[string]map[string]struct{}
	stemmingRulesByLang map[string]StemmingRules
	languageMasks       map[string]uint64

	// langsByScript stores which languages use a specific script, built at init time.
	// Example: "Cyrillic" -> ["russian", "ukrainian"]
	langsByScript map[string][]string
	// allLangsList is a simple slice of all loaded languages for fallback.
	allLangsList []string
)

// --- INITIALIZATION (runs once at startup) ---

func init() {
	// 1. Load and parse stop words and stemming rules from the embedded JSON file
	var rawData map[string]LanguageData
	if err := json.Unmarshal(stopWordsJSON, &rawData); err != nil {
		log.Fatalf("FATAL: Failed to parse embedded stopwords.json: %v", err)
	}

	// 2. Dynamically determine the list of supported languages from the JSON keys.
	var languageOrder []string
	for lang, data := range rawData {
		if len(data.Stopwords) > 0 {
			languageOrder = append(languageOrder, lang)
		}
	}
	sort.Strings(languageOrder)
	allLangsList = languageOrder // Save for fallback cases

	if len(languageOrder) > 64 {
		log.Fatalf("FATAL: Cannot support more than 64 languages due to uint64 bitmask limit. Found %d.", len(languageOrder))
	}

	// 3. Create masks for each dynamically discovered language
	languageMasks = make(map[string]uint64)
	for i, lang := range languageOrder {
		languageMasks[lang] = 1 << uint(i)
	}

	// 4. Initialize the main data structures
	invertedIndexMask = make(map[string]uint64)
	stopWordsByLang = make(map[string]map[string]struct{})
	stemmingRulesByLang = make(map[string]StemmingRules)
	langsByScript = make(map[string][]string)

	// Dynamically determine the script for each language based on its stop words.
	for lang, data := range rawData {
		if _, exists := languageMasks[lang]; !exists {
			continue // Skip languages with no stop words
		}

		determinedScript := scriptLatin // Default to Latin script
	wordLoop:
		for _, word := range data.Stopwords {
			for _, r := range word {
				// Check the rune against various Unicode scripts.
				// Once a non-Latin script is found, we can classify the language and stop checking.
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
		// Append the language to the list for its determined script.
		langsByScript[determinedScript] = append(langsByScript[determinedScript], lang)
	}

	// 5. Populate the main data structures for stop word removal and stemming.
	for lang, data := range rawData {
		langMask, maskExists := languageMasks[lang]
		if !maskExists {
			continue
		}

		// Populate stop words
		wordSet := make(map[string]struct{}, len(data.Stopwords))
		for _, word := range data.Stopwords {
			wordSet[word] = struct{}{}
			invertedIndexMask[word] |= langMask
		}
		stopWordsByLang[lang] = wordSet

		// Populate stemming rules, sorting affixes for correctness
		rules := data.Stemming
		// Sort by length descending to match longer affixes first (e.g., "ational" before "ate")
		sort.Slice(rules.Prefixes, func(i, j int) bool {
			return len(rules.Prefixes[i]) > len(rules.Prefixes[j])
		})
		sort.Slice(rules.Suffixes, func(i, j int) bool {
			return len(rules.Suffixes[i]) > len(rules.Suffixes[j])
		})
		stemmingRulesByLang[lang] = rules
	}

	// 6. Log the initialization results for debugging and verification.
	log.Printf("Initialized stop-word removal and stemming for languages: %v", languageOrder)
	for script, langs := range langsByScript {
		log.Printf("Detected script '%s' for languages: %v", script, langs)
	}
}

// --- CORE FUNCTIONS ---

// DetectLanguage identifies the language of a sentence.
// Uses text.Tokenize as the single tokenizer to ensure consistent behavior across packages.
func DetectLanguage(sentence string) string {
	// First, narrow down the potential languages based on the script.
	candidateLangs := getCandidateLangs(sentence)

	tokens := text.Tokenize(sentence)
	if len(tokens) == 0 {
		return LangUnknown
	}

	// Score each candidate language by counting stop word occurrences.
	scores := make(map[string]int)
	for _, token := range tokens {
		if mask, found := invertedIndexMask[token]; found {
			// The candidate list is a performance optimization.
			// The main check is whether the language was loaded into languageMasks.
			for lang := range languageMasks {
				// Check if the bit for this language is set in the word's mask
				// and if the language is in our candidate list.
				if (mask&languageMasks[lang]) != 0 && isCandidate(lang, candidateLangs) {
					scores[lang]++
				}
			}
		}
	}

	// Decide on the best language based on the scores.
	if len(scores) == 0 {
		return LangUnknown
	}

	bestLang := LangUnknown
	maxScore := ConfidenceThreshold - 1 // A language needs at least ConfidenceThreshold matches.
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

// RemoveStopWords removes stop words from a sentence for a given language.
// If the language is not supported or unknown, it returns the original sentence.
// Uses the unified tokenizer from internal/text.
func RemoveStopWords(sentence string, language string) string {
	stopWords, ok := stopWordsByLang[language]
	if !ok || language == LangUnknown {
		return sentence // Return as is if language is unknown or unsupported
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

// StemTokens applies stemming rules to a slice of tokens for a given language.
// It returns a new slice with the stemmed tokens.
func StemTokens(tokens []string, language string) []string {
	rules, ok := stemmingRulesByLang[language]
	// Return original tokens if language has no stemming rules or is unknown.
	if !ok || (len(rules.Prefixes) == 0 && len(rules.Suffixes) == 0) {
		return tokens
	}

	stemmedTokens := make([]string, len(tokens))
	for i, token := range tokens {
		stemmedTokens[i] = stemWord(token, rules)
	}
	return stemmedTokens
}

// stemWord applies stemming rules to a single word.
func stemWord(word string, rules StemmingRules) string {
	// Do not stem if the word is too short.
	if len(word) < rules.MinLen {
		return word
	}

	stemmed := word

	// Apply prefix stemming
	for _, prefix := range rules.Prefixes {
		if strings.HasPrefix(stemmed, prefix) {
			stemmed = strings.TrimPrefix(stemmed, prefix)
			// If one_shot is true, only remove the first matching affix.
			if rules.OneShot {
				break
			}
		}
	}

	// Re-check length after prefix removal before trying to remove suffix.
	if len(stemmed) < rules.MinLen {
		return stemmed
	}

	// Apply suffix stemming
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

// getCandidateLangs narrows down the list of possible languages by detecting the script of the input string.
// This is a performance heuristic that returns all loaded languages that use the detected script.
func getCandidateLangs(s string) []string {
	// Iterate through the string to detect a non-Latin script.
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
				// Return the list of languages for the detected script.
				return candidates
			}
		}
	}

	// If no specific script is detected, default to languages using the Latin script.
	if candidates, ok := langsByScript[scriptLatin]; ok && len(candidates) > 0 {
		return candidates
	}

	// As a last resort (e.g., if only a Cyrillic language is loaded but the text is Latin),
	// return all loaded languages to allow for a match.
	return allLangsList
}

// isCandidate checks if a language is in the list of candidates.
func isCandidate(lang string, candidates []string) bool {
	for _, c := range candidates {
		if c == lang {
			return true
		}
	}
	return false
}

