# Semseg

[![Go Report Card](https://goreportcard.com/badge/github.com/cmsdko/semseg)](https://goreportcard.com/report/github.com/cmsdko/semseg)
[![Go.Dev Reference](https://img.shields.io/badge/go.dev-reference-blue?logo=go&logoColor=white)](https://pkg.go.dev/github.com/cmsdko/semseg)
[![Build Status](https://github.com/cmsdko/semseg/actions/workflows/go.yml/badge.svg)](https://github.com/cmsdko/semseg/actions/workflows/go.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Semseg** is a lightweight, zero-dependency Go library for splitting text into semantically coherent chunks.  
It supports multi-language stopword removal, stemming, and abbreviation normalization — all in pure Go.  
Perfect for preprocessing text in RAG (Retrieval-Augmented Generation) pipelines, summarization, or any NLP tasks.

## Features

- **Semantic Splitting**: Splits at points of low semantic similarity while keeping related content together.
- **Two Embedding Modes**:
    - **Default (TF-IDF)**: Fast, lightweight, zero-dependency classical approach.
    - **External (Ollama)**: Use modern embedding models via Ollama for higher accuracy.
- **Strict Token Limit**: Ensures no chunk exceeds `MaxTokens` (unless a single sentence is larger).
- **Language Awareness**: Automatic or manual language detection, stopword removal, stemming, and abbreviation handling (for TF-IDF mode).
- **Configurable**: Fine‑tune similarity thresholds, detection modes, and preprocessing options.
- **Zero Dependencies**: 100% Go, no external models or libraries (in TF-IDF mode).
- **Fast and Lightweight**: Classic TF‑IDF approach, optimized for CPU workloads.

## Installation

```sh
go get github.com/cmsdko/semseg
```

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	"github.com/cmsdko/semseg"
)

func main() {
	text := `The solar system consists of the Sun and the planets. A rocket journey to other planets takes a long time. The ocean covers most of the Earth's surface. Amazing creatures live in the depths of the ocean.`

	opts := semseg.Options{
		MaxTokens: 15,
	}

	chunks, err := semseg.Segment(text, opts)
	if err != nil {
		log.Fatal(err)
	}

	for i, c := range chunks {
		fmt.Printf("--- Chunk %d (%d tokens) ---\n%s\n\n", i+1, c.NumTokens, c.Text)
	}
}
```

### Expected Output

```
--- Chunk 1 (14 tokens) ---
The solar system consists of the Sun and the planets. A rocket journey to other planets takes a long time.

--- Chunk 2 (13 tokens) ---
The ocean covers most of the Earth's surface. Amazing creatures live in the depths of the ocean.
```

## How It Works

1. **Sentence Splitting** → text is divided into sentences (multi‑language aware).
2. **Normalization** → abbreviations like `U.S.A.` or `т.е.` are normalized before splitting.
3. **Stopword Removal & Stemming** → optional preprocessing to reduce noise.
4. **Vectorization** → each sentence is turned into a TF‑IDF vector.
5. **Cohesion Scoring** → cosine similarity between adjacent sentences is calculated.
6. **Boundary Detection** → splits occur at local minima or below thresholds.
7. **Chunk Assembly** → sentences grouped into chunks respecting `MaxTokens`.

## Processing Pipeline (based on Options)

Depending on `Options`, the pipeline adapts as follows:

- **Language Detection**
    - `Language` set → skip detection, force specific language.
    - `LanguageDetectionTokens > 0` → detect language from first *N* tokens (slower, but enables use of JSON-based contractions/stopwords).
    - `LanguageDetectionMode` → choose detection strategy (`first_sentence`, `first_ten_sentences`, `per_sentence`, `full_text`).
    - ⚡ For **performance**, prefer `first_sentence` or `full_text`.
    - 🧩 For **flexibility**, use token-based detection — it allows leveraging custom stopwords and abbreviations.

- **Abbreviation Normalization**
    - Controlled by `PreNormalizeAbbreviations`.
    - Removes dots in known contractions and acronyms (configurable in JSON).

- **Stopword Removal**
    - Controlled by `EnableStopWordRemoval`.
    - Uses stopwords from `internal/lang/data/stopwords.json`.
    - You can **add/remove languages or stopwords** by editing this JSON.

- **Stemming**
    - Controlled by `EnableStemming`.
    - Uses simple affix-based rules per language, defined in JSON.

- **Chunk Assembly**
    - Always respects `MaxTokens`.
    - Splits on semantic boundaries or when exceeding the limit.

👉 The `stopwords.json` file is intentionally **user-editable**: you can remove or add words and even define new languages with custom rules. This makes the library flexible without depending on external NLP libraries.

## API Example

You can run Semseg as an HTTP API using the provided `example` project:

```sh
docker-compose -f example/docker-compose.yml up --build
```

Then send a request:

```sh
curl -X POST http://localhost:8080/segment -d '{
  "text": "Mars is red. Venus is hot. The ocean is blue.",
  "max_tokens": 10
}' -H "Content-Type: application/json"
```

Response:

```json
{
  "options_used":{
    "max_tokens":10,
    "min_split_similarity":0,
    "depth_threshold":0.1,
    "language":"",
    "language_detection_mode":"first_sentence",
    "language_detection_tokens":0,
    "pre_normalize_abbreviations":true,
    "enable_stop_word_removal":true,
    "enable_stemming":true
  },
  "chunks":[
    {
      "Text":"Mars is red. Venus is hot. The ocean is blue.",
      "Sentences":[
        "Mars is red.",
        "Venus is hot.",
        "The ocean is blue."
      ],
      "NumTokens":10
    }
  ],
  "stats":{
    "total_chunks":1,
    "total_tokens":10,
    "processing_time_ms":0.078,
    "chunks_per_second":12705.027379334004,
    "tokens_per_second":127050.27379334003
  }
}
```
## Known Limitations

- **Chinese and other CJK languages**:  
  The library does not implement word segmentation for Han/Hiragana/Katakana/Hangul scripts.  
  For these languages, stopword removal and stemming are not applied, and language detection will usually return `unknown`.  
  Result: text is still split into sentences, but semantic cohesion may be poor.

- **Language limit (64 max)**:  
  Internally, the library uses a `uint64` bitmask to optimize stopword lookups.  
  This means no more than 64 languages can be supported at once.  
  If `stopwords.json` defines more than 64, initialization will fail with a fatal error.

## Contributing

Contributions are welcome! Open issues or PRs to suggest features, report bugs, or improve language resources.

## License

MIT License. See [LICENSE](LICENSE).
