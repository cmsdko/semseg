# Semseg

[![Go Report Card](https://goreportcard.com/badge/github.com/cmsdko/semseg)](https://goreportcard.com/report/github.com/cmsdko/semseg)
[![Go.Dev Reference](https://img.shields.io/badge/go.dev-reference-blue?logo=go&logoColor=white)](https://pkg.go.dev/github.com/cmsdko/semseg)
[![Build Status](https://github.com/cmsdko/semseg/actions/workflows/go.yml/badge.svg)](https://github.com/cmsdko/semseg/actions/workflows/go.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Semseg** is a lightweight, zero-dependency Go library for splitting text into semantically coherent chunks. It's designed for speed and simplicity, making it perfect for preprocessing text for RAG (Retrieval-Augmented Generation) pipelines, summarization, and other NLP tasks, all while running efficiently on a CPU.

## Features

- **Semantic Splitting**: Chunks are divided at points of low semantic similarity, keeping related content together.
- **Strict Token Limit**: Guarantees that no chunk exceeds a specified `MaxTokens` limit (unless a single sentence is larger than the limit).
- **Zero Dependencies**: Pure Go implementation. Just add it to your project and go.
- **Fast and Lightweight**: Uses a classic TF-IDF approach, ideal for CPU-bound applications without heavy model downloads.

## Installation

```sh
go get github.com/cmsdko/semseg
```

## Quick Start

Here's a simple example of how to use `semseg` to split a text where the topic changes.

```go
package main

import (
	"fmt"
	"log"

	"github.com/cmsdko/semseg"
)

func main() {
	text := `The solar system consists of the Sun and the planets. A rocket journey to other planets takes a long time. The ocean covers most of the Earth's surface. Amazing creatures live in the depths of the ocean.`

	// Configure the segmenter with a token limit.
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

The library correctly identifies the topic change and creates two distinct chunks.

```
--- Chunk 1 (14 tokens) ---
The solar system consists of the Sun and the planets. A rocket journey to other planets takes a long time.

--- Chunk 2 (13 tokens) ---
The ocean covers most of the Earth's surface. Amazing creatures live in the depths of the ocean.

```

## How It Works

The library follows a simple, robust algorithm:

1.  **Sentence Splitting**: The input text is first divided into individual sentences.
2.  **Vectorization**: Each sentence is converted into a TF-IDF vector. This vector represents the sentence's topic based on its word frequencies relative to the entire text.
3.  **Cohesion Scoring**: The cosine similarity between adjacent sentence vectors is calculated. A high score means the topic continues; a low score suggests a topic change.
4.  **Boundary Detection**: The algorithm identifies significant "dips" in the similarity scores, marking these points as semantic boundaries.
5.  **Chunk Assembly**: Sentences are grouped into chunks, with splits occurring at the identified semantic boundaries or wherever necessary to strictly adhere to the `MaxTokens` limit.

## Configuration

You can customize the segmentation behavior by passing an `Options` struct to the `Segment` function.

- `MaxTokens` (int): **Required.** The hard limit on the number of tokens per chunk. The library will never create a chunk larger than this, with one exception: a single sentence that already exceeds `MaxTokens` will be placed in its own chunk.

- `MinSplitSimilarity` (float64): An optional fixed threshold (0.0 to 1.0) for splitting. If the cosine similarity between two sentences is below this value, a split will be considered. If set to `0` (the default), the more robust local minima detection method is used instead.

- `DepthThreshold` (float64): Used by the default boundary detection method. It defines the minimum "depth" of a similarity dip to be considered a valid split point. This prevents minor, insignificant fluctuations from causing a split. Default is `0.1`.

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or suggest a feature, or submit a pull request with improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
