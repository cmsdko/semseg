// file: ./cache.go

package semseg

import (
	"log"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cmsdko/semseg/internal/tfidf"
)

// EmbeddingCache defines the interface for a semantic cache.
type EmbeddingCache interface {
	Find(key map[string]float64, threshold float64) (embedding []float64, found bool)
	Set(key map[string]float64, embedding []float64, similarityThreshold float64) // Добавили threshold для инкрементального анализа
	AnalyzeSimilarity(threshold float64) int
	Close()
}

// --- ADAPTIVE CACHE MANAGER ---

type AdaptiveCacheManager interface {
	EmbeddingCache
	Start(similarityThreshold float64, activationThreshold int)
	IsActivated() bool
	QueueSet(key map[string]float64, embedding []float64)
}

type adaptiveCacheEntry struct {
	key       map[string]float64
	embedding []float64
}

type adaptiveCacheManager struct {
	cache               EmbeddingCache
	isActivated         atomic.Bool
	startOnce           sync.Once
	setQueue            chan adaptiveCacheEntry
	tickerStop          chan struct{}
	activationThreshold int
	similarityThreshold float64
}

func NewAdaptiveCacheManager(cache EmbeddingCache) AdaptiveCacheManager {
	return &adaptiveCacheManager{
		cache:      cache,
		setQueue:   make(chan adaptiveCacheEntry, 1024),
		tickerStop: make(chan struct{}),
	}
}

func (m *adaptiveCacheManager) Start(similarityThreshold float64, activationThreshold int) {
	m.startOnce.Do(func() {
		log.Println("Starting adaptive cache manager...")
		m.similarityThreshold = similarityThreshold
		m.activationThreshold = activationThreshold
		go m.asyncWriter()
		go m.activationTicker()
	})
}

func (m *adaptiveCacheManager) IsActivated() bool {
	return m.isActivated.Load()
}

func (m *adaptiveCacheManager) QueueSet(key map[string]float64, embedding []float64) {
	select {
	case m.setQueue <- adaptiveCacheEntry{key: key, embedding: embedding}:
	default:
		log.Println("Adaptive cache queue is full, dropping entry.")
	}
}

func (m *adaptiveCacheManager) Find(key map[string]float64, threshold float64) ([]float64, bool) {
	return m.cache.Find(key, threshold)
}

// Set теперь не используется напрямую, т.к. мы передаем threshold
func (m *adaptiveCacheManager) Set(key map[string]float64, embedding []float64, threshold float64) {
	m.cache.Set(key, embedding, threshold)
}

// AnalyzeSimilarity проксирует вызов
func (m *adaptiveCacheManager) AnalyzeSimilarity(threshold float64) int {
	return m.cache.AnalyzeSimilarity(threshold)
}

func (m *adaptiveCacheManager) Close() {
	m.cache.Close()
	m.startOnce.Do(func() {})
	close(m.setQueue)
	select {
	case <-m.tickerStop:
	default:
		close(m.tickerStop)
	}
}

func (m *adaptiveCacheManager) asyncWriter() {
	for entry := range m.setQueue {
		// Передаем threshold в Set для инкрементального анализа
		m.cache.Set(entry.key, entry.embedding, m.similarityThreshold)
	}
}

func (m *adaptiveCacheManager) activationTicker() {
	ticker := time.NewTicker(5 * time.Second) // Можно проверять чаще, т.к. операция стала дешевой
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if m.IsActivated() {
				return
			}
			// Теперь эта операция O(1)
			count := m.cache.AnalyzeSimilarity(m.similarityThreshold)
			if count >= m.activationThreshold {
				log.Printf("Adaptive cache activation threshold reached (%d >= %d). Activating cache.", count, m.activationThreshold)
				m.isActivated.Store(true)
				return
			}
		case <-m.tickerStop:
			return
		}
	}
}

// --- EFFICIENT IN-MEMORY CACHE IMPLEMENTATION (V3) ---

// ... (константы и вспомогательные структуры остаются теми же) ...
const (
	defaultTopK             = 16
	l0FlushThreshold        = 512
	l1CompactionTrigger     = 4
	l1CompactionTargetCount = 2
)

type cacheEntry struct {
	tfidfVector    map[string]float64
	denseEmbedding []float64
}

type termScore struct {
	term  string
	score float64
}

type l1Segment struct {
	entries []cacheEntry
	index   map[string][]int
}

type InMemoryCache struct {
	mu sync.RWMutex

	l0Entries  []cacheEntry
	l1Segments []*l1Segment

	// Новый счетчик для инкрементального анализа
	itemsWithNeighbors atomic.Int64

	topK int

	flushTrigger      chan struct{}
	compactionTrigger chan struct{}
	closeWorker       chan struct{}
}

func NewInMemoryCache() *InMemoryCache {
	c := &InMemoryCache{
		l0Entries:         make([]cacheEntry, 0, l0FlushThreshold),
		l1Segments:        make([]*l1Segment, 0),
		topK:              defaultTopK,
		flushTrigger:      make(chan struct{}, 1),
		compactionTrigger: make(chan struct{}, 1),
		closeWorker:       make(chan struct{}),
	}
	go c.backgroundWorker()
	return c
}

func (c *InMemoryCache) Close() {
	close(c.closeWorker)
}

func (c *InMemoryCache) Set(key map[string]float64, embedding []float64, similarityThreshold float64) {
	c.mu.Lock()

	// Инкрементальный анализ: ищем соседей для нового элемента только в L0
	isNewNeighborFound := false
	for _, entry := range c.l0Entries {
		if tfidf.CosineSimilarity(key, entry.tfidfVector) >= similarityThreshold {
			isNewNeighborFound = true
			break
		}
	}
	if isNewNeighborFound {
		c.itemsWithNeighbors.Add(1)
	}

	embeddingCopy := make([]float64, len(embedding))
	copy(embeddingCopy, embedding)
	c.l0Entries = append(c.l0Entries, cacheEntry{
		tfidfVector:    key,
		denseEmbedding: embeddingCopy,
	})

	shouldFlush := len(c.l0Entries) >= l0FlushThreshold
	c.mu.Unlock()

	if shouldFlush {
		select {
		case c.flushTrigger <- struct{}{}:
		default:
		}
	}
}

// AnalyzeSimilarity теперь просто возвращает значение счетчика
func (c *InMemoryCache) AnalyzeSimilarity(threshold float64) int {
	return int(c.itemsWithNeighbors.Load())
}

func (c *InMemoryCache) Find(key map[string]float64, threshold float64) ([]float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// 1. Поиск в L0 (линейный)
	for _, entry := range c.l0Entries {
		if tfidf.CosineSimilarity(key, entry.tfidfVector) >= threshold {
			return copyEmbedding(entry.denseEmbedding), true
		}
	}

	// 2. Поиск во всех L1 сегментах (по индексу), начиная с самых новых.
	topTerms := getTopK(key, c.topK)
	for i := len(c.l1Segments) - 1; i >= 0; i-- {
		segment := c.l1Segments[i]
		candidates := make(map[int]struct{})
		for _, term := range topTerms {
			if indices, ok := segment.index[term]; ok {
				for _, idx := range indices {
					candidates[idx] = struct{}{}
				}
			}
		}

		for idx := range candidates {
			entry := segment.entries[idx]
			if tfidf.CosineSimilarity(key, entry.tfidfVector) >= threshold {
				return copyEmbedding(entry.denseEmbedding), true
			}
		}
	}

	return nil, false
}

// --- Фоновые процессы (без изменений) ---

func (c *InMemoryCache) backgroundWorker() {
	for {
		select {
		case <-c.flushTrigger:
			c.flushL0()
		case <-c.compactionTrigger:
			c.compactL1()
		case <-c.closeWorker:
			return
		}
	}
}

func (c *InMemoryCache) flushL0() {
	c.mu.Lock()
	if len(c.l0Entries) == 0 {
		c.mu.Unlock()
		return
	}
	entriesToFlush := c.l0Entries
	c.l0Entries = make([]cacheEntry, 0, l0FlushThreshold)
	c.mu.Unlock()

	log.Printf("Flushing L0 with %d items to a new L1 segment...", len(entriesToFlush))
	newIndex := buildIndex(entriesToFlush, c.topK)
	newSegment := &l1Segment{
		entries: entriesToFlush,
		index:   newIndex,
	}

	c.mu.Lock()
	c.l1Segments = append(c.l1Segments, newSegment)
	shouldCompact := len(c.l1Segments) > l1CompactionTrigger
	c.mu.Unlock()

	if shouldCompact {
		select {
		case c.compactionTrigger <- struct{}{}:
		default:
		}
	}
}

func (c *InMemoryCache) compactL1() {
	c.mu.Lock()
	if len(c.l1Segments) < l1CompactionTrigger {
		c.mu.Unlock()
		return
	}
	segmentsToCompact := c.l1Segments[:l1CompactionTargetCount]
	remainingSegments := c.l1Segments[l1CompactionTargetCount:]
	c.mu.Unlock()

	log.Printf("Compacting %d L1 segments...", len(segmentsToCompact))
	var mergedEntries []cacheEntry
	for _, seg := range segmentsToCompact {
		mergedEntries = append(mergedEntries, seg.entries...)
	}

	newIndex := buildIndex(mergedEntries, c.topK)
	compactedSegment := &l1Segment{
		entries: mergedEntries,
		index:   newIndex,
	}

	c.mu.Lock()
	c.l1Segments = append([]*l1Segment{compactedSegment}, remainingSegments...)
	c.mu.Unlock()
	log.Printf("Compaction finished. New segment has %d items. Total L1 segments: %d", len(mergedEntries), len(c.l1Segments))
}

// --- Вспомогательные функции (без изменений) ---

func buildIndex(entries []cacheEntry, k int) map[string][]int {
	index := make(map[string][]int)
	for i, entry := range entries {
		topTerms := getTopK(entry.tfidfVector, k)
		for _, term := range topTerms {
			index[term] = append(index[term], i)
		}
	}
	return index
}

func getTopK(vector map[string]float64, k int) []string {
	if len(vector) <= k {
		terms := make([]string, 0, len(vector))
		for term := range vector {
			terms = append(terms, term)
		}
		return terms
	}

	scores := make([]termScore, 0, len(vector))
	for term, score := range vector {
		scores = append(scores, termScore{term, score})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	topTerms := make([]string, k)
	for i := 0; i < k; i++ {
		topTerms[i] = scores[i].term
	}
	return topTerms
}

func copyEmbedding(e []float64) []float64 {
	c := make([]float64, len(e))
	copy(c, e)
	return c
}
