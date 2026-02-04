//! Phrase extraction with canonical form selection
//!
//! Groups noun chunks by their lemmatized form, tracks variant frequencies,
//! and selects the most common surface form as canonical.

use super::chunker::{chunk_lemma, chunk_text, NounChunker};
use super::dedup::{resolve_overlaps_greedy, ScoredChunk};
use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::PageRankResult;
use crate::types::{Phrase, ScoreAggregation, TextRankConfig, Token};
use rustc_hash::FxHashMap;

/// Phrase extractor that combines chunking, scoring, and deduplication
#[derive(Debug)]
pub struct PhraseExtractor {
    config: TextRankConfig,
}

impl Default for PhraseExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PhraseExtractor {
    /// Create a new phrase extractor with default config
    pub fn new() -> Self {
        Self {
            config: TextRankConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: TextRankConfig) -> Self {
        Self { config }
    }

    /// Extract phrases from tokens using PageRank scores
    pub fn extract(
        &self,
        tokens: &[Token],
        graph: &CsrGraph,
        pagerank: &PageRankResult,
    ) -> Vec<Phrase> {
        // Extract noun chunks
        let chunker = NounChunker::new()
            .with_min_length(self.config.min_phrase_length)
            .with_max_length(self.config.max_phrase_length);
        let chunks = chunker.extract_chunks(tokens);

        // Score each chunk
        let scored_chunks = self.score_chunks(tokens, &chunks, graph, pagerank);

        // Resolve overlaps
        let deduped = resolve_overlaps_greedy(scored_chunks);

        // Group by lemma and create phrases with canonical forms
        let mut phrases = self.group_by_lemma(deduped);

        // Sort by score descending
        phrases.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Assign ranks
        for (i, phrase) in phrases.iter_mut().enumerate() {
            phrase.rank = i + 1;
        }

        // Limit to top_n if specified
        if self.config.top_n > 0 && phrases.len() > self.config.top_n {
            phrases.truncate(self.config.top_n);
        }

        phrases
    }

    /// Score a chunk based on its constituent tokens' PageRank scores
    fn score_chunks(
        &self,
        tokens: &[Token],
        chunks: &[crate::types::ChunkSpan],
        graph: &CsrGraph,
        pagerank: &PageRankResult,
    ) -> Vec<ScoredChunk> {
        chunks
            .iter()
            .map(|chunk| {
                // Get tokens in this chunk
                let chunk_tokens: Vec<_> = tokens
                    .iter()
                    .filter(|t| t.token_idx >= chunk.start_token && t.token_idx < chunk.end_token)
                    .collect();

                // Collect PageRank scores for tokens in the chunk
                let scores: Vec<f64> = chunk_tokens
                    .iter()
                    .filter_map(|t| {
                        graph
                            .get_node_by_lemma(&t.lemma)
                            .map(|node_id| pagerank.score(node_id))
                    })
                    .collect();

                // Aggregate scores
                let score = self.config.score_aggregation.aggregate(&scores);

                ScoredChunk {
                    chunk: chunk.clone(),
                    score,
                    text: chunk_text(tokens, chunk),
                    lemma: chunk_lemma(tokens, chunk),
                }
            })
            .filter(|sc| sc.score > 0.0)
            .collect()
    }

    /// Group scored chunks by their lemmatized form
    fn group_by_lemma(&self, chunks: Vec<ScoredChunk>) -> Vec<Phrase> {
        // Group by lemma
        let mut groups: FxHashMap<String, Vec<ScoredChunk>> = FxHashMap::default();

        for chunk in chunks {
            groups
                .entry(chunk.lemma.clone())
                .or_default()
                .push(chunk);
        }

        // Create phrases with canonical forms
        groups
            .into_iter()
            .map(|(lemma, variants)| {
                // Count variant frequencies
                let mut variant_counts: FxHashMap<String, usize> = FxHashMap::default();
                let mut total_score = 0.0;
                let mut offsets = Vec::new();

                for variant in &variants {
                    *variant_counts.entry(variant.text.clone()).or_insert(0) += 1;
                    total_score += variant.score;
                    offsets.push((variant.chunk.start_token, variant.chunk.end_token));
                }

                // Select canonical form (most frequent variant)
                let canonical = variant_counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(text, _)| text)
                    .unwrap_or_else(|| lemma.clone());

                // Aggregate score based on config
                let score = match self.config.score_aggregation {
                    ScoreAggregation::Sum => total_score,
                    ScoreAggregation::Mean => total_score / variants.len() as f64,
                    ScoreAggregation::Max => variants
                        .iter()
                        .map(|v| v.score)
                        .fold(f64::NEG_INFINITY, f64::max),
                    ScoreAggregation::RootMeanSquare => {
                        let sum_sq: f64 = variants.iter().map(|v| v.score * v.score).sum();
                        (sum_sq / variants.len() as f64).sqrt()
                    }
                };

                Phrase {
                    text: canonical,
                    lemma,
                    score,
                    count: variants.len(),
                    offsets,
                    rank: 0, // Will be assigned after sorting
                }
            })
            .collect()
    }
}

/// Result of keyphrase extraction including convergence info
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Extracted phrases
    pub phrases: Vec<Phrase>,
    /// Whether PageRank converged
    pub converged: bool,
    /// Number of PageRank iterations
    pub iterations: usize,
}

/// Extract phrases using the full TextRank pipeline
pub fn extract_keyphrases(tokens: &[Token], config: &TextRankConfig) -> Vec<Phrase> {
    extract_keyphrases_with_info(tokens, config).phrases
}

/// Extract phrases with PageRank convergence information
pub fn extract_keyphrases_with_info(tokens: &[Token], config: &TextRankConfig) -> ExtractionResult {
    // Build graph with POS filtering
    let include_pos = if config.include_pos.is_empty() {
        None
    } else {
        Some(config.include_pos.as_slice())
    };
    let builder = GraphBuilder::from_tokens_with_pos(
        tokens,
        config.window_size,
        config.use_edge_weights,
        include_pos,
    );

    if builder.is_empty() {
        return ExtractionResult {
            phrases: Vec::new(),
            converged: true,
            iterations: 0,
        };
    }

    // Convert to CSR
    let graph = CsrGraph::from_builder(&builder);

    // Run PageRank
    let pagerank = crate::pagerank::standard::StandardPageRank::new()
        .with_damping(config.damping)
        .with_max_iterations(config.max_iterations)
        .with_threshold(config.convergence_threshold)
        .run(&graph);

    // Extract phrases
    let extractor = PhraseExtractor::with_config(config.clone());
    let phrases = extractor.extract(tokens, &graph, &pagerank);

    ExtractionResult {
        phrases,
        converged: pagerank.converged,
        iterations: pagerank.iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    fn make_tokens() -> Vec<Token> {
        vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
            Token::new("a", "a", PosTag::Determiner, 20, 21, 0, 3),
            Token::new("subset", "subset", PosTag::Noun, 22, 28, 0, 4),
            Token::new("of", "of", PosTag::Preposition, 29, 31, 0, 5),
            Token::new("artificial", "artificial", PosTag::Adjective, 32, 42, 0, 6),
            Token::new("intelligence", "intelligence", PosTag::Noun, 43, 55, 0, 7),
        ]
    }

    #[test]
    fn test_extract_keyphrases() {
        let tokens = make_tokens();
        let config = TextRankConfig::default().with_top_n(5);
        let phrases = extract_keyphrases(&tokens, &config);

        assert!(!phrases.is_empty());
        // Should find "machine learning", "artificial intelligence", etc.
        let texts: Vec<_> = phrases.iter().map(|p| p.text.to_lowercase()).collect();
        assert!(texts.iter().any(|t| t.contains("machine") || t.contains("learning")));
    }

    #[test]
    fn test_phrase_ranking() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases(&tokens, &config);

        // Ranks should be 1-indexed and sequential
        for (i, phrase) in phrases.iter().enumerate() {
            assert_eq!(phrase.rank, i + 1);
        }
    }

    #[test]
    fn test_empty_input() {
        let tokens: Vec<Token> = Vec::new();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases(&tokens, &config);

        assert!(phrases.is_empty());
    }

    #[test]
    fn test_score_aggregation() {
        let tokens = make_tokens();

        let config_sum = TextRankConfig::default().with_score_aggregation(ScoreAggregation::Sum);
        let phrases_sum = extract_keyphrases(&tokens, &config_sum);

        let config_mean = TextRankConfig::default().with_score_aggregation(ScoreAggregation::Mean);
        let phrases_mean = extract_keyphrases(&tokens, &config_mean);

        // With Sum aggregation, multi-word phrases might score higher
        // With Mean, single-word high-scoring phrases might win
        // Just verify both produce results
        assert!(!phrases_sum.is_empty());
        assert!(!phrases_mean.is_empty());
    }

    #[test]
    fn test_top_n_limit() {
        let tokens = make_tokens();
        let config = TextRankConfig::default().with_top_n(2);
        let phrases = extract_keyphrases(&tokens, &config);

        assert!(phrases.len() <= 2);
    }
}
