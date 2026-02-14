//! Phrase extraction with canonical form selection
//!
//! Groups noun chunks by their lemmatized form, tracks variant frequencies,
//! and selects the most common surface form as canonical.

use super::chunker::{chunk_lemma, chunk_text, NounChunker};
use super::dedup::{resolve_overlaps_greedy, resolve_overlaps_greedy_with_diagnostics, ScoredChunk};
use crate::graph::csr::CsrGraph;
use crate::pagerank::PageRankResult;
use crate::pipeline::artifacts::{DroppedCandidate, ExtractionDiagnostics};
use crate::types::{Phrase, PhraseGrouping, ScoreAggregation, TextRankConfig, Token};
use rustc_hash::FxHashMap;

fn scrub_phrase_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_was_space = false;

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            last_was_space = false;
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

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

        // Group variants and create phrases with canonical forms
        let mut phrases = self.group_phrases(deduped);

        // Sort by score descending (with stable tie-breakers in deterministic mode).
        if self.config.determinism.is_deterministic() {
            phrases.sort_by(|a, b| a.stable_cmp(b));
        } else {
            phrases.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        }

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

    /// Extract phrases with full diagnostics.
    ///
    /// Only called when `debug_level >= Full`. Records:
    /// - Chunk formation events (stopword splits, POS rejections, etc.)
    /// - Overlap dedup drops
    /// - Zero-score drops
    /// - BelowTopN drops
    pub fn extract_with_diagnostics(
        &self,
        tokens: &[Token],
        graph: &CsrGraph,
        pagerank: &PageRankResult,
    ) -> (Vec<Phrase>, ExtractionDiagnostics) {
        let mut chunk_events = Vec::new();

        // Extract noun chunks with diagnostic recording
        let chunker = NounChunker::new()
            .with_min_length(self.config.min_phrase_length)
            .with_max_length(self.config.max_phrase_length);
        let chunks = chunker.extract_chunks_into(tokens, Some(&mut chunk_events));

        // Score each chunk (including zero-score for diagnostics)
        let all_scored = self.score_chunks_all(tokens, &chunks, graph, pagerank);

        // Record zero-score drops
        let mut dropped_candidates: Vec<DroppedCandidate> = Vec::new();
        let scored_chunks: Vec<ScoredChunk> = all_scored
            .into_iter()
            .filter(|sc| {
                if sc.score <= 0.0 {
                    dropped_candidates.push(DroppedCandidate {
                        text: sc.text.clone(),
                        lemma: sc.lemma.clone(),
                        score: sc.score,
                        token_range: (sc.chunk.start_token, sc.chunk.end_token),
                        reason: crate::pipeline::artifacts::DropReason::ZeroScore,
                    });
                    false
                } else {
                    true
                }
            })
            .collect();

        // Resolve overlaps with diagnostics
        let (deduped, overlap_drops) =
            resolve_overlaps_greedy_with_diagnostics(scored_chunks);
        dropped_candidates.extend(overlap_drops);

        // Group variants and create phrases with canonical forms
        let mut phrases = self.group_phrases(deduped);

        // Sort by score descending (with stable tie-breakers in deterministic mode).
        if self.config.determinism.is_deterministic() {
            phrases.sort_by(|a, b| a.stable_cmp(b));
        } else {
            phrases.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        }

        // Assign ranks
        for (i, phrase) in phrases.iter_mut().enumerate() {
            phrase.rank = i + 1;
        }

        // Limit to top_n if specified, recording BelowTopN drops
        if self.config.top_n > 0 && phrases.len() > self.config.top_n {
            for phrase in phrases.drain(self.config.top_n..) {
                dropped_candidates.push(DroppedCandidate {
                    text: phrase.text,
                    lemma: phrase.lemma,
                    score: phrase.score,
                    token_range: phrase
                        .offsets
                        .first()
                        .copied()
                        .unwrap_or((0, 0)),
                    reason: crate::pipeline::artifacts::DropReason::BelowTopN {
                        top_n: self.config.top_n,
                    },
                });
            }
        }

        let diags = ExtractionDiagnostics {
            chunk_events,
            dropped_candidates,
        };
        (phrases, diags)
    }

    /// Score all chunks (including zero-score ones).
    fn score_chunks_all(
        &self,
        tokens: &[Token],
        chunks: &[crate::types::ChunkSpan],
        graph: &CsrGraph,
        pagerank: &PageRankResult,
    ) -> Vec<ScoredChunk> {
        chunks
            .iter()
            .map(|chunk| {
                // Get tokens in this chunk via direct slice access (O(1) vs O(N) filter)
                let chunk_tokens = &tokens[chunk.start_token..chunk.end_token];

                // Collect PageRank scores for tokens in the chunk
                let scores: Vec<f64> = chunk_tokens
                    .iter()
                    .filter_map(|t| {
                        graph
                            .get_node_by_lemma(&t.graph_key(self.config.use_pos_in_nodes))
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
            .collect()
    }

    /// Score chunks and filter out zero-score entries.
    fn score_chunks(
        &self,
        tokens: &[Token],
        chunks: &[crate::types::ChunkSpan],
        graph: &CsrGraph,
        pagerank: &PageRankResult,
    ) -> Vec<ScoredChunk> {
        self.score_chunks_all(tokens, chunks, graph, pagerank)
            .into_iter()
            .filter(|sc| sc.score > 0.0)
            .collect()
    }

    /// Group scored chunks by the configured grouping strategy
    fn group_phrases(&self, chunks: Vec<ScoredChunk>) -> Vec<Phrase> {
        let mut groups: FxHashMap<String, Vec<ScoredChunk>> = FxHashMap::default();

        for chunk in chunks {
            let key = match self.config.phrase_grouping {
                PhraseGrouping::Lemma => chunk.lemma.clone(),
                PhraseGrouping::ScrubbedText => scrub_phrase_text(&chunk.text),
            };
            groups.entry(key).or_default().push(chunk);
        }

        let groups_iter: Vec<_> = if self.config.determinism.is_deterministic() {
            let mut sorted: Vec<_> = groups.into_iter().collect();
            sorted.sort_by(|(a, _), (b, _)| a.cmp(b));
            sorted
        } else {
            groups.into_iter().collect()
        };

        groups_iter
            .into_iter()
            .map(|(group_key, variants)| {
                let mut offsets = Vec::new();
                for variant in &variants {
                    offsets.push((variant.chunk.start_token, variant.chunk.end_token));
                }

                let (canonical_text, canonical_lemma) = match self.config.phrase_grouping {
                    PhraseGrouping::Lemma => {
                        let mut variant_counts: FxHashMap<String, usize> = FxHashMap::default();
                        for variant in &variants {
                            *variant_counts.entry(variant.text.clone()).or_insert(0) += 1;
                        }
                        let canonical = variant_counts
                            .into_iter()
                            .max_by(|(text_a, count_a), (text_b, count_b)| {
                                count_a.cmp(count_b).then_with(|| text_b.cmp(text_a))
                            })
                            .map(|(text, _)| text)
                            .unwrap_or_else(|| group_key.clone());
                        (canonical, group_key)
                    }
                    PhraseGrouping::ScrubbedText => {
                        let canonical_variant = variants
                            .iter()
                            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
                            .unwrap();
                        (
                            canonical_variant.text.clone(),
                            canonical_variant.lemma.clone(),
                        )
                    }
                };

                let score = match self.config.phrase_grouping {
                    PhraseGrouping::ScrubbedText => variants
                        .iter()
                        .map(|v| v.score)
                        .fold(f64::NEG_INFINITY, f64::max),
                    PhraseGrouping::Lemma => match self.config.score_aggregation {
                        ScoreAggregation::Sum => variants.iter().map(|v| v.score).sum(),
                        ScoreAggregation::Mean => {
                            variants.iter().map(|v| v.score).sum::<f64>() / variants.len() as f64
                        }
                        ScoreAggregation::Max => variants
                            .iter()
                            .map(|v| v.score)
                            .fold(f64::NEG_INFINITY, f64::max),
                        ScoreAggregation::RootMeanSquare => {
                            let sum_sq: f64 = variants.iter().map(|v| v.score * v.score).sum();
                            (sum_sq / variants.len() as f64).sqrt()
                        }
                    },
                };

                Phrase {
                    text: canonical_text,
                    lemma: canonical_lemma,
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
    /// Optional debug/inspect payload (populated when `debug_level > None`).
    pub debug: Option<crate::pipeline::artifacts::DebugPayload>,
}

/// Manual `PartialEq` that ignores `debug` — the payload contains `f64` fields
/// and is irrelevant for golden-test comparisons.
impl PartialEq for ExtractionResult {
    fn eq(&self, other: &Self) -> bool {
        self.phrases == other.phrases
            && self.converged == other.converged
            && self.iterations == other.iterations
    }
}

/// Extract phrases using the full TextRank pipeline
pub fn extract_keyphrases(tokens: &[Token], config: &TextRankConfig) -> Vec<Phrase> {
    extract_keyphrases_with_info(tokens, config).phrases
}

/// Extract phrases with PageRank convergence information.
///
/// Uses a hybrid approach for optimal performance:
///
/// 1. Fused candidate selection + graph building via legacy
///    [`GraphBuilder::from_tokens_with_pos`] (avoids `TokenStream` interning)
/// 2. [`PageRankRanker`] pipeline stage for ranking
/// 3. Direct [`PhraseExtractor`] invocation (avoids `to_legacy_tokens()` round-trip)
///
/// The `use_edge_weights` config field controls edge weight policy:
/// `true` → count-accumulating, `false` → binary.
///
/// [`GraphBuilder::from_tokens_with_pos`]: crate::graph::builder::GraphBuilder::from_tokens_with_pos
/// [`PageRankRanker`]: crate::pipeline::PageRankRanker
pub fn extract_keyphrases_with_info(tokens: &[Token], config: &TextRankConfig) -> ExtractionResult {
    use crate::graph::builder::GraphBuilder;
    use crate::pipeline::{Graph, Ranker};

    // Stage 1+2: fused candidate selection + graph building.
    // Bypasses TokenStream interning for zero-copy token access.
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
        config.use_pos_in_nodes,
    );

    if builder.is_empty() {
        return ExtractionResult {
            phrases: Vec::new(),
            converged: true,
            iterations: 0,
            debug: None,
        };
    }

    // Wrap in pipeline artifact for stage interop.
    let graph = Graph::from_builder(&builder);

    // Stage 3: ranking via pipeline stage.
    let rank_output = crate::pipeline::PageRankRanker.rank(&graph, None, config);
    let converged = rank_output.converged();
    let iterations = rank_output.iterations() as usize;

    // Build debug payload BEFORE the consuming move of rank_output.
    let mut debug = crate::pipeline::artifacts::DebugPayload::build(
        config.debug_level,
        &graph,
        &rank_output,
        config.debug_top_k,
    );

    // Stage 4: phrase extraction — use original &[Token] directly,
    // avoiding the to_legacy_tokens() round-trip.
    let pagerank_result = rank_output.into_pagerank_result();
    let extractor = PhraseExtractor::with_config(config.clone());

    let phrases = if config.debug_level.includes_full() {
        let (phrases, diags) =
            extractor.extract_with_diagnostics(tokens, graph.csr(), &pagerank_result);
        if let Some(ref mut dbg) = debug {
            dbg.phrase_diagnostics = Some(diags.chunk_events);
            dbg.dropped_candidates = Some(diags.dropped_candidates);
        }
        phrases
    } else {
        extractor.extract(tokens, graph.csr(), &pagerank_result)
    };

    ExtractionResult {
        phrases,
        converged,
        iterations,
        debug,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChunkSpan, DeterminismMode, PosTag};

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
        assert!(texts
            .iter()
            .any(|t| t.contains("machine") || t.contains("learning")));
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

    // ================================================================
    // Golden test helpers
    // ================================================================

    /// Multi-sentence document for golden tests.
    ///
    /// "Machine learning uses algorithms. Deep learning uses neural networks.
    ///  Machine learning models improve with data."
    ///
    /// Properties:
    /// - 3 sentences, repeated terms ("machine", "learning", "uses")
    /// - Mixed POS (nouns, verbs, prepositions)
    /// - Stopwords ("with")
    /// - Cross-sentence term overlap for richer graph connectivity
    fn golden_tokens() -> Vec<Token> {
        let mut tokens = vec![
            // Sentence 0: "Machine learning uses algorithms"
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("uses", "use", PosTag::Verb, 17, 21, 0, 2),
            Token::new("algorithms", "algorithm", PosTag::Noun, 22, 32, 0, 3),
            // Sentence 1: "Deep learning uses neural networks"
            Token::new("Deep", "deep", PosTag::Adjective, 34, 38, 1, 4),
            Token::new("learning", "learning", PosTag::Noun, 39, 47, 1, 5),
            Token::new("uses", "use", PosTag::Verb, 48, 52, 1, 6),
            Token::new("neural", "neural", PosTag::Adjective, 53, 59, 1, 7),
            Token::new("networks", "network", PosTag::Noun, 60, 68, 1, 8),
            // Sentence 2: "Machine learning models improve with data"
            Token::new("Machine", "machine", PosTag::Noun, 70, 77, 2, 9),
            Token::new("learning", "learning", PosTag::Noun, 78, 86, 2, 10),
            Token::new("models", "model", PosTag::Noun, 87, 93, 2, 11),
            Token::new("improve", "improve", PosTag::Verb, 94, 101, 2, 12),
            Token::new("with", "with", PosTag::Preposition, 102, 106, 2, 13),
            Token::new("data", "data", PosTag::Noun, 107, 111, 2, 14),
        ];
        // Mark stopwords
        tokens[13].is_stopword = true; // "with"
        tokens
    }

    // ================================================================
    // Golden tests — BaseTextRank pipeline regression suite
    // ================================================================

    /// Verify score within epsilon tolerance.
    fn assert_score(actual: f64, expected: f64, phrase: &str) {
        let eps = 1e-8;
        assert!(
            (actual - expected).abs() < eps,
            "Score mismatch for {:?}: got {:.10}, expected {:.10} (delta {:.2e})",
            phrase,
            actual,
            expected,
            (actual - expected).abs()
        );
    }

    /// Golden test: single-sentence document.
    ///
    /// Input: "Machine learning is a subset of artificial intelligence"
    /// Expected: 3 phrases, converged in 9 iterations.
    #[test]
    fn golden_base_textrank_single_sentence() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();
        let result = extract_keyphrases_with_info(&tokens, &config);

        // Convergence
        assert!(result.converged);
        assert_eq!(result.iterations, 9);

        // Phrase count
        assert_eq!(result.phrases.len(), 3);

        // Phrase ordering (score descending)
        assert_eq!(result.phrases[0].text, "Machine learning");
        assert_eq!(result.phrases[1].text, "artificial intelligence");
        assert_eq!(result.phrases[2].text, "subset");

        // Lemmas
        assert_eq!(result.phrases[0].lemma, "machine learning");
        assert_eq!(result.phrases[1].lemma, "artificial intelligence");
        assert_eq!(result.phrases[2].lemma, "subset");

        // Scores (within epsilon)
        assert_score(result.phrases[0].score, 0.2846505870, "Machine learning");
        assert_score(
            result.phrases[1].score,
            0.2846505870,
            "artificial intelligence",
        );
        assert_score(result.phrases[2].score, 0.2153494130, "subset");

        // Ranks (1-indexed)
        assert_eq!(result.phrases[0].rank, 1);
        assert_eq!(result.phrases[1].rank, 2);
        assert_eq!(result.phrases[2].rank, 3);
    }

    /// Golden test: multi-sentence document with repeated terms.
    ///
    /// Input: "Machine learning uses algorithms. Deep learning uses neural
    ///         networks. Machine learning models improve with data."
    /// Expected: 6 phrases, converged in 21 iterations.
    #[test]
    fn golden_base_textrank_multi_sentence() {
        let tokens = golden_tokens();
        let config = TextRankConfig::default();
        let result = extract_keyphrases_with_info(&tokens, &config);

        // Convergence
        assert!(result.converged);
        assert_eq!(result.iterations, 21);

        // Phrase count
        assert_eq!(result.phrases.len(), 6);

        // Top-5 phrase ordering (score descending)
        let texts: Vec<&str> = result.phrases.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "Machine learning models",
                "Machine learning",
                "Deep learning",
                "neural networks",
                "data",
                "algorithms",
            ]
        );

        // Scores (within epsilon)
        assert_score(
            result.phrases[0].score,
            0.4249608102,
            "Machine learning models",
        );
        assert_score(result.phrases[1].score, 0.3179496860, "Machine learning");
        assert_score(result.phrases[2].score, 0.2746293119, "Deep learning");
        assert_score(result.phrases[3].score, 0.1412982971, "neural networks");
        assert_score(result.phrases[4].score, 0.0616928699, "data");
        assert_score(result.phrases[5].score, 0.0567178525, "algorithms");

        // Ranks are 1-indexed and sequential
        for (i, p) in result.phrases.iter().enumerate() {
            assert_eq!(p.rank, i + 1, "rank mismatch at position {}", i);
        }
    }

    /// Golden test: top_n limiting on multi-sentence document.
    #[test]
    fn golden_base_textrank_top_n_3() {
        let tokens = golden_tokens();
        let config = TextRankConfig::default().with_top_n(3);
        let result = extract_keyphrases_with_info(&tokens, &config);

        assert!(result.converged);
        assert_eq!(result.phrases.len(), 3);

        // Top-3 ordering is stable
        assert_eq!(result.phrases[0].text, "Machine learning models");
        assert_eq!(result.phrases[1].text, "Machine learning");
        assert_eq!(result.phrases[2].text, "Deep learning");
    }

    /// Golden test: empty input produces empty output.
    #[test]
    fn golden_base_textrank_empty() {
        let tokens: Vec<Token> = Vec::new();
        let config = TextRankConfig::default();
        let result = extract_keyphrases_with_info(&tokens, &config);

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert!(result.phrases.is_empty());
    }

    /// Golden test: convergence metadata is propagated.
    #[test]
    fn golden_base_textrank_convergence_metadata() {
        let tokens = golden_tokens();
        let config = TextRankConfig {
            max_iterations: 1, // Force early termination
            ..TextRankConfig::default()
        };
        let result = extract_keyphrases_with_info(&tokens, &config);

        // With only 1 iteration, PageRank should NOT converge on a non-trivial graph.
        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
        // Should still produce phrases (just with less-converged scores).
        assert!(!result.phrases.is_empty());
    }

    // ================================================================
    // Determinism tests
    // ================================================================

    /// When two surface-form variants of the same lemma appear with equal
    /// frequency, the canonical text must be chosen deterministically
    /// (lexicographically smallest wins).
    #[test]
    fn test_canonical_form_tie_breaks_lexicographically() {
        fn chunk(start: usize) -> ChunkSpan {
            ChunkSpan {
                start_token: start,
                end_token: start + 1,
                start_char: start * 10,
                end_char: start * 10 + 5,
                sentence_idx: 0,
            }
        }

        // Two variants of lemma "network": "Networks" and "networks", each appearing once.
        let scored = vec![
            ScoredChunk {
                chunk: chunk(0),
                score: 1.0,
                text: "Networks".to_string(),
                lemma: "network".to_string(),
            },
            ScoredChunk {
                chunk: chunk(1),
                score: 1.0,
                text: "networks".to_string(),
                lemma: "network".to_string(),
            },
        ];

        let config = TextRankConfig {
            phrase_grouping: PhraseGrouping::Lemma,
            ..TextRankConfig::default()
        };
        let extractor = PhraseExtractor::with_config(config);
        let phrases = extractor.group_phrases(scored);

        assert_eq!(phrases.len(), 1);
        // Lexicographically smaller "Networks" < "networks" (uppercase < lowercase)
        assert_eq!(phrases[0].text, "Networks");
    }

    // ================================================================
    // Debug wiring tests
    // ================================================================

    #[test]
    fn test_debug_none_produces_no_payload() {
        let tokens = make_tokens();
        let config = TextRankConfig {
            debug_level: crate::pipeline::artifacts::DebugLevel::None,
            ..TextRankConfig::default()
        };
        let result = extract_keyphrases_with_info(&tokens, &config);
        assert!(result.debug.is_none());
    }

    #[test]
    fn test_debug_stats_produces_graph_stats_and_convergence() {
        let tokens = make_tokens();
        let config = TextRankConfig {
            debug_level: crate::pipeline::artifacts::DebugLevel::Stats,
            ..TextRankConfig::default()
        };
        let result = extract_keyphrases_with_info(&tokens, &config);
        let dbg = result.debug.as_ref().expect("debug payload should be Some at Stats level");

        // Graph stats present
        let gs = dbg.graph_stats.as_ref().expect("graph_stats should be present");
        assert!(gs.num_nodes > 0);
        assert!(gs.num_edges > 0);
        assert!(gs.avg_degree > 0.0);

        // Convergence summary present
        let cs = dbg.convergence_summary.as_ref().expect("convergence_summary should be present");
        assert!(cs.converged);
        assert!(cs.iterations > 0);

        // Full-level fields should NOT be populated at Stats level
        assert!(dbg.phrase_diagnostics.is_none());
        assert!(dbg.dropped_candidates.is_none());
        assert!(dbg.node_scores.is_none()); // TopNodes not reached
    }

    #[test]
    fn test_debug_full_produces_phrase_diagnostics() {
        let tokens = make_tokens();
        let config = TextRankConfig {
            debug_level: crate::pipeline::artifacts::DebugLevel::Full,
            ..TextRankConfig::default()
        };
        let result = extract_keyphrases_with_info(&tokens, &config);
        let dbg = result.debug.as_ref().expect("debug payload should be Some at Full level");

        // Graph stats and convergence are present (inherited from lower levels)
        assert!(dbg.graph_stats.is_some());
        assert!(dbg.convergence_summary.is_some());

        // Full-level specific: phrase diagnostics
        let pd = dbg.phrase_diagnostics.as_ref().expect("phrase_diagnostics should be present at Full level");
        // The sentence has non-noun tokens (verb, determiner, preposition) that should generate events
        assert!(!pd.is_empty(), "Should have at least one phrase split event");

        // Full-level specific: dropped candidates (may or may not have entries depending on extraction)
        assert!(dbg.dropped_candidates.is_some());

        // Node scores should also be present at Full level (includes TopNodes)
        assert!(dbg.node_scores.is_some());
    }

    #[test]
    fn test_debug_empty_input_produces_no_payload() {
        let tokens: Vec<Token> = Vec::new();
        let config = TextRankConfig {
            debug_level: crate::pipeline::artifacts::DebugLevel::Full,
            ..TextRankConfig::default()
        };
        let result = extract_keyphrases_with_info(&tokens, &config);
        // Empty input → early return with debug: None
        assert!(result.debug.is_none());
    }

    /// In deterministic mode, group_phrases produces groups in sorted key order.
    #[test]
    fn test_group_phrases_deterministic_key_order() {
        fn chunk(start: usize, sentence: usize) -> ChunkSpan {
            ChunkSpan {
                start_token: start,
                end_token: start + 1,
                start_char: start * 10,
                end_char: start * 10 + 5,
                sentence_idx: sentence,
            }
        }

        // Three different lemmas: "zebra", "alpha", "middle"
        let scored = vec![
            ScoredChunk {
                chunk: chunk(0, 0),
                score: 1.0,
                text: "zebra".to_string(),
                lemma: "zebra".to_string(),
            },
            ScoredChunk {
                chunk: chunk(1, 0),
                score: 1.0,
                text: "alpha".to_string(),
                lemma: "alpha".to_string(),
            },
            ScoredChunk {
                chunk: chunk(2, 0),
                score: 1.0,
                text: "middle".to_string(),
                lemma: "middle".to_string(),
            },
        ];

        let config = TextRankConfig {
            phrase_grouping: PhraseGrouping::Lemma,
            determinism: DeterminismMode::Deterministic,
            ..TextRankConfig::default()
        };
        let extractor = PhraseExtractor::with_config(config);
        let phrases = extractor.group_phrases(scored);

        // In deterministic mode, groups are sorted by lemma key
        let lemmas: Vec<&str> = phrases.iter().map(|p| p.lemma.as_str()).collect();
        assert_eq!(lemmas, vec!["alpha", "middle", "zebra"]);
    }
}
