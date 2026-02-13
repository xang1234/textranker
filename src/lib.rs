//! # rapid_textrank
//!
//! A high-performance TextRank implementation with Python bindings.
//!
//! This library provides keyword extraction and text summarization using
//! the TextRank algorithm and its variants (PositionRank, BiasedTextRank, TopicRank).
//!
//! ## Features
//!
//! - **Fast**: 10-100x faster than pure Python implementations
//! - **Unicode-aware**: Proper handling of CJK, emoji, and other scripts
//! - **Flexible**: Multiple algorithm variants and configuration options
//! - **Python bindings**: Seamless integration with Python via PyO3

pub mod clustering;
pub mod errors;
pub mod graph;
pub mod nlp;
pub mod pagerank;
pub mod phrase;
pub mod pipeline;
pub mod summarizer;
pub mod types;
pub mod variants;

#[cfg(feature = "python")]
pub mod python;

// Re-export commonly used types
pub use errors::{Result, TextRankError};
pub use types::{
    ChunkSpan, DeterminismMode, LemmaId, Phrase, ScoreAggregation, Sentence, StringPool,
    TextRankConfig, Token,
};

// Re-export main functionality
pub use graph::{builder::GraphBuilder, csr::CsrGraph};
pub use nlp::{stopwords::StopwordFilter, tokenizer::Tokenizer};
pub use pagerank::{
    personalized::PersonalizedPageRank, standard::StandardPageRank, PageRankResult,
};
pub use phrase::extraction::PhraseExtractor;
pub use pipeline::error_code::ErrorCode;
pub use pipeline::errors::{PipelineRuntimeError, PipelineSpecError};
pub use pipeline::spec::{
    CandidatesSpec, ClusteringSpec, EdgeWeightingSpec, ExposeSpec, FormatSpec,
    GraphSpec, GraphTransformSpec, ModuleSet, NodeScoresSpec, PageRankExposeSpec,
    PhraseGroupingSpec, PhraseSpec, PipelineSpec, PipelineSpecV1, PreprocessSpec,
    RankSpec, RuntimeSpec, ScoreAggregationSpec, TeleportSpec,
    merge_modules, resolve_preset, resolve_spec,
};
pub use pipeline::validation::{ValidationEngine, ValidationReport};
pub use pipeline::{
    AlphaBoostWeighter, CandidateGraphBuilder, CandidateSelector, ChunkPhraseBuilder,
    ClusterAssignments, Clusterer, CooccurrenceGraphBuilder, DebugLevel, DynPipeline,
    EdgeWeightPolicy, FocusTermsTeleportBuilder, IntraTopicEdgeRemover, JaccardHacClusterer,
    Linkage, MultipartitePhraseBuilder, MultipartiteRankPipeline, MultipartiteTransform,
    NoopClusterer, NoopGraphTransform, NoopPreprocessor, PhraseBuilder, SentenceCandidateSelector, SentenceRankPipeline,
    SentenceFormatter, SentenceGraphBuilder, SentencePhraseBuilder, SpecPipelineBuilder,
    TopicGraphBuilder, TopicRankPipeline, TopicRepresentativeBuilder,
    TopicWeightsTeleportBuilder, TopicalPageRankPipeline, PhraseCandidateSelector,
    PositionTeleportBuilder, Preprocessor, ResultFormatter, StandardResultFormatter,
    TeleportBuilder, TeleportType, TeleportVector, TokenEntry, TokenStream, TokenStreamRef,
    UniformTeleportBuilder, WindowGraphBuilder, WindowStrategy, WordNodeSelector,
    DEFAULT_WINDOW_SIZE,
};
// Note: pipeline::GraphBuilder trait is NOT re-exported here to avoid
// collision with graph::builder::GraphBuilder (the mutable builder struct).
// Access the trait via `pipeline::GraphBuilder` or
// `pipeline::traits::GraphBuilder`.
pub use summarizer::selector::SentenceSelector;
pub use variants::{
    biased_textrank::BiasedTextRank, multipartite_rank::MultipartiteRank,
    position_rank::PositionRank, single_rank::SingleRank, topical_pagerank::TopicalPageRank,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ================================================================
// Determinism golden tests
// ================================================================
#[cfg(test)]
mod determinism_golden_tests {
    //! Run each variant twice in deterministic mode and assert bit-exact output.
    //! These tests catch non-determinism regressions in CI.

    use crate::phrase::extraction::{
        extract_keyphrases_with_info, ExtractionResult,
    };
    use crate::types::{DeterminismMode, PosTag, TextRankConfig, Token};
    use crate::variants::biased_textrank::BiasedTextRank;
    use crate::variants::multipartite_rank::MultipartiteRank;
    use crate::variants::position_rank::PositionRank;
    use crate::variants::single_rank::SingleRank;
    use crate::variants::topic_rank::TopicRank;
    use crate::variants::topical_pagerank::TopicalPageRank;
    use std::collections::HashMap;

    /// Multi-sentence document with repeated terms, mixed POS, and stopwords.
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
        tokens[13].is_stopword = true; // "with"
        tokens
    }

    fn deterministic_config() -> TextRankConfig {
        TextRankConfig {
            determinism: DeterminismMode::Deterministic,
            ..TextRankConfig::default()
        }
    }

    /// Run extraction N times and assert all results are identical.
    fn assert_deterministic(results: &[ExtractionResult]) {
        assert!(results.len() >= 2);
        let first = &results[0];
        assert!(!first.phrases.is_empty(), "extraction produced no phrases");
        for (i, result) in results.iter().enumerate().skip(1) {
            assert_eq!(
                first.converged, result.converged,
                "run 0 vs run {}: converged mismatch",
                i
            );
            assert_eq!(
                first.iterations, result.iterations,
                "run 0 vs run {}: iterations mismatch",
                i
            );
            assert_eq!(
                first.phrases.len(),
                result.phrases.len(),
                "run 0 vs run {}: phrase count mismatch",
                i
            );
            for (j, (a, b)) in first.phrases.iter().zip(result.phrases.iter()).enumerate() {
                assert_eq!(
                    a, b,
                    "run 0 vs run {}: phrase {} differs\n  left:  {:?}\n  right: {:?}",
                    i, j, a, b
                );
            }
        }
    }

    // ── Base TextRank ───────────────────────────────────────────────

    #[test]
    fn golden_determinism_base_textrank() {
        let tokens = golden_tokens();
        let config = deterministic_config();
        let results: Vec<_> = (0..3)
            .map(|_| extract_keyphrases_with_info(&tokens, &config))
            .collect();
        assert_deterministic(&results);
    }

    // ── SingleRank ──────────────────────────────────────────────────

    #[test]
    fn golden_determinism_single_rank() {
        let tokens = golden_tokens();
        let config = deterministic_config();
        let extractor = SingleRank::with_config(config);
        let results: Vec<_> = (0..3)
            .map(|_| extractor.extract_with_info(&tokens))
            .collect();
        assert_deterministic(&results);
    }

    // ── PositionRank ────────────────────────────────────────────────

    #[test]
    fn golden_determinism_position_rank() {
        let tokens = golden_tokens();
        let config = deterministic_config();
        let extractor = PositionRank::with_config(config);
        let results: Vec<_> = (0..3)
            .map(|_| extractor.extract_with_info(&tokens))
            .collect();
        assert_deterministic(&results);
    }

    // ── BiasedTextRank ──────────────────────────────────────────────

    #[test]
    fn golden_determinism_biased_textrank() {
        let tokens = golden_tokens();
        let config = deterministic_config();
        let extractor = BiasedTextRank::with_config(config)
            .with_focus(&["machine", "learning"])
            .with_bias_weight(5.0);
        let results: Vec<_> = (0..3)
            .map(|_| extractor.extract_with_info(&tokens))
            .collect();
        assert_deterministic(&results);
    }

    // ── TopicRank ───────────────────────────────────────────────────

    #[test]
    fn golden_determinism_topic_rank() {
        let tokens = golden_tokens();
        let config = deterministic_config();
        let extractor = TopicRank::with_config(config);
        let results: Vec<_> = (0..3)
            .map(|_| extractor.extract_with_info(&tokens))
            .collect();
        assert_deterministic(&results);
    }

    // ── TopicalPageRank ─────────────────────────────────────────────

    #[test]
    fn golden_determinism_topical_pagerank() {
        let tokens = golden_tokens();
        let config = deterministic_config();
        let mut topic_weights = HashMap::new();
        topic_weights.insert("machine".to_string(), 2.0);
        topic_weights.insert("network".to_string(), 1.5);
        let extractor = TopicalPageRank::with_config(config)
            .with_topic_weights(topic_weights)
            .with_min_weight(0.1);
        let results: Vec<_> = (0..3)
            .map(|_| extractor.extract_with_info(&tokens))
            .collect();
        assert_deterministic(&results);
    }

    // ── MultipartiteRank ────────────────────────────────────────────

    #[test]
    fn golden_determinism_multipartite_rank() {
        let tokens = golden_tokens();
        let config = deterministic_config();
        let extractor = MultipartiteRank::with_config(config);
        let results: Vec<_> = (0..3)
            .map(|_| extractor.extract_with_info(&tokens))
            .collect();
        assert_deterministic(&results);
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Initialize the Python module
#[cfg(feature = "python")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
