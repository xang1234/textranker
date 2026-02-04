//! Core types for rapid_textrank
//!
//! This module defines the fundamental data structures used throughout the library,
//! including string interning, tokens, phrases, and configuration.

use crate::errors::{Result, TextRankError};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ============================================================================
// String Interning
// ============================================================================

/// A pool for string interning to reduce memory usage and enable fast comparisons.
///
/// String interning stores each unique string once and returns lightweight references.
/// This is particularly useful for lemmas which may repeat many times across a document.
#[derive(Debug, Default)]
pub struct StringPool {
    /// Maps strings to their interned IDs
    string_to_id: FxHashMap<Arc<str>, u32>,
    /// Maps IDs back to strings
    id_to_string: Vec<Arc<str>>,
}

impl StringPool {
    /// Create a new empty string pool
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a string pool with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            string_to_id: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            id_to_string: Vec::with_capacity(capacity),
        }
    }

    /// Intern a string, returning its ID
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.string_to_id.get(s) {
            return id;
        }

        let id = self.id_to_string.len() as u32;
        let arc: Arc<str> = s.into();
        self.string_to_id.insert(arc.clone(), id);
        self.id_to_string.push(arc);
        id
    }

    /// Get a string by its ID
    pub fn get(&self, id: u32) -> Option<&str> {
        self.id_to_string.get(id as usize).map(|s| s.as_ref())
    }

    /// Get the number of unique strings in the pool
    pub fn len(&self) -> usize {
        self.id_to_string.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.id_to_string.is_empty()
    }
}

// ============================================================================
// Lemma ID
// ============================================================================

/// A reference to an interned lemma string.
///
/// This is a lightweight handle (two u32s) that can be used for fast equality
/// comparisons without string allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LemmaId {
    /// The pool ID (for supporting multiple pools if needed)
    pub pool_id: u32,
    /// The string ID within the pool
    pub string_id: u32,
}

impl LemmaId {
    /// Create a new LemmaId
    pub fn new(pool_id: u32, string_id: u32) -> Self {
        Self { pool_id, string_id }
    }

    /// Create a LemmaId for the default pool (pool_id = 0)
    pub fn from_string_id(string_id: u32) -> Self {
        Self {
            pool_id: 0,
            string_id,
        }
    }
}

// ============================================================================
// Token
// ============================================================================

/// Part-of-speech tags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PosTag {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Pronoun,
    Determiner,
    Preposition,
    Conjunction,
    Interjection,
    Numeral,
    Particle,
    Punctuation,
    Symbol,
    ProperNoun,
    Other,
}

impl PosTag {
    /// Check if this POS tag is typically used for TextRank keyword extraction
    pub fn is_content_word(&self) -> bool {
        matches!(
            self,
            PosTag::Noun | PosTag::Verb | PosTag::Adjective | PosTag::ProperNoun
        )
    }

    /// Check if this tag represents a noun (common or proper)
    pub fn is_noun(&self) -> bool {
        matches!(self, PosTag::Noun | PosTag::ProperNoun)
    }

    /// Check if this tag can start a noun phrase
    pub fn can_start_noun_phrase(&self) -> bool {
        matches!(
            self,
            PosTag::Determiner | PosTag::Adjective | PosTag::Noun | PosTag::ProperNoun
        )
    }

    /// Parse from spaCy-style POS tag
    pub fn from_spacy(tag: &str) -> Self {
        match tag.to_uppercase().as_str() {
            "NOUN" => PosTag::Noun,
            "VERB" => PosTag::Verb,
            "ADJ" => PosTag::Adjective,
            "ADV" => PosTag::Adverb,
            "PRON" => PosTag::Pronoun,
            "DET" => PosTag::Determiner,
            "ADP" => PosTag::Preposition,
            "CCONJ" | "SCONJ" => PosTag::Conjunction,
            "INTJ" => PosTag::Interjection,
            "NUM" => PosTag::Numeral,
            "PART" => PosTag::Particle,
            "PUNCT" => PosTag::Punctuation,
            "SYM" => PosTag::Symbol,
            "PROPN" => PosTag::ProperNoun,
            _ => PosTag::Other,
        }
    }
}

/// A token from the input text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// The surface form (original text)
    pub text: String,
    /// The lemmatized form (normalized)
    pub lemma: String,
    /// Part-of-speech tag
    pub pos: PosTag,
    /// Character offset (start) in original text
    pub start: usize,
    /// Character offset (end) in original text
    pub end: usize,
    /// Sentence index this token belongs to
    pub sentence_idx: usize,
    /// Token index within the document
    pub token_idx: usize,
    /// Whether this token is a stopword
    pub is_stopword: bool,
}

impl Token {
    /// Create a new token
    pub fn new(
        text: impl Into<String>,
        lemma: impl Into<String>,
        pos: PosTag,
        start: usize,
        end: usize,
        sentence_idx: usize,
        token_idx: usize,
    ) -> Self {
        Self {
            text: text.into(),
            lemma: lemma.into(),
            pos,
            start,
            end,
            sentence_idx,
            token_idx,
            is_stopword: false,
        }
    }

    /// Check if this token should be included in the TextRank graph
    pub fn is_graph_candidate(&self) -> bool {
        self.pos.is_content_word() && !self.is_stopword
    }
}

// ============================================================================
// Phrase & Chunk
// ============================================================================

/// A span of text representing a noun chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSpan {
    /// Start token index (inclusive)
    pub start_token: usize,
    /// End token index (exclusive)
    pub end_token: usize,
    /// Start character offset
    pub start_char: usize,
    /// End character offset
    pub end_char: usize,
    /// The sentence this chunk belongs to
    pub sentence_idx: usize,
}

impl ChunkSpan {
    /// Check if this chunk overlaps with another
    pub fn overlaps(&self, other: &ChunkSpan) -> bool {
        self.start_char < other.end_char && other.start_char < self.end_char
    }

    /// Get the token length of this chunk
    pub fn token_len(&self) -> usize {
        self.end_token - self.start_token
    }
}

/// An extracted phrase with its score and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phrase {
    /// The canonical surface form
    pub text: String,
    /// The lemmatized form (used as key for grouping)
    pub lemma: String,
    /// The TextRank score
    pub score: f64,
    /// Number of occurrences in the document
    pub count: usize,
    /// Token offsets for each occurrence
    pub offsets: Vec<(usize, usize)>,
    /// The rank (1-indexed, based on score)
    pub rank: usize,
}

impl Phrase {
    /// Create a new phrase
    pub fn new(
        text: impl Into<String>,
        lemma: impl Into<String>,
        score: f64,
        count: usize,
    ) -> Self {
        Self {
            text: text.into(),
            lemma: lemma.into(),
            score,
            count,
            offsets: Vec::new(),
            rank: 0,
        }
    }
}

// ============================================================================
// Sentence
// ============================================================================

/// A sentence from the input text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sentence {
    /// The sentence text
    pub text: String,
    /// Start character offset in original text
    pub start: usize,
    /// End character offset in original text
    pub end: usize,
    /// Sentence index within the document
    pub index: usize,
    /// Start token index (inclusive)
    pub start_token: usize,
    /// End token index (exclusive)
    pub end_token: usize,
    /// Relevance score for summarization
    pub score: f64,
}

impl Sentence {
    /// Create a new sentence
    pub fn new(text: impl Into<String>, start: usize, end: usize, index: usize) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            index,
            start_token: 0,
            end_token: 0,
            score: 0.0,
        }
    }
}

// ============================================================================
// Score Aggregation
// ============================================================================

/// Methods for aggregating scores across multiple occurrences of a phrase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ScoreAggregation {
    /// Sum of all token scores
    #[default]
    Sum,
    /// Arithmetic mean of token scores
    Mean,
    /// Maximum token score
    Max,
    /// Root mean square of token scores
    RootMeanSquare,
}

impl ScoreAggregation {
    /// Aggregate a slice of scores
    pub fn aggregate(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }

        match self {
            ScoreAggregation::Sum => scores.iter().sum(),
            ScoreAggregation::Mean => scores.iter().sum::<f64>() / scores.len() as f64,
            ScoreAggregation::Max => scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ScoreAggregation::RootMeanSquare => {
                let sum_sq: f64 = scores.iter().map(|x| x * x).sum();
                (sum_sq / scores.len() as f64).sqrt()
            }
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for TextRank extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRankConfig {
    /// Damping factor for PageRank (typically 0.85)
    pub damping: f64,
    /// Maximum iterations for PageRank convergence
    pub max_iterations: usize,
    /// Convergence threshold (stop when delta < threshold)
    pub convergence_threshold: f64,
    /// Window size for co-occurrence graph
    pub window_size: usize,
    /// Number of top phrases to return (0 = all)
    pub top_n: usize,
    /// Minimum phrase length in tokens
    pub min_phrase_length: usize,
    /// Maximum phrase length in tokens
    pub max_phrase_length: usize,
    /// Score aggregation method
    pub score_aggregation: ScoreAggregation,
    /// Language code for stopwords (e.g., "en", "de", "fr")
    pub language: String,
    /// Include edge weights in graph (co-occurrence counts)
    pub use_edge_weights: bool,
    /// POS tags to include in graph
    pub include_pos: Vec<PosTag>,
}

impl Default for TextRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            window_size: 4,
            top_n: 10,
            min_phrase_length: 1,
            max_phrase_length: 4,
            score_aggregation: ScoreAggregation::Sum,
            language: "en".to_string(),
            use_edge_weights: true,
            include_pos: vec![PosTag::Noun, PosTag::Adjective, PosTag::ProperNoun],
        }
    }
}

impl TextRankConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.damping) {
            return Err(TextRankError::invalid_config(format!(
                "damping must be between 0 and 1, got {}",
                self.damping
            )));
        }

        if self.max_iterations == 0 {
            return Err(TextRankError::invalid_config("max_iterations must be > 0"));
        }

        if self.convergence_threshold <= 0.0 {
            return Err(TextRankError::invalid_config(
                "convergence_threshold must be > 0",
            ));
        }

        if self.window_size < 2 {
            return Err(TextRankError::invalid_config("window_size must be >= 2"));
        }

        if self.min_phrase_length == 0 {
            return Err(TextRankError::invalid_config(
                "min_phrase_length must be > 0",
            ));
        }

        if self.max_phrase_length < self.min_phrase_length {
            return Err(TextRankError::invalid_config(
                "max_phrase_length must be >= min_phrase_length",
            ));
        }

        Ok(())
    }

    /// Builder method: set damping factor
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Builder method: set max iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Builder method: set convergence threshold
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Builder method: set window size
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Builder method: set top N phrases to return
    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }

    /// Builder method: set language
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Builder method: set score aggregation
    pub fn with_score_aggregation(mut self, aggregation: ScoreAggregation) -> Self {
        self.score_aggregation = aggregation;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_pool() {
        let mut pool = StringPool::new();
        let id1 = pool.intern("hello");
        let id2 = pool.intern("world");
        let id3 = pool.intern("hello"); // duplicate

        assert_eq!(id1, id3); // same string should get same ID
        assert_ne!(id1, id2);
        assert_eq!(pool.get(id1), Some("hello"));
        assert_eq!(pool.get(id2), Some("world"));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_score_aggregation() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];

        assert!((ScoreAggregation::Sum.aggregate(&scores) - 10.0).abs() < 1e-10);
        assert!((ScoreAggregation::Mean.aggregate(&scores) - 2.5).abs() < 1e-10);
        assert!((ScoreAggregation::Max.aggregate(&scores) - 4.0).abs() < 1e-10);

        let rms = ScoreAggregation::RootMeanSquare.aggregate(&scores);
        let expected = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0_f64).sqrt();
        assert!((rms - expected).abs() < 1e-10);
    }

    #[test]
    fn test_config_validation() {
        let config = TextRankConfig::default();
        assert!(config.validate().is_ok());

        let bad_config = TextRankConfig::default().with_damping(1.5);
        assert!(bad_config.validate().is_err());

        let bad_config = TextRankConfig::default().with_window_size(1);
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_chunk_overlap() {
        let c1 = ChunkSpan {
            start_token: 0,
            end_token: 3,
            start_char: 0,
            end_char: 15,
            sentence_idx: 0,
        };
        let c2 = ChunkSpan {
            start_token: 2,
            end_token: 5,
            start_char: 10,
            end_char: 25,
            sentence_idx: 0,
        };
        let c3 = ChunkSpan {
            start_token: 5,
            end_token: 7,
            start_char: 25,
            end_char: 35,
            sentence_idx: 0,
        };

        assert!(c1.overlaps(&c2)); // overlapping
        assert!(!c1.overlaps(&c3)); // non-overlapping
        assert!(!c2.overlaps(&c3)); // adjacent, not overlapping
    }
}
