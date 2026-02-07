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

pub mod errors;
pub mod graph;
pub mod nlp;
pub mod pagerank;
pub mod phrase;
pub mod summarizer;
pub mod types;
pub mod variants;

#[cfg(feature = "python")]
pub mod python;

// Re-export commonly used types
pub use errors::{Result, TextRankError};
pub use types::{
    ChunkSpan, LemmaId, Phrase, ScoreAggregation, Sentence, StringPool, TextRankConfig, Token,
};

// Re-export main functionality
pub use graph::{builder::GraphBuilder, csr::CsrGraph};
pub use nlp::{stopwords::StopwordFilter, tokenizer::Tokenizer};
pub use pagerank::{
    personalized::PersonalizedPageRank, standard::StandardPageRank, PageRankResult,
};
pub use phrase::extraction::PhraseExtractor;
pub use summarizer::selector::SentenceSelector;
pub use variants::{
    biased_textrank::BiasedTextRank, position_rank::PositionRank, single_rank::SingleRank,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Initialize the Python module
#[cfg(feature = "python")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
