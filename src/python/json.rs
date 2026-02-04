//! JSON interface for large documents and batch processing
//!
//! For large documents or batch processing, the JSON interface
//! minimizes Python↔Rust overhead by passing pre-tokenized data.

use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::standard::StandardPageRank;
use crate::phrase::extraction::PhraseExtractor;
use crate::types::{PosTag, ScoreAggregation, TextRankConfig, Token};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Input token from JSON
#[derive(Debug, Clone, Deserialize)]
pub struct JsonToken {
    pub text: String,
    pub lemma: String,
    pub pos: String,
    pub start: usize,
    pub end: usize,
    pub sentence_idx: usize,
    pub token_idx: usize,
    #[serde(default)]
    pub is_stopword: bool,
}

impl From<JsonToken> for Token {
    fn from(jt: JsonToken) -> Self {
        Token {
            text: jt.text,
            lemma: jt.lemma,
            pos: PosTag::from_spacy(&jt.pos),
            start: jt.start,
            end: jt.end,
            sentence_idx: jt.sentence_idx,
            token_idx: jt.token_idx,
            is_stopword: jt.is_stopword,
        }
    }
}

/// Input document from JSON
#[derive(Debug, Clone, Deserialize)]
pub struct JsonDocument {
    pub tokens: Vec<JsonToken>,
    #[serde(default)]
    pub config: Option<JsonConfig>,
}

/// Configuration from JSON
#[derive(Debug, Clone, Deserialize, Default)]
pub struct JsonConfig {
    #[serde(default = "default_damping")]
    pub damping: f64,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "default_threshold")]
    pub convergence_threshold: f64,
    #[serde(default = "default_window")]
    pub window_size: usize,
    #[serde(default = "default_top_n")]
    pub top_n: usize,
    #[serde(default = "default_min_length")]
    pub min_phrase_length: usize,
    #[serde(default = "default_max_length")]
    pub max_phrase_length: usize,
    #[serde(default)]
    pub score_aggregation: String,
    #[serde(default = "default_use_edge_weights")]
    pub use_edge_weights: bool,
    /// POS tags to include (e.g., ["NOUN", "ADJ", "PROPN"])
    #[serde(default)]
    pub include_pos: Vec<String>,
}

fn default_use_edge_weights() -> bool {
    true
}

fn default_damping() -> f64 {
    0.85
}
fn default_max_iterations() -> usize {
    100
}
fn default_threshold() -> f64 {
    1e-6
}
fn default_window() -> usize {
    4
}
fn default_top_n() -> usize {
    10
}
fn default_min_length() -> usize {
    1
}
fn default_max_length() -> usize {
    4
}

impl From<JsonConfig> for TextRankConfig {
    fn from(jc: JsonConfig) -> Self {
        let aggregation = match jc.score_aggregation.to_lowercase().as_str() {
            "mean" | "average" => ScoreAggregation::Mean,
            "max" => ScoreAggregation::Max,
            "rms" => ScoreAggregation::RootMeanSquare,
            _ => ScoreAggregation::Sum,
        };

        // Parse include_pos from string tags
        let include_pos: Vec<PosTag> = if jc.include_pos.is_empty() {
            vec![PosTag::Noun, PosTag::Adjective, PosTag::ProperNoun]
        } else {
            jc.include_pos
                .iter()
                .map(|s| PosTag::from_spacy(s))
                .collect()
        };

        TextRankConfig {
            damping: jc.damping,
            max_iterations: jc.max_iterations,
            convergence_threshold: jc.convergence_threshold,
            window_size: jc.window_size,
            top_n: jc.top_n,
            min_phrase_length: jc.min_phrase_length,
            max_phrase_length: jc.max_phrase_length,
            score_aggregation: aggregation,
            language: "en".to_string(),
            use_edge_weights: jc.use_edge_weights,
            include_pos,
        }
    }
}

/// Output phrase for JSON
#[derive(Debug, Clone, Serialize)]
pub struct JsonPhrase {
    pub text: String,
    pub lemma: String,
    pub score: f64,
    pub count: usize,
    pub rank: usize,
}

/// Output result for JSON
#[derive(Debug, Clone, Serialize)]
pub struct JsonResult {
    pub phrases: Vec<JsonPhrase>,
    pub converged: bool,
    pub iterations: usize,
}

/// Extract keyphrases from JSON input
///
/// Args:
///     json_input: JSON string containing tokens and optional config
///
/// Returns:
///     JSON string with extracted phrases
#[pyfunction]
#[pyo3(signature = (json_input))]
pub fn extract_from_json(json_input: &str) -> PyResult<String> {
    // Parse input
    let doc: JsonDocument = serde_json::from_str(json_input)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

    // Convert config
    let config: TextRankConfig = doc.config.unwrap_or_default().into();

    // Convert tokens
    let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

    // Build graph
    let builder = GraphBuilder::from_tokens(&tokens, config.window_size, config.use_edge_weights);

    if builder.is_empty() {
        let result = JsonResult {
            phrases: vec![],
            converged: true,
            iterations: 0,
        };
        return serde_json::to_string(&result)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
    }

    let graph = CsrGraph::from_builder(&builder);

    // Run PageRank
    let pagerank = StandardPageRank::new()
        .with_damping(config.damping)
        .with_max_iterations(config.max_iterations)
        .with_threshold(config.convergence_threshold)
        .run(&graph);

    // Extract phrases
    let extractor = PhraseExtractor::with_config(config);
    let phrases = extractor.extract(&tokens, &graph, &pagerank);

    // Build result
    let result = JsonResult {
        phrases: phrases
            .into_iter()
            .map(|p| JsonPhrase {
                text: p.text,
                lemma: p.lemma,
                score: p.score,
                count: p.count,
                rank: p.rank,
            })
            .collect(),
        converged: pagerank.converged,
        iterations: pagerank.iterations,
    };

    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Batch extract keyphrases from multiple documents
///
/// Args:
///     json_input: JSON string containing array of documents
///
/// Returns:
///     JSON string with array of results
#[pyfunction]
#[pyo3(signature = (json_input))]
pub fn extract_batch_from_json(json_input: &str) -> PyResult<String> {
    // Parse input as array
    let docs: Vec<JsonDocument> = serde_json::from_str(json_input)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

    // Process each document
    let results: Vec<JsonResult> = docs
        .into_iter()
        .map(|doc| {
            let config: TextRankConfig = doc.config.unwrap_or_default().into();
            let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

            let builder =
                GraphBuilder::from_tokens(&tokens, config.window_size, config.use_edge_weights);

            if builder.is_empty() {
                return JsonResult {
                    phrases: vec![],
                    converged: true,
                    iterations: 0,
                };
            }

            let graph = CsrGraph::from_builder(&builder);

            let pagerank = StandardPageRank::new()
                .with_damping(config.damping)
                .with_max_iterations(config.max_iterations)
                .with_threshold(config.convergence_threshold)
                .run(&graph);

            let extractor = PhraseExtractor::with_config(config);
            let phrases = extractor.extract(&tokens, &graph, &pagerank);

            JsonResult {
                phrases: phrases
                    .into_iter()
                    .map(|p| JsonPhrase {
                        text: p.text,
                        lemma: p.lemma,
                        score: p.score,
                        count: p.count,
                        rank: p.rank,
                    })
                    .collect(),
                converged: pagerank.converged,
                iterations: pagerank.iterations,
            }
        })
        .collect();

    serde_json::to_string(&results)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
