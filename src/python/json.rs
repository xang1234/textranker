//! JSON interface for large documents and batch processing
//!
//! For large documents or batch processing, the JSON interface
//! minimizes Python↔Rust overhead by passing pre-tokenized data.

use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::standard::StandardPageRank;
use crate::phrase::extraction::PhraseExtractor;
use crate::types::{PhraseGrouping, PosTag, ScoreAggregation, TextRankConfig, Token};
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
#[derive(Debug, Clone, Deserialize)]
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
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub phrase_grouping: String,
    #[serde(default = "default_use_edge_weights")]
    pub use_edge_weights: bool,
    #[serde(default)]
    pub use_pos_in_nodes: bool,
    /// POS tags to include (e.g., ["NOUN", "ADJ", "PROPN"])
    #[serde(default)]
    pub include_pos: Vec<String>,
    /// Additional stopwords list (extends built-in list when provided)
    #[serde(default)]
    pub stopwords: Vec<String>,
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
    3
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
fn default_language() -> String {
    "en".to_string()
}
fn default_phrase_grouping() -> String {
    "scrubbed_text".to_string()
}

impl Default for JsonConfig {
    fn default() -> Self {
        Self {
            damping: default_damping(),
            max_iterations: default_max_iterations(),
            convergence_threshold: default_threshold(),
            window_size: default_window(),
            top_n: default_top_n(),
            min_phrase_length: default_min_length(),
            max_phrase_length: default_max_length(),
            score_aggregation: String::new(),
            language: default_language(),
            phrase_grouping: default_phrase_grouping(),
            use_edge_weights: default_use_edge_weights(),
            use_pos_in_nodes: true,
            include_pos: Vec::new(),
            stopwords: Vec::new(),
        }
    }
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
            vec![
                PosTag::Noun,
                PosTag::Adjective,
                PosTag::ProperNoun,
                PosTag::Verb,
            ]
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
            language: jc.language,
            use_edge_weights: jc.use_edge_weights,
            include_pos,
            stopwords: jc.stopwords,
            use_pos_in_nodes: jc.use_pos_in_nodes,
            phrase_grouping: PhraseGrouping::from_str(&jc.phrase_grouping),
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
    let mut tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

    if !config.stopwords.is_empty() {
        let stopwords = crate::nlp::stopwords::StopwordFilter::with_additional(
            &config.language,
            &config.stopwords,
        );
        for token in &mut tokens {
            if stopwords.is_stopword(&token.text) {
                token.is_stopword = true;
            }
        }
    }

    // Build graph with POS filtering from config
    let builder = GraphBuilder::from_tokens_with_pos(
        &tokens,
        config.window_size,
        config.use_edge_weights,
        Some(&config.include_pos),
        config.use_pos_in_nodes,
    );

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
            let mut tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

            if !config.stopwords.is_empty() {
                let stopwords = crate::nlp::stopwords::StopwordFilter::with_additional(
                    &config.language,
                    &config.stopwords,
                );
                for token in &mut tokens {
                    if stopwords.is_stopword(&token.text) {
                        token.is_stopword = true;
                    }
                }
            }

            let builder = GraphBuilder::from_tokens_with_pos(
                &tokens,
                config.window_size,
                config.use_edge_weights,
                Some(&config.include_pos),
                config.use_pos_in_nodes,
            );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_include_pos_filtering() {
        // Create JSON with tokens of various POS, config with include_pos=["VERB"]
        // Verify only verbs appear in results (by checking that noun-only keyphrases are excluded)
        let json_input = r#"{
            "tokens": [
                {"text": "machine", "lemma": "machine", "pos": "NOUN", "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0},
                {"text": "runs", "lemma": "run", "pos": "VERB", "start": 8, "end": 12, "sentence_idx": 0, "token_idx": 1},
                {"text": "fast", "lemma": "fast", "pos": "ADV", "start": 13, "end": 17, "sentence_idx": 0, "token_idx": 2},
                {"text": "computer", "lemma": "computer", "pos": "NOUN", "start": 18, "end": 26, "sentence_idx": 0, "token_idx": 3},
                {"text": "processes", "lemma": "process", "pos": "VERB", "start": 27, "end": 36, "sentence_idx": 0, "token_idx": 4}
            ],
            "config": {
                "include_pos": ["VERB"],
                "top_n": 10
            }
        }"#;

        // Parse and convert to TextRankConfig to verify include_pos is respected
        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let config: TextRankConfig = doc.config.unwrap().into();

        // Verify the config has only VERB in include_pos
        assert_eq!(config.include_pos.len(), 1);
        assert_eq!(config.include_pos[0], crate::types::PosTag::Verb);

        // Build graph with the config - should only include verbs
        let tokens: Vec<Token> = serde_json::from_str::<JsonDocument>(json_input)
            .unwrap()
            .tokens
            .into_iter()
            .map(Token::from)
            .collect();

        let builder = GraphBuilder::from_tokens_with_pos(
            &tokens,
            config.window_size,
            config.use_edge_weights,
            Some(&config.include_pos),
            config.use_pos_in_nodes,
        );

        // Should have 2 nodes: "run" and "process" (the verbs)
        assert_eq!(
            builder.node_count(),
            2,
            "Expected 2 verb nodes, got {}",
            builder.node_count()
        );
        assert!(
            builder.get_node_id("run|VERB").is_some(),
            "Expected 'run' verb to be in graph"
        );
        assert!(
            builder.get_node_id("process|VERB").is_some(),
            "Expected 'process' verb to be in graph"
        );
        assert!(
            builder.get_node_id("machine|NOUN").is_none(),
            "Noun 'machine' should not be in graph"
        );
        assert!(
            builder.get_node_id("computer|NOUN").is_none(),
            "Noun 'computer' should not be in graph"
        );
    }

    #[test]
    fn test_json_config_default_include_pos() {
        // When include_pos is empty/not provided, should default to [Noun, Adjective, ProperNoun, Verb]
        let json_input = r#"{
            "tokens": [
                {"text": "machine", "lemma": "machine", "pos": "NOUN", "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0}
            ],
            "config": {}
        }"#;

        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let config: TextRankConfig = doc.config.unwrap().into();

        // Should have default POS tags
        assert_eq!(config.include_pos.len(), 4);
        assert!(config.include_pos.contains(&crate::types::PosTag::Noun));
        assert!(config
            .include_pos
            .contains(&crate::types::PosTag::Adjective));
        assert!(config
            .include_pos
            .contains(&crate::types::PosTag::ProperNoun));
        assert!(config.include_pos.contains(&crate::types::PosTag::Verb));
    }

    #[test]
    fn test_json_include_pos_multiple_tags() {
        // Test with multiple POS tags in include_pos
        let json_input = r#"{
            "tokens": [
                {"text": "machine", "lemma": "machine", "pos": "NOUN", "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0},
                {"text": "learning", "lemma": "learn", "pos": "VERB", "start": 8, "end": 16, "sentence_idx": 0, "token_idx": 1},
                {"text": "smart", "lemma": "smart", "pos": "ADJ", "start": 17, "end": 22, "sentence_idx": 0, "token_idx": 2}
            ],
            "config": {
                "include_pos": ["NOUN", "ADJ"],
                "window_size": 4
            }
        }"#;

        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let config: TextRankConfig = doc.config.unwrap().into();

        let tokens: Vec<Token> = serde_json::from_str::<JsonDocument>(json_input)
            .unwrap()
            .tokens
            .into_iter()
            .map(Token::from)
            .collect();

        let builder = GraphBuilder::from_tokens_with_pos(
            &tokens,
            config.window_size,
            config.use_edge_weights,
            Some(&config.include_pos),
            config.use_pos_in_nodes,
        );

        // Should have 2 nodes: "machine" (NOUN) and "smart" (ADJ)
        assert_eq!(builder.node_count(), 2);
        assert!(builder.get_node_id("machine|NOUN").is_some());
        assert!(builder.get_node_id("smart|ADJ").is_some());
        assert!(
            builder.get_node_id("learn|VERB").is_none(),
            "Verb 'learn' should not be in graph"
        );
    }
}
