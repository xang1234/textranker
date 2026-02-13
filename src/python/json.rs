//! JSON interface for large documents and batch processing
//!
//! For large documents or batch processing, the JSON interface
//! minimizes Python↔Rust overhead by passing pre-tokenized data.

use crate::phrase::chunker::NounChunker;
use crate::phrase::extraction::extract_keyphrases_with_info;
use crate::pipeline::artifacts::TokenStream;
use crate::pipeline::observer::NoopObserver;
use crate::pipeline::spec::{PipelineSpec, PipelineSpecV1};
use crate::pipeline::spec_builder::SpecPipelineBuilder;
use crate::pipeline::validation::{ValidationEngine, ValidationReport};
use crate::types::{DeterminismMode, PhraseGrouping, PosTag, ScoreAggregation, TextRankConfig, Token};
use crate::variants::biased_textrank::BiasedTextRank;
use crate::variants::multipartite_rank::MultipartiteRank;
use crate::variants::position_rank::PositionRank;
use crate::variants::single_rank::SingleRank;
use crate::variants::topic_rank::TopicRank;
use crate::variants::topical_pagerank::TopicalPageRank;
use crate::variants::Variant;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Pre-tokenized input. Optional when `validate_only` is `true`.
    #[serde(default)]
    pub tokens: Vec<JsonToken>,
    #[serde(default)]
    pub config: Option<JsonConfig>,
    #[serde(default)]
    pub variant: Option<String>,
    /// Optional pipeline specification for modular pipeline configuration.
    #[serde(default)]
    pub pipeline: Option<PipelineSpec>,
    /// When `true`, validate the pipeline spec and return a report
    /// without running extraction.
    #[serde(default)]
    pub validate_only: bool,
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
    #[serde(default = "default_use_pos_in_nodes")]
    pub use_pos_in_nodes: bool,
    /// POS tags to include (e.g., ["NOUN", "ADJ", "PROPN"])
    #[serde(default)]
    pub include_pos: Vec<String>,
    /// Additional stopwords list (extends built-in list when provided)
    #[serde(default)]
    pub stopwords: Vec<String>,
    #[serde(default)]
    pub focus_terms: Vec<String>,
    #[serde(default = "default_bias_weight")]
    pub bias_weight: f64,
    #[serde(default = "default_topic_similarity_threshold")]
    pub topic_similarity_threshold: f64,
    #[serde(default = "default_topic_edge_weight")]
    pub topic_edge_weight: f64,
    /// Topic weights for Topical PageRank: {"lemma": weight, ...}
    #[serde(default)]
    pub topic_weights: HashMap<String, f64>,
    /// Minimum weight for OOV words in Topical PageRank (default 0.0)
    #[serde(default)]
    pub topic_min_weight: f64,
    /// Alpha weight adjustment for MultipartiteRank (default 1.1)
    #[serde(default = "default_multipartite_alpha")]
    pub multipartite_alpha: f64,
    /// Similarity threshold for MultipartiteRank clustering (default 0.26)
    #[serde(default = "default_multipartite_similarity_threshold")]
    pub multipartite_similarity_threshold: f64,
    /// Determinism mode: "default" (fastest) or "deterministic" (reproducible)
    #[serde(default)]
    pub determinism: String,
}

fn default_use_edge_weights() -> bool {
    true
}
fn default_use_pos_in_nodes() -> bool {
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

fn default_bias_weight() -> f64 {
    5.0
}

fn default_topic_similarity_threshold() -> f64 {
    0.25
}

fn default_topic_edge_weight() -> f64 {
    1.0
}

fn default_multipartite_alpha() -> f64 {
    1.1
}

fn default_multipartite_similarity_threshold() -> f64 {
    0.26
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
            focus_terms: Vec::new(),
            bias_weight: default_bias_weight(),
            topic_similarity_threshold: default_topic_similarity_threshold(),
            topic_edge_weight: default_topic_edge_weight(),
            topic_weights: HashMap::new(),
            topic_min_weight: 0.0,
            multipartite_alpha: default_multipartite_alpha(),
            multipartite_similarity_threshold: default_multipartite_similarity_threshold(),
            determinism: String::new(),
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
            phrase_grouping: jc.phrase_grouping.parse().unwrap_or(PhraseGrouping::Lemma),
            determinism: match jc.determinism.to_lowercase().as_str() {
                "deterministic" => DeterminismMode::Deterministic,
                _ => DeterminismMode::Default,
            },
            debug_level: crate::pipeline::artifacts::DebugLevel::None,
        }
    }
}

fn extract_with_variant(
    tokens: &[Token],
    config: &TextRankConfig,
    json_config: &JsonConfig,
    variant: Variant,
) -> crate::phrase::extraction::ExtractionResult {
    match variant {
        Variant::TextRank => extract_keyphrases_with_info(tokens, config),
        Variant::PositionRank => {
            PositionRank::with_config(config.clone()).extract_with_info(tokens)
        }
        Variant::BiasedTextRank => BiasedTextRank::with_config(config.clone())
            .with_focus(
                &json_config
                    .focus_terms
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )
            .with_bias_weight(json_config.bias_weight)
            .extract_with_info(tokens),
        Variant::TopicRank => TopicRank::with_config(config.clone())
            .with_similarity_threshold(json_config.topic_similarity_threshold)
            .with_edge_weight(json_config.topic_edge_weight)
            .extract_with_info(tokens),
        Variant::SingleRank => SingleRank::with_config(config.clone()).extract_with_info(tokens),
        Variant::TopicalPageRank => TopicalPageRank::with_config(config.clone())
            .with_topic_weights(json_config.topic_weights.clone())
            .with_min_weight(json_config.topic_min_weight)
            .extract_with_info(tokens),
        Variant::MultipartiteRank => MultipartiteRank::with_config(config.clone())
            .with_similarity_threshold(json_config.multipartite_similarity_threshold)
            .with_alpha(json_config.multipartite_alpha)
            .extract_with_info(tokens),
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

/// JSON response for validation-only requests.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResponse {
    /// `true` if the spec has no errors (warnings are acceptable).
    pub valid: bool,
    /// All diagnostics (errors and warnings).
    #[serde(flatten)]
    pub report: ValidationReport,
}

/// Validate a pipeline spec (pure Rust, no PyO3 dependency).
///
/// Used by both `extract_from_json` (when `validate_only` is true)
/// and `validate_pipeline_spec`.
pub fn validate_spec_impl(spec: &PipelineSpecV1) -> ValidationResponse {
    let engine = ValidationEngine::with_defaults();
    let report = engine.validate(spec);
    ValidationResponse {
        valid: report.is_valid(),
        report,
    }
}

/// Extract keyphrases from JSON input
///
/// Args:
///     json_input: JSON string containing tokens and optional config.
///         When `validate_only` is true and `pipeline` is present,
///         returns a validation report instead of extraction results.
///
/// Returns:
///     JSON string with extracted phrases, or a validation report
#[pyfunction]
#[pyo3(signature = (json_input))]
pub fn extract_from_json(py: Python<'_>, json_input: &str) -> PyResult<String> {
    // Parse input (cheap — no need to release GIL)
    let doc: JsonDocument = serde_json::from_str(json_input)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

    // Handle validate-only mode (fast path, no extraction)
    if doc.validate_only {
        let pipeline_spec = doc.pipeline.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "validate_only requires a 'pipeline' field",
            )
        })?;
        let response = match &pipeline_spec {
            PipelineSpec::V1(v1) => validate_spec_impl(v1),
            PipelineSpec::Preset(name) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Preset pipeline '{}' resolution is not yet implemented",
                    name
                )));
            }
        };
        return serde_json::to_string(&response)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
    }

    // Convert config
    let json_config = doc.config.unwrap_or_default();
    let config: TextRankConfig = json_config.clone().into();

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

    // Pipeline path — when `pipeline` is present, use the modular pipeline system.
    // This takes precedence over the `variant` field.
    if let Some(ref spec) = doc.pipeline {
        let spec_owned = spec.clone();
        let json_config_owned = json_config.clone();
        let result = py.allow_threads(move || {
            let chunks = NounChunker::new()
                .with_min_length(config.min_phrase_length)
                .with_max_length(config.max_phrase_length)
                .extract_chunks(&tokens);

            let builder = SpecPipelineBuilder::new()
                .with_chunks(chunks)
                .with_focus_terms(json_config_owned.focus_terms.clone(), json_config_owned.bias_weight)
                .with_topic_weights(json_config_owned.topic_weights.clone(), json_config_owned.topic_min_weight);

            let pipeline = builder.build_from_spec(&spec_owned, &config)?;
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            Ok::<_, crate::pipeline::errors::PipelineSpecError>(pipeline.run(stream, &config, &mut obs))
        });

        let formatted = result.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let json_result = JsonResult {
            phrases: formatted
                .phrases
                .into_iter()
                .map(|p| JsonPhrase {
                    text: p.text,
                    lemma: p.lemma,
                    score: p.score,
                    count: p.count,
                    rank: p.rank,
                })
                .collect(),
            converged: formatted.converged,
            iterations: formatted.iterations as usize,
        };

        return serde_json::to_string(&json_result)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
    }

    // Old variant dispatch path (fallback when `pipeline` is absent)
    let variant = doc
        .variant
        .as_deref()
        .and_then(|value| value.parse().ok())
        .unwrap_or(Variant::TextRank);

    // Release the GIL for CPU-intensive extraction.
    let extraction = py.allow_threads(move || {
        extract_with_variant(&tokens, &config, &json_config, variant)
    });

    // Build result (cheap serialization)
    let result = JsonResult {
        phrases: extraction
            .phrases
            .into_iter()
            .map(|p| JsonPhrase {
                text: p.text,
                lemma: p.lemma,
                score: p.score,
                count: p.count,
                rank: p.rank,
            })
            .collect(),
        converged: extraction.converged,
        iterations: extraction.iterations,
    };

    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Validate a pipeline specification without running extraction.
///
/// Args:
///     json_spec: JSON string containing a PipelineSpec object.
///
/// Returns:
///     JSON string with `{"valid": bool, "diagnostics": [...]}`
#[pyfunction]
#[pyo3(signature = (json_spec))]
pub fn validate_pipeline_spec(json_spec: &str) -> PyResult<String> {
    let pipeline_spec: PipelineSpec = serde_json::from_str(json_spec).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid pipeline spec: {}", e))
    })?;
    let response = match &pipeline_spec {
        PipelineSpec::V1(v1) => validate_spec_impl(v1),
        PipelineSpec::Preset(name) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Preset pipeline '{}' resolution is not yet implemented",
                name
            )));
        }
    };
    serde_json::to_string(&response)
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
pub fn extract_batch_from_json(py: Python<'_>, json_input: &str) -> PyResult<String> {
    // Parse input as array (cheap — no need to release GIL)
    let docs: Vec<JsonDocument> = serde_json::from_str(json_input)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

    // Release the GIL for the entire batch of CPU-intensive extractions.
    let results: Vec<JsonResult> = py.allow_threads(move || {
        docs.into_iter()
            .map(|doc| {
                let json_config = doc.config.unwrap_or_default();
                let config: TextRankConfig = json_config.clone().into();
                let variant = doc
                    .variant
                    .as_deref()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(Variant::TextRank);
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

                let extraction =
                    extract_with_variant(&tokens, &config, &json_config, variant);

                JsonResult {
                    phrases: extraction
                        .phrases
                        .into_iter()
                        .map(|p| JsonPhrase {
                            text: p.text,
                            lemma: p.lemma,
                            score: p.score,
                            count: p.count,
                            rank: p.rank,
                        })
                        .collect(),
                    converged: extraction.converged,
                    iterations: extraction.iterations,
                }
            })
            .collect()
    });

    serde_json::to_string(&results)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;
    use crate::pipeline::spec::{PipelineSpec, PipelineSpecV1};

    // ─── Validate-only mode ─────────────────────────────────────────

    #[test]
    fn test_validate_spec_impl_valid() {
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "modules": {
                "rank": { "type": "personalized_pagerank" },
                "teleport": { "type": "position" }
            }
        }"#)
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(resp.valid);
        assert!(resp.report.is_empty());
    }

    #[test]
    fn test_validate_spec_impl_invalid() {
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "modules": { "rank": { "type": "personalized_pagerank" } }
        }"#)
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(!resp.valid);
        assert!(resp.report.has_errors());
    }

    #[test]
    fn test_validate_spec_impl_response_json_shape() {
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "modules": { "rank": { "type": "personalized_pagerank" } }
        }"#)
        .unwrap();
        let resp = validate_spec_impl(&spec);
        let json = serde_json::to_value(&resp).unwrap();

        // Check the top-level shape: { "valid": bool, "diagnostics": [...] }
        assert_eq!(json["valid"], false);
        let diags = json["diagnostics"].as_array().unwrap();
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0]["severity"], "error");
        assert_eq!(diags[0]["code"], "missing_stage");
        assert_eq!(diags[0]["path"], "/modules/teleport");
        assert!(diags[0]["hint"].as_str().is_some());
    }

    #[test]
    fn test_validate_only_document_no_tokens_needed() {
        // validate_only documents should not require tokens
        let doc: JsonDocument = serde_json::from_str(r#"{
            "validate_only": true,
            "pipeline": { "v": 1, "modules": { "rank": { "type": "standard_pagerank" } } }
        }"#)
        .unwrap();
        assert!(doc.validate_only);
        assert!(doc.tokens.is_empty());
        assert!(doc.pipeline.is_some());
    }

    #[test]
    fn test_validate_only_with_warnings() {
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "strict": false,
            "bogus_field": 42
        }"#)
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(resp.valid); // warnings don't invalidate
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["valid"], true);
        let diags = json["diagnostics"].as_array().unwrap();
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0]["severity"], "warning");
    }

    #[test]
    fn test_existing_extraction_still_works_with_new_fields() {
        // Ensure adding pipeline/validate_only fields doesn't break existing behavior
        let json_input = r#"{
            "tokens": [
                {"text": "machine", "lemma": "machine", "pos": "NOUN", "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0}
            ],
            "config": {}
        }"#;
        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        assert!(!doc.validate_only);
        assert!(doc.pipeline.is_none());
        assert_eq!(doc.tokens.len(), 1);
    }

    // ─── End-to-end JSON validation (zvl.5) ────────────────────────────

    #[test]
    fn test_validate_spec_impl_multipartite_valid() {
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "modules": {
                "candidates": { "type": "phrase_candidates" },
                "clustering": { "type": "hac" },
                "graph": { "type": "candidate_graph" },
                "graph_transforms": [{ "type": "remove_intra_cluster_edges" }, { "type": "alpha_boost" }],
                "rank": { "type": "standard_pagerank" }
            }
        }"#).unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(resp.valid);
        assert_eq!(resp.report.len(), 0);
    }

    #[test]
    fn test_validate_spec_impl_multiple_errors_all_returned() {
        // personalized without teleport + topic_graph missing clustering
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "modules": {
                "rank": { "type": "personalized_pagerank" },
                "graph": { "type": "topic_graph" }
            }
        }"#).unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(!resp.valid);

        let json = serde_json::to_value(&resp).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        // At least 3: teleport missing, clustering missing, candidates wrong
        assert!(diags.len() >= 3, "expected >=3 diagnostics, got {}", diags.len());

        // All should be errors
        for d in diags {
            assert_eq!(d["severity"], "error");
        }
    }

    #[test]
    fn test_validate_spec_impl_strict_unknown_in_response() {
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "strict": true,
            "bogus_field": 42
        }"#).unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(!resp.valid);

        let json = serde_json::to_value(&resp).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        assert_eq!(diags[0]["code"], "unknown_field");
        assert_eq!(diags[0]["path"], "/bogus_field");
    }

    #[test]
    fn test_validate_only_document_with_preset_pipeline_spec() {
        // A string pipeline value now parses as PipelineSpec::Preset
        let doc: JsonDocument = serde_json::from_str(r#"{
            "validate_only": true,
            "pipeline": "textrank"
        }"#).unwrap();
        assert!(doc.validate_only);
        let spec = doc.pipeline.unwrap();
        assert!(spec.is_preset());
    }

    #[test]
    fn test_validate_only_missing_pipeline_field() {
        // validate_only=true but no pipeline field
        let doc: JsonDocument = serde_json::from_str(r#"{
            "validate_only": true
        }"#).unwrap();
        assert!(doc.validate_only);
        assert!(doc.pipeline.is_none());
    }

    #[test]
    fn test_validate_response_json_has_valid_and_diagnostics() {
        // Verify the exact JSON shape of a successful validation
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "modules": {
                "rank": { "type": "personalized_pagerank" },
                "teleport": { "type": "focus_terms" }
            }
        }"#).unwrap();
        let resp = validate_spec_impl(&spec);
        let json = serde_json::to_value(&resp).unwrap();

        // Must have "valid" and "diagnostics" at top level
        assert!(json.get("valid").is_some(), "missing 'valid' key");
        assert!(json.get("diagnostics").is_some(), "missing 'diagnostics' key");
        assert_eq!(json["valid"], true);
        assert_eq!(json["diagnostics"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_validate_response_diagnostic_has_all_fields() {
        let spec: PipelineSpecV1 = serde_json::from_str(r#"{
            "v": 1,
            "modules": { "rank": { "type": "personalized_pagerank" } }
        }"#).unwrap();
        let resp = validate_spec_impl(&spec);
        let json = serde_json::to_value(&resp).unwrap();
        let diag = &json["diagnostics"][0];

        // Every diagnostic must have: severity, code, path, message
        assert!(diag.get("severity").is_some(), "missing severity");
        assert!(diag.get("code").is_some(), "missing code");
        assert!(diag.get("path").is_some(), "missing path");
        assert!(diag.get("message").is_some(), "missing message");
        // hint is optional but should be present for this error
        assert!(diag.get("hint").is_some(), "missing hint");
    }

    // ─── Existing tests ─────────────────────────────────────────────────

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

    // ─── Topic family JSON integration tests ───────────────────────

    #[test]
    fn test_json_topic_rank_extraction() {
        let json_input = r#"{
            "tokens": [
                {"text": "Machine", "lemma": "machine", "pos": "NOUN", "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0},
                {"text": "learning", "lemma": "learning", "pos": "NOUN", "start": 8, "end": 16, "sentence_idx": 0, "token_idx": 1},
                {"text": "algorithms", "lemma": "algorithm", "pos": "NOUN", "start": 17, "end": 27, "sentence_idx": 0, "token_idx": 2},
                {"text": "Deep", "lemma": "deep", "pos": "ADJ", "start": 52, "end": 56, "sentence_idx": 1, "token_idx": 3},
                {"text": "learning", "lemma": "learning", "pos": "NOUN", "start": 57, "end": 65, "sentence_idx": 1, "token_idx": 4},
                {"text": "models", "lemma": "model", "pos": "NOUN", "start": 66, "end": 72, "sentence_idx": 1, "token_idx": 5},
                {"text": "neural", "lemma": "neural", "pos": "ADJ", "start": 77, "end": 83, "sentence_idx": 1, "token_idx": 6},
                {"text": "networks", "lemma": "network", "pos": "NOUN", "start": 84, "end": 92, "sentence_idx": 1, "token_idx": 7}
            ],
            "variant": "topic_rank",
            "config": {
                "top_n": 5,
                "determinism": "deterministic"
            }
        }"#;

        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let json_config = doc.config.clone().unwrap_or_default();
        let config: TextRankConfig = json_config.clone().into();
        let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

        let result = extract_with_variant(
            &tokens,
            &config,
            &json_config,
            crate::variants::Variant::TopicRank,
        );

        assert!(result.converged);
        assert!(!result.phrases.is_empty());
        // Scores should be in descending order
        for w in result.phrases.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // Ranks should be 1-indexed and contiguous
        for (i, p) in result.phrases.iter().enumerate() {
            assert_eq!(p.rank, i + 1);
        }
    }

    #[test]
    fn test_json_multipartite_rank_extraction() {
        let json_input = r#"{
            "tokens": [
                {"text": "Machine", "lemma": "machine", "pos": "NOUN", "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0},
                {"text": "learning", "lemma": "learning", "pos": "NOUN", "start": 8, "end": 16, "sentence_idx": 0, "token_idx": 1},
                {"text": "algorithms", "lemma": "algorithm", "pos": "NOUN", "start": 17, "end": 27, "sentence_idx": 0, "token_idx": 2},
                {"text": "neural", "lemma": "neural", "pos": "ADJ", "start": 77, "end": 83, "sentence_idx": 1, "token_idx": 3},
                {"text": "networks", "lemma": "network", "pos": "NOUN", "start": 84, "end": 92, "sentence_idx": 1, "token_idx": 4},
                {"text": "Machine", "lemma": "machine", "pos": "NOUN", "start": 94, "end": 101, "sentence_idx": 2, "token_idx": 5},
                {"text": "learning", "lemma": "learning", "pos": "NOUN", "start": 102, "end": 110, "sentence_idx": 2, "token_idx": 6},
                {"text": "techniques", "lemma": "technique", "pos": "NOUN", "start": 111, "end": 121, "sentence_idx": 2, "token_idx": 7}
            ],
            "variant": "multipartite_rank",
            "config": {
                "top_n": 5,
                "multipartite_alpha": 1.1,
                "multipartite_similarity_threshold": 0.26,
                "determinism": "deterministic"
            }
        }"#;

        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let json_config = doc.config.clone().unwrap_or_default();
        let config: TextRankConfig = json_config.clone().into();
        let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

        let result = extract_with_variant(
            &tokens,
            &config,
            &json_config,
            crate::variants::Variant::MultipartiteRank,
        );

        assert!(result.converged);
        assert!(!result.phrases.is_empty());
        for w in result.phrases.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        for (i, p) in result.phrases.iter().enumerate() {
            assert_eq!(p.rank, i + 1);
        }
    }

    #[test]
    fn test_json_topic_rank_config_parameters() {
        // Verify that topic_similarity_threshold and topic_edge_weight
        // are correctly deserialized and affect the variant dispatch.
        let json_input = r#"{
            "tokens": [
                {"text": "Machine", "lemma": "machine", "pos": "NOUN", "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0},
                {"text": "learning", "lemma": "learning", "pos": "NOUN", "start": 8, "end": 16, "sentence_idx": 0, "token_idx": 1}
            ],
            "variant": "topic_rank",
            "config": {
                "topic_similarity_threshold": 0.5,
                "topic_edge_weight": 2.0
            }
        }"#;

        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let json_config = doc.config.unwrap();
        assert!((json_config.topic_similarity_threshold - 0.5).abs() < 1e-10);
        assert!((json_config.topic_edge_weight - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_json_multipartite_rank_config_parameters() {
        let json_input = r#"{
            "tokens": [],
            "variant": "multipartite_rank",
            "config": {
                "multipartite_alpha": 2.5,
                "multipartite_similarity_threshold": 0.3
            }
        }"#;

        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let json_config = doc.config.unwrap();
        assert!((json_config.multipartite_alpha - 2.5).abs() < 1e-10);
        assert!((json_config.multipartite_similarity_threshold - 0.3).abs() < 1e-10);
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

    // ─── Pipeline execution path tests ─────────────────────────────────

    /// Helper: golden tokens for pipeline tests (multiple sentences for meaningful ranking).
    fn pipeline_test_tokens_json() -> &'static str {
        r#"[
            {"text": "Machine", "lemma": "machine", "pos": "NOUN", "start": 0, "end": 7, "sentence_idx": 0, "token_idx": 0},
            {"text": "learning", "lemma": "learning", "pos": "NOUN", "start": 8, "end": 16, "sentence_idx": 0, "token_idx": 1},
            {"text": "uses", "lemma": "use", "pos": "VERB", "start": 17, "end": 21, "sentence_idx": 0, "token_idx": 2},
            {"text": "algorithms", "lemma": "algorithm", "pos": "NOUN", "start": 22, "end": 32, "sentence_idx": 0, "token_idx": 3},
            {"text": "Deep", "lemma": "deep", "pos": "ADJ", "start": 34, "end": 38, "sentence_idx": 1, "token_idx": 4},
            {"text": "learning", "lemma": "learning", "pos": "NOUN", "start": 39, "end": 47, "sentence_idx": 1, "token_idx": 5},
            {"text": "uses", "lemma": "use", "pos": "VERB", "start": 48, "end": 52, "sentence_idx": 1, "token_idx": 6},
            {"text": "neural", "lemma": "neural", "pos": "ADJ", "start": 53, "end": 59, "sentence_idx": 1, "token_idx": 7},
            {"text": "networks", "lemma": "network", "pos": "NOUN", "start": 60, "end": 68, "sentence_idx": 1, "token_idx": 8},
            {"text": "Machine", "lemma": "machine", "pos": "NOUN", "start": 70, "end": 77, "sentence_idx": 2, "token_idx": 9},
            {"text": "learning", "lemma": "learning", "pos": "NOUN", "start": 78, "end": 86, "sentence_idx": 2, "token_idx": 10},
            {"text": "models", "lemma": "model", "pos": "NOUN", "start": 87, "end": 93, "sentence_idx": 2, "token_idx": 11}
        ]"#
    }

    #[test]
    fn test_pipeline_preset_string_execution() {
        // Pipeline with a preset string should execute through the pipeline path
        let json_input = format!(
            r#"{{"tokens": {}, "pipeline": "textrank", "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        assert!(doc.pipeline.is_some());

        // Execute through the internal extraction path (non-PyO3)
        let json_config = doc.config.unwrap_or_default();
        let config: TextRankConfig = json_config.clone().into();
        let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

        let spec = doc.pipeline.unwrap();
        let chunks = NounChunker::new()
            .with_min_length(config.min_phrase_length)
            .with_max_length(config.max_phrase_length)
            .extract_chunks(&tokens);

        let builder = SpecPipelineBuilder::new()
            .with_chunks(chunks)
            .with_focus_terms(json_config.focus_terms.clone(), json_config.bias_weight)
            .with_topic_weights(json_config.topic_weights.clone(), json_config.topic_min_weight);

        let pipeline = builder.build_from_spec(&spec, &config).unwrap();
        let stream = crate::pipeline::artifacts::TokenStream::from_tokens(&tokens);
        let mut obs = crate::pipeline::observer::NoopObserver;
        let result = pipeline.run(stream, &config, &mut obs);

        assert!(result.converged);
        assert!(!result.phrases.is_empty());
        // Scores should be in descending order
        for w in result.phrases.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_pipeline_v1_object_execution() {
        // Pipeline with a V1 object spec
        let json_input = format!(
            r#"{{"tokens": {}, "pipeline": {{"v": 1, "modules": {{}}}}, "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        assert!(matches!(doc.pipeline, Some(PipelineSpec::V1(_))));

        let json_config = doc.config.unwrap_or_default();
        let config: TextRankConfig = json_config.clone().into();
        let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

        let spec = doc.pipeline.unwrap();
        let chunks = NounChunker::new()
            .with_min_length(config.min_phrase_length)
            .with_max_length(config.max_phrase_length)
            .extract_chunks(&tokens);

        let pipeline = SpecPipelineBuilder::new()
            .with_chunks(chunks)
            .build_from_spec(&spec, &config)
            .unwrap();
        let stream = crate::pipeline::artifacts::TokenStream::from_tokens(&tokens);
        let mut obs = crate::pipeline::observer::NoopObserver;
        let result = pipeline.run(stream, &config, &mut obs);

        assert!(result.converged);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_pipeline_preset_with_module_override() {
        // V1 with preset + module override
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "preset": "position_rank",
                    "modules": {{
                        "graph": {{ "type": "cooccurrence_window", "window_size": 5 }}
                    }}
                }},
                "config": {{"determinism": "deterministic"}}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let json_config = doc.config.unwrap_or_default();
        let config: TextRankConfig = json_config.clone().into();
        let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

        let spec = doc.pipeline.unwrap();
        let chunks = NounChunker::new()
            .with_min_length(config.min_phrase_length)
            .with_max_length(config.max_phrase_length)
            .extract_chunks(&tokens);

        let pipeline = SpecPipelineBuilder::new()
            .with_chunks(chunks)
            .build_from_spec(&spec, &config)
            .unwrap();
        let stream = crate::pipeline::artifacts::TokenStream::from_tokens(&tokens);
        let mut obs = crate::pipeline::observer::NoopObserver;
        let result = pipeline.run(stream, &config, &mut obs);

        assert!(result.converged);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_pipeline_missing_uses_variant_dispatch() {
        // No pipeline field → falls through to old variant dispatch
        let json_input = format!(
            r#"{{"tokens": {}, "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        assert!(doc.pipeline.is_none());

        // Should use variant dispatch (TextRank by default)
        let json_config = doc.config.unwrap_or_default();
        let config: TextRankConfig = json_config.clone().into();
        let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();
        let result = extract_with_variant(&tokens, &config, &json_config, Variant::TextRank);
        assert!(result.converged);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_pipeline_invalid_preset_returns_error() {
        // Invalid preset name should fail at build_from_spec
        let json_input = format!(
            r#"{{"tokens": {}, "pipeline": "invalid_name"}}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let json_config = doc.config.unwrap_or_default();
        let config: TextRankConfig = json_config.clone().into();
        let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

        let spec = doc.pipeline.unwrap();
        let chunks = NounChunker::new().extract_chunks(&tokens);
        let result = SpecPipelineBuilder::new()
            .with_chunks(chunks)
            .build_from_spec(&spec, &config);
        match result {
            Err(err) => assert!(err.message.contains("invalid_name")),
            Ok(_) => panic!("expected error for invalid preset name"),
        }
    }

    #[test]
    fn test_pipeline_takes_precedence_over_variant() {
        // When both pipeline and variant are present, pipeline wins.
        // Use "textrank" pipeline (base) but "biased_textrank" variant —
        // if pipeline wins, it should run without needing focus_terms.
        let json_input = format!(
            r#"{{
                "tokens": {},
                "variant": "biased_textrank",
                "pipeline": "textrank",
                "config": {{"determinism": "deterministic"}}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        assert!(doc.pipeline.is_some());
        assert_eq!(doc.variant.as_deref(), Some("biased_textrank"));

        // Pipeline path should succeed (textrank doesn't need focus_terms)
        let json_config = doc.config.unwrap_or_default();
        let config: TextRankConfig = json_config.clone().into();
        let tokens: Vec<Token> = doc.tokens.into_iter().map(Token::from).collect();

        let spec = doc.pipeline.unwrap();
        let chunks = NounChunker::new()
            .with_min_length(config.min_phrase_length)
            .with_max_length(config.max_phrase_length)
            .extract_chunks(&tokens);

        let pipeline = SpecPipelineBuilder::new()
            .with_chunks(chunks)
            .build_from_spec(&spec, &config)
            .unwrap();
        let stream = crate::pipeline::artifacts::TokenStream::from_tokens(&tokens);
        let mut obs = crate::pipeline::observer::NoopObserver;
        let result = pipeline.run(stream, &config, &mut obs);

        assert!(result.converged);
        assert!(!result.phrases.is_empty());
    }
}
