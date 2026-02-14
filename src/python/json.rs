//! JSON interface for large documents and batch processing
//!
//! For large documents or batch processing, the JSON interface
//! minimizes Python↔Rust overhead by passing pre-tokenized data.

use crate::phrase::chunker::NounChunker;
use crate::phrase::extraction::extract_keyphrases_with_info;
use crate::pipeline::artifacts::{DebugPayload, FormattedResult, PipelineWorkspace, TokenStream};
use crate::pipeline::error_code::ErrorCode;
use crate::pipeline::errors::PipelineRuntimeError;
use crate::pipeline::observer::{NoopObserver, StageTimingObserver};
#[cfg(feature = "sentence-rank")]
use crate::pipeline::runner::SentenceRankPipeline;
use crate::pipeline::spec::{
    resolve_spec, FormatSpec, PipelineSpec, PipelineSpecV1, VALID_PRESETS,
};
use crate::pipeline::spec_builder::SpecPipelineBuilder;
use crate::pipeline::validation::{ValidationDiagnostic, ValidationEngine, ValidationReport};
use crate::types::{
    DeterminismMode, PhraseGrouping, PosTag, ScoreAggregation, TextRankConfig, Token,
};
use crate::variants::biased_textrank::BiasedTextRank;
use crate::variants::multipartite_rank::MultipartiteRank;
use crate::variants::position_rank::PositionRank;
use crate::variants::single_rank::SingleRank;
use crate::variants::topic_rank::TopicRank;
use crate::variants::topical_pagerank::TopicalPageRank;
use crate::variants::Variant;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── DocError ─────────────────────────────────────────────────────────────────
//
// Bridges two error worlds: structured `PipelineRuntimeError` (pipeline path)
// and plain `String` (legacy variant path). Callers serialize via
// `serialize_doc_error` so pipeline consumers get rich `{code, path, stage,
// message, hint}` objects while legacy consumers get backward-compatible strings.

/// Document-level processing error.
#[derive(Debug)]
enum DocError {
    /// Structured error from the modular pipeline path.
    Pipeline(PipelineRuntimeError),
    /// Plain-text error from legacy variant dispatch or non-pipeline code.
    Other(String),
}

impl std::fmt::Display for DocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocError::Pipeline(e) => write!(f, "{e}"),
            DocError::Other(s) => f.write_str(s),
        }
    }
}

/// Serialize a `DocError` into an inline JSON error object.
///
/// - **Pipeline errors** → `{"error": {code, path, stage, message, hint?}, "error_message": "..."}`
/// - **Other errors** → `{"error": "plain string"}`
fn serialize_doc_error(err: &DocError) -> String {
    match err {
        DocError::Pipeline(e) => {
            let obj = serde_json::json!({
                "error": e,
                "error_message": e.to_string(),
            });
            serde_json::to_string(&obj).unwrap()
        }
        DocError::Other(s) => serde_json::to_string(&serde_json::json!({ "error": s })).unwrap(),
    }
}

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
    /// When `true`, return a capabilities discovery response describing
    /// supported versions, presets, and module types.
    #[serde(default)]
    pub capabilities: bool,
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
            debug_top_k: crate::pipeline::artifacts::DebugLevel::DEFAULT_TOP_K,
            max_nodes: None,
            max_edges: None,
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
        #[cfg(feature = "sentence-rank")]
        Variant::SentenceRank => {
            let stream = TokenStream::from_tokens(tokens);
            let mut obs = NoopObserver;
            let formatted = SentenceRankPipeline::sentence_rank().run(stream, config, &mut obs);
            crate::phrase::extraction::ExtractionResult {
                phrases: formatted.phrases,
                converged: formatted.converged,
                iterations: formatted.iterations as usize,
            }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<crate::pipeline::artifacts::DebugPayload>,
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

/// JSON response for capability discovery requests.
#[derive(Debug, Clone, Serialize)]
pub struct CapabilitiesResponse {
    /// Crate version (from Cargo.toml).
    pub version: String,
    /// Supported pipeline spec version numbers.
    pub pipeline_spec_versions: Vec<u32>,
    /// Canonical preset names.
    pub presets: Vec<String>,
    /// Module types organized by pipeline stage.
    pub modules: HashMap<String, Vec<String>>,
}

/// Build the static capabilities response.
///
/// All data is compile-time constant — no tokens, config, or pipeline needed.
pub fn build_capabilities() -> CapabilitiesResponse {
    let mut modules = HashMap::new();
    modules.insert("preprocess".into(), vec!["default".into()]);

    let mut candidates = vec!["word_nodes".into(), "phrase_candidates".into()];
    #[cfg(feature = "sentence-rank")]
    candidates.push("sentence_candidates".into());
    modules.insert("candidates".into(), candidates);

    let mut graph = vec![
        "cooccurrence_window".into(),
        "topic_graph".into(),
        "candidate_graph".into(),
    ];
    #[cfg(feature = "sentence-rank")]
    graph.push("sentence_graph".into());
    modules.insert("graph".into(), graph);

    modules.insert(
        "graph_transforms".into(),
        vec!["remove_intra_cluster_edges".into(), "alpha_boost".into()],
    );
    modules.insert(
        "teleport".into(),
        vec![
            "uniform".into(),
            "position".into(),
            "focus_terms".into(),
            "topic_weights".into(),
        ],
    );
    modules.insert("clustering".into(), vec!["hac".into()]);
    modules.insert(
        "rank".into(),
        vec!["standard_pagerank".into(), "personalized_pagerank".into()],
    );

    let mut phrases = vec!["chunk_phrases".into()];
    #[cfg(feature = "sentence-rank")]
    phrases.push("sentence_phrases".into());
    modules.insert("phrases".into(), phrases);

    let mut format = vec!["standard_json".into(), "standard_json_with_debug".into()];
    #[cfg(feature = "sentence-rank")]
    format.push("sentence_json".into());
    modules.insert("format".into(), format);

    CapabilitiesResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        pipeline_spec_versions: vec![1],
        presets: VALID_PRESETS.iter().map(|s| s.to_string()).collect(),
        modules,
    }
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

/// Attach stage timing data from an observer to a pipeline result.
fn attach_stage_timings(
    mut result: FormattedResult,
    observer: &StageTimingObserver,
) -> FormattedResult {
    let timings: Vec<(String, f64)> = observer
        .reports()
        .iter()
        .map(|(name, report)| (name.to_string(), report.duration_ms()))
        .collect();
    match result.debug.as_mut() {
        Some(payload) => payload.stage_timings = Some(timings),
        None => {
            result.debug = Some(DebugPayload {
                stage_timings: Some(timings),
                ..Default::default()
            });
        }
    }
    result
}

/// Serialize a `JsonResult` respecting the `FormatSpec` (e.g., custom debug key).
fn serialize_result_with_format(
    result: &JsonResult,
    format: Option<&FormatSpec>,
) -> Result<String, DocError> {
    match format {
        Some(FormatSpec::StandardJsonWithDebug { debug_key }) => {
            let key = debug_key.as_deref().unwrap_or("debug");
            if key == "debug" {
                // No renaming needed
                return serde_json::to_string(result).map_err(|e| DocError::Other(e.to_string()));
            }
            let mut value =
                serde_json::to_value(result).map_err(|e| DocError::Other(e.to_string()))?;
            if let Some(debug) = value.as_object_mut().and_then(|m| m.remove("debug")) {
                value
                    .as_object_mut()
                    .unwrap()
                    .insert(key.to_string(), debug);
            }
            serde_json::to_string(&value).map_err(|e| DocError::Other(e.to_string()))
        }
        _ => serde_json::to_string(result).map_err(|e| DocError::Other(e.to_string())),
    }
}

/// Apply resolved spec settings to config and run the pipeline.
///
/// This is the shared wiring logic for both `process_single_doc` and
/// `process_single_doc_with_workspace`. It:
/// 1. Resolves + validates the spec
/// 2. Applies `expose` → `debug_level` + `debug_top_k`
/// 3. Applies `runtime.deterministic`, `max_tokens`, `max_nodes`, `max_edges`
/// 4. Builds the pipeline via `SpecPipelineBuilder`
/// 5. Runs with conditional `StageTimingObserver`
/// 6. Wraps in `runtime.scoped()` for thread control
/// 7. Serializes with `FormatSpec` awareness
fn run_pipeline_from_spec(
    spec: &PipelineSpec,
    tokens: &[Token],
    config: &mut TextRankConfig,
    json_config: &JsonConfig,
    workspace: Option<&mut PipelineWorkspace>,
    force_single_thread: bool,
) -> Result<String, DocError> {
    // 1. Resolve + validate
    let mut resolved = resolve_spec(spec).map_err(|e| DocError::Other(e.to_string()))?;
    if force_single_thread {
        resolved.runtime.single_thread = true;
    }
    let report = ValidationEngine::with_defaults().validate(&resolved);
    if let Some(err) = report.errors().next() {
        return Err(DocError::Other(err.to_string()));
    }

    // 2. Apply expose → debug_level + debug_top_k
    if let Some(ref expose) = resolved.expose {
        config.debug_level = expose.to_debug_level();
        let requested = expose.effective_top_k();
        let allowed = resolved.runtime.effective_debug_top_k();
        config.debug_top_k = requested.min(allowed);
    }

    // 3. Apply runtime controls
    if resolved.runtime.deterministic == Some(true) {
        config.determinism = DeterminismMode::Deterministic;
    }

    // 3a. max_tokens check (before heavy work)
    if let Some(max) = resolved.runtime.max_tokens {
        if tokens.len() > max {
            return Err(DocError::Pipeline(
                PipelineRuntimeError::new(
                    ErrorCode::LimitExceeded,
                    "/runtime/max_tokens",
                    "preprocess",
                    format!(
                        "token count {} exceeds runtime limit of {}",
                        tokens.len(),
                        max
                    ),
                )
                .with_hint("Increase runtime.max_tokens or reduce input size"),
            ));
        }
    }

    // 3b. Apply graph limits to config (checked inside runner)
    if let Some(max) = resolved.runtime.max_nodes {
        config.max_nodes = Some(max);
    }
    if let Some(max) = resolved.runtime.max_edges {
        config.max_edges = Some(max);
    }

    // 4. Build pipeline
    let chunks = NounChunker::new()
        .with_min_length(config.min_phrase_length)
        .with_max_length(config.max_phrase_length)
        .extract_chunks(tokens);

    let builder = SpecPipelineBuilder::new()
        .with_chunks(chunks)
        .with_focus_terms(json_config.focus_terms.clone(), json_config.bias_weight)
        .with_topic_weights(
            json_config.topic_weights.clone(),
            json_config.topic_min_weight,
        );

    let pipeline = builder
        .build(&resolved, config)
        .map_err(|e| DocError::Other(e.to_string()))?;

    let use_timings = resolved.expose.as_ref().is_some_and(|e| e.stage_timings);
    let format_spec = resolved.modules.format.as_ref();

    // 5+6. Run pipeline (conditional observer + scoped threading)
    let stream = TokenStream::from_tokens(tokens);
    let formatted = resolved.runtime.scoped(|| {
        if use_timings {
            let mut timing_obs = StageTimingObserver::new();
            let result = match workspace {
                Some(ws) => {
                    ws.clear();
                    pipeline.run_with_workspace(stream, config, &mut timing_obs, ws)
                }
                None => pipeline.run(stream, config, &mut timing_obs),
            };
            attach_stage_timings(result, &timing_obs)
        } else {
            let mut obs = NoopObserver;
            match workspace {
                Some(ws) => {
                    ws.clear();
                    pipeline.run_with_workspace(stream, config, &mut obs, ws)
                }
                None => pipeline.run(stream, config, &mut obs),
            }
        }
    });

    // Check for pipeline-level errors (graph limit exceeded)
    if let Some(err) = formatted.error {
        return Err(DocError::Pipeline(err));
    }

    // 7. Serialize with format awareness
    let json_result = formatted_to_json_result(formatted);
    serialize_result_with_format(&json_result, format_spec)
}

/// Convert an `ExtractionResult` (from legacy variant dispatch) into a `JsonResult`.
fn extraction_to_json_result(result: crate::phrase::extraction::ExtractionResult) -> JsonResult {
    JsonResult {
        phrases: result
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
        converged: result.converged,
        iterations: result.iterations,
        debug: None,
    }
}

/// Convert a `FormattedResult` (from modular pipeline) into a `JsonResult`.
fn formatted_to_json_result(result: crate::pipeline::artifacts::FormattedResult) -> JsonResult {
    JsonResult {
        phrases: result
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
        converged: result.converged,
        iterations: result.iterations as usize,
        debug: result.debug,
    }
}

/// Process a single `JsonDocument` → serialized JSON result string.
///
/// Handles capabilities, validate_only, pipeline spec, and legacy variant
/// dispatch.  Pure Rust — no PyO3 dependency — so it can be called from
/// both `extract_from_json` (single doc) and `extract_jsonl_from_json`
/// (streaming).
fn process_single_doc(doc: JsonDocument) -> Result<String, DocError> {
    // Capabilities discovery (cheapest path)
    if doc.capabilities {
        let response = build_capabilities();
        return serde_json::to_string(&response).map_err(|e| DocError::Other(e.to_string()));
    }

    // Validate-only mode (fast path, no extraction)
    if doc.validate_only {
        let pipeline_spec = doc.pipeline.ok_or_else(|| {
            DocError::Other("validate_only requires a 'pipeline' field".to_string())
        })?;
        let response = match resolve_spec(&pipeline_spec) {
            Ok(resolved) => validate_spec_impl(&resolved),
            Err(err) => ValidationResponse {
                valid: false,
                report: ValidationReport {
                    diagnostics: vec![ValidationDiagnostic::error(err)],
                },
            },
        };
        return serde_json::to_string(&response).map_err(|e| DocError::Other(e.to_string()));
    }

    // Convert config & tokens
    let json_config = doc.config.unwrap_or_default();
    let mut config: TextRankConfig = json_config.clone().into();
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

    // Pipeline path — takes precedence over `variant`
    if let Some(ref spec) = doc.pipeline {
        return run_pipeline_from_spec(spec, &tokens, &mut config, &json_config, None, false);
    }

    // Legacy variant dispatch (fallback when `pipeline` is absent)
    let variant = doc
        .variant
        .as_deref()
        .and_then(|value| value.parse().ok())
        .unwrap_or(Variant::TextRank);

    let extraction = extract_with_variant(&tokens, &config, &json_config, variant);
    let json_result = extraction_to_json_result(extraction);
    serde_json::to_string(&json_result).map_err(|e| DocError::Other(e.to_string()))
}

/// Process a single `JsonDocument` with optional workspace reuse.
///
/// When `workspace` is `Some`, pipeline-path documents call
/// `pipeline.run_with_workspace()` (clearing first) instead of `pipeline.run()`.
/// Legacy variant and non-extraction paths ignore the workspace.
///
/// When `force_single_thread` is `true`, the resolved pipeline spec's
/// `runtime.single_thread` is set to `true` before execution. This prevents
/// per-document parallelism when the batch-level caller is already using Rayon.
fn process_single_doc_with_workspace(
    doc: JsonDocument,
    workspace: Option<&mut PipelineWorkspace>,
    force_single_thread: bool,
) -> Result<String, DocError> {
    // Capabilities discovery (cheapest path)
    if doc.capabilities {
        let response = build_capabilities();
        return serde_json::to_string(&response).map_err(|e| DocError::Other(e.to_string()));
    }

    // Validate-only mode (fast path, no extraction)
    if doc.validate_only {
        let pipeline_spec = doc.pipeline.ok_or_else(|| {
            DocError::Other("validate_only requires a 'pipeline' field".to_string())
        })?;
        let response = match resolve_spec(&pipeline_spec) {
            Ok(resolved) => validate_spec_impl(&resolved),
            Err(err) => ValidationResponse {
                valid: false,
                report: ValidationReport {
                    diagnostics: vec![ValidationDiagnostic::error(err)],
                },
            },
        };
        return serde_json::to_string(&response).map_err(|e| DocError::Other(e.to_string()));
    }

    // Convert config & tokens
    let json_config = doc.config.unwrap_or_default();
    let mut config: TextRankConfig = json_config.clone().into();
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

    // Pipeline path — takes precedence over `variant`
    if let Some(ref spec) = doc.pipeline {
        return run_pipeline_from_spec(
            spec,
            &tokens,
            &mut config,
            &json_config,
            workspace,
            force_single_thread,
        );
    }

    // Legacy variant dispatch (fallback when `pipeline` is absent)
    let variant = doc
        .variant
        .as_deref()
        .and_then(|value| value.parse().ok())
        .unwrap_or(Variant::TextRank);

    let extraction = extract_with_variant(&tokens, &config, &json_config, variant);
    let json_result = extraction_to_json_result(extraction);
    serde_json::to_string(&json_result).map_err(|e| DocError::Other(e.to_string()))
}

/// Batch parallelism strategy.
enum BatchPool {
    /// Run documents sequentially (num_threads=1).
    Sequential,
    /// Use the Rayon global thread pool (num_threads=None / auto).
    Global,
    /// Use a dedicated thread pool with a specific thread count.
    Custom(rayon::ThreadPool),
}

impl BatchPool {
    /// Execute `f` inside the appropriate pool.
    ///
    /// - `Sequential` → not called (caller handles sequential path separately)
    /// - `Global` → calls `f()` directly (Rayon global pool used by default)
    /// - `Custom` → calls `pool.install(f)` so `par_iter()` uses this pool
    fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match self {
            BatchPool::Sequential => unreachable!("caller should handle sequential path"),
            BatchPool::Global => f(),
            BatchPool::Custom(pool) => pool.install(f),
        }
    }

    fn is_sequential(&self) -> bool {
        matches!(self, BatchPool::Sequential)
    }
}

/// Build a batch parallelism strategy (pure-Rust core).
///
/// - `None` → auto (Rayon global pool, no allocation)
/// - `Some(1)` → sequential
/// - `Some(n)` where n > 1 → dedicated pool with n threads
/// - `Some(0)` → error
fn build_batch_pool_inner(num_threads: Option<usize>) -> Result<BatchPool, String> {
    match num_threads {
        Some(0) => Err("num_threads must be >= 1 or None".to_string()),
        Some(1) => Ok(BatchPool::Sequential),
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map(BatchPool::Custom)
            .map_err(|e| e.to_string()),
        None => Ok(BatchPool::Global),
    }
}

/// PyO3 wrapper around [`build_batch_pool_inner`].
fn build_batch_pool(num_threads: Option<usize>) -> PyResult<BatchPool> {
    build_batch_pool_inner(num_threads).map_err(|e| {
        if e.contains("num_threads must be") {
            pyo3::exceptions::PyValueError::new_err(e)
        } else {
            pyo3::exceptions::PyRuntimeError::new_err(e)
        }
    })
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
    let doc: JsonDocument = serde_json::from_str(json_input)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

    py.allow_threads(move || process_single_doc(doc))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Extract keyphrases from JSONL (newline-delimited JSON) input.
///
/// Each non-blank input line is parsed as an independent `JsonDocument`
/// and processed through the same logic as `extract_from_json`.
/// Output is a JSONL string with one result line per non-blank input line.
///
/// Per-line error handling: a malformed or failing line produces
/// `{"error": "..."}` on the corresponding output line; processing
/// continues for subsequent lines.
///
/// Args:
///     jsonl_input: JSONL string — one JSON document per line.
///
/// Returns:
///     JSONL string with one result per non-blank input line.
#[pyfunction]
#[pyo3(signature = (jsonl_input, num_threads=None))]
pub fn extract_jsonl_from_json(
    py: Python<'_>,
    jsonl_input: &str,
    num_threads: Option<usize>,
) -> PyResult<String> {
    let pool = build_batch_pool(num_threads)?;
    let input = jsonl_input.to_owned();
    let result = py.allow_threads(move || {
        // Pre-parse non-blank lines (par_iter().collect() preserves input order)
        let lines: Vec<&str> = input
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        let process_line = |line: &str, ws: Option<&mut PipelineWorkspace>, force_st: bool| {
            match serde_json::from_str::<JsonDocument>(line) {
                Ok(doc) => match process_single_doc_with_workspace(doc, ws, force_st) {
                    Ok(json) => json,
                    Err(e) => serialize_doc_error(&e),
                },
                Err(e) => serialize_doc_error(&DocError::Other(format!("Invalid JSON: {e}"))),
            }
        };

        if pool.is_sequential() {
            let mut ws = PipelineWorkspace::new();
            let mut output = String::new();
            for line in &lines {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(&process_line(line, Some(&mut ws), false));
            }
            output
        } else {
            let results: Vec<String> = pool.install(|| {
                lines
                    .par_iter()
                    .map(|line| {
                        let mut ws = PipelineWorkspace::new();
                        process_line(line, Some(&mut ws), true)
                    })
                    .collect()
            });
            results.join("\n")
        }
    });
    Ok(result)
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
    let response = match resolve_spec(&pipeline_spec) {
        Ok(resolved) => validate_spec_impl(&resolved),
        Err(err) => ValidationResponse {
            valid: false,
            report: ValidationReport {
                diagnostics: vec![ValidationDiagnostic::error(err)],
            },
        },
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
#[pyo3(signature = (json_input, num_threads=None))]
pub fn extract_batch_from_json(
    py: Python<'_>,
    json_input: &str,
    num_threads: Option<usize>,
) -> PyResult<String> {
    let docs: Vec<JsonDocument> = serde_json::from_str(json_input)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
    let pool = build_batch_pool(num_threads)?;

    // Release the GIL for the entire batch.
    let result = py.allow_threads(move || {
        if pool.is_sequential() {
            // Sequential path: reuse a single workspace across iterations.
            let mut ws = PipelineWorkspace::new();
            let mut output = String::from("[");
            for (i, doc) in docs.into_iter().enumerate() {
                if i > 0 {
                    output.push(',');
                }
                match process_single_doc_with_workspace(doc, Some(&mut ws), false) {
                    Ok(json) => output.push_str(&json),
                    Err(e) => output.push_str(&serialize_doc_error(&e)),
                }
            }
            output.push(']');
            output
        } else {
            // Parallel path: each Rayon task gets its own workspace.
            // par_iter().collect() preserves input order (Rayon guarantee).
            // force_single_thread = true prevents nested per-document parallelism.
            let results: Vec<String> = pool.install(|| {
                docs.into_par_iter()
                    .map(|doc| {
                        let mut ws = PipelineWorkspace::new();
                        match process_single_doc_with_workspace(doc, Some(&mut ws), true) {
                            Ok(json) => json,
                            Err(e) => serialize_doc_error(&e),
                        }
                    })
                    .collect()
            });
            let mut output = String::from("[");
            for (i, r) in results.iter().enumerate() {
                if i > 0 {
                    output.push(',');
                }
                output.push_str(r);
            }
            output.push(']');
            output
        }
    });
    Ok(result)
}

/// Python iterator that yields one JSON result string per document.
///
/// Created by [`extract_batch_iter`].  When `num_threads != Some(1)`, all
/// documents are processed in parallel upfront and results are stored in
/// `precomputed`.  Otherwise, documents are processed lazily with workspace
/// reuse.
#[pyclass(name = "BatchIter")]
pub struct JsonBatchIter {
    docs: Vec<Option<JsonDocument>>,
    index: usize,
    workspace: PipelineWorkspace,
    /// When parallel mode was requested, all results are computed eagerly.
    precomputed: Option<Vec<String>>,
    precomputed_index: usize,
}

#[pymethods]
impl JsonBatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<String> {
        // Parallel eager path: take from precomputed results (avoids cloning)
        if let Some(ref mut results) = self.precomputed {
            if self.precomputed_index < results.len() {
                let idx = self.precomputed_index;
                self.precomputed_index += 1;
                return Some(std::mem::take(&mut results[idx]));
            }
            return None;
        }

        // Sequential lazy path: process one doc at a time
        while self.index < self.docs.len() {
            let idx = self.index;
            self.index += 1;
            if let Some(doc) = self.docs[idx].take() {
                // Move workspace out so it can cross into allow_threads closure.
                // std::mem::take swaps with Default (empty vecs) — zero allocation.
                let mut ws = std::mem::take(&mut self.workspace);
                let (result, ws_back) = py.allow_threads(move || {
                    let r = process_single_doc_with_workspace(doc, Some(&mut ws), false);
                    (r, ws)
                });
                self.workspace = ws_back;
                return Some(match result {
                    Ok(json) => json,
                    Err(e) => serialize_doc_error(&e),
                });
            }
        }
        None
    }

    fn __len__(&self) -> usize {
        if let Some(ref results) = self.precomputed {
            results.len()
        } else {
            self.docs.len()
        }
    }

    #[getter]
    fn remaining(&self) -> usize {
        if let Some(ref results) = self.precomputed {
            results.len().saturating_sub(self.precomputed_index)
        } else {
            self.docs.len().saturating_sub(self.index)
        }
    }
}

/// Create a lazy batch iterator over JSON documents.
///
/// When `num_threads` is `None` (default) or `> 1`, all documents are
/// processed in parallel upfront. Use `num_threads=1` for sequential
/// lazy iteration with workspace reuse.
///
/// Args:
///     json_input: JSON string containing an array of documents (same format
///         as `extract_batch_from_json`).
///     num_threads: Number of threads for parallel processing. `None` for
///         auto (Rayon default), `1` for sequential.
///
/// Returns:
///     A `BatchIter` that yields one JSON result string per document.
#[pyfunction]
#[pyo3(signature = (json_input, num_threads=None))]
pub fn extract_batch_iter(
    py: Python<'_>,
    json_input: &str,
    num_threads: Option<usize>,
) -> PyResult<JsonBatchIter> {
    let docs: Vec<JsonDocument> = serde_json::from_str(json_input)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
    let pool = build_batch_pool(num_threads)?;

    if pool.is_sequential() {
        Ok(JsonBatchIter {
            docs: docs.into_iter().map(Some).collect(),
            index: 0,
            workspace: PipelineWorkspace::new(),
            precomputed: None,
            precomputed_index: 0,
        })
    } else {
        // Eager-parallel: process all docs upfront
        let precomputed = py.allow_threads(move || {
            pool.install(|| {
                docs.into_par_iter()
                    .map(|doc| {
                        let mut ws = PipelineWorkspace::new();
                        match process_single_doc_with_workspace(doc, Some(&mut ws), true) {
                            Ok(json) => json,
                            Err(e) => serialize_doc_error(&e),
                        }
                    })
                    .collect::<Vec<_>>()
            })
        });
        Ok(JsonBatchIter {
            docs: Vec::new(),
            index: 0,
            workspace: PipelineWorkspace::new(),
            precomputed: Some(precomputed),
            precomputed_index: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;
    use crate::pipeline::spec::{PipelineSpec, PipelineSpecV1};

    // ─── Validate-only mode ─────────────────────────────────────────

    #[test]
    fn test_validate_spec_impl_valid() {
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "modules": {
                "rank": { "type": "personalized_pagerank" },
                "teleport": { "type": "position" }
            }
        }"#,
        )
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(resp.valid);
        assert!(resp.report.is_empty());
    }

    #[test]
    fn test_validate_spec_impl_invalid() {
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "modules": { "rank": { "type": "personalized_pagerank" } }
        }"#,
        )
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(!resp.valid);
        assert!(resp.report.has_errors());
    }

    #[test]
    fn test_validate_spec_impl_response_json_shape() {
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "modules": { "rank": { "type": "personalized_pagerank" } }
        }"#,
        )
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
        let doc: JsonDocument = serde_json::from_str(
            r#"{
            "validate_only": true,
            "pipeline": { "v": 1, "modules": { "rank": { "type": "standard_pagerank" } } }
        }"#,
        )
        .unwrap();
        assert!(doc.validate_only);
        assert!(doc.tokens.is_empty());
        assert!(doc.pipeline.is_some());
    }

    #[test]
    fn test_validate_only_with_warnings() {
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "strict": false,
            "bogus_field": 42
        }"#,
        )
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
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "modules": {
                "rank": { "type": "personalized_pagerank" },
                "graph": { "type": "topic_graph" }
            }
        }"#,
        )
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(!resp.valid);

        let json = serde_json::to_value(&resp).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        // At least 3: teleport missing, clustering missing, candidates wrong
        assert!(
            diags.len() >= 3,
            "expected >=3 diagnostics, got {}",
            diags.len()
        );

        // All should be errors
        for d in diags {
            assert_eq!(d["severity"], "error");
        }
    }

    #[test]
    fn test_validate_spec_impl_strict_unknown_in_response() {
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "strict": true,
            "bogus_field": 42
        }"#,
        )
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(!resp.valid);

        let json = serde_json::to_value(&resp).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        assert_eq!(diags[0]["code"], "unknown_field");
        assert_eq!(diags[0]["path"], "/bogus_field");
    }

    #[test]
    fn test_validate_only_document_with_preset_pipeline_spec() {
        // A string pipeline value parses as PipelineSpec::Preset
        // and resolves + validates successfully
        let doc: JsonDocument = serde_json::from_str(
            r#"{
            "validate_only": true,
            "pipeline": "textrank"
        }"#,
        )
        .unwrap();
        assert!(doc.validate_only);
        let spec = doc.pipeline.unwrap();
        assert!(spec.is_preset());

        // After resolve_spec, validation should succeed
        let resolved = resolve_spec(&spec).unwrap();
        let resp = validate_spec_impl(&resolved);
        assert!(resp.valid);
    }

    #[test]
    fn test_validate_only_missing_pipeline_field() {
        // validate_only=true but no pipeline field
        let doc: JsonDocument = serde_json::from_str(
            r#"{
            "validate_only": true
        }"#,
        )
        .unwrap();
        assert!(doc.validate_only);
        assert!(doc.pipeline.is_none());
    }

    #[test]
    fn test_validate_response_json_has_valid_and_diagnostics() {
        // Verify the exact JSON shape of a successful validation
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "modules": {
                "rank": { "type": "personalized_pagerank" },
                "teleport": { "type": "focus_terms" }
            }
        }"#,
        )
        .unwrap();
        let resp = validate_spec_impl(&spec);
        let json = serde_json::to_value(&resp).unwrap();

        // Must have "valid" and "diagnostics" at top level
        assert!(json.get("valid").is_some(), "missing 'valid' key");
        assert!(
            json.get("diagnostics").is_some(),
            "missing 'diagnostics' key"
        );
        assert_eq!(json["valid"], true);
        assert_eq!(json["diagnostics"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_validate_response_diagnostic_has_all_fields() {
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "modules": { "rank": { "type": "personalized_pagerank" } }
        }"#,
        )
        .unwrap();
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
            .with_topic_weights(
                json_config.topic_weights.clone(),
                json_config.topic_min_weight,
            );

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

    // ─── Preset validation (textranker-04a.6) ──────────────────────────

    #[test]
    fn test_validate_only_preset_resolves_and_validates() {
        // "textrank" preset should resolve to a valid V1 spec
        let spec = PipelineSpec::Preset("textrank".into());
        let resolved = resolve_spec(&spec).unwrap();
        let resp = validate_spec_impl(&resolved);
        assert!(resp.valid);
        assert_eq!(resp.report.len(), 0);
    }

    #[test]
    fn test_validate_only_invalid_preset_returns_error_response() {
        // An invalid preset should produce a ValidationResponse (not a panic/exception)
        let spec = PipelineSpec::Preset("nonexistent".into());
        let response = match resolve_spec(&spec) {
            Ok(resolved) => validate_spec_impl(&resolved),
            Err(err) => ValidationResponse {
                valid: false,
                report: ValidationReport {
                    diagnostics: vec![ValidationDiagnostic::error(err)],
                },
            },
        };
        assert!(!response.valid);
        let json = serde_json::to_value(&response).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0]["code"], "invalid_value");
        assert!(diags[0]["message"]
            .as_str()
            .unwrap()
            .contains("nonexistent"));
    }

    #[test]
    fn test_validate_only_v1_with_preset_resolves_before_validation() {
        // V1 spec with a preset field should merge then validate
        let spec = PipelineSpec::V1(Box::new(PipelineSpecV1 {
            v: 1,
            preset: Some("position_rank".into()),
            modules: crate::pipeline::spec::ModuleSet::default(),
            runtime: crate::pipeline::spec::RuntimeSpec::default(),
            expose: None,
            strict: false,
            unknown_fields: std::collections::HashMap::new(),
        }));
        let resolved = resolve_spec(&spec).unwrap();
        let resp = validate_spec_impl(&resolved);
        assert!(resp.valid);
        // Should have position teleport from the preset
        assert!(resolved.modules.teleport.is_some());
    }

    // ─── Capability discovery (textranker-04a.7) ────────────────────────

    #[test]
    fn test_capabilities_response_shape() {
        let json_input = r#"{"capabilities": true}"#;
        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        assert!(doc.capabilities);

        let resp = build_capabilities();
        let json = serde_json::to_value(&resp).unwrap();

        // Must have all four top-level keys
        assert!(json.get("version").is_some(), "missing 'version'");
        assert!(
            json.get("pipeline_spec_versions").is_some(),
            "missing 'pipeline_spec_versions'"
        );
        assert!(json.get("presets").is_some(), "missing 'presets'");
        assert!(json.get("modules").is_some(), "missing 'modules'");

        // version is a non-empty string
        assert!(!json["version"].as_str().unwrap().is_empty());
        // pipeline_spec_versions contains 1
        assert_eq!(json["pipeline_spec_versions"].as_array().unwrap(), &[1]);
    }

    #[test]
    fn test_capabilities_modules_has_all_stages() {
        let resp = build_capabilities();
        let expected_stages = [
            "preprocess",
            "candidates",
            "graph",
            "graph_transforms",
            "teleport",
            "clustering",
            "rank",
            "phrases",
            "format",
        ];
        assert_eq!(resp.modules.len(), expected_stages.len());
        for stage in &expected_stages {
            assert!(
                resp.modules.contains_key(*stage),
                "missing stage '{stage}' in capabilities modules"
            );
            assert!(
                !resp.modules[*stage].is_empty(),
                "stage '{stage}' has no module types"
            );
        }
    }

    #[test]
    fn test_capabilities_presets_match_valid_presets() {
        let resp = build_capabilities();
        let expected: Vec<String> = VALID_PRESETS.iter().map(|s| s.to_string()).collect();
        assert_eq!(resp.presets, expected);
    }

    #[test]
    fn test_capabilities_document_no_tokens_needed() {
        // capabilities requests should parse without tokens or config
        let doc: JsonDocument = serde_json::from_str(r#"{"capabilities": true}"#).unwrap();
        assert!(doc.capabilities);
        assert!(doc.tokens.is_empty());
        assert!(doc.config.is_none());
        assert!(doc.pipeline.is_none());
    }

    // ─── Integration: PipelineSpec round-trips (textranker-04a.8) ────────

    /// All 7 presets build and run through the pipeline path, producing phrases.
    #[test]
    fn test_all_presets_execute_through_pipeline() {
        let presets = VALID_PRESETS;
        for preset in presets {
            let json_input = format!(
                r#"{{"tokens": {}, "pipeline": "{preset}", "config": {{"determinism": "deterministic"}}}}"#,
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

            let mut builder = SpecPipelineBuilder::new().with_chunks(chunks);
            // Provide context that some presets require
            if *preset == "biased_textrank" {
                builder = builder.with_focus_terms(vec!["machine".into()], 5.0);
            }
            if *preset == "topical_pagerank" {
                builder = builder.with_topic_weights(
                    [("machine".into(), 1.0), ("learning".into(), 0.8)].into(),
                    0.0,
                );
            }

            let pipeline = builder
                .build_from_spec(&spec, &config)
                .unwrap_or_else(|e| panic!("preset '{preset}' failed to build: {e}"));
            let stream = crate::pipeline::artifacts::TokenStream::from_tokens(&tokens);
            let mut obs = crate::pipeline::observer::NoopObserver;
            let result = pipeline.run(stream, &config, &mut obs);

            assert!(result.converged, "preset '{preset}' did not converge");
            assert!(
                !result.phrases.is_empty(),
                "preset '{preset}' produced no phrases"
            );
            // Scores should be sorted descending
            for w in result.phrases.windows(2) {
                assert!(
                    w[0].score >= w[1].score,
                    "preset '{preset}' has unsorted scores: {} > {}",
                    w[0].score,
                    w[1].score
                );
            }
        }
    }

    /// All 7 presets validate successfully through validate_only mode.
    #[test]
    fn test_all_presets_validate_successfully() {
        for preset in VALID_PRESETS {
            let spec = PipelineSpec::Preset(preset.to_string());
            let resolved = resolve_spec(&spec)
                .unwrap_or_else(|e| panic!("preset '{preset}' failed to resolve: {e}"));
            let resp = validate_spec_impl(&resolved);
            assert!(
                resp.valid,
                "preset '{preset}' validation failed: {:?}",
                resp.report
            );
        }
    }

    /// Serialize a V1 spec to JSON, deserialize it back, build, and run —
    /// verify the round-trip produces identical results.
    #[test]
    fn test_spec_serialize_deserialize_build_run_roundtrip() {
        use crate::pipeline::spec::{GraphSpec, ModuleSet, RankSpec, RuntimeSpec, TeleportSpec};

        // Build a non-trivial V1 spec programmatically
        let original = PipelineSpecV1 {
            v: 1,
            preset: None,
            modules: ModuleSet {
                graph: Some(GraphSpec::CooccurrenceWindow {
                    window_size: Some(4),
                    cross_sentence: Some(true),
                    edge_weighting: None,
                }),
                rank: Some(RankSpec::PersonalizedPagerank {
                    damping: Some(0.9),
                    max_iterations: Some(150),
                    convergence_threshold: None,
                }),
                teleport: Some(TeleportSpec::Position { shape: None }),
                ..Default::default()
            },
            runtime: RuntimeSpec::default(),
            expose: None,
            strict: false,
            unknown_fields: std::collections::HashMap::new(),
        };

        // Serialize → JSON string → deserialize
        let json_str = serde_json::to_string(&original).unwrap();
        let round_tripped: PipelineSpecV1 = serde_json::from_str(&json_str).unwrap();

        // Both should build and produce identical results
        let config = TextRankConfig {
            determinism: crate::types::DeterminismMode::Deterministic,
            ..TextRankConfig::default()
        };
        let tokens_json = pipeline_test_tokens_json();
        let tokens: Vec<Token> = serde_json::from_str::<Vec<JsonToken>>(tokens_json)
            .unwrap()
            .into_iter()
            .map(Token::from)
            .collect();

        let run_with_spec = |spec: &PipelineSpecV1| {
            let chunks = NounChunker::new()
                .with_min_length(config.min_phrase_length)
                .with_max_length(config.max_phrase_length)
                .extract_chunks(&tokens);
            let pipeline = SpecPipelineBuilder::new()
                .with_chunks(chunks)
                .build(spec, &config)
                .unwrap();
            let stream = crate::pipeline::artifacts::TokenStream::from_tokens(&tokens);
            let mut obs = crate::pipeline::observer::NoopObserver;
            pipeline.run(stream, &config, &mut obs)
        };

        let result_a = run_with_spec(&original);
        let result_b = run_with_spec(&round_tripped);

        assert_eq!(result_a.phrases.len(), result_b.phrases.len());
        assert_eq!(result_a.converged, result_b.converged);
        assert_eq!(result_a.iterations, result_b.iterations);
        for (a, b) in result_a.phrases.iter().zip(&result_b.phrases) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.lemma, b.lemma);
            assert!(
                (a.score - b.score).abs() < 1e-12,
                "scores differ for '{}': {} vs {}",
                a.text,
                a.score,
                b.score
            );
            assert_eq!(a.rank, b.rank);
        }
    }

    /// Explicit module specs override preset defaults correctly.
    #[test]
    fn test_module_override_takes_precedence_over_preset() {
        use crate::pipeline::spec::{GraphSpec, ModuleSet, RuntimeSpec};

        // Start from single_rank preset (cross_sentence=true, no window_size)
        // Override with window_size=5
        let spec = PipelineSpec::V1(Box::new(PipelineSpecV1 {
            v: 1,
            preset: Some("single_rank".into()),
            modules: ModuleSet {
                graph: Some(GraphSpec::CooccurrenceWindow {
                    window_size: Some(5),
                    cross_sentence: None, // should inherit true from preset
                    edge_weighting: None,
                }),
                ..Default::default()
            },
            runtime: RuntimeSpec::default(),
            expose: None,
            strict: false,
            unknown_fields: std::collections::HashMap::new(),
        }));

        let resolved = resolve_spec(&spec).unwrap();

        // Verify deep merge: user's window_size=5 + preset's cross_sentence=true
        match &resolved.modules.graph {
            Some(GraphSpec::CooccurrenceWindow {
                window_size,
                cross_sentence,
                ..
            }) => {
                assert_eq!(*window_size, Some(5), "user override lost");
                assert_eq!(*cross_sentence, Some(true), "preset default lost");
            }
            other => panic!("expected CooccurrenceWindow, got {:?}", other),
        }

        // Should also build and run successfully
        let config = TextRankConfig {
            determinism: crate::types::DeterminismMode::Deterministic,
            ..TextRankConfig::default()
        };
        let tokens_json = pipeline_test_tokens_json();
        let tokens: Vec<Token> = serde_json::from_str::<Vec<JsonToken>>(tokens_json)
            .unwrap()
            .into_iter()
            .map(Token::from)
            .collect();
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

    /// Capability discovery modules are consistent with actual spec types:
    /// every module type listed can be embedded in a V1 spec and parsed.
    #[test]
    fn test_capabilities_modules_are_parseable_spec_types() {
        use crate::pipeline::spec::ModuleSet;

        let caps = build_capabilities();
        // For each stage→types pair, construct a JSON spec with that module
        // and verify it parses without error.
        let stage_json_templates: Vec<(&str, Vec<String>)> = vec![
            ("preprocess", caps.modules["preprocess"].clone()),
            ("candidates", caps.modules["candidates"].clone()),
            ("graph", caps.modules["graph"].clone()),
            ("teleport", caps.modules["teleport"].clone()),
            ("clustering", caps.modules["clustering"].clone()),
            ("rank", caps.modules["rank"].clone()),
            ("phrases", caps.modules["phrases"].clone()),
            ("format", caps.modules["format"].clone()),
        ];
        for (stage, types) in &stage_json_templates {
            for type_name in types {
                let json = format!(
                    r#"{{ "v": 1, "modules": {{ "{stage}": {{ "type": "{type_name}" }} }} }}"#,
                );
                let result: Result<PipelineSpecV1, _> = serde_json::from_str(&json);
                assert!(
                    result.is_ok(),
                    "capability module '{stage}.{type_name}' failed to parse: {:?}",
                    result.err()
                );
            }
        }

        // Also verify graph_transforms are parseable
        for type_name in &caps.modules["graph_transforms"] {
            let json = format!(
                r#"{{ "v": 1, "modules": {{ "graph_transforms": [{{ "type": "{type_name}" }}] }} }}"#,
            );
            let result: Result<PipelineSpecV1, _> = serde_json::from_str(&json);
            assert!(
                result.is_ok(),
                "capability transform '{type_name}' failed to parse: {:?}",
                result.err()
            );
        }
    }

    /// `capabilities` flag takes priority over `validate_only` when both are set.
    #[test]
    fn test_capabilities_takes_priority_over_validate_only() {
        let doc: JsonDocument =
            serde_json::from_str(r#"{"capabilities": true, "validate_only": true}"#).unwrap();
        assert!(doc.capabilities);
        assert!(doc.validate_only);
        // In extract_from_json dispatch, capabilities is checked first
        // so we should get a capabilities response, not a validation error
        // about missing pipeline field.
        let resp = build_capabilities();
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json.get("version").is_some());
        assert!(json.get("modules").is_some());
    }

    /// Deterministic execution produces identical results across runs.
    #[test]
    fn test_deterministic_pipeline_reproducibility() {
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "preset": "position_rank",
                    "modules": {{
                        "rank": {{ "type": "personalized_pagerank", "damping": 0.85 }}
                    }}
                }},
                "config": {{"determinism": "deterministic"}}
            }}"#,
            pipeline_test_tokens_json()
        );

        // Run twice and compare
        let run_once = || {
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
            pipeline.run(stream, &config, &mut obs)
        };

        let r1 = run_once();
        let r2 = run_once();

        assert_eq!(r1.phrases.len(), r2.phrases.len());
        assert_eq!(r1.iterations, r2.iterations);
        for (a, b) in r1.phrases.iter().zip(&r2.phrases) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.score, b.score, "non-deterministic score for '{}'", a.text);
        }
    }

    /// Invalid specs produce correct error diagnostics through validate_only.
    #[test]
    fn test_invalid_spec_errors_through_validate_only() {
        // Case 1: personalized_pagerank without teleport
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "modules": { "rank": { "type": "personalized_pagerank" } }
        }"#,
        )
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(!resp.valid);
        let json = serde_json::to_value(&resp).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        assert!(
            diags.iter().any(|d| d["code"] == "missing_stage"),
            "expected missing_stage error"
        );

        // Case 2: topic_graph without clustering or phrase_candidates
        let spec: PipelineSpecV1 = serde_json::from_str(
            r#"{
            "v": 1,
            "modules": { "graph": { "type": "topic_graph" } }
        }"#,
        )
        .unwrap();
        let resp = validate_spec_impl(&spec);
        assert!(!resp.valid);
        let json = serde_json::to_value(&resp).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        assert!(
            diags.len() >= 2,
            "expected multiple errors for topic_graph without deps"
        );

        // Case 3: unknown preset through the resolve path
        let spec = PipelineSpec::Preset("nonexistent".into());
        let response = match resolve_spec(&spec) {
            Ok(resolved) => validate_spec_impl(&resolved),
            Err(err) => ValidationResponse {
                valid: false,
                report: crate::pipeline::validation::ValidationReport {
                    diagnostics: vec![crate::pipeline::validation::ValidationDiagnostic::error(
                        err,
                    )],
                },
            },
        };
        assert!(!response.valid);
    }

    /// Config `window_size` serves as fallback for `CooccurrenceWindow` modules
    /// that omit an explicit window_size parameter.
    #[test]
    fn test_config_window_size_is_fallback_for_module_parameter() {
        // Use a V1 spec with explicit CooccurrenceWindow but window_size=None.
        // The builder should use cfg.window_size as fallback.
        let run_with_config_ws = |ws: usize| {
            let json_input = format!(
                r#"{{
                    "tokens": {},
                    "pipeline": {{
                        "v": 1,
                        "modules": {{
                            "graph": {{ "type": "cooccurrence_window" }}
                        }}
                    }},
                    "config": {{
                        "window_size": {ws},
                        "determinism": "deterministic"
                    }}
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
            pipeline.run(stream, &config, &mut obs)
        };

        // window_size=1 only connects immediate neighbors;
        // window_size=10 connects nearly all tokens within a sentence.
        let r_small = run_with_config_ws(1);
        let r_large = run_with_config_ws(10);

        assert!(r_small.converged);
        assert!(r_large.converged);
        assert!(!r_small.phrases.is_empty());
        assert!(!r_large.phrases.is_empty());

        // Different window sizes → different graph edges → different scores
        let scores_a: Vec<f64> = r_small.phrases.iter().map(|p| p.score).collect();
        let scores_b: Vec<f64> = r_large.phrases.iter().map(|p| p.score).collect();
        assert_ne!(
            scores_a, scores_b,
            "different window sizes should produce different scores"
        );
    }

    /// Pipeline-level module override trumps config-level defaults.
    #[test]
    fn test_pipeline_module_overrides_config_window_size() {
        use crate::pipeline::spec::{GraphSpec, ModuleSet, RuntimeSpec};

        // config.window_size = 3, but pipeline spec sets window_size = 6
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{
                        "graph": {{ "type": "cooccurrence_window", "window_size": 6 }}
                    }}
                }},
                "config": {{
                    "window_size": 3,
                    "determinism": "deterministic"
                }}
            }}"#,
            pipeline_test_tokens_json()
        );

        // Also run with pipeline window_size=6 AND config window_size=6
        // to verify the pipeline spec is what determines the value
        let json_input_matching = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{
                        "graph": {{ "type": "cooccurrence_window", "window_size": 6 }}
                    }}
                }},
                "config": {{
                    "window_size": 6,
                    "determinism": "deterministic"
                }}
            }}"#,
            pipeline_test_tokens_json()
        );

        let run = |input: &str| {
            let doc: JsonDocument = serde_json::from_str(input).unwrap();
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
            pipeline.run(stream, &config, &mut obs)
        };

        let r_override = run(&json_input);
        let r_matching = run(&json_input_matching);

        // Pipeline spec window_size=6 should produce same results regardless of
        // config.window_size, because the spec-level module takes precedence.
        assert_eq!(r_override.phrases.len(), r_matching.phrases.len());
        for (a, b) in r_override.phrases.iter().zip(&r_matching.phrases) {
            assert_eq!(a.text, b.text);
            assert!((a.score - b.score).abs() < 1e-12,
                "pipeline module override should produce same result regardless of config: '{}' {} vs {}",
                a.text, a.score, b.score
            );
        }
    }

    // ─── JSONL streaming tests ──────────────────────────────────────────

    /// Helper: a minimal valid JsonDocument as a compact single-line JSON string
    /// (JSONL requires no embedded newlines).
    fn jsonl_test_line() -> String {
        // Parse the pretty-printed tokens, re-serialize compact
        let tokens: serde_json::Value = serde_json::from_str(pipeline_test_tokens_json()).unwrap();
        let compact_tokens = serde_json::to_string(&tokens).unwrap();
        format!(
            r#"{{"tokens": {}, "config": {{"determinism": "deterministic"}}}}"#,
            compact_tokens
        )
    }

    #[test]
    fn test_jsonl_single_line() {
        let input = jsonl_test_line();
        let output =
            process_single_doc(serde_json::from_str::<JsonDocument>(&input).unwrap()).unwrap();

        // Should be a single JSON object with "phrases"
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert!(parsed.get("phrases").is_some());
        assert!(parsed.get("converged").is_some());
    }

    #[test]
    fn test_jsonl_multiple_lines() {
        let line = jsonl_test_line();
        let input = format!("{line}\n{line}\n{line}");

        // Simulate extract_jsonl_from_json logic (without PyO3)
        let mut output = String::new();
        for raw_line in input.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let result_line = match serde_json::from_str::<JsonDocument>(trimmed) {
                Ok(doc) => process_single_doc(doc).unwrap_or_else(|e| serialize_doc_error(&e)),
                Err(e) => serialize_doc_error(&DocError::Other(format!("Invalid JSON: {e}"))),
            };
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&result_line);
        }

        let output_lines: Vec<&str> = output.lines().collect();
        assert_eq!(
            output_lines.len(),
            3,
            "expected 3 result lines, got {}",
            output_lines.len()
        );

        // Each line should parse as valid JSON with phrases
        for (i, ol) in output_lines.iter().enumerate() {
            let parsed: serde_json::Value = serde_json::from_str(ol)
                .unwrap_or_else(|e| panic!("line {i} is not valid JSON: {e}"));
            assert!(
                parsed.get("phrases").is_some(),
                "line {i} missing 'phrases'"
            );
        }
    }

    #[test]
    fn test_jsonl_error_line() {
        let good = jsonl_test_line();
        let bad = r#"{"this is not valid json"#;
        let input = format!("{good}\n{bad}\n{good}");

        let mut output = String::new();
        for raw_line in input.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let result_line = match serde_json::from_str::<JsonDocument>(trimmed) {
                Ok(doc) => process_single_doc(doc).unwrap_or_else(|e| serialize_doc_error(&e)),
                Err(e) => serialize_doc_error(&DocError::Other(format!("Invalid JSON: {e}"))),
            };
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&result_line);
        }

        let output_lines: Vec<&str> = output.lines().collect();
        assert_eq!(
            output_lines.len(),
            3,
            "expected 3 output lines (including error)"
        );

        // Line 0: good → phrases
        let l0: serde_json::Value = serde_json::from_str(output_lines[0]).unwrap();
        assert!(l0.get("phrases").is_some(), "line 0 should have phrases");

        // Line 1: bad → error
        let l1: serde_json::Value = serde_json::from_str(output_lines[1]).unwrap();
        assert!(l1.get("error").is_some(), "line 1 should have error");
        let err_msg = l1["error"].as_str().unwrap();
        assert!(
            err_msg.contains("Invalid JSON"),
            "error should mention Invalid JSON, got: {err_msg}"
        );

        // Line 2: good → phrases
        let l2: serde_json::Value = serde_json::from_str(output_lines[2]).unwrap();
        assert!(l2.get("phrases").is_some(), "line 2 should have phrases");
    }

    #[test]
    fn test_jsonl_blank_lines_skipped() {
        let line = jsonl_test_line();
        // Input with blank lines, whitespace-only lines, and trailing newline
        let input = format!("\n  \n{line}\n\n  \t  \n{line}\n");

        let mut output = String::new();
        for raw_line in input.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let result_line = match serde_json::from_str::<JsonDocument>(trimmed) {
                Ok(doc) => process_single_doc(doc).unwrap_or_else(|e| serialize_doc_error(&e)),
                Err(e) => serialize_doc_error(&DocError::Other(format!("Invalid JSON: {e}"))),
            };
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&result_line);
        }

        let output_lines: Vec<&str> = output.lines().collect();
        assert_eq!(
            output_lines.len(),
            2,
            "blank/whitespace lines should be skipped, expected 2 result lines, got {}",
            output_lines.len()
        );
    }

    // ─── Workspace-reusing process_single_doc_with_workspace tests ────

    #[test]
    fn test_process_single_doc_with_workspace_pipeline_path() {
        // Pipeline-path document with workspace should produce identical results
        // to pipeline-path without workspace.
        let json_input = format!(
            r#"{{"tokens": {}, "pipeline": "textrank", "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let doc_a: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let doc_b: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let mut ws = PipelineWorkspace::new();

        let result_without = process_single_doc_with_workspace(doc_a, None, false).unwrap();
        let result_with = process_single_doc_with_workspace(doc_b, Some(&mut ws), false).unwrap();
        assert_eq!(result_without, result_with);

        // Verify the result is valid
        let parsed: serde_json::Value = serde_json::from_str(&result_with).unwrap();
        assert!(parsed.get("phrases").is_some());
        assert!(parsed["converged"].as_bool().unwrap());
    }

    #[test]
    fn test_process_single_doc_with_workspace_variant_path() {
        // Legacy variant path with workspace=Some should still work fine
        // (workspace is simply ignored for variant dispatch).
        let json_input = format!(
            r#"{{"tokens": {}, "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let mut ws = PipelineWorkspace::new();

        let result = process_single_doc_with_workspace(doc, Some(&mut ws), false);
        assert!(result.is_ok());
        let parsed: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert!(parsed.get("phrases").is_some());
    }

    #[test]
    fn test_process_single_doc_with_workspace_none() {
        // Passing workspace=None should work identically to process_single_doc.
        let json_input = format!(
            r#"{{"tokens": {}, "pipeline": "textrank", "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let doc_a: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let doc_b: JsonDocument = serde_json::from_str(&json_input).unwrap();

        let result_a = process_single_doc(doc_a).unwrap();
        let result_b = process_single_doc_with_workspace(doc_b, None, false).unwrap();
        assert_eq!(result_a, result_b);
    }

    // ─── SentenceRank variant dispatch ──────────────────────────────

    #[cfg(feature = "sentence-rank")]
    #[test]
    fn test_sentence_rank_variant_dispatch() {
        let json_input = format!(
            r#"{{"tokens": {}, "variant": "sentence_rank", "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let result =
            process_single_doc(serde_json::from_str::<JsonDocument>(&json_input).unwrap()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed.get("phrases").is_some());
        assert!(parsed.get("converged").is_some());
    }

    #[cfg(feature = "sentence-rank")]
    #[test]
    fn test_sentence_rank_pipeline_preset() {
        let json_input = format!(
            r#"{{"tokens": {}, "pipeline": "sentence_rank", "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let result =
            process_single_doc(serde_json::from_str::<JsonDocument>(&json_input).unwrap()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed.get("phrases").is_some());
        assert!(parsed.get("converged").is_some());
    }

    // ─── Patch 1: Preset validation via resolve_spec ─────────────────

    #[test]
    fn test_validate_preset_string_valid() {
        // "textrank" is a valid preset — should resolve and validate
        let spec: PipelineSpec = serde_json::from_str(r#""textrank""#).unwrap();
        let resolved = resolve_spec(&spec).unwrap();
        let resp = validate_spec_impl(&resolved);
        assert!(resp.valid);
    }

    #[test]
    fn test_validate_preset_string_invalid() {
        // "nonexistent" is not a valid preset — resolve should fail
        let spec: PipelineSpec = serde_json::from_str(r#""nonexistent""#).unwrap();
        assert!(resolve_spec(&spec).is_err());
    }

    #[test]
    fn test_validate_only_with_preset_string() {
        // Full document validation path with a preset string
        let json_input = r#"{"validate_only": true, "pipeline": "textrank"}"#;
        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["valid"], true);
    }

    #[test]
    fn test_validate_only_with_invalid_preset() {
        let json_input = r#"{"validate_only": true, "pipeline": "bogus_preset"}"#;
        let doc: JsonDocument = serde_json::from_str(json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["valid"], false);
    }

    // ─── Patch 8: Serde aliases ──────────────────────────────────────

    #[test]
    fn test_serde_alias_pagerank() {
        // "pagerank" should be accepted as an alias for "standard_pagerank"
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{ "rank": {{ "type": "pagerank" }} }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed["phrases"].as_array().unwrap().len() > 0);
    }

    // ─── Patch 2+3: expose → debug_level + debug_top_k ──────────────

    #[test]
    fn test_expose_graph_stats_in_output() {
        // When expose.graph_stats=true, debug payload should appear
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "expose": {{ "graph_stats": true }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed.get("debug").is_some(), "debug key should be present");
        let debug = &parsed["debug"];
        assert!(
            debug.get("graph_stats").is_some(),
            "graph_stats should be in debug"
        );
    }

    #[test]
    fn test_expose_node_scores_top_k() {
        // expose.node_scores.top_k=3 should limit node scores
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "expose": {{ "node_scores": {{ "top_k": 3 }} }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let debug = &parsed["debug"];
        let node_scores = debug["node_scores"].as_array().unwrap();
        assert!(
            node_scores.len() <= 3,
            "node_scores should be capped at top_k=3"
        );
    }

    #[test]
    fn test_expose_top_k_clamped_by_runtime() {
        // expose.node_scores.top_k=100 but runtime.max_debug_top_k=2 → clamped to 2
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "expose": {{ "node_scores": {{ "top_k": 100 }} }},
                    "runtime": {{ "max_debug_top_k": 2 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let debug = &parsed["debug"];
        let node_scores = debug["node_scores"].as_array().unwrap();
        assert!(
            node_scores.len() <= 2,
            "node_scores should be clamped to max_debug_top_k=2"
        );
    }

    #[test]
    fn test_no_debug_without_expose() {
        // Without expose, debug key should be absent (backward compat)
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{ "v": 1, "modules": {{}} }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed.get("debug").is_none(),
            "debug key should be absent without expose"
        );
    }

    // ─── Patch 4: stage_timings ──────────────────────────────────────

    #[test]
    fn test_expose_stage_timings() {
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "expose": {{ "stage_timings": true }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let debug = &parsed["debug"];
        assert!(
            debug.get("stage_timings").is_some(),
            "stage_timings should be present"
        );
        let timings = debug["stage_timings"].as_array().unwrap();
        assert!(!timings.is_empty(), "should have at least one timing entry");
    }

    // ─── Patch 5: debug field in JsonResult ──────────────────────────

    #[test]
    fn test_json_result_debug_skip_serializing_if_none() {
        // JsonResult with debug=None should not include "debug" key
        let result = JsonResult {
            phrases: vec![],
            converged: true,
            iterations: 1,
            debug: None,
        };
        let json = serde_json::to_value(&result).unwrap();
        assert!(json.get("debug").is_none());
    }

    #[test]
    fn test_json_result_debug_serialized_when_some() {
        use crate::pipeline::artifacts::DebugPayload;
        let result = JsonResult {
            phrases: vec![],
            converged: true,
            iterations: 1,
            debug: Some(DebugPayload::default()),
        };
        let json = serde_json::to_value(&result).unwrap();
        assert!(json.get("debug").is_some());
    }

    // ─── Patch 6: StandardJsonWithDebug + debug_key ──────────────────

    #[test]
    fn test_format_standard_json_with_debug_custom_key() {
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{
                        "format": {{ "type": "standard_json_with_debug", "debug_key": "introspection" }}
                    }},
                    "expose": {{ "graph_stats": true }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        // Debug should be under "introspection", not "debug"
        assert!(
            parsed.get("debug").is_none(),
            "default 'debug' key should be absent"
        );
        assert!(
            parsed.get("introspection").is_some(),
            "custom key 'introspection' should be present"
        );
    }

    #[test]
    fn test_format_standard_json_with_debug_default_key() {
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{
                        "format": {{ "type": "standard_json_with_debug" }}
                    }},
                    "expose": {{ "graph_stats": true }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed.get("debug").is_some(),
            "'debug' key should be present with default"
        );
    }

    // ─── Patch 7: Runtime controls ───────────────────────────────────

    #[test]
    fn test_runtime_max_tokens_rejects() {
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "max_tokens": 3 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc);
        assert!(
            result.is_err(),
            "should reject input with more tokens than max_tokens"
        );
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("token count"),
            "error should mention token count: {}",
            msg
        );
        assert!(
            msg.contains("exceeds runtime limit"),
            "error should mention limit: {}",
            msg
        );
    }

    #[test]
    fn test_runtime_deterministic() {
        // Two runs with deterministic=true should produce identical output
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "deterministic": true }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc1: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let doc2: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result1 = process_single_doc(doc1).unwrap();
        let result2 = process_single_doc(doc2).unwrap();
        assert_eq!(
            result1, result2,
            "deterministic runs should produce identical output"
        );
    }

    #[test]
    fn test_runtime_max_nodes_rejects() {
        // max_nodes=1 with multi-node graph should error
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "max_nodes": 1 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc);
        assert!(result.is_err(), "should reject graph exceeding max_nodes");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("node count"),
            "error should mention node count: {}",
            msg
        );
    }

    #[test]
    fn test_runtime_max_edges_rejects() {
        // max_edges=1 should reject a graph with more than 1 edge
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "max_edges": 1 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc);
        assert!(result.is_err(), "should reject graph exceeding max_edges");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("edge count"),
            "error should mention edge count: {}",
            msg
        );
    }

    #[test]
    fn test_runtime_scoped_threading() {
        // Verify max_threads=1 doesn't crash (functional test)
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "max_threads": 1 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed["phrases"].as_array().unwrap().len() > 0);
    }

    // ─── Capabilities ────────────────────────────────────────────────

    #[test]
    fn test_capabilities_includes_standard_json_with_debug() {
        let caps = build_capabilities();
        let format_list = caps.modules.get("format").unwrap();
        assert!(
            format_list.contains(&"standard_json_with_debug".to_string()),
            "capabilities should list standard_json_with_debug format"
        );
    }

    // ─── Workspace path also uses new wiring ─────────────────────────

    #[test]
    fn test_workspace_path_uses_new_wiring() {
        // Verify process_single_doc_with_workspace also applies expose
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "expose": {{ "graph_stats": true }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let mut ws = crate::pipeline::artifacts::PipelineWorkspace::new();
        let result = process_single_doc_with_workspace(doc, Some(&mut ws), false).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed.get("debug").is_some(),
            "workspace path should also produce debug output"
        );
    }

    // ─── Structured errors (Issue 3) ────────────────────────────────

    #[test]
    fn test_structured_error_shape_max_tokens() {
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "max_tokens": 3 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc);
        match result.unwrap_err() {
            DocError::Pipeline(e) => {
                assert_eq!(e.code, ErrorCode::LimitExceeded);
                assert_eq!(e.stage, "preprocess");
                assert_eq!(e.path, "/runtime/max_tokens");
                assert!(e.message.contains("token count"));
                assert!(e.hint.is_some());
            }
            DocError::Other(s) => panic!("expected Pipeline error, got Other: {s}"),
        }
    }

    #[test]
    fn test_structured_error_shape_max_nodes() {
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "max_nodes": 1 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let result = process_single_doc(doc);
        match result.unwrap_err() {
            DocError::Pipeline(e) => {
                assert_eq!(e.code, ErrorCode::LimitExceeded);
                assert_eq!(e.stage, "graph");
                assert_eq!(e.path, "/runtime/max_nodes");
                assert!(e.message.contains("node count"));
                assert!(e.hint.is_some());
            }
            DocError::Other(s) => panic!("expected Pipeline error, got Other: {s}"),
        }
    }

    #[test]
    fn test_structured_error_serialization() {
        // Pipeline errors should serialize as {"error": {code, ...}, "error_message": "..."}
        let json_input = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "max_tokens": 3 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let doc: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let err = process_single_doc(doc).unwrap_err();
        let serialized = serialize_doc_error(&err);
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        // error should be an object with structured fields
        let error_obj = parsed.get("error").expect("should have 'error' key");
        assert!(error_obj.is_object(), "pipeline error should be an object");
        assert_eq!(error_obj["code"], "limit_exceeded");
        assert_eq!(error_obj["stage"], "preprocess");
        assert_eq!(error_obj["path"], "/runtime/max_tokens");
        assert!(error_obj["message"]
            .as_str()
            .unwrap()
            .contains("token count"));

        // error_message should be a flat string for backward compat
        let error_message = parsed
            .get("error_message")
            .expect("should have 'error_message'");
        assert!(error_message.is_string());
    }

    #[test]
    fn test_legacy_error_remains_string() {
        // Non-pipeline errors should serialize as {"error": "plain string"}
        let err = DocError::Other("something went wrong".to_string());
        let serialized = serialize_doc_error(&err);
        let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        let error_val = parsed.get("error").expect("should have 'error' key");
        assert!(
            error_val.is_string(),
            "legacy error should be a plain string"
        );
        assert_eq!(error_val.as_str().unwrap(), "something went wrong");
        assert!(
            parsed.get("error_message").is_none(),
            "legacy error should not have error_message"
        );
    }

    // ─── Batch parity (Issue 4) ─────────────────────────────────────

    #[test]
    fn test_batch_with_pipeline_spec() {
        // Batch array with pipeline docs should work (previously only legacy was supported)
        let doc_json = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{ "v": 1, "modules": {{}} }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let batch_input = format!("[{doc_json},{doc_json}]");
        let docs: Vec<JsonDocument> = serde_json::from_str(&batch_input).unwrap();
        let mut ws = PipelineWorkspace::new();
        let mut output = String::from("[");
        for (i, doc) in docs.into_iter().enumerate() {
            if i > 0 {
                output.push(',');
            }
            match process_single_doc_with_workspace(doc, Some(&mut ws), false) {
                Ok(json) => output.push_str(&json),
                Err(e) => output.push_str(&serialize_doc_error(&e)),
            }
        }
        output.push(']');
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        for item in arr {
            assert!(
                item["phrases"].as_array().unwrap().len() > 0,
                "each doc should have phrases"
            );
        }
    }

    #[test]
    fn test_batch_mixed_legacy_and_pipeline() {
        let pipeline_doc = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{ "v": 1, "modules": {{}} }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let legacy_doc = format!(
            r#"{{
                "tokens": {},
                "variant": "textrank",
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let batch = format!("[{pipeline_doc},{legacy_doc}]");
        let docs: Vec<JsonDocument> = serde_json::from_str(&batch).unwrap();
        let mut ws = PipelineWorkspace::new();
        let mut output = String::from("[");
        for (i, doc) in docs.into_iter().enumerate() {
            if i > 0 {
                output.push(',');
            }
            match process_single_doc_with_workspace(doc, Some(&mut ws), false) {
                Ok(json) => output.push_str(&json),
                Err(e) => output.push_str(&serialize_doc_error(&e)),
            }
        }
        output.push(']');
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // Both should produce phrases (pipeline path and legacy path)
        for item in arr {
            assert!(item["phrases"].as_array().unwrap().len() > 0);
        }
    }

    #[test]
    fn test_batch_pipeline_with_debug() {
        let doc_json = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "expose": {{ "graph_stats": true }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let batch = format!("[{doc_json}]");
        let docs: Vec<JsonDocument> = serde_json::from_str(&batch).unwrap();
        let mut ws = PipelineWorkspace::new();
        let mut output = String::from("[");
        for (i, doc) in docs.into_iter().enumerate() {
            if i > 0 {
                output.push(',');
            }
            match process_single_doc_with_workspace(doc, Some(&mut ws), false) {
                Ok(json) => output.push_str(&json),
                Err(e) => output.push_str(&serialize_doc_error(&e)),
            }
        }
        output.push(']');
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let item = &parsed.as_array().unwrap()[0];
        assert!(
            item.get("debug").is_some(),
            "batch pipeline doc with expose should have debug"
        );
    }

    #[test]
    fn test_batch_pipeline_with_error() {
        // One doc should fail limit, the other should succeed
        let good_doc = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{ "v": 1, "modules": {{}} }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let bad_doc = format!(
            r#"{{
                "tokens": {},
                "pipeline": {{
                    "v": 1,
                    "modules": {{}},
                    "runtime": {{ "max_tokens": 1 }}
                }},
                "config": {{ "determinism": "deterministic" }}
            }}"#,
            pipeline_test_tokens_json()
        );
        let batch = format!("[{good_doc},{bad_doc}]");
        let docs: Vec<JsonDocument> = serde_json::from_str(&batch).unwrap();
        let mut ws = PipelineWorkspace::new();
        let mut output = String::from("[");
        for (i, doc) in docs.into_iter().enumerate() {
            if i > 0 {
                output.push(',');
            }
            match process_single_doc_with_workspace(doc, Some(&mut ws), false) {
                Ok(json) => output.push_str(&json),
                Err(e) => output.push_str(&serialize_doc_error(&e)),
            }
        }
        output.push(']');
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // First doc succeeds
        assert!(arr[0].get("phrases").is_some(), "first doc should succeed");
        // Second doc fails with structured error
        let error_obj = arr[1].get("error").expect("second doc should have error");
        assert!(error_obj.is_object(), "pipeline error should be structured");
        assert_eq!(error_obj["code"], "limit_exceeded");
    }

    #[test]
    fn test_jsonl_workspace_reuse() {
        // 3 JSONL lines through workspace path, all should succeed.
        // Must compact the tokens JSON to a single line for JSONL format.
        let tokens_compact: String = pipeline_test_tokens_json()
            .chars()
            .filter(|c| *c != '\n')
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ");
        let line = format!(
            r#"{{"tokens": {}, "pipeline": {{ "v": 1, "modules": {{}} }}, "config": {{ "determinism": "deterministic" }}}}"#,
            tokens_compact
        );
        // Verify the line is actually a single line
        assert!(!line.contains('\n'), "JSONL line must be a single line");
        let input = format!("{line}\n{line}\n{line}");

        let mut ws = PipelineWorkspace::new();
        let mut output = String::new();
        for raw_line in input.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let result_line = match serde_json::from_str::<JsonDocument>(trimmed) {
                Ok(doc) => match process_single_doc_with_workspace(doc, Some(&mut ws), false) {
                    Ok(json) => json,
                    Err(e) => serialize_doc_error(&e),
                },
                Err(e) => serialize_doc_error(&DocError::Other(format!("Invalid JSON: {e}"))),
            };
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&result_line);
        }

        let output_lines: Vec<&str> = output.lines().collect();
        assert_eq!(output_lines.len(), 3, "expected 3 result lines");
        for line_str in &output_lines {
            let parsed: serde_json::Value = serde_json::from_str(line_str).unwrap();
            assert!(
                parsed["phrases"].as_array().unwrap().len() > 0,
                "each workspace-reusing line should produce phrases"
            );
        }
    }

    // ─── Batch parallelism tests ──────────────────────────────────────

    #[test]
    fn test_build_batch_pool_zero_errors() {
        let err = build_batch_pool_inner(Some(0));
        assert!(err.is_err());
    }

    #[test]
    fn test_build_batch_pool_one_returns_sequential() {
        let pool = build_batch_pool_inner(Some(1)).unwrap();
        assert!(pool.is_sequential(), "Some(1) should be Sequential");
    }

    #[test]
    fn test_build_batch_pool_explicit_threads() {
        let pool = build_batch_pool_inner(Some(4)).unwrap();
        assert!(
            matches!(pool, BatchPool::Custom(_)),
            "Some(4) should return Custom pool"
        );
    }

    #[test]
    fn test_build_batch_pool_auto_uses_global() {
        let pool = build_batch_pool_inner(None).unwrap();
        assert!(
            matches!(pool, BatchPool::Global),
            "None should return Global"
        );
    }

    #[test]
    fn test_force_single_thread_produces_valid_results() {
        // Verify that force_single_thread=true still produces correct output
        // (exercises the full run_pipeline_from_spec path with the flag).
        let json_input = format!(
            r#"{{"tokens": {}, "pipeline": "textrank", "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let doc_normal: JsonDocument = serde_json::from_str(&json_input).unwrap();
        let doc_forced: JsonDocument = serde_json::from_str(&json_input).unwrap();

        let result_normal = process_single_doc_with_workspace(doc_normal, None, false).unwrap();
        let result_forced = process_single_doc_with_workspace(doc_forced, None, true).unwrap();

        // Both should produce identical results (deterministic mode)
        let v_normal: serde_json::Value = serde_json::from_str(&result_normal).unwrap();
        let v_forced: serde_json::Value = serde_json::from_str(&result_forced).unwrap();
        assert_eq!(v_normal, v_forced);
    }

    #[test]
    fn test_parallel_batch_produces_same_results_as_sequential() {
        let doc_json = format!(
            r#"{{"tokens": {}, "pipeline": "textrank", "config": {{"determinism": "deterministic"}}}}"#,
            pipeline_test_tokens_json()
        );
        let batch_input = format!("[{doc_json},{doc_json},{doc_json}]");
        let docs_seq: Vec<JsonDocument> = serde_json::from_str(&batch_input).unwrap();
        let docs_par: Vec<JsonDocument> = serde_json::from_str(&batch_input).unwrap();

        // Sequential
        let mut ws = PipelineWorkspace::new();
        let seq_results: Vec<String> = docs_seq
            .into_iter()
            .map(|doc| process_single_doc_with_workspace(doc, Some(&mut ws), false).unwrap())
            .collect();

        // Parallel (using a 2-thread custom pool)
        let pool = build_batch_pool_inner(Some(2)).unwrap();
        let par_results: Vec<String> = pool.install(|| {
            docs_par
                .into_par_iter()
                .map(|doc| {
                    let mut ws = PipelineWorkspace::new();
                    process_single_doc_with_workspace(doc, Some(&mut ws), true).unwrap()
                })
                .collect()
        });

        assert_eq!(seq_results.len(), par_results.len());
        for (s, p) in seq_results.iter().zip(par_results.iter()) {
            // Parse as JSON values so field order doesn't matter
            let sv: serde_json::Value = serde_json::from_str(s).unwrap();
            let pv: serde_json::Value = serde_json::from_str(p).unwrap();
            assert_eq!(
                sv, pv,
                "parallel and sequential should produce identical results"
            );
        }
    }
}
