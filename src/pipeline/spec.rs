//! Pipeline specification types.
//!
//! A [`PipelineSpec`] is either a preset string (e.g., `"textrank"`) or a full
//! V1 object ([`PipelineSpecV1`]) describing which modules to use for each
//! pipeline stage, runtime execution limits, and strictness settings.
//!
//! # JSON shapes
//!
//! **Preset:**
//! ```json
//! "textrank"
//! ```
//!
//! **Full V1:**
//! ```json
//! {
//!   "v": 1,
//!   "preset": "textrank",
//!   "modules": {
//!     "candidates": { "type": "word_nodes" },
//!     "graph": { "type": "cooccurrence_window", "window_size": 3 },
//!     "rank": { "type": "standard_pagerank" }
//!   },
//!   "runtime": { "max_tokens": 200000 },
//!   "strict": false
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::artifacts::DebugLevel;
use super::error_code::ErrorCode;
use super::errors::PipelineSpecError;

// ─── PipelineSpec (untagged enum) ──────────────────────────────────────────

/// Top-level pipeline specification.
///
/// Accepts either a preset string (e.g., `"textrank"`) or a full V1 object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PipelineSpec {
    /// A preset name resolved at execution time.
    Preset(String),
    /// A full V1 specification with explicit module selections.
    V1(PipelineSpecV1),
}

impl PipelineSpec {
    /// Returns a reference to the inner `PipelineSpecV1` if this is a `V1` variant.
    pub fn as_v1(&self) -> Option<&PipelineSpecV1> {
        match self {
            PipelineSpec::V1(v1) => Some(v1),
            _ => None,
        }
    }

    /// Consumes `self` and returns the inner `PipelineSpecV1` if this is a `V1` variant.
    pub fn into_v1(self) -> Option<PipelineSpecV1> {
        match self {
            PipelineSpec::V1(v1) => Some(v1),
            _ => None,
        }
    }

    /// Returns `true` if this is a `Preset` variant.
    pub fn is_preset(&self) -> bool {
        matches!(self, PipelineSpec::Preset(_))
    }
}

// ─── PipelineSpecV1 ────────────────────────────────────────────────────────

/// Full pipeline specification (v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSpecV1 {
    /// Spec version (currently `1`).
    pub v: u32,

    /// Optional preset name used as a starting point (e.g., `"textrank"`).
    #[serde(default)]
    pub preset: Option<String>,

    /// Explicit module selections. Omitted modules inherit from the preset.
    #[serde(default)]
    pub modules: ModuleSet,

    /// Runtime execution limits.
    #[serde(default)]
    pub runtime: RuntimeSpec,

    /// Debug / introspection requests.
    #[serde(default)]
    pub expose: Option<ExposeSpec>,

    /// If `true`, unrecognized fields are errors; if `false`, warnings.
    #[serde(default)]
    pub strict: bool,

    /// Captures any fields not recognized by the schema.
    /// Used by the strict-mode validation rule.
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, serde_json::Value>,
}

/// The set of modules selected for the pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModuleSet {
    #[serde(default)]
    pub preprocess: Option<PreprocessSpec>,

    #[serde(default)]
    pub candidates: Option<CandidatesSpec>,

    #[serde(default)]
    pub graph: Option<GraphSpec>,

    #[serde(default)]
    pub graph_transforms: Vec<GraphTransformSpec>,

    #[serde(default)]
    pub teleport: Option<TeleportSpec>,

    #[serde(default)]
    pub clustering: Option<ClusteringSpec>,

    #[serde(default)]
    pub rank: Option<RankSpec>,

    #[serde(default)]
    pub phrases: Option<PhraseSpec>,

    #[serde(default)]
    pub format: Option<FormatSpec>,

    /// Captures any fields not recognized by the schema.
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, serde_json::Value>,
}

// ─── Preset resolution ──────────────────────────────────────────────────────

/// Valid canonical preset names, used in error hints and capability discovery.
#[cfg(feature = "sentence-rank")]
pub const VALID_PRESETS: &[&str] = &[
    "textrank",
    "position_rank",
    "biased_textrank",
    "single_rank",
    "topical_pagerank",
    "topic_rank",
    "multipartite_rank",
    "sentence_rank",
];

/// Valid canonical preset names, used in error hints and capability discovery.
#[cfg(not(feature = "sentence-rank"))]
pub const VALID_PRESETS: &[&str] = &[
    "textrank",
    "position_rank",
    "biased_textrank",
    "single_rank",
    "topical_pagerank",
    "topic_rank",
    "multipartite_rank",
];

/// Resolve a preset name to its default [`ModuleSet`].
///
/// Accepts the same aliases as [`Variant::parse()`](crate::variants::Variant::parse),
/// but returns `Err(PipelineSpecError)` for unknown names instead of silently
/// falling back to `TextRank`.
///
/// # Examples
///
/// ```
/// # use textranker::pipeline::spec::resolve_preset;
/// let ms = resolve_preset("position_rank").unwrap();
/// assert!(ms.teleport.is_some());
/// ```
pub fn resolve_preset(name: &str) -> Result<ModuleSet, PipelineSpecError> {
    match name.to_lowercase().as_str() {
        // ── TextRank (base) ─────────────────────────────────────────
        "textrank" | "text_rank" | "base" => Ok(ModuleSet::default()),

        // ── PositionRank ────────────────────────────────────────────
        "position_rank" | "positionrank" | "position" => Ok(ModuleSet {
            teleport: Some(TeleportSpec::Position { shape: None }),
            ..Default::default()
        }),

        // ── BiasedTextRank ──────────────────────────────────────────
        "biased_textrank" | "biased" | "biasedtextrank" => Ok(ModuleSet {
            teleport: Some(TeleportSpec::FocusTerms),
            ..Default::default()
        }),

        // ── SingleRank ──────────────────────────────────────────────
        "single_rank" | "singlerank" | "single" => Ok(ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: None,
                cross_sentence: Some(true),
                edge_weighting: None,
            }),
            ..Default::default()
        }),

        // ── TopicalPageRank ─────────────────────────────────────────
        "topical_pagerank" | "topicalpagerank" | "single_tpr" | "tpr" => Ok(ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: None,
                cross_sentence: Some(true),
                edge_weighting: None,
            }),
            teleport: Some(TeleportSpec::TopicWeights),
            ..Default::default()
        }),

        // ── TopicRank ───────────────────────────────────────────────
        "topic_rank" | "topicrank" | "topic" => Ok(ModuleSet {
            candidates: Some(CandidatesSpec::PhraseCandidates),
            graph: Some(GraphSpec::TopicGraph),
            clustering: Some(ClusteringSpec::Hac { threshold: Some(0.25) }),
            ..Default::default()
        }),

        // ── MultipartiteRank ────────────────────────────────────────
        "multipartite_rank" | "multipartiterank" | "multipartite" | "mpr" => Ok(ModuleSet {
            candidates: Some(CandidatesSpec::PhraseCandidates),
            graph: Some(GraphSpec::CandidateGraph),
            clustering: Some(ClusteringSpec::Hac { threshold: None }),
            graph_transforms: vec![
                GraphTransformSpec::RemoveIntraClusterEdges,
                GraphTransformSpec::AlphaBoost,
            ],
            ..Default::default()
        }),

        // ── SentenceRank ────────────────────────────────────────────
        #[cfg(feature = "sentence-rank")]
        "sentence_rank" | "sentencerank" | "sentence" => Ok(ModuleSet {
            candidates: Some(CandidatesSpec::SentenceCandidates),
            graph: Some(GraphSpec::SentenceGraph { min_similarity: None }),
            phrases: Some(PhraseSpec::SentencePhrases),
            format: Some(FormatSpec::SentenceJson { sort_by_position: None }),
            ..Default::default()
        }),

        // ── Unknown ─────────────────────────────────────────────────
        _ => Err(PipelineSpecError::new(
            ErrorCode::InvalidValue,
            "/preset",
            format!("unknown preset name: '{name}'"),
        )
        .with_hint(format!("valid presets: {}", VALID_PRESETS.join(", ")))),
    }
}

// ─── Spec resolution ────────────────────────────────────────────────────────

/// Resolve a [`PipelineSpec`] to an effective [`PipelineSpecV1`] with preset
/// defaults merged into the module set.
///
/// This performs the first two steps of the pipeline build lifecycle:
///
/// 1. **Preset string** → resolve name to a full `PipelineSpecV1` with
///    preset-default modules (via [`resolve_preset()`]).
/// 2. **V1 with preset** → resolve preset, then merge user modules over
///    preset defaults (via [`merge_modules()`]).
/// 3. **V1 without preset** → return as-is (user modules are used directly).
///
/// After resolution, the returned spec is ready for validation and stage
/// construction.
///
/// # Errors
///
/// Returns `Err(PipelineSpecError)` if the preset name is unrecognized.
pub fn resolve_spec(spec: &PipelineSpec) -> Result<PipelineSpecV1, PipelineSpecError> {
    match spec {
        PipelineSpec::Preset(name) => {
            let modules = resolve_preset(name)?;
            Ok(PipelineSpecV1 {
                v: 1,
                preset: Some(name.clone()),
                modules,
                runtime: RuntimeSpec::default(),
                expose: None,
                strict: false,
                unknown_fields: HashMap::new(),
            })
        }
        PipelineSpec::V1(v1) => {
            if let Some(preset_name) = &v1.preset {
                let preset_modules = resolve_preset(preset_name)?;
                let merged = merge_modules(&v1.modules, &preset_modules);
                Ok(PipelineSpecV1 {
                    v: v1.v,
                    preset: v1.preset.clone(),
                    modules: merged,
                    runtime: v1.runtime.clone(),
                    expose: v1.expose.clone(),
                    strict: v1.strict,
                    unknown_fields: v1.unknown_fields.clone(),
                })
            } else {
                Ok(v1.clone())
            }
        }
    }
}

// ─── Module-set merging (config precedence) ────────────────────────────────

/// Merge two [`ModuleSet`]s with `user` fields taking precedence over `preset`.
///
/// Implements the config precedence model:
///
/// 1. **`user`** — explicit module selections from `pipeline.modules.*` (highest)
/// 2. **`preset`** — defaults from [`resolve_preset()`] (lowest)
///
/// For each `Option` field, `user.Some` wins; `None` falls through to preset.
/// For `graph_transforms`, a non-empty user vec wins; empty inherits the preset.
///
/// When both sides carry the **same module variant** (e.g., both
/// `CooccurrenceWindow`), optional parameters are deep-merged: user params
/// override preset params, and `None` inherits the preset's value.  When
/// variants differ, the user's variant wins entirely.
///
/// Level 2 of the precedence model — `TextRankConfig` defaults for `None`
/// parameters — is handled downstream by [`SpecPipelineBuilder::build()`]
/// (e.g., `window_size.unwrap_or(cfg.window_size)`).
pub fn merge_modules(user: &ModuleSet, preset: &ModuleSet) -> ModuleSet {
    // Helper: merge two Option<T> where T has a merge_with method.
    macro_rules! merge_opt {
        ($user:expr, $preset:expr, deep) => {
            match (&$user, &$preset) {
                (Some(u), Some(p)) => Some(u.merge_with(p)),
                (Some(u), None) => Some(u.clone()),
                (None, Some(p)) => Some(p.clone()),
                (None, None) => None,
            }
        };
        ($user:expr, $preset:expr) => {
            $user.clone().or_else(|| $preset.clone())
        };
    }

    let graph_transforms = if user.graph_transforms.is_empty() {
        preset.graph_transforms.clone()
    } else {
        user.graph_transforms.clone()
    };

    let mut unknown_fields = preset.unknown_fields.clone();
    unknown_fields.extend(user.unknown_fields.clone());

    ModuleSet {
        preprocess: merge_opt!(user.preprocess, preset.preprocess),
        candidates: merge_opt!(user.candidates, preset.candidates),
        graph: merge_opt!(user.graph, preset.graph, deep),
        graph_transforms,
        teleport: merge_opt!(user.teleport, preset.teleport, deep),
        clustering: merge_opt!(user.clustering, preset.clustering, deep),
        rank: merge_opt!(user.rank, preset.rank, deep),
        phrases: merge_opt!(user.phrases, preset.phrases, deep),
        format: merge_opt!(user.format, preset.format),
        unknown_fields,
    }
}

// ─── Module spec enums (internally tagged) ─────────────────────────────────

/// Preprocessing strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PreprocessSpec {
    /// Default preprocessing pipeline.
    Default,
}

impl PreprocessSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Default => "default",
        }
    }
}

/// Candidate selection strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CandidatesSpec {
    /// Individual word tokens as candidates (standard TextRank family).
    WordNodes,
    /// Noun-phrase chunks as candidates (TopicRank/MultipartiteRank family).
    PhraseCandidates,
    /// Whole sentences as candidates (SentenceRank / extractive summarization).
    #[cfg(feature = "sentence-rank")]
    SentenceCandidates,
}

impl CandidatesSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::WordNodes => "word_nodes",
            Self::PhraseCandidates => "phrase_candidates",
            #[cfg(feature = "sentence-rank")]
            Self::SentenceCandidates => "sentence_candidates",
        }
    }
}

/// Graph construction strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GraphSpec {
    /// Word co-occurrence within a sliding window.
    CooccurrenceWindow {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_size: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cross_sentence: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        edge_weighting: Option<EdgeWeightingSpec>,
    },
    /// Topic-level graph where nodes are phrase clusters (TopicRank).
    TopicGraph,
    /// Candidate-level graph with inter-cluster edges (MultipartiteRank).
    CandidateGraph,
    /// Sentence-level graph with Jaccard-similarity edges (SentenceRank).
    #[cfg(feature = "sentence-rank")]
    SentenceGraph {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min_similarity: Option<f64>,
    },
}

impl GraphSpec {
    /// Returns the user-facing name used in JSON and error messages.
    pub fn as_str(&self) -> &'static str {
        self.type_name()
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::CooccurrenceWindow { .. } => "cooccurrence_window",
            Self::TopicGraph => "topic_graph",
            Self::CandidateGraph => "candidate_graph",
            #[cfg(feature = "sentence-rank")]
            Self::SentenceGraph { .. } => "sentence_graph",
        }
    }

    /// Deep-merge optional parameters when both sides are the same variant.
    /// `self` (user) takes precedence; `fallback` (preset) fills `None` gaps.
    pub fn merge_with(&self, fallback: &Self) -> Self {
        match (self, fallback) {
            (
                Self::CooccurrenceWindow { window_size, cross_sentence, edge_weighting },
                Self::CooccurrenceWindow {
                    window_size: fb_ws,
                    cross_sentence: fb_cs,
                    edge_weighting: fb_ew,
                },
            ) => Self::CooccurrenceWindow {
                window_size: window_size.or(*fb_ws),
                cross_sentence: cross_sentence.or(*fb_cs),
                edge_weighting: edge_weighting.clone().or(fb_ew.clone()),
            },
            #[cfg(feature = "sentence-rank")]
            (
                Self::SentenceGraph { min_similarity },
                Self::SentenceGraph { min_similarity: fb_ms },
            ) => Self::SentenceGraph {
                min_similarity: min_similarity.or(*fb_ms),
            },
            // Different variants — user wins entirely.
            _ => self.clone(),
        }
    }
}

/// Edge weighting strategy for co-occurrence graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeWeightingSpec {
    Binary,
    Count,
}

/// Graph post-processing transforms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GraphTransformSpec {
    /// Remove edges between candidates in the same cluster.
    RemoveIntraClusterEdges,
    /// Apply alpha-boost weighting to first-occurring cluster members.
    AlphaBoost,
}

impl GraphTransformSpec {
    pub fn as_str(&self) -> &'static str {
        self.type_name()
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::RemoveIntraClusterEdges => "remove_intra_cluster_edges",
            Self::AlphaBoost => "alpha_boost",
        }
    }
}

/// Teleport (personalization) strategy for PageRank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TeleportSpec {
    /// Uniform distribution (equivalent to no personalization).
    Uniform,
    /// Position-weighted: earlier tokens get higher teleport probability.
    Position {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shape: Option<String>,
    },
    /// Focus-terms-biased: specified terms get boosted teleport probability.
    FocusTerms,
    /// Topic-weighted: per-lemma weights from external topic model.
    TopicWeights,
}

impl TeleportSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Position { .. } => "position",
            Self::FocusTerms => "focus_terms",
            Self::TopicWeights => "topic_weights",
        }
    }

    /// Deep-merge optional parameters when both sides are the same variant.
    pub fn merge_with(&self, fallback: &Self) -> Self {
        match (self, fallback) {
            (Self::Position { shape }, Self::Position { shape: fb_shape }) => Self::Position {
                shape: shape.clone().or_else(|| fb_shape.clone()),
            },
            _ => self.clone(),
        }
    }
}

/// Clustering strategy for phrase candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClusteringSpec {
    /// Hierarchical agglomerative clustering with Jaccard distance.
    Hac {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        threshold: Option<f64>,
    },
}

impl ClusteringSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Hac { .. } => "hac",
        }
    }

    /// Deep-merge optional parameters when both sides are the same variant.
    pub fn merge_with(&self, fallback: &Self) -> Self {
        match (self, fallback) {
            (Self::Hac { threshold }, Self::Hac { threshold: fb_th }) => Self::Hac {
                threshold: threshold.or(*fb_th),
            },
        }
    }
}

/// PageRank variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RankSpec {
    /// Standard (unpersonalized) PageRank.
    StandardPagerank,
    /// Personalized PageRank with a teleport distribution.
    PersonalizedPagerank {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        damping: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_iterations: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        convergence_threshold: Option<f64>,
    },
}

impl RankSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::StandardPagerank => "standard_pagerank",
            Self::PersonalizedPagerank { .. } => "personalized_pagerank",
        }
    }

    /// Deep-merge optional parameters when both sides are the same variant.
    pub fn merge_with(&self, fallback: &Self) -> Self {
        match (self, fallback) {
            (
                Self::PersonalizedPagerank { damping, max_iterations, convergence_threshold },
                Self::PersonalizedPagerank {
                    damping: fb_d,
                    max_iterations: fb_mi,
                    convergence_threshold: fb_ct,
                },
            ) => Self::PersonalizedPagerank {
                damping: damping.or(*fb_d),
                max_iterations: max_iterations.or(*fb_mi),
                convergence_threshold: convergence_threshold.or(*fb_ct),
            },
            _ => self.clone(),
        }
    }
}

/// Phrase assembly strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PhraseSpec {
    /// Chunk-based phrase assembly.
    ChunkPhrases {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min_phrase_length: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_phrase_length: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        score_aggregation: Option<ScoreAggregationSpec>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phrase_grouping: Option<PhraseGroupingSpec>,
    },
    /// Sentence-level phrase assembly (SentenceRank / extractive summarization).
    #[cfg(feature = "sentence-rank")]
    SentencePhrases,
}

impl PhraseSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::ChunkPhrases { .. } => "chunk_phrases",
            #[cfg(feature = "sentence-rank")]
            Self::SentencePhrases => "sentence_phrases",
        }
    }

    /// Deep-merge optional parameters when both sides are the same variant.
    pub fn merge_with(&self, fallback: &Self) -> Self {
        match (self, fallback) {
            (
                Self::ChunkPhrases {
                    min_phrase_length,
                    max_phrase_length,
                    score_aggregation,
                    phrase_grouping,
                },
                Self::ChunkPhrases {
                    min_phrase_length: fb_min,
                    max_phrase_length: fb_max,
                    score_aggregation: fb_sa,
                    phrase_grouping: fb_pg,
                },
            ) => Self::ChunkPhrases {
                min_phrase_length: min_phrase_length.or(*fb_min),
                max_phrase_length: max_phrase_length.or(*fb_max),
                score_aggregation: score_aggregation.or(*fb_sa),
                phrase_grouping: phrase_grouping.or(*fb_pg),
            },
            // Different variants or SentencePhrases — no deep merge needed.
            _ => self.clone(),
        }
    }
}

/// Score aggregation strategy for phrases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreAggregationSpec {
    Sum,
    Max,
    Mean,
}

/// Phrase grouping strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhraseGroupingSpec {
    ScrubbedText,
    Lemma,
}

/// Format / output strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FormatSpec {
    /// Standard JSON output format.
    StandardJson,
    /// Sentence-level JSON output with optional position-based sorting.
    #[cfg(feature = "sentence-rank")]
    SentenceJson {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sort_by_position: Option<bool>,
    },
}

impl FormatSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::StandardJson => "standard_json",
            #[cfg(feature = "sentence-rank")]
            Self::SentenceJson { .. } => "sentence_json",
        }
    }
}

// ─── Expose (debug/introspection) spec ────────────────────────────────────

/// Declarative specification for debug/introspection output.
///
/// Users include an `expose` section in the pipeline spec to request specific
/// intermediate artifacts in the response.  Each field maps to a subset of
/// [`DebugPayload`](super::DebugPayload) fields.
///
/// # JSON shape
///
/// ```json
/// "expose": {
///   "node_scores": { "top_k": 50 },
///   "graph_stats": true,
///   "pagerank": { "residuals": true },
///   "clusters": true,
///   "stage_timings": true
/// }
/// ```
///
/// An empty `expose: {}` enables nothing — each sub-field must be explicitly
/// set to request its data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExposeSpec {
    /// Request top-K ranked node scores in the debug output.
    #[serde(default)]
    pub node_scores: Option<NodeScoresSpec>,

    /// Request graph statistics (node/edge counts, transform status).
    #[serde(default)]
    pub graph_stats: bool,

    /// Request PageRank convergence data.
    #[serde(default)]
    pub pagerank: Option<PageRankExposeSpec>,

    /// Request cluster membership arrays (topic-family pipelines only).
    #[serde(default)]
    pub clusters: bool,

    /// Request per-stage timing measurements.
    #[serde(default)]
    pub stage_timings: bool,
}

/// Sub-spec for node score exposure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeScoresSpec {
    /// Maximum number of node scores to include.
    /// Defaults to [`DebugLevel::DEFAULT_TOP_K`] (50) when absent.
    #[serde(default)]
    pub top_k: Option<usize>,
}

/// Sub-spec for PageRank convergence data exposure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageRankExposeSpec {
    /// Include per-iteration residual deltas.
    #[serde(default)]
    pub residuals: bool,
}

impl ExposeSpec {
    /// Returns `true` if any debug output is requested.
    pub fn is_enabled(&self) -> bool {
        self.graph_stats
            || self.stage_timings
            || self.clusters
            || self.node_scores.is_some()
            || self.pagerank.is_some()
    }

    /// Map this declarative spec to the coarsest [`DebugLevel`] that satisfies
    /// all requested fields.
    ///
    /// The mapping follows the [`DebugPayload`](super::DebugPayload) population
    /// table:
    ///
    /// | Requested field   | Minimum `DebugLevel` |
    /// |-------------------|----------------------|
    /// | `graph_stats`     | `Stats`              |
    /// | `stage_timings`   | `Stats`              |
    /// | `node_scores`     | `TopNodes`           |
    /// | `pagerank.residuals` | `Full`            |
    /// | `clusters`        | `Full`               |
    pub fn to_debug_level(&self) -> DebugLevel {
        if !self.is_enabled() {
            return DebugLevel::None;
        }

        // Full: residuals or clusters
        if self.clusters {
            return DebugLevel::Full;
        }
        if let Some(ref pr) = self.pagerank {
            if pr.residuals {
                return DebugLevel::Full;
            }
        }

        // TopNodes: node_scores
        if self.node_scores.is_some() {
            return DebugLevel::TopNodes;
        }

        // Stats: graph_stats, stage_timings, or pagerank (without residuals)
        DebugLevel::Stats
    }

    /// Resolve the effective top-K for node scores.
    ///
    /// Priority order:
    /// 1. `expose.node_scores.top_k` (explicit in expose spec)
    /// 2. [`DebugLevel::DEFAULT_TOP_K`] (fallback)
    ///
    /// Callers that also have a [`RuntimeSpec`] should use
    /// [`RuntimeSpec::effective_debug_top_k`] as a secondary fallback.
    pub fn effective_top_k(&self) -> usize {
        self.node_scores
            .as_ref()
            .and_then(|ns| ns.top_k)
            .unwrap_or(DebugLevel::DEFAULT_TOP_K)
    }
}

// ─── Runtime spec ─────────────────────────────────────────────────────────

/// Runtime execution limits and threading controls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeSpec {
    /// Maximum number of input tokens before rejecting.
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Maximum number of graph nodes before rejecting.
    #[serde(default)]
    pub max_nodes: Option<usize>,

    /// Maximum number of graph edges before rejecting.
    #[serde(default)]
    pub max_edges: Option<usize>,

    /// Maximum number of Rayon threads for parallel work.
    /// `None` uses Rayon's default (all logical cores).
    #[serde(default)]
    pub max_threads: Option<usize>,

    /// Disable parallelism entirely (equivalent to `max_threads: 1`).
    /// When `true`, overrides `max_threads`.
    #[serde(default)]
    pub single_thread: bool,

    /// Maximum number of node scores to include in debug output.
    /// Defaults to [`DebugLevel::DEFAULT_TOP_K`] (50) when `None`.
    #[serde(default)]
    pub max_debug_top_k: Option<usize>,

    /// Request deterministic execution (reproducible output).
    #[serde(default)]
    pub deterministic: Option<bool>,

    /// Captures any fields not recognized by the schema.
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, serde_json::Value>,
}

impl RuntimeSpec {
    /// Resolve the effective thread count.
    ///
    /// - `single_thread == true` → `Some(1)`
    /// - `max_threads == Some(n)` → `Some(n)`
    /// - otherwise → `None` (use Rayon default)
    pub fn effective_threads(&self) -> Option<usize> {
        if self.single_thread {
            Some(1)
        } else {
            self.max_threads
        }
    }

    /// Build a scoped Rayon thread pool matching this config.
    ///
    /// Returns `None` when no thread limit is set (use global pool).
    pub fn build_thread_pool(&self) -> Option<rayon::ThreadPool> {
        self.effective_threads().map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("failed to build Rayon thread pool")
        })
    }

    /// Execute `f` within a scoped Rayon thread pool matching this config.
    ///
    /// If no thread limit is set, `f` runs directly (using the global pool).
    /// Otherwise, a custom pool is created and `f` runs inside
    /// [`rayon::ThreadPool::install`], so any `par_iter()` within `f`
    /// uses the scoped pool.
    pub fn scoped<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match self.build_thread_pool() {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// Resolve the effective debug top-K limit.
    ///
    /// Returns the explicit `max_debug_top_k` if set, otherwise
    /// [`DebugLevel::DEFAULT_TOP_K`] (50).
    pub fn effective_debug_top_k(&self) -> usize {
        self.max_debug_top_k
            .unwrap_or(crate::pipeline::artifacts::DebugLevel::DEFAULT_TOP_K)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PipelineSpec enum ────────────────────────────────────────────

    #[test]
    fn test_preset_string_parsing() {
        let json = r#""textrank""#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert!(spec.is_preset());
        assert!(spec.as_v1().is_none());
        match spec {
            PipelineSpec::Preset(name) => assert_eq!(name, "textrank"),
            _ => panic!("expected Preset"),
        }
    }

    #[test]
    fn test_v1_object_parsing() {
        let json = r#"{ "v": 1 }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert!(!spec.is_preset());
        let v1 = spec.as_v1().unwrap();
        assert_eq!(v1.v, 1);
    }

    #[test]
    fn test_into_v1() {
        let json = r#"{ "v": 1 }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        let v1 = spec.into_v1().unwrap();
        assert_eq!(v1.v, 1);
    }

    #[test]
    fn test_preset_into_v1_is_none() {
        let spec = PipelineSpec::Preset("textrank".into());
        assert!(spec.into_v1().is_none());
    }

    // ─── PipelineSpecV1 deserialization ───────────────────────────────

    #[test]
    fn test_deserialize_minimal_spec() {
        let json = r#"{ "v": 1 }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert_eq!(spec.v, 1);
        assert!(spec.modules.rank.is_none());
        assert!(!spec.strict);
    }

    #[test]
    fn test_deserialize_full_spec() {
        let json = r#"{
            "v": 1,
            "preset": "textrank",
            "modules": {
                "candidates": { "type": "word_nodes" },
                "graph": { "type": "cooccurrence_window" },
                "rank": { "type": "personalized_pagerank" },
                "teleport": { "type": "position" }
            },
            "runtime": { "max_tokens": 100000 },
            "strict": true
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert_eq!(spec.preset.as_deref(), Some("textrank"));
        assert!(matches!(spec.modules.rank, Some(RankSpec::PersonalizedPagerank { .. })));
        assert!(matches!(spec.modules.teleport, Some(TeleportSpec::Position { .. })));
        assert_eq!(spec.runtime.max_tokens, Some(100000));
        assert!(spec.strict);
    }

    #[test]
    fn test_deserialize_parameterized_modules() {
        let json = r#"{
            "v": 1,
            "modules": {
                "graph": { "type": "cooccurrence_window", "window_size": 5, "edge_weighting": "count" },
                "rank": { "type": "personalized_pagerank", "damping": 0.9, "max_iterations": 200 },
                "clustering": { "type": "hac", "threshold": 0.3 },
                "teleport": { "type": "position", "shape": "exponential" },
                "phrases": { "type": "chunk_phrases", "min_phrase_length": 2, "score_aggregation": "mean" }
            }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();

        match &spec.modules.graph {
            Some(GraphSpec::CooccurrenceWindow { window_size, edge_weighting, .. }) => {
                assert_eq!(*window_size, Some(5));
                assert_eq!(*edge_weighting, Some(EdgeWeightingSpec::Count));
            }
            other => panic!("expected CooccurrenceWindow, got {:?}", other),
        }

        match &spec.modules.rank {
            Some(RankSpec::PersonalizedPagerank { damping, max_iterations, .. }) => {
                assert_eq!(*damping, Some(0.9));
                assert_eq!(*max_iterations, Some(200));
            }
            other => panic!("expected PersonalizedPagerank, got {:?}", other),
        }

        match &spec.modules.clustering {
            Some(ClusteringSpec::Hac { threshold }) => {
                assert_eq!(*threshold, Some(0.3));
            }
            other => panic!("expected Hac, got {:?}", other),
        }

        match &spec.modules.teleport {
            Some(TeleportSpec::Position { shape }) => {
                assert_eq!(shape.as_deref(), Some("exponential"));
            }
            other => panic!("expected Position, got {:?}", other),
        }

        match &spec.modules.phrases {
            Some(PhraseSpec::ChunkPhrases { min_phrase_length, score_aggregation, .. }) => {
                assert_eq!(*min_phrase_length, Some(2));
                assert_eq!(*score_aggregation, Some(ScoreAggregationSpec::Mean));
            }
            other => panic!("expected ChunkPhrases, got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_fields_captured() {
        let json = r#"{
            "v": 1,
            "bogus_top_level": 42,
            "modules": {
                "rank": { "type": "standard_pagerank" },
                "bogus_module": "xyz"
            }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert!(spec.unknown_fields.contains_key("bogus_top_level"));
        assert!(spec.modules.unknown_fields.contains_key("bogus_module"));
    }

    #[test]
    fn test_serde_roundtrip() {
        let json = r#"{
            "v": 1,
            "modules": {
                "rank": { "type": "personalized_pagerank" },
                "teleport": { "type": "focus_terms" }
            }
        }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        let v1 = spec.as_v1().unwrap();
        let back = serde_json::to_value(v1).unwrap();
        assert_eq!(back["modules"]["rank"]["type"], "personalized_pagerank");
        assert_eq!(back["modules"]["teleport"]["type"], "focus_terms");
    }

    #[test]
    fn test_serde_roundtrip_with_params() {
        let json = r#"{
            "v": 1,
            "modules": {
                "graph": { "type": "cooccurrence_window", "window_size": 5 },
                "rank": { "type": "personalized_pagerank", "damping": 0.9 }
            }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        let back = serde_json::to_value(&spec).unwrap();
        assert_eq!(back["modules"]["graph"]["type"], "cooccurrence_window");
        assert_eq!(back["modules"]["graph"]["window_size"], 5);
        assert_eq!(back["modules"]["rank"]["type"], "personalized_pagerank");
        assert!((back["modules"]["rank"]["damping"].as_f64().unwrap() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_new_stages_deserialize() {
        let json = r#"{
            "v": 1,
            "modules": {
                "preprocess": { "type": "default" },
                "format": { "type": "standard_json" }
            }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert!(matches!(spec.modules.preprocess, Some(PreprocessSpec::Default)));
        assert!(matches!(spec.modules.format, Some(FormatSpec::StandardJson)));
    }

    // ─── RuntimeSpec threading ──────────────────────────────────────

    #[test]
    fn test_effective_threads_default() {
        let rt = RuntimeSpec::default();
        assert_eq!(rt.effective_threads(), None);
    }

    #[test]
    fn test_effective_threads_max_threads() {
        let rt = RuntimeSpec {
            max_threads: Some(4),
            ..Default::default()
        };
        assert_eq!(rt.effective_threads(), Some(4));
    }

    #[test]
    fn test_effective_threads_single_thread_overrides() {
        let rt = RuntimeSpec {
            max_threads: Some(8),
            single_thread: true,
            ..Default::default()
        };
        assert_eq!(rt.effective_threads(), Some(1));
    }

    #[test]
    fn test_build_thread_pool_none_when_default() {
        let rt = RuntimeSpec::default();
        assert!(rt.build_thread_pool().is_none());
    }

    #[test]
    fn test_build_thread_pool_some_when_configured() {
        let rt = RuntimeSpec {
            max_threads: Some(2),
            ..Default::default()
        };
        let pool = rt.build_thread_pool();
        assert!(pool.is_some());
        assert_eq!(pool.unwrap().current_num_threads(), 2);
    }

    #[test]
    fn test_build_thread_pool_single_thread() {
        let rt = RuntimeSpec {
            single_thread: true,
            ..Default::default()
        };
        let pool = rt.build_thread_pool();
        assert!(pool.is_some());
        assert_eq!(pool.unwrap().current_num_threads(), 1);
    }

    #[test]
    fn test_scoped_runs_in_pool() {
        let rt = RuntimeSpec {
            max_threads: Some(2),
            ..Default::default()
        };
        let thread_count = rt.scoped(|| rayon::current_num_threads());
        assert_eq!(thread_count, 2);
    }

    #[test]
    fn test_scoped_default_uses_global() {
        let rt = RuntimeSpec::default();
        // Should run without error; uses global pool
        let result = rt.scoped(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_deserialize_runtime_threading() {
        let json = r#"{
            "v": 1,
            "runtime": { "max_threads": 4, "single_thread": false }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert_eq!(spec.runtime.max_threads, Some(4));
        assert!(!spec.runtime.single_thread);
    }

    #[test]
    fn test_deserialize_runtime_single_thread() {
        let json = r#"{
            "v": 1,
            "runtime": { "single_thread": true }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert!(spec.runtime.single_thread);
        assert_eq!(spec.runtime.effective_threads(), Some(1));
    }

    #[test]
    fn test_deserialize_runtime_deterministic() {
        let json = r#"{
            "v": 1,
            "runtime": { "deterministic": true }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert_eq!(spec.runtime.deterministic, Some(true));
    }

    // ─── RuntimeSpec debug top-K ─────────────────────────────────────

    #[test]
    fn test_effective_debug_top_k_default() {
        let rt = RuntimeSpec::default();
        assert_eq!(
            rt.effective_debug_top_k(),
            crate::pipeline::artifacts::DebugLevel::DEFAULT_TOP_K
        );
    }

    #[test]
    fn test_effective_debug_top_k_explicit() {
        let rt = RuntimeSpec {
            max_debug_top_k: Some(25),
            ..Default::default()
        };
        assert_eq!(rt.effective_debug_top_k(), 25);
    }

    #[test]
    fn test_deserialize_runtime_max_debug_top_k() {
        let json = r#"{
            "v": 1,
            "runtime": { "max_debug_top_k": 100 }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert_eq!(spec.runtime.max_debug_top_k, Some(100));
        assert_eq!(spec.runtime.effective_debug_top_k(), 100);
    }

    // ─── ExposeSpec ─────────────────────────────────────────────────────

    #[test]
    fn test_expose_default_is_disabled() {
        let expose = ExposeSpec::default();
        assert!(!expose.is_enabled());
        assert_eq!(expose.to_debug_level(), DebugLevel::None);
    }

    #[test]
    fn test_expose_graph_stats_maps_to_stats() {
        let expose = ExposeSpec {
            graph_stats: true,
            ..Default::default()
        };
        assert!(expose.is_enabled());
        assert_eq!(expose.to_debug_level(), DebugLevel::Stats);
    }

    #[test]
    fn test_expose_stage_timings_maps_to_stats() {
        let expose = ExposeSpec {
            stage_timings: true,
            ..Default::default()
        };
        assert_eq!(expose.to_debug_level(), DebugLevel::Stats);
    }

    #[test]
    fn test_expose_pagerank_without_residuals_maps_to_stats() {
        let expose = ExposeSpec {
            pagerank: Some(PageRankExposeSpec { residuals: false }),
            ..Default::default()
        };
        assert!(expose.is_enabled());
        assert_eq!(expose.to_debug_level(), DebugLevel::Stats);
    }

    #[test]
    fn test_expose_node_scores_maps_to_top_nodes() {
        let expose = ExposeSpec {
            node_scores: Some(NodeScoresSpec { top_k: Some(25) }),
            ..Default::default()
        };
        assert_eq!(expose.to_debug_level(), DebugLevel::TopNodes);
    }

    #[test]
    fn test_expose_residuals_maps_to_full() {
        let expose = ExposeSpec {
            pagerank: Some(PageRankExposeSpec { residuals: true }),
            ..Default::default()
        };
        assert_eq!(expose.to_debug_level(), DebugLevel::Full);
    }

    #[test]
    fn test_expose_clusters_maps_to_full() {
        let expose = ExposeSpec {
            clusters: true,
            ..Default::default()
        };
        assert_eq!(expose.to_debug_level(), DebugLevel::Full);
    }

    #[test]
    fn test_expose_combined_uses_highest_level() {
        // node_scores (TopNodes) + clusters (Full) → Full wins
        let expose = ExposeSpec {
            node_scores: Some(NodeScoresSpec { top_k: None }),
            graph_stats: true,
            clusters: true,
            ..Default::default()
        };
        assert_eq!(expose.to_debug_level(), DebugLevel::Full);
    }

    #[test]
    fn test_expose_effective_top_k_default() {
        let expose = ExposeSpec::default();
        assert_eq!(expose.effective_top_k(), DebugLevel::DEFAULT_TOP_K);
    }

    #[test]
    fn test_expose_effective_top_k_explicit() {
        let expose = ExposeSpec {
            node_scores: Some(NodeScoresSpec { top_k: Some(10) }),
            ..Default::default()
        };
        assert_eq!(expose.effective_top_k(), 10);
    }

    #[test]
    fn test_expose_effective_top_k_none_in_spec() {
        let expose = ExposeSpec {
            node_scores: Some(NodeScoresSpec { top_k: None }),
            ..Default::default()
        };
        assert_eq!(expose.effective_top_k(), DebugLevel::DEFAULT_TOP_K);
    }

    #[test]
    fn test_deserialize_expose_empty() {
        let json = r#"{ "v": 1, "expose": {} }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        let expose = spec.expose.unwrap();
        assert!(!expose.is_enabled());
    }

    #[test]
    fn test_deserialize_expose_full_example() {
        let json = r#"{
            "v": 1,
            "expose": {
                "node_scores": { "top_k": 50 },
                "graph_stats": true,
                "pagerank": { "residuals": true },
                "clusters": true,
                "stage_timings": true
            }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        let expose = spec.expose.unwrap();
        assert!(expose.is_enabled());
        assert!(expose.graph_stats);
        assert!(expose.stage_timings);
        assert!(expose.clusters);
        assert_eq!(expose.node_scores.as_ref().unwrap().top_k, Some(50));
        assert!(expose.pagerank.as_ref().unwrap().residuals);
        assert_eq!(expose.to_debug_level(), DebugLevel::Full);
        assert_eq!(expose.effective_top_k(), 50);
    }

    #[test]
    fn test_deserialize_expose_partial() {
        let json = r#"{
            "v": 1,
            "expose": { "node_scores": {}, "graph_stats": true }
        }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        let expose = spec.expose.unwrap();
        assert!(expose.is_enabled());
        assert_eq!(expose.to_debug_level(), DebugLevel::TopNodes);
        // node_scores with no top_k → default
        assert_eq!(expose.effective_top_k(), DebugLevel::DEFAULT_TOP_K);
    }

    #[test]
    fn test_deserialize_no_expose() {
        let json = r#"{ "v": 1 }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert!(spec.expose.is_none());
    }

    #[test]
    fn test_expose_serde_roundtrip() {
        let expose = ExposeSpec {
            node_scores: Some(NodeScoresSpec { top_k: Some(25) }),
            graph_stats: true,
            pagerank: Some(PageRankExposeSpec { residuals: true }),
            clusters: false,
            stage_timings: true,
        };
        let json = serde_json::to_string(&expose).unwrap();
        let back: ExposeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.to_debug_level(), DebugLevel::Full);
        assert_eq!(back.effective_top_k(), 25);
        assert!(back.graph_stats);
        assert!(back.stage_timings);
        assert!(!back.clusters);
    }

    #[test]
    fn test_expose_not_captured_as_unknown_field() {
        let json = r#"{ "v": 1, "strict": true, "expose": { "graph_stats": true } }"#;
        let spec: PipelineSpecV1 = serde_json::from_str(json).unwrap();
        assert!(!spec.unknown_fields.contains_key("expose"));
        assert!(spec.expose.is_some());
    }

    // ─── type_name coverage ──────────────────────────────────────────

    #[test]
    fn test_type_names() {
        assert_eq!(PreprocessSpec::Default.type_name(), "default");
        assert_eq!(CandidatesSpec::WordNodes.type_name(), "word_nodes");
        assert_eq!(CandidatesSpec::PhraseCandidates.type_name(), "phrase_candidates");
        #[cfg(feature = "sentence-rank")]
        assert_eq!(CandidatesSpec::SentenceCandidates.type_name(), "sentence_candidates");
        assert_eq!(GraphSpec::TopicGraph.type_name(), "topic_graph");
        assert_eq!(GraphSpec::CandidateGraph.type_name(), "candidate_graph");
        #[cfg(feature = "sentence-rank")]
        assert_eq!(
            GraphSpec::SentenceGraph { min_similarity: None }.type_name(),
            "sentence_graph"
        );
        assert_eq!(
            GraphSpec::CooccurrenceWindow {
                window_size: None,
                cross_sentence: None,
                edge_weighting: None,
            }
            .type_name(),
            "cooccurrence_window"
        );
        assert_eq!(GraphTransformSpec::RemoveIntraClusterEdges.type_name(), "remove_intra_cluster_edges");
        assert_eq!(GraphTransformSpec::AlphaBoost.type_name(), "alpha_boost");
        assert_eq!(TeleportSpec::Uniform.type_name(), "uniform");
        assert_eq!(TeleportSpec::Position { shape: None }.type_name(), "position");
        assert_eq!(TeleportSpec::FocusTerms.type_name(), "focus_terms");
        assert_eq!(TeleportSpec::TopicWeights.type_name(), "topic_weights");
        assert_eq!(ClusteringSpec::Hac { threshold: None }.type_name(), "hac");
        assert_eq!(RankSpec::StandardPagerank.type_name(), "standard_pagerank");
        assert_eq!(
            RankSpec::PersonalizedPagerank {
                damping: None,
                max_iterations: None,
                convergence_threshold: None,
            }
            .type_name(),
            "personalized_pagerank"
        );
        assert_eq!(
            PhraseSpec::ChunkPhrases {
                min_phrase_length: None,
                max_phrase_length: None,
                score_aggregation: None,
                phrase_grouping: None,
            }
            .type_name(),
            "chunk_phrases"
        );
        #[cfg(feature = "sentence-rank")]
        assert_eq!(PhraseSpec::SentencePhrases.type_name(), "sentence_phrases");
        assert_eq!(FormatSpec::StandardJson.type_name(), "standard_json");
        #[cfg(feature = "sentence-rank")]
        assert_eq!(
            FormatSpec::SentenceJson { sort_by_position: None }.type_name(),
            "sentence_json"
        );
    }

    // ─── resolve_preset ───────────────────────────────────────────────

    #[test]
    fn test_resolve_preset_textrank() {
        let ms = resolve_preset("textrank").unwrap();
        assert!(ms.candidates.is_none());
        assert!(ms.graph.is_none());
        assert!(ms.teleport.is_none());
        assert!(ms.clustering.is_none());
        assert!(ms.graph_transforms.is_empty());
    }

    #[test]
    fn test_resolve_preset_position_rank() {
        let ms = resolve_preset("position_rank").unwrap();
        assert!(ms.candidates.is_none());
        assert!(ms.graph.is_none());
        assert!(matches!(ms.teleport, Some(TeleportSpec::Position { .. })));
        assert!(ms.clustering.is_none());
        assert!(ms.graph_transforms.is_empty());
    }

    #[test]
    fn test_resolve_preset_biased_textrank() {
        let ms = resolve_preset("biased_textrank").unwrap();
        assert!(ms.candidates.is_none());
        assert!(ms.graph.is_none());
        assert!(matches!(ms.teleport, Some(TeleportSpec::FocusTerms)));
        assert!(ms.clustering.is_none());
        assert!(ms.graph_transforms.is_empty());
    }

    #[test]
    fn test_resolve_preset_single_rank() {
        let ms = resolve_preset("single_rank").unwrap();
        assert!(ms.candidates.is_none());
        match &ms.graph {
            Some(GraphSpec::CooccurrenceWindow { cross_sentence, .. }) => {
                assert_eq!(*cross_sentence, Some(true));
            }
            other => panic!("expected CooccurrenceWindow, got {:?}", other),
        }
        assert!(ms.teleport.is_none());
        assert!(ms.clustering.is_none());
        assert!(ms.graph_transforms.is_empty());
    }

    #[test]
    fn test_resolve_preset_topical_pagerank() {
        let ms = resolve_preset("topical_pagerank").unwrap();
        assert!(ms.candidates.is_none());
        match &ms.graph {
            Some(GraphSpec::CooccurrenceWindow { cross_sentence, .. }) => {
                assert_eq!(*cross_sentence, Some(true));
            }
            other => panic!("expected CooccurrenceWindow, got {:?}", other),
        }
        assert!(matches!(ms.teleport, Some(TeleportSpec::TopicWeights)));
        assert!(ms.clustering.is_none());
        assert!(ms.graph_transforms.is_empty());
    }

    #[test]
    fn test_resolve_preset_topic_rank() {
        let ms = resolve_preset("topic_rank").unwrap();
        assert!(matches!(ms.candidates, Some(CandidatesSpec::PhraseCandidates)));
        assert!(matches!(ms.graph, Some(GraphSpec::TopicGraph)));
        match &ms.clustering {
            Some(ClusteringSpec::Hac { threshold }) => {
                assert_eq!(*threshold, Some(0.25));
            }
            other => panic!("expected Hac, got {:?}", other),
        }
        assert!(ms.teleport.is_none());
        assert!(ms.graph_transforms.is_empty());
    }

    #[test]
    fn test_resolve_preset_multipartite_rank() {
        let ms = resolve_preset("multipartite_rank").unwrap();
        assert!(matches!(ms.candidates, Some(CandidatesSpec::PhraseCandidates)));
        assert!(matches!(ms.graph, Some(GraphSpec::CandidateGraph)));
        assert!(ms.teleport.is_none());
        assert!(matches!(ms.clustering, Some(ClusteringSpec::Hac { threshold: None })));
        assert_eq!(ms.graph_transforms.len(), 2);
        assert!(matches!(ms.graph_transforms[0], GraphTransformSpec::RemoveIntraClusterEdges));
        assert!(matches!(ms.graph_transforms[1], GraphTransformSpec::AlphaBoost));
    }

    #[test]
    fn test_resolve_preset_aliases_match_canonical() {
        // Each alias should produce the same ModuleSet as its canonical name.
        let canonical = resolve_preset("textrank").unwrap();
        for alias in &["text_rank", "base"] {
            let aliased = resolve_preset(alias).unwrap();
            assert_eq!(
                format!("{:?}", canonical),
                format!("{:?}", aliased),
                "alias '{alias}' differs from canonical 'textrank'"
            );
        }

        let canonical = resolve_preset("position_rank").unwrap();
        for alias in &["positionrank", "position"] {
            let aliased = resolve_preset(alias).unwrap();
            assert_eq!(
                format!("{:?}", canonical),
                format!("{:?}", aliased),
                "alias '{alias}' differs from canonical 'position_rank'"
            );
        }

        let canonical = resolve_preset("topical_pagerank").unwrap();
        for alias in &["topicalpagerank", "single_tpr", "tpr"] {
            let aliased = resolve_preset(alias).unwrap();
            assert_eq!(
                format!("{:?}", canonical),
                format!("{:?}", aliased),
                "alias '{alias}' differs from canonical 'topical_pagerank'"
            );
        }

        let canonical = resolve_preset("multipartite_rank").unwrap();
        for alias in &["multipartiterank", "multipartite", "mpr"] {
            let aliased = resolve_preset(alias).unwrap();
            assert_eq!(
                format!("{:?}", canonical),
                format!("{:?}", aliased),
                "alias '{alias}' differs from canonical 'multipartite_rank'"
            );
        }
    }

    #[test]
    fn test_resolve_preset_case_insensitive() {
        let lower = resolve_preset("textrank").unwrap();
        let upper = resolve_preset("TEXTRANK").unwrap();
        let mixed = resolve_preset("TextRank").unwrap();
        let dbg_lower = format!("{:?}", lower);
        assert_eq!(dbg_lower, format!("{:?}", upper));
        assert_eq!(dbg_lower, format!("{:?}", mixed));
    }

    #[test]
    fn test_resolve_preset_unknown_returns_error() {
        let err = resolve_preset("nonexistent").unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidValue);
        assert_eq!(err.path, "/preset");
        assert!(err.message.contains("nonexistent"));
        assert!(err.hint.as_ref().unwrap().contains("textrank"));
        assert!(err.hint.as_ref().unwrap().contains("multipartite_rank"));
    }

    #[test]
    fn test_resolve_preset_empty_string_returns_error() {
        let err = resolve_preset("").unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidValue);
    }

    #[test]
    fn test_resolve_preset_all_fields_none_except_set() {
        // Verify that fields not set by a preset are None/empty/default.
        for name in VALID_PRESETS {
            let ms = resolve_preset(name).unwrap();
            assert!(ms.preprocess.is_none(), "preset '{name}' set preprocess");
            assert!(ms.rank.is_none(), "preset '{name}' set rank");
            // sentence_rank sets phrases and format (sentence-level stages)
            #[cfg(feature = "sentence-rank")]
            if *name == "sentence_rank" {
                // sentence_rank is expected to set phrases and format
            } else {
                assert!(ms.phrases.is_none(), "preset '{name}' set phrases");
                assert!(ms.format.is_none(), "preset '{name}' set format");
            }
            #[cfg(not(feature = "sentence-rank"))]
            {
                assert!(ms.phrases.is_none(), "preset '{name}' set phrases");
                assert!(ms.format.is_none(), "preset '{name}' set format");
            }
            assert!(ms.unknown_fields.is_empty(), "preset '{name}' has unknown fields");
        }
    }

    // ─── merge_modules ────────────────────────────────────────────────

    #[test]
    fn test_merge_user_overrides_preset() {
        // User sets teleport; preset has graph. Both should appear in result.
        let user = ModuleSet {
            teleport: Some(TeleportSpec::Position { shape: None }),
            ..Default::default()
        };
        let preset = ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: None,
                cross_sentence: Some(true),
                edge_weighting: None,
            }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        assert!(matches!(merged.teleport, Some(TeleportSpec::Position { .. })));
        assert!(matches!(merged.graph, Some(GraphSpec::CooccurrenceWindow { .. })));
    }

    #[test]
    fn test_merge_user_replaces_preset_field() {
        // Both set teleport — user wins.
        let user = ModuleSet {
            teleport: Some(TeleportSpec::FocusTerms),
            ..Default::default()
        };
        let preset = ModuleSet {
            teleport: Some(TeleportSpec::Position { shape: None }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        assert!(matches!(merged.teleport, Some(TeleportSpec::FocusTerms)));
    }

    #[test]
    fn test_merge_empty_user_inherits_preset() {
        let user = ModuleSet::default();
        let preset = resolve_preset("single_rank").unwrap();
        let merged = merge_modules(&user, &preset);
        // Should be identical to preset.
        assert_eq!(format!("{:?}", merged.graph), format!("{:?}", preset.graph));
    }

    #[test]
    fn test_merge_graph_deep_params() {
        // Preset: cross_sentence=true. User: window_size=5.
        // Merged: both params present.
        let user = ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: Some(5),
                cross_sentence: None,
                edge_weighting: None,
            }),
            ..Default::default()
        };
        let preset = ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: None,
                cross_sentence: Some(true),
                edge_weighting: None,
            }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        match &merged.graph {
            Some(GraphSpec::CooccurrenceWindow {
                window_size,
                cross_sentence,
                ..
            }) => {
                assert_eq!(*window_size, Some(5));
                assert_eq!(*cross_sentence, Some(true));
            }
            other => panic!("expected CooccurrenceWindow, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_graph_user_param_overrides_preset_param() {
        // Both set cross_sentence — user's value wins.
        let user = ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: None,
                cross_sentence: Some(false),
                edge_weighting: None,
            }),
            ..Default::default()
        };
        let preset = ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: Some(3),
                cross_sentence: Some(true),
                edge_weighting: None,
            }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        match &merged.graph {
            Some(GraphSpec::CooccurrenceWindow {
                window_size,
                cross_sentence,
                ..
            }) => {
                assert_eq!(*window_size, Some(3)); // from preset
                assert_eq!(*cross_sentence, Some(false)); // user overrides
            }
            other => panic!("expected CooccurrenceWindow, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_different_graph_variants_user_wins() {
        // User: TopicGraph. Preset: CooccurrenceWindow.
        // User wins entirely (no deep merge across different variants).
        let user = ModuleSet {
            graph: Some(GraphSpec::TopicGraph),
            ..Default::default()
        };
        let preset = ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: Some(3),
                cross_sentence: Some(true),
                edge_weighting: None,
            }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        assert!(matches!(merged.graph, Some(GraphSpec::TopicGraph)));
    }

    #[test]
    fn test_merge_graph_transforms_user_nonempty_wins() {
        let user = ModuleSet {
            graph_transforms: vec![GraphTransformSpec::AlphaBoost],
            ..Default::default()
        };
        let preset = ModuleSet {
            graph_transforms: vec![
                GraphTransformSpec::RemoveIntraClusterEdges,
                GraphTransformSpec::AlphaBoost,
            ],
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        assert_eq!(merged.graph_transforms.len(), 1);
        assert!(matches!(merged.graph_transforms[0], GraphTransformSpec::AlphaBoost));
    }

    #[test]
    fn test_merge_graph_transforms_user_empty_inherits() {
        let user = ModuleSet::default();
        let preset = ModuleSet {
            graph_transforms: vec![
                GraphTransformSpec::RemoveIntraClusterEdges,
                GraphTransformSpec::AlphaBoost,
            ],
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        assert_eq!(merged.graph_transforms.len(), 2);
    }

    #[test]
    fn test_merge_rank_deep_params() {
        let user = ModuleSet {
            rank: Some(RankSpec::PersonalizedPagerank {
                damping: Some(0.9),
                max_iterations: None,
                convergence_threshold: None,
            }),
            ..Default::default()
        };
        let preset = ModuleSet {
            rank: Some(RankSpec::PersonalizedPagerank {
                damping: None,
                max_iterations: Some(200),
                convergence_threshold: Some(1e-5),
            }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        match &merged.rank {
            Some(RankSpec::PersonalizedPagerank {
                damping,
                max_iterations,
                convergence_threshold,
            }) => {
                assert_eq!(*damping, Some(0.9));
                assert_eq!(*max_iterations, Some(200));
                assert_eq!(*convergence_threshold, Some(1e-5));
            }
            other => panic!("expected PersonalizedPagerank, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_clustering_deep_params() {
        let user = ModuleSet {
            clustering: Some(ClusteringSpec::Hac { threshold: Some(0.3) }),
            ..Default::default()
        };
        let preset = ModuleSet {
            clustering: Some(ClusteringSpec::Hac { threshold: Some(0.25) }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        match &merged.clustering {
            Some(ClusteringSpec::Hac { threshold }) => {
                assert_eq!(*threshold, Some(0.3)); // user wins
            }
            other => panic!("expected Hac, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_clustering_inherit_preset_threshold() {
        let user = ModuleSet {
            clustering: Some(ClusteringSpec::Hac { threshold: None }),
            ..Default::default()
        };
        let preset = ModuleSet {
            clustering: Some(ClusteringSpec::Hac { threshold: Some(0.25) }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        match &merged.clustering {
            Some(ClusteringSpec::Hac { threshold }) => {
                assert_eq!(*threshold, Some(0.25)); // from preset
            }
            other => panic!("expected Hac, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_phrases_deep_params() {
        let user = ModuleSet {
            phrases: Some(PhraseSpec::ChunkPhrases {
                min_phrase_length: Some(2),
                max_phrase_length: None,
                score_aggregation: None,
                phrase_grouping: None,
            }),
            ..Default::default()
        };
        let preset = ModuleSet {
            phrases: Some(PhraseSpec::ChunkPhrases {
                min_phrase_length: None,
                max_phrase_length: Some(5),
                score_aggregation: Some(ScoreAggregationSpec::Mean),
                phrase_grouping: Some(PhraseGroupingSpec::Lemma),
            }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        match &merged.phrases {
            Some(PhraseSpec::ChunkPhrases {
                min_phrase_length,
                max_phrase_length,
                score_aggregation,
                phrase_grouping,
            }) => {
                assert_eq!(*min_phrase_length, Some(2));
                assert_eq!(*max_phrase_length, Some(5));
                assert_eq!(*score_aggregation, Some(ScoreAggregationSpec::Mean));
                assert_eq!(*phrase_grouping, Some(PhraseGroupingSpec::Lemma));
            }
            other => panic!("expected ChunkPhrases, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_teleport_position_deep_params() {
        let user = ModuleSet {
            teleport: Some(TeleportSpec::Position { shape: None }),
            ..Default::default()
        };
        let preset = ModuleSet {
            teleport: Some(TeleportSpec::Position {
                shape: Some("exponential".to_string()),
            }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        match &merged.teleport {
            Some(TeleportSpec::Position { shape }) => {
                assert_eq!(shape.as_deref(), Some("exponential"));
            }
            other => panic!("expected Position, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_realistic_single_rank_with_overrides() {
        // Start from single_rank preset, override window_size.
        let preset = resolve_preset("single_rank").unwrap();
        let user = ModuleSet {
            graph: Some(GraphSpec::CooccurrenceWindow {
                window_size: Some(5),
                cross_sentence: None,
                edge_weighting: Some(EdgeWeightingSpec::Binary),
            }),
            ..Default::default()
        };
        let merged = merge_modules(&user, &preset);
        match &merged.graph {
            Some(GraphSpec::CooccurrenceWindow {
                window_size,
                cross_sentence,
                edge_weighting,
            }) => {
                assert_eq!(*window_size, Some(5)); // user
                assert_eq!(*cross_sentence, Some(true)); // preset
                assert_eq!(*edge_weighting, Some(EdgeWeightingSpec::Binary)); // user
            }
            other => panic!("expected CooccurrenceWindow, got {:?}", other),
        }
    }

    #[test]
    fn test_merge_both_empty_returns_empty() {
        let merged = merge_modules(&ModuleSet::default(), &ModuleSet::default());
        assert!(merged.candidates.is_none());
        assert!(merged.graph.is_none());
        assert!(merged.teleport.is_none());
        assert!(merged.clustering.is_none());
        assert!(merged.rank.is_none());
        assert!(merged.phrases.is_none());
        assert!(merged.format.is_none());
        assert!(merged.preprocess.is_none());
        assert!(merged.graph_transforms.is_empty());
    }

    // ─── resolve_spec ──────────────────────────────────────────────────

    #[test]
    fn test_resolve_spec_preset_string() {
        let spec = PipelineSpec::Preset("position_rank".into());
        let v1 = resolve_spec(&spec).unwrap();
        assert_eq!(v1.v, 1);
        assert_eq!(v1.preset.as_deref(), Some("position_rank"));
        assert!(matches!(v1.modules.teleport, Some(TeleportSpec::Position { .. })));
        assert!(v1.modules.graph.is_none());
    }

    #[test]
    fn test_resolve_spec_v1_with_preset_merges_modules() {
        // Start from single_rank preset (cross-sentence graph),
        // override with position teleport.
        let spec = PipelineSpec::V1(PipelineSpecV1 {
            v: 1,
            preset: Some("single_rank".into()),
            modules: ModuleSet {
                teleport: Some(TeleportSpec::Position { shape: None }),
                ..Default::default()
            },
            runtime: Default::default(),
            expose: None,
            strict: false,
            unknown_fields: HashMap::new(),
        });
        let v1 = resolve_spec(&spec).unwrap();
        // Teleport comes from user override
        assert!(matches!(v1.modules.teleport, Some(TeleportSpec::Position { .. })));
        // Graph comes from preset (single_rank → cross-sentence)
        match &v1.modules.graph {
            Some(GraphSpec::CooccurrenceWindow { cross_sentence, .. }) => {
                assert_eq!(*cross_sentence, Some(true));
            }
            other => panic!("expected CooccurrenceWindow from preset, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_spec_v1_without_preset_passes_through() {
        let original = PipelineSpecV1 {
            v: 1,
            preset: None,
            modules: ModuleSet {
                teleport: Some(TeleportSpec::FocusTerms),
                ..Default::default()
            },
            runtime: Default::default(),
            expose: None,
            strict: true,
            unknown_fields: HashMap::new(),
        };
        let spec = PipelineSpec::V1(original.clone());
        let v1 = resolve_spec(&spec).unwrap();
        assert!(v1.preset.is_none());
        assert!(matches!(v1.modules.teleport, Some(TeleportSpec::FocusTerms)));
        assert!(v1.strict);
    }

    #[test]
    fn test_resolve_spec_invalid_preset_string() {
        let spec = PipelineSpec::Preset("nonexistent".into());
        let err = resolve_spec(&spec).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidValue);
        assert!(err.message.contains("nonexistent"));
    }

    #[test]
    fn test_resolve_spec_v1_invalid_preset() {
        let spec = PipelineSpec::V1(PipelineSpecV1 {
            v: 1,
            preset: Some("bogus".into()),
            modules: Default::default(),
            runtime: Default::default(),
            expose: None,
            strict: false,
            unknown_fields: HashMap::new(),
        });
        let err = resolve_spec(&spec).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidValue);
    }

    #[test]
    fn test_resolve_spec_preserves_runtime_and_expose() {
        let spec = PipelineSpec::V1(PipelineSpecV1 {
            v: 1,
            preset: Some("textrank".into()),
            modules: Default::default(),
            runtime: RuntimeSpec {
                max_tokens: Some(50000),
                deterministic: Some(true),
                ..Default::default()
            },
            expose: Some(ExposeSpec {
                graph_stats: true,
                ..Default::default()
            }),
            strict: true,
            unknown_fields: HashMap::new(),
        });
        let v1 = resolve_spec(&spec).unwrap();
        assert_eq!(v1.runtime.max_tokens, Some(50000));
        assert_eq!(v1.runtime.deterministic, Some(true));
        assert!(v1.expose.as_ref().unwrap().graph_stats);
        assert!(v1.strict);
    }

    #[test]
    fn test_resolve_spec_v1_deep_merges_graph_params() {
        // Preset: single_rank → graph with cross_sentence=true
        // User: graph with window_size=5 (but no cross_sentence)
        // Result: window_size=5 + cross_sentence=true (deep merge)
        let spec = PipelineSpec::V1(PipelineSpecV1 {
            v: 1,
            preset: Some("single_rank".into()),
            modules: ModuleSet {
                graph: Some(GraphSpec::CooccurrenceWindow {
                    window_size: Some(5),
                    cross_sentence: None,
                    edge_weighting: None,
                }),
                ..Default::default()
            },
            runtime: Default::default(),
            expose: None,
            strict: false,
            unknown_fields: HashMap::new(),
        });
        let v1 = resolve_spec(&spec).unwrap();
        match &v1.modules.graph {
            Some(GraphSpec::CooccurrenceWindow { window_size, cross_sentence, .. }) => {
                assert_eq!(*window_size, Some(5)); // from user
                assert_eq!(*cross_sentence, Some(true)); // from preset
            }
            other => panic!("expected CooccurrenceWindow, got {:?}", other),
        }
    }

    // ─── resolve_preset sentence_rank ───────────────────────────────

    #[cfg(feature = "sentence-rank")]
    #[test]
    fn test_resolve_preset_sentence_rank() {
        let ms = resolve_preset("sentence_rank").unwrap();
        assert!(matches!(ms.candidates, Some(CandidatesSpec::SentenceCandidates)));
        assert!(matches!(ms.graph, Some(GraphSpec::SentenceGraph { min_similarity: None })));
        assert!(matches!(ms.phrases, Some(PhraseSpec::SentencePhrases)));
        assert!(matches!(ms.format, Some(FormatSpec::SentenceJson { sort_by_position: None })));
        // Not set by sentence_rank
        assert!(ms.teleport.is_none());
        assert!(ms.clustering.is_none());
        assert!(ms.graph_transforms.is_empty());
    }

    #[cfg(feature = "sentence-rank")]
    #[test]
    fn test_resolve_preset_sentence_rank_aliases() {
        let canonical = resolve_preset("sentence_rank").unwrap();
        for alias in &["sentencerank", "sentence"] {
            let aliased = resolve_preset(alias).unwrap();
            assert_eq!(
                format!("{:?}", canonical),
                format!("{:?}", aliased),
                "alias '{alias}' differs from canonical 'sentence_rank'"
            );
        }
    }
}
