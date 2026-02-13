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
}

impl CandidatesSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::WordNodes => "word_nodes",
            Self::PhraseCandidates => "phrase_candidates",
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
}

impl PhraseSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::ChunkPhrases { .. } => "chunk_phrases",
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
}

impl FormatSpec {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::StandardJson => "standard_json",
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
        assert_eq!(GraphSpec::TopicGraph.type_name(), "topic_graph");
        assert_eq!(GraphSpec::CandidateGraph.type_name(), "candidate_graph");
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
        assert_eq!(FormatSpec::StandardJson.type_name(), "standard_json");
    }
}
