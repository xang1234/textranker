//! Pipeline specification types.
//!
//! A [`PipelineSpec`] describes which modules to use for each pipeline stage,
//! runtime execution limits, and strictness settings. These types are the
//! input to the [`super::validation::ValidationEngine`].
//!
//! # JSON shape
//!
//! ```json
//! {
//!   "v": 1,
//!   "preset": "textrank",
//!   "modules": {
//!     "candidates": "word_nodes",
//!     "graph": "cooccurrence_window",
//!     "rank": "standard_pagerank"
//!   },
//!   "runtime": { "max_tokens": 200000 },
//!   "strict": false
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::artifacts::DebugLevel;

/// Top-level pipeline specification (v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSpec {
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
    pub candidates: Option<CandidateModuleType>,

    #[serde(default)]
    pub graph: Option<GraphModuleType>,

    #[serde(default)]
    pub graph_transforms: Vec<GraphTransformType>,

    #[serde(default)]
    pub teleport: Option<TeleportModuleType>,

    #[serde(default)]
    pub clustering: Option<ClusteringModuleType>,

    #[serde(default)]
    pub rank: Option<RankModuleType>,

    /// Captures any fields not recognized by the schema.
    #[serde(flatten)]
    pub unknown_fields: HashMap<String, serde_json::Value>,
}

// ─── Module type enums ──────────────────────────────────────────────────────

/// Candidate selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateModuleType {
    /// Individual word tokens as candidates (standard TextRank family).
    WordNodes,
    /// Noun-phrase chunks as candidates (TopicRank/MultipartiteRank family).
    PhraseCandidates,
}

/// Graph construction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphModuleType {
    /// Word co-occurrence within a sliding window.
    CooccurrenceWindow,
    /// Topic-level graph where nodes are phrase clusters (TopicRank).
    TopicGraph,
    /// Candidate-level graph with inter-cluster edges (MultipartiteRank).
    CandidateGraph,
}

impl GraphModuleType {
    /// Returns the user-facing name used in JSON and error messages.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CooccurrenceWindow => "cooccurrence_window",
            Self::TopicGraph => "topic_graph",
            Self::CandidateGraph => "candidate_graph",
        }
    }
}

/// Graph post-processing transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphTransformType {
    /// Remove edges between candidates in the same cluster.
    RemoveIntraClusterEdges,
    /// Apply alpha-boost weighting to first-occurring cluster members.
    AlphaBoost,
}

impl GraphTransformType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RemoveIntraClusterEdges => "remove_intra_cluster_edges",
            Self::AlphaBoost => "alpha_boost",
        }
    }
}

/// Teleport (personalization) strategy for PageRank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeleportModuleType {
    /// Uniform distribution (equivalent to no personalization).
    Uniform,
    /// Position-weighted: earlier tokens get higher teleport probability.
    Position,
    /// Focus-terms-biased: specified terms get boosted teleport probability.
    FocusTerms,
    /// Topic-weighted: per-lemma weights from external topic model.
    TopicWeights,
}

/// Clustering strategy for phrase candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusteringModuleType {
    /// Hierarchical agglomerative clustering with Jaccard distance.
    Hac,
}

/// PageRank variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankModuleType {
    /// Standard (unpersonalized) PageRank.
    StandardPagerank,
    /// Personalized PageRank with a teleport distribution.
    PersonalizedPagerank,
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

    #[test]
    fn test_deserialize_minimal_spec() {
        let json = r#"{ "v": 1 }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
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
                "candidates": "word_nodes",
                "graph": "cooccurrence_window",
                "rank": "personalized_pagerank",
                "teleport": "position"
            },
            "runtime": { "max_tokens": 100000 },
            "strict": true
        }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.preset.as_deref(), Some("textrank"));
        assert_eq!(spec.modules.rank, Some(RankModuleType::PersonalizedPagerank));
        assert_eq!(spec.modules.teleport, Some(TeleportModuleType::Position));
        assert_eq!(spec.runtime.max_tokens, Some(100000));
        assert!(spec.strict);
    }

    #[test]
    fn test_unknown_fields_captured() {
        let json = r#"{
            "v": 1,
            "bogus_top_level": 42,
            "modules": {
                "rank": "standard_pagerank",
                "bogus_module": "xyz"
            }
        }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert!(spec.unknown_fields.contains_key("bogus_top_level"));
        assert!(spec.modules.unknown_fields.contains_key("bogus_module"));
    }

    #[test]
    fn test_serde_roundtrip() {
        let json = r#"{"v":1,"modules":{"rank":"personalized_pagerank","teleport":"focus_terms"}}"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        let back = serde_json::to_value(&spec).unwrap();
        assert_eq!(back["modules"]["rank"], "personalized_pagerank");
        assert_eq!(back["modules"]["teleport"], "focus_terms");
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
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.runtime.max_threads, Some(4));
        assert!(!spec.runtime.single_thread);
    }

    #[test]
    fn test_deserialize_runtime_single_thread() {
        let json = r#"{
            "v": 1,
            "runtime": { "single_thread": true }
        }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert!(spec.runtime.single_thread);
        assert_eq!(spec.runtime.effective_threads(), Some(1));
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
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
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
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
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
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
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
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        let expose = spec.expose.unwrap();
        assert!(expose.is_enabled());
        assert_eq!(expose.to_debug_level(), DebugLevel::TopNodes);
        // node_scores with no top_k → default
        assert_eq!(expose.effective_top_k(), DebugLevel::DEFAULT_TOP_K);
    }

    #[test]
    fn test_deserialize_no_expose() {
        let json = r#"{ "v": 1 }"#;
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
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
        let spec: PipelineSpec = serde_json::from_str(json).unwrap();
        assert!(!spec.unknown_fields.contains_key("expose"));
        assert!(spec.expose.is_some());
    }
}
