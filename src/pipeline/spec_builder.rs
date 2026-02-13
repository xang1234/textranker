//! Spec-driven pipeline builder — maps a [`PipelineSpecV1`] to a runnable
//! [`DynPipeline`] using trait-object dispatch.
//!
//! The [`SpecPipelineBuilder`] is a runtime bridge between the declarative
//! JSON pipeline specification and the concrete stage implementations.
//! It constructs a [`DynPipeline`] (`Pipeline<Box<dyn Preprocessor>, ...>`)
//! whose stages are selected according to the spec's module set.
//!
//! # Usage
//!
//! ```ignore
//! let spec: PipelineSpecV1 = serde_json::from_str(json)?;
//! let cfg = TextRankConfig::default();
//! let pipeline = SpecPipelineBuilder::new()
//!     .with_chunks(chunks)
//!     .build(&spec, &cfg)?;
//! let result = pipeline.run(tokens, &cfg, &mut NoopObserver);
//! ```

use std::collections::HashMap;

use crate::pipeline::errors::PipelineSpecError;
use crate::pipeline::error_code::ErrorCode;
use crate::pipeline::runner::Pipeline;
use crate::pipeline::spec::{
    CandidatesSpec, ClusteringSpec, EdgeWeightingSpec, GraphSpec, GraphTransformSpec,
    PipelineSpec, PipelineSpecV1, TeleportSpec, resolve_spec,
};
use crate::pipeline::validation::ValidationEngine;
use crate::pipeline::traits::{
    CandidateGraphBuilder, CandidateSelector, ChunkPhraseBuilder, Clusterer,
    EdgeWeightPolicy, FocusTermsTeleportBuilder, GraphBuilder, GraphTransform,
    JaccardHacClusterer, MultipartitePhraseBuilder, MultipartiteTransform, NoopGraphTransform,
    NoopPreprocessor, PageRankRanker, PhraseBuilder, PhraseCandidateSelector,
    PositionTeleportBuilder, Preprocessor, Ranker, ResultFormatter,
    StandardResultFormatter, TeleportBuilder,
    TopicGraphBuilder, TopicRepresentativeBuilder, TopicWeightsTeleportBuilder,
    UniformTeleportBuilder, WindowGraphBuilder, WindowStrategy, WordNodeSelector,
};
#[cfg(feature = "sentence-rank")]
use crate::pipeline::traits::{
    SentenceCandidateSelector, SentenceFormatter, SentenceGraphBuilder, SentencePhraseBuilder,
};
use crate::pipeline::artifacts::{
    CandidateSetRef, Graph, TokenStreamRef,
};
use crate::types::{ChunkSpan, TextRankConfig};

// ─── DynPipeline type alias ────────────────────────────────────────────────

/// A pipeline whose stages are all trait objects — supports runtime composition.
///
/// This is the dynamic-dispatch counterpart of the static pipeline type aliases
/// (e.g., [`BaseTextRankPipeline`], [`PositionRankPipeline`]).  It can represent
/// any combination of stages, at the cost of virtual dispatch per stage call.
///
/// Construct via [`SpecPipelineBuilder::build`].
pub type DynPipeline = Pipeline<
    Box<dyn Preprocessor>,
    Box<dyn CandidateSelector>,
    Box<dyn GraphBuilder>,
    Box<dyn GraphTransform>,
    Box<dyn TeleportBuilder>,
    Box<dyn Ranker>,
    Box<dyn PhraseBuilder>,
    Box<dyn ResultFormatter>,
>;

// ─── ChainedGraphTransform (private) ───────────────────────────────────────

/// Applies multiple graph transforms in sequence.
///
/// Used when the spec's `graph_transforms` array has more than one entry.
struct ChainedGraphTransform {
    transforms: Vec<Box<dyn GraphTransform>>,
}

impl GraphTransform for ChainedGraphTransform {
    fn transform(
        &self,
        graph: &mut Graph,
        tokens: TokenStreamRef<'_>,
        candidates: CandidateSetRef<'_>,
        cfg: &TextRankConfig,
    ) {
        for t in &self.transforms {
            t.transform(graph, tokens, candidates, cfg);
        }
    }
}

// ─── SpecPipelineBuilder ───────────────────────────────────────────────────

/// Fluent builder that maps a [`PipelineSpecV1`] to a [`DynPipeline`].
///
/// Runtime context (data from the input document, not the spec) is supplied
/// via `with_*` methods before calling [`build`](Self::build).
pub struct SpecPipelineBuilder {
    /// Pre-computed noun-phrase chunks (required for PhraseCandidateSelector).
    chunks: Vec<ChunkSpan>,
    /// Focus terms for BiasedTextRank-style teleportation.
    focus_terms: Vec<String>,
    /// Bias weight for focus-term teleportation.
    bias_weight: f64,
    /// Per-lemma topic weights for TopicalPageRank-style teleportation.
    topic_weights: HashMap<String, f64>,
    /// Floor weight for topic-weight teleportation.
    topic_min_weight: f64,
}

impl SpecPipelineBuilder {
    /// Create a new builder with empty defaults.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            focus_terms: Vec::new(),
            bias_weight: 5.0,
            topic_weights: HashMap::new(),
            topic_min_weight: 0.01,
        }
    }

    /// Supply pre-computed noun-phrase chunks.
    ///
    /// Required when the spec uses `candidates: phrase_candidates`.
    pub fn with_chunks(mut self, chunks: Vec<ChunkSpan>) -> Self {
        self.chunks = chunks;
        self
    }

    /// Supply focus terms and bias weight for `teleport: focus_terms`.
    pub fn with_focus_terms(mut self, terms: Vec<String>, weight: f64) -> Self {
        self.focus_terms = terms;
        self.bias_weight = weight;
        self
    }

    /// Supply per-lemma topic weights for `teleport: topic_weights`.
    pub fn with_topic_weights(mut self, weights: HashMap<String, f64>, min_weight: f64) -> Self {
        self.topic_weights = weights;
        self.topic_min_weight = min_weight;
        self
    }

    /// Build a [`DynPipeline`] from the given spec and config.
    ///
    /// Returns `Err(PipelineSpecError)` if the spec requires runtime context
    /// that was not provided (e.g., `teleport: focus_terms` without calling
    /// `with_focus_terms`).
    pub fn build(
        &self,
        spec: &PipelineSpecV1,
        cfg: &TextRankConfig,
    ) -> Result<DynPipeline, PipelineSpecError> {
        let modules = &spec.modules;

        // ── Preprocessor ──────────────────────────────────────────────
        let preprocessor: Box<dyn Preprocessor> = Box::new(NoopPreprocessor);

        // ── Candidates ────────────────────────────────────────────────
        let selector: Box<dyn CandidateSelector> = match &modules.candidates {
            None | Some(CandidatesSpec::WordNodes) => Box::new(WordNodeSelector),
            Some(CandidatesSpec::PhraseCandidates) => {
                if self.chunks.is_empty() {
                    return Err(PipelineSpecError::new(
                        ErrorCode::InvalidValue,
                        "/modules/candidates",
                        "phrase_candidates requires non-empty chunks (call with_chunks())",
                    )
                    .with_hint("Supply chunks via SpecPipelineBuilder::with_chunks()"));
                }
                Box::new(PhraseCandidateSelector::new(self.chunks.clone()))
            }
            #[cfg(feature = "sentence-rank")]
            Some(CandidatesSpec::SentenceCandidates) => {
                Box::new(SentenceCandidateSelector)
            }
        };

        // ── Clustering (resolved early — graph builders may need it) ──
        let clustering_spec = modules.clustering.clone().or_else(|| {
            // Auto-infer clustering for topic-family graphs.
            match &modules.graph {
                Some(GraphSpec::TopicGraph) | Some(GraphSpec::CandidateGraph) => {
                    Some(ClusteringSpec::Hac { threshold: None })
                }
                _ => None,
            }
        });

        let make_clusterer = |spec: &ClusteringSpec| -> Box<dyn Clusterer> {
            match spec {
                ClusteringSpec::Hac { threshold } => {
                    Box::new(JaccardHacClusterer::new(threshold.unwrap_or(0.25)))
                }
            }
        };

        // ── Graph ─────────────────────────────────────────────────────
        let graph_builder: Box<dyn GraphBuilder> = match &modules.graph {
            None => {
                // Default: sentence-bounded, count-accumulating (base_textrank).
                Box::new(WindowGraphBuilder::base_textrank())
            }
            Some(GraphSpec::CooccurrenceWindow {
                window_size,
                cross_sentence,
                edge_weighting,
            }) => {
                let ws = window_size.unwrap_or(cfg.window_size);
                let strategy = if cross_sentence.unwrap_or(false) {
                    WindowStrategy::CrossSentence { window_size: ws }
                } else {
                    WindowStrategy::SentenceBounded { window_size: ws }
                };
                let policy = match edge_weighting {
                    Some(EdgeWeightingSpec::Binary) => EdgeWeightPolicy::Binary,
                    Some(EdgeWeightingSpec::Count) | None => EdgeWeightPolicy::CountAccumulating,
                };
                Box::new(WindowGraphBuilder {
                    window_strategy: strategy,
                    edge_weight_policy: policy,
                })
            }
            Some(GraphSpec::TopicGraph) => {
                let clust_spec = clustering_spec.as_ref().unwrap(); // guaranteed by auto-infer
                Box::new(TopicGraphBuilder::new(make_clusterer(clust_spec)))
            }
            Some(GraphSpec::CandidateGraph) => {
                let clust_spec = clustering_spec.as_ref().unwrap();
                Box::new(CandidateGraphBuilder::new(make_clusterer(clust_spec)))
            }
            #[cfg(feature = "sentence-rank")]
            Some(GraphSpec::SentenceGraph { min_similarity }) => {
                let mut b = SentenceGraphBuilder::default();
                if let Some(ms) = min_similarity {
                    b = b.with_min_similarity(*ms);
                }
                Box::new(b)
            }
        };

        // ── Graph Transforms ──────────────────────────────────────────
        let graph_transform: Box<dyn GraphTransform> = if modules.graph_transforms.is_empty() {
            Box::new(NoopGraphTransform)
        } else if modules.graph_transforms.len() == 1 {
            self.make_graph_transform(&modules.graph_transforms[0])
        } else {
            let transforms: Vec<Box<dyn GraphTransform>> = modules
                .graph_transforms
                .iter()
                .map(|spec| self.make_graph_transform(spec))
                .collect();
            Box::new(ChainedGraphTransform { transforms })
        };

        // ── Teleport ──────────────────────────────────────────────────
        let teleport_builder: Box<dyn TeleportBuilder> = match &modules.teleport {
            None | Some(TeleportSpec::Uniform) => Box::new(UniformTeleportBuilder),
            Some(TeleportSpec::Position { .. }) => Box::new(PositionTeleportBuilder),
            Some(TeleportSpec::FocusTerms) => {
                if self.focus_terms.is_empty() {
                    return Err(PipelineSpecError::new(
                        ErrorCode::InvalidValue,
                        "/modules/teleport",
                        "focus_terms teleport requires non-empty focus terms (call with_focus_terms())",
                    )
                    .with_hint("Supply focus terms via SpecPipelineBuilder::with_focus_terms()"));
                }
                Box::new(FocusTermsTeleportBuilder::new(
                    self.focus_terms.clone(),
                    self.bias_weight,
                ))
            }
            Some(TeleportSpec::TopicWeights) => {
                if self.topic_weights.is_empty() {
                    return Err(PipelineSpecError::new(
                        ErrorCode::InvalidValue,
                        "/modules/teleport",
                        "topic_weights teleport requires non-empty topic weights (call with_topic_weights())",
                    )
                    .with_hint("Supply topic weights via SpecPipelineBuilder::with_topic_weights()"));
                }
                Box::new(TopicWeightsTeleportBuilder::new(
                    self.topic_weights.clone(),
                    self.topic_min_weight,
                ))
            }
        };

        // ── Ranker ────────────────────────────────────────────────────
        let ranker: Box<dyn Ranker> = Box::new(PageRankRanker);

        // ── Phrases ───────────────────────────────────────────────────
        let phrase_builder: Box<dyn PhraseBuilder> = match &modules.phrases {
            Some(crate::pipeline::spec::PhraseSpec::ChunkPhrases { .. }) => {
                Box::new(ChunkPhraseBuilder)
            }
            #[cfg(feature = "sentence-rank")]
            Some(crate::pipeline::spec::PhraseSpec::SentencePhrases) => {
                Box::new(SentencePhraseBuilder)
            }
            None => {
                // Infer from graph type.
                match &modules.graph {
                    Some(GraphSpec::TopicGraph) => Box::new(TopicRepresentativeBuilder),
                    Some(GraphSpec::CandidateGraph) => Box::new(MultipartitePhraseBuilder),
                    #[cfg(feature = "sentence-rank")]
                    Some(GraphSpec::SentenceGraph { .. }) => Box::new(SentencePhraseBuilder),
                    _ => Box::new(ChunkPhraseBuilder),
                }
            }
        };

        // ── Format ────────────────────────────────────────────────────
        let formatter: Box<dyn ResultFormatter> = match &modules.format {
            #[cfg(feature = "sentence-rank")]
            Some(crate::pipeline::spec::FormatSpec::SentenceJson { sort_by_position }) => {
                Box::new(SentenceFormatter {
                    sort_by_position: sort_by_position.unwrap_or(false),
                })
            }
            _ => Box::new(StandardResultFormatter),
        };

        Ok(Pipeline {
            preprocessor,
            selector,
            graph_builder,
            graph_transform,
            teleport_builder,
            ranker,
            phrase_builder,
            formatter,
        })
    }

    /// Build a [`DynPipeline`] from a [`PipelineSpec`] — the main entry point
    /// for the JSON→Pipeline bridge.
    ///
    /// Orchestrates the full lifecycle:
    ///
    /// 1. **Resolve** — normalize the spec (preset string or V1 with preset)
    ///    to an effective [`PipelineSpecV1`] via [`resolve_spec()`].
    /// 2. **Merge** — user module overrides are merged over preset defaults
    ///    (handled inside `resolve_spec`).
    /// 3. **Validate** — run the default [`ValidationEngine`] rules against
    ///    the effective spec. Returns the first error if validation fails.
    /// 4. **Build** — construct stage implementations via [`build()`](Self::build).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let spec: PipelineSpec = serde_json::from_str(r#""position_rank""#)?;
    /// let cfg = TextRankConfig::default();
    /// let pipeline = SpecPipelineBuilder::new()
    ///     .build_from_spec(&spec, &cfg)?;
    /// ```
    pub fn build_from_spec(
        &self,
        spec: &PipelineSpec,
        cfg: &TextRankConfig,
    ) -> Result<DynPipeline, PipelineSpecError> {
        // 1-2. Resolve preset + merge modules.
        let effective = resolve_spec(spec)?;

        // 3. Validate.
        let report = ValidationEngine::with_defaults().validate(&effective);
        if let Some(err) = report.errors().next() {
            return Err(err.clone());
        }

        // 4. Build stage implementations.
        self.build(&effective, cfg)
    }

    /// Map a single `GraphTransformSpec` to a boxed impl.
    fn make_graph_transform(&self, spec: &GraphTransformSpec) -> Box<dyn GraphTransform> {
        match spec {
            GraphTransformSpec::RemoveIntraClusterEdges => {
                // IntraTopicEdgeRemover needs assignments at runtime, but
                // MultipartiteTransform reads them from the graph.
                // Use the combined transform for both cases.
                Box::new(MultipartiteTransform::with_alpha(0.0))
            }
            GraphTransformSpec::AlphaBoost => {
                // Alpha-boost only (no edge removal).
                // MultipartiteTransform with alpha > 0 also does edge removal,
                // so we use it standalone only when paired with removal in a chain.
                Box::new(MultipartiteTransform::new())
            }
        }
    }
}

impl Default for SpecPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::artifacts::TokenStream;
    use crate::pipeline::error_code::ErrorCode;
    use crate::pipeline::observer::NoopObserver;
    use crate::pipeline::runner::{BaseTextRankPipeline, SingleRankPipeline, TopicRankPipeline};
    use crate::pipeline::spec::{ModuleSet, PipelineSpec, PipelineSpecV1};
    use crate::types::{DeterminismMode, PosTag, Token};

    // ── Helpers ──────────────────────────────────────────────────────

    fn minimal_spec() -> PipelineSpecV1 {
        PipelineSpecV1 {
            v: 1,
            preset: None,
            modules: ModuleSet::default(),
            runtime: Default::default(),
            expose: None,
            strict: false,
            unknown_fields: HashMap::new(),
        }
    }

    fn deterministic_config() -> TextRankConfig {
        TextRankConfig {
            determinism: DeterminismMode::Deterministic,
            ..TextRankConfig::default()
        }
    }

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

    fn golden_chunks() -> Vec<ChunkSpan> {
        vec![
            ChunkSpan { start_token: 0, end_token: 2, start_char: 0, end_char: 16, sentence_idx: 0 },
            ChunkSpan { start_token: 3, end_token: 4, start_char: 22, end_char: 32, sentence_idx: 0 },
            ChunkSpan { start_token: 4, end_token: 6, start_char: 34, end_char: 47, sentence_idx: 1 },
            ChunkSpan { start_token: 7, end_token: 9, start_char: 53, end_char: 68, sentence_idx: 1 },
            ChunkSpan { start_token: 9, end_token: 12, start_char: 70, end_char: 93, sentence_idx: 2 },
            ChunkSpan { start_token: 14, end_token: 15, start_char: 107, end_char: 111, sentence_idx: 2 },
        ]
    }

    // ── Build validation tests ──────────────────────────────────────

    #[test]
    fn test_build_default_matches_base_textrank() {
        let spec = minimal_spec();
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new().build(&spec, &cfg).unwrap();
        // Should construct without error — DynPipeline is valid.
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let _result = pipeline.run(stream, &cfg, &mut NoopObserver);
    }

    #[test]
    fn test_build_position_rank() {
        let mut spec = minimal_spec();
        spec.modules.teleport = Some(TeleportSpec::Position { shape: None });
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new().build(&spec, &cfg).unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_single_rank() {
        let mut spec = minimal_spec();
        spec.modules.graph = Some(GraphSpec::CooccurrenceWindow {
            window_size: None,
            cross_sentence: Some(true),
            edge_weighting: Some(EdgeWeightingSpec::Count),
        });
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new().build(&spec, &cfg).unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_biased_textrank() {
        let mut spec = minimal_spec();
        spec.modules.teleport = Some(TeleportSpec::FocusTerms);
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .with_focus_terms(vec!["machine".to_string(), "learning".to_string()], 5.0)
            .build(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_topical_pagerank() {
        let mut spec = minimal_spec();
        spec.modules.graph = Some(GraphSpec::CooccurrenceWindow {
            window_size: None,
            cross_sentence: Some(true),
            edge_weighting: Some(EdgeWeightingSpec::Count),
        });
        spec.modules.teleport = Some(TeleportSpec::TopicWeights);
        let cfg = deterministic_config();
        let mut tw = HashMap::new();
        tw.insert("machine".to_string(), 2.0);
        tw.insert("network".to_string(), 1.5);
        let pipeline = SpecPipelineBuilder::new()
            .with_topic_weights(tw, 0.1)
            .build(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_topic_rank() {
        let mut spec = minimal_spec();
        spec.modules.candidates = Some(CandidatesSpec::PhraseCandidates);
        spec.modules.graph = Some(GraphSpec::TopicGraph);
        spec.modules.clustering = Some(ClusteringSpec::Hac { threshold: Some(0.25) });
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .with_chunks(golden_chunks())
            .build(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_multipartite_rank() {
        let mut spec = minimal_spec();
        spec.modules.candidates = Some(CandidatesSpec::PhraseCandidates);
        spec.modules.graph = Some(GraphSpec::CandidateGraph);
        spec.modules.graph_transforms = vec![
            GraphTransformSpec::RemoveIntraClusterEdges,
            GraphTransformSpec::AlphaBoost,
        ];
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .with_chunks(golden_chunks())
            .build(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_with_parameterized_graph() {
        let mut spec = minimal_spec();
        spec.modules.graph = Some(GraphSpec::CooccurrenceWindow {
            window_size: Some(5),
            cross_sentence: Some(false),
            edge_weighting: Some(EdgeWeightingSpec::Binary),
        });
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new().build(&spec, &cfg).unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_chained_transforms() {
        let mut spec = minimal_spec();
        spec.modules.candidates = Some(CandidatesSpec::PhraseCandidates);
        spec.modules.graph = Some(GraphSpec::CandidateGraph);
        spec.modules.graph_transforms = vec![
            GraphTransformSpec::RemoveIntraClusterEdges,
            GraphTransformSpec::AlphaBoost,
        ];
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .with_chunks(golden_chunks())
            .build(&spec, &cfg)
            .unwrap();
        // Verify the pipeline runs successfully with chained transforms.
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    // ── Error tests ─────────────────────────────────────────────────

    #[test]
    fn test_error_focus_terms_without_context() {
        let mut spec = minimal_spec();
        spec.modules.teleport = Some(TeleportSpec::FocusTerms);
        let cfg = TextRankConfig::default();
        let err = match SpecPipelineBuilder::new().build(&spec, &cfg) {
            Err(e) => e,
            Ok(_) => panic!("expected error for focus_terms without context"),
        };
        assert_eq!(err.code, ErrorCode::InvalidValue);
        assert!(err.message.contains("focus_terms"));
    }

    #[test]
    fn test_error_topic_weights_without_context() {
        let mut spec = minimal_spec();
        spec.modules.teleport = Some(TeleportSpec::TopicWeights);
        let cfg = TextRankConfig::default();
        let err = match SpecPipelineBuilder::new().build(&spec, &cfg) {
            Err(e) => e,
            Ok(_) => panic!("expected error for topic_weights without context"),
        };
        assert_eq!(err.code, ErrorCode::InvalidValue);
        assert!(err.message.contains("topic_weights"));
    }

    #[test]
    fn test_error_phrase_candidates_without_chunks() {
        let mut spec = minimal_spec();
        spec.modules.candidates = Some(CandidatesSpec::PhraseCandidates);
        let cfg = TextRankConfig::default();
        let err = match SpecPipelineBuilder::new().build(&spec, &cfg) {
            Err(e) => e,
            Ok(_) => panic!("expected error for phrase_candidates without chunks"),
        };
        assert_eq!(err.code, ErrorCode::InvalidValue);
        assert!(err.message.contains("chunks"));
    }

    // ── Golden tests (DynPipeline vs static pipeline bit-exact) ─────

    #[test]
    fn test_golden_dyn_matches_static_base() {
        let cfg = deterministic_config();
        let tokens = golden_tokens();

        // Static pipeline
        let static_pipeline = BaseTextRankPipeline::base_textrank();
        let stream1 = TokenStream::from_tokens(&tokens);
        let static_result = static_pipeline.run(stream1, &cfg, &mut NoopObserver);

        // DynPipeline via spec
        let spec = minimal_spec();
        let dyn_pipeline = SpecPipelineBuilder::new().build(&spec, &cfg).unwrap();
        let stream2 = TokenStream::from_tokens(&tokens);
        let dyn_result = dyn_pipeline.run(stream2, &cfg, &mut NoopObserver);

        assert_eq!(static_result.phrases.len(), dyn_result.phrases.len());
        for (s, d) in static_result.phrases.iter().zip(dyn_result.phrases.iter()) {
            assert_eq!(s.text, d.text, "phrase text mismatch");
            assert_eq!(s.lemma, d.lemma, "phrase lemma mismatch");
            assert!(
                (s.score - d.score).abs() < 1e-10,
                "score mismatch: {} vs {}",
                s.score,
                d.score
            );
            assert_eq!(s.rank, d.rank, "rank mismatch");
        }
    }

    #[test]
    fn test_golden_dyn_matches_static_single_rank() {
        let cfg = deterministic_config();
        let tokens = golden_tokens();

        // Static pipeline
        let static_pipeline = SingleRankPipeline::single_rank();
        let stream1 = TokenStream::from_tokens(&tokens);
        let static_result = static_pipeline.run(stream1, &cfg, &mut NoopObserver);

        // DynPipeline via spec
        let mut spec = minimal_spec();
        spec.modules.graph = Some(GraphSpec::CooccurrenceWindow {
            window_size: None,
            cross_sentence: Some(true),
            edge_weighting: Some(EdgeWeightingSpec::Count),
        });
        let dyn_pipeline = SpecPipelineBuilder::new().build(&spec, &cfg).unwrap();
        let stream2 = TokenStream::from_tokens(&tokens);
        let dyn_result = dyn_pipeline.run(stream2, &cfg, &mut NoopObserver);

        assert_eq!(static_result.phrases.len(), dyn_result.phrases.len());
        for (s, d) in static_result.phrases.iter().zip(dyn_result.phrases.iter()) {
            assert_eq!(s.text, d.text, "phrase text mismatch");
            assert!(
                (s.score - d.score).abs() < 1e-10,
                "score mismatch for '{}': {} vs {}",
                s.text,
                s.score,
                d.score
            );
        }
    }

    #[test]
    fn test_golden_dyn_matches_static_topic_rank() {
        let cfg = deterministic_config();
        let tokens = golden_tokens();
        let chunks = golden_chunks();

        // Static pipeline
        let static_pipeline = TopicRankPipeline::topic_rank(chunks.clone());
        let stream1 = TokenStream::from_tokens(&tokens);
        let static_result = static_pipeline.run(stream1, &cfg, &mut NoopObserver);

        // DynPipeline via spec
        let mut spec = minimal_spec();
        spec.modules.candidates = Some(CandidatesSpec::PhraseCandidates);
        spec.modules.graph = Some(GraphSpec::TopicGraph);
        spec.modules.clustering = Some(ClusteringSpec::Hac { threshold: Some(0.25) });
        let dyn_pipeline = SpecPipelineBuilder::new()
            .with_chunks(chunks)
            .build(&spec, &cfg)
            .unwrap();
        let stream2 = TokenStream::from_tokens(&tokens);
        let dyn_result = dyn_pipeline.run(stream2, &cfg, &mut NoopObserver);

        assert_eq!(
            static_result.phrases.len(),
            dyn_result.phrases.len(),
            "phrase count mismatch: static={}, dyn={}",
            static_result.phrases.len(),
            dyn_result.phrases.len()
        );
        for (s, d) in static_result.phrases.iter().zip(dyn_result.phrases.iter()) {
            assert_eq!(s.text, d.text, "phrase text mismatch");
            assert!(
                (s.score - d.score).abs() < 1e-10,
                "score mismatch for '{}': {} vs {}",
                s.text,
                s.score,
                d.score
            );
        }
    }

    // ── build_from_spec tests ─────────────────────────────────────────

    #[test]
    fn test_build_from_spec_preset_string() {
        let spec = PipelineSpec::Preset("textrank".into());
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .build_from_spec(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_from_spec_preset_position_rank() {
        let spec = PipelineSpec::Preset("position_rank".into());
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .build_from_spec(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_from_spec_v1_with_preset_and_overrides() {
        // Start from single_rank preset (cross-sentence graph),
        // override window_size to 5.
        let spec = PipelineSpec::V1(PipelineSpecV1 {
            v: 1,
            preset: Some("single_rank".into()),
            modules: ModuleSet {
                graph: Some(GraphSpec::CooccurrenceWindow {
                    window_size: Some(5),
                    cross_sentence: None, // inherits from preset
                    edge_weighting: None,
                }),
                ..Default::default()
            },
            runtime: Default::default(),
            expose: None,
            strict: false,
            unknown_fields: HashMap::new(),
        });
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .build_from_spec(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_from_spec_v1_without_preset() {
        // V1 without preset — modules used directly.
        let spec = PipelineSpec::V1(minimal_spec());
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .build_from_spec(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_from_spec_invalid_preset_returns_error() {
        let spec = PipelineSpec::Preset("nonexistent".into());
        let cfg = TextRankConfig::default();
        let err = match SpecPipelineBuilder::new().build_from_spec(&spec, &cfg) {
            Err(e) => e,
            Ok(_) => panic!("expected error for invalid preset"),
        };
        assert_eq!(err.code, ErrorCode::InvalidValue);
        assert!(err.message.contains("nonexistent"));
    }

    #[test]
    fn test_build_from_spec_validation_catches_invalid_combo() {
        // personalized_pagerank without teleport → validation error.
        let spec = PipelineSpec::V1(PipelineSpecV1 {
            v: 1,
            preset: None,
            modules: ModuleSet {
                rank: Some(crate::pipeline::spec::RankSpec::PersonalizedPagerank {
                    damping: None,
                    max_iterations: None,
                    convergence_threshold: None,
                }),
                ..Default::default()
            },
            runtime: Default::default(),
            expose: None,
            strict: false,
            unknown_fields: HashMap::new(),
        });
        let cfg = TextRankConfig::default();
        let err = match SpecPipelineBuilder::new().build_from_spec(&spec, &cfg) {
            Err(e) => e,
            Ok(_) => panic!("expected validation error for missing teleport"),
        };
        assert_eq!(err.code, ErrorCode::MissingStage);
        assert!(err.path.contains("teleport"));
    }

    #[test]
    fn test_build_from_spec_golden_preset_matches_direct_build() {
        // A preset "textrank" built via build_from_spec should produce
        // identical results to build() with a minimal_spec.
        let cfg = deterministic_config();
        let tokens = golden_tokens();

        // Via build_from_spec
        let spec = PipelineSpec::Preset("textrank".into());
        let pipeline_a = SpecPipelineBuilder::new()
            .build_from_spec(&spec, &cfg)
            .unwrap();
        let stream_a = TokenStream::from_tokens(&tokens);
        let result_a = pipeline_a.run(stream_a, &cfg, &mut NoopObserver);

        // Via build (direct PipelineSpecV1)
        let pipeline_b = SpecPipelineBuilder::new()
            .build(&minimal_spec(), &cfg)
            .unwrap();
        let stream_b = TokenStream::from_tokens(&tokens);
        let result_b = pipeline_b.run(stream_b, &cfg, &mut NoopObserver);

        assert_eq!(result_a.phrases.len(), result_b.phrases.len());
        for (a, b) in result_a.phrases.iter().zip(result_b.phrases.iter()) {
            assert_eq!(a.text, b.text, "phrase text mismatch");
            assert!(
                (a.score - b.score).abs() < 1e-10,
                "score mismatch for '{}': {} vs {}",
                a.text,
                a.score,
                b.score
            );
        }
    }

    #[test]
    fn test_build_from_spec_topic_rank_preset() {
        let spec = PipelineSpec::Preset("topic_rank".into());
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .with_chunks(golden_chunks())
            .build_from_spec(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }

    #[test]
    fn test_build_from_spec_multipartite_preset() {
        let spec = PipelineSpec::Preset("multipartite_rank".into());
        let cfg = deterministic_config();
        let pipeline = SpecPipelineBuilder::new()
            .with_chunks(golden_chunks())
            .build_from_spec(&spec, &cfg)
            .unwrap();
        let tokens = golden_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let result = pipeline.run(stream, &cfg, &mut NoopObserver);
        assert!(!result.phrases.is_empty());
    }
}
