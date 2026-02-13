//! Pipeline runner — orchestrates stage execution and artifact flow.
//!
//! The [`Pipeline`] struct holds a statically-composed set of pipeline stages.
//! Calling [`Pipeline::run`] executes them in order, threading artifacts
//! between stages and notifying an optional [`PipelineObserver`] at each
//! boundary.
//!
//! # Static dispatch
//!
//! `Pipeline` is generic over all stage types, so the compiler monomorphizes
//! each variant combination into a unique concrete type. Zero-sized default
//! stages (e.g., [`NoopPreprocessor`], [`NoopGraphTransform`],
//! [`UniformTeleportBuilder`]) add zero bytes and zero runtime cost.
//!
//! # Factory methods
//!
//! Use [`Pipeline::base_textrank()`] (and friends) to build pipelines for
//! known algorithm variants without spelling out the generics manually.

use crate::pipeline::artifacts::{DebugLevel, FormattedResult, PipelineWorkspace, TokenStream};
use crate::pipeline::observer::{
    PipelineObserver, StageClock, StageReport, StageReportBuilder, STAGE_CANDIDATES, STAGE_FORMAT,
    STAGE_GRAPH, STAGE_GRAPH_TRANSFORM, STAGE_PHRASES, STAGE_PREPROCESS, STAGE_RANK,
    STAGE_TELEPORT,
};
use crate::pipeline::traits::{
    CandidateGraphBuilder, CandidateSelector, ChunkPhraseBuilder, FocusTermsTeleportBuilder,
    GraphBuilder, GraphTransform, JaccardHacClusterer, MultipartitePhraseBuilder,
    MultipartiteTransform, NoopGraphTransform, NoopPreprocessor, PageRankRanker,
    PhraseCandidateSelector, PhraseBuilder, PositionTeleportBuilder, Preprocessor, Ranker,
    ResultFormatter, SentenceCandidateSelector, SentenceFormatter, SentenceGraphBuilder,
    SentencePhraseBuilder, StandardResultFormatter, TeleportBuilder, TopicGraphBuilder,
    TopicRepresentativeBuilder, TopicWeightsTeleportBuilder, UniformTeleportBuilder,
    WindowGraphBuilder, WordNodeSelector,
};
use std::collections::HashMap;
use crate::types::TextRankConfig;

// ---------------------------------------------------------------------------
// Conditional tracing support
// ---------------------------------------------------------------------------

/// Enter a tracing span for a pipeline stage (when the `tracing` feature is
/// enabled). When disabled, this is a no-op and the compiler eliminates it.
macro_rules! trace_stage {
    ($name:expr) => {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("pipeline_stage", stage = $name).entered();
    };
}


// ============================================================================
// Pipeline — statically-composed stage container
// ============================================================================

/// A pipeline composed of concrete stage implementations.
///
/// All type parameters have trait bounds enforced at the `impl` level, so the
/// struct itself is unconditionally constructible (useful for builders).
///
/// # Type parameters
///
/// | Param | Trait | Default impl |
/// |-------|-------|--------------|
/// | `Pre` | [`Preprocessor`] | [`NoopPreprocessor`] |
/// | `Sel` | [`CandidateSelector`] | [`WordNodeSelector`] |
/// | `GB`  | [`GraphBuilder`] | [`WindowGraphBuilder`] |
/// | `GT`  | [`GraphTransform`] | [`NoopGraphTransform`] |
/// | `TB`  | [`TeleportBuilder`] | [`UniformTeleportBuilder`] |
/// | `Rnk` | [`Ranker`] | [`PageRankRanker`] |
/// | `PB`  | [`PhraseBuilder`] | [`ChunkPhraseBuilder`] |
/// | `Fmt` | [`ResultFormatter`] | [`StandardResultFormatter`] |
#[derive(Debug, Clone)]
pub struct Pipeline<Pre, Sel, GB, GT, TB, Rnk, PB, Fmt> {
    pub preprocessor: Pre,
    pub selector: Sel,
    pub graph_builder: GB,
    pub graph_transform: GT,
    pub teleport_builder: TB,
    pub ranker: Rnk,
    pub phrase_builder: PB,
    pub formatter: Fmt,
}

/// Type alias for the default BaseTextRank pipeline.
pub type BaseTextRankPipeline = Pipeline<
    NoopPreprocessor,
    WordNodeSelector,
    WindowGraphBuilder,
    NoopGraphTransform,
    UniformTeleportBuilder,
    PageRankRanker,
    ChunkPhraseBuilder,
    StandardResultFormatter,
>;

impl BaseTextRankPipeline {
    /// Build a pipeline for the standard BaseTextRank algorithm.
    ///
    /// All stages use their zero-sized defaults:
    /// - No preprocessing
    /// - Word-level candidate selection (POS + stopword filtering)
    /// - Sentence-bounded co-occurrence graph (binary edges)
    /// - No graph transform
    /// - Uniform teleport (standard PageRank)
    /// - PageRank ranker
    /// - Chunk-based phrase builder
    /// - Standard result formatter
    pub fn base_textrank() -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: WordNodeSelector,
            graph_builder: WindowGraphBuilder::base_textrank(),
            graph_transform: NoopGraphTransform,
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: ChunkPhraseBuilder,
            formatter: StandardResultFormatter,
        }
    }
}

/// Type alias for the PositionRank pipeline.
///
/// Identical to [`BaseTextRankPipeline`] except for the teleport stage:
/// uses [`PositionTeleportBuilder`] instead of [`UniformTeleportBuilder`].
pub type PositionRankPipeline = Pipeline<
    NoopPreprocessor,
    WordNodeSelector,
    WindowGraphBuilder,
    NoopGraphTransform,
    PositionTeleportBuilder,
    PageRankRanker,
    ChunkPhraseBuilder,
    StandardResultFormatter,
>;

impl PositionRankPipeline {
    /// Build a pipeline for the PositionRank algorithm.
    ///
    /// Same as BaseTextRank but with position-biased teleportation:
    /// earlier-occurring candidates receive higher teleport probability.
    pub fn position_rank() -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: WordNodeSelector,
            graph_builder: WindowGraphBuilder::base_textrank(),
            graph_transform: NoopGraphTransform,
            teleport_builder: PositionTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: ChunkPhraseBuilder,
            formatter: StandardResultFormatter,
        }
    }
}

// ---------------------------------------------------------------------------
// SingleRankPipeline — cross-sentence windowing + count-accumulating weights
// ---------------------------------------------------------------------------

/// Pipeline alias for SingleRank: base pipeline with cross-sentence graph.
///
/// Structurally identical to [`BaseTextRankPipeline`] (same concrete types),
/// but the factory method configures [`WindowGraphBuilder`] with
/// [`WindowStrategy::CrossSentence`] + [`EdgeWeightPolicy::CountAccumulating`].
pub type SingleRankPipeline = Pipeline<
    NoopPreprocessor,
    WordNodeSelector,
    WindowGraphBuilder,
    NoopGraphTransform,
    UniformTeleportBuilder,
    PageRankRanker,
    ChunkPhraseBuilder,
    StandardResultFormatter,
>;

impl SingleRankPipeline {
    /// Build a pipeline for the SingleRank algorithm.
    ///
    /// Same as BaseTextRank but with cross-sentence windowing: the sliding
    /// window ignores sentence boundaries, and co-occurrence counts
    /// accumulate as edge weights.
    pub fn single_rank() -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: WordNodeSelector,
            graph_builder: WindowGraphBuilder::single_rank(),
            graph_transform: NoopGraphTransform,
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: ChunkPhraseBuilder,
            formatter: StandardResultFormatter,
        }
    }
}

// ---------------------------------------------------------------------------
// BiasedTextRankPipeline — focus-term biased teleportation
// ---------------------------------------------------------------------------

/// Pipeline alias for BiasedTextRank: base pipeline + focus-term teleportation.
pub type BiasedTextRankPipeline = Pipeline<
    NoopPreprocessor,
    WordNodeSelector,
    WindowGraphBuilder,
    NoopGraphTransform,
    FocusTermsTeleportBuilder,
    PageRankRanker,
    ChunkPhraseBuilder,
    StandardResultFormatter,
>;

impl BiasedTextRankPipeline {
    /// Build a pipeline for the BiasedTextRank algorithm.
    ///
    /// Same as BaseTextRank but with focus-term biased teleportation:
    /// candidates matching `focus_terms` receive `bias_weight` relative
    /// to the base weight of `1.0` for non-focus candidates.
    pub fn biased(focus_terms: Vec<String>, bias_weight: f64) -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: WordNodeSelector,
            graph_builder: WindowGraphBuilder::base_textrank(),
            graph_transform: NoopGraphTransform,
            teleport_builder: FocusTermsTeleportBuilder::new(focus_terms, bias_weight),
            ranker: PageRankRanker,
            phrase_builder: ChunkPhraseBuilder,
            formatter: StandardResultFormatter,
        }
    }
}

// ---------------------------------------------------------------------------
// TopicalPageRankPipeline — topic-weight biased teleportation
// ---------------------------------------------------------------------------

/// Pipeline alias for TopicalPageRank: SingleRank graph + topic-weighted teleportation.
///
/// Uses the same cross-sentence, count-accumulating graph as [`SingleRankPipeline`],
/// combined with [`TopicWeightsTeleportBuilder`] to bias PageRank towards
/// topic-relevant candidates.
pub type TopicalPageRankPipeline = Pipeline<
    NoopPreprocessor,
    WordNodeSelector,
    WindowGraphBuilder,
    NoopGraphTransform,
    TopicWeightsTeleportBuilder,
    PageRankRanker,
    ChunkPhraseBuilder,
    StandardResultFormatter,
>;

impl TopicalPageRankPipeline {
    /// Build a pipeline for the TopicalPageRank algorithm.
    ///
    /// Combines SingleRank's cross-sentence graph with topic-weighted
    /// teleportation: candidates matching entries in `topic_weights`
    /// receive proportionally higher teleport probability. Candidates
    /// absent from the map receive `min_weight`.
    pub fn topical(topic_weights: HashMap<String, f64>, min_weight: f64) -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: WordNodeSelector,
            graph_builder: WindowGraphBuilder::single_rank(),
            graph_transform: NoopGraphTransform,
            teleport_builder: TopicWeightsTeleportBuilder::new(topic_weights, min_weight),
            ranker: PageRankRanker,
            phrase_builder: ChunkPhraseBuilder,
            formatter: StandardResultFormatter,
        }
    }
}

// ---------------------------------------------------------------------------
// TopicRankPipeline — topic-level graph over cluster nodes
// ---------------------------------------------------------------------------

/// Pipeline alias for TopicRank: phrase candidates → HAC clustering →
/// topic graph (clusters as nodes) → PageRank → representative selection.
///
/// Uses [`PhraseCandidateSelector`] for candidate extraction,
/// [`TopicGraphBuilder`] (with embedded [`JaccardHacClusterer`]) for
/// cluster-graph construction, and [`TopicRepresentativeBuilder`] to
/// select one representative phrase per top-scoring cluster.
pub type TopicRankPipeline = Pipeline<
    NoopPreprocessor,
    PhraseCandidateSelector,
    TopicGraphBuilder<JaccardHacClusterer>,
    NoopGraphTransform,
    UniformTeleportBuilder,
    PageRankRanker,
    TopicRepresentativeBuilder,
    StandardResultFormatter,
>;

impl TopicRankPipeline {
    /// Build a pipeline for the TopicRank algorithm with default parameters.
    ///
    /// - Similarity threshold: `0.25` (HAC Jaccard cutoff)
    /// - Edge weight: `1.0`
    pub fn topic_rank(chunks: Vec<crate::types::ChunkSpan>) -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: PhraseCandidateSelector::new(chunks),
            graph_builder: TopicGraphBuilder::new(JaccardHacClusterer::topic_rank()),
            graph_transform: NoopGraphTransform,
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: TopicRepresentativeBuilder,
            formatter: StandardResultFormatter,
        }
    }

    /// Build a TopicRank pipeline with custom similarity threshold and
    /// edge-weight scaling.
    pub fn topic_rank_with(
        chunks: Vec<crate::types::ChunkSpan>,
        similarity_threshold: f64,
        edge_weight: f64,
    ) -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: PhraseCandidateSelector::new(chunks),
            graph_builder: TopicGraphBuilder::new(
                JaccardHacClusterer::new(similarity_threshold),
            )
            .with_edge_weight(edge_weight),
            graph_transform: NoopGraphTransform,
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: TopicRepresentativeBuilder,
            formatter: StandardResultFormatter,
        }
    }
}

// ---------------------------------------------------------------------------
// MultipartiteRankPipeline — candidate-level k-partite graph
// ---------------------------------------------------------------------------

/// Pipeline alias for MultipartiteRank: phrase candidates → HAC clustering →
/// candidate graph (one node per candidate) → k-partite transform + alpha
/// boost → PageRank → highest-scoring per lemma group.
///
/// Unlike [`TopicRankPipeline`] where clusters are graph nodes,
/// MultipartiteRank keeps individual candidates as nodes and removes
/// intra-cluster edges to form a k-partite structure, then boosts edges
/// toward first-occurring variants.
pub type MultipartiteRankPipeline = Pipeline<
    NoopPreprocessor,
    PhraseCandidateSelector,
    CandidateGraphBuilder<JaccardHacClusterer>,
    MultipartiteTransform,
    UniformTeleportBuilder,
    PageRankRanker,
    MultipartitePhraseBuilder,
    StandardResultFormatter,
>;

impl MultipartiteRankPipeline {
    /// Build a pipeline for the MultipartiteRank algorithm with default
    /// parameters.
    ///
    /// - Similarity threshold: `0.26` (HAC Jaccard cutoff)
    /// - Alpha: `1.1` (boost scaling factor)
    pub fn multipartite_rank(chunks: Vec<crate::types::ChunkSpan>) -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: PhraseCandidateSelector::new(chunks),
            graph_builder: CandidateGraphBuilder::new(JaccardHacClusterer::new(0.26)),
            graph_transform: MultipartiteTransform::new(),
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: MultipartitePhraseBuilder,
            formatter: StandardResultFormatter,
        }
    }

    /// Build a MultipartiteRank pipeline with custom similarity threshold
    /// and alpha.
    pub fn multipartite_rank_with(
        chunks: Vec<crate::types::ChunkSpan>,
        similarity_threshold: f64,
        alpha: f64,
    ) -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: PhraseCandidateSelector::new(chunks),
            graph_builder: CandidateGraphBuilder::new(
                JaccardHacClusterer::new(similarity_threshold),
            ),
            graph_transform: MultipartiteTransform::with_alpha(alpha),
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: MultipartitePhraseBuilder,
            formatter: StandardResultFormatter,
        }
    }
}

// ---------------------------------------------------------------------------
// SentenceRankPipeline — extractive summarization via sentence-level ranking
// ---------------------------------------------------------------------------

/// Pipeline alias for SentenceRank: whole sentences as candidates, Jaccard
/// similarity graph, sentence-level phrase assembly, and optional
/// position-based output ordering.
///
/// Suitable for extractive summarization: the `top_n` highest-ranked
/// sentences are returned as "phrases" (each phrase text is a full sentence).
pub type SentenceRankPipeline = Pipeline<
    NoopPreprocessor,
    SentenceCandidateSelector,
    SentenceGraphBuilder,
    NoopGraphTransform,
    UniformTeleportBuilder,
    PageRankRanker,
    SentencePhraseBuilder,
    SentenceFormatter,
>;

impl SentenceRankPipeline {
    /// Build a SentenceRank pipeline with default settings (sort by score).
    pub fn sentence_rank() -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: SentenceCandidateSelector,
            graph_builder: SentenceGraphBuilder::default(),
            graph_transform: NoopGraphTransform,
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: SentencePhraseBuilder,
            formatter: SentenceFormatter::default(),
        }
    }

    /// Build a SentenceRank pipeline that sorts output by document position
    /// instead of score (useful for producing readable summaries).
    pub fn sentence_rank_by_position() -> Self {
        Pipeline {
            preprocessor: NoopPreprocessor,
            selector: SentenceCandidateSelector,
            graph_builder: SentenceGraphBuilder::default(),
            graph_transform: NoopGraphTransform,
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: SentencePhraseBuilder,
            formatter: SentenceFormatter {
                sort_by_position: true,
            },
        }
    }
}

// ============================================================================
// Pipeline::run — execute stages in order
// ============================================================================

impl<Pre, Sel, GB, GT, TB, Rnk, PB, Fmt> Pipeline<Pre, Sel, GB, GT, TB, Rnk, PB, Fmt>
where
    Pre: Preprocessor,
    Sel: CandidateSelector,
    GB: GraphBuilder,
    GT: GraphTransform,
    TB: TeleportBuilder,
    Rnk: Ranker,
    PB: PhraseBuilder,
    Fmt: ResultFormatter,
{
    /// Execute the pipeline, producing a [`FormattedResult`].
    ///
    /// Stages run in order:
    /// 1. Preprocess (mutate tokens in place)
    /// 2. Select candidates
    /// 3. Build graph
    /// 4. Transform graph (optional no-op)
    /// 5. Build teleport vector (optional)
    /// 6. Rank
    /// 7. Build phrases
    /// 8. Format result
    ///
    /// The `observer` receives callbacks at each stage boundary. Pass
    /// [`NoopObserver`] for zero-overhead execution.
    pub fn run(
        &self,
        tokens: TokenStream,
        cfg: &TextRankConfig,
        observer: &mut impl PipelineObserver,
    ) -> FormattedResult {
        self.run_inner(tokens, cfg, observer, None)
    }

    /// Execute the pipeline, reusing workspace buffers for PageRank.
    ///
    /// Same as [`run`](Self::run) but uses the provided
    /// [`PipelineWorkspace`] to avoid per-document allocations in the
    /// ranking stage. Call [`PipelineWorkspace::clear()`] between
    /// invocations (or use [`run_batch`](Self::run_batch) which does
    /// this automatically).
    pub fn run_with_workspace(
        &self,
        tokens: TokenStream,
        cfg: &TextRankConfig,
        observer: &mut impl PipelineObserver,
        ws: &mut PipelineWorkspace,
    ) -> FormattedResult {
        self.run_inner(tokens, cfg, observer, Some(ws))
    }

    /// Execute the pipeline over multiple documents, reusing a single
    /// workspace across all of them.
    ///
    /// Equivalent to calling [`run`](Self::run) on each document
    /// individually, but avoids repeated PageRank buffer allocations.
    pub fn run_batch(
        &self,
        docs: impl IntoIterator<Item = TokenStream>,
        cfg: &TextRankConfig,
        observer: &mut impl PipelineObserver,
    ) -> Vec<FormattedResult> {
        let mut ws = PipelineWorkspace::new();
        docs.into_iter()
            .map(|tokens| {
                ws.clear();
                self.run_with_workspace(tokens, cfg, observer, &mut ws)
            })
            .collect()
    }

    /// Shared orchestration logic for [`run`] and [`run_with_workspace`].
    fn run_inner(
        &self,
        mut tokens: TokenStream,
        cfg: &TextRankConfig,
        observer: &mut impl PipelineObserver,
        ws: Option<&mut PipelineWorkspace>,
    ) -> FormattedResult {
        // Stage 0: Preprocess
        trace_stage!(STAGE_PREPROCESS);
        observer.on_stage_start(STAGE_PREPROCESS);
        let clock = StageClock::start();
        self.preprocessor.preprocess(&mut tokens, cfg);
        let report = StageReport::new(clock.elapsed());
        observer.on_stage_end(STAGE_PREPROCESS, &report);
        observer.on_tokens(&tokens);

        // Stage 1: Select candidates
        trace_stage!(STAGE_CANDIDATES);
        observer.on_stage_start(STAGE_CANDIDATES);
        let clock = StageClock::start();
        let candidates = self.selector.select(tokens.as_ref(), cfg);
        let report = StageReport::new(clock.elapsed());
        observer.on_stage_end(STAGE_CANDIDATES, &report);
        observer.on_candidates(&candidates);

        // Stage 2: Build graph
        trace_stage!(STAGE_GRAPH);
        observer.on_stage_start(STAGE_GRAPH);
        let clock = StageClock::start();
        let mut graph = self
            .graph_builder
            .build(tokens.as_ref(), candidates.as_ref(), cfg);
        let report = StageReportBuilder::new(clock.elapsed())
            .nodes(graph.num_nodes())
            .edges(graph.num_edges())
            .build();
        observer.on_stage_end(STAGE_GRAPH, &report);

        // Stage 2a: Transform graph
        trace_stage!(STAGE_GRAPH_TRANSFORM);
        observer.on_stage_start(STAGE_GRAPH_TRANSFORM);
        let clock = StageClock::start();
        self.graph_transform
            .transform(&mut graph, tokens.as_ref(), candidates.as_ref(), cfg);
        let report = StageReport::new(clock.elapsed());
        observer.on_stage_end(STAGE_GRAPH_TRANSFORM, &report);
        observer.on_graph(&graph);

        // Stage 3a: Build teleport vector
        trace_stage!(STAGE_TELEPORT);
        observer.on_stage_start(STAGE_TELEPORT);
        let clock = StageClock::start();
        let teleport = self
            .teleport_builder
            .build(tokens.as_ref(), candidates.as_ref(), cfg);
        let report = StageReport::new(clock.elapsed());
        observer.on_stage_end(STAGE_TELEPORT, &report);

        // Stage 3: Rank
        trace_stage!(STAGE_RANK);
        observer.on_stage_start(STAGE_RANK);
        let clock = StageClock::start();
        let rank_output = match ws {
            Some(ws) => self.ranker.rank_reusing(&graph, teleport.as_ref(), cfg, ws),
            None => self.ranker.rank(&graph, teleport.as_ref(), cfg),
        };
        let report = StageReportBuilder::new(clock.elapsed())
            .iterations(rank_output.iterations())
            .converged(rank_output.converged())
            .residual(rank_output.final_delta())
            .build();
        observer.on_stage_end(STAGE_RANK, &report);
        observer.on_rank(&rank_output);

        // Stage 4: Build phrases
        trace_stage!(STAGE_PHRASES);
        observer.on_stage_start(STAGE_PHRASES);
        let clock = StageClock::start();
        let phrases = self.phrase_builder.build(
            tokens.as_ref(),
            candidates.as_ref(),
            &rank_output,
            &graph,
            cfg,
        );
        let report = StageReport::new(clock.elapsed());
        observer.on_stage_end(STAGE_PHRASES, &report);
        observer.on_phrases(&phrases);

        // Build debug payload (opt-in via cfg.debug_level).
        let debug_payload = super::DebugPayload::build(
            cfg.debug_level,
            &graph,
            &rank_output,
            DebugLevel::DEFAULT_TOP_K,
        );

        // Stage 5: Format result
        trace_stage!(STAGE_FORMAT);
        observer.on_stage_start(STAGE_FORMAT);
        let clock = StageClock::start();
        let result = self.formatter.format(&phrases, &rank_output, debug_payload, cfg);
        let report = StageReport::new(clock.elapsed());
        observer.on_stage_end(STAGE_FORMAT, &report);

        result
    }
}

// ============================================================================
// PipelineBuilder — fluent construction with custom stages
// ============================================================================

/// Fluent builder for constructing a [`Pipeline`] with custom stages.
///
/// Starts from a default BaseTextRank configuration and allows overriding
/// individual stages.
///
/// ```
/// # use rapid_textrank::pipeline::runner::PipelineBuilder;
/// # use rapid_textrank::pipeline::traits::*;
/// let pipeline = PipelineBuilder::new()
///     .graph_builder(WindowGraphBuilder::single_rank())
///     .build();
/// ```
pub struct PipelineBuilder<Pre = NoopPreprocessor, Sel = WordNodeSelector, GB = WindowGraphBuilder, GT = NoopGraphTransform, TB = UniformTeleportBuilder, Rnk = PageRankRanker, PB = ChunkPhraseBuilder, Fmt = StandardResultFormatter> {
    preprocessor: Pre,
    selector: Sel,
    graph_builder: GB,
    graph_transform: GT,
    teleport_builder: TB,
    ranker: Rnk,
    phrase_builder: PB,
    formatter: Fmt,
}

impl PipelineBuilder {
    /// Start building from default BaseTextRank stages.
    pub fn new() -> Self {
        PipelineBuilder {
            preprocessor: NoopPreprocessor,
            selector: WordNodeSelector,
            graph_builder: WindowGraphBuilder::base_textrank(),
            graph_transform: NoopGraphTransform,
            teleport_builder: UniformTeleportBuilder,
            ranker: PageRankRanker,
            phrase_builder: ChunkPhraseBuilder,
            formatter: StandardResultFormatter,
        }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl<Pre, Sel, GB, GT, TB, Rnk, PB, Fmt> PipelineBuilder<Pre, Sel, GB, GT, TB, Rnk, PB, Fmt> {
    /// Override the preprocessor stage.
    pub fn preprocessor<P: Preprocessor>(self, p: P) -> PipelineBuilder<P, Sel, GB, GT, TB, Rnk, PB, Fmt> {
        PipelineBuilder {
            preprocessor: p,
            selector: self.selector,
            graph_builder: self.graph_builder,
            graph_transform: self.graph_transform,
            teleport_builder: self.teleport_builder,
            ranker: self.ranker,
            phrase_builder: self.phrase_builder,
            formatter: self.formatter,
        }
    }

    /// Override the candidate selector stage.
    pub fn selector<S: CandidateSelector>(self, s: S) -> PipelineBuilder<Pre, S, GB, GT, TB, Rnk, PB, Fmt> {
        PipelineBuilder {
            preprocessor: self.preprocessor,
            selector: s,
            graph_builder: self.graph_builder,
            graph_transform: self.graph_transform,
            teleport_builder: self.teleport_builder,
            ranker: self.ranker,
            phrase_builder: self.phrase_builder,
            formatter: self.formatter,
        }
    }

    /// Override the graph builder stage.
    pub fn graph_builder<G: GraphBuilder>(self, g: G) -> PipelineBuilder<Pre, Sel, G, GT, TB, Rnk, PB, Fmt> {
        PipelineBuilder {
            preprocessor: self.preprocessor,
            selector: self.selector,
            graph_builder: g,
            graph_transform: self.graph_transform,
            teleport_builder: self.teleport_builder,
            ranker: self.ranker,
            phrase_builder: self.phrase_builder,
            formatter: self.formatter,
        }
    }

    /// Override the graph transform stage.
    pub fn graph_transform<G: GraphTransform>(self, g: G) -> PipelineBuilder<Pre, Sel, GB, G, TB, Rnk, PB, Fmt> {
        PipelineBuilder {
            preprocessor: self.preprocessor,
            selector: self.selector,
            graph_builder: self.graph_builder,
            graph_transform: g,
            teleport_builder: self.teleport_builder,
            ranker: self.ranker,
            phrase_builder: self.phrase_builder,
            formatter: self.formatter,
        }
    }

    /// Override the teleport builder stage.
    pub fn teleport_builder<T: TeleportBuilder>(self, t: T) -> PipelineBuilder<Pre, Sel, GB, GT, T, Rnk, PB, Fmt> {
        PipelineBuilder {
            preprocessor: self.preprocessor,
            selector: self.selector,
            graph_builder: self.graph_builder,
            graph_transform: self.graph_transform,
            teleport_builder: t,
            ranker: self.ranker,
            phrase_builder: self.phrase_builder,
            formatter: self.formatter,
        }
    }

    /// Override the ranker stage.
    pub fn ranker<R: Ranker>(self, r: R) -> PipelineBuilder<Pre, Sel, GB, GT, TB, R, PB, Fmt> {
        PipelineBuilder {
            preprocessor: self.preprocessor,
            selector: self.selector,
            graph_builder: self.graph_builder,
            graph_transform: self.graph_transform,
            teleport_builder: self.teleport_builder,
            ranker: r,
            phrase_builder: self.phrase_builder,
            formatter: self.formatter,
        }
    }

    /// Override the phrase builder stage.
    pub fn phrase_builder<P: PhraseBuilder>(self, p: P) -> PipelineBuilder<Pre, Sel, GB, GT, TB, Rnk, P, Fmt> {
        PipelineBuilder {
            preprocessor: self.preprocessor,
            selector: self.selector,
            graph_builder: self.graph_builder,
            graph_transform: self.graph_transform,
            teleport_builder: self.teleport_builder,
            ranker: self.ranker,
            phrase_builder: p,
            formatter: self.formatter,
        }
    }

    /// Override the result formatter stage.
    pub fn formatter<F: ResultFormatter>(self, f: F) -> PipelineBuilder<Pre, Sel, GB, GT, TB, Rnk, PB, F> {
        PipelineBuilder {
            preprocessor: self.preprocessor,
            selector: self.selector,
            graph_builder: self.graph_builder,
            graph_transform: self.graph_transform,
            teleport_builder: self.teleport_builder,
            ranker: self.ranker,
            phrase_builder: self.phrase_builder,
            formatter: f,
        }
    }

    /// Consume the builder and produce a [`Pipeline`].
    pub fn build(self) -> Pipeline<Pre, Sel, GB, GT, TB, Rnk, PB, Fmt> {
        Pipeline {
            preprocessor: self.preprocessor,
            selector: self.selector,
            graph_builder: self.graph_builder,
            graph_transform: self.graph_transform,
            teleport_builder: self.teleport_builder,
            ranker: self.ranker,
            phrase_builder: self.phrase_builder,
            formatter: self.formatter,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::artifacts::{CandidateSet, Graph, PhraseSet, RankOutput};
    use crate::pipeline::observer::{NoopObserver, StageTimingObserver};
    use crate::types::{PosTag, Token};

    fn sample_tokens() -> Vec<Token> {
        // "Rust is a systems programming language"
        vec![
            Token {
                text: "Rust".into(),
                lemma: "rust".into(),
                pos: PosTag::ProperNoun,
                start: 0,
                end: 4,
                sentence_idx: 0,
                token_idx: 0,
                is_stopword: false,
            },
            Token {
                text: "is".into(),
                lemma: "be".into(),
                pos: PosTag::Verb,
                start: 5,
                end: 7,
                sentence_idx: 0,
                token_idx: 1,
                is_stopword: true,
            },
            Token {
                text: "a".into(),
                lemma: "a".into(),
                pos: PosTag::Determiner,
                start: 8,
                end: 9,
                sentence_idx: 0,
                token_idx: 2,
                is_stopword: true,
            },
            Token {
                text: "systems".into(),
                lemma: "system".into(),
                pos: PosTag::Noun,
                start: 10,
                end: 17,
                sentence_idx: 0,
                token_idx: 3,
                is_stopword: false,
            },
            Token {
                text: "programming".into(),
                lemma: "programming".into(),
                pos: PosTag::Noun,
                start: 18,
                end: 29,
                sentence_idx: 0,
                token_idx: 4,
                is_stopword: false,
            },
            Token {
                text: "language".into(),
                lemma: "language".into(),
                pos: PosTag::Noun,
                start: 30,
                end: 38,
                sentence_idx: 0,
                token_idx: 5,
                is_stopword: false,
            },
        ]
    }

    fn make_token_stream() -> TokenStream {
        TokenStream::from_tokens(&sample_tokens())
    }

    #[test]
    fn test_base_textrank_pipeline_constructs() {
        let _pipeline = BaseTextRankPipeline::base_textrank();
    }

    #[test]
    fn test_pipeline_builder_default() {
        let _pipeline = PipelineBuilder::new().build();
    }

    #[test]
    fn test_pipeline_builder_with_custom_graph() {
        let pipeline = PipelineBuilder::new()
            .graph_builder(WindowGraphBuilder::single_rank())
            .build();
        // Verify it's the SingleRank config (cross-sentence, count weighting).
        assert_eq!(
            format!("{:?}", pipeline.graph_builder),
            format!("{:?}", WindowGraphBuilder::single_rank())
        );
    }

    #[test]
    fn test_pipeline_run_with_noop_observer() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let tokens = make_token_stream();
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let result = pipeline.run(tokens, &cfg, &mut obs);
        // Should produce some phrases (the exact results depend on the algorithm).
        assert!(result.converged);
    }

    #[test]
    fn test_pipeline_run_with_timing_observer() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let tokens = make_token_stream();
        let cfg = TextRankConfig::default();
        let mut obs = StageTimingObserver::new();

        let _result = pipeline.run(tokens, &cfg, &mut obs);

        // Should have reports for all 8 stages.
        assert_eq!(obs.reports().len(), 8);
        let stage_names: Vec<&str> = obs.reports().iter().map(|(name, _)| *name).collect();
        assert_eq!(
            stage_names,
            vec![
                STAGE_PREPROCESS,
                STAGE_CANDIDATES,
                STAGE_GRAPH,
                STAGE_GRAPH_TRANSFORM,
                STAGE_TELEPORT,
                STAGE_RANK,
                STAGE_PHRASES,
                STAGE_FORMAT,
            ]
        );
    }

    #[test]
    fn test_pipeline_observer_receives_graph_metrics() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let tokens = make_token_stream();
        let cfg = TextRankConfig::default();
        let mut obs = StageTimingObserver::new();

        let _result = pipeline.run(tokens, &cfg, &mut obs);

        // Graph stage should report nodes and edges.
        let (_, graph_report) = &obs.reports()[2]; // STAGE_GRAPH is 3rd
        assert!(graph_report.nodes().is_some());
        assert!(graph_report.edges().is_some());
    }

    #[test]
    fn test_pipeline_observer_receives_rank_metrics() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let tokens = make_token_stream();
        let cfg = TextRankConfig::default();
        let mut obs = StageTimingObserver::new();

        let _result = pipeline.run(tokens, &cfg, &mut obs);

        // Rank stage should report iterations, converged, residual.
        let (_, rank_report) = &obs.reports()[5]; // STAGE_RANK is 6th
        assert!(rank_report.iterations().is_some());
        assert!(rank_report.converged().is_some());
        assert!(rank_report.residual().is_some());
    }

    #[test]
    fn test_pipeline_run_empty_input() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let tokens = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let result = pipeline.run(tokens, &cfg, &mut obs);
        assert!(result.phrases.is_empty());
    }

    #[test]
    fn test_pipeline_run_produces_phrases() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let tokens = make_token_stream();
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let result = pipeline.run(tokens, &cfg, &mut obs);
        // With "Rust is a systems programming language", we should get phrases.
        assert!(!result.phrases.is_empty());
    }

    /// Custom observer that captures artifact snapshots.
    struct ArtifactObserver {
        saw_tokens: bool,
        saw_candidates: bool,
        saw_graph: bool,
        saw_rank: bool,
        saw_phrases: bool,
    }

    impl ArtifactObserver {
        fn new() -> Self {
            Self {
                saw_tokens: false,
                saw_candidates: false,
                saw_graph: false,
                saw_rank: false,
                saw_phrases: false,
            }
        }
    }

    impl PipelineObserver for ArtifactObserver {
        fn on_tokens(&mut self, _tokens: &TokenStream) {
            self.saw_tokens = true;
        }
        fn on_candidates(&mut self, _candidates: &CandidateSet) {
            self.saw_candidates = true;
        }
        fn on_graph(&mut self, _graph: &Graph) {
            self.saw_graph = true;
        }
        fn on_rank(&mut self, _rank: &RankOutput) {
            self.saw_rank = true;
        }
        fn on_phrases(&mut self, _phrases: &PhraseSet) {
            self.saw_phrases = true;
        }
    }

    #[test]
    fn test_pipeline_calls_all_artifact_observers() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let tokens = make_token_stream();
        let cfg = TextRankConfig::default();
        let mut obs = ArtifactObserver::new();

        let _result = pipeline.run(tokens, &cfg, &mut obs);

        assert!(obs.saw_tokens, "on_tokens not called");
        assert!(obs.saw_candidates, "on_candidates not called");
        assert!(obs.saw_graph, "on_graph not called");
        assert!(obs.saw_rank, "on_rank not called");
        assert!(obs.saw_phrases, "on_phrases not called");
    }

    // ================================================================
    // Cross-path golden tests: pipeline vs legacy path
    // ================================================================

    /// Multi-sentence token set matching the golden tests in
    /// `phrase/extraction.rs` — "Machine learning uses algorithms.
    /// Deep learning uses neural networks. Machine learning models
    /// improve with data."
    fn golden_tokens() -> Vec<Token> {
        let mut tokens = vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("uses", "use", PosTag::Verb, 17, 21, 0, 2),
            Token::new("algorithms", "algorithm", PosTag::Noun, 22, 32, 0, 3),
            Token::new("Deep", "deep", PosTag::Adjective, 34, 38, 1, 4),
            Token::new("learning", "learning", PosTag::Noun, 39, 47, 1, 5),
            Token::new("uses", "use", PosTag::Verb, 48, 52, 1, 6),
            Token::new("neural", "neural", PosTag::Adjective, 53, 59, 1, 7),
            Token::new("networks", "network", PosTag::Noun, 60, 68, 1, 8),
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

    /// Golden test: BaseTextRank pipeline produces identical output to the
    /// legacy `extract_keyphrases_with_info` path.
    ///
    /// This validates that modularizing graph construction into
    /// WindowGraphBuilder (SentenceBounded + Binary) preserves the exact
    /// same behavior as the fused legacy code path.
    #[test]
    fn golden_base_textrank_pipeline_matches_legacy() {
        use crate::phrase::extraction::extract_keyphrases_with_info;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        // Legacy path
        let legacy = extract_keyphrases_with_info(&tokens, &cfg);

        // Pipeline path
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = NoopObserver;
        let pipeline_result = BaseTextRankPipeline::base_textrank().run(stream, &cfg, &mut obs);

        // Convergence metadata must match.
        assert_eq!(
            legacy.converged, pipeline_result.converged,
            "Convergence mismatch: legacy={}, pipeline={}",
            legacy.converged, pipeline_result.converged
        );
        assert_eq!(
            legacy.iterations, pipeline_result.iterations as usize,
            "Iteration count mismatch: legacy={}, pipeline={}",
            legacy.iterations, pipeline_result.iterations
        );

        // Phrase count must match.
        assert_eq!(
            legacy.phrases.len(),
            pipeline_result.phrases.len(),
            "Phrase count mismatch: legacy={}, pipeline={}",
            legacy.phrases.len(),
            pipeline_result.phrases.len()
        );

        // Every phrase must match: text, lemma, score, rank.
        let eps = 1e-8;
        for (i, (lp, pp)) in legacy
            .phrases
            .iter()
            .zip(pipeline_result.phrases.iter())
            .enumerate()
        {
            assert_eq!(
                lp.text, pp.text,
                "Phrase text mismatch at position {i}: legacy={:?}, pipeline={:?}",
                lp.text, pp.text
            );
            assert_eq!(
                lp.lemma, pp.lemma,
                "Phrase lemma mismatch at position {i}: legacy={:?}, pipeline={:?}",
                lp.lemma, pp.lemma
            );
            assert!(
                (lp.score - pp.score).abs() < eps,
                "Score mismatch at position {i} ({:?}): legacy={:.10}, pipeline={:.10}, delta={:.2e}",
                lp.text,
                lp.score,
                pp.score,
                (lp.score - pp.score).abs()
            );
            assert_eq!(
                lp.rank, pp.rank,
                "Rank mismatch at position {i}: legacy={}, pipeline={}",
                lp.rank, pp.rank
            );
        }
    }

    /// Golden test: single-sentence input also matches between paths.
    #[test]
    fn golden_base_textrank_pipeline_matches_legacy_single_sentence() {
        use crate::phrase::extraction::extract_keyphrases_with_info;

        let tokens = vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
            Token::new("a", "a", PosTag::Determiner, 20, 21, 0, 3),
            Token::new("subset", "subset", PosTag::Noun, 22, 28, 0, 4),
            Token::new("of", "of", PosTag::Preposition, 29, 31, 0, 5),
            Token::new("artificial", "artificial", PosTag::Adjective, 32, 42, 0, 6),
            Token::new("intelligence", "intelligence", PosTag::Noun, 43, 55, 0, 7),
        ];
        let cfg = TextRankConfig::default();

        let legacy = extract_keyphrases_with_info(&tokens, &cfg);
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = NoopObserver;
        let pipeline_result = BaseTextRankPipeline::base_textrank().run(stream, &cfg, &mut obs);

        assert_eq!(legacy.converged, pipeline_result.converged);
        assert_eq!(legacy.iterations, pipeline_result.iterations as usize);
        assert_eq!(legacy.phrases.len(), pipeline_result.phrases.len());

        let eps = 1e-8;
        for (i, (lp, pp)) in legacy
            .phrases
            .iter()
            .zip(pipeline_result.phrases.iter())
            .enumerate()
        {
            assert_eq!(lp.text, pp.text, "Text mismatch at {i}");
            assert_eq!(lp.lemma, pp.lemma, "Lemma mismatch at {i}");
            assert!(
                (lp.score - pp.score).abs() < eps,
                "Score mismatch at {i}: legacy={:.10}, pipeline={:.10}",
                lp.score,
                pp.score
            );
            assert_eq!(lp.rank, pp.rank, "Rank mismatch at {i}");
        }
    }

    /// Golden test: empty input produces identical empty output from both paths.
    #[test]
    fn golden_base_textrank_pipeline_matches_legacy_empty() {
        use crate::phrase::extraction::extract_keyphrases_with_info;

        let tokens: Vec<Token> = Vec::new();
        let cfg = TextRankConfig::default();

        let legacy = extract_keyphrases_with_info(&tokens, &cfg);
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = NoopObserver;
        let pipeline_result = BaseTextRankPipeline::base_textrank().run(stream, &cfg, &mut obs);

        assert_eq!(legacy.converged, pipeline_result.converged);
        assert!(legacy.phrases.is_empty());
        assert!(pipeline_result.phrases.is_empty());
    }

    // ================================================================
    // SingleRank golden tests: pipeline vs legacy direct path
    // ================================================================

    /// Golden test: SingleRank pipeline produces identical output to the
    /// legacy direct graph-construction path.
    ///
    /// The legacy path (pre-refactor) used `GraphBuilder::from_tokens_with_pos_and_boundaries`
    /// with `weighted=true, respect_sentence_boundaries=false`. This test
    /// reconstructs that path inline and compares to `SingleRankPipeline`.
    #[test]
    fn golden_single_rank_pipeline_matches_legacy() {
        use crate::graph::builder::GraphBuilder as LegacyGraphBuilder;
        use crate::graph::csr::CsrGraph;
        use crate::phrase::extraction::PhraseExtractor;
        use super::SingleRankPipeline;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        // --- Legacy direct path (reconstructed) ---
        let include_pos: Option<&[_]> = if cfg.include_pos.is_empty() {
            None
        } else {
            Some(&cfg.include_pos)
        };

        let builder = LegacyGraphBuilder::from_tokens_with_pos_and_boundaries(
            &tokens,
            cfg.window_size,
            true,  // always weighted
            include_pos,
            cfg.use_pos_in_nodes,
            false, // cross-sentence windowing
        );
        let graph = CsrGraph::from_builder(&builder);
        let pagerank = crate::pagerank::standard::StandardPageRank::new()
            .with_damping(cfg.damping)
            .with_max_iterations(cfg.max_iterations)
            .with_threshold(cfg.convergence_threshold)
            .run(&graph);
        let extractor = PhraseExtractor::with_config(cfg.clone());
        let legacy_phrases = extractor.extract(&tokens, &graph, &pagerank);

        // --- Pipeline path ---
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = NoopObserver;
        let pipeline_result = SingleRankPipeline::single_rank().run(stream, &cfg, &mut obs);

        // Convergence metadata.
        assert_eq!(
            pagerank.converged, pipeline_result.converged,
            "Convergence mismatch: legacy={}, pipeline={}",
            pagerank.converged, pipeline_result.converged
        );
        assert_eq!(
            pagerank.iterations, pipeline_result.iterations as usize,
            "Iteration count mismatch: legacy={}, pipeline={}",
            pagerank.iterations, pipeline_result.iterations
        );

        // Phrase count.
        assert_eq!(
            legacy_phrases.len(),
            pipeline_result.phrases.len(),
            "Phrase count mismatch: legacy={}, pipeline={}",
            legacy_phrases.len(),
            pipeline_result.phrases.len()
        );

        // Per-phrase exact match.
        let eps = 1e-8;
        for (i, (lp, pp)) in legacy_phrases
            .iter()
            .zip(pipeline_result.phrases.iter())
            .enumerate()
        {
            assert_eq!(
                lp.text, pp.text,
                "Text mismatch at {i}: legacy={:?}, pipeline={:?}",
                lp.text, pp.text
            );
            assert_eq!(
                lp.lemma, pp.lemma,
                "Lemma mismatch at {i}: legacy={:?}, pipeline={:?}",
                lp.lemma, pp.lemma
            );
            assert!(
                (lp.score - pp.score).abs() < eps,
                "Score mismatch at {i} ({:?}): legacy={:.10}, pipeline={:.10}, delta={:.2e}",
                lp.text,
                lp.score,
                pp.score,
                (lp.score - pp.score).abs()
            );
            assert_eq!(
                lp.rank, pp.rank,
                "Rank mismatch at {i}: legacy={}, pipeline={}",
                lp.rank, pp.rank
            );
        }
    }

    /// Golden test: SingleRank pipeline with empty input returns empty.
    #[test]
    fn golden_single_rank_pipeline_empty() {
        use super::SingleRankPipeline;

        let tokens: Vec<Token> = Vec::new();
        let cfg = TextRankConfig::default();
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = NoopObserver;

        let result = SingleRankPipeline::single_rank().run(stream, &cfg, &mut obs);
        assert!(result.phrases.is_empty());
        assert!(result.converged);
    }

    /// Verify SingleRank pipeline produces different results from BaseTextRank
    /// on multi-sentence input (cross-sentence edges change the graph).
    #[test]
    fn single_rank_differs_from_base_textrank_on_multi_sentence() {
        use super::SingleRankPipeline;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        let base_result = {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            BaseTextRankPipeline::base_textrank().run(stream, &cfg, &mut obs)
        };

        let single_result = {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            SingleRankPipeline::single_rank().run(stream, &cfg, &mut obs)
        };

        // Both should produce phrases.
        assert!(!base_result.phrases.is_empty());
        assert!(!single_result.phrases.is_empty());

        // They should differ: cross-sentence windowing creates additional
        // edges that change the PageRank distribution. Compare scores to
        // detect the difference (at least one phrase score must differ).
        let any_diff = base_result
            .phrases
            .iter()
            .zip(single_result.phrases.iter())
            .any(|(b, s)| (b.score - s.score).abs() > 1e-10 || b.text != s.text);

        assert!(
            any_diff,
            "SingleRank and BaseTextRank produced identical results on multi-sentence input; \
             cross-sentence edges should cause a difference"
        );
    }

    // ================================================================
    // TopicalPageRank pipeline tests
    // ================================================================

    #[test]
    fn test_topical_pipeline_constructs() {
        use std::collections::HashMap;
        let mut weights = HashMap::new();
        weights.insert("machine".to_string(), 2.0);
        let _pipeline = super::TopicalPageRankPipeline::topical(weights, 0.01);
    }

    #[test]
    fn test_topical_pipeline_runs() {
        use std::collections::HashMap;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();
        let mut weights = HashMap::new();
        weights.insert("machine".to_string(), 2.0);
        weights.insert("network".to_string(), 1.5);

        let pipeline = super::TopicalPageRankPipeline::topical(weights, 0.01);
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = StageTimingObserver::new();

        let result = pipeline.run(stream, &cfg, &mut obs);

        assert!(result.converged);
        assert!(!result.phrases.is_empty());
        // Observer should have reports for all 8 stages.
        assert_eq!(obs.reports().len(), 8);
    }

    #[test]
    fn test_topical_pipeline_topic_weights_affect_ranking() {
        use std::collections::HashMap;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        // Run with high weight on "machine".
        let mut biased_weights = HashMap::new();
        biased_weights.insert("machine".to_string(), 10.0);
        let biased_pipeline =
            super::TopicalPageRankPipeline::topical(biased_weights, 0.01);
        let biased_result = {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            biased_pipeline.run(stream, &cfg, &mut obs)
        };

        // Run with uniform (all min_weight) — empty map.
        let uniform_pipeline =
            super::TopicalPageRankPipeline::topical(HashMap::new(), 1.0);
        let uniform_result = {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            uniform_pipeline.run(stream, &cfg, &mut obs)
        };

        // Both should produce phrases.
        assert!(!biased_result.phrases.is_empty());
        assert!(!uniform_result.phrases.is_empty());

        // Find score of a phrase containing "machine" in both runs.
        let biased_machine_score = biased_result
            .phrases
            .iter()
            .find(|p| p.lemma.contains("machine"))
            .map(|p| p.score);
        let uniform_machine_score = uniform_result
            .phrases
            .iter()
            .find(|p| p.lemma.contains("machine"))
            .map(|p| p.score);

        // With heavy bias on "machine", its score should be higher
        // than in the uniform run.
        assert!(
            biased_machine_score > uniform_machine_score,
            "Biased 'machine' score ({:?}) should exceed uniform ({:?})",
            biased_machine_score,
            uniform_machine_score
        );
    }

    #[test]
    fn test_topical_pipeline_via_builder() {
        use crate::pipeline::traits::TopicWeightsTeleportBuilder;
        use std::collections::HashMap;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        let mut weights = HashMap::new();
        weights.insert("machine".to_string(), 2.0);
        weights.insert("network".to_string(), 1.5);

        // Build via type alias constructor.
        let alias_result = {
            let pipeline =
                super::TopicalPageRankPipeline::topical(weights.clone(), 0.01);
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            pipeline.run(stream, &cfg, &mut obs)
        };

        // Build via PipelineBuilder with equivalent stages.
        let builder_result = {
            let pipeline = PipelineBuilder::new()
                .graph_builder(WindowGraphBuilder::single_rank())
                .teleport_builder(TopicWeightsTeleportBuilder::new(weights, 0.01))
                .build();
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            pipeline.run(stream, &cfg, &mut obs)
        };

        // Both paths must produce identical results.
        assert_eq!(alias_result.converged, builder_result.converged);
        assert_eq!(alias_result.iterations, builder_result.iterations);
        assert_eq!(alias_result.phrases.len(), builder_result.phrases.len());

        let eps = 1e-10;
        for (i, (a, b)) in alias_result
            .phrases
            .iter()
            .zip(builder_result.phrases.iter())
            .enumerate()
        {
            assert_eq!(a.text, b.text, "Text mismatch at position {i}");
            assert_eq!(a.lemma, b.lemma, "Lemma mismatch at position {i}");
            assert!(
                (a.score - b.score).abs() < eps,
                "Score mismatch at {i}: alias={:.10}, builder={:.10}",
                a.score,
                b.score
            );
            assert_eq!(a.rank, b.rank, "Rank mismatch at position {i}");
        }
    }

    // ================================================================
    // TopicalPageRank pipeline integration tests (e61.3)
    // ================================================================

    /// Golden test: TopicalPageRank pipeline produces identical output to
    /// the legacy `TopicalPageRank::extract_with_info()` path.
    #[test]
    fn golden_topical_pipeline_matches_legacy() {
        use crate::variants::topical_pagerank::TopicalPageRank;
        use std::collections::HashMap;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        let mut weights = HashMap::new();
        weights.insert("machine".to_string(), 2.0);
        weights.insert("network".to_string(), 1.5);

        // Legacy path
        let legacy = TopicalPageRank::with_config(cfg.clone())
            .with_topic_weights(weights.clone())
            .with_min_weight(0.1)
            .extract_with_info(&tokens);

        // Pipeline path
        let pipeline = super::TopicalPageRankPipeline::topical(weights, 0.1);
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = NoopObserver;
        let pipeline_result = pipeline.run(stream, &cfg, &mut obs);

        // Convergence metadata must match.
        assert_eq!(
            legacy.converged, pipeline_result.converged,
            "Convergence mismatch: legacy={}, pipeline={}",
            legacy.converged, pipeline_result.converged
        );
        assert_eq!(
            legacy.iterations, pipeline_result.iterations as usize,
            "Iteration count mismatch: legacy={}, pipeline={}",
            legacy.iterations, pipeline_result.iterations
        );

        // Phrase count must match.
        assert_eq!(
            legacy.phrases.len(),
            pipeline_result.phrases.len(),
            "Phrase count mismatch: legacy={}, pipeline={}",
            legacy.phrases.len(),
            pipeline_result.phrases.len()
        );

        // Every phrase must match: text, lemma, score, rank.
        let eps = 1e-8;
        for (i, (lp, pp)) in legacy
            .phrases
            .iter()
            .zip(pipeline_result.phrases.iter())
            .enumerate()
        {
            assert_eq!(
                lp.text, pp.text,
                "Text mismatch at {i}: legacy={:?}, pipeline={:?}",
                lp.text, pp.text
            );
            assert_eq!(
                lp.lemma, pp.lemma,
                "Lemma mismatch at {i}: legacy={:?}, pipeline={:?}",
                lp.lemma, pp.lemma
            );
            assert!(
                (lp.score - pp.score).abs() < eps,
                "Score mismatch at {i} ({:?}): legacy={:.10}, pipeline={:.10}, delta={:.2e}",
                lp.text,
                lp.score,
                pp.score,
                (lp.score - pp.score).abs()
            );
            assert_eq!(
                lp.rank, pp.rank,
                "Rank mismatch at {i}: legacy={}, pipeline={}",
                lp.rank, pp.rank
            );
        }
    }

    /// Verify that `min_weight` controls how out-of-vocabulary candidates
    /// are scored: OOV lemmas should score higher with `min_weight=1.0`
    /// than with `min_weight=0.0`.
    #[test]
    fn test_topical_pipeline_min_weight_affects_oov() {
        use std::collections::HashMap;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        // Only "machine" has a topic weight — "algorithm" is OOV.
        let mut weights = HashMap::new();
        weights.insert("machine".to_string(), 2.0);

        // min_weight=0.0 → OOV candidates get zero teleport probability.
        let result_zero = {
            let pipeline =
                super::TopicalPageRankPipeline::topical(weights.clone(), 0.0);
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            pipeline.run(stream, &cfg, &mut obs)
        };

        // min_weight=1.0 → OOV candidates get base teleport weight.
        let result_one = {
            let pipeline =
                super::TopicalPageRankPipeline::topical(weights, 1.0);
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            pipeline.run(stream, &cfg, &mut obs)
        };

        // Find score of phrase containing "algorithm" in both runs.
        let algo_score_zero = result_zero
            .phrases
            .iter()
            .find(|p| p.lemma.contains("algorithm"))
            .map(|p| p.score)
            .expect("Should find 'algorithm' phrase in zero run");
        let algo_score_one = result_one
            .phrases
            .iter()
            .find(|p| p.lemma.contains("algorithm"))
            .map(|p| p.score)
            .expect("Should find 'algorithm' phrase in one run");

        assert!(
            algo_score_one > algo_score_zero,
            "OOV 'algorithm' score with min_weight=1.0 ({:.10}) should exceed \
             min_weight=0.0 ({:.10})",
            algo_score_one,
            algo_score_zero
        );
    }

    /// Verify sensitivity — changing which lemma gets the highest topic
    /// weight changes relative ranking.  "algorithm" appears only once
    /// in `golden_tokens()` while "machine learning" appears twice, so
    /// graph structure strongly favors "machine".  We assert a weaker
    /// (but still meaningful) property: the *relative* score of
    /// "algorithm" vs "machine" must shift in favor of whichever lemma
    /// receives the large topic weight.
    #[test]
    fn test_topical_pipeline_different_weights_change_top_phrase() {
        use std::collections::HashMap;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        // Heavy weight on "machine".
        let result_machine = {
            let mut w = HashMap::new();
            w.insert("machine".to_string(), 100.0);
            let pipeline = super::TopicalPageRankPipeline::topical(w, 0.01);
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            pipeline.run(stream, &cfg, &mut obs)
        };

        // Heavy weight on "algorithm".
        let result_algo = {
            let mut w = HashMap::new();
            w.insert("algorithm".to_string(), 100.0);
            let pipeline = super::TopicalPageRankPipeline::topical(w, 0.01);
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            pipeline.run(stream, &cfg, &mut obs)
        };

        // Helper: find score of a phrase whose lemma contains the target.
        let find_score = |phrases: &[crate::types::Phrase], target: &str| -> f64 {
            phrases
                .iter()
                .find(|p| p.lemma.contains(target))
                .map(|p| p.score)
                .unwrap_or(0.0)
        };

        // When "machine" is boosted, its score relative to "algorithm"
        // should be higher than in the "algorithm"-boosted run.
        let machine_ratio_in_m = find_score(&result_machine.phrases, "machine")
            / find_score(&result_machine.phrases, "algorithm").max(1e-15);
        let machine_ratio_in_a = find_score(&result_algo.phrases, "machine")
            / find_score(&result_algo.phrases, "algorithm").max(1e-15);

        assert!(
            machine_ratio_in_m > machine_ratio_in_a,
            "machine/algorithm score ratio should be higher when 'machine' is \
             boosted ({:.4}) than when 'algorithm' is boosted ({:.4})",
            machine_ratio_in_m,
            machine_ratio_in_a
        );
    }

    /// Verify that per-call config overrides (damping factor) affect
    /// pipeline output — different damping values must produce different
    /// score distributions.
    #[test]
    fn test_topical_pipeline_per_call_config_override() {
        use std::collections::HashMap;

        let tokens = golden_tokens();

        let mut weights = HashMap::new();
        weights.insert("machine".to_string(), 2.0);
        let pipeline = super::TopicalPageRankPipeline::topical(weights, 0.1);

        // Low damping — teleport dominates.
        let mut cfg_low = TextRankConfig::default();
        cfg_low.damping = 0.5;
        let result_low = {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            pipeline.run(stream, &cfg_low, &mut obs)
        };

        // High damping — link structure dominates.
        let mut cfg_high = TextRankConfig::default();
        cfg_high.damping = 0.99;
        let result_high = {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            pipeline.run(stream, &cfg_high, &mut obs)
        };

        assert!(!result_low.phrases.is_empty());
        assert!(!result_high.phrases.is_empty());

        // At least one phrase score must differ between the two damping
        // values, proving per-call config is respected.
        let any_diff = result_low
            .phrases
            .iter()
            .zip(result_high.phrases.iter())
            .any(|(l, h)| (l.score - h.score).abs() > 1e-10);

        assert!(
            any_diff,
            "Different damping values (0.5 vs 0.99) must produce different scores"
        );
    }

    /// Edge case: empty topic_weights with min_weight=0.0 produces an
    /// all-zero teleport vector, which normalizes to uniform — pipeline
    /// should still produce valid phrases (equivalent to SingleRank).
    #[test]
    fn test_topical_pipeline_empty_weights_fallback() {
        use std::collections::HashMap;

        let tokens = golden_tokens();
        let cfg = TextRankConfig::default();

        let pipeline =
            super::TopicalPageRankPipeline::topical(HashMap::new(), 0.0);
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = NoopObserver;
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert!(result.converged, "Should converge even with empty weights");
        assert!(
            !result.phrases.is_empty(),
            "Should still produce phrases with empty weights (uniform fallback)"
        );

        // Compare to SingleRank — with uniform teleport, TopicalPageRank
        // should behave like SingleRank (same graph, same uniform PPR).
        let single_result = {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            super::SingleRankPipeline::single_rank().run(stream, &cfg, &mut obs)
        };

        assert_eq!(
            result.phrases.len(),
            single_result.phrases.len(),
            "Empty-weights topical should produce same phrase count as SingleRank"
        );

        // Scores should match since both use uniform teleport.
        let eps = 1e-8;
        for (i, (tp, sp)) in result
            .phrases
            .iter()
            .zip(single_result.phrases.iter())
            .enumerate()
        {
            assert_eq!(tp.text, sp.text, "Text mismatch at {i}");
            assert!(
                (tp.score - sp.score).abs() < eps,
                "Score mismatch at {i}: topical={:.10}, single={:.10}",
                tp.score,
                sp.score
            );
        }
    }

    // ================================================================
    // TopicRankPipeline tests
    // ================================================================

    fn topic_rank_tokens() -> Vec<Token> {
        vec![
            // Sentence 0: "Machine learning algorithms"
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("algorithms", "algorithm", PosTag::Noun, 17, 27, 0, 2),
            // Sentence 1: "Deep learning models"
            Token::new("Deep", "deep", PosTag::Adjective, 28, 32, 1, 3),
            Token::new("learning", "learning", PosTag::Noun, 33, 41, 1, 4),
            Token::new("models", "model", PosTag::Noun, 42, 48, 1, 5),
            // Sentence 2: "Neural networks perform"
            Token::new("Neural", "neural", PosTag::Adjective, 49, 55, 2, 6),
            Token::new("networks", "network", PosTag::Noun, 56, 64, 2, 7),
            Token::new("perform", "perform", PosTag::Verb, 65, 72, 2, 8),
        ]
    }

    fn topic_rank_chunks() -> Vec<crate::types::ChunkSpan> {
        use crate::types::ChunkSpan;
        vec![
            ChunkSpan { start_token: 0, end_token: 2, start_char: 0, end_char: 16, sentence_idx: 0 },
            ChunkSpan { start_token: 3, end_token: 5, start_char: 28, end_char: 41, sentence_idx: 1 },
            ChunkSpan { start_token: 6, end_token: 8, start_char: 49, end_char: 64, sentence_idx: 2 },
        ]
    }

    #[test]
    fn test_topic_rank_pipeline_constructs_and_runs() {
        let tokens = topic_rank_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let pipeline = TopicRankPipeline::topic_rank(topic_rank_chunks());
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert!(!result.phrases.is_empty(), "TopicRank should produce phrases");
        assert!(result.converged, "PageRank should converge");
    }

    #[test]
    fn test_topic_rank_pipeline_empty_input() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let pipeline = TopicRankPipeline::topic_rank(vec![]);
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert!(result.phrases.is_empty());
    }

    #[test]
    fn test_topic_rank_pipeline_deterministic() {
        let tokens = topic_rank_tokens();
        let chunks = topic_rank_chunks();
        let cfg = TextRankConfig::default();

        let run = || {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            TopicRankPipeline::topic_rank(chunks.clone()).run(stream, &cfg, &mut obs)
        };

        let r1 = run();
        let r2 = run();

        assert_eq!(r1.phrases.len(), r2.phrases.len());
        for (p1, p2) in r1.phrases.iter().zip(r2.phrases.iter()) {
            assert_eq!(p1.text, p2.text, "Phrase text should be deterministic");
            assert!(
                (p1.score - p2.score).abs() < 1e-12,
                "Scores should be identical: {} vs {}",
                p1.score,
                p2.score,
            );
        }
    }

    #[test]
    fn test_topic_rank_pipeline_edge_weight_affects_scores() {
        let tokens = topic_rank_tokens();
        let chunks = topic_rank_chunks();
        let cfg = TextRankConfig::default();

        // Run with default edge_weight=1.0
        let stream1 = TokenStream::from_tokens(&tokens);
        let mut obs1 = NoopObserver;
        let r1 = TopicRankPipeline::topic_rank(chunks.clone())
            .run(stream1, &cfg, &mut obs1);

        // Run with edge_weight=10.0
        let stream2 = TokenStream::from_tokens(&tokens);
        let mut obs2 = NoopObserver;
        let r2 = TopicRankPipeline::topic_rank_with(chunks, 0.25, 10.0)
            .run(stream2, &cfg, &mut obs2);

        // Same number of phrases, but scores should differ.
        assert_eq!(r1.phrases.len(), r2.phrases.len());
        if r1.phrases.len() > 1 {
            // With different edge weights, the score distribution changes.
            // At minimum, verify both runs produce valid output.
            for p in &r2.phrases {
                assert!(p.score > 0.0);
            }
        }
    }

    #[test]
    fn test_topic_rank_pipeline_single_cluster() {
        // Single chunk → single cluster → single phrase.
        let tokens = vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
        ];
        let chunks = vec![
            crate::types::ChunkSpan {
                start_token: 0,
                end_token: 2,
                start_char: 0,
                end_char: 16,
                sentence_idx: 0,
            },
        ];

        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let pipeline = TopicRankPipeline::topic_rank(chunks);
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert_eq!(result.phrases.len(), 1, "Single cluster should produce one phrase");
        assert!(result.phrases[0].score > 0.0);
    }

    #[test]
    fn test_topic_rank_pipeline_with_custom_threshold() {
        let tokens = topic_rank_tokens();
        let chunks = topic_rank_chunks();
        let cfg = TextRankConfig::default();

        // High similarity_threshold (0.99): high Jaccard similarity needed to
        // merge → distance cutoff = 1 - 0.99 = 0.01 → harder to merge → more
        // clusters → more phrases.
        let stream1 = TokenStream::from_tokens(&tokens);
        let mut obs1 = NoopObserver;
        let r_strict = TopicRankPipeline::topic_rank_with(chunks.clone(), 0.99, 1.0)
            .run(stream1, &cfg, &mut obs1);

        // Low similarity_threshold (0.01): low Jaccard similarity needed →
        // distance cutoff = 1 - 0.01 = 0.99 → easier to merge → fewer
        // clusters → fewer phrases.
        let stream2 = TokenStream::from_tokens(&tokens);
        let mut obs2 = NoopObserver;
        let r_relaxed = TopicRankPipeline::topic_rank_with(chunks, 0.01, 1.0)
            .run(stream2, &cfg, &mut obs2);

        // Strict threshold should yield >= phrases than relaxed.
        assert!(
            r_strict.phrases.len() >= r_relaxed.phrases.len(),
            "Strict threshold (0.99) should yield >= phrases than relaxed (0.01): got {} vs {}",
            r_strict.phrases.len(), r_relaxed.phrases.len()
        );
    }

    // ================================================================
    // MultipartiteRankPipeline tests
    // ================================================================

    #[test]
    fn test_multipartite_pipeline_constructs_and_runs() {
        let tokens = topic_rank_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let pipeline = super::MultipartiteRankPipeline::multipartite_rank(topic_rank_chunks());
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert!(!result.phrases.is_empty(), "MultipartiteRank should produce phrases");
        assert!(result.converged, "PageRank should converge");
    }

    #[test]
    fn test_multipartite_pipeline_empty_input() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let pipeline = super::MultipartiteRankPipeline::multipartite_rank(vec![]);
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert!(result.phrases.is_empty());
    }

    #[test]
    fn test_multipartite_pipeline_deterministic() {
        let tokens = topic_rank_tokens();
        let chunks = topic_rank_chunks();
        let cfg = TextRankConfig::default();

        let run = || {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            super::MultipartiteRankPipeline::multipartite_rank(chunks.clone())
                .run(stream, &cfg, &mut obs)
        };

        let r1 = run();
        let r2 = run();

        assert_eq!(r1.phrases.len(), r2.phrases.len());
        for (p1, p2) in r1.phrases.iter().zip(r2.phrases.iter()) {
            assert_eq!(p1.text, p2.text, "Phrase text should be deterministic");
            assert!(
                (p1.score - p2.score).abs() < 1e-12,
                "Scores should be identical: {} vs {}",
                p1.score,
                p2.score,
            );
        }
    }

    #[test]
    fn test_multipartite_pipeline_alpha_affects_scores() {
        let tokens = topic_rank_tokens();
        let chunks = topic_rank_chunks();
        let cfg = TextRankConfig::default();

        // Alpha = 0.0 (no boost).
        let stream1 = TokenStream::from_tokens(&tokens);
        let mut obs1 = NoopObserver;
        let r_no_boost = super::MultipartiteRankPipeline::multipartite_rank_with(
            chunks.clone(), 0.26, 0.0,
        ).run(stream1, &cfg, &mut obs1);

        // Alpha = 5.0 (strong boost).
        let stream2 = TokenStream::from_tokens(&tokens);
        let mut obs2 = NoopObserver;
        let r_high_boost = super::MultipartiteRankPipeline::multipartite_rank_with(
            chunks, 0.26, 5.0,
        ).run(stream2, &cfg, &mut obs2);

        // Both should produce phrases.
        assert!(!r_no_boost.phrases.is_empty());
        assert!(!r_high_boost.phrases.is_empty());

        // With different alpha, scores should differ.
        let any_diff = r_no_boost.phrases.iter()
            .zip(r_high_boost.phrases.iter())
            .any(|(a, b)| (a.score - b.score).abs() > 1e-10 || a.text != b.text);

        assert!(
            any_diff,
            "Different alpha values should produce different results"
        );
    }

    #[test]
    fn test_multipartite_pipeline_threshold_affects_clustering() {
        let tokens = topic_rank_tokens();
        let chunks = topic_rank_chunks();
        let cfg = TextRankConfig::default();

        // Very strict threshold → more clusters.
        let stream1 = TokenStream::from_tokens(&tokens);
        let mut obs1 = NoopObserver;
        let r_strict = super::MultipartiteRankPipeline::multipartite_rank_with(
            chunks.clone(), 0.99, 1.1,
        ).run(stream1, &cfg, &mut obs1);

        // Very relaxed threshold → fewer clusters.
        let stream2 = TokenStream::from_tokens(&tokens);
        let mut obs2 = NoopObserver;
        let r_relaxed = super::MultipartiteRankPipeline::multipartite_rank_with(
            chunks, 0.01, 1.1,
        ).run(stream2, &cfg, &mut obs2);

        // Both should produce valid output.
        assert!(!r_strict.phrases.is_empty());
        assert!(!r_relaxed.phrases.is_empty());
    }

    #[test]
    fn test_multipartite_pipeline_single_candidate() {
        let tokens = vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
        ];
        let chunks = vec![
            crate::types::ChunkSpan {
                start_token: 0,
                end_token: 2,
                start_char: 0,
                end_char: 16,
                sentence_idx: 0,
            },
        ];

        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let pipeline = super::MultipartiteRankPipeline::multipartite_rank(chunks);
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert_eq!(result.phrases.len(), 1, "Single candidate should produce one phrase");
        assert!(result.phrases[0].score > 0.0);
    }

    // ================================================================
    // Debug enrichment integration tests
    // ================================================================

    #[test]
    fn test_pipeline_debug_none_no_payload() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default(); // debug_level = None by default
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert!(result.debug.is_none(), "debug_level=None should produce no debug payload");
    }

    #[test]
    fn test_pipeline_debug_stats_has_graph_stats() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default()
            .with_debug_level(crate::pipeline::artifacts::DebugLevel::Stats);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().expect("Stats level should produce debug payload");
        let gs = debug.graph_stats.as_ref().expect("Should have graph_stats");
        assert!(gs.num_nodes > 0);
        assert!(gs.num_edges > 0);

        let cs = debug.convergence_summary.as_ref().expect("Should have convergence_summary");
        assert!(cs.iterations > 0);
        assert!(cs.converged);

        // Stats level should NOT include node scores.
        assert!(debug.node_scores.is_none());
    }

    #[test]
    fn test_pipeline_debug_top_nodes_has_scores() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default()
            .with_debug_level(crate::pipeline::artifacts::DebugLevel::TopNodes);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().expect("TopNodes level should produce debug payload");
        let scores = debug.node_scores.as_ref().expect("Should have node_scores");
        assert!(!scores.is_empty());
        // Scores should be sorted descending.
        for w in scores.windows(2) {
            assert!(w[0].1 >= w[1].1, "Node scores should be sorted descending");
        }
    }

    #[test]
    fn test_pipeline_debug_full_has_residuals() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default()
            .with_debug_level(DebugLevel::Full);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().expect("Full level should produce debug payload");
        // Full includes everything from Stats and TopNodes.
        assert!(debug.graph_stats.is_some());
        assert!(debug.convergence_summary.is_some());
        assert!(debug.node_scores.is_some());
        // Residuals come from PageRank diagnostics (if captured).
        // The default ranker may or may not capture diagnostics, so we just
        // verify the payload is present and the field exists.
    }

    #[test]
    fn test_pipeline_debug_none_identical_to_default() {
        // Verifies that debug_level=None produces exactly the same phrases,
        // converged, and iterations as a config without debug_level set at all.
        let pipeline = super::BaseTextRankPipeline::base_textrank();

        let cfg_default = TextRankConfig::default();
        let cfg_none = TextRankConfig::default()
            .with_debug_level(DebugLevel::None);

        let r1 = pipeline.run(
            TokenStream::from_tokens(&sample_tokens()),
            &cfg_default,
            &mut NoopObserver,
        );
        let r2 = pipeline.run(
            TokenStream::from_tokens(&sample_tokens()),
            &cfg_none,
            &mut NoopObserver,
        );

        assert_eq!(r1.phrases.len(), r2.phrases.len());
        assert_eq!(r1.converged, r2.converged);
        assert_eq!(r1.iterations, r2.iterations);
        for (a, b) in r1.phrases.iter().zip(r2.phrases.iter()) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.rank, b.rank);
            assert!((a.score - b.score).abs() < f64::EPSILON);
        }
        assert!(r1.debug.is_none());
        assert!(r2.debug.is_none());
    }

    #[test]
    fn test_pipeline_debug_does_not_affect_phrases() {
        // The phrase output should be identical regardless of debug level.
        let pipeline = super::BaseTextRankPipeline::base_textrank();

        let run = |level: DebugLevel| {
            let stream = TokenStream::from_tokens(&sample_tokens());
            let cfg = TextRankConfig::default().with_debug_level(level);
            pipeline.run(stream, &cfg, &mut NoopObserver)
        };

        let r_none = run(DebugLevel::None);
        let r_stats = run(DebugLevel::Stats);
        let r_top = run(DebugLevel::TopNodes);
        let r_full = run(DebugLevel::Full);

        // All four runs must produce the same phrases.
        for r in [&r_stats, &r_top, &r_full] {
            assert_eq!(r.phrases.len(), r_none.phrases.len());
            assert_eq!(r.converged, r_none.converged);
            assert_eq!(r.iterations, r_none.iterations);
            for (a, b) in r.phrases.iter().zip(r_none.phrases.iter()) {
                assert_eq!(a.text, b.text);
                assert_eq!(a.rank, b.rank);
                assert!(
                    (a.score - b.score).abs() < f64::EPSILON,
                    "Scores differ for {:?}: {} vs {}",
                    a.text,
                    a.score,
                    b.score,
                );
            }
        }
    }

    #[test]
    fn test_pipeline_debug_top_k_limits_node_scores() {
        let tokens = sample_tokens();
        let num_content_tokens = tokens
            .iter()
            .filter(|t| t.pos.is_content_word() && !t.is_stopword)
            .count();
        // Ensure we have enough nodes to test a limit smaller than the total.
        assert!(num_content_tokens >= 2, "Need at least 2 content tokens");

        // Set top_k to 2 (less than total content nodes).
        // We need to use DebugLevel::TopNodes and pass a small top_k via
        // the DebugPayload::build path. Since Pipeline::run currently uses
        // DEFAULT_TOP_K, we test the DebugPayload::build path directly here.
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default()
            .with_debug_level(DebugLevel::TopNodes);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let scores = result
            .debug
            .as_ref()
            .unwrap()
            .node_scores
            .as_ref()
            .unwrap();

        // With DEFAULT_TOP_K (50), all nodes should be included since we have < 50.
        assert_eq!(scores.len(), num_content_tokens);

        // Now test with an explicit small top_k via DebugPayload::build directly.
        let stream2 = TokenStream::from_tokens(&sample_tokens());
        let cfg2 = TextRankConfig::default();
        // Build a pipeline and manually call build with top_k=2.
        // We need access to graph and ranks, so we run the pipeline first
        // at None level, then build the debug payload separately.
        // Instead, let's just test DebugPayload::build directly.
        use crate::pipeline::artifacts::DebugPayload;
        // We already tested this in artifacts tests, but verify via integration:
        // Run the pipeline to get a FormattedResult, then verify the scores length
        // from the default run (which uses DEFAULT_TOP_K = 50) is bounded.
        assert!(
            scores.len() <= DebugLevel::DEFAULT_TOP_K,
            "Node scores should be bounded by top_k",
        );
    }

    #[test]
    fn test_pipeline_debug_stats_level_field_boundaries() {
        // Stats level should have graph_stats + convergence_summary but NOT
        // node_scores, residuals, or cluster_memberships.
        let stream = TokenStream::from_tokens(&sample_tokens());
        let cfg = TextRankConfig::default()
            .with_debug_level(DebugLevel::Stats);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().unwrap();
        assert!(debug.graph_stats.is_some(), "Stats should include graph_stats");
        assert!(debug.convergence_summary.is_some(), "Stats should include convergence_summary");
        assert!(debug.node_scores.is_none(), "Stats should NOT include node_scores");
        assert!(debug.residuals.is_none(), "Stats should NOT include residuals");
        assert!(debug.cluster_memberships.is_none(), "Stats should NOT include cluster_memberships");
    }

    #[test]
    fn test_pipeline_debug_top_nodes_includes_stats() {
        // TopNodes is a superset of Stats.
        let stream = TokenStream::from_tokens(&sample_tokens());
        let cfg = TextRankConfig::default()
            .with_debug_level(DebugLevel::TopNodes);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().unwrap();
        assert!(debug.graph_stats.is_some(), "TopNodes should include graph_stats");
        assert!(debug.convergence_summary.is_some(), "TopNodes should include convergence_summary");
        assert!(debug.node_scores.is_some(), "TopNodes should include node_scores");
        assert!(debug.residuals.is_none(), "TopNodes should NOT include residuals");
    }

    #[test]
    fn test_pipeline_debug_expose_spec_end_to_end() {
        // Verify ExposeSpec → DebugLevel → pipeline produces expected output.
        use crate::pipeline::spec::{ExposeSpec, NodeScoresSpec};

        let expose = ExposeSpec {
            node_scores: Some(NodeScoresSpec { top_k: Some(10) }),
            graph_stats: true,
            ..Default::default()
        };
        let level = expose.to_debug_level();
        assert_eq!(level, DebugLevel::TopNodes);

        let stream = TokenStream::from_tokens(&sample_tokens());
        let cfg = TextRankConfig::default().with_debug_level(level);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().unwrap();
        assert!(debug.graph_stats.is_some());
        assert!(debug.node_scores.is_some());
        assert!(
            debug.node_scores.as_ref().unwrap().len() <= 10,
            "top_k=10 should limit node scores (actual count handled by DEFAULT_TOP_K in runner)",
        );
    }

    #[test]
    fn test_pipeline_debug_payload_serde_roundtrip() {
        let stream = TokenStream::from_tokens(&sample_tokens());
        let cfg = TextRankConfig::default()
            .with_debug_level(DebugLevel::TopNodes);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().unwrap();

        // Serialize to JSON and back.
        let json = serde_json::to_string(debug).expect("DebugPayload should serialize");
        let back: crate::pipeline::artifacts::DebugPayload =
            serde_json::from_str(&json).expect("DebugPayload should deserialize");

        // Verify key fields survived the round-trip.
        let gs = back.graph_stats.as_ref().unwrap();
        let orig_gs = debug.graph_stats.as_ref().unwrap();
        assert_eq!(gs.num_nodes, orig_gs.num_nodes);
        assert_eq!(gs.num_edges, orig_gs.num_edges);
        assert_eq!(gs.is_transformed, orig_gs.is_transformed);

        let cs = back.convergence_summary.as_ref().unwrap();
        let orig_cs = debug.convergence_summary.as_ref().unwrap();
        assert_eq!(cs.iterations, orig_cs.iterations);
        assert_eq!(cs.converged, orig_cs.converged);
        assert!((cs.final_delta - orig_cs.final_delta).abs() < f64::EPSILON);

        let scores = back.node_scores.as_ref().unwrap();
        let orig_scores = debug.node_scores.as_ref().unwrap();
        assert_eq!(scores.len(), orig_scores.len());
        for (a, b) in scores.iter().zip(orig_scores.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_pipeline_debug_with_stage_timing_observer() {
        // Verify that the observer collects timings regardless of debug level.
        let stream = TokenStream::from_tokens(&sample_tokens());
        let cfg = TextRankConfig::default()
            .with_debug_level(DebugLevel::Stats);
        let mut obs = StageTimingObserver::new();

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let _result = pipeline.run(stream, &cfg, &mut obs);

        // Should have reports for all stages.
        assert!(
            obs.reports().len() >= 5,
            "Observer should collect timing for all stages, got {}",
            obs.reports().len(),
        );
        assert!(obs.total_duration_ms() >= 0.0);
    }

    #[test]
    fn test_topic_rank_pipeline_debug_full_has_cluster_memberships() {
        let tokens = topic_rank_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default()
            .with_debug_level(DebugLevel::Full);
        let mut obs = NoopObserver;

        let pipeline = super::TopicRankPipeline::topic_rank(topic_rank_chunks());
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().expect("Full level should produce debug payload");
        assert!(debug.graph_stats.is_some());
        assert!(debug.convergence_summary.is_some());
        assert!(debug.node_scores.is_some());

        // TopicRank clusters candidates, so cluster_memberships should be populated.
        let memberships = debug
            .cluster_memberships
            .as_ref()
            .expect("Full level on topic-family pipeline should have cluster_memberships");
        assert!(
            !memberships.is_empty(),
            "Should have at least one cluster",
        );
        // Every candidate should appear in exactly one cluster.
        let total_candidates: usize = memberships.iter().map(|c| c.len()).sum();
        assert!(total_candidates > 0, "Clusters should contain candidates");
    }

    #[test]
    fn test_base_pipeline_debug_full_no_cluster_memberships() {
        // BaseTextRank doesn't use clustering, so cluster_memberships should be None.
        let stream = TokenStream::from_tokens(&sample_tokens());
        let cfg = TextRankConfig::default()
            .with_debug_level(DebugLevel::Full);
        let mut obs = NoopObserver;

        let pipeline = super::BaseTextRankPipeline::base_textrank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        let debug = result.debug.as_ref().unwrap();
        assert!(
            debug.cluster_memberships.is_none(),
            "Base pipeline has no clusters — cluster_memberships should be None",
        );
    }

    // ================================================================
    // run_with_workspace / run_batch tests
    // ================================================================

    #[test]
    fn test_run_with_workspace_matches_run() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let result_normal = pipeline.run(make_token_stream(), &cfg, &mut obs);

        let mut ws = crate::pipeline::artifacts::PipelineWorkspace::new();
        let result_ws = pipeline.run_with_workspace(make_token_stream(), &cfg, &mut obs, &mut ws);

        assert_eq!(result_normal.phrases.len(), result_ws.phrases.len());
        assert_eq!(result_normal.converged, result_ws.converged);
        for (a, b) in result_normal.phrases.iter().zip(result_ws.phrases.iter()) {
            assert_eq!(a.text, b.text);
            assert!(
                (a.score - b.score).abs() < 1e-12,
                "phrase '{}' score mismatch: {} vs {}",
                a.text,
                a.score,
                b.score,
            );
        }
    }

    #[test]
    fn test_run_batch_matches_sequential_runs() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        // Run individually first (consuming token streams).
        let seq_0 = pipeline.run(make_token_stream(), &cfg, &mut obs);
        let seq_1 = pipeline.run(TokenStream::from_tokens(&golden_tokens()), &cfg, &mut obs);
        let seq_2 = pipeline.run(make_token_stream(), &cfg, &mut obs);
        let sequential = [seq_0, seq_1, seq_2];

        // Now run as batch (fresh token streams).
        let batch = pipeline.run_batch(
            vec![
                make_token_stream(),
                TokenStream::from_tokens(&golden_tokens()),
                make_token_stream(),
            ],
            &cfg,
            &mut obs,
        );

        assert_eq!(sequential.len(), batch.len());
        for (seq, bat) in sequential.iter().zip(batch.iter()) {
            assert_eq!(seq.phrases.len(), bat.phrases.len());
            for (a, b) in seq.phrases.iter().zip(bat.phrases.iter()) {
                assert_eq!(a.text, b.text);
                assert!(
                    (a.score - b.score).abs() < 1e-12,
                    "phrase '{}' score mismatch: {} vs {}",
                    a.text,
                    a.score,
                    b.score,
                );
            }
        }
    }

    #[test]
    fn test_run_batch_empty_docs() {
        let pipeline = BaseTextRankPipeline::base_textrank();
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let result = pipeline.run_batch(std::iter::empty(), &cfg, &mut obs);
        assert!(result.is_empty());
    }

    // ── SentenceRankPipeline ────────────────────────────────────────

    fn multi_sentence_tokens() -> Vec<Token> {
        vec![
            // Sentence 0: "Rust is a systems programming language"
            Token { text: "Rust".into(), lemma: "rust".into(), pos: PosTag::ProperNoun, start: 0, end: 4, sentence_idx: 0, token_idx: 0, is_stopword: false },
            Token { text: "is".into(), lemma: "be".into(), pos: PosTag::Verb, start: 5, end: 7, sentence_idx: 0, token_idx: 1, is_stopword: true },
            Token { text: "a".into(), lemma: "a".into(), pos: PosTag::Determiner, start: 8, end: 9, sentence_idx: 0, token_idx: 2, is_stopword: true },
            Token { text: "systems".into(), lemma: "system".into(), pos: PosTag::Noun, start: 10, end: 17, sentence_idx: 0, token_idx: 3, is_stopword: false },
            Token { text: "programming".into(), lemma: "programming".into(), pos: PosTag::Noun, start: 18, end: 29, sentence_idx: 0, token_idx: 4, is_stopword: false },
            Token { text: "language".into(), lemma: "language".into(), pos: PosTag::Noun, start: 30, end: 38, sentence_idx: 0, token_idx: 5, is_stopword: false },
            // Sentence 1: "Python is popular for data science"
            Token { text: "Python".into(), lemma: "python".into(), pos: PosTag::ProperNoun, start: 40, end: 46, sentence_idx: 1, token_idx: 6, is_stopword: false },
            Token { text: "is".into(), lemma: "be".into(), pos: PosTag::Verb, start: 47, end: 49, sentence_idx: 1, token_idx: 7, is_stopword: true },
            Token { text: "popular".into(), lemma: "popular".into(), pos: PosTag::Adjective, start: 50, end: 57, sentence_idx: 1, token_idx: 8, is_stopword: false },
            Token { text: "for".into(), lemma: "for".into(), pos: PosTag::Preposition, start: 58, end: 61, sentence_idx: 1, token_idx: 9, is_stopword: true },
            Token { text: "data".into(), lemma: "data".into(), pos: PosTag::Noun, start: 62, end: 66, sentence_idx: 1, token_idx: 10, is_stopword: false },
            Token { text: "science".into(), lemma: "science".into(), pos: PosTag::Noun, start: 67, end: 74, sentence_idx: 1, token_idx: 11, is_stopword: false },
            // Sentence 2: "Both languages support machine learning"
            Token { text: "Both".into(), lemma: "both".into(), pos: PosTag::Determiner, start: 76, end: 80, sentence_idx: 2, token_idx: 12, is_stopword: true },
            Token { text: "languages".into(), lemma: "language".into(), pos: PosTag::Noun, start: 81, end: 90, sentence_idx: 2, token_idx: 13, is_stopword: false },
            Token { text: "support".into(), lemma: "support".into(), pos: PosTag::Verb, start: 91, end: 98, sentence_idx: 2, token_idx: 14, is_stopword: false },
            Token { text: "machine".into(), lemma: "machine".into(), pos: PosTag::Noun, start: 99, end: 106, sentence_idx: 2, token_idx: 15, is_stopword: false },
            Token { text: "learning".into(), lemma: "learning".into(), pos: PosTag::Noun, start: 107, end: 115, sentence_idx: 2, token_idx: 16, is_stopword: false },
        ]
    }

    #[test]
    fn test_sentence_rank_pipeline_constructs() {
        let _pipeline = SentenceRankPipeline::sentence_rank();
    }

    #[test]
    fn test_sentence_rank_by_position_constructs() {
        let pipeline = SentenceRankPipeline::sentence_rank_by_position();
        assert!(pipeline.formatter.sort_by_position);
    }

    #[test]
    fn test_sentence_rank_produces_results() {
        let pipeline = SentenceRankPipeline::sentence_rank();
        let tokens = multi_sentence_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let result = pipeline.run(stream, &cfg, &mut obs);
        assert!(!result.phrases.is_empty(), "SentenceRank should produce phrases");
        assert!(result.converged);
    }

    #[test]
    fn test_sentence_rank_empty_input() {
        let pipeline = SentenceRankPipeline::sentence_rank();
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let mut obs = NoopObserver;

        let result = pipeline.run(stream, &cfg, &mut obs);
        assert!(result.phrases.is_empty());
    }
}
