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

use crate::pipeline::artifacts::{FormattedResult, TokenStream};
use crate::pipeline::observer::{
    PipelineObserver, StageClock, StageReport, StageReportBuilder, STAGE_CANDIDATES, STAGE_FORMAT,
    STAGE_GRAPH, STAGE_GRAPH_TRANSFORM, STAGE_PHRASES, STAGE_PREPROCESS, STAGE_RANK,
    STAGE_TELEPORT,
};
use crate::pipeline::traits::{
    CandidateSelector, ChunkPhraseBuilder, FocusTermsTeleportBuilder, GraphBuilder, GraphTransform,
    NoopGraphTransform, NoopPreprocessor, PageRankRanker, PhraseBuilder, PositionTeleportBuilder,
    Preprocessor, Ranker, ResultFormatter, StandardResultFormatter, TeleportBuilder,
    UniformTeleportBuilder, WindowGraphBuilder, WordNodeSelector,
};
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
        mut tokens: TokenStream,
        cfg: &TextRankConfig,
        observer: &mut impl PipelineObserver,
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
        let rank_output = self.ranker.rank(&graph, teleport.as_ref(), cfg);
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

        // Stage 5: Format result
        trace_stage!(STAGE_FORMAT);
        observer.on_stage_start(STAGE_FORMAT);
        let clock = StageClock::start();
        let result = self.formatter.format(&phrases, &rank_output, None, cfg);
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
}
