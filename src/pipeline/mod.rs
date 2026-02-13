//! Pipeline specification, validation, and execution.
//!
//! This module provides the foundation for declarative pipeline configuration,
//! error handling, and (in future phases) modular execution.
//!
//! ## Submodules
//!
//! - [`artifacts`] — First-class typed intermediates flowing between stages
//! - [`traits`] — Stage trait definitions (E3)
//! - [`runner`] — Pipeline orchestration and artifact threading (E4)
//! - [`observer`] — Logging, profiling, and debug hooks (E4)

pub mod artifacts;
pub mod error_code;
pub mod errors;
pub mod observer;
pub mod runner;
pub mod spec;
pub mod spec_builder;
pub mod traits;
pub mod validation;

// Re-export artifact types for convenient access.
pub use artifacts::{
    CandidateKind, CandidateSet, CandidateSetRef, ClusterAssignments, ConvergenceSummary,
    DebugLevel, DebugPayload, FormattedResult, Graph, GraphStats, PhraseCandidate, PhraseEntry,
    PhraseSet, PhraseSetRef, PipelineWorkspace, RankDiagnostics, RankOutput, SentenceCandidate,
    TeleportType, TeleportVector, TokenEntry, TokenStream, TokenStreamRef, WordCandidate,
};

// Re-export observer types.
pub use observer::{
    NoopObserver, PipelineObserver, StageClock, StageReport, StageReportBuilder,
    StageTimingObserver, STAGE_CANDIDATES, STAGE_FORMAT, STAGE_GRAPH, STAGE_GRAPH_TRANSFORM,
    STAGE_PHRASES, STAGE_PREPROCESS, STAGE_RANK, STAGE_TELEPORT,
};

// Re-export runner types (Pipeline, builder, type alias).
pub use runner::{
    BaseTextRankPipeline, BiasedTextRankPipeline, MultipartiteRankPipeline, Pipeline,
    PipelineBuilder, PositionRankPipeline, SentenceRankPipeline, SingleRankPipeline,
    TopicRankPipeline, TopicalPageRankPipeline,
};

// Re-export preset resolution, spec resolution, and module merging.
pub use spec::{merge_modules, resolve_preset, resolve_spec};

// Re-export spec builder types.
pub use spec_builder::{DynPipeline, SpecPipelineBuilder};

// Re-export stage traits and default implementations.
pub use traits::{
    AlphaBoostWeighter, CandidateGraphBuilder, CandidateSelector, ChunkPhraseBuilder, Clusterer,
    CooccurrenceGraphBuilder, EdgeWeightPolicy, FocusTermsTeleportBuilder, GraphBuilder,
    GraphTransform, IntraTopicEdgeRemover, JaccardHacClusterer, Linkage,
    MultipartitePhraseBuilder, MultipartiteTransform, NoopClusterer, NoopGraphTransform,
    NoopPreprocessor, PageRankRanker, PhraseBuilder, PhraseCandidateSelector,
    PositionTeleportBuilder, Preprocessor, Ranker, ResultFormatter, SentenceCandidateSelector,
    SentenceFormatter, SentenceGraphBuilder, SentencePhraseBuilder, StandardResultFormatter,
    TeleportBuilder, TopicGraphBuilder, TopicRepresentativeBuilder, TopicWeightsTeleportBuilder,
    UniformTeleportBuilder, WindowGraphBuilder, WindowStrategy, WordNodeSelector,
    DEFAULT_WINDOW_SIZE,
};
