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
pub mod traits;
pub mod validation;

// Re-export artifact types for convenient access.
pub use artifacts::{
    CandidateKind, CandidateSet, CandidateSetRef, DebugPayload, FormattedResult, Graph,
    GraphStats, PhraseCandidate, PhraseEntry, PhraseSet, PhraseSetRef, PipelineWorkspace,
    RankDiagnostics, RankOutput, TeleportVector, TokenEntry, TokenStream, TokenStreamRef,
    WordCandidate,
};

// Re-export stage traits and default implementations.
pub use traits::{
    CandidateSelector, CooccurrenceGraphBuilder, EdgeWeightPolicy, GraphBuilder, GraphTransform,
    NoopGraphTransform, NoopPreprocessor, PhraseCandidateSelector, Preprocessor, TeleportBuilder,
    UniformTeleportBuilder, WindowStrategy, WordNodeSelector,
};
