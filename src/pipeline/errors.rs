//! Pipeline error types for validation and runtime failures.
//!
//! Two error types cover the full pipeline lifecycle:
//!
//! - [`PipelineSpecError`] — build-time problems found during spec validation
//!   (missing stages, invalid values, incompatible modules, etc.)
//! - [`PipelineRuntimeError`] — failures that occur during pipeline execution
//!   (stage crashes, convergence failures, etc.)
//!
//! Both types carry a stable [`ErrorCode`] for programmatic matching, a JSON
//! pointer `path` for locating the problem in the spec, a human-readable
//! `message`, and an optional `hint` suggesting a fix.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::error_code::ErrorCode;

// ─── Spec (build-time) errors ───────────────────────────────────────────────

/// A validation error found in a pipeline spec before execution begins.
///
/// # Display format
///
/// ```text
/// [missing_stage] /modules/rank: A ranking module is required
/// ```
///
/// # JSON format
///
/// ```json
/// {
///   "code": "missing_stage",
///   "path": "/modules/rank",
///   "message": "A ranking module is required",
///   "hint": "Add a 'rank' entry to the modules section"
/// }
/// ```
#[derive(Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[error("[{code}] {path}: {message}")]
pub struct PipelineSpecError {
    /// Stable error code for programmatic matching.
    pub code: ErrorCode,

    /// JSON pointer into the spec identifying the problematic location.
    ///
    /// Examples: `"/modules/rank/type"`, `"/top_n"`, `""` (root).
    pub path: String,

    /// Human-readable description of the problem.
    pub message: String,

    /// Optional suggestion for how to fix the problem.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl PipelineSpecError {
    /// Create a new spec error.
    pub fn new(
        code: ErrorCode,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
            hint: None,
        }
    }

    /// Attach a hint suggesting how to fix the problem.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

// ─── Runtime (execution-time) errors ────────────────────────────────────────

/// A failure that occurred while executing a pipeline stage.
///
/// # Display format
///
/// ```text
/// [convergence_failed] /modules/rank (stage: rank): PageRank did not converge after 100 iterations
/// ```
///
/// # JSON format
///
/// ```json
/// {
///   "code": "convergence_failed",
///   "path": "/modules/rank",
///   "stage": "rank",
///   "message": "PageRank did not converge after 100 iterations",
///   "hint": "Increase max_iterations or relax the convergence threshold"
/// }
/// ```
#[derive(Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[error("[{code}] {path} (stage: {stage}): {message}")]
pub struct PipelineRuntimeError {
    /// Stable error code for programmatic matching.
    pub code: ErrorCode,

    /// JSON pointer into the spec identifying the stage that failed.
    pub path: String,

    /// Name of the pipeline stage that failed (e.g., `"tokenize"`, `"rank"`).
    pub stage: String,

    /// Human-readable description of the failure.
    pub message: String,

    /// Optional suggestion for how to fix or work around the failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl PipelineRuntimeError {
    /// Create a new runtime error.
    pub fn new(
        code: ErrorCode,
        path: impl Into<String>,
        stage: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            path: path.into(),
            stage: stage.into(),
            message: message.into(),
            hint: None,
        }
    }

    /// Attach a hint suggesting how to fix or work around the failure.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PipelineSpecError ──────────────────────────────────────────────

    #[test]
    fn test_spec_error_display() {
        let err = PipelineSpecError::new(
            ErrorCode::MissingStage,
            "/modules/rank",
            "A ranking module is required",
        );
        assert_eq!(
            err.to_string(),
            "[missing_stage] /modules/rank: A ranking module is required"
        );
    }

    #[test]
    fn test_spec_error_with_hint() {
        let err = PipelineSpecError::new(
            ErrorCode::InvalidValue,
            "/top_n",
            "top_n must be positive",
        )
        .with_hint("Set top_n to a value >= 1");

        assert_eq!(err.hint.as_deref(), Some("Set top_n to a value >= 1"));
    }

    #[test]
    fn test_spec_error_serde_roundtrip() {
        let err = PipelineSpecError::new(
            ErrorCode::LimitExceeded,
            "/top_n",
            "top_n exceeds maximum of 1000",
        )
        .with_hint("Use a value <= 1000");

        let json = serde_json::to_string(&err).unwrap();
        let back: PipelineSpecError = serde_json::from_str(&json).unwrap();
        assert_eq!(back, err);
    }

    #[test]
    fn test_spec_error_json_format() {
        let err = PipelineSpecError::new(
            ErrorCode::UnknownField,
            "/modules/rank/typo_field",
            "Unknown field 'typo_field'",
        );

        let value: serde_json::Value = serde_json::to_value(&err).unwrap();
        assert_eq!(value["code"], "unknown_field");
        assert_eq!(value["path"], "/modules/rank/typo_field");
        assert_eq!(value["message"], "Unknown field 'typo_field'");
        // hint is None → should be absent from JSON
        assert!(value.get("hint").is_none());
    }

    #[test]
    fn test_spec_error_json_with_hint() {
        let err = PipelineSpecError::new(
            ErrorCode::InvalidCombo,
            "/modules",
            "Cannot combine SingleRank with BiasedTextRank",
        )
        .with_hint("Choose one ranking algorithm");

        let value: serde_json::Value = serde_json::to_value(&err).unwrap();
        assert_eq!(value["hint"], "Choose one ranking algorithm");
    }

    #[test]
    fn test_spec_error_is_std_error() {
        let err = PipelineSpecError::new(
            ErrorCode::ValidationFailed,
            "",
            "Spec validation failed",
        );
        // Verify it implements std::error::Error via thiserror
        let _: &dyn std::error::Error = &err;
    }

    // ─── PipelineRuntimeError ───────────────────────────────────────────

    #[test]
    fn test_runtime_error_display() {
        let err = PipelineRuntimeError::new(
            ErrorCode::ConvergenceFailed,
            "/modules/rank",
            "rank",
            "PageRank did not converge after 100 iterations",
        );
        assert_eq!(
            err.to_string(),
            "[convergence_failed] /modules/rank (stage: rank): \
             PageRank did not converge after 100 iterations"
        );
    }

    #[test]
    fn test_runtime_error_with_hint() {
        let err = PipelineRuntimeError::new(
            ErrorCode::StageFailed,
            "/modules/tokenize",
            "tokenize",
            "Tokenizer returned no tokens",
        )
        .with_hint("Check that the input text is not empty");

        assert_eq!(
            err.hint.as_deref(),
            Some("Check that the input text is not empty")
        );
    }

    #[test]
    fn test_runtime_error_serde_roundtrip() {
        let err = PipelineRuntimeError::new(
            ErrorCode::StageFailed,
            "/modules/rank",
            "rank",
            "Internal panic in ranking stage",
        )
        .with_hint("Report this as a bug");

        let json = serde_json::to_string(&err).unwrap();
        let back: PipelineRuntimeError = serde_json::from_str(&json).unwrap();
        assert_eq!(back, err);
    }

    #[test]
    fn test_runtime_error_json_format() {
        let err = PipelineRuntimeError::new(
            ErrorCode::ConvergenceFailed,
            "/modules/rank",
            "rank",
            "Did not converge",
        );

        let value: serde_json::Value = serde_json::to_value(&err).unwrap();
        assert_eq!(value["code"], "convergence_failed");
        assert_eq!(value["path"], "/modules/rank");
        assert_eq!(value["stage"], "rank");
        assert_eq!(value["message"], "Did not converge");
        assert!(value.get("hint").is_none());
    }

    #[test]
    fn test_runtime_error_is_std_error() {
        let err = PipelineRuntimeError::new(
            ErrorCode::StageFailed,
            "/modules/rank",
            "rank",
            "stage failed",
        );
        let _: &dyn std::error::Error = &err;
    }

    // ─── Cross-cutting ──────────────────────────────────────────────────

    #[test]
    fn test_error_code_shared_across_types() {
        let spec = PipelineSpecError::new(
            ErrorCode::InvalidValue,
            "/damping",
            "damping must be in (0, 1)",
        );
        let runtime = PipelineRuntimeError::new(
            ErrorCode::InvalidValue,
            "/damping",
            "rank",
            "damping factor caused numeric instability",
        );

        // Same code, different error types
        assert_eq!(spec.code, runtime.code);
        assert_eq!(spec.code, ErrorCode::InvalidValue);
    }
}
