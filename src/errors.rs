//! Error types for rapid_textrank
//!
//! This module defines the error types used throughout the library.
//! All errors are designed to be informative and actionable.

use thiserror::Error;

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, TextRankError>;

/// Main error type for rapid_textrank
#[derive(Error, Debug, Clone)]
pub enum TextRankError {
    /// Input text is empty or contains no processable content
    #[error("Empty input: {message}")]
    EmptyInput { message: String },

    /// No candidate phrases were found after filtering
    #[error("No candidates found: {message}")]
    NoCandidates { message: String },

    /// PageRank did not converge within the maximum iterations
    /// Note: This is often not fatal - partial results may still be useful
    #[error("Convergence failure after {iterations} iterations (delta={delta:.6})")]
    ConvergenceFailure { iterations: usize, delta: f64 },

    /// Configuration validation failed
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    /// JSON serialization/deserialization error
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// Internal error (should not occur in normal usage)
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl TextRankError {
    /// Create an empty input error
    pub fn empty_input(message: impl Into<String>) -> Self {
        Self::EmptyInput {
            message: message.into(),
        }
    }

    /// Create a no candidates error
    pub fn no_candidates(message: impl Into<String>) -> Self {
        Self::NoCandidates {
            message: message.into(),
        }
    }

    /// Create a convergence failure error
    pub fn convergence_failure(iterations: usize, delta: f64) -> Self {
        Self::ConvergenceFailure { iterations, delta }
    }

    /// Create an invalid config error
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Check if this error indicates non-convergence
    /// (which may still have usable partial results)
    pub fn is_convergence_failure(&self) -> bool {
        matches!(self, Self::ConvergenceFailure { .. })
    }
}

impl From<serde_json::Error> for TextRankError {
    fn from(err: serde_json::Error) -> Self {
        Self::serialization(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TextRankError::empty_input("no text provided");
        assert!(err.to_string().contains("Empty input"));
        assert!(err.to_string().contains("no text provided"));

        let err = TextRankError::convergence_failure(100, 0.001);
        assert!(err.to_string().contains("100 iterations"));
        assert!(err.to_string().contains("0.001"));
    }

    #[test]
    fn test_is_convergence_failure() {
        let err = TextRankError::convergence_failure(100, 0.001);
        assert!(err.is_convergence_failure());

        let err = TextRankError::empty_input("test");
        assert!(!err.is_convergence_failure());
    }
}
