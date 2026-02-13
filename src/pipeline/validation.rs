//! Validation engine for pipeline specifications.
//!
//! The engine runs all registered [`ValidationRule`]s against a
//! [`PipelineSpecV1`](super::spec::PipelineSpecV1) and collects every diagnostic
//! into a [`ValidationReport`] — it never short-circuits on the first error,
//! so users see all problems at once.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use rapid_textrank::pipeline::validation::ValidationEngine;
//!
//! let engine = ValidationEngine::with_defaults();
//! let report = engine.validate(&spec);
//! if report.has_errors() {
//!     for err in report.errors() {
//!         eprintln!("{err}");
//!     }
//! }
//! ```

use serde::Serialize;

use super::error_code::ErrorCode;
use super::errors::PipelineSpecError;
use super::spec::*;

// ─── Severity ───────────────────────────────────────────────────────────────

/// Whether a diagnostic is a hard error or a soft warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
}

// ─── Diagnostic ─────────────────────────────────────────────────────────────

/// A single validation finding — an error or warning attached to a
/// [`PipelineSpecError`] that carries the code, path, message, and hint.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationDiagnostic {
    pub severity: Severity,
    #[serde(flatten)]
    pub error: PipelineSpecError,
}

impl ValidationDiagnostic {
    pub fn error(err: PipelineSpecError) -> Self {
        Self {
            severity: Severity::Error,
            error: err,
        }
    }

    pub fn warning(err: PipelineSpecError) -> Self {
        Self {
            severity: Severity::Warning,
            error: err,
        }
    }
}

// ─── Report ─────────────────────────────────────────────────────────────────

/// Collected diagnostics from running all validation rules.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ValidationReport {
    pub diagnostics: Vec<ValidationDiagnostic>,
}

impl ValidationReport {
    /// Iterate over error-severity diagnostics.
    pub fn errors(&self) -> impl Iterator<Item = &PipelineSpecError> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| &d.error)
    }

    /// Iterate over warning-severity diagnostics.
    pub fn warnings(&self) -> impl Iterator<Item = &PipelineSpecError> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .map(|d| &d.error)
    }

    /// Returns `true` if any diagnostic is an error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    /// Returns `true` if there are no errors (warnings are acceptable).
    pub fn is_valid(&self) -> bool {
        !self.has_errors()
    }

    /// Total number of diagnostics (errors + warnings).
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Returns `true` if there are no diagnostics at all.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

// ─── Rule trait ─────────────────────────────────────────────────────────────

/// A single validation rule that inspects a [`PipelineSpecV1`] and returns
/// zero or more diagnostics.
///
/// Rules are stateless and must be `Send + Sync` so they can be shared
/// across threads (e.g., in a long-lived validation engine).
pub trait ValidationRule: Send + Sync {
    /// Short, stable identifier for this rule (e.g., `"rank_teleport"`).
    fn name(&self) -> &str;

    /// Inspect `spec` and return any findings.
    fn validate(&self, spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic>;
}

// ─── Engine ─────────────────────────────────────────────────────────────────

/// Runs a set of [`ValidationRule`]s against a [`PipelineSpecV1`] and collects
/// all diagnostics into a [`ValidationReport`].
pub struct ValidationEngine {
    rules: Vec<Box<dyn ValidationRule>>,
}

impl ValidationEngine {
    /// Create an empty engine with no rules.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create an engine pre-loaded with the default rule set.
    pub fn with_defaults() -> Self {
        let mut engine = Self::new();
        engine.add_rule(Box::new(RankTeleportRule));
        engine.add_rule(Box::new(TopicGraphDepsRule));
        engine.add_rule(Box::new(GraphTransformDepsRule));
        engine.add_rule(Box::new(RuntimeLimitsRule));
        engine.add_rule(Box::new(UnknownFieldsRule));
        engine
    }

    /// Register an additional rule.
    pub fn add_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.rules.push(rule);
    }

    /// Run all rules against `spec` and return the collected report.
    pub fn validate(&self, spec: &PipelineSpecV1) -> ValidationReport {
        let mut report = ValidationReport::default();
        for rule in &self.rules {
            report.diagnostics.extend(rule.validate(spec));
        }
        report
    }
}

impl Default for ValidationEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Concrete rules
// ═══════════════════════════════════════════════════════════════════════════

// ─── 1. personalized_pagerank requires teleport ─────────────────────────────

struct RankTeleportRule;

impl ValidationRule for RankTeleportRule {
    fn name(&self) -> &str {
        "rank_teleport"
    }

    fn validate(&self, spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic> {
        let is_personalized =
            matches!(spec.modules.rank, Some(RankSpec::PersonalizedPagerank { .. }));

        if is_personalized && spec.modules.teleport.is_none() {
            vec![ValidationDiagnostic::error(
                PipelineSpecError::new(
                    ErrorCode::MissingStage,
                    "/modules/teleport",
                    "personalized_pagerank requires a teleport module",
                )
                .with_hint(
                    "Add a teleport module: position, focus_terms, \
                     topic_weights, or uniform",
                ),
            )]
        } else {
            vec![]
        }
    }
}

// ─── 2. topic_graph / candidate_graph require clustering + phrase_candidates ─

struct TopicGraphDepsRule;

impl ValidationRule for TopicGraphDepsRule {
    fn name(&self) -> &str {
        "topic_graph_deps"
    }

    fn validate(&self, spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic> {
        let graph = match &spec.modules.graph {
            Some(g) if matches!(g, GraphSpec::TopicGraph | GraphSpec::CandidateGraph) => g,
            _ => return vec![],
        };

        let mut out = Vec::new();

        if spec.modules.clustering.is_none() {
            out.push(ValidationDiagnostic::error(
                PipelineSpecError::new(
                    ErrorCode::MissingStage,
                    "/modules/clustering",
                    format!("{} requires a clustering module", graph.type_name()),
                )
                .with_hint("Add clustering: \"hac\""),
            ));
        }

        if !matches!(spec.modules.candidates, Some(CandidatesSpec::PhraseCandidates)) {
            out.push(ValidationDiagnostic::error(
                PipelineSpecError::new(
                    ErrorCode::InvalidCombo,
                    "/modules/candidates",
                    format!(
                        "{} requires phrase_candidates, not word_nodes",
                        graph.type_name()
                    ),
                )
                .with_hint("Set candidates to \"phrase_candidates\""),
            ));
        }

        out
    }
}

// ─── 3. remove_intra_cluster_edges requires clustering ──────────────────────

struct GraphTransformDepsRule;

impl ValidationRule for GraphTransformDepsRule {
    fn name(&self) -> &str {
        "graph_transform_deps"
    }

    fn validate(&self, spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic> {
        let needs_clusters = spec
            .modules
            .graph_transforms
            .iter()
            .any(|t| matches!(t, GraphTransformSpec::RemoveIntraClusterEdges));

        if needs_clusters && spec.modules.clustering.is_none() {
            vec![ValidationDiagnostic::error(
                PipelineSpecError::new(
                    ErrorCode::MissingStage,
                    "/modules/clustering",
                    "remove_intra_cluster_edges requires a clustering module",
                )
                .with_hint("Add clustering: \"hac\""),
            )]
        } else {
            vec![]
        }
    }
}

// ─── 4. Runtime limits must be positive when set ────────────────────────────

struct RuntimeLimitsRule;

impl ValidationRule for RuntimeLimitsRule {
    fn name(&self) -> &str {
        "runtime_limits"
    }

    fn validate(&self, spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic> {
        let mut out = Vec::new();

        let checks: &[(&str, Option<usize>)] = &[
            ("max_tokens", spec.runtime.max_tokens),
            ("max_nodes", spec.runtime.max_nodes),
            ("max_edges", spec.runtime.max_edges),
            ("max_threads", spec.runtime.max_threads),
        ];

        for &(field, value) in checks {
            if value == Some(0) {
                out.push(ValidationDiagnostic::error(
                    PipelineSpecError::new(
                        ErrorCode::LimitExceeded,
                        format!("/runtime/{field}"),
                        format!("{field} must be greater than 0"),
                    )
                    .with_hint(format!("Remove {field} to disable the limit, or set it to a positive value")),
                ));
            }
        }

        out
    }
}

// ─── 5. Unknown fields (strict → error, non-strict → warning) ──────────────

struct UnknownFieldsRule;

impl UnknownFieldsRule {
    /// Collect unknown-field diagnostics at the given JSON pointer `path`
    /// from a `HashMap` of extra fields captured by `#[serde(flatten)]`.
    fn check_unknowns(
        path: &str,
        unknowns: &std::collections::HashMap<String, serde_json::Value>,
        strict: bool,
    ) -> Vec<ValidationDiagnostic> {
        unknowns
            .keys()
            .map(|key| {
                let diag_fn = if strict {
                    ValidationDiagnostic::error
                } else {
                    ValidationDiagnostic::warning
                };
                diag_fn(
                    PipelineSpecError::new(
                        ErrorCode::UnknownField,
                        format!("{path}/{key}"),
                        format!("unrecognized field \"{key}\""),
                    )
                    .with_hint("Check spelling or remove this field"),
                )
            })
            .collect()
    }
}

impl ValidationRule for UnknownFieldsRule {
    fn name(&self) -> &str {
        "unknown_fields"
    }

    fn validate(&self, spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic> {
        let mut out = Vec::new();
        out.extend(Self::check_unknowns("", &spec.unknown_fields, spec.strict));
        out.extend(Self::check_unknowns(
            "/modules",
            &spec.modules.unknown_fields,
            spec.strict,
        ));
        out.extend(Self::check_unknowns(
            "/runtime",
            &spec.runtime.unknown_fields,
            spec.strict,
        ));
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a PipelineSpecV1 from JSON.
    fn spec(json: &str) -> PipelineSpecV1 {
        serde_json::from_str(json).unwrap()
    }

    fn engine() -> ValidationEngine {
        ValidationEngine::with_defaults()
    }

    // ─── Valid specs ────────────────────────────────────────────────────

    #[test]
    fn test_minimal_spec_is_valid() {
        let report = engine().validate(&spec(r#"{ "v": 1 }"#));
        assert!(report.is_valid());
        assert!(report.is_empty());
    }

    #[test]
    fn test_standard_pagerank_without_teleport_is_valid() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": { "type": "standard_pagerank" } } }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_personalized_with_teleport_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "rank": { "type": "personalized_pagerank" },
                    "teleport": { "type": "position" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_topic_graph_with_all_deps_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "phrase_candidates" },
                    "graph": { "type": "topic_graph" },
                    "clustering": { "type": "hac" },
                    "rank": { "type": "standard_pagerank" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_candidate_graph_with_all_deps_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "phrase_candidates" },
                    "graph": { "type": "candidate_graph" },
                    "clustering": { "type": "hac" },
                    "graph_transforms": [{ "type": "remove_intra_cluster_edges" }],
                    "rank": { "type": "standard_pagerank" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_runtime_limits_positive_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "runtime": { "max_tokens": 100000, "max_nodes": 50000, "max_edges": 1000000 }
            }"#,
        ));
        assert!(report.is_valid());
    }

    // ─── Rule: rank_teleport ────────────────────────────────────────────

    #[test]
    fn test_personalized_without_teleport_fails() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": { "type": "personalized_pagerank" } } }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::MissingStage);
        assert_eq!(errs[0].path, "/modules/teleport");
    }

    #[test]
    fn test_personalized_with_uniform_teleport_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "rank": { "type": "personalized_pagerank" },
                    "teleport": { "type": "uniform" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    // ─── Rule: topic_graph_deps ─────────────────────────────────────────

    #[test]
    fn test_topic_graph_without_clustering_fails() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "phrase_candidates" },
                    "graph": { "type": "topic_graph" }
                }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::MissingStage);
        assert!(errs[0].path.contains("clustering"));
    }

    #[test]
    fn test_topic_graph_without_phrase_candidates_fails() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "word_nodes" },
                    "graph": { "type": "topic_graph" },
                    "clustering": { "type": "hac" }
                }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::InvalidCombo);
    }

    #[test]
    fn test_topic_graph_missing_both_deps_reports_two_errors() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "graph": { "type": "topic_graph" } } }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_candidate_graph_has_same_deps() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "graph": { "type": "candidate_graph" } } }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_cooccurrence_window_needs_no_clustering() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "graph": { "type": "cooccurrence_window" },
                    "candidates": { "type": "word_nodes" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    // ─── Rule: graph_transform_deps ─────────────────────────────────────

    #[test]
    fn test_intra_cluster_removal_without_clustering_fails() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "graph_transforms": [{ "type": "remove_intra_cluster_edges" }]
                }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::MissingStage);
    }

    #[test]
    fn test_alpha_boost_without_clustering_is_valid() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": { "graph_transforms": [{ "type": "alpha_boost" }] }
            }"#,
        ));
        // alpha_boost doesn't require clustering (it's a weight modifier)
        assert!(
            !report
                .errors()
                .any(|e| e.message.contains("remove_intra_cluster"))
        );
    }

    // ─── Rule: runtime_limits ───────────────────────────────────────────

    #[test]
    fn test_zero_max_tokens_fails() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "runtime": { "max_tokens": 0 } }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::LimitExceeded);
        assert!(errs[0].path.contains("max_tokens"));
    }

    #[test]
    fn test_zero_max_nodes_and_edges_reports_two_errors() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "runtime": { "max_nodes": 0, "max_edges": 0 } }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_absent_limits_are_fine() {
        let report = engine().validate(&spec(r#"{ "v": 1, "runtime": {} }"#));
        assert!(report.is_valid());
    }

    // ─── Rule: unknown_fields (strict mode) ─────────────────────────────

    #[test]
    fn test_unknown_fields_non_strict_are_warnings() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": false, "bogus": 42 }"#,
        ));
        assert!(report.is_valid()); // warnings don't make it invalid
        let warns: Vec<_> = report.warnings().collect();
        assert_eq!(warns.len(), 1);
        assert_eq!(warns[0].code, ErrorCode::UnknownField);
        assert!(warns[0].path.contains("bogus"));
    }

    #[test]
    fn test_unknown_fields_strict_are_errors() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": true, "bogus": 42 }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].code, ErrorCode::UnknownField);
    }

    #[test]
    fn test_unknown_module_field_strict() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": true,
                "modules": { "bogus_module": "xyz" }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].path.contains("bogus_module"));
    }

    #[test]
    fn test_unknown_runtime_field_strict() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": true,
                "runtime": { "bogus_runtime_field": 42 }
            }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].path.contains("bogus_runtime_field"));
    }

    #[test]
    fn test_max_threads_valid() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": true, "runtime": { "max_threads": 8 } }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_max_threads_zero_fails() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "runtime": { "max_threads": 0 } }"#,
        ));
        assert!(report.has_errors());
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].path.contains("max_threads"));
    }

    #[test]
    fn test_no_unknown_fields_clean() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": true, "modules": { "rank": { "type": "standard_pagerank" } } }"#,
        ));
        assert!(report.is_empty());
    }

    // ─── Report helpers ─────────────────────────────────────────────────

    #[test]
    fn test_report_len_and_empty() {
        let report = engine().validate(&spec(r#"{ "v": 1 }"#));
        assert_eq!(report.len(), 0);
        assert!(report.is_empty());

        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": { "type": "personalized_pagerank" } } }"#,
        ));
        assert_eq!(report.len(), 1);
        assert!(!report.is_empty());
    }

    #[test]
    fn test_multiple_rules_fire_independently() {
        // personalized without teleport + zero max_tokens + unknown field strict
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": true,
                "bogus": true,
                "modules": { "rank": { "type": "personalized_pagerank" } },
                "runtime": { "max_tokens": 0 }
            }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 3);
    }

    // ─── Engine: custom rules ───────────────────────────────────────────

    #[test]
    fn test_custom_rule() {
        struct AlwaysWarnRule;
        impl ValidationRule for AlwaysWarnRule {
            fn name(&self) -> &str {
                "always_warn"
            }
            fn validate(&self, _spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic> {
                vec![ValidationDiagnostic::warning(PipelineSpecError::new(
                    ErrorCode::ValidationFailed,
                    "",
                    "custom warning",
                ))]
            }
        }

        let mut eng = ValidationEngine::new();
        eng.add_rule(Box::new(AlwaysWarnRule));
        let report = eng.validate(&spec(r#"{ "v": 1 }"#));
        assert!(report.is_valid()); // warnings only
        assert_eq!(report.warnings().count(), 1);
    }

    // ─── Serialization ──────────────────────────────────────────────────

    #[test]
    fn test_report_serializes_to_json() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": { "type": "personalized_pagerank" } } }"#,
        ));
        let json = serde_json::to_value(&report).unwrap();
        let diags = json["diagnostics"].as_array().unwrap();
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0]["severity"], "error");
        assert_eq!(diags[0]["code"], "missing_stage");
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Comprehensive validation tests (zvl.5)
    // ═════════════════════════════════════════════════════════════════════

    // ─── Exact path + hint verification (public contract) ───────────────

    #[test]
    fn test_rank_teleport_path_and_hint() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": { "type": "personalized_pagerank" } } }"#,
        ));
        let err = report.errors().next().unwrap();
        assert_eq!(err.path, "/modules/teleport");
        assert_eq!(err.code, ErrorCode::MissingStage);
        let hint = err.hint.as_deref().unwrap();
        assert!(hint.contains("position"), "hint should mention position teleport: {hint}");
        assert!(hint.contains("focus_terms"), "hint should mention focus_terms: {hint}");
        assert!(hint.contains("uniform"), "hint should mention uniform: {hint}");
    }

    #[test]
    fn test_topic_graph_clustering_path_and_hint() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "phrase_candidates" },
                    "graph": { "type": "topic_graph" }
                }
            }"#,
        ));
        let err = report.errors().next().unwrap();
        assert_eq!(err.path, "/modules/clustering");
        assert_eq!(err.code, ErrorCode::MissingStage);
        let hint = err.hint.as_deref().unwrap();
        assert!(hint.contains("hac"), "hint should suggest hac clustering: {hint}");
    }

    #[test]
    fn test_topic_graph_candidates_path_and_hint() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "word_nodes" },
                    "graph": { "type": "topic_graph" },
                    "clustering": { "type": "hac" }
                }
            }"#,
        ));
        let err = report.errors().next().unwrap();
        assert_eq!(err.path, "/modules/candidates");
        assert_eq!(err.code, ErrorCode::InvalidCombo);
        let hint = err.hint.as_deref().unwrap();
        assert!(
            hint.contains("phrase_candidates"),
            "hint should suggest phrase_candidates: {hint}"
        );
    }

    #[test]
    fn test_graph_transform_deps_path_and_hint() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": { "graph_transforms": [{ "type": "remove_intra_cluster_edges" }] }
            }"#,
        ));
        let err = report.errors().next().unwrap();
        assert_eq!(err.path, "/modules/clustering");
        assert_eq!(err.code, ErrorCode::MissingStage);
        assert!(err.message.contains("remove_intra_cluster_edges"));
        let hint = err.hint.as_deref().unwrap();
        assert!(hint.contains("hac"), "hint should suggest hac: {hint}");
    }

    #[test]
    fn test_runtime_limit_paths_are_specific() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "runtime": { "max_tokens": 0, "max_nodes": 0, "max_edges": 0 } }"#,
        ));
        let paths: Vec<_> = report.errors().map(|e| e.path.clone()).collect();
        assert!(paths.contains(&"/runtime/max_tokens".to_string()));
        assert!(paths.contains(&"/runtime/max_nodes".to_string()));
        assert!(paths.contains(&"/runtime/max_edges".to_string()));
    }

    #[test]
    fn test_runtime_limit_hints_mention_removal() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "runtime": { "max_tokens": 0 } }"#,
        ));
        let err = report.errors().next().unwrap();
        let hint = err.hint.as_deref().unwrap();
        assert!(
            hint.contains("Remove") || hint.contains("positive"),
            "hint should suggest removal or positive value: {hint}"
        );
    }

    #[test]
    fn test_unknown_field_path_includes_field_name() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": true, "bogus_field": 42 }"#,
        ));
        let err = report.errors().next().unwrap();
        assert_eq!(err.path, "/bogus_field");
        assert_eq!(err.code, ErrorCode::UnknownField);
        assert!(err.message.contains("bogus_field"));
    }

    #[test]
    fn test_unknown_field_hint_suggests_removal() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": true, "typo": 1 }"#,
        ));
        let err = report.errors().next().unwrap();
        let hint = err.hint.as_deref().unwrap();
        assert!(
            hint.contains("remove") || hint.contains("spelling"),
            "hint should suggest fix: {hint}"
        );
    }

    // ─── Realistic variant compositions (valid specs) ───────────────────

    #[test]
    fn test_base_textrank_composition() {
        // BaseTextRank: word_nodes + cooccurrence + standard pagerank
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "word_nodes" },
                    "graph": { "type": "cooccurrence_window" },
                    "rank": { "type": "standard_pagerank" }
                }
            }"#,
        ));
        assert!(report.is_valid());
        assert!(report.is_empty());
    }

    #[test]
    fn test_position_rank_composition() {
        // PositionRank: word_nodes + cooccurrence + position teleport + personalized
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "word_nodes" },
                    "graph": { "type": "cooccurrence_window" },
                    "teleport": { "type": "position" },
                    "rank": { "type": "personalized_pagerank" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_biased_textrank_composition() {
        // BiasedTextRank: word_nodes + cooccurrence + focus_terms teleport + personalized
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "word_nodes" },
                    "graph": { "type": "cooccurrence_window" },
                    "teleport": { "type": "focus_terms" },
                    "rank": { "type": "personalized_pagerank" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_topical_pagerank_composition() {
        // TopicalPageRank: word_nodes + cooccurrence + topic_weights teleport + personalized
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "word_nodes" },
                    "graph": { "type": "cooccurrence_window" },
                    "teleport": { "type": "topic_weights" },
                    "rank": { "type": "personalized_pagerank" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_topic_rank_composition() {
        // TopicRank: phrase_candidates + clustering + topic_graph + standard
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "phrase_candidates" },
                    "clustering": { "type": "hac" },
                    "graph": { "type": "topic_graph" },
                    "rank": { "type": "standard_pagerank" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_multipartite_rank_composition() {
        // MultipartiteRank: phrase_candidates + clustering + candidate_graph
        // + remove_intra_cluster_edges + alpha_boost + standard
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "phrase_candidates" },
                    "clustering": { "type": "hac" },
                    "graph": { "type": "candidate_graph" },
                    "graph_transforms": [{ "type": "remove_intra_cluster_edges" }, { "type": "alpha_boost" }],
                    "rank": { "type": "standard_pagerank" }
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_novel_combination_singlerank_graph_with_position_teleport() {
        // A novel mix not matching any existing variant:
        // cooccurrence_window (could be cross-sentence) + position teleport
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "word_nodes" },
                    "graph": { "type": "cooccurrence_window" },
                    "teleport": { "type": "position" },
                    "rank": { "type": "personalized_pagerank" }
                },
                "runtime": { "max_tokens": 200000 }
            }"#,
        ));
        assert!(report.is_valid());
    }

    // ─── All teleport types accepted with personalized ──────────────────

    #[test]
    fn test_all_teleport_types_valid_with_personalized() {
        for teleport in &["uniform", "position", "focus_terms", "topic_weights"] {
            let json = format!(
                r#"{{
                    "v": 1,
                    "modules": {{
                        "rank": {{ "type": "personalized_pagerank" }},
                        "teleport": {{ "type": "{teleport}" }}
                    }}
                }}"#
            );
            let report = engine().validate(&spec(&json));
            assert!(
                report.is_valid(),
                "teleport={teleport} should be valid with personalized_pagerank"
            );
        }
    }

    // ─── module_unavailable via custom rule ──────────────────────────────

    #[test]
    fn test_module_unavailable_custom_rule() {
        // Demonstrates how a module-availability rule works:
        // check that requested modules are in a set of available ones.
        struct ModuleAvailabilityRule {
            available_rank_types: Vec<String>,
        }

        impl ValidationRule for ModuleAvailabilityRule {
            fn name(&self) -> &str {
                "module_availability"
            }
            fn validate(&self, spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic> {
                if let Some(rank) = &spec.modules.rank {
                    if !self.available_rank_types.iter().any(|t| t == rank.type_name()) {
                        return vec![ValidationDiagnostic::error(
                            PipelineSpecError::new(
                                ErrorCode::ModuleUnavailable,
                                "/modules/rank",
                                format!(
                                    "rank module {} is not available in this build",
                                    rank.type_name()
                                ),
                            )
                            .with_hint(
                                "Check available modules with capabilities() or use a different rank module",
                            ),
                        )];
                    }
                }
                vec![]
            }
        }

        // Engine where only standard_pagerank is "available"
        let mut eng = ValidationEngine::new();
        eng.add_rule(Box::new(ModuleAvailabilityRule {
            available_rank_types: vec!["standard_pagerank".to_string()],
        }));

        // Requesting personalized → unavailable
        let report = eng.validate(&spec(
            r#"{ "v": 1, "modules": { "rank": { "type": "personalized_pagerank" } } }"#,
        ));
        assert!(report.has_errors());
        let err = report.errors().next().unwrap();
        assert_eq!(err.code, ErrorCode::ModuleUnavailable);
        assert_eq!(err.path, "/modules/rank");
        assert!(err.hint.as_deref().unwrap().contains("capabilities"));

        // Requesting standard → available
        let report = eng.validate(&spec(
            r#"{ "v": 1, "modules": { "rank": { "type": "standard_pagerank" } } }"#,
        ));
        assert!(report.is_valid());
    }

    // ─── Maximum error accumulation (all rules fire) ────────────────────

    #[test]
    fn test_worst_case_spec_triggers_all_rules() {
        // A spec that violates every rule at once:
        // - personalized without teleport (rank_teleport)
        // - topic_graph without clustering or phrase_candidates (topic_graph_deps × 2)
        // - remove_intra_cluster_edges without clustering (graph_transform_deps)
        // - zero max_tokens (runtime_limits)
        // - unknown field in strict mode (unknown_fields)
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": true,
                "bogus": true,
                "modules": {
                    "rank": { "type": "personalized_pagerank" },
                    "graph": { "type": "topic_graph" },
                    "graph_transforms": [{ "type": "remove_intra_cluster_edges" }]
                },
                "runtime": { "max_tokens": 0 }
            }"#,
        ));

        let errs: Vec<_> = report.errors().collect();

        // Collect all error codes present
        let codes: Vec<_> = errs.iter().map(|e| e.code).collect();
        assert!(codes.contains(&ErrorCode::MissingStage)); // teleport + clustering
        assert!(codes.contains(&ErrorCode::InvalidCombo)); // candidates mismatch
        assert!(codes.contains(&ErrorCode::LimitExceeded)); // max_tokens=0
        assert!(codes.contains(&ErrorCode::UnknownField)); // bogus field

        // Should be at least 5 errors (teleport, clustering, candidates, transform, limit, unknown)
        assert!(
            errs.len() >= 5,
            "Expected at least 5 errors, got {}: {:?}",
            errs.len(),
            codes
        );
    }

    // ─── Error code coverage: every code used by default rules ──────────

    #[test]
    fn test_missing_stage_code_used() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "rank": { "type": "personalized_pagerank" } } }"#,
        ));
        assert!(report.errors().any(|e| e.code == ErrorCode::MissingStage));
    }

    #[test]
    fn test_invalid_combo_code_used() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "word_nodes" },
                    "graph": { "type": "topic_graph" },
                    "clustering": { "type": "hac" }
                }
            }"#,
        ));
        assert!(report.errors().any(|e| e.code == ErrorCode::InvalidCombo));
    }

    #[test]
    fn test_limit_exceeded_code_used() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "runtime": { "max_edges": 0 } }"#,
        ));
        assert!(report.errors().any(|e| e.code == ErrorCode::LimitExceeded));
    }

    #[test]
    fn test_unknown_field_code_used() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "strict": true, "nope": 1 }"#,
        ));
        assert!(report.errors().any(|e| e.code == ErrorCode::UnknownField));
    }

    // ─── Strict vs non-strict with multiple unknowns ────────────────────

    #[test]
    fn test_multiple_unknown_fields_all_reported() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": true,
                "foo": 1,
                "bar": 2,
                "modules": { "baz": 3 }
            }"#,
        ));
        let errs: Vec<_> = report.errors().collect();
        assert_eq!(errs.len(), 3); // foo, bar, baz
        let paths: Vec<_> = errs.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"/foo"));
        assert!(paths.contains(&"/bar"));
        assert!(paths.contains(&"/modules/baz"));
    }

    #[test]
    fn test_non_strict_unknown_fields_are_all_warnings() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "strict": false,
                "a": 1,
                "b": 2
            }"#,
        ));
        assert!(report.is_valid()); // no errors
        assert_eq!(report.warnings().count(), 2);
    }

    // ─── Spec deserialization edge cases ─────────────────────────────────

    #[test]
    fn test_empty_modules_is_valid() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": {} }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_empty_graph_transforms_is_valid() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "graph_transforms": [] } }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_spec_with_preset_is_valid() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "preset": "textrank" }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_multiple_graph_transforms() {
        // Both transforms present with clustering → valid
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "phrase_candidates" },
                    "clustering": { "type": "hac" },
                    "graph": { "type": "candidate_graph" },
                    "graph_transforms": [{ "type": "remove_intra_cluster_edges" }, { "type": "alpha_boost" }]
                }
            }"#,
        ));
        assert!(report.is_valid());
    }

    #[test]
    fn test_candidate_graph_message_names_graph_type() {
        let report = engine().validate(&spec(
            r#"{
                "v": 1,
                "modules": {
                    "candidates": { "type": "phrase_candidates" },
                    "graph": { "type": "candidate_graph" }
                }
            }"#,
        ));
        let err = report.errors().next().unwrap();
        assert!(
            err.message.contains("candidate_graph"),
            "message should name the graph type: {}",
            err.message
        );
    }

    #[test]
    fn test_topic_graph_message_names_graph_type() {
        let report = engine().validate(&spec(
            r#"{ "v": 1, "modules": { "graph": { "type": "topic_graph" } } }"#,
        ));
        // First error should mention topic_graph
        let errs: Vec<_> = report.errors().collect();
        assert!(errs.iter().any(|e| e.message.contains("topic_graph")));
    }
}
