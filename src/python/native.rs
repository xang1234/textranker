//! Native Python interface
//!
//! Direct Python classes for small documents where Python↔Rust
//! overhead is negligible compared to processing time.

use crate::nlp::stopwords::StopwordFilter;
use crate::nlp::tokenizer::Tokenizer;
use crate::phrase::extraction::extract_keyphrases_with_info;
use crate::pipeline::artifacts::{DebugLevel, DebugPayload, TokenStream};
use crate::pipeline::observer::NoopObserver;
#[cfg(feature = "sentence-rank")]
use crate::pipeline::runner::SentenceRankPipeline;
use crate::types::{Phrase, PhraseGrouping, ScoreAggregation, TextRankConfig};
use crate::variants::biased_textrank::BiasedTextRank;
use crate::variants::multipartite_rank::MultipartiteRank;
use crate::variants::position_rank::PositionRank;
use crate::variants::single_rank::SingleRank;
use crate::variants::topical_pagerank::TopicalPageRank;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

/// A phrase extracted by TextRank
#[pyclass(name = "Phrase")]
#[derive(Clone)]
pub struct PyPhrase {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub lemma: String,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub rank: usize,
}

#[pymethods]
impl PyPhrase {
    fn __repr__(&self) -> String {
        format!(
            "Phrase(text='{}', score={:.4}, rank={})",
            self.text, self.score, self.rank
        )
    }

    fn __str__(&self) -> String {
        self.text.clone()
    }
}

impl From<Phrase> for PyPhrase {
    fn from(p: Phrase) -> Self {
        Self {
            text: p.text,
            lemma: p.lemma,
            score: p.score,
            count: p.count,
            rank: p.rank,
        }
    }
}

// ─── Debug / Inspect Python classes ──────────────────────────────────────

/// Graph statistics from the co-occurrence graph.
#[pyclass(name = "GraphStats")]
#[derive(Clone)]
pub struct PyGraphStats {
    #[pyo3(get)]
    pub num_nodes: usize,
    #[pyo3(get)]
    pub num_edges: usize,
    #[pyo3(get)]
    pub avg_degree: f64,
    #[pyo3(get)]
    pub is_transformed: bool,
}

#[pymethods]
impl PyGraphStats {
    fn __repr__(&self) -> String {
        format!(
            "GraphStats(num_nodes={}, num_edges={}, avg_degree={:.2})",
            self.num_nodes, self.num_edges, self.avg_degree
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("num_nodes", self.num_nodes)?;
        d.set_item("num_edges", self.num_edges)?;
        d.set_item("avg_degree", self.avg_degree)?;
        d.set_item("is_transformed", self.is_transformed)?;
        Ok(d)
    }
}

/// Convergence summary from PageRank.
#[pyclass(name = "ConvergenceSummary")]
#[derive(Clone)]
pub struct PyConvergenceSummary {
    #[pyo3(get)]
    pub iterations: u32,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub final_delta: f64,
}

#[pymethods]
impl PyConvergenceSummary {
    fn __repr__(&self) -> String {
        format!(
            "ConvergenceSummary(iterations={}, converged={}, final_delta={:.2e})",
            self.iterations, self.converged, self.final_delta
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("iterations", self.iterations)?;
        d.set_item("converged", self.converged)?;
        d.set_item("final_delta", self.final_delta)?;
        Ok(d)
    }
}

/// Records why a potential phrase was split or rejected during chunking.
#[pyclass(name = "PhraseSplitEvent")]
#[derive(Clone)]
pub struct PyPhraseSplitEvent {
    #[pyo3(get)]
    pub token_range: (usize, usize),
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub reason: String,
}

#[pymethods]
impl PyPhraseSplitEvent {
    fn __repr__(&self) -> String {
        format!(
            "PhraseSplitEvent(text='{}', reason='{}')",
            self.text, self.reason
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("token_range", self.token_range)?;
        d.set_item("text", &self.text)?;
        d.set_item("reason", &self.reason)?;
        Ok(d)
    }
}

/// Records a phrase candidate that was dropped.
#[pyclass(name = "DroppedCandidate")]
#[derive(Clone)]
pub struct PyDroppedCandidate {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub lemma: String,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub token_range: (usize, usize),
    #[pyo3(get)]
    pub reason: String,
}

#[pymethods]
impl PyDroppedCandidate {
    fn __repr__(&self) -> String {
        format!(
            "DroppedCandidate(text='{}', score={:.4}, reason='{}')",
            self.text, self.score, self.reason
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("text", &self.text)?;
        d.set_item("lemma", &self.lemma)?;
        d.set_item("score", self.score)?;
        d.set_item("token_range", self.token_range)?;
        d.set_item("reason", &self.reason)?;
        Ok(d)
    }
}

/// Enriched cluster member with text metadata.
#[pyclass(name = "ClusterMember")]
#[derive(Clone)]
pub struct PyClusterMember {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub lemma: String,
}

#[pymethods]
impl PyClusterMember {
    fn __repr__(&self) -> String {
        format!(
            "ClusterMember(index={}, text='{}', lemma='{}')",
            self.index, self.text, self.lemma
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("index", self.index)?;
        d.set_item("text", &self.text)?;
        d.set_item("lemma", &self.lemma)?;
        Ok(d)
    }
}

/// Cluster detail: a cluster index and its members.
#[pyclass(name = "ClusterDetail")]
#[derive(Clone)]
pub struct PyClusterDetail {
    #[pyo3(get)]
    pub cluster_index: usize,
    #[pyo3(get)]
    pub members: Vec<PyClusterMember>,
}

#[pymethods]
impl PyClusterDetail {
    fn __repr__(&self) -> String {
        format!(
            "ClusterDetail(cluster_index={}, members={})",
            self.cluster_index,
            self.members.len()
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("cluster_index", self.cluster_index)?;
        let members: Vec<_> = self
            .members
            .iter()
            .map(|m| m.to_dict(py))
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("members", members)?;
        Ok(d)
    }
}

/// Debug/inspect payload from the extraction pipeline.
#[pyclass(name = "DebugPayload")]
#[derive(Clone)]
pub struct PyDebugPayload {
    #[pyo3(get)]
    pub graph_stats: Option<PyGraphStats>,
    #[pyo3(get)]
    pub convergence_summary: Option<PyConvergenceSummary>,
    #[pyo3(get)]
    pub node_scores: Option<Vec<(String, f64)>>,
    #[pyo3(get)]
    pub stage_timings: Option<Vec<(String, f64)>>,
    #[pyo3(get)]
    pub residuals: Option<Vec<f64>>,
    #[pyo3(get)]
    pub cluster_memberships: Option<Vec<Vec<usize>>>,
    #[pyo3(get)]
    pub cluster_details: Option<Vec<PyClusterDetail>>,
    #[pyo3(get)]
    pub phrase_diagnostics: Option<Vec<PyPhraseSplitEvent>>,
    #[pyo3(get)]
    pub dropped_candidates: Option<Vec<PyDroppedCandidate>>,
}

#[pymethods]
impl PyDebugPayload {
    fn __repr__(&self) -> String {
        let mut parts = Vec::new();
        if self.graph_stats.is_some() {
            parts.push("graph_stats");
        }
        if self.convergence_summary.is_some() {
            parts.push("convergence_summary");
        }
        if self.node_scores.is_some() {
            parts.push("node_scores");
        }
        if self.phrase_diagnostics.is_some() {
            parts.push("phrase_diagnostics");
        }
        if self.dropped_candidates.is_some() {
            parts.push("dropped_candidates");
        }
        if self.cluster_details.is_some() {
            parts.push("cluster_details");
        }
        format!("DebugPayload(fields=[{}])", parts.join(", "))
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        if let Some(ref gs) = self.graph_stats {
            d.set_item("graph_stats", gs.to_dict(py)?)?;
        }
        if let Some(ref cs) = self.convergence_summary {
            d.set_item("convergence_summary", cs.to_dict(py)?)?;
        }
        if let Some(ref ns) = self.node_scores {
            d.set_item("node_scores", ns.clone())?;
        }
        if let Some(ref st) = self.stage_timings {
            d.set_item("stage_timings", st.clone())?;
        }
        if let Some(ref r) = self.residuals {
            d.set_item("residuals", r.clone())?;
        }
        if let Some(ref cm) = self.cluster_memberships {
            d.set_item("cluster_memberships", cm.clone())?;
        }
        if let Some(ref cd) = self.cluster_details {
            let details: Vec<_> = cd
                .iter()
                .map(|c| c.to_dict(py))
                .collect::<PyResult<Vec<_>>>()?;
            d.set_item("cluster_details", details)?;
        }
        if let Some(ref pd) = self.phrase_diagnostics {
            let events: Vec<_> = pd
                .iter()
                .map(|e| e.to_dict(py))
                .collect::<PyResult<Vec<_>>>()?;
            d.set_item("phrase_diagnostics", events)?;
        }
        if let Some(ref dc) = self.dropped_candidates {
            let drops: Vec<_> = dc
                .iter()
                .map(|c| c.to_dict(py))
                .collect::<PyResult<Vec<_>>>()?;
            d.set_item("dropped_candidates", drops)?;
        }
        Ok(d)
    }
}

/// Convert a Rust `DebugPayload` into the Python wrapper.
fn convert_debug_payload(payload: DebugPayload) -> PyDebugPayload {
    let graph_stats = payload.graph_stats.map(|gs| PyGraphStats {
        num_nodes: gs.num_nodes,
        num_edges: gs.num_edges,
        avg_degree: gs.avg_degree,
        is_transformed: gs.is_transformed,
    });

    let convergence_summary = payload.convergence_summary.map(|cs| PyConvergenceSummary {
        iterations: cs.iterations,
        converged: cs.converged,
        final_delta: cs.final_delta,
    });

    let cluster_details = payload.cluster_details.map(|clusters| {
        clusters
            .into_iter()
            .enumerate()
            .map(|(i, members)| PyClusterDetail {
                cluster_index: i,
                members: members
                    .into_iter()
                    .map(|m| PyClusterMember {
                        index: m.index,
                        text: m.text,
                        lemma: m.lemma,
                    })
                    .collect(),
            })
            .collect()
    });

    let phrase_diagnostics = payload.phrase_diagnostics.map(|events| {
        events
            .into_iter()
            .map(|e| PyPhraseSplitEvent {
                token_range: e.token_range,
                text: e.text,
                reason: format!("{:?}", e.reason),
            })
            .collect()
    });

    let dropped_candidates = payload.dropped_candidates.map(|drops| {
        drops
            .into_iter()
            .map(|d| PyDroppedCandidate {
                text: d.text,
                lemma: d.lemma,
                score: d.score,
                token_range: d.token_range,
                reason: format!("{:?}", d.reason),
            })
            .collect()
    });

    PyDebugPayload {
        graph_stats,
        convergence_summary,
        node_scores: payload.node_scores,
        stage_timings: payload.stage_timings,
        residuals: payload.residuals,
        cluster_memberships: payload.cluster_memberships,
        cluster_details,
        phrase_diagnostics,
        dropped_candidates,
    }
}

/// Convert an `ExtractionResult` into a `PyTextRankResult`.
fn extraction_result_to_py(
    result: crate::phrase::extraction::ExtractionResult,
) -> PyTextRankResult {
    PyTextRankResult {
        phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
        converged: result.converged,
        iterations: result.iterations,
        debug: result.debug.map(convert_debug_payload),
    }
}

// ─── End debug classes ──────────────────────────────────────────────────

/// Result of TextRank extraction
#[pyclass(name = "TextRankResult")]
#[derive(Clone)]
pub struct PyTextRankResult {
    #[pyo3(get)]
    pub phrases: Vec<PyPhrase>,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub debug: Option<PyDebugPayload>,
}

#[pymethods]
impl PyTextRankResult {
    fn __repr__(&self) -> String {
        format!(
            "TextRankResult(phrases={}, converged={}, iterations={})",
            self.phrases.len(),
            self.converged,
            self.iterations
        )
    }

    fn __len__(&self) -> usize {
        self.phrases.len()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<PyPhrase> {
        self.phrases
            .get(idx)
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("index out of range"))
    }

    /// Get phrases as a list of (text, score) tuples
    fn as_tuples(&self) -> Vec<(String, f64)> {
        self.phrases
            .iter()
            .map(|p| (p.text.clone(), p.score))
            .collect()
    }
}

/// Get the built-in stopword list for a language.
#[pyfunction]
#[pyo3(signature = (language = "en"))]
pub fn get_stopwords(language: &str) -> PyResult<Vec<String>> {
    Ok(StopwordFilter::built_in_list(language))
}

/// Build an optional dedicated rayon thread pool.
fn build_thread_pool(max_threads: Option<usize>) -> PyResult<Option<Arc<rayon::ThreadPool>>> {
    match max_threads {
        None => Ok(None),
        Some(0) => Err(pyo3::exceptions::PyValueError::new_err(
            "max_threads must be >= 1 (use None for the global pool)",
        )),
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create thread pool: {}",
                        e
                    ))
                })?;
            Ok(Some(Arc::new(pool)))
        }
    }
}

/// Run a closure in the given pool, or on the global pool if `None`.
fn run_in_pool<R: Send>(pool: &Option<Arc<rayon::ThreadPool>>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// Configuration for TextRank
#[pyclass(name = "TextRankConfig")]
#[derive(Clone)]
pub struct PyTextRankConfig {
    inner: TextRankConfig,
}

#[pymethods]
impl PyTextRankConfig {
    #[new]
    #[pyo3(signature = (
        damping=0.85,
        max_iterations=100,
        convergence_threshold=1e-6,
        window_size=3,
        top_n=10,
        min_phrase_length=1,
        max_phrase_length=4,
        score_aggregation="sum",
        language="en",
        use_edge_weights=true,
        include_pos=None,
        stopwords=None,
        use_pos_in_nodes=true,
        phrase_grouping="scrubbed_text",
        determinism="default",
        debug_level="none",
        debug_top_k=50
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        damping: f64,
        max_iterations: usize,
        convergence_threshold: f64,
        window_size: usize,
        top_n: usize,
        min_phrase_length: usize,
        max_phrase_length: usize,
        score_aggregation: &str,
        language: &str,
        use_edge_weights: bool,
        include_pos: Option<Vec<String>>,
        stopwords: Option<Vec<String>>,
        use_pos_in_nodes: bool,
        phrase_grouping: &str,
        determinism: &str,
        debug_level: &str,
        debug_top_k: usize,
    ) -> PyResult<Self> {
        let aggregation = match score_aggregation.to_lowercase().as_str() {
            "sum" => ScoreAggregation::Sum,
            "mean" | "average" => ScoreAggregation::Mean,
            "max" => ScoreAggregation::Max,
            "rms" | "root_mean_square" => ScoreAggregation::RootMeanSquare,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown score_aggregation: {}. Use 'sum', 'mean', 'max', or 'rms'",
                    score_aggregation
                )))
            }
        };

        // Parse include_pos from string tags
        let pos_tags: Vec<crate::types::PosTag> = match include_pos {
            Some(tags) => tags
                .iter()
                .map(|s| crate::types::PosTag::from_spacy(s))
                .collect(),
            None => vec![
                crate::types::PosTag::Noun,
                crate::types::PosTag::Adjective,
                crate::types::PosTag::ProperNoun,
                crate::types::PosTag::Verb,
            ],
        };

        let det_mode = match determinism.to_lowercase().as_str() {
            "deterministic" => crate::types::DeterminismMode::Deterministic,
            "default" => crate::types::DeterminismMode::Default,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown determinism mode: {}. Use 'default' or 'deterministic'",
                    determinism
                )))
            }
        };

        let dbg_level = DebugLevel::parse_str(debug_level).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown debug_level: {}. Use 'none', 'stats', 'top_nodes', or 'full'",
                debug_level
            ))
        })?;

        let config = TextRankConfig {
            damping,
            max_iterations,
            convergence_threshold,
            window_size,
            top_n,
            min_phrase_length,
            max_phrase_length,
            score_aggregation: aggregation,
            language: language.to_string(),
            use_edge_weights,
            include_pos: pos_tags,
            stopwords: stopwords.unwrap_or_default(),
            use_pos_in_nodes,
            phrase_grouping: phrase_grouping.parse().unwrap_or(PhraseGrouping::Lemma),
            determinism: det_mode,
            debug_level: dbg_level,
            debug_top_k,
            max_nodes: None,
            max_edges: None,
        };

        config
            .validate()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner: config })
    }

    fn __repr__(&self) -> String {
        format!(
            "TextRankConfig(damping={}, window_size={}, top_n={})",
            self.inner.damping, self.inner.window_size, self.inner.top_n
        )
    }
}

/// Base TextRank keyword extractor
#[pyclass(name = "BaseTextRank")]
pub struct PyBaseTextRank {
    config: TextRankConfig,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

#[pymethods]
impl PyBaseTextRank {
    #[new]
    #[pyo3(signature = (config=None, top_n=None, language=None, max_threads=None))]
    fn new(
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
        max_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner_config = config.map(|c| c.inner).unwrap_or_default();

        if let Some(n) = top_n {
            inner_config.top_n = n;
        }
        if let Some(lang) = language {
            inner_config.language = lang.to_string();
        }

        Ok(Self {
            config: inner_config,
            thread_pool: build_thread_pool(max_threads)?,
        })
    }

    /// Extract keywords from text
    ///
    /// For small documents, this uses the built-in tokenizer.
    /// For best results with large documents, use the JSON interface
    /// with pre-tokenized data from spaCy.
    #[pyo3(signature = (text))]
    fn extract_keywords(&self, py: Python<'_>, text: &str) -> PyResult<PyTextRankResult> {
        let config = self.config.clone();
        let text = text.to_owned();
        let pool = self.thread_pool.clone();

        // Release the GIL for CPU-intensive extraction.
        let result = py.allow_threads(move || {
            run_in_pool(&pool, move || {
                let tokenizer = Tokenizer::new();
                let (_sentences, mut tokens) = tokenizer.tokenize(&text);

                let stopwords = if config.stopwords.is_empty() {
                    StopwordFilter::new(&config.language)
                } else {
                    StopwordFilter::with_additional(&config.language, &config.stopwords)
                };
                for token in &mut tokens {
                    token.is_stopword = stopwords.is_stopword(&token.text);
                }

                extract_keyphrases_with_info(&tokens, &config)
            })
        });

        Ok(extraction_result_to_py(result))
    }

    /// Get the number of threads in the dedicated pool, or None if using the global pool.
    #[getter]
    fn max_threads(&self) -> Option<usize> {
        self.thread_pool.as_ref().map(|p| p.current_num_threads())
    }

    /// Replace the dedicated thread pool. Pass None to revert to the global pool.
    #[pyo3(signature = (max_threads=None))]
    fn set_max_threads(&mut self, max_threads: Option<usize>) -> PyResult<()> {
        self.thread_pool = build_thread_pool(max_threads)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "BaseTextRank(top_n={}, language='{}')",
            self.config.top_n, self.config.language
        )
    }
}

/// PositionRank keyword extractor
#[pyclass(name = "PositionRank")]
pub struct PyPositionRank {
    config: TextRankConfig,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

#[pymethods]
impl PyPositionRank {
    #[new]
    #[pyo3(signature = (config=None, top_n=None, language=None, max_threads=None))]
    fn new(
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
        max_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner_config = config.map(|c| c.inner).unwrap_or_default();

        if let Some(n) = top_n {
            inner_config.top_n = n;
        }
        if let Some(lang) = language {
            inner_config.language = lang.to_string();
        }

        Ok(Self {
            config: inner_config,
            thread_pool: build_thread_pool(max_threads)?,
        })
    }

    /// Extract keywords using PositionRank
    #[pyo3(signature = (text))]
    fn extract_keywords(&self, py: Python<'_>, text: &str) -> PyResult<PyTextRankResult> {
        let config = self.config.clone();
        let text = text.to_owned();
        let pool = self.thread_pool.clone();

        let result = py.allow_threads(move || {
            run_in_pool(&pool, move || {
                let tokenizer = Tokenizer::new();
                let (_, mut tokens) = tokenizer.tokenize(&text);

                let stopwords = if config.stopwords.is_empty() {
                    StopwordFilter::new(&config.language)
                } else {
                    StopwordFilter::with_additional(&config.language, &config.stopwords)
                };
                for token in &mut tokens {
                    token.is_stopword = stopwords.is_stopword(&token.text);
                }

                PositionRank::with_config(config).extract_with_info(&tokens)
            })
        });

        Ok(extraction_result_to_py(result))
    }

    #[getter]
    fn max_threads(&self) -> Option<usize> {
        self.thread_pool.as_ref().map(|p| p.current_num_threads())
    }

    #[pyo3(signature = (max_threads=None))]
    fn set_max_threads(&mut self, max_threads: Option<usize>) -> PyResult<()> {
        self.thread_pool = build_thread_pool(max_threads)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "PositionRank(top_n={}, language='{}')",
            self.config.top_n, self.config.language
        )
    }
}

/// BiasedTextRank keyword extractor
#[pyclass(name = "BiasedTextRank")]
pub struct PyBiasedTextRank {
    config: TextRankConfig,
    focus_terms: Vec<String>,
    bias_weight: f64,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

#[pymethods]
impl PyBiasedTextRank {
    #[new]
    #[pyo3(signature = (focus_terms=None, bias_weight=5.0, config=None, top_n=None, language=None, max_threads=None))]
    fn new(
        focus_terms: Option<Vec<String>>,
        bias_weight: f64,
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
        max_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner_config = config.map(|c| c.inner).unwrap_or_default();

        if let Some(n) = top_n {
            inner_config.top_n = n;
        }
        if let Some(lang) = language {
            inner_config.language = lang.to_string();
        }

        Ok(Self {
            config: inner_config,
            focus_terms: focus_terms.unwrap_or_default(),
            bias_weight,
            thread_pool: build_thread_pool(max_threads)?,
        })
    }

    /// Set focus terms for biased extraction
    fn set_focus(&mut self, terms: Vec<String>) {
        self.focus_terms = terms;
    }

    /// Extract keywords using BiasedTextRank
    #[pyo3(signature = (text, focus_terms=None))]
    fn extract_keywords(
        &mut self,
        py: Python<'_>,
        text: &str,
        focus_terms: Option<Vec<String>>,
    ) -> PyResult<PyTextRankResult> {
        // Update focus terms if provided
        if let Some(terms) = focus_terms {
            self.focus_terms = terms;
        }

        let config = self.config.clone();
        let text = text.to_owned();
        let focus_terms = self.focus_terms.clone();
        let bias_weight = self.bias_weight;
        let pool = self.thread_pool.clone();

        let result = py.allow_threads(move || {
            run_in_pool(&pool, move || {
                let tokenizer = Tokenizer::new();
                let (_, mut tokens) = tokenizer.tokenize(&text);

                let stopwords = if config.stopwords.is_empty() {
                    StopwordFilter::new(&config.language)
                } else {
                    StopwordFilter::with_additional(&config.language, &config.stopwords)
                };
                for token in &mut tokens {
                    token.is_stopword = stopwords.is_stopword(&token.text);
                }

                let focus_refs: Vec<&str> = focus_terms.iter().map(|s| s.as_str()).collect();
                BiasedTextRank::with_config(config)
                    .with_focus(&focus_refs)
                    .with_bias_weight(bias_weight)
                    .extract_with_info(&tokens)
            })
        });

        Ok(extraction_result_to_py(result))
    }

    #[getter]
    fn max_threads(&self) -> Option<usize> {
        self.thread_pool.as_ref().map(|p| p.current_num_threads())
    }

    #[pyo3(signature = (max_threads=None))]
    fn set_max_threads(&mut self, max_threads: Option<usize>) -> PyResult<()> {
        self.thread_pool = build_thread_pool(max_threads)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "BiasedTextRank(focus_terms={:?}, bias_weight={}, top_n={})",
            self.focus_terms, self.bias_weight, self.config.top_n
        )
    }
}

/// SingleRank keyword extractor
///
/// SingleRank uses weighted co-occurrence edges and cross-sentence
/// windowing. The rest of the pipeline is identical to base TextRank.
#[pyclass(name = "SingleRank")]
pub struct PySingleRank {
    config: TextRankConfig,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

#[pymethods]
impl PySingleRank {
    #[new]
    #[pyo3(signature = (config=None, top_n=None, language=None, max_threads=None))]
    fn new(
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
        max_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner_config = config.map(|c| c.inner).unwrap_or_default();

        if let Some(n) = top_n {
            inner_config.top_n = n;
        }
        if let Some(lang) = language {
            inner_config.language = lang.to_string();
        }

        Ok(Self {
            config: inner_config,
            thread_pool: build_thread_pool(max_threads)?,
        })
    }

    /// Extract keywords using SingleRank
    #[pyo3(signature = (text))]
    fn extract_keywords(&self, py: Python<'_>, text: &str) -> PyResult<PyTextRankResult> {
        let config = self.config.clone();
        let text = text.to_owned();
        let pool = self.thread_pool.clone();

        let result = py.allow_threads(move || {
            run_in_pool(&pool, move || {
                let tokenizer = Tokenizer::new();
                let (_, mut tokens) = tokenizer.tokenize(&text);

                let stopwords = if config.stopwords.is_empty() {
                    StopwordFilter::new(&config.language)
                } else {
                    StopwordFilter::with_additional(&config.language, &config.stopwords)
                };
                for token in &mut tokens {
                    token.is_stopword = stopwords.is_stopword(&token.text);
                }

                SingleRank::with_config(config).extract_with_info(&tokens)
            })
        });

        Ok(extraction_result_to_py(result))
    }

    #[getter]
    fn max_threads(&self) -> Option<usize> {
        self.thread_pool.as_ref().map(|p| p.current_num_threads())
    }

    #[pyo3(signature = (max_threads=None))]
    fn set_max_threads(&mut self, max_threads: Option<usize>) -> PyResult<()> {
        self.thread_pool = build_thread_pool(max_threads)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "SingleRank(top_n={}, language='{}')",
            self.config.top_n, self.config.language
        )
    }
}

/// Topical PageRank keyword extractor
///
/// Uses topic-importance weights to bias the random walk towards
/// topically relevant words. Combines SingleRank's weighted graph
/// with personalized PageRank.
#[pyclass(name = "TopicalPageRank")]
pub struct PyTopicalPageRank {
    config: TextRankConfig,
    topic_weights: HashMap<String, f64>,
    min_weight: f64,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

#[pymethods]
impl PyTopicalPageRank {
    #[new]
    #[pyo3(signature = (topic_weights=None, min_weight=0.0, config=None, top_n=None, language=None, max_threads=None))]
    fn new(
        topic_weights: Option<HashMap<String, f64>>,
        min_weight: f64,
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
        max_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner_config = config.map(|c| c.inner).unwrap_or_default();

        if let Some(n) = top_n {
            inner_config.top_n = n;
        }
        if let Some(lang) = language {
            inner_config.language = lang.to_string();
        }

        Ok(Self {
            config: inner_config,
            topic_weights: topic_weights.unwrap_or_default(),
            min_weight,
            thread_pool: build_thread_pool(max_threads)?,
        })
    }

    /// Set topic weights for extraction
    fn set_topic_weights(&mut self, weights: HashMap<String, f64>) {
        self.topic_weights = weights;
    }

    /// Extract keywords using Topical PageRank
    #[pyo3(signature = (text, topic_weights=None))]
    fn extract_keywords(
        &mut self,
        py: Python<'_>,
        text: &str,
        topic_weights: Option<HashMap<String, f64>>,
    ) -> PyResult<PyTextRankResult> {
        if let Some(weights) = topic_weights {
            self.topic_weights = weights;
        }

        let config = self.config.clone();
        let text = text.to_owned();
        let topic_weights = self.topic_weights.clone();
        let min_weight = self.min_weight;
        let pool = self.thread_pool.clone();

        let result = py.allow_threads(move || {
            run_in_pool(&pool, move || {
                let tokenizer = Tokenizer::new();
                let (_, mut tokens) = tokenizer.tokenize(&text);

                let stopwords = if config.stopwords.is_empty() {
                    StopwordFilter::new(&config.language)
                } else {
                    StopwordFilter::with_additional(&config.language, &config.stopwords)
                };
                for token in &mut tokens {
                    token.is_stopword = stopwords.is_stopword(&token.text);
                }

                TopicalPageRank::with_config(config)
                    .with_topic_weights(topic_weights)
                    .with_min_weight(min_weight)
                    .extract_with_info(&tokens)
            })
        });

        Ok(extraction_result_to_py(result))
    }

    #[getter]
    fn max_threads(&self) -> Option<usize> {
        self.thread_pool.as_ref().map(|p| p.current_num_threads())
    }

    #[pyo3(signature = (max_threads=None))]
    fn set_max_threads(&mut self, max_threads: Option<usize>) -> PyResult<()> {
        self.thread_pool = build_thread_pool(max_threads)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "TopicalPageRank(topic_weights={}, min_weight={}, top_n={})",
            self.topic_weights.len(),
            self.min_weight,
            self.config.top_n
        )
    }
}

/// MultipartiteRank keyword extractor
///
/// Builds a k-partite directed graph where candidates from different
/// topic clusters are connected. An alpha adjustment boosts the
/// first-occurring variant in each topic.
#[pyclass(name = "MultipartiteRank")]
pub struct PyMultipartiteRank {
    config: TextRankConfig,
    similarity_threshold: f64,
    alpha: f64,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

#[pymethods]
impl PyMultipartiteRank {
    #[new]
    #[pyo3(signature = (similarity_threshold=0.26, alpha=1.1, config=None, top_n=None, language=None, max_threads=None))]
    fn new(
        similarity_threshold: f64,
        alpha: f64,
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
        max_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner_config = config.map(|c| c.inner).unwrap_or_default();

        if let Some(n) = top_n {
            inner_config.top_n = n;
        }
        if let Some(lang) = language {
            inner_config.language = lang.to_string();
        }

        Ok(Self {
            config: inner_config,
            similarity_threshold,
            alpha,
            thread_pool: build_thread_pool(max_threads)?,
        })
    }

    /// Extract keywords using MultipartiteRank
    #[pyo3(signature = (text))]
    fn extract_keywords(&self, py: Python<'_>, text: &str) -> PyResult<PyTextRankResult> {
        let config = self.config.clone();
        let text = text.to_owned();
        let similarity_threshold = self.similarity_threshold;
        let alpha = self.alpha;
        let pool = self.thread_pool.clone();

        let result = py.allow_threads(move || {
            run_in_pool(&pool, move || {
                let tokenizer = Tokenizer::new();
                let (_, mut tokens) = tokenizer.tokenize(&text);

                let stopwords = if config.stopwords.is_empty() {
                    StopwordFilter::new(&config.language)
                } else {
                    StopwordFilter::with_additional(&config.language, &config.stopwords)
                };
                for token in &mut tokens {
                    token.is_stopword = stopwords.is_stopword(&token.text);
                }

                MultipartiteRank::with_config(config)
                    .with_similarity_threshold(similarity_threshold)
                    .with_alpha(alpha)
                    .extract_with_info(&tokens)
            })
        });

        Ok(extraction_result_to_py(result))
    }

    #[getter]
    fn max_threads(&self) -> Option<usize> {
        self.thread_pool.as_ref().map(|p| p.current_num_threads())
    }

    #[pyo3(signature = (max_threads=None))]
    fn set_max_threads(&mut self, max_threads: Option<usize>) -> PyResult<()> {
        self.thread_pool = build_thread_pool(max_threads)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "MultipartiteRank(alpha={}, similarity_threshold={}, top_n={})",
            self.alpha, self.similarity_threshold, self.config.top_n
        )
    }
}

/// SentenceRank extractive summarizer
///
/// Ranks whole sentences using TextRank with Jaccard-similarity edges.
/// Returns the top-N most important sentences as an extractive summary.
#[cfg(feature = "sentence-rank")]
#[pyclass(name = "SentenceRank")]
pub struct PySentenceRank {
    config: TextRankConfig,
    sort_by_position: bool,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

#[cfg(feature = "sentence-rank")]
#[pymethods]
impl PySentenceRank {
    #[new]
    #[pyo3(signature = (config=None, top_n=None, language=None, sort_by_position=false, max_threads=None))]
    fn new(
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
        sort_by_position: bool,
        max_threads: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner_config = config.map(|c| c.inner).unwrap_or_default();

        if let Some(n) = top_n {
            inner_config.top_n = n;
        }
        if let Some(lang) = language {
            inner_config.language = lang.to_string();
        }

        Ok(Self {
            config: inner_config,
            sort_by_position,
            thread_pool: build_thread_pool(max_threads)?,
        })
    }

    /// Extract the most important sentences from text
    #[pyo3(signature = (text))]
    fn extract_sentences(&self, py: Python<'_>, text: &str) -> PyResult<PyTextRankResult> {
        let config = self.config.clone();
        let text = text.to_owned();
        let sort_by_position = self.sort_by_position;
        let pool = self.thread_pool.clone();

        let result = py.allow_threads(move || {
            run_in_pool(&pool, move || {
                let tokenizer = Tokenizer::new();
                let (_, mut tokens) = tokenizer.tokenize(&text);

                let stopwords = if config.stopwords.is_empty() {
                    StopwordFilter::new(&config.language)
                } else {
                    StopwordFilter::with_additional(&config.language, &config.stopwords)
                };
                for token in &mut tokens {
                    token.is_stopword = stopwords.is_stopword(&token.text);
                }

                let stream = TokenStream::from_tokens(&tokens);
                let mut obs = NoopObserver;
                let pipeline = if sort_by_position {
                    SentenceRankPipeline::sentence_rank_by_position()
                } else {
                    SentenceRankPipeline::sentence_rank()
                };
                pipeline.run(stream, &config, &mut obs)
            })
        });

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations as usize,
            debug: result.debug.map(convert_debug_payload),
        })
    }

    #[getter]
    fn max_threads(&self) -> Option<usize> {
        self.thread_pool.as_ref().map(|p| p.current_num_threads())
    }

    #[pyo3(signature = (max_threads=None))]
    fn set_max_threads(&mut self, max_threads: Option<usize>) -> PyResult<()> {
        self.thread_pool = build_thread_pool(max_threads)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "SentenceRank(top_n={}, sort_by_position={}, language='{}')",
            self.config.top_n, self.sort_by_position, self.config.language
        )
    }
}
