//! Native Python interface
//!
//! Direct Python classes for small documents where Python↔Rust
//! overhead is negligible compared to processing time.

use crate::nlp::stopwords::StopwordFilter;
use crate::nlp::tokenizer::Tokenizer;
use crate::phrase::extraction::extract_keyphrases_with_info;
use crate::pipeline::artifacts::TokenStream;
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
        determinism="default"
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
            debug_level: crate::pipeline::artifacts::DebugLevel::None,
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

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
        })
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

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
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

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
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

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
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

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
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

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
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
