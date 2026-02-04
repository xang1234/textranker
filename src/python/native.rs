//! Native Python interface
//!
//! Direct Python classes for small documents where Python↔Rust
//! overhead is negligible compared to processing time.

use crate::nlp::stopwords::StopwordFilter;
use crate::nlp::tokenizer::Tokenizer;
use crate::phrase::extraction::extract_keyphrases_with_info;
use crate::types::{Phrase, PhraseGrouping, ScoreAggregation, TextRankConfig};
use crate::variants::biased_textrank::BiasedTextRank;
use crate::variants::position_rank::PositionRank;
use pyo3::prelude::*;

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
        phrase_grouping="scrubbed_text"
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
            phrase_grouping: PhraseGrouping::from_str(phrase_grouping),
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
}

#[pymethods]
impl PyBaseTextRank {
    #[new]
    #[pyo3(signature = (config=None, top_n=None, language=None))]
    fn new(
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
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
        })
    }

    /// Extract keywords from text
    ///
    /// For small documents, this uses the built-in tokenizer.
    /// For best results with large documents, use the JSON interface
    /// with pre-tokenized data from spaCy.
    #[pyo3(signature = (text))]
    fn extract_keywords(&self, text: &str) -> PyResult<PyTextRankResult> {
        // Tokenize
        let tokenizer = Tokenizer::new();
        let (_sentences, mut tokens) = tokenizer.tokenize(text);

        // Apply stopword filter
        let stopwords = if self.config.stopwords.is_empty() {
            StopwordFilter::new(&self.config.language)
        } else {
            StopwordFilter::with_additional(&self.config.language, &self.config.stopwords)
        };
        for token in &mut tokens {
            token.is_stopword = stopwords.is_stopword(&token.text);
        }

        // Extract phrases with convergence info
        let result = extract_keyphrases_with_info(&tokens, &self.config);

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
        })
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
}

#[pymethods]
impl PyPositionRank {
    #[new]
    #[pyo3(signature = (config=None, top_n=None, language=None))]
    fn new(
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
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
        })
    }

    /// Extract keywords using PositionRank
    #[pyo3(signature = (text))]
    fn extract_keywords(&self, text: &str) -> PyResult<PyTextRankResult> {
        let tokenizer = Tokenizer::new();
        let (_, mut tokens) = tokenizer.tokenize(text);

        let stopwords = if self.config.stopwords.is_empty() {
            StopwordFilter::new(&self.config.language)
        } else {
            StopwordFilter::with_additional(&self.config.language, &self.config.stopwords)
        };
        for token in &mut tokens {
            token.is_stopword = stopwords.is_stopword(&token.text);
        }

        let extractor = PositionRank::with_config(self.config.clone());
        let result = extractor.extract_with_info(&tokens);

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
        })
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
}

#[pymethods]
impl PyBiasedTextRank {
    #[new]
    #[pyo3(signature = (focus_terms=None, bias_weight=5.0, config=None, top_n=None, language=None))]
    fn new(
        focus_terms: Option<Vec<String>>,
        bias_weight: f64,
        config: Option<PyTextRankConfig>,
        top_n: Option<usize>,
        language: Option<&str>,
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
        text: &str,
        focus_terms: Option<Vec<String>>,
    ) -> PyResult<PyTextRankResult> {
        // Update focus terms if provided
        if let Some(terms) = focus_terms {
            self.focus_terms = terms;
        }

        let tokenizer = Tokenizer::new();
        let (_, mut tokens) = tokenizer.tokenize(text);

        let stopwords = if self.config.stopwords.is_empty() {
            StopwordFilter::new(&self.config.language)
        } else {
            StopwordFilter::with_additional(&self.config.language, &self.config.stopwords)
        };
        for token in &mut tokens {
            token.is_stopword = stopwords.is_stopword(&token.text);
        }

        let focus_refs: Vec<&str> = self.focus_terms.iter().map(|s| s.as_str()).collect();
        let mut extractor = BiasedTextRank::with_config(self.config.clone())
            .with_focus(&focus_refs)
            .with_bias_weight(self.bias_weight);

        let result = extractor.extract_with_info(&tokens);

        Ok(PyTextRankResult {
            phrases: result.phrases.into_iter().map(PyPhrase::from).collect(),
            converged: result.converged,
            iterations: result.iterations,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BiasedTextRank(focus_terms={:?}, bias_weight={}, top_n={})",
            self.focus_terms, self.bias_weight, self.config.top_n
        )
    }
}
