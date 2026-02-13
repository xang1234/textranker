//! Python bindings via PyO3
//!
//! This module provides the Python interface for rapid_textrank.

pub mod json;
pub mod native;

use pyo3::prelude::*;

/// Register all Python classes and functions
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Native interface classes
    m.add_class::<native::PyPhrase>()?;
    m.add_class::<native::PyTextRankResult>()?;
    m.add_class::<native::PyTextRankConfig>()?;
    m.add_class::<native::PyBaseTextRank>()?;
    m.add_class::<native::PyPositionRank>()?;
    m.add_class::<native::PyBiasedTextRank>()?;
    m.add_class::<native::PySingleRank>()?;
    m.add_class::<native::PyTopicalPageRank>()?;
    m.add_class::<native::PyMultipartiteRank>()?;
    m.add_function(wrap_pyfunction!(native::get_stopwords, m)?)?;

    // JSON interface functions
    m.add_function(wrap_pyfunction!(json::extract_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(json::extract_batch_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(json::extract_jsonl_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(json::validate_pipeline_spec, m)?)?;
    m.add_class::<json::JsonBatchIter>()?;
    m.add_function(wrap_pyfunction!(json::extract_batch_iter, m)?)?;

    Ok(())
}
