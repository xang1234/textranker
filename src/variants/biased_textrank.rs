//! BiasedTextRank variant
//!
//! BiasedTextRank allows specifying "focus" terms that should be prioritized.
//! This is useful for topic-specific keyword extraction where you want
//! keywords related to a particular concept.
//!
//! Focus terms receive a higher personalization weight, biasing the random
//! walk towards them and their neighboring nodes.
//!
//! Internally this is BaseTextRank + [`FocusTermsTeleportBuilder`]: the only
//! difference from the base algorithm is the teleport (personalization)
//! strategy.

use crate::phrase::extraction::ExtractionResult;
use crate::pipeline::artifacts::TokenStream;
use crate::pipeline::observer::NoopObserver;
use crate::pipeline::runner::BiasedTextRankPipeline;
use crate::types::{Phrase, TextRankConfig, Token};

/// BiasedTextRank implementation
#[derive(Debug)]
pub struct BiasedTextRank {
    config: TextRankConfig,
    /// Focus terms (lemmatized)
    focus_terms: Vec<String>,
    /// Weight multiplier for focus terms
    bias_weight: f64,
}

impl Default for BiasedTextRank {
    fn default() -> Self {
        Self::new()
    }
}

impl BiasedTextRank {
    /// Create a new BiasedTextRank extractor
    pub fn new() -> Self {
        Self {
            config: TextRankConfig::default(),
            focus_terms: Vec::new(),
            bias_weight: 5.0,
        }
    }

    /// Create with custom config
    pub fn with_config(config: TextRankConfig) -> Self {
        Self {
            config,
            focus_terms: Vec::new(),
            bias_weight: 5.0,
        }
    }

    /// Set focus terms
    pub fn with_focus(mut self, terms: &[&str]) -> Self {
        self.focus_terms = terms.iter().map(|s| s.to_lowercase()).collect();
        self
    }

    /// Set bias weight for focus terms
    pub fn with_bias_weight(mut self, weight: f64) -> Self {
        self.bias_weight = weight;
        self
    }

    /// Extract keyphrases with current focus terms
    pub fn extract(&self, tokens: &[Token]) -> Vec<Phrase> {
        self.extract_with_info(tokens).phrases
    }

    /// Extract keyphrases with PageRank convergence information
    pub fn extract_with_info(&self, tokens: &[Token]) -> ExtractionResult {
        let pipeline = BiasedTextRankPipeline::biased(self.focus_terms.clone(), self.bias_weight);
        let stream = TokenStream::from_tokens(tokens);
        let mut obs = NoopObserver;
        let result = pipeline.run(stream, &self.config, &mut obs);

        ExtractionResult {
            phrases: result.phrases,
            converged: result.converged,
            iterations: result.iterations as usize,
            debug: result.debug,
        }
    }

    /// Change focus and re-rank
    ///
    /// Runs a fresh pipeline with the new focus terms.
    pub fn change_focus(&mut self, new_focus: &[&str], tokens: &[Token]) -> Option<Vec<Phrase>> {
        self.focus_terms = new_focus.iter().map(|s| s.to_lowercase()).collect();
        Some(self.extract(tokens))
    }

    /// Get the current focus terms
    pub fn focus_terms(&self) -> &[String] {
        &self.focus_terms
    }

    /// Get the bias weight
    pub fn bias_weight(&self) -> f64 {
        self.bias_weight
    }
}

/// Convenience function to extract keyphrases with focus terms
pub fn extract_keyphrases_biased(
    tokens: &[Token],
    config: &TextRankConfig,
    focus_terms: &[&str],
    bias_weight: f64,
) -> Vec<Phrase> {
    BiasedTextRank::with_config(config.clone())
        .with_focus(focus_terms)
        .with_bias_weight(bias_weight)
        .extract(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    fn make_tokens() -> Vec<Token> {
        // "Machine learning uses algorithms. Deep learning uses neural networks."
        vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("uses", "use", PosTag::Verb, 17, 21, 0, 2),
            Token::new("algorithms", "algorithm", PosTag::Noun, 22, 32, 0, 3),
            Token::new("Deep", "deep", PosTag::Adjective, 34, 38, 1, 4),
            Token::new("learning", "learning", PosTag::Noun, 39, 47, 1, 5),
            Token::new("uses", "use", PosTag::Verb, 48, 52, 1, 6),
            Token::new("neural", "neural", PosTag::Adjective, 53, 59, 1, 7),
            Token::new("networks", "network", PosTag::Noun, 60, 68, 1, 8),
        ]
    }

    #[test]
    fn test_biased_textrank() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_biased(&tokens, &config, &["machine"], 5.0);

        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_focus_affects_ranking() {
        let tokens = make_tokens();
        let config = TextRankConfig::default().with_top_n(10);

        // Extract with "machine" focus
        let phrases_machine = extract_keyphrases_biased(&tokens, &config, &["machine"], 10.0);

        // Extract with "neural" focus
        let phrases_neural = extract_keyphrases_biased(&tokens, &config, &["neural"], 10.0);

        // The focused term should rank higher in its respective extraction
        let machine_rank_1 = phrases_machine
            .iter()
            .find(|p| p.lemma.contains("machine"))
            .map(|p| p.rank);
        let machine_rank_2 = phrases_neural
            .iter()
            .find(|p| p.lemma.contains("machine"))
            .map(|p| p.rank);

        // "machine" should rank higher when it's the focus
        if let (Some(rank_1), Some(rank_2)) = (machine_rank_1, machine_rank_2) {
            assert!(rank_1 <= rank_2);
        }
    }

    #[test]
    fn test_change_focus() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();

        let mut extractor = BiasedTextRank::with_config(config)
            .with_focus(&["machine"])
            .with_bias_weight(5.0);

        // First extraction builds the graph
        let phrases1 = extractor.extract(&tokens);
        assert!(!phrases1.is_empty());

        // Change focus and re-rank
        let phrases2 = extractor.change_focus(&["neural"], &tokens);
        assert!(phrases2.is_some());
        assert!(!phrases2.unwrap().is_empty());
    }

    #[test]
    fn test_empty_focus() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();

        // Empty focus should work (equivalent to standard TextRank)
        let phrases = extract_keyphrases_biased(&tokens, &config, &[], 5.0);
        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_nonexistent_focus() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();

        // Focus on term not in document should still work
        let phrases = extract_keyphrases_biased(&tokens, &config, &["nonexistent"], 5.0);
        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_convergence_info() {
        let tokens = make_tokens();
        let result = BiasedTextRank::new()
            .with_focus(&["machine"])
            .with_bias_weight(5.0)
            .extract_with_info(&tokens);

        assert!(!result.phrases.is_empty());
        assert!(result.converged);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_biased_pipeline_constructs() {
        let _pipeline = BiasedTextRankPipeline::biased(vec!["machine".to_string()], 5.0);
    }
}
