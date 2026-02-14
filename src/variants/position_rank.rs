//! PositionRank variant
//!
//! PositionRank biases PageRank towards words that appear earlier in the document.
//! The intuition is that important keywords often appear early (title, introduction).
//!
//! Bias formula: weight = 1 / (position + 1)
//! where position is the first occurrence position of the word.
//!
//! Internally this is BaseTextRank + [`PositionTeleportBuilder`]: the only
//! difference from the base algorithm is the teleport (personalization)
//! strategy.

use crate::phrase::extraction::ExtractionResult;
use crate::pipeline::artifacts::TokenStream;
use crate::pipeline::observer::NoopObserver;
use crate::pipeline::runner::PositionRankPipeline;
use crate::types::{Phrase, TextRankConfig, Token};

/// PositionRank implementation
#[derive(Debug)]
pub struct PositionRank {
    config: TextRankConfig,
}

impl Default for PositionRank {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionRank {
    /// Create a new PositionRank extractor with default config
    pub fn new() -> Self {
        Self {
            config: TextRankConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: TextRankConfig) -> Self {
        Self { config }
    }

    /// Extract keyphrases using PositionRank
    pub fn extract(&self, tokens: &[Token]) -> Vec<Phrase> {
        self.extract_with_info(tokens).phrases
    }

    /// Extract keyphrases with PageRank convergence information
    pub fn extract_with_info(&self, tokens: &[Token]) -> ExtractionResult {
        let pipeline = PositionRankPipeline::position_rank();
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
}

/// Convenience function to extract keyphrases using PositionRank
pub fn extract_keyphrases_position(tokens: &[Token], config: &TextRankConfig) -> Vec<Phrase> {
    PositionRank::with_config(config.clone()).extract(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    fn make_tokens() -> Vec<Token> {
        // "Important topic first. Then details. Important topic again."
        vec![
            Token::new("Important", "important", PosTag::Adjective, 0, 9, 0, 0),
            Token::new("topic", "topic", PosTag::Noun, 10, 15, 0, 1),
            Token::new("first", "first", PosTag::Adverb, 16, 21, 0, 2),
            Token::new("Then", "then", PosTag::Adverb, 23, 27, 1, 3),
            Token::new("details", "detail", PosTag::Noun, 28, 35, 1, 4),
            Token::new("Important", "important", PosTag::Adjective, 37, 46, 2, 5),
            Token::new("topic", "topic", PosTag::Noun, 47, 52, 2, 6),
            Token::new("again", "again", PosTag::Adverb, 53, 58, 2, 7),
        ]
    }

    #[test]
    fn test_position_rank() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_position(&tokens, &config);

        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_position_rank_convergence_info() {
        let tokens = make_tokens();
        let result = PositionRank::new().extract_with_info(&tokens);

        assert!(!result.phrases.is_empty());
        assert!(result.converged);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_empty_input() {
        let tokens: Vec<Token> = Vec::new();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_position(&tokens, &config);

        assert!(phrases.is_empty());
    }

    #[test]
    fn test_earlier_words_preferred() {
        // Create tokens where an "early" word and a "late" word have similar context
        let tokens = vec![
            Token::new("Early", "early", PosTag::Noun, 0, 5, 0, 0),
            Token::new("topic", "topic", PosTag::Noun, 6, 11, 0, 1),
            Token::new("is", "be", PosTag::Verb, 12, 14, 0, 2),
            Token::new("important", "important", PosTag::Adjective, 15, 24, 0, 3),
            Token::new("Late", "late", PosTag::Noun, 26, 30, 1, 4),
            Token::new("topic", "topic", PosTag::Noun, 31, 36, 1, 5),
            Token::new("is", "be", PosTag::Verb, 37, 39, 1, 6),
            Token::new("important", "important", PosTag::Adjective, 40, 49, 1, 7),
        ];

        let config = TextRankConfig::default().with_top_n(10);
        let phrases = extract_keyphrases_position(&tokens, &config);

        // Early words should generally rank higher
        // Find if "early" appears before "late" in the ranking
        let early_rank = phrases.iter().find(|p| p.lemma == "early").map(|p| p.rank);
        let late_rank = phrases.iter().find(|p| p.lemma == "late").map(|p| p.rank);

        // If both exist, early should rank higher (lower rank number)
        if let (Some(early), Some(late)) = (early_rank, late_rank) {
            assert!(early < late);
        }
    }

    #[test]
    fn test_position_rank_pipeline_constructs() {
        let _pipeline = PositionRankPipeline::position_rank();
    }
}
