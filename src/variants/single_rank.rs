//! SingleRank variant
//!
//! SingleRank (Wan & Xiao, 2008) extends TextRank with two modifications:
//! 1. **Weighted edges**: Co-occurrence counts are used as edge weights
//!    (always forced, regardless of `use_edge_weights` config).
//! 2. **Cross-sentence windowing**: The sliding window ignores sentence
//!    boundaries, so tokens at the end of one sentence can co-occur with
//!    tokens at the start of the next.
//!
//! The rest of the pipeline (standard PageRank, phrase extraction) is
//! identical to base TextRank.

use crate::pipeline::artifacts::TokenStream;
use crate::pipeline::observer::NoopObserver;
use crate::pipeline::runner::SingleRankPipeline;
use crate::phrase::extraction::ExtractionResult;
use crate::types::{Phrase, TextRankConfig, Token};

/// SingleRank implementation
#[derive(Debug)]
pub struct SingleRank {
    config: TextRankConfig,
}

impl Default for SingleRank {
    fn default() -> Self {
        Self::new()
    }
}

impl SingleRank {
    /// Create a new SingleRank extractor with default config
    pub fn new() -> Self {
        Self {
            config: TextRankConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: TextRankConfig) -> Self {
        Self { config }
    }

    /// Extract keyphrases using SingleRank
    pub fn extract(&self, tokens: &[Token]) -> Vec<Phrase> {
        self.extract_with_info(tokens).phrases
    }

    /// Extract keyphrases with PageRank convergence information
    pub fn extract_with_info(&self, tokens: &[Token]) -> ExtractionResult {
        let pipeline = SingleRankPipeline::single_rank();
        let stream = TokenStream::from_tokens(tokens);
        let mut obs = NoopObserver;
        let result = pipeline.run(stream, &self.config, &mut obs);

        ExtractionResult {
            phrases: result.phrases,
            converged: result.converged,
            iterations: result.iterations as usize,
        }
    }
}

/// Convenience function to extract keyphrases using SingleRank
pub fn extract_keyphrases_singlerank(tokens: &[Token], config: &TextRankConfig) -> Vec<Phrase> {
    SingleRank::with_config(config.clone()).extract(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    fn make_token(text: &str, lemma: &str, pos: PosTag, sent: usize, idx: usize) -> Token {
        Token {
            text: text.to_string(),
            lemma: lemma.to_string(),
            pos,
            start: 0,
            end: text.len(),
            sentence_idx: sent,
            token_idx: idx,
            is_stopword: false,
        }
    }

    #[test]
    fn test_singlerank_basic() {
        let tokens = vec![
            make_token("Machine", "machine", PosTag::Noun, 0, 0),
            make_token("learning", "learning", PosTag::Noun, 0, 1),
            make_token("is", "be", PosTag::Verb, 0, 2),
            make_token("artificial", "artificial", PosTag::Adjective, 0, 3),
            make_token("intelligence", "intelligence", PosTag::Noun, 0, 4),
        ];

        let config = TextRankConfig::default().with_top_n(5);
        let phrases = extract_keyphrases_singlerank(&tokens, &config);
        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let tokens: Vec<Token> = Vec::new();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_singlerank(&tokens, &config);
        assert!(phrases.is_empty());
    }

    #[test]
    fn test_cross_sentence_connections() {
        // Tokens spanning two sentences. SingleRank should create edges
        // between "learning" (sent 0) and "deep" (sent 1) because it
        // ignores sentence boundaries.
        let tokens = vec![
            make_token("machine", "machine", PosTag::Noun, 0, 0),
            make_token("learning", "learning", PosTag::Noun, 0, 1),
            make_token("deep", "deep", PosTag::Adjective, 1, 2),
            make_token("neural", "neural", PosTag::Adjective, 1, 3),
            make_token("network", "network", PosTag::Noun, 1, 4),
        ];

        let sr = SingleRank::with_config(TextRankConfig::default().with_top_n(10));
        let result = sr.extract_with_info(&tokens);

        // All tokens should participate in the graph (cross-sentence
        // windowing connects them), so we expect non-empty results.
        assert!(!result.phrases.is_empty());
        assert!(result.converged);
    }

    #[test]
    fn test_weighted_edges_affect_ranking() {
        // Build a scenario where a pair co-occurs many times.
        // With weighted edges the pair's edge weight is higher,
        // which should influence PageRank scores.
        let mut tokens = Vec::new();
        let mut idx = 0;
        for sent in 0..5 {
            tokens.push(make_token("data", "data", PosTag::Noun, sent, idx));
            idx += 1;
            tokens.push(make_token("science", "science", PosTag::Noun, sent, idx));
            idx += 1;
            tokens.push(make_token("field", "field", PosTag::Noun, sent, idx));
            idx += 1;
        }

        let config = TextRankConfig::default().with_top_n(5);
        let sr = SingleRank::with_config(config);
        let result = sr.extract_with_info(&tokens);

        assert!(!result.phrases.is_empty());

        // "data" and "science" co-occur 5 times adjacently, so they
        // should appear among the top phrases.
        let top_lemmas: Vec<&str> = result.phrases.iter().map(|p| p.lemma.as_str()).collect();
        assert!(
            top_lemmas
                .iter()
                .any(|l| l.contains("data") || l.contains("science")),
            "Expected 'data' or 'science' in top phrases, got: {:?}",
            top_lemmas
        );
    }

    #[test]
    fn test_deterministic_toy_graph() {
        // Triangle: A-B-C where A-B co-occurs 3x, A-C 1x, B-C 1x.
        // With weighted edges A and B should score highest.
        let tokens = vec![
            make_token("alpha", "alpha", PosTag::Noun, 0, 0),
            make_token("beta", "beta", PosTag::Noun, 0, 1),
            make_token("alpha", "alpha", PosTag::Noun, 0, 2),
            make_token("beta", "beta", PosTag::Noun, 0, 3),
            make_token("alpha", "alpha", PosTag::Noun, 0, 4),
            make_token("beta", "beta", PosTag::Noun, 0, 5),
            make_token("gamma", "gamma", PosTag::Noun, 0, 6),
        ];

        let config = TextRankConfig::default().with_top_n(3);
        let sr = SingleRank::with_config(config);
        let result = sr.extract_with_info(&tokens);

        assert!(!result.phrases.is_empty());

        // The top phrase should be alpha or beta (strongly connected pair)
        let top = &result.phrases[0];
        assert!(
            top.lemma.contains("alpha") || top.lemma.contains("beta"),
            "Expected alpha or beta as top phrase, got: {}",
            top.lemma
        );
    }
}
