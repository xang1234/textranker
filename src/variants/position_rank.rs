//! PositionRank variant
//!
//! PositionRank biases PageRank towards words that appear earlier in the document.
//! The intuition is that important keywords often appear early (title, introduction).
//!
//! Bias formula: weight = 1 / (position + 1)
//! where position is the first occurrence position of the word.

use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::personalized::{position_based_personalization, PersonalizedPageRank};
use crate::phrase::extraction::{ExtractionResult, PhraseExtractor};
use crate::types::{Phrase, TextRankConfig, Token};
use rustc_hash::FxHashMap;

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
        // Build graph with POS filtering
        let include_pos = if self.config.include_pos.is_empty() {
            None
        } else {
            Some(self.config.include_pos.as_slice())
        };
        let builder = GraphBuilder::from_tokens_with_pos(
            tokens,
            self.config.window_size,
            self.config.use_edge_weights,
            include_pos,
        );

        if builder.is_empty() {
            return ExtractionResult {
                phrases: Vec::new(),
                converged: true,
                iterations: 0,
            };
        }

        let graph = CsrGraph::from_builder(&builder);

        // Calculate first occurrence positions for each lemma
        let first_positions = self.get_first_positions(tokens, &graph);

        // Build personalization vector
        let personalization =
            position_based_personalization(&first_positions, graph.num_nodes);

        // Run personalized PageRank
        let pagerank = PersonalizedPageRank::new()
            .with_damping(self.config.damping)
            .with_max_iterations(self.config.max_iterations)
            .with_threshold(self.config.convergence_threshold)
            .with_personalization(personalization)
            .run(&graph);

        // Extract phrases
        let extractor = PhraseExtractor::with_config(self.config.clone());
        let phrases = extractor.extract(tokens, &graph, &pagerank);

        ExtractionResult {
            phrases,
            converged: pagerank.converged,
            iterations: pagerank.iterations,
        }
    }

    /// Get the first occurrence position for each lemma in the graph
    fn get_first_positions(&self, tokens: &[Token], graph: &CsrGraph) -> Vec<(u32, usize)> {
        let mut first_positions: FxHashMap<String, usize> = FxHashMap::default();

        // Find first occurrence of each lemma (respecting include_pos config)
        let include_pos = &self.config.include_pos;
        for token in tokens.iter().filter(|t| {
            if t.is_stopword {
                return false;
            }
            if include_pos.is_empty() {
                t.pos.is_content_word()
            } else {
                include_pos.contains(&t.pos)
            }
        }) {
            first_positions
                .entry(token.lemma.clone())
                .or_insert(token.token_idx);
        }

        // Convert to (node_id, position) pairs
        first_positions
            .into_iter()
            .filter_map(|(lemma, position)| {
                graph
                    .get_node_by_lemma(&lemma)
                    .map(|node_id| (node_id, position))
            })
            .collect()
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
    fn test_first_positions() {
        let tokens = make_tokens();
        let builder = GraphBuilder::from_tokens(&tokens, 4, true);
        let graph = CsrGraph::from_builder(&builder);

        let pr = PositionRank::new();
        let first_pos = pr.get_first_positions(&tokens, &graph);

        // "topic" first appears at position 1
        let topic_node = graph.get_node_by_lemma("topic");
        if let Some(node_id) = topic_node {
            let pos = first_pos.iter().find(|(id, _)| *id == node_id);
            assert!(pos.is_some());
            assert_eq!(pos.unwrap().1, 1); // First at token index 1
        }
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
        if early_rank.is_some() && late_rank.is_some() {
            assert!(early_rank.unwrap() < late_rank.unwrap());
        }
    }
}
