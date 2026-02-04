//! BiasedTextRank variant
//!
//! BiasedTextRank allows specifying "focus" terms that should be prioritized.
//! This is useful for topic-specific keyword extraction where you want
//! keywords related to a particular concept.
//!
//! Focus terms receive a higher personalization weight, biasing the random
//! walk towards them and their neighboring nodes.

use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::personalized::{focus_based_personalization, PersonalizedPageRank};
use crate::phrase::extraction::{ExtractionResult, PhraseExtractor};
use crate::types::{Phrase, PosTag, TextRankConfig, Token};

/// BiasedTextRank implementation
#[derive(Debug)]
pub struct BiasedTextRank {
    config: TextRankConfig,
    /// Focus terms (lemmatized)
    focus_terms: Vec<String>,
    /// Weight multiplier for focus terms
    bias_weight: f64,
    /// Cached graph and PageRank result for re-ranking
    cached_graph: Option<CsrGraph>,
    cached_builder: Option<GraphBuilder>,
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
            cached_graph: None,
            cached_builder: None,
        }
    }

    /// Create with custom config
    pub fn with_config(config: TextRankConfig) -> Self {
        Self {
            config,
            focus_terms: Vec::new(),
            bias_weight: 5.0,
            cached_graph: None,
            cached_builder: None,
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
    pub fn extract(&mut self, tokens: &[Token]) -> Vec<Phrase> {
        self.extract_with_info(tokens).phrases
    }

    /// Extract keyphrases with PageRank convergence information
    pub fn extract_with_info(&mut self, tokens: &[Token]) -> ExtractionResult {
        // Build graph (cache it for re-ranking) with POS filtering
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
            self.config.use_pos_in_nodes,
        );

        if builder.is_empty() {
            return ExtractionResult {
                phrases: Vec::new(),
                converged: true,
                iterations: 0,
            };
        }

        let graph = CsrGraph::from_builder(&builder);

        // Find focus term node IDs
        let focus_nodes = self.focus_node_ids(&graph);

        // Build personalization vector
        let personalization =
            focus_based_personalization(&focus_nodes, self.bias_weight, graph.num_nodes);

        // Run personalized PageRank
        let pagerank = PersonalizedPageRank::new()
            .with_damping(self.config.damping)
            .with_max_iterations(self.config.max_iterations)
            .with_threshold(self.config.convergence_threshold)
            .with_personalization(personalization)
            .run(&graph);

        // Cache for potential re-ranking
        self.cached_graph = Some(graph.clone());
        self.cached_builder = Some(builder);

        // Extract phrases
        let extractor = PhraseExtractor::with_config(self.config.clone());
        let phrases = extractor.extract(tokens, &graph, &pagerank);

        ExtractionResult {
            phrases,
            converged: pagerank.converged,
            iterations: pagerank.iterations,
        }
    }

    /// Change focus and re-rank without rebuilding the graph
    ///
    /// This is efficient for exploring different topic focuses on the same document.
    pub fn change_focus(&mut self, new_focus: &[&str], tokens: &[Token]) -> Option<Vec<Phrase>> {
        let graph = self.cached_graph.as_ref()?;

        // Update focus terms
        self.focus_terms = new_focus.iter().map(|s| s.to_lowercase()).collect();

        // Find new focus term node IDs
        let focus_nodes = self.focus_node_ids(graph);

        // Build new personalization vector
        let personalization =
            focus_based_personalization(&focus_nodes, self.bias_weight, graph.num_nodes);

        // Re-run PageRank with new personalization
        let pagerank = PersonalizedPageRank::new()
            .with_damping(self.config.damping)
            .with_max_iterations(self.config.max_iterations)
            .with_threshold(self.config.convergence_threshold)
            .with_personalization(personalization)
            .run(graph);

        // Extract phrases
        let extractor = PhraseExtractor::with_config(self.config.clone());
        Some(extractor.extract(tokens, graph, &pagerank))
    }

    fn focus_node_ids(&self, graph: &CsrGraph) -> Vec<u32> {
        if !self.config.use_pos_in_nodes {
            return self
                .focus_terms
                .iter()
                .filter_map(|term| graph.get_node_by_lemma(term))
                .collect();
        }

        let default_pos = [PosTag::Noun, PosTag::Adjective, PosTag::ProperNoun];
        let pos_tags: &[PosTag] = if self.config.include_pos.is_empty() {
            &default_pos
        } else {
            self.config.include_pos.as_slice()
        };

        let mut nodes = Vec::new();
        for term in &self.focus_terms {
            for pos in pos_tags {
                let key = format!("{}|{}", term, pos.as_str());
                if let Some(node_id) = graph.get_node_by_lemma(&key) {
                    nodes.push(node_id);
                }
            }
        }
        nodes
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
}
