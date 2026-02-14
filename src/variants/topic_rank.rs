//! TopicRank variant
//!
//! TopicRank groups similar keyphrases into topics before ranking.
//! This produces more diverse keywords by ensuring each "topic cluster"
//! contributes at most one representative phrase.
//!
//! Process:
//! 1. Extract candidate phrases
//! 2. Cluster candidates with HAC (average linkage) over Jaccard distance
//! 3. Build a complete graph where nodes are clusters
//! 4. Run PageRank on the cluster graph
//! 5. Select the first occurring phrase from each top cluster

use crate::clustering::{self, compute_gap, extract_candidates, PhraseCandidate};
use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::standard::StandardPageRank;
use crate::phrase::extraction::ExtractionResult;
use crate::types::{Phrase, TextRankConfig, Token};

/// TopicRank implementation
#[derive(Debug)]
pub struct TopicRank {
    config: TextRankConfig,
    /// Jaccard similarity threshold for clustering
    similarity_threshold: f64,
    /// Edge weight multiplier for topic graph edges
    edge_weight: f64,
    /// Maximum number of phrases to cluster (for performance)
    max_phrases: usize,
}

impl Default for TopicRank {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicRank {
    /// Create a new TopicRank extractor
    pub fn new() -> Self {
        Self {
            config: TextRankConfig::default(),
            similarity_threshold: 0.25,
            edge_weight: 1.0,
            max_phrases: 200,
        }
    }

    /// Create with custom config
    pub fn with_config(config: TextRankConfig) -> Self {
        Self {
            config,
            similarity_threshold: 0.25,
            edge_weight: 1.0,
            max_phrases: 200,
        }
    }

    /// Set similarity threshold for clustering
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set edge weight multiplier for topic graph edges
    pub fn with_edge_weight(mut self, weight: f64) -> Self {
        self.edge_weight = weight.max(0.0);
        self
    }

    /// Set maximum phrases to process
    pub fn with_max_phrases(mut self, max: usize) -> Self {
        self.max_phrases = max;
        self
    }

    /// Extract keyphrases using TopicRank
    pub fn extract(&self, tokens: &[Token]) -> Vec<Phrase> {
        self.extract_with_info(tokens).phrases
    }

    /// Extract keyphrases with PageRank convergence information
    pub fn extract_with_info(&self, tokens: &[Token]) -> ExtractionResult {
        // Extract candidate phrases
        let candidates = extract_candidates(
            tokens,
            self.config.min_phrase_length,
            self.config.max_phrase_length,
            self.max_phrases,
            &self.config.include_pos,
        );

        if candidates.is_empty() {
            return ExtractionResult {
                phrases: Vec::new(),
                converged: true,
                iterations: 0,
                debug: None,
            };
        }

        // Cluster similar phrases
        let clusters = clustering::cluster_phrases(&candidates, self.similarity_threshold);

        if clusters.is_empty() {
            return ExtractionResult {
                phrases: Vec::new(),
                converged: true,
                iterations: 0,
                debug: None,
            };
        }

        // Build cluster graph
        let (cluster_graph, cluster_members) = self.build_cluster_graph(&clusters, &candidates);

        // Run PageRank on cluster graph
        let pagerank = StandardPageRank::new()
            .with_damping(self.config.damping)
            .with_max_iterations(self.config.max_iterations)
            .with_threshold(self.config.convergence_threshold)
            .run(&cluster_graph);

        // Select best phrase from each top cluster
        let mut phrases = self.select_representatives(&cluster_members, &candidates, &pagerank);

        // Sort by score (with stable tie-breakers in deterministic mode).
        if self.config.determinism.is_deterministic() {
            phrases.sort_by(|a, b| a.stable_cmp(b));
        } else {
            phrases.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        }

        // Assign ranks
        for (i, phrase) in phrases.iter_mut().enumerate() {
            phrase.rank = i + 1;
        }

        // Limit to top_n
        if self.config.top_n > 0 && phrases.len() > self.config.top_n {
            phrases.truncate(self.config.top_n);
        }

        // Build debug payload from legacy types if requested.
        let mut debug = crate::pipeline::artifacts::DebugPayload::build_from_legacy(
            self.config.debug_level,
            &cluster_graph,
            &pagerank,
            self.config.debug_top_k,
        );

        // At Full level, enrich with cluster details.
        if self.config.debug_level.includes_full() {
            if let Some(ref mut dbg) = debug {
                let details: Vec<Vec<crate::pipeline::artifacts::ClusterMember>> = cluster_members
                    .iter()
                    .map(|members| {
                        members
                            .iter()
                            .map(|&idx| crate::pipeline::artifacts::ClusterMember {
                                index: idx,
                                text: candidates[idx].text.clone(),
                                lemma: candidates[idx].lemma.clone(),
                            })
                            .collect()
                    })
                    .collect();
                dbg.cluster_details = Some(details);

                // Also populate cluster_memberships at Full level.
                dbg.cluster_memberships = Some(cluster_members.clone());
            }
        }

        ExtractionResult {
            phrases,
            converged: pagerank.converged,
            iterations: pagerank.iterations,
            debug,
        }
    }

    /// Build a graph where nodes are clusters and edges connect co-occurring clusters
    fn build_cluster_graph(
        &self,
        clusters: &[Vec<usize>],
        candidates: &[PhraseCandidate],
    ) -> (CsrGraph, Vec<Vec<usize>>) {
        let mut builder = GraphBuilder::with_capacity(clusters.len());

        // Create node for each cluster
        for (i, _) in clusters.iter().enumerate() {
            builder.get_or_create_node(&format!("cluster_{}", i));
        }

        // Complete graph with inverse-distance weights
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let mut weight = 0.0;
                for &source_idx in &clusters[i] {
                    for &target_idx in &clusters[j] {
                        let gap = compute_gap(
                            &candidates[source_idx].chunk,
                            &candidates[target_idx].chunk,
                        );
                        weight += 1.0 / gap as f64;
                    }
                }
                if weight > 0.0 {
                    builder.increment_edge(i as u32, j as u32, weight * self.edge_weight);
                }
            }
        }

        let graph = CsrGraph::from_builder(&builder);
        (graph, clusters.to_vec())
    }

    /// Select the best representative phrase from each cluster
    fn select_representatives(
        &self,
        clusters: &[Vec<usize>],
        candidates: &[PhraseCandidate],
        pagerank: &crate::pagerank::PageRankResult,
    ) -> Vec<Phrase> {
        clusters
            .iter()
            .enumerate()
            .map(|(cluster_idx, members)| {
                // Get cluster score from PageRank
                let cluster_score = pagerank.score(cluster_idx as u32);

                // Select representative: first occurring candidate
                let best_idx = members
                    .iter()
                    .min_by_key(|&&idx| candidates[idx].chunk.start_token)
                    .copied()
                    .unwrap_or(members[0]);

                let candidate = &candidates[best_idx];
                let mut offsets: Vec<(usize, usize)> = members
                    .iter()
                    .map(|&idx| {
                        (
                            candidates[idx].chunk.start_token,
                            candidates[idx].chunk.end_token,
                        )
                    })
                    .collect();
                offsets.sort_by_key(|(start, _)| *start);

                Phrase {
                    text: candidate.text.clone(),
                    lemma: candidate.lemma.clone(),
                    score: cluster_score,
                    count: members.len(),
                    offsets,
                    rank: 0,
                }
            })
            .collect()
    }
}

/// Convenience function
pub fn extract_keyphrases_topic(tokens: &[Token], config: &TextRankConfig) -> Vec<Phrase> {
    TopicRank::with_config(config.clone()).extract(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::{jaccard_distance, PhraseCandidate};
    use crate::types::{ChunkSpan, PosTag};

    fn make_tokens() -> Vec<Token> {
        vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("algorithms", "algorithm", PosTag::Noun, 17, 27, 0, 2),
            Token::new("Deep", "deep", PosTag::Adjective, 29, 33, 1, 3),
            Token::new("learning", "learning", PosTag::Noun, 34, 42, 1, 4),
            Token::new("models", "model", PosTag::Noun, 43, 49, 1, 5),
            Token::new("Neural", "neural", PosTag::Adjective, 51, 57, 2, 6),
            Token::new("networks", "network", PosTag::Noun, 58, 66, 2, 7),
        ]
    }

    #[test]
    fn test_topic_rank() {
        let tokens = make_tokens();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_topic(&tokens, &config);

        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_jaccard_distance() {
        use rustc_hash::FxHashSet;
        let a: FxHashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b: FxHashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();

        let dist = jaccard_distance(&a, &b);
        // Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        // Jaccard distance = 1 - (2/4) = 0.5
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_distance_identical() {
        use rustc_hash::FxHashSet;
        let a: FxHashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let dist = jaccard_distance(&a, &a);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_distance_disjoint() {
        use rustc_hash::FxHashSet;
        let a: FxHashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let b: FxHashSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();
        let dist = jaccard_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let tokens: Vec<Token> = Vec::new();
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_topic(&tokens, &config);

        assert!(phrases.is_empty());
    }

    #[test]
    fn test_candidate_trim_by_pos() {
        let tokens = vec![
            Token::new("Deep", "deep", PosTag::Adjective, 0, 4, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 5, 13, 0, 1),
        ];
        let config = TextRankConfig {
            include_pos: vec![PosTag::Noun],
            ..Default::default()
        };
        let phrases = extract_keyphrases_topic(&tokens, &config);
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].text, "learning");
    }

    #[test]
    fn test_representative_first_occurrence_and_offsets() {
        let tokens = vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("Machine", "machine", PosTag::Noun, 17, 24, 1, 2),
            Token::new("learning", "learning", PosTag::Noun, 25, 33, 1, 3),
        ];
        let config = TextRankConfig::default();
        let phrases = extract_keyphrases_topic(&tokens, &config);
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].text, "Machine learning");
        assert_eq!(phrases[0].offsets.len(), 2);
        assert_eq!(phrases[0].offsets[0].0, 0);
    }

    #[test]
    fn test_hac_clustering_thresholds() {
        fn make_candidate(terms: &[&str], start_token: usize) -> PhraseCandidate {
            PhraseCandidate {
                text: terms.join(" "),
                lemma: terms.join(" "),
                terms: terms.iter().map(|s| s.to_string()).collect(),
                chunk: ChunkSpan {
                    start_token,
                    end_token: start_token + 1,
                    start_char: 0,
                    end_char: 1,
                    sentence_idx: 0,
                },
            }
        }

        // Disjoint candidates stay separate (distance 1.0 > 0.74)
        {
            let candidates = vec![
                make_candidate(&["a", "b"], 0),
                make_candidate(&["c", "d"], 1),
                make_candidate(&["e", "f"], 2),
            ];
            let clusters = clustering::cluster_phrases(&candidates, 0.25);
            assert_eq!(clusters.len(), 3);
        }

        // Identical candidates merge (distance 0.0 <= 0.74)
        {
            let candidates = vec![
                make_candidate(&["a", "b"], 0),
                make_candidate(&["a", "b"], 1),
            ];
            let clusters = clustering::cluster_phrases(&candidates, 0.25);
            assert_eq!(clusters.len(), 1);
        }

        // High overlap merges, low overlap stays separate
        {
            let candidates = vec![
                make_candidate(&["a", "b"], 0),
                make_candidate(&["a", "b", "c"], 1),
                make_candidate(&["x", "y", "z"], 2),
            ];
            let clusters = clustering::cluster_phrases(&candidates, 0.25);
            assert_eq!(clusters.len(), 2);
        }

        // Average linkage prevents chaining when distance exceeds cutoff
        {
            let candidates = vec![
                make_candidate(&["a", "b"], 0),
                make_candidate(&["a", "b", "c"], 1),
                make_candidate(&["c", "d"], 2),
            ];
            let clusters = clustering::cluster_phrases(&candidates, 0.25);
            assert_eq!(clusters.len(), 2);
        }

        // Low overlap with shared term stays separate
        {
            let candidates = vec![
                make_candidate(&["a", "b", "c", "d", "e"], 0),
                make_candidate(&["a", "x", "y", "z", "w"], 1),
            ];
            let clusters = clustering::cluster_phrases(&candidates, 0.25);
            assert_eq!(clusters.len(), 2);
        }
    }

    // ─── Integration test helpers ─────────────────────────────────

    /// Richer 4-sentence corpus with repeated phrases across topics.
    /// Used by golden tests, determinism tests, and structural checks.
    fn make_rich_tokens() -> Vec<Token> {
        vec![
            // Sentence 0: "Machine learning algorithms process large datasets"
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("algorithms", "algorithm", PosTag::Noun, 17, 27, 0, 2),
            Token::new("process", "process", PosTag::Verb, 28, 35, 0, 3),
            Token::new("large", "large", PosTag::Adjective, 36, 41, 0, 4),
            Token::new("datasets", "dataset", PosTag::Noun, 42, 50, 0, 5),
            // Sentence 1: "Deep learning models use neural networks"
            Token::new("Deep", "deep", PosTag::Adjective, 52, 56, 1, 6),
            Token::new("learning", "learning", PosTag::Noun, 57, 65, 1, 7),
            Token::new("models", "model", PosTag::Noun, 66, 72, 1, 8),
            Token::new("use", "use", PosTag::Verb, 73, 76, 1, 9),
            Token::new("neural", "neural", PosTag::Adjective, 77, 83, 1, 10),
            Token::new("networks", "network", PosTag::Noun, 84, 92, 1, 11),
            // Sentence 2: "Machine learning techniques improve data analysis"
            Token::new("Machine", "machine", PosTag::Noun, 94, 101, 2, 12),
            Token::new("learning", "learning", PosTag::Noun, 102, 110, 2, 13),
            Token::new("techniques", "technique", PosTag::Noun, 111, 121, 2, 14),
            Token::new("improve", "improve", PosTag::Verb, 122, 129, 2, 15),
            Token::new("data", "data", PosTag::Noun, 130, 134, 2, 16),
            Token::new("analysis", "analysis", PosTag::Noun, 135, 143, 2, 17),
            // Sentence 3: "Neural networks enable deep learning applications"
            Token::new("Neural", "neural", PosTag::Adjective, 145, 151, 3, 18),
            Token::new("networks", "network", PosTag::Noun, 152, 160, 3, 19),
            Token::new("enable", "enable", PosTag::Verb, 161, 167, 3, 20),
            Token::new("deep", "deep", PosTag::Adjective, 168, 172, 3, 21),
            Token::new("learning", "learning", PosTag::Noun, 173, 181, 3, 22),
            Token::new("applications", "application", PosTag::Noun, 182, 194, 3, 23),
        ]
    }

    // ─── Clustering determinism ──────────────────────────────────

    #[test]
    fn test_clustering_determinism() {
        let tokens = make_rich_tokens();
        let config = TextRankConfig {
            determinism: crate::types::DeterminismMode::Deterministic,
            ..Default::default()
        };
        let baseline = TopicRank::with_config(config.clone()).extract_with_info(&tokens);

        for _ in 0..10 {
            let result = TopicRank::with_config(config.clone()).extract_with_info(&tokens);
            assert_eq!(
                result.phrases.len(),
                baseline.phrases.len(),
                "Number of phrases should be stable across runs"
            );
            for (b, r) in baseline.phrases.iter().zip(result.phrases.iter()) {
                assert_eq!(b.text, r.text, "Phrase text order should be deterministic");
                assert_eq!(b.lemma, r.lemma);
                assert!(
                    (b.score - r.score).abs() < 1e-12,
                    "Scores should be bitwise-identical: {} vs {}",
                    b.score,
                    r.score,
                );
                assert_eq!(b.rank, r.rank);
                assert_eq!(b.offsets, r.offsets);
            }
        }
    }

    // ─── TopicRank golden test ───────────────────────────────────

    #[test]
    fn test_topic_rank_golden() {
        let tokens = make_rich_tokens();
        let config = TextRankConfig {
            determinism: crate::types::DeterminismMode::Deterministic,
            ..Default::default()
        };
        let result = TopicRank::with_config(config).extract_with_info(&tokens);

        assert!(result.converged, "PageRank should converge");
        assert_eq!(result.iterations, 17);
        assert_eq!(result.phrases.len(), 6);

        // Pin exact phrase order, lemmas, and approximate scores.
        let expected = [
            (1, "neural network", 0.2579807),
            (2, "machine learning algorithm", 0.2114519),
            (3, "deep learning model", 0.1531721),
            (4, "data analysis", 0.1479700),
            (5, "large dataset", 0.1399592),
            (6, "deep learning application", 0.0894658),
        ];
        for (phrase, &(rank, lemma, approx_score)) in result.phrases.iter().zip(expected.iter()) {
            assert_eq!(phrase.rank, rank, "rank mismatch for {:?}", phrase.lemma);
            assert_eq!(phrase.lemma, lemma, "lemma mismatch at rank {}", rank);
            assert!(
                (phrase.score - approx_score).abs() < 1e-5,
                "score mismatch for {}: expected ~{}, got {}",
                lemma,
                approx_score,
                phrase.score,
            );
        }

        // Verify multi-occurrence phrases have correct offset counts.
        let neural = &result.phrases[0];
        assert_eq!(neural.count, 2, "neural network appears 2x");
        assert_eq!(neural.offsets.len(), 2);

        let ml = &result.phrases[1];
        assert_eq!(
            ml.count, 2,
            "machine learning algorithm cluster has 2 members"
        );
    }

    #[test]
    fn test_compute_gap() {
        fn span(start: usize, end: usize) -> ChunkSpan {
            ChunkSpan {
                start_token: start,
                end_token: end,
                start_char: 0,
                end_char: 0,
                sentence_idx: 0,
            }
        }

        // Adjacent single-token phrases: gap = |0 - 1| - 0 = 1
        assert_eq!(compute_gap(&span(0, 1), &span(1, 2)), 1);

        // Adjacent multi-token phrases: A=[0,2), B=[2,4)
        // raw=2, span_adjust=(2-0)-1=1, gap=2-1=1
        assert_eq!(compute_gap(&span(0, 2), &span(2, 4)), 1);

        // Same position: gap floors at 1
        assert_eq!(compute_gap(&span(3, 5), &span(3, 5)), 1);

        // Distant phrases: A=[0,3), B=[10,12)
        // raw=10, span_adjust=(3-0)-1=2, gap=10-2=8
        assert_eq!(compute_gap(&span(0, 3), &span(10, 12)), 8);

        // Reversed order: A=[10,12), B=[0,3)
        // raw=10, span_adjust=(3-0)-1=2, gap=10-2=8
        assert_eq!(compute_gap(&span(10, 12), &span(0, 3)), 8);

        // Overlapping phrases: A=[0,5), B=[3,7)
        // raw=3, span_adjust=(5-0)-1=4, gap=max(3-4,0)=0 -> floored to 1
        assert_eq!(compute_gap(&span(0, 5), &span(3, 7)), 1);
    }
}
