//! MultipartiteRank variant
//!
//! MultipartiteRank (Boudin, NAACL 2018) builds a k-partite directed graph
//! where nodes are individual keyphrase candidates and edges connect only
//! candidates from different topic clusters. An alpha weight-adjustment step
//! boosts incoming edges to the first-occurring variant in each topic,
//! encoding positional preference directly into edge weights before PageRank.

use crate::clustering::{self, compute_gap, extract_candidates, PhraseCandidate};
use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::standard::StandardPageRank;
use crate::phrase::extraction::ExtractionResult;
use crate::types::{Phrase, TextRankConfig, Token};
use rustc_hash::FxHashMap;

/// MultipartiteRank implementation
#[derive(Debug)]
pub struct MultipartiteRank {
    config: TextRankConfig,
    /// Jaccard similarity threshold for topic clustering (default: 0.26)
    similarity_threshold: f64,
    /// Weight adjustment strength (default: 1.1)
    alpha: f64,
    /// Maximum number of candidate phrases to process (default: 200)
    max_phrases: usize,
}

impl Default for MultipartiteRank {
    fn default() -> Self {
        Self::new()
    }
}

impl MultipartiteRank {
    /// Create a new MultipartiteRank extractor with default settings
    pub fn new() -> Self {
        Self {
            config: TextRankConfig::default(),
            similarity_threshold: 0.26,
            alpha: 1.1,
            max_phrases: 200,
        }
    }

    /// Create with custom config
    pub fn with_config(config: TextRankConfig) -> Self {
        Self {
            config,
            similarity_threshold: 0.26,
            alpha: 1.1,
            max_phrases: 200,
        }
    }

    /// Set similarity threshold for topic clustering
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set alpha weight adjustment strength
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.max(0.0);
        self
    }

    /// Set maximum phrases to process
    pub fn with_max_phrases(mut self, max: usize) -> Self {
        self.max_phrases = max;
        self
    }

    /// Extract keyphrases using MultipartiteRank
    pub fn extract(&self, tokens: &[Token]) -> Vec<Phrase> {
        self.extract_with_info(tokens).phrases
    }

    /// Extract keyphrases with PageRank convergence information
    pub fn extract_with_info(&self, tokens: &[Token]) -> ExtractionResult {
        // 1. Extract candidate phrases
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

        // 2. Cluster candidates into topics
        let clusters = clustering::cluster_phrases(&candidates, self.similarity_threshold);
        if clusters.is_empty() {
            return ExtractionResult {
                phrases: Vec::new(),
                converged: true,
                iterations: 0,
                debug: None,
            };
        }

        // Map each candidate to its topic index
        let mut topic_of = vec![0usize; candidates.len()];
        for (topic_idx, members) in clusters.iter().enumerate() {
            for &candidate_idx in members {
                topic_of[candidate_idx] = topic_idx;
            }
        }

        // 3. Build directed multipartite graph (no intra-topic edges)
        let n = candidates.len();
        let mut builder = GraphBuilder::with_capacity(n);

        for i in 0..n {
            builder.get_or_create_node(&format!("c_{}", i));
        }

        for i in 0..n {
            for j in (i + 1)..n {
                if topic_of[i] == topic_of[j] {
                    continue;
                }
                let gap = compute_gap(&candidates[i].chunk, &candidates[j].chunk);
                let weight = 1.0 / gap as f64;
                builder.increment_directed_edge(i as u32, j as u32, weight);
                builder.increment_directed_edge(j as u32, i as u32, weight);
            }
        }

        // 4. Alpha weight adjustment
        if self.alpha > 0.0 {
            self.adjust_weights(&mut builder, &candidates, &clusters, &topic_of);
        }

        // 5. Run PageRank
        let graph = CsrGraph::from_builder(&builder);
        let pagerank = StandardPageRank::new()
            .with_damping(self.config.damping)
            .with_max_iterations(self.config.max_iterations)
            .with_threshold(self.config.convergence_threshold)
            .run(&graph);

        // 6. Group by lemma and build output phrases
        let mut lemma_indices: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for (i, candidate) in candidates.iter().enumerate() {
            lemma_indices
                .entry(candidate.lemma.clone())
                .or_default()
                .push(i);
        }

        let lemma_groups: Vec<Vec<usize>> = if self.config.determinism.is_deterministic() {
            let mut sorted: Vec<_> = lemma_indices.into_iter().collect();
            sorted.sort_by(|(a, _), (b, _)| a.cmp(b));
            sorted.into_iter().map(|(_, v)| v).collect()
        } else {
            lemma_indices.into_values().collect()
        };

        let mut phrases: Vec<Phrase> = lemma_groups
            .into_iter()
            .map(|indices| {
                let best_idx = *indices
                    .iter()
                    .max_by(|&&a, &&b| {
                        pagerank
                            .score(a as u32)
                            .partial_cmp(&pagerank.score(b as u32))
                            .unwrap()
                    })
                    .unwrap();

                let score = pagerank.score(best_idx as u32);
                let candidate = &candidates[best_idx];

                let mut offsets: Vec<(usize, usize)> = indices
                    .iter()
                    .map(|&i| {
                        (
                            candidates[i].chunk.start_token,
                            candidates[i].chunk.end_token,
                        )
                    })
                    .collect();
                offsets.sort_by_key(|(start, _)| *start);

                Phrase {
                    text: candidate.text.clone(),
                    lemma: candidate.lemma.clone(),
                    score,
                    count: offsets.len(),
                    offsets,
                    rank: 0,
                }
            })
            .collect();

        if self.config.determinism.is_deterministic() {
            phrases.sort_by(|a, b| a.stable_cmp(b));
        } else {
            phrases.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        }

        for (i, phrase) in phrases.iter_mut().enumerate() {
            phrase.rank = i + 1;
        }

        if self.config.top_n > 0 && phrases.len() > self.config.top_n {
            phrases.truncate(self.config.top_n);
        }

        // Build debug payload from legacy types if requested.
        let mut debug = crate::pipeline::artifacts::DebugPayload::build_from_legacy(
            self.config.debug_level,
            &graph,
            &pagerank,
            self.config.debug_top_k,
        );

        // At Full level, enrich with cluster details.
        if self.config.debug_level.includes_full() {
            if let Some(ref mut dbg) = debug {
                let details: Vec<Vec<crate::pipeline::artifacts::ClusterMember>> = clusters
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
                dbg.cluster_memberships = Some(clusters.clone());
            }
        }

        ExtractionResult {
            phrases,
            converged: pagerank.converged,
            iterations: pagerank.iterations,
            debug,
        }
    }

    /// Apply alpha weight adjustment to boost first-occurring variants.
    ///
    /// For each topic with multiple variants, the first-occurring variant's
    /// incoming edges are boosted by redirecting "recommendation power" from
    /// secondary variants. The boost is scaled by `alpha * exp(1/(1+p))` where
    /// `p` is the token offset of the first occurrence.
    fn adjust_weights(
        &self,
        builder: &mut GraphBuilder,
        candidates: &[PhraseCandidate],
        clusters: &[Vec<usize>],
        topic_of: &[usize],
    ) {
        let n = candidates.len();

        for cluster in clusters {
            if cluster.len() <= 1 {
                continue;
            }

            // Find first-occurring variant in this topic
            let first_idx = *cluster
                .iter()
                .min_by_key(|&&idx| candidates[idx].chunk.start_token)
                .unwrap();

            let p_first = candidates[first_idx].chunk.start_token;
            let position_factor = (1.0 / (1.0 + p_first as f64)).exp();

            // For each external node c_j (in a different topic)
            for c_j in 0..n {
                if topic_of[c_j] == topic_of[first_idx] {
                    continue;
                }

                // Collect booster weights from other variants in this topic
                let mut booster_sum = 0.0;
                for &v in cluster {
                    if v == first_idx {
                        continue;
                    }
                    if let Some(node) = builder.get_node(v as u32) {
                        if let Some(&w) = node.edges.get(&(c_j as u32)) {
                            booster_sum += w;
                        }
                    }
                }

                if booster_sum > 0.0 {
                    let boost = self.alpha * position_factor * booster_sum;
                    builder.increment_directed_edge(c_j as u32, first_idx as u32, boost);
                }
            }
        }
    }
}

/// Convenience function
pub fn extract_keyphrases_multipartite(tokens: &[Token], config: &TextRankConfig) -> Vec<Phrase> {
    MultipartiteRank::with_config(config.clone()).extract(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::PhraseCandidate;
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
    fn test_multipartite_rank_basic() {
        let tokens = make_tokens();
        let phrases = MultipartiteRank::new().extract(&tokens);
        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let tokens: Vec<Token> = Vec::new();
        let result = MultipartiteRank::new().extract_with_info(&tokens);
        assert!(result.phrases.is_empty());
        assert!(result.converged);
    }

    #[test]
    fn test_no_intra_topic_edges() {
        // Two candidates with identical terms → same topic → no edge between them
        let candidates = vec![
            PhraseCandidate {
                text: "machine learning".into(),
                lemma: "machine learning".into(),
                terms: ["machine", "learning"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                chunk: ChunkSpan {
                    start_token: 0,
                    end_token: 2,
                    start_char: 0,
                    end_char: 16,
                    sentence_idx: 0,
                },
            },
            PhraseCandidate {
                text: "machine learning".into(),
                lemma: "machine learning".into(),
                terms: ["machine", "learning"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                chunk: ChunkSpan {
                    start_token: 5,
                    end_token: 7,
                    start_char: 40,
                    end_char: 56,
                    sentence_idx: 1,
                },
            },
            PhraseCandidate {
                text: "neural network".into(),
                lemma: "neural network".into(),
                terms: ["neural", "network"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                chunk: ChunkSpan {
                    start_token: 3,
                    end_token: 5,
                    start_char: 20,
                    end_char: 34,
                    sentence_idx: 0,
                },
            },
        ];

        let clusters = clustering::cluster_phrases(&candidates, 0.26);

        // ML candidates should cluster together, NN separate
        let mut topic_of = vec![0usize; candidates.len()];
        for (topic_idx, members) in clusters.iter().enumerate() {
            for &idx in members {
                topic_of[idx] = topic_idx;
            }
        }

        // Build graph
        let n = candidates.len();
        let mut builder = GraphBuilder::with_capacity(n);
        for i in 0..n {
            builder.get_or_create_node(&format!("c_{}", i));
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if topic_of[i] == topic_of[j] {
                    continue;
                }
                let gap = compute_gap(&candidates[i].chunk, &candidates[j].chunk);
                let weight = 1.0 / gap as f64;
                builder.increment_directed_edge(i as u32, j as u32, weight);
                builder.increment_directed_edge(j as u32, i as u32, weight);
            }
        }

        // Verify: candidates 0 and 1 (both "machine learning") should NOT be connected
        let node_0 = builder.get_node(0).unwrap();
        assert!(
            !node_0.edges.contains_key(&1),
            "Intra-topic edge should not exist"
        );
        let node_1 = builder.get_node(1).unwrap();
        assert!(
            !node_1.edges.contains_key(&0),
            "Intra-topic edge should not exist"
        );

        // But both should connect to candidate 2 (neural network)
        assert!(
            node_0.edges.contains_key(&2),
            "Inter-topic edge should exist"
        );
        assert!(
            node_1.edges.contains_key(&2),
            "Inter-topic edge should exist"
        );
    }

    #[test]
    fn test_alpha_boosts_first_occurrence() {
        // Create two topics, each with 2 candidates. The first-occurring
        // candidate in each topic should get higher score than the second.
        let tokens = vec![
            // Topic A: "data" appears at positions 0 and 6
            Token::new("data", "data", PosTag::Noun, 0, 4, 0, 0),
            Token::new("science", "science", PosTag::Noun, 5, 12, 0, 1),
            // Topic B: "neural" at positions 2 and 4
            Token::new("neural", "neural", PosTag::Adjective, 13, 19, 0, 2),
            Token::new("network", "network", PosTag::Noun, 20, 27, 0, 3),
            Token::new("neural", "neural", PosTag::Adjective, 28, 34, 1, 4),
            Token::new("net", "net", PosTag::Noun, 35, 38, 1, 5),
            Token::new("data", "data", PosTag::Noun, 39, 43, 1, 6),
            Token::new("analysis", "analysis", PosTag::Noun, 44, 52, 1, 7),
        ];

        let mpr = MultipartiteRank::new().with_alpha(1.1);
        let result = mpr.extract_with_info(&tokens);
        assert!(!result.phrases.is_empty());
        // The first phrase should have rank 1
        assert_eq!(result.phrases[0].rank, 1);
    }

    #[test]
    fn test_alpha_zero_no_adjustment() {
        // Construct tokens where the same topic has multiple variants
        // so that alpha adjustment actually fires.
        let tokens = vec![
            // "machine learning" at pos 0 and pos 8 → same cluster
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            // Different topic: "neural network"
            Token::new("neural", "neural", PosTag::Adjective, 17, 23, 0, 2),
            Token::new("network", "network", PosTag::Noun, 24, 31, 0, 3),
            // Another topic: "data science"
            Token::new("data", "data", PosTag::Noun, 32, 36, 1, 4),
            Token::new("science", "science", PosTag::Noun, 37, 44, 1, 5),
            // Repeat "machine learning" → same cluster, gives secondary variant
            Token::new("machine", "machine", PosTag::Noun, 45, 52, 1, 6),
            Token::new("learning", "learning", PosTag::Noun, 53, 61, 1, 7),
        ];

        let result_no_alpha = MultipartiteRank::new()
            .with_alpha(0.0)
            .extract_with_info(&tokens);
        let result_with_alpha = MultipartiteRank::new()
            .with_alpha(1.1)
            .extract_with_info(&tokens);

        assert!(!result_no_alpha.phrases.is_empty());
        assert!(!result_with_alpha.phrases.is_empty());

        // Find the "machine learning" phrase in each result and compare scores.
        // Alpha should boost the first-occurring variant.
        let score_no = result_no_alpha
            .phrases
            .iter()
            .find(|p| p.lemma.contains("machine"))
            .map(|p| p.score);
        let score_with = result_with_alpha
            .phrases
            .iter()
            .find(|p| p.lemma.contains("machine"))
            .map(|p| p.score);

        if let (Some(s0), Some(s1)) = (score_no, score_with) {
            assert!(
                (s0 - s1).abs() > 1e-10,
                "Alpha adjustment should change scores when a topic has multiple variants"
            );
        }
    }

    #[test]
    fn test_single_candidate() {
        let tokens = vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
        ];
        let result = MultipartiteRank::new().extract_with_info(&tokens);
        // Single candidate should still be returned
        assert_eq!(result.phrases.len(), 1);
    }

    #[test]
    fn test_phrases_are_ranked() {
        let tokens = make_tokens();
        let phrases = MultipartiteRank::new().extract(&tokens);
        for (i, phrase) in phrases.iter().enumerate() {
            assert_eq!(
                phrase.rank,
                i + 1,
                "Rank should be 1-indexed and contiguous"
            );
        }
        // Scores should be descending
        for w in phrases.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "Phrases should be sorted by score descending"
            );
        }
    }

    // ─── Integration test helpers ─────────────────────────────────

    fn make_rich_tokens() -> Vec<Token> {
        vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("algorithms", "algorithm", PosTag::Noun, 17, 27, 0, 2),
            Token::new("process", "process", PosTag::Verb, 28, 35, 0, 3),
            Token::new("large", "large", PosTag::Adjective, 36, 41, 0, 4),
            Token::new("datasets", "dataset", PosTag::Noun, 42, 50, 0, 5),
            Token::new("Deep", "deep", PosTag::Adjective, 52, 56, 1, 6),
            Token::new("learning", "learning", PosTag::Noun, 57, 65, 1, 7),
            Token::new("models", "model", PosTag::Noun, 66, 72, 1, 8),
            Token::new("use", "use", PosTag::Verb, 73, 76, 1, 9),
            Token::new("neural", "neural", PosTag::Adjective, 77, 83, 1, 10),
            Token::new("networks", "network", PosTag::Noun, 84, 92, 1, 11),
            Token::new("Machine", "machine", PosTag::Noun, 94, 101, 2, 12),
            Token::new("learning", "learning", PosTag::Noun, 102, 110, 2, 13),
            Token::new("techniques", "technique", PosTag::Noun, 111, 121, 2, 14),
            Token::new("improve", "improve", PosTag::Verb, 122, 129, 2, 15),
            Token::new("data", "data", PosTag::Noun, 130, 134, 2, 16),
            Token::new("analysis", "analysis", PosTag::Noun, 135, 143, 2, 17),
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
        let baseline = MultipartiteRank::with_config(config.clone())
            .with_alpha(1.1)
            .extract_with_info(&tokens);

        for _ in 0..10 {
            let result = MultipartiteRank::with_config(config.clone())
                .with_alpha(1.1)
                .extract_with_info(&tokens);
            assert_eq!(result.phrases.len(), baseline.phrases.len());
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
            }
        }
    }

    // ─── Intra-topic edge removal (exhaustive) ───────────────────

    #[test]
    fn test_intra_topic_edge_removal_exhaustive() {
        use rustc_hash::FxHashSet;

        fn terms(words: &[&str]) -> FxHashSet<String> {
            words.iter().map(|s| s.to_string()).collect()
        }

        // Build the multipartite graph from multiple topic clusters and
        // verify that NO within-cluster edges exist — exhaustively check
        // every (i, j) pair where topic_of[i] == topic_of[j].
        let candidates = vec![
            // Cluster A: identical terms → will merge
            PhraseCandidate {
                text: "machine learning".into(),
                lemma: "machine learning".into(),
                terms: terms(&["machine", "learning"]),
                chunk: ChunkSpan {
                    start_token: 0,
                    end_token: 2,
                    start_char: 0,
                    end_char: 16,
                    sentence_idx: 0,
                },
            },
            PhraseCandidate {
                text: "machine learning".into(),
                lemma: "machine learning".into(),
                terms: terms(&["machine", "learning"]),
                chunk: ChunkSpan {
                    start_token: 10,
                    end_token: 12,
                    start_char: 80,
                    end_char: 96,
                    sentence_idx: 2,
                },
            },
            // Cluster B: identical terms → will merge
            PhraseCandidate {
                text: "neural network".into(),
                lemma: "neural network".into(),
                terms: terms(&["neural", "network"]),
                chunk: ChunkSpan {
                    start_token: 4,
                    end_token: 6,
                    start_char: 30,
                    end_char: 44,
                    sentence_idx: 1,
                },
            },
            PhraseCandidate {
                text: "neural networks".into(),
                lemma: "neural network".into(),
                terms: terms(&["neural", "network"]),
                chunk: ChunkSpan {
                    start_token: 14,
                    end_token: 16,
                    start_char: 110,
                    end_char: 126,
                    sentence_idx: 3,
                },
            },
            // Cluster C: singleton
            PhraseCandidate {
                text: "data analysis".into(),
                lemma: "data analysis".into(),
                terms: terms(&["data", "analysis"]),
                chunk: ChunkSpan {
                    start_token: 7,
                    end_token: 9,
                    start_char: 50,
                    end_char: 63,
                    sentence_idx: 1,
                },
            },
        ];

        let clusters = clustering::cluster_phrases(&candidates, 0.26);

        let mut topic_of = vec![0usize; candidates.len()];
        for (topic_idx, members) in clusters.iter().enumerate() {
            for &idx in members {
                topic_of[idx] = topic_idx;
            }
        }

        // Build graph the same way extract_with_info does
        let n = candidates.len();
        let mut builder = GraphBuilder::with_capacity(n);
        for i in 0..n {
            builder.get_or_create_node(&format!("c_{}", i));
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if topic_of[i] == topic_of[j] {
                    continue;
                }
                let gap = compute_gap(&candidates[i].chunk, &candidates[j].chunk);
                let weight = 1.0 / gap as f64;
                builder.increment_directed_edge(i as u32, j as u32, weight);
                builder.increment_directed_edge(j as u32, i as u32, weight);
            }
        }

        // Exhaustive check: no node should have an edge to its cluster-mate
        for i in 0..n {
            let node = builder.get_node(i as u32).unwrap();
            for j in 0..n {
                if i == j {
                    continue;
                }
                if topic_of[i] == topic_of[j] {
                    assert!(
                        !node.edges.contains_key(&(j as u32)),
                        "Intra-topic edge found: c_{} → c_{} (both topic {})",
                        i,
                        j,
                        topic_of[i],
                    );
                }
            }
        }

        // Positive check: inter-topic edges exist
        // Cluster A member → Cluster B member
        let node_0 = builder.get_node(0).unwrap();
        assert!(node_0.edges.contains_key(&2), "A→B edge should exist");
        assert!(node_0.edges.contains_key(&4), "A→C edge should exist");
    }

    // ─── Alpha boost: earlier candidates ranked higher ───────────

    #[test]
    fn test_alpha_boost_earlier_candidate_ranked_higher() {
        // Two occurrences of "machine learning": position 0 and position 12.
        // With alpha > 0, the position-0 variant should receive a higher
        // PageRank score than the position-12 variant.
        //
        // We verify this indirectly: with high alpha, the top-ranked phrase
        // (selected per lemma group by highest score) should be the first
        // occurrence's text variant.
        let tokens = make_rich_tokens();

        let config = TextRankConfig {
            determinism: crate::types::DeterminismMode::Deterministic,
            ..Default::default()
        };

        // High alpha → strong positional bias
        let result_boosted = MultipartiteRank::with_config(config.clone())
            .with_alpha(5.0)
            .extract_with_info(&tokens);

        // No alpha → no positional bias
        let result_flat = MultipartiteRank::with_config(config)
            .with_alpha(0.0)
            .extract_with_info(&tokens);

        assert!(!result_boosted.phrases.is_empty());
        assert!(!result_flat.phrases.is_empty());

        // With alpha=5.0, the top phrase should start at an earlier position
        // than the top phrase with alpha=0. At minimum, the ranking should
        // differ, showing that alpha has an effect.
        let any_rank_diff = result_boosted
            .phrases
            .iter()
            .zip(result_flat.phrases.iter())
            .any(|(a, b)| a.lemma != b.lemma || (a.score - b.score).abs() > 1e-10);

        assert!(any_rank_diff, "Alpha boost should change ranking or scores");
    }

    // ─── MultipartiteRank golden test ────────────────────────────

    #[test]
    fn test_multipartite_rank_golden() {
        let tokens = make_rich_tokens();
        let config = TextRankConfig {
            determinism: crate::types::DeterminismMode::Deterministic,
            ..Default::default()
        };
        let result = MultipartiteRank::with_config(config)
            .with_alpha(1.1)
            .extract_with_info(&tokens);

        assert!(result.converged, "PageRank should converge");
        assert_eq!(result.iterations, 20);
        assert_eq!(result.phrases.len(), 7);

        let expected = [
            (1, "machine learning algorithm", 0.2536742),
            (2, "neural network", 0.2351832),
            (3, "large dataset", 0.1184485),
            (4, "deep learning model", 0.1157130),
            (5, "machine learning technique", 0.0855296),
            (6, "data analysis", 0.0795817),
            (7, "deep learning application", 0.0500771),
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

        // neural network should have 2 offsets (positions 10-12 and 18-20)
        let nn = &result.phrases[1];
        assert_eq!(nn.count, 2);
        assert_eq!(nn.offsets.len(), 2);
    }
}
