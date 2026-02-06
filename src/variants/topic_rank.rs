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

use crate::graph::builder::GraphBuilder;
use crate::graph::csr::CsrGraph;
use crate::pagerank::standard::StandardPageRank;
use crate::phrase::chunker::{chunk_lemma, chunk_text, NounChunker};
use crate::types::{Phrase, TextRankConfig, Token};
use rustc_hash::FxHashSet;

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

    fn is_kept_token(&self, token: &Token) -> bool {
        if token.is_stopword {
            return false;
        }
        if self.config.include_pos.is_empty() {
            token.pos.is_content_word()
        } else {
            self.config.include_pos.contains(&token.pos)
        }
    }

    fn trim_chunk_to_first_kept(
        &self,
        tokens: &[Token],
        chunk: &crate::types::ChunkSpan,
    ) -> Option<crate::types::ChunkSpan> {
        let mut start = chunk.start_token;
        let end = chunk.end_token;

        while start < end && !self.is_kept_token(&tokens[start]) {
            start += 1;
        }

        if start >= end {
            return None;
        }

        Some(crate::types::ChunkSpan {
            start_token: start,
            end_token: end,
            start_char: tokens[start].start,
            end_char: tokens[end - 1].end,
            sentence_idx: tokens[start].sentence_idx,
        })
    }

    /// Extract keyphrases using TopicRank
    pub fn extract(&self, tokens: &[Token]) -> Vec<Phrase> {
        // Extract candidate phrases
        let chunker = NounChunker::new()
            .with_min_length(self.config.min_phrase_length)
            .with_max_length(self.config.max_phrase_length);
        let chunks = chunker.extract_chunks(tokens);

        if chunks.is_empty() {
            return Vec::new();
        }

        // Create phrase candidates
        let mut candidates: Vec<PhraseCandidate> = Vec::new();
        for chunk in chunks.iter().take(self.max_phrases) {
            let trimmed = match self.trim_chunk_to_first_kept(tokens, chunk) {
                Some(span) => span,
                None => continue,
            };
            let text = chunk_text(tokens, &trimmed);
            let lemma = chunk_lemma(tokens, &trimmed);
            let terms: FxHashSet<String> = tokens[trimmed.start_token..trimmed.end_token]
                .iter()
                .filter(|t| !t.is_stopword)
                .map(|t| t.text.clone())
                .collect();
            candidates.push(PhraseCandidate {
                text,
                lemma,
                terms,
                chunk: trimmed,
            });
        }

        if candidates.is_empty() {
            return Vec::new();
        }

        // Cluster similar phrases
        let clusters = self.cluster_phrases(&candidates);

        if clusters.is_empty() {
            return Vec::new();
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

        // Sort by score
        phrases.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Assign ranks
        for (i, phrase) in phrases.iter_mut().enumerate() {
            phrase.rank = i + 1;
        }

        // Limit to top_n
        if self.config.top_n > 0 && phrases.len() > self.config.top_n {
            phrases.truncate(self.config.top_n);
        }

        phrases
    }

    /// Cluster phrases using HAC (average linkage) over Jaccard distance
    fn cluster_phrases(&self, candidates: &[PhraseCandidate]) -> Vec<Vec<usize>> {
        let n = candidates.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![vec![0]];
        }

        let mut base_dist = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = jaccard_distance(&candidates[i].terms, &candidates[j].terms);
                base_dist[i][j] = dist;
                base_dist[j][i] = dist;
            }
        }

        let cutoff = (0.99 - self.similarity_threshold).clamp(0.0, 1.0);
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        loop {
            if clusters.len() <= 1 {
                break;
            }

            let mut best_i = 0;
            let mut best_j = 0;
            let mut best_dist = f64::INFINITY;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let dist = average_linkage_distance(&base_dist, &clusters[i], &clusters[j]);
                    if dist < best_dist {
                        best_dist = dist;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if best_dist > cutoff {
                break;
            }

            let mut merged = clusters.remove(best_j);
            clusters[best_i].append(&mut merged);
        }

        clusters
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
                        let source_start = candidates[source_idx].chunk.start_token;
                        let target_start = candidates[target_idx].chunk.start_token;
                        let distance = source_start.abs_diff(target_start);
                        if distance > 0 {
                            weight += 1.0 / distance as f64;
                        }
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

/// A phrase candidate with its stems for clustering
#[derive(Debug)]
struct PhraseCandidate {
    text: String,
    lemma: String,
    terms: FxHashSet<String>,
    chunk: crate::types::ChunkSpan,
}

/// Jaccard distance between two sets
fn jaccard_distance(a: &FxHashSet<String>, b: &FxHashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        1.0
    } else {
        1.0 - (intersection as f64 / union as f64)
    }
}

fn average_linkage_distance(
    base_dist: &[Vec<f64>],
    cluster_a: &[usize],
    cluster_b: &[usize],
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;

    for &i in cluster_a {
        for &j in cluster_b {
            sum += base_dist[i][j];
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Convenience function
pub fn extract_keyphrases_topic(tokens: &[Token], config: &TextRankConfig) -> Vec<Phrase> {
    TopicRank::with_config(config.clone()).extract(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

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
        let a: FxHashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b: FxHashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();

        let dist = jaccard_distance(&a, &b);
        // Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        // Jaccard distance = 1 - (2/4) = 0.5
        assert!((dist - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_distance_identical() {
        let a: FxHashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let dist = jaccard_distance(&a, &a);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_distance_disjoint() {
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
        use crate::types::ChunkSpan;

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

        let topic_rank = TopicRank::new().with_similarity_threshold(0.25);

        // Disjoint candidates stay separate (distance 1.0 > 0.74)
        {
            let candidates = vec![
                make_candidate(&["a", "b"], 0),
                make_candidate(&["c", "d"], 1),
                make_candidate(&["e", "f"], 2),
            ];
            let clusters = topic_rank.cluster_phrases(&candidates);
            assert_eq!(clusters.len(), 3);
        }

        // Identical candidates merge (distance 0.0 <= 0.74)
        {
            let candidates = vec![
                make_candidate(&["a", "b"], 0),
                make_candidate(&["a", "b"], 1),
            ];
            let clusters = topic_rank.cluster_phrases(&candidates);
            assert_eq!(clusters.len(), 1);
        }

        // High overlap merges, low overlap stays separate
        {
            let candidates = vec![
                make_candidate(&["a", "b"], 0),
                make_candidate(&["a", "b", "c"], 1),
                make_candidate(&["x", "y", "z"], 2),
            ];
            let clusters = topic_rank.cluster_phrases(&candidates);
            assert_eq!(clusters.len(), 2);
        }

        // Average linkage prevents chaining when distance exceeds cutoff
        {
            let candidates = vec![
                make_candidate(&["a", "b"], 0),
                make_candidate(&["a", "b", "c"], 1),
                make_candidate(&["c", "d"], 2),
            ];
            let clusters = topic_rank.cluster_phrases(&candidates);
            assert_eq!(clusters.len(), 2);
        }

        // Low overlap with shared term stays separate
        {
            let candidates = vec![
                make_candidate(&["a", "b", "c", "d", "e"], 0),
                make_candidate(&["a", "x", "y", "z", "w"], 1),
            ];
            let clusters = topic_rank.cluster_phrases(&candidates);
            assert_eq!(clusters.len(), 2);
        }
    }
}
