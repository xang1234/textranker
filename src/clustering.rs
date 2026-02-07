//! Shared clustering utilities for topic-based keyword extraction
//!
//! Used by TopicRank and MultipartiteRank to cluster keyphrase candidates
//! into topic groups using hierarchical agglomerative clustering (HAC)
//! with Jaccard distance and average linkage.

use crate::phrase::chunker::{chunk_lemma, chunk_text, NounChunker};
use crate::types::{ChunkSpan, PosTag, Token};
use rustc_hash::FxHashSet;

/// A keyphrase candidate with its term set for clustering
#[derive(Debug)]
pub struct PhraseCandidate {
    pub text: String,
    pub lemma: String,
    pub terms: FxHashSet<String>,
    pub chunk: ChunkSpan,
}

/// Extract keyphrase candidates from tokens using noun chunking.
///
/// Candidates are noun phrases trimmed to start at the first POS-matching token.
/// At most `max_phrases` chunks are considered.
pub fn extract_candidates(
    tokens: &[Token],
    min_phrase_length: usize,
    max_phrase_length: usize,
    max_phrases: usize,
    include_pos: &[PosTag],
) -> Vec<PhraseCandidate> {
    let chunker = NounChunker::new()
        .with_min_length(min_phrase_length)
        .with_max_length(max_phrase_length);
    let chunks = chunker.extract_chunks(tokens);

    let mut candidates = Vec::new();
    for chunk in chunks.iter().take(max_phrases) {
        let trimmed = match trim_chunk_to_first_kept(tokens, chunk, include_pos) {
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
    candidates
}

fn is_kept_token(token: &Token, include_pos: &[PosTag]) -> bool {
    if token.is_stopword {
        return false;
    }
    if include_pos.is_empty() {
        token.pos.is_content_word()
    } else {
        include_pos.contains(&token.pos)
    }
}

fn trim_chunk_to_first_kept(
    tokens: &[Token],
    chunk: &ChunkSpan,
    include_pos: &[PosTag],
) -> Option<ChunkSpan> {
    let mut start = chunk.start_token;
    let end = chunk.end_token;

    while start < end && !is_kept_token(&tokens[start], include_pos) {
        start += 1;
    }

    if start >= end {
        return None;
    }

    Some(ChunkSpan {
        start_token: start,
        end_token: end,
        start_char: tokens[start].start,
        end_char: tokens[end - 1].end,
        sentence_idx: tokens[start].sentence_idx,
    })
}

/// Cluster phrases using HAC (average linkage) over Jaccard distance.
///
/// Returns clusters as vectors of indices into the candidates slice.
/// `similarity_threshold` is the Jaccard similarity cutoff (e.g. 0.25 for
/// TopicRank, 0.26 for MultipartiteRank). Internally converted to distance
/// cutoff `0.99 - threshold`.
pub fn cluster_phrases(
    candidates: &[PhraseCandidate],
    similarity_threshold: f64,
) -> Vec<Vec<usize>> {
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

    let cutoff = (0.99 - similarity_threshold).clamp(0.0, 1.0);
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

/// Jaccard distance between two term sets
pub fn jaccard_distance(a: &FxHashSet<String>, b: &FxHashSet<String>) -> f64 {
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

/// Average linkage distance between two clusters
pub fn average_linkage_distance(
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

/// Compute PKE-style positional gap between two chunks.
/// Measures distance from the end of the earlier phrase to the start of the later,
/// floored at 1 to avoid division by zero.
pub fn compute_gap(a: &ChunkSpan, b: &ChunkSpan) -> usize {
    let raw = a.start_token.abs_diff(b.start_token);
    let span_adjust = if a.start_token < b.start_token {
        (a.end_token - a.start_token).saturating_sub(1)
    } else if a.start_token > b.start_token {
        (b.end_token - b.start_token).saturating_sub(1)
    } else {
        0
    };
    raw.saturating_sub(span_adjust).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_distance() {
        let a: FxHashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b: FxHashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let dist = jaccard_distance(&a, &b);
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

        assert_eq!(compute_gap(&span(0, 1), &span(1, 2)), 1);
        assert_eq!(compute_gap(&span(0, 2), &span(2, 4)), 1);
        assert_eq!(compute_gap(&span(3, 5), &span(3, 5)), 1);
        assert_eq!(compute_gap(&span(0, 3), &span(10, 12)), 8);
        assert_eq!(compute_gap(&span(10, 12), &span(0, 3)), 8);
        assert_eq!(compute_gap(&span(0, 5), &span(3, 7)), 1);
    }

    #[test]
    fn test_cluster_disjoint_stay_separate() {
        let candidates = vec![
            make_candidate(&["a", "b"], 0),
            make_candidate(&["c", "d"], 1),
            make_candidate(&["e", "f"], 2),
        ];
        let clusters = cluster_phrases(&candidates, 0.25);
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn test_cluster_identical_merge() {
        let candidates = vec![
            make_candidate(&["a", "b"], 0),
            make_candidate(&["a", "b"], 1),
        ];
        let clusters = cluster_phrases(&candidates, 0.25);
        assert_eq!(clusters.len(), 1);
    }

    #[test]
    fn test_cluster_high_overlap_merges() {
        let candidates = vec![
            make_candidate(&["a", "b"], 0),
            make_candidate(&["a", "b", "c"], 1),
            make_candidate(&["x", "y", "z"], 2),
        ];
        let clusters = cluster_phrases(&candidates, 0.25);
        assert_eq!(clusters.len(), 2);
    }

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
}
