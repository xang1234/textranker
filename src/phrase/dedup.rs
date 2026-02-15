//! Overlap resolution for phrase extraction
//!
//! When multiple noun chunks overlap, we need to select the best ones.
//! Strategy: sort by position, then by score descending, keep non-overlapping.

use crate::pipeline::artifacts::DroppedCandidate;
use crate::types::ChunkSpan;

/// A chunk with an associated score
#[derive(Debug, Clone)]
pub struct ScoredChunk {
    pub chunk: ChunkSpan,
    pub score: f64,
    pub text: String,
    pub lemma: String,
}

/// Remove overlapping chunks, keeping higher-scored ones
///
/// Strategy:
/// 1. Sort by start position
/// 2. For overlapping chunks, keep the one with higher score
/// 3. Return non-overlapping chunks in document order
pub fn resolve_overlaps(mut chunks: Vec<ScoredChunk>) -> Vec<ScoredChunk> {
    if chunks.is_empty() {
        return chunks;
    }

    // Sort by start position, then by score descending for ties
    chunks.sort_by(|a, b| {
        a.chunk
            .start_char
            .cmp(&b.chunk.start_char)
            .then_with(|| b.score.partial_cmp(&a.score).unwrap())
    });

    let mut result = Vec::new();
    let mut last_end = 0;

    for chunk in chunks {
        // If this chunk doesn't overlap with the last accepted chunk
        if chunk.chunk.start_char >= last_end {
            last_end = chunk.chunk.end_char;
            result.push(chunk);
        } else if chunk.score > result.last().map(|c| c.score).unwrap_or(0.0) {
            // This chunk overlaps but has higher score - replace the lower-scoring one
            result.pop();
            last_end = chunk.chunk.end_char;
            result.push(chunk);
        }
    }

    result
}

/// Alternative strategy: greedy by score
///
/// 1. Sort all chunks by score descending
/// 2. Greedily select chunks that don't overlap with already selected ones
pub fn resolve_overlaps_greedy(mut chunks: Vec<ScoredChunk>) -> Vec<ScoredChunk> {
    if chunks.is_empty() {
        return chunks;
    }

    // Sort by score descending
    chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut result = Vec::new();

    for chunk in chunks {
        // Check if this chunk overlaps with any already selected chunk
        let overlaps = result
            .iter()
            .any(|selected: &ScoredChunk| chunk.chunk.overlaps(&selected.chunk));

        if !overlaps {
            result.push(chunk);
        }
    }

    // Sort result by position for natural reading order
    result.sort_by_key(|c| c.chunk.start_char);

    result
}

/// Greedy overlap resolution with diagnostics.
///
/// Returns the kept chunks **and** a record of every chunk that was dropped
/// (with the reason: overlapped a higher-scored chunk).
pub fn resolve_overlaps_greedy_with_diagnostics(
    mut chunks: Vec<ScoredChunk>,
) -> (Vec<ScoredChunk>, Vec<DroppedCandidate>) {
    let mut dropped = Vec::new();
    if chunks.is_empty() {
        return (chunks, dropped);
    }

    // Sort by score descending
    chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut result = Vec::new();

    for chunk in chunks {
        // Find the first selected chunk that overlaps with this one
        let blocking = result
            .iter()
            .find(|selected: &&ScoredChunk| chunk.chunk.overlaps(&selected.chunk));

        if let Some(blocker) = blocking {
            dropped.push(DroppedCandidate {
                text: chunk.text.clone(),
                lemma: chunk.lemma.clone(),
                score: chunk.score,
                token_range: (chunk.chunk.start_token, chunk.chunk.end_token),
                reason: crate::pipeline::artifacts::DropReason::OverlapWithHigherScored {
                    kept_text: blocker.text.clone(),
                    kept_score: blocker.score,
                },
            });
        } else {
            result.push(chunk);
        }
    }

    // Sort result by position for natural reading order
    result.sort_by_key(|c| c.chunk.start_char);

    (result, dropped)
}

/// Merge adjacent chunks that share the same lemma pattern
pub fn merge_adjacent_chunks(chunks: &[ScoredChunk]) -> Vec<ScoredChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut current = chunks[0].clone();

    for chunk in chunks.iter().skip(1) {
        // Check if chunks are adjacent and in same sentence
        if chunk.chunk.sentence_idx == current.chunk.sentence_idx
            && chunk.chunk.start_token == current.chunk.end_token
        {
            // Merge the chunks
            current.chunk.end_token = chunk.chunk.end_token;
            current.chunk.end_char = chunk.chunk.end_char;
            current.text = format!("{} {}", current.text, chunk.text);
            current.lemma = format!("{} {}", current.lemma, chunk.lemma);
            current.score += chunk.score;
        } else {
            result.push(current);
            current = chunk.clone();
        }
    }
    result.push(current);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(start: usize, end: usize, score: f64, text: &str) -> ScoredChunk {
        ScoredChunk {
            chunk: ChunkSpan {
                start_token: 0,
                end_token: 1,
                start_char: start,
                end_char: end,
                sentence_idx: 0,
            },
            score,
            text: text.to_string(),
            lemma: text.to_lowercase(),
        }
    }

    #[test]
    fn test_no_overlap() {
        let chunks = vec![
            make_chunk(0, 10, 1.0, "first"),
            make_chunk(15, 25, 2.0, "second"),
            make_chunk(30, 40, 1.5, "third"),
        ];

        let result = resolve_overlaps(chunks);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_simple_overlap() {
        let chunks = vec![
            make_chunk(0, 15, 1.0, "first phrase"),
            make_chunk(10, 25, 2.0, "overlapping phrase"), // Overlaps with first, higher score
        ];

        let result = resolve_overlaps(chunks);
        // Higher-scoring chunk replaces the lower-scoring one
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "overlapping phrase");
    }

    #[test]
    fn test_greedy_prefers_higher_score() {
        let chunks = vec![
            make_chunk(0, 15, 1.0, "lower score"),
            make_chunk(5, 20, 2.0, "higher score"), // Overlaps but higher score
        ];

        let result = resolve_overlaps_greedy(chunks);
        // Greedy keeps higher score
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "higher score");
    }

    #[test]
    fn test_empty_input() {
        let result = resolve_overlaps(vec![]);
        assert!(result.is_empty());

        let result = resolve_overlaps_greedy(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_multiple_overlaps() {
        let chunks = vec![
            make_chunk(0, 10, 1.0, "a"),
            make_chunk(5, 15, 1.5, "b"),  // Overlaps a
            make_chunk(12, 20, 2.0, "c"), // Overlaps b
            make_chunk(25, 35, 1.0, "d"), // No overlap
        ];

        let result = resolve_overlaps_greedy(chunks);
        // Should get "c" (highest score among overlapping) and "d"
        assert!(result.len() >= 2);
        assert!(result.iter().any(|c| c.text == "c"));
        assert!(result.iter().any(|c| c.text == "d"));
    }

    #[test]
    fn test_resolve_overlaps_higher_score_replaces() {
        // Test that resolve_overlaps correctly replaces lower-score chunk with higher-score
        let chunks = vec![
            make_chunk(0, 15, 1.0, "lower score"),   // First chunk
            make_chunk(10, 25, 3.0, "higher score"), // Overlaps, higher score
        ];

        let result = resolve_overlaps(chunks);
        // The higher-scoring chunk should replace the lower-scoring one
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "higher score");
        assert_eq!(result[0].score, 3.0);
    }

    #[test]
    fn test_resolve_overlaps_lower_score_ignored() {
        // Test that later chunk with lower score doesn't replace earlier chunk
        let chunks = vec![
            make_chunk(0, 15, 3.0, "higher score"), // First chunk, high score
            make_chunk(10, 25, 1.0, "lower score"), // Overlaps, lower score
        ];

        let result = resolve_overlaps(chunks);
        // The first (higher-scoring) chunk should be kept
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "higher score");
    }

    // ─── Diagnostics tests ──────────────────────────────────────────

    #[test]
    fn test_greedy_diagnostics_records_overlap_drops() {
        let chunks = vec![
            make_chunk(0, 15, 2.0, "higher"),
            make_chunk(5, 20, 1.0, "lower"), // Overlaps, lower score → should be dropped
            make_chunk(25, 35, 1.5, "no_overlap"), // No overlap → kept
        ];

        let (kept, dropped) = resolve_overlaps_greedy_with_diagnostics(chunks);

        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|c| c.text == "higher"));
        assert!(kept.iter().any(|c| c.text == "no_overlap"));

        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].text, "lower");
        assert_eq!(dropped[0].score, 1.0);
        match &dropped[0].reason {
            crate::pipeline::artifacts::DropReason::OverlapWithHigherScored {
                kept_text,
                kept_score,
            } => {
                assert_eq!(kept_text, "higher");
                assert_eq!(*kept_score, 2.0);
            }
            other => panic!("Expected OverlapWithHigherScored, got {:?}", other),
        }
    }

    #[test]
    fn test_greedy_diagnostics_no_overlap_produces_empty_drops() {
        let chunks = vec![
            make_chunk(0, 10, 1.0, "a"),
            make_chunk(15, 25, 2.0, "b"),
            make_chunk(30, 40, 1.5, "c"),
        ];

        let (kept, dropped) = resolve_overlaps_greedy_with_diagnostics(chunks);

        assert_eq!(kept.len(), 3);
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_greedy_diagnostics_empty_input() {
        let (kept, dropped) = resolve_overlaps_greedy_with_diagnostics(vec![]);
        assert!(kept.is_empty());
        assert!(dropped.is_empty());
    }

    #[test]
    fn test_greedy_diagnostics_multiple_drops() {
        // "best" (score 3.0) overlaps with both "mid" (1.5) and "low" (0.5)
        let chunks = vec![
            make_chunk(0, 20, 3.0, "best"),
            make_chunk(5, 15, 1.5, "mid"),
            make_chunk(10, 25, 0.5, "low"),
        ];

        let (kept, dropped) = resolve_overlaps_greedy_with_diagnostics(chunks);

        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].text, "best");

        assert_eq!(dropped.len(), 2);
        // Both should reference "best" as the blocker
        for d in &dropped {
            match &d.reason {
                crate::pipeline::artifacts::DropReason::OverlapWithHigherScored {
                    kept_text,
                    ..
                } => {
                    assert_eq!(kept_text, "best");
                }
                other => panic!("Expected OverlapWithHigherScored, got {:?}", other),
            }
        }
    }
}
