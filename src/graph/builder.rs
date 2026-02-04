//! Graph builder with efficient edge handling
//!
//! This module provides a mutable graph builder that uses FxHashMap
//! for O(1) edge lookups during construction.

use crate::types::{PosTag, Token};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

type EdgeWeights = FxHashMap<(Arc<str>, Arc<str>), f64>;

/// A node in the graph builder
#[derive(Debug, Clone)]
pub struct BuilderNode {
    /// The lemma for this node
    pub lemma: String,
    /// Adjacency list: target node ID -> edge weight
    pub edges: FxHashMap<u32, f64>,
}

impl BuilderNode {
    /// Create a new node
    pub fn new(lemma: impl Into<String>) -> Self {
        Self {
            lemma: lemma.into(),
            edges: FxHashMap::default(),
        }
    }
}

/// A mutable graph builder optimized for incremental construction
#[derive(Debug)]
pub struct GraphBuilder {
    /// Maps lemma -> node ID
    lemma_to_id: FxHashMap<String, u32>,
    /// Node storage
    nodes: Vec<BuilderNode>,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    /// Create a new empty graph builder
    pub fn new() -> Self {
        Self {
            lemma_to_id: FxHashMap::default(),
            nodes: Vec::new(),
        }
    }

    /// Create a graph builder with pre-allocated capacity
    pub fn with_capacity(node_capacity: usize) -> Self {
        Self {
            lemma_to_id: FxHashMap::with_capacity_and_hasher(node_capacity, Default::default()),
            nodes: Vec::with_capacity(node_capacity),
        }
    }

    /// Get or create a node for the given lemma, returning its ID
    pub fn get_or_create_node(&mut self, lemma: &str) -> u32 {
        if let Some(&id) = self.lemma_to_id.get(lemma) {
            return id;
        }

        let id = self.nodes.len() as u32;
        self.lemma_to_id.insert(lemma.to_string(), id);
        self.nodes.push(BuilderNode::new(lemma));
        id
    }

    /// Increment the edge weight between two nodes
    ///
    /// If the edge doesn't exist, it's created with the given weight.
    /// If it exists, the weight is added to the existing weight.
    pub fn increment_edge(&mut self, from: u32, to: u32, weight: f64) {
        if from == to {
            return; // No self-loops
        }

        // Add edge in both directions (undirected graph)
        if let Some(node) = self.nodes.get_mut(from as usize) {
            *node.edges.entry(to).or_insert(0.0) += weight;
        }
        if let Some(node) = self.nodes.get_mut(to as usize) {
            *node.edges.entry(from).or_insert(0.0) += weight;
        }
    }

    /// Set the edge weight between two nodes (binary/unweighted mode)
    ///
    /// If the edge doesn't exist, it's created with the given weight.
    /// If it exists, the weight is NOT modified (edge already exists).
    pub fn set_edge(&mut self, from: u32, to: u32, weight: f64) {
        if from == to {
            return; // No self-loops
        }

        // Add edge in both directions (undirected graph), but don't overwrite
        if let Some(node) = self.nodes.get_mut(from as usize) {
            node.edges.entry(to).or_insert(weight);
        }
        if let Some(node) = self.nodes.get_mut(to as usize) {
            node.edges.entry(from).or_insert(weight);
        }
    }

    /// Build a graph from tokens using a sliding window
    ///
    /// This creates edges between tokens that co-occur within the window.
    /// Uses the default POS filter (Noun, Verb, Adjective, ProperNoun).
    pub fn from_tokens(tokens: &[Token], window_size: usize, use_weights: bool) -> Self {
        Self::from_tokens_with_pos(tokens, window_size, use_weights, None, false)
    }

    /// Build a graph from tokens using a sliding window with custom POS filter
    ///
    /// This creates edges between tokens that co-occur within the window.
    /// If `include_pos` is None, uses the default content word filter.
    /// If `include_pos` is Some, only includes tokens with matching POS tags.
    pub fn from_tokens_with_pos(
        tokens: &[Token],
        window_size: usize,
        use_weights: bool,
        include_pos: Option<&[PosTag]>,
        use_pos_in_nodes: bool,
    ) -> Self {
        let mut builder = Self::with_capacity(tokens.len() / 2);

        // Filter to graph candidates based on POS tags
        let candidates: Vec<_> = tokens
            .iter()
            .filter(|t| {
                if t.is_stopword {
                    return false;
                }
                match include_pos {
                    Some(pos_tags) => pos_tags.contains(&t.pos),
                    None => t.pos.is_content_word(),
                }
            })
            .collect();
        let candidate_keys: Vec<String> = candidates
            .iter()
            .map(|t| t.graph_key(use_pos_in_nodes))
            .collect();

        // Process each sentence separately (don't create edges across sentences)
        let mut i = 0;
        while i < candidates.len() {
            let sent_idx = candidates[i].sentence_idx;

            // Find all candidates in this sentence
            let sent_start = i;
            while i < candidates.len() && candidates[i].sentence_idx == sent_idx {
                i += 1;
            }
            let sent_end = i;

            // Create nodes and edges within the sentence
            for j in sent_start..sent_end {
                let node_j = builder.get_or_create_node(&candidate_keys[j]);

                // Window extends forward
                let window_end = std::cmp::min(j + window_size, sent_end);
                for k in (j + 1)..window_end {
                    let node_k = builder.get_or_create_node(&candidate_keys[k]);
                    if use_weights {
                        // Weighted mode: accumulate co-occurrence counts
                        builder.increment_edge(node_j, node_k, 1.0);
                    } else {
                        // Binary mode: edge exists (1.0) or doesn't, no accumulation
                        builder.set_edge(node_j, node_k, 1.0);
                    }
                }
            }
        }

        builder
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the total number of edges (counting each undirected edge once)
    pub fn edge_count(&self) -> usize {
        self.nodes.iter().map(|n| n.edges.len()).sum::<usize>() / 2
    }

    /// Get a node by ID
    pub fn get_node(&self, id: u32) -> Option<&BuilderNode> {
        self.nodes.get(id as usize)
    }

    /// Get a node ID by lemma
    pub fn get_node_id(&self, lemma: &str) -> Option<u32> {
        self.lemma_to_id.get(lemma).copied()
    }

    /// Get the lemma for a node ID
    pub fn get_lemma(&self, id: u32) -> Option<&str> {
        self.nodes.get(id as usize).map(|n| n.lemma.as_str())
    }

    /// Iterate over all nodes
    pub fn nodes(&self) -> impl Iterator<Item = (u32, &BuilderNode)> {
        self.nodes.iter().enumerate().map(|(i, n)| (i as u32, n))
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Thread-safe counter for parallel graph building
#[derive(Debug, Default)]
pub struct AtomicCounter {
    value: AtomicU32,
}

impl AtomicCounter {
    /// Create a new counter
    pub fn new() -> Self {
        Self {
            value: AtomicU32::new(0),
        }
    }

    /// Increment and return the new value
    pub fn increment(&self) -> u32 {
        self.value.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the current value
    pub fn get(&self) -> u32 {
        self.value.load(Ordering::SeqCst)
    }
}

/// Build a graph from tokens in parallel (for large documents)
///
/// This splits the document into chunks, builds partial graphs in parallel,
/// and then merges them. Uses the default POS filter.
pub fn build_graph_parallel(
    tokens: &[Token],
    window_size: usize,
    use_weights: bool,
) -> GraphBuilder {
    build_graph_parallel_with_pos(tokens, window_size, use_weights, None, false)
}

/// Build a graph from tokens in parallel with custom POS filter
///
/// This splits the document into chunks, builds partial graphs in parallel,
/// and then merges them.
pub fn build_graph_parallel_with_pos(
    tokens: &[Token],
    window_size: usize,
    use_weights: bool,
    include_pos: Option<&[PosTag]>,
    use_pos_in_nodes: bool,
) -> GraphBuilder {
    // For small documents, sequential is faster
    if tokens.len() < 1000 {
        return GraphBuilder::from_tokens_with_pos(
            tokens,
            window_size,
            use_weights,
            include_pos,
            use_pos_in_nodes,
        );
    }

    // Group tokens by sentence for parallel processing
    let mut sentences: Vec<Vec<&Token>> = Vec::new();
    let mut current_sent = Vec::new();
    let mut current_idx = None;

    // Filter tokens by POS tags
    let token_filter = |t: &&Token| {
        if t.is_stopword {
            return false;
        }
        match include_pos {
            Some(pos_tags) => pos_tags.contains(&t.pos),
            None => t.pos.is_content_word(),
        }
    };

    for token in tokens.iter().filter(token_filter) {
        if current_idx != Some(token.sentence_idx) {
            if !current_sent.is_empty() {
                sentences.push(std::mem::take(&mut current_sent));
            }
            current_idx = Some(token.sentence_idx);
        }
        current_sent.push(token);
    }
    if !current_sent.is_empty() {
        sentences.push(current_sent);
    }

    // Use specialized unweighted path for better deduplication with HashSet
    if !use_weights {
        return build_unweighted_parallel(sentences, window_size, use_pos_in_nodes);
    }

    // Convert lemmas to Arc<str> once before parallel processing
    // This avoids expensive String clones during edge pair creation
    let sentences_with_arcs: Vec<Vec<(&Token, Arc<str>)>> = sentences
        .iter()
        .map(|sent_tokens| {
            sent_tokens
                .iter()
                .map(|&t| (t, Arc::from(t.graph_key(use_pos_in_nodes))))
                .collect()
        })
        .collect();

    // Build partial graphs in parallel using Arc<str> for cheap cloning
    let partial_graphs: Vec<EdgeWeights> = sentences_with_arcs
        .par_iter()
        .map(|sent_token_pairs| {
            let mut edges = FxHashMap::default();
            for i in 0..sent_token_pairs.len() {
                for j in (i + 1)..std::cmp::min(i + window_size, sent_token_pairs.len()) {
                    let arc_a = &sent_token_pairs[i].1;
                    let arc_b = &sent_token_pairs[j].1;

                    let (key_a, key_b) = if arc_a <= arc_b {
                        (Arc::clone(arc_a), Arc::clone(arc_b))
                    } else {
                        (Arc::clone(arc_b), Arc::clone(arc_a))
                    };

                    if key_a != key_b {
                        // Weighted mode: accumulate co-occurrence counts
                        *edges.entry((key_a, key_b)).or_insert(0.0) += 1.0;
                    }
                }
            }
            edges
        })
        .collect();

    // Merge partial graphs - accumulate weights across sentences
    let mut builder = GraphBuilder::new();
    for partial in partial_graphs {
        for ((a, b), weight) in partial {
            let id_a = builder.get_or_create_node(a.as_ref());
            let id_b = builder.get_or_create_node(b.as_ref());
            builder.increment_edge(id_a, id_b, weight);
        }
    }

    builder
}

/// Build unweighted graph in parallel using HashSet for efficient deduplication
fn build_unweighted_parallel(
    sentences: Vec<Vec<&Token>>,
    window_size: usize,
    use_pos_in_nodes: bool,
) -> GraphBuilder {
    // Build partial edge sets in parallel - HashSet automatically deduplicates
    let partial_sets: Vec<FxHashSet<(Arc<str>, Arc<str>)>> = sentences
        .par_iter()
        .map(|sent_tokens| {
            // Convert lemmas to Arc<str> once per sentence
            let lemma_arcs: Vec<Arc<str>> = sent_tokens
                .iter()
                .map(|t| Arc::from(t.graph_key(use_pos_in_nodes)))
                .collect();

            let mut edge_set = FxHashSet::default();
            for i in 0..sent_tokens.len() {
                for j in (i + 1)..std::cmp::min(i + window_size, sent_tokens.len()) {
                    let arc_a = &lemma_arcs[i];
                    let arc_b = &lemma_arcs[j];

                    if arc_a != arc_b {
                        let edge = if arc_a <= arc_b {
                            (Arc::clone(arc_a), Arc::clone(arc_b))
                        } else {
                            (Arc::clone(arc_b), Arc::clone(arc_a))
                        };
                        edge_set.insert(edge);
                    }
                }
            }
            edge_set
        })
        .collect();

    // Merge all edge sets - extend automatically handles deduplication
    let mut merged_edges = FxHashSet::default();
    for partial_set in partial_sets {
        merged_edges.extend(partial_set);
    }

    // Build final graph from unique edges
    let mut builder = GraphBuilder::new();
    for (a, b) in merged_edges {
        let id_a = builder.get_or_create_node(a.as_ref());
        let id_b = builder.get_or_create_node(b.as_ref());
        builder.set_edge(id_a, id_b, 1.0);
    }

    builder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    fn make_token(text: &str, lemma: &str, sent_idx: usize, tok_idx: usize) -> Token {
        Token {
            text: text.to_string(),
            lemma: lemma.to_string(),
            pos: PosTag::Noun,
            start: 0,
            end: text.len(),
            sentence_idx: sent_idx,
            token_idx: tok_idx,
            is_stopword: false,
        }
    }

    #[test]
    fn test_graph_builder_basic() {
        let mut builder = GraphBuilder::new();

        let id_a = builder.get_or_create_node("machine");
        let id_b = builder.get_or_create_node("learning");
        let id_c = builder.get_or_create_node("machine"); // duplicate

        assert_eq!(id_a, id_c); // Same lemma should get same ID
        assert_ne!(id_a, id_b);
        assert_eq!(builder.node_count(), 2);
    }

    #[test]
    fn test_edge_incrementing() {
        let mut builder = GraphBuilder::new();

        let id_a = builder.get_or_create_node("machine");
        let id_b = builder.get_or_create_node("learning");

        builder.increment_edge(id_a, id_b, 1.0);
        builder.increment_edge(id_a, id_b, 1.0);

        // Should have weight 2.0 in both directions
        assert_eq!(builder.get_node(id_a).unwrap().edges.get(&id_b), Some(&2.0));
        assert_eq!(builder.get_node(id_b).unwrap().edges.get(&id_a), Some(&2.0));
    }

    #[test]
    fn test_from_tokens() {
        let tokens = vec![
            make_token("machine", "machine", 0, 0),
            make_token("learning", "learning", 0, 1),
            make_token("is", "is", 0, 2), // Will be filtered if stopword
            make_token("great", "great", 0, 3),
        ];

        let builder = GraphBuilder::from_tokens(&tokens, 3, true);

        assert_eq!(builder.node_count(), 4);
        // "machine" should be connected to "learning" and "is" (within window)
        let machine_id = builder.get_node_id("machine").unwrap();
        let node = builder.get_node(machine_id).unwrap();
        assert!(!node.edges.is_empty());
    }

    #[test]
    fn test_no_cross_sentence_edges() {
        let tokens = vec![
            make_token("machine", "machine", 0, 0),
            make_token("learning", "learning", 0, 1),
            make_token("deep", "deep", 1, 2), // Different sentence
            make_token("neural", "neural", 1, 3),
        ];

        let builder = GraphBuilder::from_tokens(&tokens, 3, true);

        // "learning" and "deep" should NOT be connected (different sentences)
        let learning_id = builder.get_node_id("learning").unwrap();
        let deep_id = builder.get_node_id("deep").unwrap();
        let learning_node = builder.get_node(learning_id).unwrap();
        assert!(!learning_node.edges.contains_key(&deep_id));
    }

    #[test]
    fn test_self_loops_prevented() {
        let mut builder = GraphBuilder::new();
        let id_a = builder.get_or_create_node("test");

        builder.increment_edge(id_a, id_a, 1.0);

        // No self-loop should be created
        let node = builder.get_node(id_a).unwrap();
        assert!(node.edges.is_empty());
    }

    #[test]
    fn test_set_edge_no_accumulation() {
        let mut builder = GraphBuilder::new();
        let id_a = builder.get_or_create_node("machine");
        let id_b = builder.get_or_create_node("learning");

        // set_edge should not accumulate
        builder.set_edge(id_a, id_b, 1.0);
        builder.set_edge(id_a, id_b, 1.0);
        builder.set_edge(id_a, id_b, 1.0);

        // Should still have weight 1.0 (not 3.0)
        assert_eq!(builder.get_node(id_a).unwrap().edges.get(&id_b), Some(&1.0));
        assert_eq!(builder.get_node(id_b).unwrap().edges.get(&id_a), Some(&1.0));
    }

    #[test]
    fn test_use_edge_weights_true_accumulates() {
        // Create tokens where same pair co-occurs multiple times
        let tokens = vec![
            make_token("machine", "machine", 0, 0),
            make_token("learning", "learning", 0, 1),
            make_token("machine", "machine", 0, 2),
            make_token("learning", "learning", 0, 3),
        ];

        let builder = GraphBuilder::from_tokens(&tokens, 2, true);

        let machine_id = builder.get_node_id("machine").unwrap();
        let learning_id = builder.get_node_id("learning").unwrap();

        // With use_weights=true, edge weight should be > 1.0 due to accumulation
        let weight = builder
            .get_node(machine_id)
            .unwrap()
            .edges
            .get(&learning_id);
        assert!(weight.is_some());
        assert!(*weight.unwrap() > 1.0, "Expected accumulated weight > 1.0");
    }

    #[test]
    fn test_use_edge_weights_false_binary() {
        // Create tokens where same pair co-occurs multiple times
        let tokens = vec![
            make_token("machine", "machine", 0, 0),
            make_token("learning", "learning", 0, 1),
            make_token("machine", "machine", 0, 2),
            make_token("learning", "learning", 0, 3),
        ];

        let builder = GraphBuilder::from_tokens(&tokens, 2, false);

        let machine_id = builder.get_node_id("machine").unwrap();
        let learning_id = builder.get_node_id("learning").unwrap();

        // With use_weights=false, edge weight should be exactly 1.0 (binary)
        let weight = builder
            .get_node(machine_id)
            .unwrap()
            .edges
            .get(&learning_id);
        assert!(weight.is_some());
        assert_eq!(*weight.unwrap(), 1.0, "Expected binary weight = 1.0");
    }

    #[test]
    fn test_include_pos_filters_by_tag() {
        // Create tokens with different POS tags
        let tokens = vec![
            Token {
                text: "machine".to_string(),
                lemma: "machine".to_string(),
                pos: PosTag::Noun,
                start: 0,
                end: 7,
                sentence_idx: 0,
                token_idx: 0,
                is_stopword: false,
            },
            Token {
                text: "runs".to_string(),
                lemma: "run".to_string(),
                pos: PosTag::Verb,
                start: 8,
                end: 12,
                sentence_idx: 0,
                token_idx: 1,
                is_stopword: false,
            },
            Token {
                text: "fast".to_string(),
                lemma: "fast".to_string(),
                pos: PosTag::Adverb,
                start: 13,
                end: 17,
                sentence_idx: 0,
                token_idx: 2,
                is_stopword: false,
            },
        ];

        // Only include Nouns - should only have "machine"
        let builder_nouns =
            GraphBuilder::from_tokens_with_pos(&tokens, 3, true, Some(&[PosTag::Noun]), false);
        assert_eq!(builder_nouns.node_count(), 1);
        assert!(builder_nouns.get_node_id("machine").is_some());
        assert!(builder_nouns.get_node_id("run").is_none());

        // Only include Verbs - should only have "run"
        let builder_verbs =
            GraphBuilder::from_tokens_with_pos(&tokens, 3, true, Some(&[PosTag::Verb]), false);
        assert_eq!(builder_verbs.node_count(), 1);
        assert!(builder_verbs.get_node_id("run").is_some());
        assert!(builder_verbs.get_node_id("machine").is_none());

        // Include Nouns and Verbs - should have both
        let builder_both = GraphBuilder::from_tokens_with_pos(
            &tokens,
            3,
            true,
            Some(&[PosTag::Noun, PosTag::Verb]),
            false,
        );
        assert_eq!(builder_both.node_count(), 2);
        assert!(builder_both.get_node_id("machine").is_some());
        assert!(builder_both.get_node_id("run").is_some());
    }

    #[test]
    fn test_parallel_use_edge_weights_false_across_sentences() {
        // Create tokens spanning multiple sentences with the same pair
        // to verify the parallel merge respects use_weights=false
        // We need > 1000 tokens to trigger parallel path, so we'll test
        // by calling build_graph_parallel_with_pos directly with a smaller threshold
        // For this test, we verify the merge logic works correctly by building
        // a scenario with multiple sentences containing the same word pair.

        // Build many tokens to ensure we hit the parallel code path
        let mut tokens = Vec::with_capacity(1500);
        for sent_idx in 0..500 {
            // Each sentence has "machine learning" pair
            tokens.push(Token {
                text: "machine".to_string(),
                lemma: "machine".to_string(),
                pos: PosTag::Noun,
                start: 0,
                end: 7,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3,
                is_stopword: false,
            });
            tokens.push(Token {
                text: "learning".to_string(),
                lemma: "learning".to_string(),
                pos: PosTag::Noun,
                start: 8,
                end: 16,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3 + 1,
                is_stopword: false,
            });
            tokens.push(Token {
                text: "system".to_string(),
                lemma: "system".to_string(),
                pos: PosTag::Noun,
                start: 17,
                end: 23,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3 + 2,
                is_stopword: false,
            });
        }

        // With use_weights=false, the edge weight should be 1.0, not 500.0
        let builder = build_graph_parallel_with_pos(&tokens, 2, false, None, false);

        let machine_id = builder.get_node_id("machine").unwrap();
        let learning_id = builder.get_node_id("learning").unwrap();

        let weight = builder
            .get_node(machine_id)
            .unwrap()
            .edges
            .get(&learning_id);
        assert!(weight.is_some());
        assert_eq!(
            *weight.unwrap(),
            1.0,
            "Expected binary weight = 1.0, got {} (parallel merge should not accumulate with use_weights=false)",
            weight.unwrap()
        );
    }

    #[test]
    fn test_parallel_use_edge_weights_true_across_sentences() {
        // Same setup, but with use_weights=true - should accumulate
        let mut tokens = Vec::with_capacity(1500);
        for sent_idx in 0..500 {
            tokens.push(Token {
                text: "machine".to_string(),
                lemma: "machine".to_string(),
                pos: PosTag::Noun,
                start: 0,
                end: 7,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3,
                is_stopword: false,
            });
            tokens.push(Token {
                text: "learning".to_string(),
                lemma: "learning".to_string(),
                pos: PosTag::Noun,
                start: 8,
                end: 16,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3 + 1,
                is_stopword: false,
            });
            tokens.push(Token {
                text: "system".to_string(),
                lemma: "system".to_string(),
                pos: PosTag::Noun,
                start: 17,
                end: 23,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3 + 2,
                is_stopword: false,
            });
        }

        // With use_weights=true, the edge weight should be 500.0 (accumulated)
        let builder = build_graph_parallel_with_pos(&tokens, 2, true, None, false);

        let machine_id = builder.get_node_id("machine").unwrap();
        let learning_id = builder.get_node_id("learning").unwrap();

        let weight = builder
            .get_node(machine_id)
            .unwrap()
            .edges
            .get(&learning_id);
        assert!(weight.is_some());
        assert_eq!(
            *weight.unwrap(),
            500.0,
            "Expected accumulated weight = 500.0 (parallel merge should accumulate with use_weights=true)"
        );
    }

    #[test]
    fn test_parallel_weighted_uses_arc_str() {
        // This test verifies the parallel weighted path uses Arc<str> for edge keys
        // by exercising the code path and checking functional correctness.
        // The Arc<str> usage is enforced at compile time via type annotations in
        // build_graph_parallel_with_pos (line 310: FxHashMap<(Arc<str>, Arc<str>), f64>)

        // Build enough tokens to trigger parallel path (>1000)
        let mut tokens = Vec::with_capacity(1200);
        for sent_idx in 0..400 {
            tokens.push(Token {
                text: "alpha".to_string(),
                lemma: "alpha".to_string(),
                pos: PosTag::Noun,
                start: 0,
                end: 5,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3,
                is_stopword: false,
            });
            tokens.push(Token {
                text: "beta".to_string(),
                lemma: "beta".to_string(),
                pos: PosTag::Noun,
                start: 6,
                end: 10,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3 + 1,
                is_stopword: false,
            });
            tokens.push(Token {
                text: "gamma".to_string(),
                lemma: "gamma".to_string(),
                pos: PosTag::Noun,
                start: 11,
                end: 16,
                sentence_idx: sent_idx,
                token_idx: sent_idx * 3 + 2,
                is_stopword: false,
            });
        }

        // With use_weights=true, parallel path uses Arc<str> in FxHashMap keys
        let builder = build_graph_parallel_with_pos(&tokens, 2, true, None, false);

        // Verify graph structure
        assert_eq!(builder.node_count(), 3, "Should have 3 unique lemmas");

        let alpha_id = builder.get_node_id("alpha").expect("alpha should exist");
        let beta_id = builder.get_node_id("beta").expect("beta should exist");

        // Each sentence has alpha-beta edge, 400 sentences = weight 400
        let weight = builder.get_node(alpha_id).unwrap().edges.get(&beta_id);
        assert_eq!(
            *weight.unwrap(),
            400.0,
            "Arc<str> parallel path should correctly accumulate weights"
        );
    }
}
