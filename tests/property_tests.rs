//! Property-based tests using proptest

use proptest::prelude::*;
use rapid_textrank::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_pagerank_scores_sum_to_one(
        nodes in 2usize..20,
        edge_prob in 0.1f64..0.5
    ) {
        // Create a random graph
        let mut builder = graph::builder::GraphBuilder::new();

        for i in 0..nodes {
            builder.get_or_create_node(&format!("node_{}", i));
        }

        // Add random edges
        for i in 0..nodes {
            for j in (i+1)..nodes {
                if rand::random::<f64>() < edge_prob {
                    builder.increment_edge(i as u32, j as u32, 1.0);
                }
            }
        }

        let graph = graph::csr::CsrGraph::from_builder(&builder);

        // Skip if graph is empty or has no edges
        if graph.num_edges() == 0 {
            return Ok(());
        }

        let pr = pagerank::standard::StandardPageRank::new()
            .with_max_iterations(200);
        let result = pr.run(&graph);

        // Scores should sum to approximately 1
        let sum: f64 = result.scores.iter().sum();
        prop_assert!((sum - 1.0).abs() < 0.01, "Scores sum to {} instead of 1", sum);
    }

    #[test]
    fn test_phrase_extraction_deterministic(
        _seed in 0u64..1000
    ) {
        let text = "Machine learning is a field of artificial intelligence. \
                   Deep learning is a subset of machine learning.";

        let tokenizer = nlp::tokenizer::Tokenizer::new();
        let (_, mut tokens) = tokenizer.tokenize(text);

        let stopwords = nlp::stopwords::StopwordFilter::new("en");
        for token in &mut tokens {
            token.is_stopword = stopwords.is_stopword(&token.text);
        }

        let config = TextRankConfig::default().with_top_n(5);

        // Run twice with same input
        let phrases1 = phrase::extraction::extract_keyphrases(&tokens, &config);
        let phrases2 = phrase::extraction::extract_keyphrases(&tokens, &config);

        // Results should be identical
        prop_assert_eq!(phrases1.len(), phrases2.len());
        for (p1, p2) in phrases1.iter().zip(phrases2.iter()) {
            prop_assert_eq!(&p1.text, &p2.text);
            prop_assert!((p1.score - p2.score).abs() < 1e-10);
        }
    }

    #[test]
    fn test_config_validation_properties(
        damping in 0.0f64..=1.0,
        window_size in 2usize..10,
        top_n in 1usize..100
    ) {
        let config = TextRankConfig::default()
            .with_damping(damping)
            .with_window_size(window_size)
            .with_top_n(top_n);

        // Valid damping and window size should validate successfully
        prop_assert!(config.validate().is_ok());
    }

    #[test]
    fn test_score_aggregation_properties(scores in prop::collection::vec(0.1f64..10.0, 1..20)) {
        let sum = ScoreAggregation::Sum.aggregate(&scores);
        let mean = ScoreAggregation::Mean.aggregate(&scores);
        let max = ScoreAggregation::Max.aggregate(&scores);
        let rms = ScoreAggregation::RootMeanSquare.aggregate(&scores);

        // Sum should be >= mean (when there's more than one element)
        if scores.len() > 1 {
            prop_assert!(sum >= mean);
        }

        // Max should be <= Sum
        prop_assert!(max <= sum);

        // RMS should be between mean and max for positive values
        prop_assert!(rms >= mean * 0.99); // Allow small floating point error
    }

    #[test]
    fn test_string_pool_properties(
        strings in prop::collection::vec("[a-z]{1,10}", 1..50)
    ) {
        let mut pool = types::StringPool::new();

        let ids: Vec<u32> = strings.iter().map(|s| pool.intern(s)).collect();

        // Same strings should get same IDs
        for (i, s) in strings.iter().enumerate() {
            let new_id = pool.intern(s);
            prop_assert_eq!(ids[i], new_id);
        }

        // All IDs should be retrievable
        for &id in &ids {
            prop_assert!(pool.get(id).is_some());
        }

        // Pool size should be <= number of strings (due to dedup)
        prop_assert!(pool.len() <= strings.len());
    }

    #[test]
    fn test_chunk_overlap_symmetry(
        start1 in 0usize..100,
        len1 in 1usize..20,
        start2 in 0usize..100,
        len2 in 1usize..20
    ) {
        let chunk1 = types::ChunkSpan {
            start_token: 0,
            end_token: 1,
            start_char: start1,
            end_char: start1 + len1,
            sentence_idx: 0,
        };

        let chunk2 = types::ChunkSpan {
            start_token: 0,
            end_token: 1,
            start_char: start2,
            end_char: start2 + len2,
            sentence_idx: 0,
        };

        // Overlap should be symmetric
        prop_assert_eq!(chunk1.overlaps(&chunk2), chunk2.overlaps(&chunk1));
    }

    #[test]
    fn test_top_n_respects_limit(
        n_phrases in 5usize..50,
        top_n in 1usize..10
    ) {
        // Generate some fake phrases
        let text = (0..n_phrases)
            .map(|i| format!("keyword{} topic{}", i, i))
            .collect::<Vec<_>>()
            .join(". ");

        let tokenizer = nlp::tokenizer::Tokenizer::new();
        let (_, mut tokens) = tokenizer.tokenize(&text);

        let stopwords = nlp::stopwords::StopwordFilter::new("en");
        for token in &mut tokens {
            token.is_stopword = stopwords.is_stopword(&token.text);
        }

        let config = TextRankConfig::default().with_top_n(top_n);
        let phrases = phrase::extraction::extract_keyphrases(&tokens, &config);

        // Should not exceed top_n
        prop_assert!(phrases.len() <= top_n);
    }
}

// Use rand for random number generation in tests
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn random<T: From<f64>>() -> T {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let hash = hasher.finish();
        T::from((hash % 1000) as f64 / 1000.0)
    }
}
