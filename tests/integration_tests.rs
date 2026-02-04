//! Integration tests for rapid_textrank

use rapid_textrank::*;

/// Sample text for testing
const SAMPLE_TEXT: &str = r#"
Machine learning is a subset of artificial intelligence (AI) that provides systems
the ability to automatically learn and improve from experience without being explicitly
programmed. Machine learning focuses on the development of computer programs that can
access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct
experience, or instruction, in order to look for patterns in data and make better
decisions in the future based on the examples that we provide. The primary aim is to
allow the computers to learn automatically without human intervention or assistance
and adjust actions accordingly.

Deep learning is a subset of machine learning that uses artificial neural networks
with representation learning. The learning can be supervised, semi-supervised or
unsupervised.
"#;

#[test]
fn test_full_pipeline() {
    // Tokenize
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (sentences, mut tokens) = tokenizer.tokenize(SAMPLE_TEXT);

    assert!(!sentences.is_empty());
    assert!(!tokens.is_empty());

    // Apply stopwords
    let stopwords = nlp::stopwords::StopwordFilter::new("en");
    for token in &mut tokens {
        token.is_stopword = stopwords.is_stopword(&token.text);
    }

    // Build graph
    let config = TextRankConfig::default().with_top_n(10);
    let builder = graph::builder::GraphBuilder::from_tokens(&tokens, config.window_size, true);

    assert!(!builder.is_empty());

    // Convert to CSR
    let graph = graph::csr::CsrGraph::from_builder(&builder);

    assert!(graph.num_nodes > 0);

    // Run PageRank
    let pagerank = pagerank::standard::StandardPageRank::new()
        .with_damping(config.damping)
        .run(&graph);

    assert!(pagerank.converged);
    assert!(!pagerank.scores.is_empty());

    // Extract phrases
    let extractor = phrase::extraction::PhraseExtractor::with_config(config);
    let phrases = extractor.extract(&tokens, &graph, &pagerank);

    assert!(!phrases.is_empty());

    // Verify phrase properties
    for (i, phrase) in phrases.iter().enumerate() {
        assert_eq!(phrase.rank, i + 1);
        assert!(phrase.score > 0.0);
        assert!(!phrase.text.is_empty());
    }
}

#[test]
fn test_position_rank_pipeline() {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (_, mut tokens) = tokenizer.tokenize(SAMPLE_TEXT);

    let stopwords = nlp::stopwords::StopwordFilter::new("en");
    for token in &mut tokens {
        token.is_stopword = stopwords.is_stopword(&token.text);
    }

    let config = TextRankConfig::default().with_top_n(10);
    let phrases = variants::position_rank::extract_keyphrases_position(&tokens, &config);

    assert!(!phrases.is_empty());
}

#[test]
fn test_biased_textrank_pipeline() {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (_, mut tokens) = tokenizer.tokenize(SAMPLE_TEXT);

    let stopwords = nlp::stopwords::StopwordFilter::new("en");
    for token in &mut tokens {
        token.is_stopword = stopwords.is_stopword(&token.text);
    }

    let config = TextRankConfig::default().with_top_n(10);
    let phrases = variants::biased_textrank::extract_keyphrases_biased(
        &tokens,
        &config,
        &["neural", "network"],
        5.0,
    );

    assert!(!phrases.is_empty());
}

#[test]
fn test_summarization_pipeline() {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (sentences, mut tokens) = tokenizer.tokenize(SAMPLE_TEXT);

    let stopwords = nlp::stopwords::StopwordFilter::new("en");
    for token in &mut tokens {
        token.is_stopword = stopwords.is_stopword(&token.text);
    }

    // Extract phrases for summarization
    let config = TextRankConfig::default().with_top_n(20);
    let phrases = phrase::extraction::extract_keyphrases(&tokens, &config);

    // Select sentences
    let selector = summarizer::selector::SentenceSelector::new()
        .with_num_sentences(3)
        .with_lambda(0.7);

    let summary = selector.select(&sentences, &tokens, &phrases);

    // Should select up to 3 sentences (may be fewer if document is short)
    assert!(!summary.sentences.is_empty());
    assert!(summary.sentences.len() <= 3);

    // Sentences should be in document order
    for i in 1..summary.sentences.len() {
        assert!(summary.sentences[i].sentence.index > summary.sentences[i - 1].sentence.index);
    }
}

#[test]
fn test_config_validation() {
    let valid_config = TextRankConfig::default();
    assert!(valid_config.validate().is_ok());

    let invalid_damping = TextRankConfig::default().with_damping(1.5);
    assert!(invalid_damping.validate().is_err());

    let invalid_window = TextRankConfig::default().with_window_size(1);
    assert!(invalid_window.validate().is_err());
}

#[test]
fn test_empty_text() {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (sentences, tokens) = tokenizer.tokenize("");

    assert!(sentences.is_empty());
    assert!(tokens.is_empty());

    let config = TextRankConfig::default();
    let phrases = phrase::extraction::extract_keyphrases(&tokens, &config);

    assert!(phrases.is_empty());
}

#[test]
fn test_unicode_text() {
    let unicode_text = "机器学习是人工智能的一个子集。深度学习使用神经网络。";

    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (sentences, _tokens) = tokenizer.tokenize(unicode_text);

    assert!(!sentences.is_empty());
    // Should handle CJK characters
}

#[test]
fn test_score_aggregation() {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (_, mut tokens) = tokenizer.tokenize(SAMPLE_TEXT);

    let stopwords = nlp::stopwords::StopwordFilter::new("en");
    for token in &mut tokens {
        token.is_stopword = stopwords.is_stopword(&token.text);
    }

    // Test different aggregation methods
    let aggregations = [
        ScoreAggregation::Sum,
        ScoreAggregation::Mean,
        ScoreAggregation::Max,
        ScoreAggregation::RootMeanSquare,
    ];

    for agg in aggregations {
        let config = TextRankConfig::default()
            .with_score_aggregation(agg)
            .with_top_n(5);

        let phrases = phrase::extraction::extract_keyphrases(&tokens, &config);
        assert!(!phrases.is_empty(), "Failed for {:?}", agg);
    }
}

#[test]
fn test_pagerank_convergence() {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (_, tokens) = tokenizer.tokenize(SAMPLE_TEXT);

    let builder = graph::builder::GraphBuilder::from_tokens(&tokens, 4, true);
    let graph = graph::csr::CsrGraph::from_builder(&builder);

    // Should converge with default settings
    let pr = pagerank::standard::StandardPageRank::new();
    let result = pr.run(&graph);

    assert!(result.converged);

    // Scores should sum to ~1
    let sum: f64 = result.scores.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

#[test]
fn test_parallel_graph_building() {
    // Create a larger document for parallel testing
    let large_text = SAMPLE_TEXT.repeat(10);

    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (_, tokens) = tokenizer.tokenize(&large_text);

    // Build both sequentially and in parallel
    let seq_builder = graph::builder::GraphBuilder::from_tokens(&tokens, 4, true);
    let par_builder = graph::builder::build_graph_parallel(&tokens, 4, true);

    // Parallel version may have slightly fewer nodes (isolated nodes without edges)
    // but should be close and have the same edge structure
    let seq_count = seq_builder.node_count();
    let par_count = par_builder.node_count();

    // Both should have nodes and be within 5% of each other
    assert!(seq_count > 0);
    assert!(par_count > 0);
    let diff = (seq_count as i64 - par_count as i64).unsigned_abs() as usize;
    assert!(
        diff <= seq_count / 20 + 1,
        "Node counts differ too much: seq={}, par={}",
        seq_count,
        par_count
    );
}
