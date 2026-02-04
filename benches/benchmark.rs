//! Benchmarks for rapid_textrank

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rapid_textrank::*;

/// Sample text for benchmarking
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
unsupervised. Deep learning has been applied to various fields including computer
vision, speech recognition, natural language processing, and drug design.

Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language. NLP techniques are used to analyze, understand, and generate human
language in a valuable way. Key applications include sentiment analysis, machine
translation, and text summarization.
"#;

fn benchmark_tokenization(c: &mut Criterion) {
    let tokenizer = nlp::tokenizer::Tokenizer::new();

    c.bench_function("tokenize_sample", |b| {
        b.iter(|| tokenizer.tokenize(black_box(SAMPLE_TEXT)))
    });

    // Benchmark different document sizes
    let mut group = c.benchmark_group("tokenize_by_size");
    for size in [1, 5, 10, 20].iter() {
        let text = SAMPLE_TEXT.repeat(*size);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text, |b, text| {
            b.iter(|| tokenizer.tokenize(black_box(text)))
        });
    }
    group.finish();
}

fn benchmark_graph_building(c: &mut Criterion) {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (_, tokens) = tokenizer.tokenize(SAMPLE_TEXT);

    c.bench_function("graph_build", |b| {
        b.iter(|| graph::builder::GraphBuilder::from_tokens(black_box(&tokens), 4, true))
    });

    // Benchmark parallel vs sequential
    let large_text = SAMPLE_TEXT.repeat(10);
    let (_, large_tokens) = tokenizer.tokenize(&large_text);

    let mut group = c.benchmark_group("graph_build_parallel");
    group.bench_function("sequential", |b| {
        b.iter(|| graph::builder::GraphBuilder::from_tokens(black_box(&large_tokens), 4, true))
    });
    group.bench_function("parallel", |b| {
        b.iter(|| graph::builder::build_graph_parallel(black_box(&large_tokens), 4, true))
    });
    group.finish();
}

fn benchmark_pagerank(c: &mut Criterion) {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (_, tokens) = tokenizer.tokenize(SAMPLE_TEXT);
    let builder = graph::builder::GraphBuilder::from_tokens(&tokens, 4, true);
    let csr_graph = graph::csr::CsrGraph::from_builder(&builder);

    c.bench_function("pagerank", |b| {
        b.iter(|| pagerank::standard::StandardPageRank::new().run(black_box(&csr_graph)))
    });

    // Benchmark with different damping factors
    let mut group = c.benchmark_group("pagerank_damping");
    for damping in [0.5, 0.85, 0.95].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(damping),
            damping,
            |b, &damping| {
                b.iter(|| {
                    pagerank::standard::StandardPageRank::new()
                        .with_damping(damping)
                        .run(black_box(&csr_graph))
                })
            },
        );
    }
    group.finish();
}

fn benchmark_phrase_extraction(c: &mut Criterion) {
    let tokenizer = nlp::tokenizer::Tokenizer::new();
    let (_, mut tokens) = tokenizer.tokenize(SAMPLE_TEXT);

    let stopwords = nlp::stopwords::StopwordFilter::new("en");
    for token in &mut tokens {
        token.is_stopword = stopwords.is_stopword(&token.text);
    }

    let config = TextRankConfig::default().with_top_n(10);

    c.bench_function("extract_keyphrases", |b| {
        b.iter(|| phrase::extraction::extract_keyphrases(black_box(&tokens), black_box(&config)))
    });

    // Compare different variants
    let mut group = c.benchmark_group("extraction_variants");

    group.bench_function("standard", |b| {
        b.iter(|| phrase::extraction::extract_keyphrases(black_box(&tokens), black_box(&config)))
    });

    group.bench_function("position_rank", |b| {
        b.iter(|| {
            variants::position_rank::extract_keyphrases_position(
                black_box(&tokens),
                black_box(&config),
            )
        })
    });

    group.bench_function("biased_textrank", |b| {
        b.iter(|| {
            variants::biased_textrank::extract_keyphrases_biased(
                black_box(&tokens),
                black_box(&config),
                &["machine", "learning"],
                5.0,
            )
        })
    });

    group.finish();
}

fn benchmark_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    for size in [1, 5, 10].iter() {
        let text = SAMPLE_TEXT.repeat(*size);
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &text, |b, text| {
            b.iter(|| {
                let tokenizer = nlp::tokenizer::Tokenizer::new();
                let (_, mut tokens) = tokenizer.tokenize(text);

                let stopwords = nlp::stopwords::StopwordFilter::new("en");
                for token in &mut tokens {
                    token.is_stopword = stopwords.is_stopword(&token.text);
                }

                let config = TextRankConfig::default().with_top_n(10);
                phrase::extraction::extract_keyphrases(&tokens, &config)
            })
        });
    }

    group.finish();
}

fn benchmark_stopwords(c: &mut Criterion) {
    let filter = nlp::stopwords::StopwordFilter::new("en");

    // Words to check
    let words: Vec<&str> = vec![
        "the",
        "and",
        "is",
        "a",
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "deep",
        "neural",
        "network",
        "data",
        "algorithm",
        "computer",
        "science",
    ];

    c.bench_function("stopword_check", |b| {
        b.iter(|| {
            for word in &words {
                black_box(filter.is_stopword(word));
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_tokenization,
    benchmark_graph_building,
    benchmark_pagerank,
    benchmark_phrase_extraction,
    benchmark_full_pipeline,
    benchmark_stopwords,
);

criterion_main!(benches);
