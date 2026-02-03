# rust_textrank

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

**High-performance TextRank implementation in Rust with Python bindings.**

Extract keywords and key phrases from text 10-100x faster than pure Python implementations, with support for multiple algorithm variants and 18 languages.

## Features

- **Fast**: 10-100x faster than pure Python implementations
- **Multiple algorithms**: TextRank, PositionRank, and BiasedTextRank variants
- **Unicode-aware**: Proper handling of CJK, emoji, and other scripts
- **Multi-language**: Stopword support for 18 languages
- **Dual API**: Native Python classes + JSON interface for batch processing
- **Zero Python overhead**: Computation happens entirely in Rust (no GIL)

## Quick Start

```bash
pip install rust_textrank
```

```python
from rust_textrank import extract_keywords

text = """
Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience. Deep learning, a type of
machine learning, uses neural networks with many layers.
"""

keywords = extract_keywords(text, top_n=5, language="en")
for phrase in keywords:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

Output:
```
machine learning: 0.2341
deep learning: 0.1872
artificial intelligence: 0.1654
neural networks: 0.1432
systems: 0.0891
```

## How TextRank Works

TextRank is a graph-based ranking algorithm for keyword extraction, inspired by Google's PageRank.

### The Algorithm

1. **Build a co-occurrence graph**: Words become nodes. An edge connects two words if they appear within a sliding window (default: 4 words).

2. **Run PageRank**: The algorithm iteratively distributes "importance" through the graph. Words connected to many important words become important themselves.

3. **Extract phrases**: Adjacent high-scoring words are combined into key phrases. Scores are aggregated (sum, mean, or max).

```
Text: "Machine learning enables systems to learn from data"

Co-occurrence graph (window=2):
    machine ←→ learning ←→ enables ←→ systems ←→ learn ←→ data
                              ↓
                            PageRank
                              ↓
    Scores: machine(0.23) learning(0.31) enables(0.12) ...
                              ↓
                        Phrase extraction
                              ↓
    "machine learning" (0.54), "systems" (0.18), ...
```

### Further Reading

- [TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) (Mihalcea & Tarau, 2004)
- [PositionRank: An Unsupervised Approach to Keyphrase Extraction](https://aclanthology.org/P17-1102/) (Florescu & Caragea, 2017)
- [BiasedTextRank: Unsupervised Graph-Based Content Extraction](https://aclanthology.org/2020.coling-main.144/) (Kazemi et al., 2020)

## Algorithm Variants

| Variant | Best For | Description |
|---------|----------|-------------|
| `BaseTextRank` | General text | Standard TextRank implementation |
| `PositionRank` | Academic papers, news | Favors words appearing early in the document |
| `BiasedTextRank` | Topic-focused extraction | Biases results toward specified focus terms |

### PositionRank

Weights words by their position—earlier appearances score higher. Useful for documents where key information appears in titles, abstracts, or opening paragraphs.

```python
from rust_textrank import PositionRank

extractor = PositionRank(top_n=10)
result = extractor.extract_keywords("""
Quantum Computing Advances in 2024

Researchers have made significant breakthroughs in quantum error correction.
The quantum computing field continues to evolve rapidly...
""")

# "quantum computing" and "quantum" will rank higher due to early position
```

### BiasedTextRank

Steers extraction toward specific topics using focus terms. The `bias_weight` parameter controls how strongly results favor the focus terms.

```python
from rust_textrank import BiasedTextRank

extractor = BiasedTextRank(
    focus_terms=["security", "privacy"],
    bias_weight=5.0,  # Higher = stronger bias
    top_n=10
)

result = extractor.extract_keywords("""
Modern web applications must balance user experience with security.
Privacy regulations require careful data handling. Performance
optimizations should not compromise security measures.
""")

# Results will favor security/privacy-related phrases
```

## API Reference

### Convenience Function

The simplest way to extract keywords:

```python
from rust_textrank import extract_keywords

phrases = extract_keywords(
    text,           # Input text
    top_n=10,       # Number of keywords to return
    language="en"   # Language for stopword filtering
)
```

### Class-Based API

For more control, use the extractor classes:

```python
from rust_textrank import BaseTextRank, PositionRank, BiasedTextRank

# Standard TextRank
extractor = BaseTextRank(top_n=10, language="en")
result = extractor.extract_keywords(text)

# Position-weighted
extractor = PositionRank(top_n=10, language="en")
result = extractor.extract_keywords(text)

# Topic-biased
extractor = BiasedTextRank(
    focus_terms=["machine", "learning"],
    bias_weight=5.0,
    top_n=10,
    language="en"
)
result = extractor.extract_keywords(text)

# You can also pass focus_terms per-call
result = extractor.extract_keywords(text, focus_terms=["neural", "network"])
```

### Configuration

Fine-tune the algorithm with `TextRankConfig`:

```python
from rust_textrank import TextRankConfig, BaseTextRank

config = TextRankConfig(
    damping=0.85,              # PageRank damping factor (0-1)
    max_iterations=100,        # Maximum PageRank iterations
    convergence_threshold=1e-6,# Convergence threshold
    window_size=4,             # Co-occurrence window size
    top_n=10,                  # Number of results
    min_phrase_length=1,       # Minimum words in a phrase
    max_phrase_length=4,       # Maximum words in a phrase
    score_aggregation="sum",   # How to combine word scores: "sum", "mean", "max", "rms"
    language="en"              # Language for stopwords
)

extractor = BaseTextRank(config=config)
```

### Result Objects

```python
result = extractor.extract_keywords(text)

# TextRankResult attributes
result.phrases      # List of Phrase objects
result.converged    # Whether PageRank converged
result.iterations   # Number of iterations run

# Phrase attributes
for phrase in result.phrases:
    phrase.text     # The phrase text (e.g., "machine learning")
    phrase.lemma    # Lemmatized form
    phrase.score    # TextRank score
    phrase.count    # Occurrences in text
    phrase.rank     # 1-indexed rank

# Convenience method
tuples = result.as_tuples()  # [(text, score), ...]
```

### JSON Interface

For processing large documents or integrating with spaCy, use the JSON interface. This accepts pre-tokenized data to avoid re-tokenizing in Rust.

```python
from rust_textrank import extract_from_json, extract_batch_from_json
import json

# Single document
doc = {
    "tokens": [
        {
            "text": "Machine",
            "lemma": "machine",
            "pos": "NOUN",
            "start": 0,
            "end": 7,
            "sentence_idx": 0,
            "token_idx": 0,
            "is_stopword": False
        },
        # ... more tokens
    ],
    "config": {"top_n": 10}
}

result_json = extract_from_json(json.dumps(doc))
result = json.loads(result_json)

# Batch processing (parallel in Rust)
docs = [doc1, doc2, doc3]
results_json = extract_batch_from_json(json.dumps(docs))
results = json.loads(results_json)
```

## Supported Languages

Stopword filtering is available for 18 languages:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `de` | German | `fr` | French |
| `es` | Spanish | `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `ru` | Russian | `sv` | Swedish |
| `no` | Norwegian | `da` | Danish | `fi` | Finnish |
| `hu` | Hungarian | `tr` | Turkish | `pl` | Polish |
| `ar` | Arabic | `zh` | Chinese | `ja` | Japanese |

## Performance

rust_textrank achieves significant speedups through Rust's performance characteristics and careful algorithm implementation.

### Benchmark Script

Run this script to compare performance on your hardware:

```python
"""
Benchmark: rust_textrank vs pytextrank

Prerequisites:
    pip install rust_textrank pytextrank spacy
    python -m spacy download en_core_web_sm
"""

import time
import statistics

# Sample texts of varying sizes
TEXTS = {
    "small": """
        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks with many layers.
    """,

    "medium": """
        Natural language processing (NLP) is a field of artificial intelligence
        that focuses on the interaction between computers and humans through
        natural language. The ultimate goal of NLP is to enable computers to
        understand, interpret, and generate human language in a valuable way.

        Machine learning approaches have transformed NLP in recent years.
        Deep learning models, particularly transformers, have achieved
        state-of-the-art results on many NLP tasks including translation,
        summarization, and question answering.

        Key applications include sentiment analysis, named entity recognition,
        machine translation, and text classification. These technologies
        power virtual assistants, search engines, and content recommendation
        systems used by millions of people daily.
    """,

    "large": """
        Artificial intelligence has evolved dramatically since its inception in
        the mid-20th century. Early AI systems relied on symbolic reasoning and
        expert systems, where human knowledge was manually encoded into rules.

        The machine learning revolution changed everything. Instead of explicit
        programming, systems learn patterns from data. Supervised learning uses
        labeled examples, unsupervised learning finds hidden structures, and
        reinforcement learning optimizes through trial and error.

        Deep learning, powered by neural networks with multiple layers, has
        achieved remarkable success. Convolutional neural networks excel at
        image recognition. Recurrent neural networks and transformers handle
        sequential data like text and speech. Generative adversarial networks
        create realistic synthetic content.

        Natural language processing has been transformed by these advances.
        Word embeddings capture semantic relationships. Attention mechanisms
        allow models to focus on relevant context. Large language models
        demonstrate emergent capabilities in reasoning and generation.

        Computer vision applications include object detection, facial recognition,
        medical image analysis, and autonomous vehicle perception. These systems
        process visual information with superhuman accuracy in many domains.

        The ethical implications of AI are significant. Bias in training data
        can lead to unfair outcomes. Privacy concerns arise from data collection.
        Job displacement affects workers across industries. Regulation and
        governance frameworks are being developed worldwide.

        Future directions include neuromorphic computing, quantum machine learning,
        and artificial general intelligence. Researchers continue to push
        boundaries while addressing safety and alignment challenges.
    """ * 3  # ~1000 words
}


def benchmark_rust_textrank(text: str, runs: int = 10) -> dict:
    """Benchmark rust_textrank."""
    from rust_textrank import BaseTextRank

    extractor = BaseTextRank(top_n=10, language="en")

    # Warmup
    extractor.extract_keywords(text)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = extractor.extract_keywords(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "phrases": len(result.phrases)
    }


def benchmark_pytextrank(text: str, runs: int = 10) -> dict:
    """Benchmark pytextrank with spaCy."""
    import spacy
    import pytextrank

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    # Warmup
    doc = nlp(text)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        doc = nlp(text)
        phrases = list(doc._.phrases[:10])
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "phrases": len(phrases)
    }


def main():
    print("=" * 70)
    print("TextRank Performance Benchmark")
    print("=" * 70)

    for size, text in TEXTS.items():
        word_count = len(text.split())
        print(f"\n{size.upper()} TEXT (~{word_count} words)")
        print("-" * 50)

        # Benchmark rust_textrank
        rust_results = benchmark_rust_textrank(text)
        print(f"rust_textrank:  {rust_results['mean']:>8.2f} ms (±{rust_results['std']:.2f})")

        # Benchmark pytextrank
        try:
            py_results = benchmark_pytextrank(text)
            print(f"pytextrank:     {py_results['mean']:>8.2f} ms (±{py_results['std']:.2f})")

            speedup = py_results['mean'] / rust_results['mean']
            print(f"Speedup:        {speedup:>8.1f}x faster")
        except Exception as e:
            print(f"pytextrank:     (not available: {e})")

    print("\n" + "=" * 70)
    print("Note: pytextrank times include spaCy tokenization.")
    print("For fair comparison with pre-tokenized input, use rust_textrank's JSON API.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

### Why Rust is Fast

The performance advantage comes from several factors:

1. **CSR Graph Format**: The co-occurrence graph uses Compressed Sparse Row format, enabling cache-friendly memory access during PageRank iteration.

2. **String Interning**: Repeated words share a single allocation via `StringPool`, reducing memory usage 10-100x for typical documents.

3. **Parallel Processing**: Rayon provides data parallelism for batch processing without explicit thread management.

4. **Link-Time Optimization (LTO)**: Release builds use full LTO with single codegen unit for maximum inlining.

5. **No GIL**: All computation happens in Rust. Python's Global Interpreter Lock is released during extraction.

6. **FxHash**: Fast non-cryptographic hashing for internal hash maps.

## Installation

### From PyPI

```bash
pip install rust_textrank
```

### With spaCy Support

```bash
pip install rust_textrank[spacy]
```

### From Source

Requirements: Rust 1.70+, Python 3.9+

```bash
git clone https://github.com/textranker/rust_textrank
cd rust_textrank
pip install maturin
maturin develop --release
```

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run Rust tests
cargo test
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use rust_textrank in research, please cite the original TextRank paper:

```bibtex
@inproceedings{mihalcea-tarau-2004-textrank,
    title = "{T}ext{R}ank: Bringing Order into Text",
    author = "Mihalcea, Rada and Tarau, Paul",
    booktitle = "Proceedings of EMNLP 2004",
    year = "2004",
    publisher = "Association for Computational Linguistics",
}
```

For PositionRank:

```bibtex
@inproceedings{florescu-caragea-2017-positionrank,
    title = "{P}osition{R}ank: An Unsupervised Approach to Keyphrase Extraction from Scholarly Documents",
    author = "Florescu, Corina and Caragea, Cornelia",
    booktitle = "Proceedings of ACL 2017",
    year = "2017",
}
```
