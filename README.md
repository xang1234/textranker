# rapid_textrank

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

**High-performance TextRank implementation in Rust with Python bindings.**

Extract keywords and key phrases from text up to 10-100x faster than pure Python implementations (depending on document size and tokenization), with support for multiple algorithm variants and 18 languages.

## Features

- **Fast**: Up to 10-100x faster than pure Python implementations (see benchmarks)
- **Multiple algorithms**: TextRank, PositionRank, BiasedTextRank, TopicRank, SingleRank, and TopicalPageRank variants
- **Unicode-aware**: Proper handling of CJK and other scripts (emoji are ignored by the built-in tokenizer)
- **Multi-language**: Stopword support for 18 languages
- **Dual API**: Native Python classes + JSON interface for batch processing
- **Rust core**: Computation happens in Rust (the Python GIL is currently held during extraction)

## Quick Start

```bash
pip install rapid_textrank
```

```python
from rapid_textrank import extract_keywords

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

3. **Extract phrases**: High-scoring words are grouped into noun chunks (POS-filtered) to form key phrases. Scores are aggregated (sum, mean, or max).

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
- [TopicRank: Graph-Based Topic Ranking for Keyphrase Extraction](https://aclanthology.org/I13-1062/) (Bougouin et al., 2013)
- [SingleRank: Single Document Keyphrase Extraction Using Neighborhood Knowledge](https://ojs.aaai.org/index.php/AAAI/article/view/7798) (Wan & Xiao, 2008)
- [Topical Word Importance for Fast Keyphrase Extraction](https://aclanthology.org/W15-3605/) (Sterckx et al., 2015)

## Algorithm Variants

| Variant | Best For | Description |
|---------|----------|-------------|
| `BaseTextRank` | General text | Standard TextRank implementation |
| `PositionRank` | Academic papers, news | Favors words appearing early in the document |
| `BiasedTextRank` | Topic-focused extraction | Biases results toward specified focus terms |
| `TopicRank` | Multi-topic documents | Clusters similar phrases into topics and ranks the topics |
| `SingleRank` | Longer documents | Uses weighted co-occurrence edges and cross-sentence windowing |
| `TopicalPageRank` | Topic-model-guided extraction | Biases SingleRank towards topically important words via personalized PageRank |

### PositionRank

Weights words by their position—earlier appearances score higher. Useful for documents where key information appears in titles, abstracts, or opening paragraphs.

```python
from rapid_textrank import PositionRank

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
from rapid_textrank import BiasedTextRank

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

### TopicRank

TopicRank clusters similar candidate phrases into topics, then ranks the topics. It is exposed via the JSON interface (useful for spaCy-tokenized input).

```python
import json
import spacy
from rapid_textrank import extract_from_json

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

tokens = []
for sent_idx, sent in enumerate(doc.sents):
    for token in sent:
        tokens.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "start": token.idx,
            "end": token.idx + len(token.text),
            "sentence_idx": sent_idx,
            "token_idx": token.i,
            "is_stopword": token.is_stop,
        })

payload = {
    "tokens": tokens,
    "variant": "topic_rank",
    "config": {
        "top_n": 10,
        "language": "en",
        "topic_similarity_threshold": 0.25,
        "topic_edge_weight": 1.0,
    },
}

result = json.loads(extract_from_json(json.dumps(payload)))
for phrase in result["phrases"][:10]:
    print(phrase["text"], phrase["score"])
```

### SingleRank

SingleRank (Wan & Xiao, 2008) extends TextRank in two ways: edges are weighted by co-occurrence count (repeated neighbors get stronger connections), and the sliding window ignores sentence boundaries so that terms at the end of one sentence connect to terms at the start of the next.

```python
from rapid_textrank import SingleRank

extractor = SingleRank(top_n=10)
result = extractor.extract_keywords("""
Machine learning is a powerful tool. Deep learning is a subset of
machine learning. Neural networks power deep learning systems.
""")

# Cross-sentence co-occurrences strengthen "machine learning" edges
for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

SingleRank is also available via the JSON interface with `variant="single_rank"`.

**When to use SingleRank over BaseTextRank:** SingleRank works well on longer documents where important terms co-occur across sentence boundaries. The weighted edges amplify frequently co-occurring pairs, giving a clearer signal than the binary edges used by BaseTextRank.

### Topical PageRank

Topical PageRank (Sterckx et al., 2015) extends SingleRank by biasing the random walk towards topically important words. Instead of uniform teleportation, PageRank uses a personalization vector derived from per-word topic weights.

Users supply pre-computed topic weights as a `{lemma: weight}` dictionary. These typically come from a topic model (e.g., LDA via gensim or sklearn), but any source of word importance scores works. Words absent from the dictionary receive a configurable minimum weight (`min_weight`, default 0.0 — matching PKE's OOV behavior).

```python
from rapid_textrank import TopicalPageRank

# Topic weights from an external topic model or manual assignment
topic_weights = {
    "neural": 0.9,
    "network": 0.8,
    "learning": 0.7,
    "deep": 0.6,
}

extractor = TopicalPageRank(
    topic_weights=topic_weights,
    min_weight=0.01,  # Floor for out-of-vocabulary words
    top_n=10
)

result = extractor.extract_keywords("""
Deep learning is a subset of machine learning that uses artificial neural
networks. Neural networks with many layers can learn complex patterns.
Convolutional neural networks excel at image recognition tasks.
""")

for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")

# Update topic weights for a different document/topic
result = extractor.extract_keywords(
    "Machine learning enables data-driven decisions...",
    topic_weights={"machine": 0.9, "data": 0.8}
)
```

TopicalPageRank is also available via the JSON interface with `variant="topical_pagerank"` (aliases: `"tpr"`, `"single_tpr"`). Set `topic_weights` and optionally `topic_min_weight` in the JSON config:

```python
import json
from rapid_textrank import extract_from_json

payload = {
    "tokens": tokens,  # Pre-tokenized (e.g., from spaCy)
    "variant": "topical_pagerank",
    "config": {
        "top_n": 10,
        "topic_weights": {"neural": 0.9, "network": 0.8, "learning": 0.7},
        "topic_min_weight": 0.01,
    },
}

result = json.loads(extract_from_json(json.dumps(payload)))
```

#### Computing topic weights from LDA

The `topic_weights_from_lda` helper computes per-lemma weights from a trained [gensim](https://radimrehurek.com/gensim/) LDA model, so you can go from corpus to keywords in a few lines:

```bash
pip install rapid_textrank[topic]   # installs gensim
```

```python
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from rapid_textrank import TopicalPageRank, topic_weights_from_lda

# 1. Train (or load) an LDA model
texts = [doc.split() for doc in corpus]      # list of token lists
dictionary = Dictionary(texts)
bow_corpus = [dictionary.doc2bow(t) for t in texts]
lda = LdaModel(bow_corpus, num_topics=10, id2word=dictionary)

# 2. Compute topic weights for a single document
weights = topic_weights_from_lda(lda, bow_corpus[0], dictionary)

# 3. Extract keywords using those weights
extractor = TopicalPageRank(topic_weights=weights, top_n=10)
result = extractor.extract_keywords(raw_text)
for phrase in result.phrases:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

`topic_weights_from_lda` accepts an optional `aggregation` parameter (`"max"` or `"mean"`) and `top_n_words` to control how many words per topic are considered. See the docstring for details.

**TopicalPageRank vs BiasedTextRank:** Both bias extraction towards specific terms, but they differ in how:
- **BiasedTextRank** takes a list of focus terms and a single bias weight. It's manual and direct — good when you know exactly which terms matter.
- **TopicalPageRank** takes per-word weights, typically from a topic model. It's data-driven — good when you want the topic distribution to guide extraction automatically.

**Topic modeling is optional.** You can supply any word-importance scores: TF-IDF weights, embedding similarities, domain relevance scores, or hand-picked values.

## API Reference

### Convenience Function

The simplest way to extract keywords:

```python
from rapid_textrank import extract_keywords

phrases = extract_keywords(
    text,           # Input text
    top_n=10,       # Number of keywords to return
    language="en"   # Language for stopword filtering
)
```

### Class-Based API

For more control, use the extractor classes:

```python
from rapid_textrank import BaseTextRank, PositionRank, BiasedTextRank, SingleRank, TopicalPageRank

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

# SingleRank: weighted edges + cross-sentence windowing
extractor = SingleRank(top_n=10, language="en")
result = extractor.extract_keywords(text)

# Topical PageRank: topic-weight-biased extraction
extractor = TopicalPageRank(
    topic_weights={"neural": 0.9, "network": 0.8},
    min_weight=0.01,
    top_n=10,
    language="en"
)
result = extractor.extract_keywords(text)
```

TopicRank is available via the JSON interface using `variant="topic_rank"` (see below).

### Configuration

Fine-tune the algorithm with `TextRankConfig`:

```python
from rapid_textrank import TextRankConfig, BaseTextRank

config = TextRankConfig(
    damping=0.85,              # PageRank damping factor (0-1)
    max_iterations=100,        # Maximum PageRank iterations
    convergence_threshold=1e-6,# Convergence threshold
    window_size=3,             # Co-occurrence window size
    top_n=10,                  # Number of results
    min_phrase_length=1,       # Minimum words in a phrase
    max_phrase_length=4,       # Maximum words in a phrase
    score_aggregation="sum",   # How to combine word scores: "sum", "mean", "max", "rms"
    language="en",             # Language for stopwords
    include_pos=["NOUN","ADJ","PROPN","VERB"],  # POS tags to include in the graph
    use_pos_in_nodes=True,     # If True, graph nodes are lemma+POS
    phrase_grouping="scrubbed_text",   # "lemma" or "scrubbed_text"
    stopwords=["custom", "terms"]  # Additional stopwords (extends built-in list)
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

For processing large documents or integrating with spaCy, use the JSON interface. This accepts pre-tokenized data to avoid re-tokenizing in Rust. Stopword handling can use each token's `is_stopword` field and/or a `config.language` plus `config.stopwords` (additional words that extend the built-in list). Language codes follow the Supported Languages table below.

```python
from rapid_textrank import extract_from_json, extract_batch_from_json
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
    "variant": "textrank",
    "config": {
        "top_n": 10,
        "language": "en",
        "stopwords": ["nlp", "transformers"]
    }
}

result_json = extract_from_json(json.dumps(doc))
result = json.loads(result_json)

# Batch processing (Rust core; per-document processing is sequential)
docs = [doc1, doc2, doc3]
results_json = extract_batch_from_json(json.dumps(docs))
results = json.loads(results_json)
```

`variant` can be `"textrank"` (default), `"position_rank"`, `"biased_textrank"`, `"topic_rank"`, `"single_rank"`, or `"topical_pagerank"` (aliases: `"tpr"`, `"single_tpr"`). For `"biased_textrank"`, set `focus_terms` and `bias_weight` in the JSON config. For `"topic_rank"`, set `topic_similarity_threshold` and `topic_edge_weight` in the JSON config. For `"topical_pagerank"`, set `topic_weights` and optionally `topic_min_weight` in the JSON config.

## Supported Languages

Stopword filtering is available for 18 languages. Use these codes for the `language` parameter in all APIs (including JSON config):

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `de` | German | `fr` | French |
| `es` | Spanish | `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `ru` | Russian | `sv` | Swedish |
| `no` | Norwegian | `da` | Danish | `fi` | Finnish |
| `hu` | Hungarian | `tr` | Turkish | `pl` | Polish |
| `ar` | Arabic | `zh` | Chinese | `ja` | Japanese |

You can inspect the built-in stopword list with:

```python
import rapid_textrank as rt
rt.get_stopwords("en")
```

## Performance

rapid_textrank achieves significant speedups through Rust's performance characteristics and careful algorithm implementation.

### Benchmark Script

Run this script to compare performance on your hardware:

```python
"""
Benchmark: rapid_textrank vs pytextrank

Prerequisites:
    pip install rapid_textrank pytextrank spacy
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


def benchmark_rapid_textrank(text: str, runs: int = 10) -> dict:
    """Benchmark rapid_textrank."""
    from rapid_textrank import BaseTextRank

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

        # Benchmark rapid_textrank
        rust_results = benchmark_rapid_textrank(text)
        print(f"rapid_textrank:  {rust_results['mean']:>8.2f} ms (±{rust_results['std']:.2f})")

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
    print("For fair comparison with pre-tokenized input, use rapid_textrank's JSON API.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

### Why Rust is Fast

The performance advantage comes from several factors:

1. **CSR Graph Format**: The co-occurrence graph uses Compressed Sparse Row format, enabling cache-friendly memory access during PageRank iteration.

2. **String Interning**: Repeated words share a single allocation via `StringPool`, reducing memory usage 10-100x for typical documents.

3. **Parallel Processing**: Rayon provides data parallelism in internal graph construction without explicit thread management.

4. **Link-Time Optimization (LTO)**: Release builds use full LTO with single codegen unit for maximum inlining.

5. **Rust core**: Most computation happens in Rust, minimizing Python-level overhead.

6. **FxHash**: Fast non-cryptographic hashing for internal hash maps.

## Installation

### From PyPI

```bash
pip install rapid_textrank
```

Import name is `rapid_textrank`.

### With spaCy Support

```bash
pip install rapid_textrank[spacy]
```

```python
import spacy
import rapid_textrank.spacy_component  # registers the pipeline factory

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("rapid_textrank")

doc = nlp("Machine learning is a subset of artificial intelligence.")
for phrase in doc._.phrases[:5]:
    print(f"{phrase.text}: {phrase.score:.4f}")
```

### From Source

Requirements: Rust 1.70+, Python 3.9+

```bash
git clone https://github.com/xang1234/rapid-textrank
cd rapid_textrank
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

## Publishing

Publishing is automated with GitHub Actions using Trusted Publishing (OIDC), so no API tokens are stored.

TestPyPI release (push a tag):

```bash
git tag -a test-0.1.0 -m "TestPyPI 0.1.0"
git push origin test-0.1.0
```

Tag pattern: `test-*`

PyPI release (push a tag):

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

Tag pattern: `v*`

Wheel builds

GitHub Actions builds wheels for Python 3.9–3.12 on Linux, macOS, and Windows.

Before the first publish, add Trusted Publishers on TestPyPI and PyPI:

- Repo: `xang1234/textranker`
- Workflows: `.github/workflows/publish-testpypi.yml` and `.github/workflows/publish-pypi.yml`
- Environments: `testpypi` and `pypi`

You can also trigger either workflow manually via GitHub Actions if needed.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use rapid_textrank in research, please cite the original TextRank paper:

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

For SingleRank:

```bibtex
@inproceedings{wan-xiao-2008-singlerank,
    title = "Single Document Keyphrase Extraction Using Neighborhood Knowledge",
    author = "Wan, Xiaojun and Xiao, Jianguo",
    booktitle = "Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence (AAAI 2008)",
    year = "2008",
    pages = "855--860",
}
```

For TopicRank:

```bibtex
@inproceedings{bougouin-boudin-daille-2013-topicrank,
    title = "{T}opic{R}ank: Graph-Based Topic Ranking for Keyphrase Extraction",
    author = "Bougouin, Adrien and Boudin, Florian and Daille, B{\\'e}atrice",
    booktitle = "Proceedings of the Sixth International Joint Conference on Natural Language Processing",
    year = "2013",
    pages = "543--551",
    publisher = "Asian Federation of Natural Language Processing",
}
```

For Topical PageRank:

```bibtex
@inproceedings{sterckx-etal-2015-topical,
    title = "Topical Word Importance for Fast Keyphrase Extraction",
    author = "Sterckx, Lucas and Demeester, Thomas and Deleu, Johannes and Develder, Chris",
    booktitle = "Proceedings of the 24th International Conference on World Wide Web (Companion Volume)",
    year = "2015",
    pages = "121--122",
}
```
