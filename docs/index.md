# rapid_textrank

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

**High-performance TextRank keyword extraction in Rust with Python bindings.**

rapid_textrank extracts keywords and key phrases from text up to 10--100x faster than pure-Python implementations, with support for seven algorithm variants and 18 languages.

---

## Key Features

- **Fast** -- Rust core with CSR graph format, string interning, and FxHash for 10--100x speedups over pure Python (depending on document size and tokenization).
- **Eight algorithm variants** -- BaseTextRank, PositionRank, BiasedTextRank, TopicRank, SingleRank, TopicalPageRank, MultipartiteRank, and SentenceRank.
- **Unicode-aware** -- proper handling of CJK and other scripts.
- **18 languages** -- built-in stopword lists from English to Japanese.
- **Dual API** -- native Python classes for quick use, plus a JSON interface for pre-tokenized / spaCy input.
- **Configurable** -- control damping, window size, POS filtering, phrase length, score aggregation, and more via `TextRankConfig`.

---

## Install

```bash
pip install rapid_textrank
```

## Quick Example

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

```
machine learning: 0.2341
deep learning: 0.1872
artificial intelligence: 0.1654
neural networks: 0.1432
systems: 0.0891
```

---

## Documentation

| Section | What you will find |
|---------|--------------------|
| [Getting Started](getting-started/index.md) | Installation, quick start guide, and grab-and-go recipes |
| [Algorithms](algorithms/index.md) | How TextRank works and details on each variant |
| [API Reference](api/index.md) | `extract_keywords()`, extractor classes, `TextRankConfig`, JSON interface, spaCy integration |
| [Performance](performance/index.md) | Benchmarks, why Rust is fast, comparison with alternatives |
| [Architecture](architecture/pipeline.md) | Pipeline stages, variant composition, debug introspection, production hardening |

---

## Links

- **GitHub**: <https://github.com/xang1234/rapid-textrank>
- **PyPI**: <https://pypi.org/project/rapid-textrank/>
- **License**: [MIT](reference/license.md)
