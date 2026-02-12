"""
Benchmark: GIL release effectiveness for concurrent extraction.

Compares sequential vs ThreadPoolExecutor throughput to measure the
real-world benefit of py.allow_threads() in the Rust extraction code.

If the GIL were still held, threads would serialize and the concurrent
path would show no speedup (or even a slowdown from overhead). A
speedup > 1x proves the GIL is truly released during extraction.

Usage:
    python python/benchmarks/bench_gil_release.py [--threads 4] [--docs 32] [--warmup 2] [--rounds 5]
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from rapid_textrank import BaseTextRank


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------

# Realistic multi-sentence paragraphs (varied vocabulary so extraction
# is non-trivial).
_PARAGRAPHS = [
    (
        "Machine learning algorithms process large datasets to identify patterns. "
        "Supervised learning uses labeled training data to make predictions. "
        "Neural networks form the backbone of modern deep learning systems. "
        "Gradient descent optimizes model parameters during training. "
        "Feature engineering transforms raw data into useful representations."
    ),
    (
        "Natural language processing enables computers to understand human text. "
        "Tokenization splits sentences into individual words or subwords. "
        "Named entity recognition extracts people, places, and organizations. "
        "Sentiment analysis determines the emotional tone of documents. "
        "Language models predict the probability of word sequences."
    ),
    (
        "Computer vision systems interpret visual information from images. "
        "Convolutional neural networks excel at image classification tasks. "
        "Object detection identifies and locates items within photographs. "
        "Image segmentation assigns labels to individual pixels. "
        "Transfer learning adapts pretrained models to new visual domains."
    ),
    (
        "Distributed systems coordinate multiple computing nodes for reliability. "
        "Consensus algorithms ensure data consistency across replicas. "
        "Load balancing distributes incoming requests to available servers. "
        "Microservice architectures decompose applications into small services. "
        "Container orchestration automates deployment and scaling of workloads."
    ),
    (
        "Database optimization improves query performance through indexing. "
        "Relational databases use structured query language for data access. "
        "NoSQL databases handle unstructured and semi-structured data flexibly. "
        "Transaction processing guarantees atomicity and consistency of updates. "
        "Replication strategies protect against data loss during failures."
    ),
    (
        "Cybersecurity protects networks and systems from digital threats. "
        "Encryption algorithms transform plaintext into unreadable ciphertext. "
        "Authentication mechanisms verify the identity of users and services. "
        "Intrusion detection systems monitor network traffic for anomalies. "
        "Vulnerability scanning identifies weaknesses in software configurations."
    ),
    (
        "Cloud computing provides scalable resources over the internet. "
        "Serverless functions execute code without managing infrastructure. "
        "Object storage services handle massive volumes of unstructured data. "
        "Virtual machines emulate complete hardware environments in software. "
        "Edge computing processes data closer to the source for lower latency."
    ),
    (
        "Reinforcement learning agents learn by interacting with environments. "
        "Reward signals guide policy optimization toward desired behaviors. "
        "Exploration strategies balance novelty with exploiting known rewards. "
        "Multi-agent systems coordinate independent learners in shared spaces. "
        "Model-based planning uses learned dynamics for efficient decision making."
    ),
]


def generate_documents(n: int) -> list[str]:
    """Generate n documents by cycling through paragraphs."""
    return [_PARAGRAPHS[i % len(_PARAGRAPHS)] for i in range(n)]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_sequential(extractor: BaseTextRank, documents: list[str]) -> list[float]:
    """Extract keywords from each document sequentially. Returns per-doc times."""
    per_doc: list[float] = []
    for doc in documents:
        t0 = time.perf_counter()
        extractor.extract_keywords(doc)
        per_doc.append(time.perf_counter() - t0)
    return per_doc


def run_concurrent(
    extractor: BaseTextRank, documents: list[str], threads: int
) -> list[float]:
    """Extract keywords concurrently via ThreadPoolExecutor. Returns per-doc times."""
    per_doc: list[float] = [0.0] * len(documents)

    def _work(idx_doc: tuple[int, str]) -> tuple[int, float]:
        idx, doc = idx_doc
        t0 = time.perf_counter()
        extractor.extract_keywords(doc)
        return idx, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for idx, elapsed in pool.map(_work, enumerate(documents)):
            per_doc[idx] = elapsed

    return per_doc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def report(
    label: str,
    wall_times: list[float],
    per_doc_times: list[list[float]],
) -> dict:
    """Print a summary block and return stats dict."""
    walls = wall_times
    all_per_doc = [t for round_times in per_doc_times for t in round_times]

    stats = {
        "wall_mean": statistics.mean(walls),
        "wall_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "per_doc_mean": statistics.mean(all_per_doc),
        "per_doc_p50": statistics.median(all_per_doc),
        "per_doc_p99": sorted(all_per_doc)[int(len(all_per_doc) * 0.99)],
    }

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  Wall clock  : {fmt_ms(stats['wall_mean'])} ± {fmt_ms(stats['wall_stdev'])}")
    print(f"  Per-doc mean: {fmt_ms(stats['per_doc_mean'])}")
    print(f"  Per-doc p50 : {fmt_ms(stats['per_doc_p50'])}")
    print(f"  Per-doc p99 : {fmt_ms(stats['per_doc_p99'])}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GIL release effectiveness")
    parser.add_argument("--threads", type=int, default=4, help="Thread pool size (default: 4)")
    parser.add_argument("--docs", type=int, default=32, help="Number of documents (default: 32)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup rounds (default: 2)")
    parser.add_argument("--rounds", type=int, default=5, help="Measurement rounds (default: 5)")
    parser.add_argument("--top-n", type=int, default=10, help="Keywords per document (default: 10)")
    args = parser.parse_args()

    documents = generate_documents(args.docs)
    extractor = BaseTextRank(top_n=args.top_n)

    print(f"GIL Release Benchmark")
    print(f"  Documents : {args.docs}")
    print(f"  Threads   : {args.threads}")
    print(f"  Rounds    : {args.rounds} (+ {args.warmup} warmup)")
    print(f"  Doc length: ~{len(documents[0])} chars")

    # ── Warmup ────────────────────────────────────────────────────────
    for _ in range(args.warmup):
        run_sequential(extractor, documents)
        run_concurrent(extractor, documents, args.threads)

    # ── Sequential measurement ────────────────────────────────────────
    seq_walls: list[float] = []
    seq_per_docs: list[list[float]] = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        per_doc = run_sequential(extractor, documents)
        seq_walls.append(time.perf_counter() - t0)
        seq_per_docs.append(per_doc)

    # ── Concurrent measurement ────────────────────────────────────────
    con_walls: list[float] = []
    con_per_docs: list[list[float]] = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        per_doc = run_concurrent(extractor, documents, args.threads)
        con_walls.append(time.perf_counter() - t0)
        con_per_docs.append(per_doc)

    # ── Report ────────────────────────────────────────────────────────
    seq_stats = report("Sequential", seq_walls, seq_per_docs)
    con_stats = report(f"Concurrent ({args.threads} threads)", con_walls, con_per_docs)

    speedup = seq_stats["wall_mean"] / con_stats["wall_mean"] if con_stats["wall_mean"] > 0 else 0
    efficiency = speedup / args.threads * 100

    print(f"\n{'━' * 50}")
    print(f"  Speedup    : {speedup:.2f}x")
    print(f"  Efficiency : {efficiency:.1f}% (of {args.threads} threads)")
    print(f"{'━' * 50}")

    if speedup > 1.2:
        print(f"\n  ✓ GIL release confirmed: {speedup:.2f}x speedup with {args.threads} threads")
    elif speedup > 1.0:
        print(f"\n  ~ Marginal speedup ({speedup:.2f}x). Document may be too small for parallelism benefit.")
    else:
        print(f"\n  ✗ No speedup detected. GIL may not be released, or overhead dominates.")


if __name__ == "__main__":
    main()
