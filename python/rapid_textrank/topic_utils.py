"""
Utility for computing per-lemma topic weights from a Gensim LDA model.

These weights can be passed directly to TopicalPageRank(topic_weights=...).

Requires the ``gensim`` optional dependency::

    pip install rapid_textrank[topic]
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def topic_weights_from_lda(
    lda_model: Any,
    corpus_entry: list[tuple[int, int | float]],
    dictionary: Any,
    top_n_words: int = 50,
    aggregation: str = "max",
) -> dict[str, float]:
    """Compute per-lemma topic weights from a Gensim LDA model.

    For each topic that the document belongs to, this function retrieves the
    top words and computes ``P(topic|doc) * P(word|topic)`` for every word.
    Scores are aggregated across topics using either *max* or *mean*.

    Parameters
    ----------
    lda_model
        A trained ``gensim.models.LdaModel`` (or ``LdaMulticore``).
    corpus_entry
        Bag-of-words representation of a single document, i.e. a list of
        ``(token_id, count)`` tuples as returned by
        ``dictionary.doc2bow(tokens)``.
    dictionary
        A ``gensim.corpora.Dictionary`` mapping token IDs to words.
    top_n_words
        Number of top words to retrieve per topic (default 50).
    aggregation
        How to aggregate a word's weight across multiple topics.
        ``"max"`` (default) keeps the highest weight; ``"mean"`` averages.

    Returns
    -------
    dict[str, float]
        Mapping from lemma strings to importance weights, suitable for
        ``TopicalPageRank(topic_weights=...)``.

    Raises
    ------
    ImportError
        If ``gensim`` is not installed.
    ValueError
        If *aggregation* is not ``"max"`` or ``"mean"``.

    Example
    -------
    >>> from gensim.models import LdaModel
    >>> from gensim.corpora import Dictionary
    >>> weights = topic_weights_from_lda(lda, corpus[0], dictionary)
    >>> from rapid_textrank import TopicalPageRank
    >>> extractor = TopicalPageRank(topic_weights=weights)
    """
    try:
        import gensim  # noqa: F401
    except ImportError:
        raise ImportError(
            "gensim is required for topic_weights_from_lda. "
            "Install it with: pip install rapid_textrank[topic]"
        )

    if aggregation not in ("max", "mean"):
        raise ValueError(
            f"aggregation must be 'max' or 'mean', got {aggregation!r}"
        )

    # Get P(topic | document) distribution
    doc_topics = lda_model.get_document_topics(corpus_entry)
    if not doc_topics:
        return {}

    # Accumulate word weights across topics
    word_weights: dict[str, list[float]] = defaultdict(list)

    for topic_id, topic_prob in doc_topics:
        # show_topic returns list of (word, word_prob_in_topic)
        top_words = lda_model.show_topic(topic_id, topn=top_n_words)
        for word, word_prob in top_words:
            weight = topic_prob * word_prob
            word_weights[word].append(weight)

    # Aggregate across topics
    result: dict[str, float] = {}
    for word, weights in word_weights.items():
        if aggregation == "max":
            result[word] = max(weights)
        else:  # mean
            result[word] = sum(weights) / len(weights)

    return result
