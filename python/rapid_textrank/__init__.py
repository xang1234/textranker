"""
rapid_textrank - High-performance TextRank implementation

A fast TextRank implementation in Rust with Python bindings,
providing keyword extraction and text summarization.
"""

from rapid_textrank._rust import (
    __version__,
    Phrase,
    TextRankResult,
    TextRankConfig,
    BaseTextRank,
    PositionRank,
    BiasedTextRank,
    extract_from_json,
    extract_batch_from_json,
)

__all__ = [
    "__version__",
    "Phrase",
    "TextRankResult",
    "TextRankConfig",
    "BaseTextRank",
    "PositionRank",
    "BiasedTextRank",
    "extract_from_json",
    "extract_batch_from_json",
]


def extract_keywords(text: str, top_n: int = 10, language: str = "en") -> list:
    """
    Extract keywords from text using TextRank.

    Args:
        text: The input text to extract keywords from
        top_n: Number of top keywords to return
        language: Language code for stopword filtering

    Returns:
        List of Phrase objects with text, score, and rank

    Example:
        >>> from rapid_textrank import extract_keywords
        >>> phrases = extract_keywords("Machine learning is a subset of AI.")
        >>> for phrase in phrases:
        ...     print(f"{phrase.text}: {phrase.score:.4f}")
    """
    extractor = BaseTextRank(top_n=top_n, language=language)
    result = extractor.extract_keywords(text)
    return list(result.phrases)
