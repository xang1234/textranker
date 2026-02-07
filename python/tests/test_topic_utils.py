"""
Tests for rapid_textrank.topic_utils — topic weight helpers for TopicalPageRank.

All tests use mock objects so gensim is NOT required to run them.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock LDA helpers
# ---------------------------------------------------------------------------

def _make_lda(doc_topics, topic_words):
    """Create a mock LDA model.

    Parameters
    ----------
    doc_topics : list[(int, float)]
        Document-topic distribution returned by get_document_topics.
    topic_words : dict[int, list[(str, float)]]
        Per-topic word distributions returned by show_topic.
    """
    model = MagicMock()
    model.get_document_topics.return_value = doc_topics
    model.show_topic.side_effect = lambda tid, topn=10: topic_words.get(tid, [])[:topn]
    return model


def _make_dictionary():
    """Create a mock gensim Dictionary (unused directly by the helper)."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_gensim_importable(monkeypatch):
    """Ensure `import gensim` succeeds during tests by injecting a stub."""
    if "gensim" not in sys.modules:
        monkeypatch.setitem(sys.modules, "gensim", ModuleType("gensim"))


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestTopicWeightsFromLda:
    """Core tests for topic_weights_from_lda."""

    def test_single_topic(self):
        """Single topic produces word weights = topic_prob * word_prob."""
        from rapid_textrank.topic_utils import topic_weights_from_lda

        lda = _make_lda(
            doc_topics=[(0, 0.8)],
            topic_words={0: [("machine", 0.1), ("learning", 0.05)]},
        )
        result = topic_weights_from_lda(lda, [(0, 1)], _make_dictionary())

        assert pytest.approx(result["machine"]) == 0.8 * 0.1
        assert pytest.approx(result["learning"]) == 0.8 * 0.05

    def test_multiple_topics_max_aggregation(self):
        """With aggregation='max', the highest weight wins."""
        from rapid_textrank.topic_utils import topic_weights_from_lda

        lda = _make_lda(
            doc_topics=[(0, 0.6), (1, 0.4)],
            topic_words={
                0: [("data", 0.2)],
                1: [("data", 0.5)],
            },
        )
        result = topic_weights_from_lda(
            lda, [(0, 1)], _make_dictionary(), aggregation="max"
        )

        # topic 0: 0.6 * 0.2 = 0.12; topic 1: 0.4 * 0.5 = 0.20
        assert pytest.approx(result["data"]) == 0.20

    def test_multiple_topics_mean_aggregation(self):
        """With aggregation='mean', weights are averaged."""
        from rapid_textrank.topic_utils import topic_weights_from_lda

        lda = _make_lda(
            doc_topics=[(0, 0.6), (1, 0.4)],
            topic_words={
                0: [("data", 0.2)],
                1: [("data", 0.5)],
            },
        )
        result = topic_weights_from_lda(
            lda, [(0, 1)], _make_dictionary(), aggregation="mean"
        )

        # (0.12 + 0.20) / 2 = 0.16
        assert pytest.approx(result["data"]) == 0.16

    def test_top_n_words_limits_output(self):
        """top_n_words limits how many words are retrieved per topic."""
        from rapid_textrank.topic_utils import topic_weights_from_lda

        words = [(f"w{i}", 0.01) for i in range(100)]
        lda = _make_lda(doc_topics=[(0, 1.0)], topic_words={0: words})

        result = topic_weights_from_lda(
            lda, [(0, 1)], _make_dictionary(), top_n_words=5
        )
        assert len(result) == 5

    def test_all_values_positive(self):
        """All returned weights must be positive floats."""
        from rapid_textrank.topic_utils import topic_weights_from_lda

        lda = _make_lda(
            doc_topics=[(0, 0.5), (1, 0.5)],
            topic_words={
                0: [("alpha", 0.3), ("beta", 0.2)],
                1: [("gamma", 0.4)],
            },
        )
        result = topic_weights_from_lda(lda, [(0, 1)], _make_dictionary())

        assert all(isinstance(v, float) for v in result.values())
        assert all(v > 0 for v in result.values())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_corpus(self):
        """Empty document-topic distribution returns empty dict."""
        from rapid_textrank.topic_utils import topic_weights_from_lda

        lda = _make_lda(doc_topics=[], topic_words={})
        result = topic_weights_from_lda(lda, [], _make_dictionary())

        assert result == {}

    def test_topic_with_no_words(self):
        """Topic that returns no words is silently skipped."""
        from rapid_textrank.topic_utils import topic_weights_from_lda

        lda = _make_lda(
            doc_topics=[(0, 1.0)],
            topic_words={0: []},
        )
        result = topic_weights_from_lda(lda, [(0, 1)], _make_dictionary())

        assert result == {}

    def test_invalid_aggregation(self):
        """Invalid aggregation value raises ValueError."""
        from rapid_textrank.topic_utils import topic_weights_from_lda

        lda = _make_lda(doc_topics=[(0, 1.0)], topic_words={0: [("x", 0.1)]})

        with pytest.raises(ValueError, match="aggregation must be"):
            topic_weights_from_lda(
                lda, [(0, 1)], _make_dictionary(), aggregation="sum"
            )

    def test_import_error_without_gensim(self, monkeypatch):
        """Clear error message when gensim is not installed."""
        from rapid_textrank import topic_utils

        # Temporarily make `import gensim` fail
        import builtins

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "gensim":
                raise ImportError("No module named 'gensim'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        # Remove cached module so the guarded import runs again
        monkeypatch.delitem(sys.modules, "gensim", raising=False)

        with pytest.raises(ImportError, match="rapid_textrank\\[topic\\]"):
            topic_utils.topic_weights_from_lda(
                MagicMock(), [(0, 1)], MagicMock()
            )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

class TestPublicApi:
    """Verify the lazy re-export in rapid_textrank.__init__."""

    def test_importable_from_top_level(self):
        """topic_weights_from_lda is accessible from rapid_textrank."""
        from rapid_textrank import topic_weights_from_lda

        assert callable(topic_weights_from_lda)

    def test_in_all(self):
        """topic_weights_from_lda appears in __all__."""
        import rapid_textrank

        assert "topic_weights_from_lda" in rapid_textrank.__all__
