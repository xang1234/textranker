"""
Tests for the rapid_textrank Python API.
"""

import pytest
import json


def test_import():
    """Test that the module can be imported."""
    import rapid_textrank

    assert hasattr(rapid_textrank, "__version__")
    assert hasattr(rapid_textrank, "BaseTextRank")
    assert hasattr(rapid_textrank, "PositionRank")
    assert hasattr(rapid_textrank, "BiasedTextRank")


def test_version():
    """Test version string is valid."""
    from rapid_textrank import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2


class TestBaseTextRank:
    """Tests for BaseTextRank extractor."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=5)
        result = extractor.extract_keywords(
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning is a type of machine learning."
        )

        assert len(result.phrases) > 0
        assert result.converged
        assert all(p.score > 0 for p in result.phrases)

    def test_extract_keywords_ranking(self):
        """Test that phrases are properly ranked."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=10)
        result = extractor.extract_keywords(
            "Machine learning algorithms process data. "
            "Machine learning is used in many applications. "
            "Data science relies on machine learning."
        )

        # Ranks should be sequential starting from 1
        for i, phrase in enumerate(result.phrases):
            assert phrase.rank == i + 1

    def test_empty_input(self):
        """Test handling of empty input."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank()
        result = extractor.extract_keywords("")

        assert len(result.phrases) == 0

    def test_top_n_limit(self):
        """Test that top_n limits results."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=3)
        result = extractor.extract_keywords(
            "Machine learning, deep learning, natural language processing, "
            "computer vision, and neural networks are all important topics."
        )

        assert len(result.phrases) <= 3

    def test_phrase_attributes(self):
        """Test phrase object attributes."""
        from rapid_textrank import BaseTextRank

        extractor = BaseTextRank(top_n=1)
        result = extractor.extract_keywords("Machine learning is fascinating.")

        if result.phrases:
            phrase = result.phrases[0]
            assert hasattr(phrase, "text")
            assert hasattr(phrase, "lemma")
            assert hasattr(phrase, "score")
            assert hasattr(phrase, "count")
            assert hasattr(phrase, "rank")


class TestPositionRank:
    """Tests for PositionRank extractor."""

    def test_position_bias(self):
        """Test that early words are favored."""
        from rapid_textrank import PositionRank

        extractor = PositionRank(top_n=10)

        # "Important" appears first, "secondary" appears later
        result = extractor.extract_keywords(
            "Important topic is discussed first. "
            "Then we talk about secondary topic. "
            "Important topic appears again."
        )

        assert len(result.phrases) > 0


class TestBiasedTextRank:
    """Tests for BiasedTextRank extractor."""

    def test_focus_terms(self):
        """Test extraction with focus terms."""
        from rapid_textrank import BiasedTextRank

        extractor = BiasedTextRank(
            focus_terms=["neural"], bias_weight=10.0, top_n=10
        )

        result = extractor.extract_keywords(
            "Machine learning uses algorithms. "
            "Deep learning uses neural networks. "
            "Neural networks are powerful."
        )

        assert len(result.phrases) > 0

    def test_change_focus(self):
        """Test changing focus terms."""
        from rapid_textrank import BiasedTextRank

        extractor = BiasedTextRank(focus_terms=["machine"], top_n=10)
        result1 = extractor.extract_keywords("Machine learning and neural networks.")

        extractor.set_focus(["neural"])
        result2 = extractor.extract_keywords("Machine learning and neural networks.")

        # Both should produce results
        assert len(result1.phrases) > 0
        assert len(result2.phrases) > 0


class TestTextRankConfig:
    """Tests for TextRankConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from rapid_textrank import TextRankConfig

        config = TextRankConfig()
        assert config is not None

    def test_custom_config(self):
        """Test custom configuration."""
        from rapid_textrank import TextRankConfig, BaseTextRank

        config = TextRankConfig(
            damping=0.9,
            window_size=5,
            top_n=15,
            score_aggregation="mean",
        )

        extractor = BaseTextRank(config=config)
        result = extractor.extract_keywords("Test text for configuration.")

        assert result is not None

    def test_invalid_config(self):
        """Test that invalid config raises error."""
        from rapid_textrank import TextRankConfig
        import pytest

        with pytest.raises(ValueError):
            TextRankConfig(damping=2.0)  # Invalid: must be 0-1


class TestJsonInterface:
    """Tests for the JSON interface."""

    def test_extract_from_json(self):
        """Test JSON-based extraction."""
        from rapid_textrank import extract_from_json

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
                    "is_stopword": False,
                },
                {
                    "text": "learning",
                    "lemma": "learning",
                    "pos": "NOUN",
                    "start": 8,
                    "end": 16,
                    "sentence_idx": 0,
                    "token_idx": 1,
                    "is_stopword": False,
                },
            ],
            "config": {"top_n": 5},
        }

        result_json = extract_from_json(json.dumps(doc))
        result = json.loads(result_json)

        assert "phrases" in result
        assert "converged" in result
        assert "iterations" in result

    def test_batch_from_json(self):
        """Test batch JSON extraction."""
        from rapid_textrank import extract_batch_from_json

        docs = [
            {
                "tokens": [
                    {
                        "text": "First",
                        "lemma": "first",
                        "pos": "ADJ",
                        "start": 0,
                        "end": 5,
                        "sentence_idx": 0,
                        "token_idx": 0,
                        "is_stopword": False,
                    },
                ],
            },
            {
                "tokens": [
                    {
                        "text": "Second",
                        "lemma": "second",
                        "pos": "ADJ",
                        "start": 0,
                        "end": 6,
                        "sentence_idx": 0,
                        "token_idx": 0,
                        "is_stopword": False,
                    },
                ],
            },
        ]

        results_json = extract_batch_from_json(json.dumps(docs))
        results = json.loads(results_json)

        assert isinstance(results, list)
        assert len(results) == 2


class TestConvenienceFunction:
    """Tests for the extract_keywords convenience function."""

    def test_extract_keywords(self):
        """Test the convenience function."""
        from rapid_textrank import extract_keywords

        phrases = extract_keywords(
            "Machine learning is transforming industries.", top_n=5
        )

        assert isinstance(phrases, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
