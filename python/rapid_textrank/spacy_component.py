"""
spaCy pipeline component for rapid_textrank.

This module provides a spaCy pipeline component that uses rapid_textrank
for keyword extraction. It can be used as a drop-in replacement for
pytextrank with significantly better performance.

Example:
    >>> import spacy
    >>> from rapid_textrank.spacy_component import RustTextRank
    >>>
    >>> nlp = spacy.load("en_core_web_sm")
    >>> nlp.add_pipe("rapid_textrank")
    >>>
    >>> doc = nlp("Machine learning is a subset of artificial intelligence.")
    >>> for phrase in doc._.phrases[:5]:
    ...     print(f"{phrase.text}: {phrase.score:.4f}")
"""

from typing import List, Optional, Dict, Any
import json

try:
    from spacy.tokens import Doc
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Doc = None
    Language = None

from rapid_textrank._rust import extract_from_json


class Phrase:
    """
    A keyphrase extracted by RustTextRank.

    Compatible with pytextrank's Phrase interface.
    """

    def __init__(self, text: str, lemma: str, score: float, count: int, rank: int):
        self.text = text
        self.lemma = lemma
        self.score = score
        self.count = count
        self.rank = rank
        # For pytextrank compatibility
        self.chunks = []

    def __repr__(self) -> str:
        return f"Phrase(text='{self.text}', score={self.score:.4f}, rank={self.rank})"

    def __str__(self) -> str:
        return self.text


class RustTextRankResult:
    """Result container for TextRank extraction."""

    def __init__(self, phrases: List[Phrase], converged: bool, iterations: int):
        self.phrases = phrases
        self.converged = converged
        self.iterations = iterations

    def __len__(self) -> int:
        return len(self.phrases)

    def __iter__(self):
        return iter(self.phrases)


if SPACY_AVAILABLE:

    @Language.factory(
        "rapid_textrank",
        default_config={
            "damping": 0.85,
            "max_iterations": 100,
            "convergence_threshold": 1e-6,
            "window_size": 3,
            "top_n": 10,
            "min_phrase_length": 1,
            "max_phrase_length": 4,
            "score_aggregation": "sum",
            "include_pos": ["ADJ", "NOUN", "PROPN", "VERB"],
            "use_pos_in_nodes": True,
            "phrase_grouping": "scrubbed_text",
            "language": "en",
            "stopwords": None,
        },
    )
    def create_rapid_textrank(
        nlp: Language,
        name: str,
        damping: float,
        max_iterations: int,
        convergence_threshold: float,
        window_size: int,
        top_n: int,
        min_phrase_length: int,
        max_phrase_length: int,
        score_aggregation: str,
        include_pos: Optional[List[str]],
        use_pos_in_nodes: bool,
        phrase_grouping: str,
        language: str,
        stopwords: Optional[List[str]],
    ):
        """Create a RustTextRank pipeline component."""
        return RustTextRank(
            nlp=nlp,
            name=name,
            damping=damping,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            window_size=window_size,
            top_n=top_n,
            min_phrase_length=min_phrase_length,
            max_phrase_length=max_phrase_length,
            score_aggregation=score_aggregation,
            include_pos=include_pos,
            use_pos_in_nodes=use_pos_in_nodes,
            phrase_grouping=phrase_grouping,
            language=language,
            stopwords=stopwords,
        )

    class RustTextRank:
        """
        spaCy pipeline component for TextRank keyword extraction.

        This component uses the Rust implementation for fast extraction
        while integrating seamlessly with spaCy's NLP pipeline.

        Example:
            >>> import spacy
            >>> nlp = spacy.load("en_core_web_sm")
            >>> nlp.add_pipe("rapid_textrank")
            >>> doc = nlp("Machine learning is transforming industries.")
            >>> for phrase in doc._.phrases:
            ...     print(phrase.text, phrase.score)
        """

        def __init__(
            self,
            nlp: Language,
            name: str = "rapid_textrank",
            damping: float = 0.85,
            max_iterations: int = 100,
            convergence_threshold: float = 1e-6,
            window_size: int = 3,
            top_n: int = 10,
            min_phrase_length: int = 1,
            max_phrase_length: int = 4,
            score_aggregation: str = "sum",
            include_pos: Optional[List[str]] = None,
            use_pos_in_nodes: bool = True,
            phrase_grouping: str = "scrubbed_text",
            language: str = "en",
            stopwords: Optional[List[str]] = None,
        ):
            self.nlp = nlp
            self.name = name
            self.config = {
                "damping": damping,
                "max_iterations": max_iterations,
                "convergence_threshold": convergence_threshold,
                "window_size": window_size,
                "top_n": top_n,
                "min_phrase_length": min_phrase_length,
                "max_phrase_length": max_phrase_length,
                "score_aggregation": score_aggregation,
                "use_pos_in_nodes": use_pos_in_nodes,
                "phrase_grouping": phrase_grouping,
                "language": language,
            }
            if include_pos is not None:
                self.config["include_pos"] = include_pos
            if stopwords is not None:
                self.config["stopwords"] = stopwords

            # Register custom extensions
            if not Doc.has_extension("phrases"):
                Doc.set_extension("phrases", default=[])
            if not Doc.has_extension("textrank_result"):
                Doc.set_extension("textrank_result", default=None)

        def __call__(self, doc: Doc) -> Doc:
            """Process a spaCy Doc and extract keyphrases."""
            # Convert spaCy tokens to JSON format
            tokens = []
            for sent_idx, sent in enumerate(doc.sents):
                for token in sent:
                    tokens.append(
                        {
                            "text": token.text,
                            "lemma": token.lemma_,
                            "pos": token.pos_,
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "sentence_idx": sent_idx,
                            "token_idx": token.i,
                            "is_stopword": token.is_stop,
                        }
                    )

            # Create JSON input
            json_input = json.dumps({"tokens": tokens, "config": self.config})

            # Extract keyphrases using Rust
            json_output = extract_from_json(json_input)
            result = json.loads(json_output)

            # Convert to Phrase objects
            phrases = [
                Phrase(
                    text=p["text"],
                    lemma=p["lemma"],
                    score=p["score"],
                    count=p["count"],
                    rank=p["rank"],
                )
                for p in result["phrases"]
            ]

            # Store results
            doc._.phrases = phrases
            doc._.textrank_result = RustTextRankResult(
                phrases=phrases,
                converged=result["converged"],
                iterations=result["iterations"],
            )

            return doc

        def to_disk(self, path, **kwargs):
            """Save component configuration to disk."""
            import json
            from pathlib import Path

            config_path = Path(path) / "config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f)

        def from_disk(self, path, **kwargs):
            """Load component configuration from disk."""
            import json
            from pathlib import Path

            config_path = Path(path) / "config.json"
            with open(config_path, "r") as f:
                self.config = json.load(f)
            return self

else:
    # Fallback when spaCy is not available
    class RustTextRank:
        """Placeholder when spaCy is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "spaCy is required for the RustTextRank pipeline component. "
                "Install it with: pip install spacy"
            )
