//! Stage trait definitions for the pipeline.
//!
//! Each trait represents one processing stage boundary. Implementations are
//! statically dispatched for performance; trait objects are available behind a
//! feature gate for dynamic composition.

use crate::pipeline::artifacts::{
    CandidateKind, CandidateSet, PhraseCandidate, TokenStream, TokenStreamRef, WordCandidate,
};
use crate::types::{ChunkSpan, TextRankConfig};

// ============================================================================
// Preprocessor — optional token normalization (stage 0)
// ============================================================================

/// Optional preprocessing / normalization stage.
///
/// Centralizes normalization differences between the built-in tokenizer
/// (Unicode-aware) and spaCy / JSON-provided tokens, without duplicating
/// rules across downstream stages (CandidateSelector, GraphBuilder,
/// PhraseBuilder).
///
/// Most variants don't need custom preprocessing — the provided
/// [`NoopPreprocessor`] is the default.
///
/// # Contract
///
/// - **Input**: a mutable [`TokenStream`] (modify in-place to avoid
///   allocating a second stream).
/// - **Output**: none — the stream is mutated.
/// - **Idempotent**: calling `preprocess` twice should produce the same
///   result as calling it once.
///
/// # Examples
///
/// A concrete preprocessor might:
/// - Re-lemmatize tokens using a different strategy
/// - Override POS tags for known domain terms
/// - Mark additional stopwords based on a custom list
/// - Normalize Unicode forms (NFC/NFKC)
pub trait Preprocessor {
    /// Preprocess the token stream in place.
    fn preprocess(&self, tokens: &mut TokenStream, cfg: &TextRankConfig);
}

/// No-op preprocessor — the default for most pipeline configurations.
///
/// Passes the token stream through unchanged, with zero overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopPreprocessor;

impl Preprocessor for NoopPreprocessor {
    #[inline]
    fn preprocess(&self, _tokens: &mut TokenStream, _cfg: &TextRankConfig) {
        // Intentionally empty.
    }
}

// ============================================================================
// CandidateSelector — token stream to candidate nodes (stage 1)
// ============================================================================

/// Selects candidate nodes from a token stream for graph construction.
///
/// Two families of candidate selection exist in the TextRank ecosystem:
///
/// - **Word-level** ([`WordNodeSelector`]): filters individual tokens by POS
///   tag and stopword status, deduplicates by graph key. Used by BaseTextRank,
///   PositionRank, BiasedTextRank, SingleRank, and TopicalPageRank.
///
/// - **Phrase-level** ([`PhraseCandidateSelector`]): extracts noun-phrase
///   chunks as multi-token candidates for topic-based clustering. Used by
///   TopicRank and MultipartiteRank.
///
/// # Contract
///
/// - **Input**: a borrowed [`TokenStreamRef`] (read-only) and config.
/// - **Output**: a [`CandidateSet`] containing either word or phrase
///   candidates.
/// - **Deterministic**: same input → same output (no internal randomness).
pub trait CandidateSelector {
    /// Select candidates from the token stream.
    fn select(&self, tokens: TokenStreamRef<'_>, cfg: &TextRankConfig) -> CandidateSet;
}

/// Word-level candidate selector for the TextRank family.
///
/// Filters tokens by POS tag and stopword status, deduplicates by graph key
/// (`lemma` or `lemma|POS`), and records the first occurrence position.
///
/// This is the selector used by BaseTextRank, PositionRank, BiasedTextRank,
/// SingleRank, and TopicalPageRank. It reads `include_pos` and
/// `use_pos_in_nodes` from [`TextRankConfig`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WordNodeSelector;

impl CandidateSelector for WordNodeSelector {
    fn select(&self, tokens: TokenStreamRef<'_>, cfg: &TextRankConfig) -> CandidateSet {
        use rustc_hash::FxHashMap;
        use crate::types::PosTag;

        // Key: (lemma_id, optional POS discriminant) → index into `words`.
        let mut seen: FxHashMap<(u32, Option<PosTag>), usize> = FxHashMap::default();
        let mut words = Vec::new();

        for entry in tokens.tokens() {
            if entry.is_stopword {
                continue;
            }
            let pass = if cfg.include_pos.is_empty() {
                entry.pos.is_content_word()
            } else {
                cfg.include_pos.contains(&entry.pos)
            };
            if !pass {
                continue;
            }

            let key = if cfg.use_pos_in_nodes {
                (entry.lemma_id, Some(entry.pos))
            } else {
                (entry.lemma_id, None)
            };

            if !seen.contains_key(&key) {
                seen.insert(key, words.len());
                words.push(WordCandidate {
                    lemma_id: entry.lemma_id,
                    pos: entry.pos,
                    first_position: entry.token_idx,
                });
            }
        }

        CandidateSet::from_kind(CandidateKind::Words(words))
    }
}

/// Phrase-level candidate selector for the TopicRank / MultipartiteRank family.
///
/// Operates on pre-computed noun-phrase chunk spans ([`ChunkSpan`]) and the
/// token stream to build [`PhraseCandidate`] entries with interned lemma and
/// term IDs for downstream Jaccard-based clustering.
///
/// # Construction
///
/// Pass pre-computed chunks via [`PhraseCandidateSelector::new`]. In the full
/// pipeline, these chunks come from the [`NounChunker`](crate::phrase::chunker::NounChunker)
/// run on the original token data.
#[derive(Debug, Clone)]
pub struct PhraseCandidateSelector {
    chunks: Vec<ChunkSpan>,
}

impl PhraseCandidateSelector {
    /// Create a phrase selector with pre-computed chunk spans.
    pub fn new(chunks: Vec<ChunkSpan>) -> Self {
        Self { chunks }
    }
}

impl CandidateSelector for PhraseCandidateSelector {
    fn select(&self, tokens: TokenStreamRef<'_>, _cfg: &TextRankConfig) -> CandidateSet {
        let mut phrases = Vec::with_capacity(self.chunks.len());

        for chunk in &self.chunks {
            let start = chunk.start_token;
            let end = chunk.end_token;
            if start >= end || end > tokens.len() {
                continue;
            }

            let mut lemma_ids = Vec::with_capacity(end - start);
            let mut term_ids = Vec::new();

            for &entry in &tokens.tokens()[start..end] {
                lemma_ids.push(entry.lemma_id);
                if !entry.is_stopword {
                    // Use text_id for term set (matches legacy PhraseCandidate).
                    if !term_ids.contains(&entry.text_id) {
                        term_ids.push(entry.text_id);
                    }
                }
            }

            phrases.push(PhraseCandidate {
                start_token: start as u32,
                end_token: end as u32,
                start_char: chunk.start_char as u32,
                end_char: chunk.end_char as u32,
                sentence_idx: chunk.sentence_idx as u32,
                lemma_ids,
                term_ids,
            });
        }

        CandidateSet::from_kind(CandidateKind::Phrases(phrases))
    }
}

// Future E3 stage traits will be added below:
//   - GraphBuilder          (textranker-4a0.3)
//   - GraphTransform        (textranker-4a0.4)
//   - TeleportBuilder       (textranker-4a0.5)
//   - Ranker                (textranker-4a0.6)
//   - PhraseBuilder         (textranker-4a0.7)
//   - ResultFormatter       (textranker-4a0.8)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::artifacts::CandidateKind;
    use crate::types::{ChunkSpan, PosTag, Token};

    fn sample_tokens() -> Vec<Token> {
        vec![
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
        ]
    }

    /// Richer token set with stopwords, multiple sentences, mixed POS.
    fn rich_tokens() -> Vec<Token> {
        let mut tokens = vec![
            // Sentence 0: "Machine learning is great"
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
            Token::new("great", "great", PosTag::Adjective, 20, 25, 0, 3),
            // Sentence 1: "Rust is fast"
            Token::new("Rust", "rust", PosTag::ProperNoun, 27, 31, 1, 4),
            Token::new("is", "be", PosTag::Verb, 32, 34, 1, 5),
            Token::new("fast", "fast", PosTag::Adjective, 35, 39, 1, 6),
        ];
        tokens[2].is_stopword = true; // "is" sentence 0
        tokens[5].is_stopword = true; // "is" sentence 1
        tokens
    }

    // ================================================================
    // Preprocessor tests
    // ================================================================

    #[test]
    fn test_noop_preprocessor_preserves_tokens() {
        let tokens = sample_tokens();
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let snapshot_len = stream.len();
        let snapshot_text0 = stream.text(&stream.tokens()[0]).to_string();
        let snapshot_pos0 = stream.tokens()[0].pos;

        NoopPreprocessor.preprocess(&mut stream, &cfg);

        assert_eq!(stream.len(), snapshot_len);
        assert_eq!(stream.text(&stream.tokens()[0]), snapshot_text0);
        assert_eq!(stream.tokens()[0].pos, snapshot_pos0);
    }

    #[test]
    fn test_noop_preprocessor_is_idempotent() {
        let tokens = sample_tokens();
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        NoopPreprocessor.preprocess(&mut stream, &cfg);
        let after_first: Vec<_> = stream.tokens().to_vec();

        NoopPreprocessor.preprocess(&mut stream, &cfg);
        let after_second: Vec<_> = stream.tokens().to_vec();

        assert_eq!(after_first, after_second);
    }

    #[test]
    fn test_noop_preprocessor_on_empty_stream() {
        let mut stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();

        NoopPreprocessor.preprocess(&mut stream, &cfg);

        assert!(stream.is_empty());
        assert_eq!(stream.num_sentences(), 0);
    }

    #[test]
    fn test_custom_preprocessor_marks_stopwords() {
        struct MarkVerbsAsStopwords;

        impl Preprocessor for MarkVerbsAsStopwords {
            fn preprocess(&self, tokens: &mut TokenStream, _cfg: &TextRankConfig) {
                for entry in tokens.tokens_mut() {
                    if entry.pos == PosTag::Verb {
                        entry.is_stopword = true;
                    }
                }
            }
        }

        let tokens = sample_tokens();
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        assert!(!stream.tokens()[2].is_stopword);
        MarkVerbsAsStopwords.preprocess(&mut stream, &cfg);
        assert!(stream.tokens()[2].is_stopword);
    }

    #[test]
    fn test_custom_preprocessor_relemmatize() {
        struct Lowercaser;

        impl Preprocessor for Lowercaser {
            fn preprocess(&self, tokens: &mut TokenStream, _cfg: &TextRankConfig) {
                let new_lemmas: Vec<String> = tokens
                    .tokens()
                    .iter()
                    .map(|e| tokens.pool().get(e.lemma_id).unwrap_or("").to_lowercase())
                    .collect();

                for (i, new_lemma) in new_lemmas.into_iter().enumerate() {
                    let new_id = tokens.pool_mut().intern(&new_lemma);
                    tokens.tokens_mut()[i].lemma_id = new_id;
                }
            }
        }

        let tokens = vec![
            Token::new("Machine", "Machine", PosTag::Noun, 0, 7, 0, 0),
        ];
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        assert_eq!(stream.lemma(&stream.tokens()[0]), "Machine");
        Lowercaser.preprocess(&mut stream, &cfg);
        assert_eq!(stream.lemma(&stream.tokens()[0]), "machine");
    }

    #[test]
    fn test_preprocessor_as_trait_object() {
        let preprocessor: Box<dyn Preprocessor> = Box::new(NoopPreprocessor);

        let tokens = sample_tokens();
        let mut stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        preprocessor.preprocess(&mut stream, &cfg);
        assert_eq!(stream.len(), 3);
    }

    #[test]
    fn test_noop_preprocessor_default() {
        let _p = NoopPreprocessor::default();
    }

    // ================================================================
    // CandidateSelector — WordNodeSelector tests
    // ================================================================

    #[test]
    fn test_word_selector_default_pos_filtering() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);

        assert!(matches!(cs.kind(), CandidateKind::Words(_)));
        // Content words (non-stopword): machine NOUN, learning NOUN, great ADJ,
        // rust PROPN, fast ADJ → 5 candidates.
        assert_eq!(cs.len(), 5);
    }

    #[test]
    fn test_word_selector_custom_include_pos() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.include_pos = vec![PosTag::Noun]; // Only nouns.

        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);

        // Only "machine" NOUN and "learning" NOUN pass.
        let words = cs.words();
        assert_eq!(words.len(), 2);
        for w in words {
            assert_eq!(w.pos, PosTag::Noun);
        }
    }

    #[test]
    fn test_word_selector_dedup_with_pos_in_nodes() {
        let tokens = vec![
            Token::new("fast", "fast", PosTag::Adjective, 0, 4, 0, 0),
            Token::new("fast", "fast", PosTag::Adverb, 5, 9, 0, 1),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.include_pos = vec![PosTag::Adjective, PosTag::Adverb];

        // With POS: "fast|ADJ" and "fast|ADV" are distinct.
        cfg.use_pos_in_nodes = true;
        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        assert_eq!(cs.len(), 2);

        // Without POS: same lemma → deduplicated to one.
        cfg.use_pos_in_nodes = false;
        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        assert_eq!(cs.len(), 1);
    }

    #[test]
    fn test_word_selector_records_first_position() {
        let tokens = vec![
            Token::new("great", "great", PosTag::Adjective, 0, 5, 0, 0),
            Token::new("world", "world", PosTag::Noun, 6, 11, 0, 1),
            Token::new("great", "great", PosTag::Adjective, 12, 17, 0, 2),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        let words = cs.words();

        let great = words.iter().find(|w| {
            stream.pool().get(w.lemma_id) == Some("great")
        }).unwrap();
        assert_eq!(great.first_position, 0); // First occurrence, not 2.
    }

    #[test]
    fn test_word_selector_excludes_stopwords() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        // Include Verb so we can check stopword "is" is still excluded.
        cfg.include_pos = vec![PosTag::Noun, PosTag::Verb, PosTag::Adjective, PosTag::ProperNoun];

        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        let words = cs.words();

        // "be" (verb, stopword) should be excluded despite Verb being in include_pos.
        for w in words {
            let lemma = stream.pool().get(w.lemma_id).unwrap_or("");
            assert_ne!(lemma, "be", "stopword 'be' should be excluded");
        }
    }

    #[test]
    fn test_word_selector_empty_stream() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();

        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_word_selector_all_stopwords() {
        let mut tokens = vec![
            Token::new("the", "the", PosTag::Determiner, 0, 3, 0, 0),
            Token::new("is", "be", PosTag::Verb, 4, 6, 0, 1),
        ];
        tokens[0].is_stopword = true;
        tokens[1].is_stopword = true;

        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_word_selector_matches_artifact_bridge() {
        // WordNodeSelector should produce the same result as
        // CandidateSet::from_word_tokens for the same input.
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let from_selector = WordNodeSelector.select(stream.as_ref(), &cfg);
        let from_bridge = CandidateSet::from_word_tokens(
            &stream,
            &cfg.include_pos,
            cfg.use_pos_in_nodes,
        );

        assert_eq!(from_selector.len(), from_bridge.len());
        let sel_words = from_selector.words();
        let bridge_words = from_bridge.words();
        for (s, b) in sel_words.iter().zip(bridge_words.iter()) {
            assert_eq!(s.lemma_id, b.lemma_id);
            assert_eq!(s.pos, b.pos);
            assert_eq!(s.first_position, b.first_position);
        }
    }

    // ================================================================
    // CandidateSelector — PhraseCandidateSelector tests
    // ================================================================

    #[test]
    fn test_phrase_selector_basic() {
        let tokens = vec![
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
            Token::new("very", "very", PosTag::Adverb, 20, 24, 0, 3),
            Token::new("fast", "fast", PosTag::Adjective, 25, 29, 0, 4),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let chunks = vec![ChunkSpan {
            start_token: 0,
            end_token: 2,
            start_char: 0,
            end_char: 16,
            sentence_idx: 0,
        }];
        let selector = PhraseCandidateSelector::new(chunks);
        let cs = selector.select(stream.as_ref(), &cfg);

        assert!(matches!(cs.kind(), CandidateKind::Phrases(_)));
        assert_eq!(cs.len(), 1);

        let p = &cs.phrases()[0];
        assert_eq!(p.start_token, 0);
        assert_eq!(p.end_token, 2);
        assert_eq!(p.lemma_ids.len(), 2);
        assert_eq!(p.term_ids.len(), 2);
    }

    #[test]
    fn test_phrase_selector_multiple_chunks() {
        let tokens = vec![
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
            Token::new("deep", "deep", PosTag::Adjective, 20, 24, 0, 3),
            Token::new("learning", "learning", PosTag::Noun, 25, 33, 0, 4),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let chunks = vec![
            ChunkSpan { start_token: 0, end_token: 2, start_char: 0, end_char: 16, sentence_idx: 0 },
            ChunkSpan { start_token: 3, end_token: 5, start_char: 20, end_char: 33, sentence_idx: 0 },
        ];

        let cs = PhraseCandidateSelector::new(chunks).select(stream.as_ref(), &cfg);

        assert_eq!(cs.len(), 2);
        assert_eq!(cs.phrases()[0].start_token, 0);
        assert_eq!(cs.phrases()[1].start_token, 3);
    }

    #[test]
    fn test_phrase_selector_stopwords_excluded_from_terms() {
        let mut tokens = vec![
            Token::new("the", "the", PosTag::Determiner, 0, 3, 0, 0),
            Token::new("big", "big", PosTag::Adjective, 4, 7, 0, 1),
            Token::new("cat", "cat", PosTag::Noun, 8, 11, 0, 2),
        ];
        tokens[0].is_stopword = true;

        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let chunks = vec![ChunkSpan {
            start_token: 0, end_token: 3, start_char: 0, end_char: 11, sentence_idx: 0,
        }];

        let cs = PhraseCandidateSelector::new(chunks).select(stream.as_ref(), &cfg);
        let p = &cs.phrases()[0];

        assert_eq!(p.lemma_ids.len(), 3); // All tokens in lemma_ids.
        assert_eq!(p.term_ids.len(), 2);  // Only non-stopword in term_ids.
    }

    #[test]
    fn test_phrase_selector_empty_chunks() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let cs = PhraseCandidateSelector::new(vec![]).select(stream.as_ref(), &cfg);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_phrase_selector_empty_stream() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();

        // Chunks reference tokens that don't exist → skipped.
        let chunks = vec![ChunkSpan {
            start_token: 0, end_token: 2, start_char: 0, end_char: 10, sentence_idx: 0,
        }];
        let cs = PhraseCandidateSelector::new(chunks).select(stream.as_ref(), &cfg);
        assert!(cs.is_empty());
    }

    #[test]
    fn test_phrase_selector_invalid_chunk_skipped() {
        let tokens = vec![
            Token::new("hello", "hello", PosTag::Noun, 0, 5, 0, 0),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let chunks = vec![
            // Valid chunk.
            ChunkSpan { start_token: 0, end_token: 1, start_char: 0, end_char: 5, sentence_idx: 0 },
            // Invalid: end > stream length.
            ChunkSpan { start_token: 0, end_token: 99, start_char: 0, end_char: 100, sentence_idx: 0 },
            // Invalid: start >= end.
            ChunkSpan { start_token: 5, end_token: 3, start_char: 0, end_char: 0, sentence_idx: 0 },
        ];

        let cs = PhraseCandidateSelector::new(chunks).select(stream.as_ref(), &cfg);
        assert_eq!(cs.len(), 1); // Only the valid chunk.
    }

    // ================================================================
    // CandidateSelector — trait object tests
    // ================================================================

    #[test]
    fn test_candidate_selector_as_trait_object_word() {
        let selector: Box<dyn CandidateSelector> = Box::new(WordNodeSelector);
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let cs = selector.select(stream.as_ref(), &cfg);
        assert!(matches!(cs.kind(), CandidateKind::Words(_)));
        assert_eq!(cs.len(), 5);
    }

    #[test]
    fn test_candidate_selector_as_trait_object_phrase() {
        let tokens = vec![
            Token::new("big", "big", PosTag::Adjective, 0, 3, 0, 0),
            Token::new("cat", "cat", PosTag::Noun, 4, 7, 0, 1),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let chunks = vec![ChunkSpan {
            start_token: 0, end_token: 2, start_char: 0, end_char: 7, sentence_idx: 0,
        }];
        let selector: Box<dyn CandidateSelector> = Box::new(PhraseCandidateSelector::new(chunks));

        let cs = selector.select(stream.as_ref(), &cfg);
        assert!(matches!(cs.kind(), CandidateKind::Phrases(_)));
        assert_eq!(cs.len(), 1);
    }
}
