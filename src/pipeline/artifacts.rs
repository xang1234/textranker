//! First-class pipeline artifacts.
//!
//! Each type represents a typed intermediate result flowing between pipeline
//! stages. Artifacts use interned IDs internally; string materialization is
//! deferred to the formatting boundary ([`FormattedResult`]).
//!
//! **Owned vs Borrowed**: Hot-path stage interfaces accept `*Ref<'a>` borrows;
//! the pipeline retains ownership of the corresponding owned artifacts.

use crate::types::{PosTag, StringPool, Token};

// ============================================================================
// TokenStream — interned, compact token representation
// ============================================================================

/// A single token stored as interned IDs — compact and `Copy`.
///
/// All string data lives in the parent [`TokenStream`]'s [`StringPool`]; this
/// struct stores only `u32` handles.  At 28 bytes per entry (plus padding to
/// 32) it fits two entries per 64-byte cache line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEntry {
    /// Interned ID for the surface form in the parent pool.
    pub text_id: u32,
    /// Interned ID for the lemmatized form.
    pub lemma_id: u32,
    /// Part-of-speech tag (already an enum — no interning needed).
    pub pos: PosTag,
    /// Character byte-offset of the first byte in the source text.
    pub start: u32,
    /// Character byte-offset one past the last byte.
    pub end: u32,
    /// Sentence index (0-based).
    pub sentence_idx: u32,
    /// Token index within the document (0-based, monotonically increasing).
    pub token_idx: u32,
    /// Whether this token is a stopword.
    pub is_stopword: bool,
}

impl TokenEntry {
    /// Build the graph-node key, mirroring [`Token::graph_key`].
    ///
    /// When `use_pos_in_nodes` is true the key is `"lemma|POS"`, otherwise
    /// just the lemma string.  Requires access to the owning pool.
    #[inline]
    pub fn graph_key(&self, pool: &StringPool, use_pos_in_nodes: bool) -> String {
        let lemma = pool.get(self.lemma_id).unwrap_or("");
        if use_pos_in_nodes {
            format!("{}|{}", lemma, self.pos.as_str())
        } else {
            lemma.to_owned()
        }
    }

    /// Whether this token is a content-word candidate for the graph.
    #[inline]
    pub fn is_graph_candidate(&self) -> bool {
        self.pos.is_content_word() && !self.is_stopword
    }
}

/// Canonical token stream produced by the preprocessor stage.
///
/// Stores tokens as interned [`TokenEntry`] values backed by a shared
/// [`StringPool`].  Sentence boundaries use a CSR-style offset array:
/// `sentence_offsets[i]..sentence_offsets[i+1]` gives the token index range
/// for sentence `i`.
///
/// # Construction
///
/// Use [`TokenStream::from_tokens`] to convert from the legacy `&[Token]`
/// representation (used by the existing tokenizer and JSON input path).
#[derive(Debug)]
pub struct TokenStream {
    /// Interned string storage for text and lemma values.
    pool: StringPool,
    /// Compact, ID-based token entries.
    tokens: Vec<TokenEntry>,
    /// CSR-style sentence boundary offsets.
    ///
    /// Length = `num_sentences + 1`.  The last element equals `tokens.len()`.
    sentence_offsets: Vec<u32>,
}

impl TokenStream {
    /// Build a `TokenStream` from the legacy `Token` slice.
    ///
    /// This is the primary bridge between the existing tokenizer / JSON
    /// deserialization path and the new pipeline artifact system.  It interns
    /// all text and lemma strings into a single pool and computes sentence
    /// boundary offsets in a single pass.
    pub fn from_tokens(tokens: &[Token]) -> Self {
        let mut pool = StringPool::with_capacity(tokens.len());
        let mut entries = Vec::with_capacity(tokens.len());

        if tokens.is_empty() {
            return Self {
                pool,
                tokens: entries,
                sentence_offsets: Vec::new(),
            };
        }

        // Determine number of sentences for offset pre-allocation.
        let num_sentences = tokens
            .iter()
            .map(|t| t.sentence_idx)
            .max()
            .map_or(0, |m| m + 1);
        // +1 for the sentinel entry at the end.
        let mut sentence_offsets = Vec::with_capacity(num_sentences + 1);

        let mut current_sentence: usize = 0;
        sentence_offsets.push(0); // sentence 0 starts at token 0

        for (i, t) in tokens.iter().enumerate() {
            // Emit boundary markers for any new sentences.
            while t.sentence_idx > current_sentence {
                sentence_offsets.push(i as u32);
                current_sentence += 1;
            }

            let text_id = pool.intern(&t.text);
            let lemma_id = pool.intern(&t.lemma);

            entries.push(TokenEntry {
                text_id,
                lemma_id,
                pos: t.pos,
                start: t.start as u32,
                end: t.end as u32,
                sentence_idx: t.sentence_idx as u32,
                token_idx: t.token_idx as u32,
                is_stopword: t.is_stopword,
            });
        }

        // Sentinel: marks the end of the last sentence.
        sentence_offsets.push(entries.len() as u32);

        Self {
            pool,
            tokens: entries,
            sentence_offsets,
        }
    }

    /// Create a borrowed view for passing to pipeline stages.
    #[inline]
    pub fn as_ref(&self) -> TokenStreamRef<'_> {
        TokenStreamRef {
            pool: &self.pool,
            tokens: &self.tokens,
            sentence_offsets: &self.sentence_offsets,
        }
    }

    /// Number of tokens in the stream.
    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the stream contains no tokens.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Number of sentences.
    #[inline]
    pub fn num_sentences(&self) -> usize {
        // offsets has num_sentences + 1 entries.
        self.sentence_offsets.len().saturating_sub(1)
    }

    /// Access the shared string pool.
    #[inline]
    pub fn pool(&self) -> &StringPool {
        &self.pool
    }

    /// Access all token entries as a slice.
    #[inline]
    pub fn tokens(&self) -> &[TokenEntry] {
        &self.tokens
    }

    /// Access the CSR-style sentence boundary offsets.
    #[inline]
    pub fn sentence_offsets(&self) -> &[u32] {
        &self.sentence_offsets
    }

    /// Get the token index range for sentence `idx`.
    ///
    /// Returns `None` if `idx >= num_sentences()`.
    #[inline]
    pub fn sentence_token_range(&self, idx: usize) -> Option<std::ops::Range<usize>> {
        if idx + 1 < self.sentence_offsets.len() {
            let start = self.sentence_offsets[idx] as usize;
            let end = self.sentence_offsets[idx + 1] as usize;
            Some(start..end)
        } else {
            None
        }
    }

    /// Resolve the surface text for a token entry.
    #[inline]
    pub fn text(&self, entry: &TokenEntry) -> &str {
        self.pool.get(entry.text_id).unwrap_or("")
    }

    /// Resolve the lemma string for a token entry.
    #[inline]
    pub fn lemma(&self, entry: &TokenEntry) -> &str {
        self.pool.get(entry.lemma_id).unwrap_or("")
    }
}

/// Borrowed view into a [`TokenStream`].
///
/// This is the primary interface stages use to read tokens without requiring
/// ownership or allocation.  All three inner references point into the parent
/// `TokenStream`'s storage.
#[derive(Debug, Clone, Copy)]
pub struct TokenStreamRef<'a> {
    pool: &'a StringPool,
    tokens: &'a [TokenEntry],
    sentence_offsets: &'a [u32],
}

impl<'a> TokenStreamRef<'a> {
    /// Number of tokens.
    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the stream is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Number of sentences.
    #[inline]
    pub fn num_sentences(&self) -> usize {
        self.sentence_offsets.len().saturating_sub(1)
    }

    /// Access the shared string pool.
    #[inline]
    pub fn pool(&self) -> &'a StringPool {
        self.pool
    }

    /// Access all token entries as a slice.
    #[inline]
    pub fn tokens(&self) -> &'a [TokenEntry] {
        self.tokens
    }

    /// Access the sentence boundary offsets.
    #[inline]
    pub fn sentence_offsets(&self) -> &'a [u32] {
        self.sentence_offsets
    }

    /// Get the token index range for sentence `idx`.
    #[inline]
    pub fn sentence_token_range(&self, idx: usize) -> Option<std::ops::Range<usize>> {
        if idx + 1 < self.sentence_offsets.len() {
            let start = self.sentence_offsets[idx] as usize;
            let end = self.sentence_offsets[idx + 1] as usize;
            Some(start..end)
        } else {
            None
        }
    }

    /// Resolve the surface text for a token entry.
    #[inline]
    pub fn text(&self, entry: &TokenEntry) -> &'a str {
        self.pool.get(entry.text_id).unwrap_or("")
    }

    /// Resolve the lemma string for a token entry.
    #[inline]
    pub fn lemma(&self, entry: &TokenEntry) -> &'a str {
        self.pool.get(entry.lemma_id).unwrap_or("")
    }
}

/// Set of candidate nodes (word-level or phrase-level) selected for graph
/// construction.
///
/// Supports both word-node candidates (TextRank, PositionRank, etc.) and
/// phrase-level candidates (TopicRank, MultipartiteRank).
pub struct CandidateSet {
    _private: (),
}

/// Borrowed view into a [`CandidateSet`].
pub struct CandidateSetRef<'a> {
    _private: std::marker::PhantomData<&'a ()>,
}

/// Pipeline-level graph artifact wrapping the CSR-backed adjacency + weights.
///
/// Includes a node-index mapping from internal candidate IDs to CSR indices,
/// preserving cache-friendly iteration.
pub struct Graph {
    _private: (),
}

/// PageRank output: per-node scores, convergence info, and optional diagnostics.
pub struct RankOutput {
    _private: (),
}

/// Pre-format phrase collection: scored phrases with interned lemma IDs.
///
/// Surface forms are lazily materialized only when needed for formatting.
pub struct PhraseSet {
    _private: (),
}

/// Borrowed view into a [`PhraseSet`].
pub struct PhraseSetRef<'a> {
    _private: std::marker::PhantomData<&'a ()>,
}

/// Public-facing formatted output — the stability boundary.
///
/// Everything before this type is internal and may change; this type is the
/// public contract exposed to Python and JSON consumers.
pub struct FormattedResult {
    _private: (),
}

/// Reusable scratch buffers for reducing allocator churn across repeated
/// pipeline invocations (common in Python batch processing).
pub struct PipelineWorkspace {
    _private: (),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PosTag;

    /// Helper: build a small `Vec<Token>` for testing.
    fn sample_tokens() -> Vec<Token> {
        vec![
            // Sentence 0: "Machine learning is great"
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
            Token::new("great", "great", PosTag::Adjective, 20, 25, 0, 3),
            // Sentence 1: "Rust is fast"
            Token::new("Rust", "rust", PosTag::ProperNoun, 27, 31, 1, 4),
            Token::new("is", "be", PosTag::Verb, 32, 34, 1, 5),
            Token::new("fast", "fast", PosTag::Adjective, 35, 39, 1, 6),
        ]
    }

    #[test]
    fn test_from_tokens_basic() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);

        assert_eq!(stream.len(), 7);
        assert!(!stream.is_empty());
        assert_eq!(stream.num_sentences(), 2);
    }

    #[test]
    fn test_from_tokens_empty() {
        let stream = TokenStream::from_tokens(&[]);

        assert_eq!(stream.len(), 0);
        assert!(stream.is_empty());
        assert_eq!(stream.num_sentences(), 0);
    }

    #[test]
    fn test_string_interning_deduplicates() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);

        // "be" appears twice (tokens 2 and 5), "is" appears twice.
        // The pool should have fewer entries than total token count.
        // 7 tokens but: "Machine","machine","learning","is","be","great","Rust",
        // "rust","fast" = 9 unique strings (text + lemma combined).
        // "is" repeated, "be" repeated → 9 unique.
        assert!(stream.pool().len() <= 9);

        // Verify the duplicate lemma "be" gets the same ID.
        let e2 = &stream.tokens()[2]; // "is" / "be"
        let e5 = &stream.tokens()[5]; // "is" / "be"
        assert_eq!(e2.lemma_id, e5.lemma_id);
        assert_eq!(e2.text_id, e5.text_id);
    }

    #[test]
    fn test_sentence_offsets() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);

        // Sentence 0: tokens 0..4
        assert_eq!(stream.sentence_token_range(0), Some(0..4));
        // Sentence 1: tokens 4..7
        assert_eq!(stream.sentence_token_range(1), Some(4..7));
        // Out of range
        assert_eq!(stream.sentence_token_range(2), None);

        // Raw offsets: [0, 4, 7]
        assert_eq!(stream.sentence_offsets(), &[0, 4, 7]);
    }

    #[test]
    fn test_text_and_lemma_resolution() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);

        let e0 = &stream.tokens()[0];
        assert_eq!(stream.text(e0), "Machine");
        assert_eq!(stream.lemma(e0), "machine");

        let e4 = &stream.tokens()[4];
        assert_eq!(stream.text(e4), "Rust");
        assert_eq!(stream.lemma(e4), "rust");
    }

    #[test]
    fn test_token_entry_fields() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);

        let e3 = &stream.tokens()[3]; // "great"
        assert_eq!(e3.pos, PosTag::Adjective);
        assert_eq!(e3.start, 20);
        assert_eq!(e3.end, 25);
        assert_eq!(e3.sentence_idx, 0);
        assert_eq!(e3.token_idx, 3);
        assert!(!e3.is_stopword);
    }

    #[test]
    fn test_graph_key_with_pos() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);

        let e0 = &stream.tokens()[0]; // "machine" NOUN
        assert_eq!(e0.graph_key(stream.pool(), true), "machine|NOUN");
        assert_eq!(e0.graph_key(stream.pool(), false), "machine");

        let e4 = &stream.tokens()[4]; // "rust" PROPN
        assert_eq!(e4.graph_key(stream.pool(), true), "rust|PROPN");
    }

    #[test]
    fn test_is_graph_candidate() {
        let mut tokens = sample_tokens();
        tokens[2].is_stopword = true; // "is" — verb + stopword

        let stream = TokenStream::from_tokens(&tokens);

        // Noun, not stopword → candidate
        assert!(stream.tokens()[0].is_graph_candidate());
        // Verb + stopword → not candidate
        assert!(!stream.tokens()[2].is_graph_candidate());
        // Adjective, not stopword → candidate
        assert!(stream.tokens()[3].is_graph_candidate());
    }

    #[test]
    fn test_ref_mirrors_owned() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let r = stream.as_ref();

        assert_eq!(r.len(), stream.len());
        assert_eq!(r.num_sentences(), stream.num_sentences());
        assert_eq!(r.sentence_offsets(), stream.sentence_offsets());
        assert_eq!(r.tokens().len(), stream.tokens().len());

        // Resolve strings through the ref.
        let e0 = &r.tokens()[0];
        assert_eq!(r.text(e0), "Machine");
        assert_eq!(r.lemma(e0), "machine");
        assert_eq!(r.sentence_token_range(1), Some(4..7));
    }

    #[test]
    fn test_single_sentence() {
        let tokens = vec![
            Token::new("hello", "hello", PosTag::Noun, 0, 5, 0, 0),
            Token::new("world", "world", PosTag::Noun, 6, 11, 0, 1),
        ];
        let stream = TokenStream::from_tokens(&tokens);

        assert_eq!(stream.num_sentences(), 1);
        assert_eq!(stream.sentence_token_range(0), Some(0..2));
        assert_eq!(stream.sentence_offsets(), &[0, 2]);
    }

    #[test]
    fn test_three_sentences() {
        let tokens = vec![
            Token::new("a", "a", PosTag::Noun, 0, 1, 0, 0),
            Token::new("b", "b", PosTag::Noun, 2, 3, 1, 1),
            Token::new("c", "c", PosTag::Noun, 4, 5, 1, 2),
            Token::new("d", "d", PosTag::Noun, 6, 7, 2, 3),
        ];
        let stream = TokenStream::from_tokens(&tokens);

        assert_eq!(stream.num_sentences(), 3);
        assert_eq!(stream.sentence_token_range(0), Some(0..1));
        assert_eq!(stream.sentence_token_range(1), Some(1..3));
        assert_eq!(stream.sentence_token_range(2), Some(3..4));
    }

    #[test]
    fn test_ref_is_copy() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let r1 = stream.as_ref();
        let r2 = r1; // Copy
        assert_eq!(r1.len(), r2.len()); // both usable
    }
}
