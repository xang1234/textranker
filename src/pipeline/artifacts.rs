//! First-class pipeline artifacts.
//!
//! Each type represents a typed intermediate result flowing between pipeline
//! stages. Artifacts use interned IDs internally; string materialization is
//! deferred to the formatting boundary ([`FormattedResult`]).
//!
//! **Owned vs Borrowed**: Hot-path stage interfaces accept `*Ref<'a>` borrows;
//! the pipeline retains ownership of the corresponding owned artifacts.

use crate::types::{PosTag, StringPool, Token};
use serde::{Deserialize, Serialize};

use super::errors::PipelineRuntimeError;

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
    /// Construct directly from entries and a pool (for tests and stage impls).
    pub fn new(tokens: Vec<TokenEntry>, pool: StringPool) -> Self {
        Self {
            pool,
            tokens,
            sentence_offsets: Vec::new(),
        }
    }

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

        // Tokens are in document order, so the last token has the highest
        // sentence index — O(1) instead of a full scan.
        let num_sentences = tokens.last().map_or(0, |t| t.sentence_idx + 1);
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

    /// Mutable access to all token entries.
    ///
    /// Used by [`Preprocessor`](super::traits::Preprocessor) stages to modify
    /// token metadata in place (e.g., normalizing POS tags, toggling stopword
    /// flags).
    #[inline]
    pub fn tokens_mut(&mut self) -> &mut [TokenEntry] {
        &mut self.tokens
    }

    /// Mutable access to the string pool.
    ///
    /// Allows [`Preprocessor`](super::traits::Preprocessor) stages to intern
    /// new strings (e.g., re-lemmatizing tokens with a different strategy).
    #[inline]
    pub fn pool_mut(&mut self) -> &mut StringPool {
        &mut self.pool
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

    /// Convert back to the legacy `Vec<Token>` representation.
    ///
    /// This is the bridge for pipeline stages that delegate to existing code
    /// paths operating on `&[Token]`.  The conversion materializes all interned
    /// strings, so it should only be used in adapter implementations — native
    /// pipeline stages should work directly with [`TokenEntry`].
    pub fn to_legacy_tokens(&self) -> Vec<Token> {
        self.tokens
            .iter()
            .map(|e| {
                let text = self.pool.get(e.text_id).unwrap_or("").to_string();
                let lemma = self.pool.get(e.lemma_id).unwrap_or("").to_string();
                let mut t = Token::new(
                    text,
                    lemma,
                    e.pos,
                    e.start as usize,
                    e.end as usize,
                    e.sentence_idx as usize,
                    e.token_idx as usize,
                );
                t.is_stopword = e.is_stopword;
                t
            })
            .collect()
    }
}

// ============================================================================
// CandidateSet — unified word-level and phrase-level candidates
// ============================================================================

/// A single word-level candidate node (TextRank / PositionRank / BiasedTextRank
/// / SingleRank / TopicalPageRank families).
///
/// One entry per unique graph key (`lemma` or `lemma|POS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordCandidate {
    /// Interned lemma ID in the parent [`TokenStream`]'s pool.
    pub lemma_id: u32,
    /// Part-of-speech tag.
    pub pos: PosTag,
    /// Token index of the first occurrence in the document.
    pub first_position: u32,
}

impl WordCandidate {
    /// Build the graph-node key, mirroring [`TokenEntry::graph_key`].
    #[inline]
    pub fn graph_key(&self, pool: &StringPool, use_pos_in_nodes: bool) -> String {
        let lemma = pool.get(self.lemma_id).unwrap_or("");
        if use_pos_in_nodes {
            format!("{}|{}", lemma, self.pos.as_str())
        } else {
            lemma.to_owned()
        }
    }
}

/// A single phrase-level candidate (TopicRank / MultipartiteRank families).
///
/// Represents a noun chunk with its token span, surface forms, and the term
/// set used for Jaccard-based clustering.
#[derive(Debug, Clone)]
pub struct PhraseCandidate {
    /// Start token index (inclusive) in the parent token stream.
    pub start_token: u32,
    /// End token index (exclusive).
    pub end_token: u32,
    /// Character byte-offset of the first byte.
    pub start_char: u32,
    /// Character byte-offset one past the last byte.
    pub end_char: u32,
    /// Sentence this phrase belongs to.
    pub sentence_idx: u32,
    /// Interned IDs for individual lemma tokens within the phrase.
    ///
    /// Used for Jaccard similarity during clustering.  Ordered to match the
    /// token sequence; equality/hashing uses the set of unique IDs.
    pub lemma_ids: Vec<u32>,
    /// Interned IDs for non-stopword surface forms (the "term set").
    ///
    /// This mirrors the `FxHashSet<String>` in the legacy `PhraseCandidate`
    /// but uses interned IDs for cheaper set operations.
    pub term_ids: Vec<u32>,
}

impl PhraseCandidate {
    /// Token span length.
    #[inline]
    pub fn token_len(&self) -> u32 {
        self.end_token - self.start_token
    }
}

/// A single sentence-level candidate (SentenceRank family).
///
/// Each sentence in the document becomes a candidate node for sentence-level
/// TextRank (extractive summarization).
#[derive(Debug, Clone)]
pub struct SentenceCandidate {
    /// 0-based sentence index in the parent token stream.
    pub sentence_idx: u32,
    /// Start token index (inclusive).
    pub start_token: u32,
    /// End token index (exclusive).
    pub end_token: u32,
    /// Character byte-offset of the first byte.
    pub start_char: u32,
    /// Character byte-offset one past the last byte.
    pub end_char: u32,
    /// Non-stopword lemma IDs within this sentence, preserving duplicates for TF.
    pub lemma_ids: Vec<u32>,
}

impl SentenceCandidate {
    /// Token span length.
    #[inline]
    pub fn token_len(&self) -> u32 {
        self.end_token - self.start_token
    }
}

/// Distinguishes the three candidate families.
///
/// Downstream stages (GraphBuilder, TeleportBuilder, etc.) can match on this
/// to select the appropriate processing strategy.
#[derive(Debug, Clone)]
pub enum CandidateKind {
    /// Word-level candidates (one per unique graph key).
    Words(Vec<WordCandidate>),
    /// Phrase-level candidates (one per noun chunk).
    Phrases(Vec<PhraseCandidate>),
    /// Sentence-level candidates (one per sentence, for extractive summarization).
    Sentences(Vec<SentenceCandidate>),
}

/// Set of candidate nodes (word-level or phrase-level) selected for graph
/// construction.
///
/// Wraps a [`CandidateKind`] enum so downstream stages can work with
/// candidate indices uniformly while dispatching on the variant family.
///
/// # Construction
///
/// Use [`CandidateSet::from_word_tokens`] to build from a token stream
/// (word-level), or [`CandidateSet::from_phrase_chunks`] for phrase-level.
#[derive(Debug, Clone)]
pub struct CandidateSet {
    kind: CandidateKind,
}

impl CandidateSet {
    /// Empty candidate set (word-level, zero candidates).
    pub fn empty() -> Self {
        Self {
            kind: CandidateKind::Words(Vec::new()),
        }
    }

    /// Construct from a pre-built [`CandidateKind`].
    ///
    /// This is the low-level constructor used by [`CandidateSelector`]
    /// implementations that build candidates from [`TokenStreamRef`].
    pub fn from_kind(kind: CandidateKind) -> Self {
        Self { kind }
    }

    /// Build a word-level candidate set from a token stream.
    ///
    /// Filters tokens by `include_pos` (or default content-word check when
    /// empty) and stopword flag, deduplicates by graph key, and records the
    /// first occurrence position for each unique key.
    pub fn from_word_tokens(
        stream: &TokenStream,
        include_pos: &[PosTag],
        use_pos_in_nodes: bool,
    ) -> Self {
        use rustc_hash::FxHashMap;

        // Key: (lemma_id, optional POS discriminant) → index into `words`.
        let mut seen: FxHashMap<(u32, Option<PosTag>), usize> = FxHashMap::default();
        let mut words = Vec::new();

        for entry in stream.tokens() {
            if entry.is_stopword {
                continue;
            }
            let pass = if include_pos.is_empty() {
                entry.pos.is_content_word()
            } else {
                include_pos.contains(&entry.pos)
            };
            if !pass {
                continue;
            }

            let key = if use_pos_in_nodes {
                (entry.lemma_id, Some(entry.pos))
            } else {
                (entry.lemma_id, None)
            };

            if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
                e.insert(words.len());
                words.push(WordCandidate {
                    lemma_id: entry.lemma_id,
                    pos: entry.pos,
                    first_position: entry.token_idx,
                });
            }
        }

        Self {
            kind: CandidateKind::Words(words),
        }
    }

    /// Build a phrase-level candidate set from pre-computed chunk spans.
    ///
    /// Each chunk is turned into a [`PhraseCandidate`] whose `lemma_ids` and
    /// `term_ids` are resolved against the token stream's interning pool.
    pub fn from_phrase_chunks(stream: &TokenStream, chunks: &[crate::types::ChunkSpan]) -> Self {
        let mut phrases = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let start = chunk.start_token;
            let end = chunk.end_token;

            let mut lemma_ids = Vec::with_capacity(end - start);
            let mut term_ids = Vec::new();

            for &entry in &stream.tokens()[start..end] {
                lemma_ids.push(entry.lemma_id);
                if !entry.is_stopword {
                    // Use text_id for term set (matches legacy `t.text.clone()`)
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

        Self {
            kind: CandidateKind::Phrases(phrases),
        }
    }

    /// The candidate variant (word-level or phrase-level).
    #[inline]
    pub fn kind(&self) -> &CandidateKind {
        &self.kind
    }

    /// Number of candidates.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.kind {
            CandidateKind::Words(w) => w.len(),
            CandidateKind::Phrases(p) => p.len(),
            CandidateKind::Sentences(s) => s.len(),
        }
    }

    /// Whether there are no candidates.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow as a [`CandidateSetRef`].
    #[inline]
    pub fn as_ref(&self) -> CandidateSetRef<'_> {
        CandidateSetRef { kind: &self.kind }
    }

    /// Access word candidates (panics on wrong variant).
    #[inline]
    pub fn words(&self) -> &[WordCandidate] {
        match &self.kind {
            CandidateKind::Words(w) => w,
            _ => panic!("called words() on non-word CandidateSet"),
        }
    }

    /// Access phrase candidates (panics on wrong variant).
    #[inline]
    pub fn phrases(&self) -> &[PhraseCandidate] {
        match &self.kind {
            CandidateKind::Phrases(p) => p,
            _ => panic!("called phrases() on non-phrase CandidateSet"),
        }
    }

    /// Access sentence candidates (panics on wrong variant).
    #[inline]
    pub fn sentences(&self) -> &[SentenceCandidate] {
        match &self.kind {
            CandidateKind::Sentences(s) => s,
            _ => panic!("called sentences() on non-sentence CandidateSet"),
        }
    }

    /// Build a sentence-level candidate set from a token stream's sentence boundaries.
    ///
    /// Each sentence becomes a [`SentenceCandidate`] whose `lemma_ids` are the
    /// non-stopword lemma IDs within the sentence (preserving duplicates for TF).
    pub fn from_sentence_boundaries(stream: &TokenStream) -> Self {
        let num_sentences = stream.num_sentences();
        let mut sentences = Vec::with_capacity(num_sentences);

        for idx in 0..num_sentences {
            let range = match stream.sentence_token_range(idx) {
                Some(r) => r,
                None => continue,
            };
            let tokens_slice = &stream.tokens()[range.clone()];
            if tokens_slice.is_empty() {
                continue;
            }

            let start_char = tokens_slice.first().map_or(0, |t| t.start);
            let end_char = tokens_slice.last().map_or(0, |t| t.end);

            let lemma_ids: Vec<u32> = tokens_slice
                .iter()
                .filter(|t| !t.is_stopword)
                .map(|t| t.lemma_id)
                .collect();

            sentences.push(SentenceCandidate {
                sentence_idx: idx as u32,
                start_token: range.start as u32,
                end_token: range.end as u32,
                start_char,
                end_char,
                lemma_ids,
            });
        }

        Self {
            kind: CandidateKind::Sentences(sentences),
        }
    }
}

/// Borrowed view into a [`CandidateSet`].
///
/// Stages accept this to avoid requiring ownership of the candidate data.
#[derive(Debug, Clone, Copy)]
pub struct CandidateSetRef<'a> {
    kind: &'a CandidateKind,
}

impl<'a> CandidateSetRef<'a> {
    /// The candidate variant.
    #[inline]
    pub fn kind(&self) -> &'a CandidateKind {
        self.kind
    }

    /// Number of candidates.
    #[inline]
    pub fn len(&self) -> usize {
        match self.kind {
            CandidateKind::Words(w) => w.len(),
            CandidateKind::Phrases(p) => p.len(),
            CandidateKind::Sentences(s) => s.len(),
        }
    }

    /// Whether there are no candidates.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Access word candidates.
    #[inline]
    pub fn words(&self) -> &'a [WordCandidate] {
        match self.kind {
            CandidateKind::Words(w) => w,
            _ => panic!("called words() on non-word CandidateSetRef"),
        }
    }

    /// Access phrase candidates.
    #[inline]
    pub fn phrases(&self) -> &'a [PhraseCandidate] {
        match self.kind {
            CandidateKind::Phrases(p) => p,
            _ => panic!("called phrases() on non-phrase CandidateSetRef"),
        }
    }

    /// Access sentence candidates.
    #[inline]
    pub fn sentences(&self) -> &'a [SentenceCandidate] {
        match self.kind {
            CandidateKind::Sentences(s) => s,
            _ => panic!("called sentences() on non-sentence CandidateSetRef"),
        }
    }
}

// ============================================================================
// ClusterAssignments — candidate → cluster mapping (topic families)
// ============================================================================

/// Maps each phrase candidate to a cluster (topic) ID.
///
/// Used by the TopicRank and MultipartiteRank families to group similar
/// phrase candidates into topic clusters.  Downstream stages
/// ([`GraphTransform`](crate::pipeline::traits::GraphTransform) for
/// intra-cluster edge removal, alpha-boost weighting, and
/// [`PhraseBuilder`](crate::pipeline::traits::PhraseBuilder) for
/// topic-representative selection) consume this artifact.
///
/// # Layout
///
/// Internally a flat `Vec<u32>` indexed by candidate position in the
/// parent [`CandidateSet`] (phrase family).  The value at index `i` is
/// the 0-based cluster ID for candidate `i`.
///
/// The number of clusters is tracked separately so consumers can iterate
/// over cluster IDs (`0..num_clusters`) without scanning the vector.
///
/// # Empty clusters
///
/// An empty `ClusterAssignments` (no candidates) has `num_clusters == 0`
/// and an empty assignment vector.  This is the result of the
/// [`NoopClusterer`](crate::pipeline::traits::NoopClusterer).
#[derive(Debug, Clone)]
pub struct ClusterAssignments {
    /// `assignments[i]` = cluster ID for candidate `i`.
    assignments: Vec<u32>,
    /// Total number of distinct clusters.
    num_clusters: u32,
}

impl ClusterAssignments {
    /// Create an empty assignment (zero candidates, zero clusters).
    pub fn empty() -> Self {
        Self {
            assignments: Vec::new(),
            num_clusters: 0,
        }
    }

    /// Build from the legacy `Vec<Vec<usize>>` cluster representation
    /// where `clusters[c]` contains the candidate indices in cluster `c`.
    pub fn from_cluster_vecs(clusters: &[Vec<usize>], num_candidates: usize) -> Self {
        let mut assignments = vec![0u32; num_candidates];
        for (cluster_id, members) in clusters.iter().enumerate() {
            for &candidate_idx in members {
                if candidate_idx < num_candidates {
                    assignments[candidate_idx] = cluster_id as u32;
                }
            }
        }
        Self {
            assignments,
            num_clusters: clusters.len() as u32,
        }
    }

    /// Cluster ID for candidate at `index`.
    #[inline]
    pub fn cluster_of(&self, index: usize) -> u32 {
        self.assignments[index]
    }

    /// Number of distinct clusters.
    #[inline]
    pub fn num_clusters(&self) -> u32 {
        self.num_clusters
    }

    /// Number of candidates (length of the assignment vector).
    #[inline]
    pub fn num_candidates(&self) -> usize {
        self.assignments.len()
    }

    /// Whether there are no candidates assigned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Raw slice of cluster assignments indexed by candidate position.
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.assignments
    }

    /// Collect the candidate indices belonging to a given cluster.
    pub fn members_of(&self, cluster_id: u32) -> Vec<usize> {
        self.assignments
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c == cluster_id)
            .map(|(i, _)| i)
            .collect()
    }
}

// ============================================================================
// Graph — pipeline artifact wrapping the CSR representation
// ============================================================================

/// Pipeline-level graph artifact wrapping the existing CSR-backed adjacency
/// + weights.
///
/// This is a thin wrapper around [`CsrGraph`] that turns the graph into a
/// first-class pipeline artifact.  The inner CSR storage is fully accessible
/// for hot-path PageRank iteration, and the wrapper provides attachment
/// points for metadata (transform history, observer hooks, etc.).
///
/// # Construction
///
/// Use [`Graph::from_builder`] to convert from an existing [`GraphBuilder`],
/// or [`Graph::from_csr`] if you already have a [`CsrGraph`].
#[derive(Debug, Clone)]
pub struct Graph {
    /// The underlying CSR graph.
    csr: crate::graph::csr::CsrGraph,
    /// Whether this graph has been modified by any [`GraphTransform`] stage.
    ///
    /// Set to `true` after transforms such as intra-cluster edge removal
    /// (MultipartiteRank) or alpha-boost weighting.  Observers can use this
    /// to log whether the graph they see is the original build or a
    /// transformed variant.
    transformed: bool,
    /// Optional cluster assignments embedded during topic-graph construction.
    ///
    /// Present only when the graph was built by [`TopicGraphBuilder`] (or
    /// another topic-family graph builder).  The downstream
    /// [`TopicRepresentativeBuilder`] reads this to select one representative
    /// phrase per cluster.
    ///
    /// `None` for all non-topic pipelines — zero overhead.
    cluster_assignments: Option<ClusterAssignments>,
}

impl Graph {
    /// Empty graph (zero nodes, zero edges).
    pub fn empty() -> Self {
        Self {
            csr: crate::graph::csr::CsrGraph {
                num_nodes: 0,
                row_ptr: vec![0],
                col_idx: Vec::new(),
                weights: Vec::new(),
                out_degree: Vec::new(),
                total_weight: Vec::new(),
                lemmas: Vec::new(),
                lemma_to_id: Default::default(),
            },
            transformed: false,
            cluster_assignments: None,
        }
    }

    /// Build from an existing [`GraphBuilder`].
    ///
    /// This is the primary construction path — the existing graph builder
    /// infrastructure produces a `GraphBuilder`, and this method converts it
    /// to the pipeline artifact via [`CsrGraph::from_builder`].
    pub fn from_builder(builder: &crate::graph::builder::GraphBuilder) -> Self {
        Self {
            csr: crate::graph::csr::CsrGraph::from_builder(builder),
            transformed: false,
            cluster_assignments: None,
        }
    }

    /// Wrap a pre-existing [`CsrGraph`].
    pub fn from_csr(csr: crate::graph::csr::CsrGraph) -> Self {
        Self {
            csr,
            transformed: false,
            cluster_assignments: None,
        }
    }

    /// Access the inner CSR graph (immutable).
    ///
    /// Use this for hot-path operations like PageRank iteration where you
    /// need direct access to the CSR arrays.
    #[inline]
    pub fn csr(&self) -> &crate::graph::csr::CsrGraph {
        &self.csr
    }

    /// Mutable access to the inner CSR graph.
    ///
    /// Used by [`GraphTransform`] stages that modify edge weights in-place
    /// (e.g., intra-cluster edge removal, alpha-boost weighting).
    /// Automatically sets the `transformed` flag.
    #[inline]
    pub fn csr_mut(&mut self) -> &mut crate::graph::csr::CsrGraph {
        self.transformed = true;
        &mut self.csr
    }

    /// Consume this wrapper and return the inner [`CsrGraph`].
    #[inline]
    pub fn into_csr(self) -> crate::graph::csr::CsrGraph {
        self.csr
    }

    /// Whether any transform stage has modified this graph.
    #[inline]
    pub fn is_transformed(&self) -> bool {
        self.transformed
    }

    /// Mark this graph as transformed (e.g., after an external modification).
    #[inline]
    pub fn set_transformed(&mut self) {
        self.transformed = true;
    }

    /// Access the embedded cluster assignments (if any).
    ///
    /// Returns `Some` only when this graph was built by a topic-family
    /// graph builder (e.g., [`TopicGraphBuilder`]).
    #[inline]
    pub fn cluster_assignments(&self) -> Option<&ClusterAssignments> {
        self.cluster_assignments.as_ref()
    }

    /// Attach cluster assignments to this graph.
    ///
    /// Called by topic-family graph builders after constructing the cluster
    /// graph so downstream stages (e.g., [`TopicRepresentativeBuilder`])
    /// can access them.
    #[inline]
    pub fn set_cluster_assignments(&mut self, assignments: ClusterAssignments) {
        self.cluster_assignments = Some(assignments);
    }

    // ------------------------------------------------------------------
    // Delegated convenience methods
    // ------------------------------------------------------------------

    /// Number of nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.csr.num_nodes
    }

    /// Number of directed edge entries in the CSR arrays.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.csr.num_edges()
    }

    /// Whether the graph has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.csr.is_empty()
    }

    /// Look up a node by its graph key (lemma or `lemma|POS`).
    #[inline]
    pub fn get_node_by_lemma(&self, lemma: &str) -> Option<u32> {
        self.csr.get_node_by_lemma(lemma)
    }

    /// Iterate over (neighbor_id, weight) pairs for a node.
    #[inline]
    pub fn neighbors(&self, node: u32) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.csr.neighbors(node)
    }

    /// Get the lemma / graph key for a node.
    #[inline]
    pub fn lemma(&self, node: u32) -> &str {
        self.csr.lemma(node)
    }

    /// Dangling nodes (no outgoing edges).
    #[inline]
    pub fn dangling_nodes(&self) -> Vec<u32> {
        self.csr.dangling_nodes()
    }
}

// ============================================================================
// TeleportVector — personalization distribution for PageRank
// ============================================================================

/// The kind of teleport (personalization) strategy that produced this vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeleportType {
    /// Uniform distribution — equivalent to standard (non-personalized) PageRank.
    Uniform,
    /// Position-weighted — earlier tokens get higher teleport probability (PositionRank).
    Position,
    /// Focus-terms — specified terms get boosted teleport probability (BiasedTextRank).
    Focus,
    /// Topic-weighted — per-lemma weights from an external topic model (TopicalPageRank).
    Topic,
}

impl std::fmt::Display for TeleportType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform => write!(f, "uniform"),
            Self::Position => write!(f, "position"),
            Self::Focus => write!(f, "focus"),
            Self::Topic => write!(f, "topic"),
        }
    }
}

/// A personalization (teleport) vector for PageRank.
///
/// Each entry `v[i]` is the teleport probability for CSR node `i`.  When used
/// with Personalized PageRank, the random surfer jumps to node `i` with
/// probability proportional to `v[i]` instead of uniformly.
///
/// # Normalization
///
/// The [`new`](TeleportVector::new) constructor **enforces** normalization —
/// it panics if the values do not sum to approximately 1.0.  Use
/// [`zeros`](TeleportVector::zeros) for sparse construction followed by
/// [`normalize`](TeleportVector::normalize).
///
/// # Construction
///
/// - [`TeleportVector::new`] — from a pre-normalized `Vec<f64>` (validates)
/// - [`TeleportVector::uniform`] — uniform distribution of given length
/// - [`TeleportVector::zeros`] — zero vector (for sparse construction)
///
/// # Metadata
///
/// Each vector carries a [`TeleportType`] describing the strategy that
/// produced it, and an optional `debug_source` string for diagnostic output.
#[derive(Debug, Clone, PartialEq)]
pub struct TeleportVector {
    values: Vec<f64>,
    teleport_type: TeleportType,
    debug_source: Option<String>,
}

impl TeleportVector {
    /// Normalization tolerance used by [`new`](Self::new).
    const NORM_EPSILON: f64 = 1e-6;

    /// Create from a pre-normalized probability vector.
    ///
    /// # Panics
    ///
    /// Panics if `values` is non-empty and does not sum to approximately 1.0
    /// (tolerance: 1e-6).  Use [`zeros`](Self::zeros) for sparse construction
    /// followed by [`normalize`](Self::normalize).
    pub fn new(values: Vec<f64>, teleport_type: TeleportType) -> Self {
        if !values.is_empty() {
            let sum: f64 = values.iter().sum();
            assert!(
                (sum - 1.0).abs() < Self::NORM_EPSILON,
                "TeleportVector::new requires a normalized vector (sum ≈ 1.0), got sum = {sum}"
            );
        }
        Self {
            values,
            teleport_type,
            debug_source: None,
        }
    }

    /// Create a uniform distribution of length `n`.
    ///
    /// Each entry is `1.0 / n`.  Returns an empty vector when `n == 0`.
    /// The [`teleport_type`](Self::teleport_type) is [`TeleportType::Uniform`].
    pub fn uniform(n: usize) -> Self {
        if n == 0 {
            return Self {
                values: Vec::new(),
                teleport_type: TeleportType::Uniform,
                debug_source: None,
            };
        }
        Self {
            values: vec![1.0 / n as f64; n],
            teleport_type: TeleportType::Uniform,
            debug_source: None,
        }
    }

    /// Create a zero vector of length `n` for sparse construction.
    ///
    /// The caller should set individual entries and then call [`normalize`].
    pub fn zeros(n: usize, teleport_type: TeleportType) -> Self {
        Self {
            values: vec![0.0; n],
            teleport_type,
            debug_source: None,
        }
    }

    /// Number of entries (should equal the number of graph nodes).
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrow the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Mutable access to the underlying slice.
    ///
    /// Useful for sparse construction: create with [`zeros`], set individual
    /// entries, then [`normalize`].
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Get the value at index `i`, or `0.0` if out of bounds.
    #[inline]
    pub fn get(&self, i: usize) -> f64 {
        self.values.get(i).copied().unwrap_or(0.0)
    }

    /// Set the value at index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()`.
    #[inline]
    pub fn set(&mut self, i: usize, value: f64) {
        self.values[i] = value;
    }

    /// Check whether the vector sums to approximately 1.0.
    pub fn is_normalized(&self, epsilon: f64) -> bool {
        if self.values.is_empty() {
            return true;
        }
        let sum: f64 = self.values.iter().sum();
        (sum - 1.0).abs() < epsilon
    }

    /// Normalize in place so entries sum to 1.0.
    ///
    /// If the sum is zero (or the vector is empty), falls back to a uniform
    /// distribution.
    pub fn normalize(&mut self) {
        if self.values.is_empty() {
            return;
        }
        let sum: f64 = self.values.iter().sum();
        if sum > 0.0 {
            for v in &mut self.values {
                *v /= sum;
            }
        } else {
            let uniform = 1.0 / self.values.len() as f64;
            for v in &mut self.values {
                *v = uniform;
            }
        }
    }

    /// The teleport strategy that produced this vector.
    #[inline]
    pub fn teleport_type(&self) -> TeleportType {
        self.teleport_type
    }

    /// Optional description of how this vector was built (for debug output).
    #[inline]
    pub fn debug_source(&self) -> Option<&str> {
        self.debug_source.as_deref()
    }

    /// Attach a debug-source description (builder pattern).
    #[inline]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.debug_source = Some(source.into());
        self
    }

    /// Consume and return the inner `Vec<f64>`.
    #[inline]
    pub fn into_inner(self) -> Vec<f64> {
        self.values
    }

    /// Iterate over values.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.values.iter()
    }
}

impl std::ops::Index<usize> for TeleportVector {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &f64 {
        &self.values[index]
    }
}

impl std::ops::IndexMut<usize> for TeleportVector {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.values[index]
    }
}

// ============================================================================
// RankOutput — PageRank scores + convergence metadata
// ============================================================================

/// Optional diagnostics captured during PageRank iteration.
///
/// These are only populated when debug/expose mode is enabled, so the hot
/// path incurs zero allocation overhead.
#[derive(Debug, Clone, Default)]
pub struct RankDiagnostics {
    /// Per-iteration residual (L1 norm of score delta).
    ///
    /// `residuals[i]` is the residual after iteration `i`.  Empty when
    /// diagnostics are not requested.
    pub residuals: Vec<f64>,
}

/// PageRank output: per-node scores, convergence info, and optional
/// diagnostics.
///
/// Scores are stored as a `Vec<f64>` indexed by node ID for cache-friendly
/// access — the same layout used by the existing [`PageRankResult`].
///
/// # Construction
///
/// Use [`RankOutput::from_pagerank_result`] to bridge from the existing
/// `PageRankResult` type.
#[derive(Debug, Clone)]
pub struct RankOutput {
    /// Per-node scores indexed by CSR node ID.
    scores: Vec<f64>,
    /// Whether the power iteration converged within the threshold.
    converged: bool,
    /// Number of iterations actually performed.
    iterations: u32,
    /// Final L1-norm convergence delta between the last two iterations.
    final_delta: f64,
    /// Optional debug diagnostics (empty by default).
    diagnostics: Option<RankDiagnostics>,
}

impl RankOutput {
    /// Construct from the existing [`PageRankResult`].
    pub fn from_pagerank_result(pr: &crate::pagerank::PageRankResult) -> Self {
        Self {
            scores: pr.scores.clone(),
            converged: pr.converged,
            iterations: pr.iterations as u32,
            final_delta: pr.delta,
            diagnostics: None,
        }
    }

    /// Build directly (for tests or future stage implementations).
    pub fn new(scores: Vec<f64>, converged: bool, iterations: u32, final_delta: f64) -> Self {
        Self {
            scores,
            converged,
            iterations,
            final_delta,
            diagnostics: None,
        }
    }

    /// Attach diagnostics (call after construction when debug is enabled).
    pub fn with_diagnostics(mut self, diag: RankDiagnostics) -> Self {
        self.diagnostics = Some(diag);
        self
    }

    /// Per-node score slice, indexed by CSR node ID.
    #[inline]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Score for a specific node.
    #[inline]
    pub fn score(&self, node_id: u32) -> f64 {
        self.scores.get(node_id as usize).copied().unwrap_or(0.0)
    }

    /// Whether PageRank converged.
    #[inline]
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Iteration count.
    #[inline]
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Final convergence delta.
    #[inline]
    pub fn final_delta(&self) -> f64 {
        self.final_delta
    }

    /// Access diagnostics (if attached).
    #[inline]
    pub fn diagnostics(&self) -> Option<&RankDiagnostics> {
        self.diagnostics.as_ref()
    }

    /// Number of nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.scores.len()
    }

    /// Convert to the legacy [`PageRankResult`] type.
    ///
    /// This is the bridge for pipeline stages that delegate to existing code
    /// paths.  Clones the score vector.
    pub fn to_pagerank_result(&self) -> crate::pagerank::PageRankResult {
        crate::pagerank::PageRankResult {
            scores: self.scores.clone(),
            iterations: self.iterations as usize,
            delta: self.final_delta,
            converged: self.converged,
        }
    }

    /// Consuming conversion to the legacy [`PageRankResult`] type.
    ///
    /// Moves the score vector instead of cloning, avoiding allocation.
    pub fn into_pagerank_result(self) -> crate::pagerank::PageRankResult {
        crate::pagerank::PageRankResult {
            scores: self.scores,
            iterations: self.iterations as usize,
            delta: self.final_delta,
            converged: self.converged,
        }
    }
}

// ============================================================================
// PhraseSet — pre-format phrase collection
// ============================================================================

/// A single scored phrase entry in a [`PhraseSet`].
///
/// Uses interned IDs where possible; surface forms are optional and lazily
/// materialized only when needed for formatting.
#[derive(Debug, Clone)]
pub struct PhraseEntry {
    /// Interned lemma IDs for the tokens in this phrase (ordered).
    pub lemma_ids: Vec<u32>,
    /// Aggregated score from PageRank.
    pub score: f64,
    /// Number of occurrences in the document.
    pub count: u32,
    /// Optional canonical surface form (materialized lazily).
    pub surface: Option<String>,
    /// Optional lemma string (materialized lazily for grouping key).
    pub lemma_text: Option<String>,
    /// Optional token-span pairs for each occurrence (debug only).
    pub spans: Option<Vec<(u32, u32)>>,
}

/// Pre-format phrase collection: scored phrases with interned lemma IDs.
///
/// This is the last internal artifact before [`FormattedResult`] produces the
/// public output.  Surface forms are lazily materialized — the hot path
/// operates on interned IDs only.
///
/// # Construction
///
/// Use [`PhraseSet::from_phrases`] to bridge from the existing `Vec<Phrase>`.
#[derive(Debug, Clone)]
pub struct PhraseSet {
    entries: Vec<PhraseEntry>,
}

impl PhraseSet {
    /// Empty phrase set (zero phrases).
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Construct from the existing `Vec<Phrase>` type.
    ///
    /// Interns lemma tokens via the provided pool and eagerly stores the
    /// surface/lemma strings (since they're already materialized in the
    /// legacy path).
    pub fn from_phrases(phrases: &[crate::types::Phrase], pool: &mut StringPool) -> Self {
        let entries = phrases
            .iter()
            .map(|p| {
                let lemma_ids: Vec<u32> =
                    p.lemma.split_whitespace().map(|w| pool.intern(w)).collect();
                let spans = if p.offsets.is_empty() {
                    None
                } else {
                    Some(
                        p.offsets
                            .iter()
                            .map(|&(s, e)| (s as u32, e as u32))
                            .collect(),
                    )
                };
                PhraseEntry {
                    lemma_ids,
                    score: p.score,
                    count: p.count as u32,
                    surface: Some(p.text.clone()),
                    lemma_text: Some(p.lemma.clone()),
                    spans,
                }
            })
            .collect();
        Self { entries }
    }

    /// Build from raw entries.
    pub fn from_entries(entries: Vec<PhraseEntry>) -> Self {
        Self { entries }
    }

    /// Access the phrase entries.
    #[inline]
    pub fn entries(&self) -> &[PhraseEntry] {
        &self.entries
    }

    /// Number of phrases.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no phrases.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow as a [`PhraseSetRef`].
    #[inline]
    pub fn as_ref(&self) -> PhraseSetRef<'_> {
        PhraseSetRef {
            entries: &self.entries,
        }
    }
}

/// Borrowed view into a [`PhraseSet`].
#[derive(Debug, Clone, Copy)]
pub struct PhraseSetRef<'a> {
    entries: &'a [PhraseEntry],
}

impl<'a> PhraseSetRef<'a> {
    /// Access the phrase entries.
    #[inline]
    pub fn entries(&self) -> &'a [PhraseEntry] {
        self.entries
    }

    /// Number of phrases.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no phrases.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================================
// FormattedResult — public-facing output (stability boundary)
// ============================================================================

/// Public-facing formatted output — the stability boundary.
///
/// Everything before this type is internal and may change; **this type is the
/// public contract** exposed to Python and JSON consumers.  It wraps the
/// existing `Vec<Phrase>` + convergence metadata, plus an optional debug
/// payload for power users.
///
/// # Construction
///
/// Use [`FormattedResult::from_extraction`] to bridge from the existing
/// `ExtractionResult`.
#[derive(Debug, Clone)]
pub struct FormattedResult {
    /// The ranked phrases (public output).
    pub phrases: Vec<crate::types::Phrase>,
    /// Whether PageRank converged.
    pub converged: bool,
    /// Number of PageRank iterations.
    pub iterations: u32,
    /// Optional debug payload (opt-in via `expose` config).
    pub debug: Option<DebugPayload>,
    /// Optional structured error (e.g., graph limit exceeded).
    /// When set, `phrases` is empty and the caller should surface the error.
    pub error: Option<PipelineRuntimeError>,
}

// ============================================================================
// DebugLevel — tiered debug output control
// ============================================================================

/// Controls how much debug information is attached to a [`FormattedResult`].
///
/// Each level is a strict superset of the previous one:
///
/// | Level      | Includes                                                         |
/// |------------|------------------------------------------------------------------|
/// | `None`     | Nothing (default — zero overhead).                               |
/// | `Stats`    | Graph statistics + convergence summary (nodes, edges, iters).    |
/// | `TopNodes` | Stats + top-K node scores (bounded by `max_debug_top_k`).        |
/// | `Full`     | TopNodes + convergence residuals + bounded adjacency samples.    |
///
/// The ordering derives from variant declaration order, so
/// `DebugLevel::Stats >= DebugLevel::None` holds naturally via [`Ord`].
///
/// # Examples
///
/// ```
/// use rapid_textrank::pipeline::artifacts::DebugLevel;
///
/// let level = DebugLevel::TopNodes;
/// assert!(level.includes_stats());
/// assert!(level.includes_node_scores());
/// assert!(!level.includes_full());
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DebugLevel {
    /// No debug output (default). Zero overhead on the hot path.
    #[default]
    None,
    /// Graph statistics (node/edge counts) and convergence summary
    /// (iterations, converged flag, final delta).
    Stats,
    /// Everything in [`Stats`](DebugLevel::Stats), plus the top-K node
    /// scores (bounded by `max_debug_top_k` from [`RuntimeSpec`]).
    TopNodes,
    /// Everything in [`TopNodes`](DebugLevel::TopNodes), plus convergence
    /// residuals per iteration and bounded adjacency samples. Use with
    /// caution on large graphs — output is bounded but can still be
    /// substantial.
    Full,
}

impl DebugLevel {
    /// Returns `true` when any debug output is requested (level > `None`).
    #[inline]
    pub fn is_enabled(self) -> bool {
        self > DebugLevel::None
    }

    /// Returns `true` when graph stats and convergence summary are included.
    #[inline]
    pub fn includes_stats(self) -> bool {
        self >= DebugLevel::Stats
    }

    /// Returns `true` when top-K node scores are included.
    #[inline]
    pub fn includes_node_scores(self) -> bool {
        self >= DebugLevel::TopNodes
    }

    /// Returns `true` when full diagnostics (residuals, adjacency) are included.
    #[inline]
    pub fn includes_full(self) -> bool {
        self >= DebugLevel::Full
    }
}

impl DebugLevel {
    /// String representation matching the serde/JSON contract.
    pub fn as_str(self) -> &'static str {
        match self {
            DebugLevel::None => "none",
            DebugLevel::Stats => "stats",
            DebugLevel::TopNodes => "top_nodes",
            DebugLevel::Full => "full",
        }
    }

    /// Parse from a string (case-insensitive).
    pub fn parse_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Some(DebugLevel::None),
            "stats" => Some(DebugLevel::Stats),
            "top_nodes" | "topnodes" => Some(DebugLevel::TopNodes),
            "full" => Some(DebugLevel::Full),
            _ => None,
        }
    }

    /// Default top-K limit for node scores when no explicit limit is set.
    pub const DEFAULT_TOP_K: usize = 50;
}

/// Optional debug information attached to a [`FormattedResult`].
///
/// Power users can request this via the `expose` config key.  Fields are
/// individually optional so callers pay only for what they ask for.
///
/// # Population by [`DebugLevel`]
///
/// | Field                 | `Stats` | `TopNodes` | `Full` |
/// |-----------------------|:-------:|:----------:|:------:|
/// | `graph_stats`         |    ✓    |     ✓      |   ✓    |
/// | `convergence_summary` |    ✓    |     ✓      |   ✓    |
/// | `stage_timings`       |    ✓    |     ✓      |   ✓    |
/// | `node_scores`         |         |     ✓      |   ✓    |
/// | `residuals`           |         |            |   ✓    |
/// | `cluster_memberships` |         |            |   ✓    |
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugPayload {
    /// Top-K node scores (node lemma → score), sorted by score descending.
    pub node_scores: Option<Vec<(String, f64)>>,
    /// Graph statistics.
    pub graph_stats: Option<GraphStats>,
    /// Per-stage timing in milliseconds.
    pub stage_timings: Option<Vec<(String, f64)>>,
    /// PageRank convergence residuals per iteration.
    pub residuals: Option<Vec<f64>>,
    /// Convergence summary (iterations, converged, final delta).
    pub convergence_summary: Option<ConvergenceSummary>,
    /// Cluster memberships: `cluster_memberships[i]` lists the candidate
    /// indices belonging to cluster `i`.  Only populated for topic-family
    /// pipelines (TopicRank, MultipartiteRank).
    pub cluster_memberships: Option<Vec<Vec<usize>>>,
}

/// Summary statistics for the co-occurrence graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    /// Whether the graph was modified by a [`GraphTransform`] stage.
    pub is_transformed: bool,
}

/// Convergence summary from PageRank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceSummary {
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether PageRank converged within the threshold.
    pub converged: bool,
    /// Final L1-norm delta between last two iterations.
    pub final_delta: f64,
}

impl DebugPayload {
    /// Build a [`DebugPayload`] from pipeline artifacts based on the
    /// requested [`DebugLevel`].
    ///
    /// Returns `None` when `level` is [`DebugLevel::None`].
    ///
    /// # Arguments
    ///
    /// * `level` — Controls which fields are populated.
    /// * `graph` — The co-occurrence graph (for stats, node labels).
    /// * `ranks` — PageRank output (for scores, convergence, residuals).
    /// * `max_top_k` — Maximum number of node scores to include.
    pub fn build(
        level: DebugLevel,
        graph: &Graph,
        ranks: &RankOutput,
        max_top_k: usize,
    ) -> Option<Self> {
        if !level.is_enabled() {
            return None;
        }

        let mut payload = DebugPayload::default();

        // --- Stats level: graph stats + convergence summary ---
        if level.includes_stats() {
            payload.graph_stats = Some(GraphStats {
                num_nodes: graph.num_nodes(),
                num_edges: graph.num_edges(),
                is_transformed: graph.is_transformed(),
            });

            payload.convergence_summary = Some(ConvergenceSummary {
                iterations: ranks.iterations(),
                converged: ranks.converged(),
                final_delta: ranks.final_delta(),
            });
        }

        // --- TopNodes level: top-K node scores ---
        if level.includes_node_scores() {
            let num_nodes = ranks.num_nodes();
            let mut scored: Vec<(String, f64)> = (0..num_nodes as u32)
                .map(|id| (graph.lemma(id).to_string(), ranks.score(id)))
                .collect();

            // Sort by score descending, then by lemma ascending for stability.
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });

            scored.truncate(max_top_k);
            payload.node_scores = Some(scored);
        }

        // --- Full level: residuals + cluster memberships ---
        if level.includes_full() {
            // Convergence residuals (if diagnostics were captured).
            if let Some(diag) = ranks.diagnostics() {
                payload.residuals = Some(diag.residuals.clone());
            }

            // Cluster memberships (topic-family pipelines only).
            if let Some(assignments) = graph.cluster_assignments() {
                let num_clusters = assignments.num_clusters() as usize;
                let mut memberships: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
                for (cand_idx, &cluster_id) in assignments.as_slice().iter().enumerate() {
                    memberships[cluster_id as usize].push(cand_idx);
                }
                payload.cluster_memberships = Some(memberships);
            }
        }

        Some(payload)
    }
}

impl FormattedResult {
    /// Construct from the existing `ExtractionResult`.
    pub fn from_extraction(er: &crate::phrase::extraction::ExtractionResult) -> Self {
        Self {
            phrases: er.phrases.clone(),
            converged: er.converged,
            iterations: er.iterations as u32,
            debug: None,
            error: None,
        }
    }

    /// Build directly.
    pub fn new(phrases: Vec<crate::types::Phrase>, converged: bool, iterations: u32) -> Self {
        Self {
            phrases,
            converged,
            iterations,
            debug: None,
            error: None,
        }
    }

    /// Attach debug payload.
    pub fn with_debug(mut self, debug: DebugPayload) -> Self {
        self.debug = Some(debug);
        self
    }
}

// ============================================================================
// PipelineWorkspace — reusable scratch buffers
// ============================================================================

/// Reusable scratch buffers for reducing allocator churn across repeated
/// pipeline invocations (common in Python batch processing).
///
/// The workspace owns heap-allocated buffers that are **cleared but not
/// deallocated** between runs via [`PipelineWorkspace::clear`].  This avoids
/// repeated `malloc`/`free` cycles for the most allocation-heavy stages.
///
/// # Usage
///
/// Create once, pass `&mut workspace` into `PipelineRunner`, and reuse across
/// documents in a batch loop.
#[derive(Debug)]
pub struct PipelineWorkspace {
    /// Scratch buffer for edge pairs during graph construction.
    pub edge_buf: Vec<(u32, u32, f64)>,
    /// Scratch buffer for PageRank score vector.
    pub score_buf: Vec<f64>,
    /// Scratch buffer for PageRank normalization / dangling-mass.
    pub norm_buf: Vec<f64>,
    /// Scratch buffer for phrase grouping keys.
    pub group_keys: Vec<String>,
}

impl PipelineWorkspace {
    /// Create a new workspace with default (empty) buffers.
    pub fn new() -> Self {
        Self {
            edge_buf: Vec::new(),
            score_buf: Vec::new(),
            norm_buf: Vec::new(),
            group_keys: Vec::new(),
        }
    }

    /// Create a workspace with pre-allocated buffer capacities.
    ///
    /// Useful when approximate document sizes are known up front.
    pub fn with_capacity(edge_cap: usize, node_cap: usize, phrase_cap: usize) -> Self {
        Self {
            edge_buf: Vec::with_capacity(edge_cap),
            score_buf: Vec::with_capacity(node_cap),
            norm_buf: Vec::with_capacity(node_cap),
            group_keys: Vec::with_capacity(phrase_cap),
        }
    }

    /// Clear all buffers without deallocating.
    ///
    /// After this call, all buffers have `len() == 0` but retain their
    /// allocated capacity for the next pipeline invocation.
    pub fn clear(&mut self) {
        self.edge_buf.clear();
        self.score_buf.clear();
        self.norm_buf.clear();
        self.group_keys.clear();
    }

    /// Total heap capacity held by all buffers (in bytes, approximate).
    pub fn capacity_bytes(&self) -> usize {
        self.edge_buf.capacity() * std::mem::size_of::<(u32, u32, f64)>()
            + self.score_buf.capacity() * std::mem::size_of::<f64>()
            + self.norm_buf.capacity() * std::mem::size_of::<f64>()
            + self.group_keys.capacity() * std::mem::size_of::<String>()
    }
}

impl Default for PipelineWorkspace {
    fn default() -> Self {
        Self::new()
    }
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

    // ================================================================
    // CandidateSet — word-level tests
    // ================================================================

    /// Helper: tokens with a stopword ("is") for candidate filtering tests.
    fn tokens_with_stopword() -> Vec<Token> {
        let mut tokens = sample_tokens();
        tokens[2].is_stopword = true; // "is" sentence 0
        tokens[5].is_stopword = true; // "is" sentence 1
        tokens
    }

    #[test]
    fn test_word_candidates_default_pos() {
        let tokens = tokens_with_stopword();
        let stream = TokenStream::from_tokens(&tokens);
        // Empty include_pos → use default is_content_word() (Noun, Verb, Adj, ProperNoun).
        // Stopwords are excluded, so "be" (verb, stopword) is out.
        let cs = CandidateSet::from_word_tokens(&stream, &[], true);

        assert!(matches!(cs.kind(), CandidateKind::Words(_)));
        // "machine" NOUN, "learning" NOUN, "great" ADJ, "rust" PROPN, "fast" ADJ
        assert_eq!(cs.len(), 5);
        assert!(!cs.is_empty());

        // First positions should be monotonically increasing (dedup by graph key).
        let words = cs.words();
        for w in words {
            assert!(stream.pool().get(w.lemma_id).is_some());
        }
    }

    #[test]
    fn test_word_candidates_custom_pos() {
        let tokens = tokens_with_stopword();
        let stream = TokenStream::from_tokens(&tokens);
        // Only Nouns.
        let cs = CandidateSet::from_word_tokens(&stream, &[PosTag::Noun], true);

        let words = cs.words();
        // "machine" NOUN, "learning" NOUN
        assert_eq!(words.len(), 2);
        for w in words {
            assert_eq!(w.pos, PosTag::Noun);
        }
    }

    #[test]
    fn test_word_candidates_dedup_with_pos() {
        // Two tokens with same lemma but different POS.
        let tokens = vec![
            Token::new("fast", "fast", PosTag::Adjective, 0, 4, 0, 0),
            Token::new("fast", "fast", PosTag::Adverb, 5, 9, 0, 1),
        ];
        let stream = TokenStream::from_tokens(&tokens);

        // With POS in nodes: "fast|ADJ" and "fast|ADV" are distinct.
        let cs_pos = CandidateSet::from_word_tokens(&stream, &[], true);
        // Adverb is not a content word, so only ADJ passes default filter.
        assert_eq!(cs_pos.len(), 1);

        // Allow both ADJ and ADV explicitly:
        let cs_both =
            CandidateSet::from_word_tokens(&stream, &[PosTag::Adjective, PosTag::Adverb], true);
        assert_eq!(cs_both.len(), 2);

        // Without POS in nodes: same lemma → deduplicated to one.
        let cs_no_pos =
            CandidateSet::from_word_tokens(&stream, &[PosTag::Adjective, PosTag::Adverb], false);
        assert_eq!(cs_no_pos.len(), 1);
    }

    #[test]
    fn test_word_candidates_first_position() {
        let tokens = vec![
            Token::new("great", "great", PosTag::Adjective, 0, 5, 0, 0),
            Token::new("world", "world", PosTag::Noun, 6, 11, 0, 1),
            Token::new("great", "great", PosTag::Adjective, 12, 17, 0, 2), // duplicate
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cs = CandidateSet::from_word_tokens(&stream, &[], true);

        let words = cs.words();
        let great = words
            .iter()
            .find(|w| stream.pool().get(w.lemma_id) == Some("great"))
            .unwrap();
        // First position is token 0, not 2.
        assert_eq!(great.first_position, 0);
    }

    #[test]
    fn test_word_candidates_empty_stream() {
        let stream = TokenStream::from_tokens(&[]);
        let cs = CandidateSet::from_word_tokens(&stream, &[], true);

        assert!(cs.is_empty());
        assert_eq!(cs.len(), 0);
    }

    #[test]
    fn test_word_candidate_graph_key() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cs = CandidateSet::from_word_tokens(&stream, &[], true);

        let words = cs.words();
        let w0 = &words[0]; // first candidate: "machine" NOUN
        assert_eq!(w0.graph_key(stream.pool(), true), "machine|NOUN");
        assert_eq!(w0.graph_key(stream.pool(), false), "machine");
    }

    // ================================================================
    // CandidateSet — phrase-level tests
    // ================================================================

    #[test]
    fn test_phrase_candidates_from_chunks() {
        let tokens = vec![
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("is", "be", PosTag::Verb, 17, 19, 0, 2),
            Token::new("very", "very", PosTag::Adverb, 20, 24, 0, 3),
            Token::new("fast", "fast", PosTag::Adjective, 25, 29, 0, 4),
        ];
        let stream = TokenStream::from_tokens(&tokens);

        let chunks = vec![crate::types::ChunkSpan {
            start_token: 0,
            end_token: 2,
            start_char: 0,
            end_char: 16,
            sentence_idx: 0,
        }];

        let cs = CandidateSet::from_phrase_chunks(&stream, &chunks);
        assert!(matches!(cs.kind(), CandidateKind::Phrases(_)));
        assert_eq!(cs.len(), 1);

        let p = &cs.phrases()[0];
        assert_eq!(p.start_token, 0);
        assert_eq!(p.end_token, 2);
        assert_eq!(p.start_char, 0);
        assert_eq!(p.end_char, 16);
        assert_eq!(p.sentence_idx, 0);
        assert_eq!(p.token_len(), 2);
        // lemma_ids: "machine", "learning"
        assert_eq!(p.lemma_ids.len(), 2);
        // Both tokens are non-stopword, so term_ids has 2 unique text_ids.
        assert_eq!(p.term_ids.len(), 2);
    }

    #[test]
    fn test_phrase_candidates_stopword_excluded_from_terms() {
        let mut tokens = vec![
            Token::new("the", "the", PosTag::Determiner, 0, 3, 0, 0),
            Token::new("big", "big", PosTag::Adjective, 4, 7, 0, 1),
            Token::new("cat", "cat", PosTag::Noun, 8, 11, 0, 2),
        ];
        tokens[0].is_stopword = true; // "the" is a stopword

        let stream = TokenStream::from_tokens(&tokens);
        let chunks = vec![crate::types::ChunkSpan {
            start_token: 0,
            end_token: 3,
            start_char: 0,
            end_char: 11,
            sentence_idx: 0,
        }];

        let cs = CandidateSet::from_phrase_chunks(&stream, &chunks);
        let p = &cs.phrases()[0];

        // All 3 tokens contribute lemma_ids.
        assert_eq!(p.lemma_ids.len(), 3);
        // Only non-stopword tokens contribute to term_ids.
        assert_eq!(p.term_ids.len(), 2); // "big", "cat"
    }

    #[test]
    fn test_phrase_candidates_empty() {
        let stream = TokenStream::from_tokens(&[]);
        let cs = CandidateSet::from_phrase_chunks(&stream, &[]);

        assert!(cs.is_empty());
        assert_eq!(cs.len(), 0);
    }

    // ================================================================
    // CandidateSetRef tests
    // ================================================================

    #[test]
    fn test_candidate_set_ref_mirrors_owned() {
        let tokens = tokens_with_stopword();
        let stream = TokenStream::from_tokens(&tokens);
        let cs = CandidateSet::from_word_tokens(&stream, &[], true);
        let r = cs.as_ref();

        assert_eq!(r.len(), cs.len());
        assert_eq!(r.words().len(), cs.words().len());
        assert!(matches!(r.kind(), CandidateKind::Words(_)));
    }

    #[test]
    fn test_candidate_set_ref_is_copy() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cs = CandidateSet::from_word_tokens(&stream, &[], true);
        let r1 = cs.as_ref();
        let r2 = r1; // Copy
        assert_eq!(r1.len(), r2.len());
    }

    #[test]
    #[should_panic(expected = "called phrases() on non-phrase")]
    fn test_words_panics_on_phrases_access() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cs = CandidateSet::from_word_tokens(&stream, &[], true);
        let _ = cs.phrases(); // should panic
    }

    #[test]
    #[should_panic(expected = "called words() on non-word")]
    fn test_phrases_panics_on_words_access() {
        let stream = TokenStream::from_tokens(&[Token::new("a", "a", PosTag::Noun, 0, 1, 0, 0)]);
        let chunks = vec![crate::types::ChunkSpan {
            start_token: 0,
            end_token: 1,
            start_char: 0,
            end_char: 1,
            sentence_idx: 0,
        }];
        let cs = CandidateSet::from_phrase_chunks(&stream, &chunks);
        let _ = cs.words(); // should panic
    }

    // ================================================================
    // Graph artifact tests
    // ================================================================

    /// Helper: build a small graph via the existing builder.
    fn sample_graph_builder() -> crate::graph::builder::GraphBuilder {
        let mut builder = crate::graph::builder::GraphBuilder::new();
        let a = builder.get_or_create_node("machine|NOUN");
        let b = builder.get_or_create_node("learning|NOUN");
        let c = builder.get_or_create_node("great|ADJ");
        builder.increment_edge(a, b, 1.0);
        builder.increment_edge(b, c, 1.0);
        builder.increment_edge(a, c, 1.0);
        builder
    }

    #[test]
    fn test_graph_from_builder() {
        let builder = sample_graph_builder();
        let graph = Graph::from_builder(&builder);

        assert_eq!(graph.num_nodes(), 3);
        assert!(!graph.is_empty());
        // Each undirected edge is stored in both directions → 6 directed entries.
        assert_eq!(graph.num_edges(), 6);
        assert!(!graph.is_transformed());
    }

    #[test]
    fn test_graph_from_csr() {
        let builder = sample_graph_builder();
        let csr = crate::graph::csr::CsrGraph::from_builder(&builder);
        let graph = Graph::from_csr(csr);

        assert_eq!(graph.num_nodes(), 3);
        assert!(!graph.is_transformed());
    }

    #[test]
    fn test_graph_empty() {
        let builder = crate::graph::builder::GraphBuilder::new();
        let graph = Graph::from_builder(&builder);

        assert!(graph.is_empty());
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_graph_node_lookup() {
        let graph = Graph::from_builder(&sample_graph_builder());

        assert_eq!(graph.get_node_by_lemma("machine|NOUN"), Some(0));
        assert_eq!(graph.get_node_by_lemma("learning|NOUN"), Some(1));
        assert_eq!(graph.get_node_by_lemma("great|ADJ"), Some(2));
        assert_eq!(graph.get_node_by_lemma("nonexistent"), None);
    }

    #[test]
    fn test_graph_neighbors() {
        let graph = Graph::from_builder(&sample_graph_builder());

        // Node 0 ("machine|NOUN") should neighbor nodes 1 and 2.
        let neighbors: Vec<(u32, f64)> = graph.neighbors(0).collect();
        assert_eq!(neighbors.len(), 2);
        let target_ids: Vec<u32> = neighbors.iter().map(|(id, _)| *id).collect();
        assert!(target_ids.contains(&1));
        assert!(target_ids.contains(&2));
    }

    #[test]
    fn test_graph_lemma() {
        let graph = Graph::from_builder(&sample_graph_builder());

        assert_eq!(graph.lemma(0), "machine|NOUN");
        assert_eq!(graph.lemma(1), "learning|NOUN");
        assert_eq!(graph.lemma(2), "great|ADJ");
    }

    #[test]
    fn test_graph_dangling_nodes() {
        let mut builder = crate::graph::builder::GraphBuilder::new();
        let a = builder.get_or_create_node("a");
        let b = builder.get_or_create_node("b");
        let _c = builder.get_or_create_node("c"); // dangling
        builder.increment_edge(a, b, 1.0);

        let graph = Graph::from_builder(&builder);
        let dangling = graph.dangling_nodes();
        assert!(dangling.contains(&2)); // "c" has no edges
    }

    #[test]
    fn test_graph_transformed_flag() {
        let mut graph = Graph::from_builder(&sample_graph_builder());

        assert!(!graph.is_transformed());

        // Accessing csr_mut should flip the flag.
        let _ = graph.csr_mut();
        assert!(graph.is_transformed());
    }

    #[test]
    fn test_graph_set_transformed() {
        let mut graph = Graph::from_builder(&sample_graph_builder());
        assert!(!graph.is_transformed());

        graph.set_transformed();
        assert!(graph.is_transformed());
    }

    #[test]
    fn test_graph_into_csr() {
        let graph = Graph::from_builder(&sample_graph_builder());
        let csr = graph.into_csr();

        // Should still have the same data.
        assert_eq!(csr.num_nodes, 3);
        assert_eq!(csr.get_node_by_lemma("machine|NOUN"), Some(0));
    }

    #[test]
    fn test_graph_csr_direct_access() {
        let graph = Graph::from_builder(&sample_graph_builder());

        // Direct CSR access for hot-path-style iteration.
        let csr = graph.csr();
        assert_eq!(csr.num_nodes, 3);
        assert!((csr.node_total_weight(0) - 2.0).abs() < 1e-10); // edges to b and c, weight 1 each
    }

    #[test]
    fn test_graph_clone() {
        let graph = Graph::from_builder(&sample_graph_builder());
        let cloned = graph.clone();

        assert_eq!(cloned.num_nodes(), graph.num_nodes());
        assert_eq!(cloned.num_edges(), graph.num_edges());
        assert_eq!(cloned.is_transformed(), graph.is_transformed());
    }

    // ================================================================
    // TeleportVector tests
    // ================================================================

    #[test]
    fn test_teleport_vector_new() {
        let tv = TeleportVector::new(vec![0.5, 0.3, 0.2], TeleportType::Uniform);

        assert_eq!(tv.len(), 3);
        assert!(!tv.is_empty());
        assert_eq!(tv.as_slice(), &[0.5, 0.3, 0.2]);
        assert_eq!(tv.teleport_type(), TeleportType::Uniform);
        assert!(tv.debug_source().is_none());
    }

    #[test]
    fn test_teleport_vector_metadata() {
        let tv = TeleportVector::new(vec![0.5, 0.3, 0.2], TeleportType::Focus)
            .with_source("focus on 'machine learning'");

        assert_eq!(tv.teleport_type(), TeleportType::Focus);
        assert_eq!(tv.debug_source(), Some("focus on 'machine learning'"));
    }

    #[test]
    #[should_panic(expected = "normalized")]
    fn test_teleport_vector_new_rejects_unnormalized() {
        TeleportVector::new(vec![1.0, 1.0, 1.0], TeleportType::Uniform);
    }

    #[test]
    fn test_teleport_vector_uniform() {
        let tv = TeleportVector::uniform(4);

        assert_eq!(tv.len(), 4);
        assert!(tv.is_normalized(1e-15));
        assert_eq!(tv.teleport_type(), TeleportType::Uniform);
        for &v in tv.as_slice() {
            assert!((v - 0.25).abs() < 1e-15);
        }
    }

    #[test]
    fn test_teleport_vector_uniform_zero() {
        let tv = TeleportVector::uniform(0);

        assert!(tv.is_empty());
        assert_eq!(tv.len(), 0);
        assert!(tv.is_normalized(1e-10));
    }

    #[test]
    fn test_teleport_vector_zeros() {
        let tv = TeleportVector::zeros(5, TeleportType::Position);

        assert_eq!(tv.len(), 5);
        assert_eq!(tv.teleport_type(), TeleportType::Position);
        for &v in tv.as_slice() {
            assert!(v.abs() < 1e-15);
        }
    }

    #[test]
    fn test_teleport_vector_get() {
        let tv = TeleportVector::new(vec![0.1, 0.9], TeleportType::Uniform);

        assert!((tv.get(0) - 0.1).abs() < 1e-15);
        assert!((tv.get(1) - 0.9).abs() < 1e-15);
        // Out of bounds returns 0.0.
        assert!(tv.get(99).abs() < 1e-15);
    }

    #[test]
    fn test_teleport_vector_set() {
        let mut tv = TeleportVector::zeros(3, TeleportType::Uniform);

        tv.set(0, 0.5);
        tv.set(2, 0.5);

        assert!((tv[0] - 0.5).abs() < 1e-15);
        assert!(tv[1].abs() < 1e-15);
        assert!((tv[2] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_teleport_vector_index() {
        let tv = TeleportVector::new(vec![0.3, 0.7], TeleportType::Uniform);

        assert!((tv[0] - 0.3).abs() < 1e-15);
        assert!((tv[1] - 0.7).abs() < 1e-15);
    }

    #[test]
    fn test_teleport_vector_index_mut() {
        let mut tv = TeleportVector::new(vec![0.5, 0.5], TeleportType::Uniform);

        tv[0] = 0.8;
        tv[1] = 0.2;

        assert!((tv[0] - 0.8).abs() < 1e-15);
        assert!((tv[1] - 0.2).abs() < 1e-15);
    }

    #[test]
    fn test_teleport_vector_is_normalized() {
        let tv = TeleportVector::new(vec![0.5, 0.3, 0.2], TeleportType::Uniform);
        assert!(tv.is_normalized(1e-10));

        // Use zeros + set to build an unnormalized vector for testing is_normalized.
        let mut tv2 = TeleportVector::zeros(3, TeleportType::Uniform);
        tv2.set(0, 1.0);
        tv2.set(1, 1.0);
        tv2.set(2, 1.0);
        assert!(!tv2.is_normalized(1e-10));
    }

    #[test]
    fn test_teleport_vector_normalize() {
        let mut tv = TeleportVector::zeros(3, TeleportType::Uniform);
        tv.set(0, 2.0);
        tv.set(1, 3.0);
        tv.set(2, 5.0);
        tv.normalize();

        assert!(tv.is_normalized(1e-10));
        assert!((tv[0] - 0.2).abs() < 1e-10);
        assert!((tv[1] - 0.3).abs() < 1e-10);
        assert!((tv[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_teleport_vector_normalize_zero_sum() {
        let mut tv = TeleportVector::zeros(3, TeleportType::Uniform);
        tv.normalize();

        // Falls back to uniform.
        assert!(tv.is_normalized(1e-10));
        for &v in tv.as_slice() {
            assert!((v - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_teleport_vector_normalize_empty() {
        let mut tv = TeleportVector::new(vec![], TeleportType::Uniform);
        tv.normalize();

        assert!(tv.is_empty());
        assert!(tv.is_normalized(1e-10));
    }

    #[test]
    fn test_teleport_vector_into_inner() {
        let tv = TeleportVector::new(vec![0.1, 0.2, 0.7], TeleportType::Uniform);
        let inner = tv.into_inner();

        assert_eq!(inner, vec![0.1, 0.2, 0.7]);
    }

    #[test]
    fn test_teleport_vector_iter() {
        let tv = TeleportVector::new(vec![0.3, 0.7], TeleportType::Uniform);
        let sum: f64 = tv.iter().sum();

        assert!((sum - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_teleport_vector_as_mut_slice() {
        let mut tv = TeleportVector::zeros(3, TeleportType::Uniform);
        let slice = tv.as_mut_slice();
        slice[0] = 0.5;
        slice[1] = 0.3;
        slice[2] = 0.2;

        assert!(tv.is_normalized(1e-10));
    }

    #[test]
    fn test_teleport_vector_clone_eq() {
        let tv = TeleportVector::new(vec![0.25, 0.25, 0.5], TeleportType::Position);
        let cloned = tv.clone();

        assert_eq!(tv, cloned);
    }

    #[test]
    #[should_panic]
    fn test_teleport_vector_index_out_of_bounds() {
        let tv = TeleportVector::new(vec![1.0], TeleportType::Uniform);
        let _ = tv[5]; // should panic
    }

    #[test]
    fn test_teleport_vector_single_node() {
        let tv = TeleportVector::uniform(1);

        assert_eq!(tv.len(), 1);
        assert!((tv[0] - 1.0).abs() < 1e-15);
        assert!(tv.is_normalized(1e-15));
    }

    // ================================================================
    // RankOutput tests
    // ================================================================

    #[test]
    fn test_rank_output_new() {
        let ro = RankOutput::new(vec![0.3, 0.5, 0.2], true, 42, 1e-7);

        assert_eq!(ro.scores(), &[0.3, 0.5, 0.2]);
        assert!(ro.converged());
        assert_eq!(ro.iterations(), 42);
        assert!((ro.final_delta() - 1e-7).abs() < 1e-15);
        assert_eq!(ro.num_nodes(), 3);
        assert!(ro.diagnostics().is_none());
    }

    #[test]
    fn test_rank_output_score_lookup() {
        let ro = RankOutput::new(vec![0.1, 0.9], true, 10, 0.0);

        assert!((ro.score(0) - 0.1).abs() < 1e-15);
        assert!((ro.score(1) - 0.9).abs() < 1e-15);
        // Out-of-bounds returns 0.0.
        assert!((ro.score(999)).abs() < 1e-15);
    }

    #[test]
    fn test_rank_output_from_pagerank_result() {
        let pr = crate::pagerank::PageRankResult {
            scores: vec![0.25, 0.25, 0.25, 0.25],
            iterations: 50,
            delta: 5e-7,
            converged: true,
        };
        let ro = RankOutput::from_pagerank_result(&pr);

        assert_eq!(ro.scores(), &[0.25, 0.25, 0.25, 0.25]);
        assert!(ro.converged());
        assert_eq!(ro.iterations(), 50);
        assert!((ro.final_delta() - 5e-7).abs() < 1e-15);
    }

    #[test]
    fn test_rank_output_with_diagnostics() {
        let diag = RankDiagnostics {
            residuals: vec![0.5, 0.1, 0.01, 0.001],
        };
        let ro = RankOutput::new(vec![1.0], true, 4, 0.001).with_diagnostics(diag);

        let d = ro.diagnostics().unwrap();
        assert_eq!(d.residuals.len(), 4);
        assert!((d.residuals[0] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_rank_output_not_converged() {
        let ro = RankOutput::new(vec![0.5, 0.5], false, 100, 0.05);

        assert!(!ro.converged());
        assert_eq!(ro.iterations(), 100);
    }

    // ================================================================
    // PhraseSet tests
    // ================================================================

    fn sample_phrases() -> Vec<crate::types::Phrase> {
        vec![
            crate::types::Phrase {
                text: "machine learning".to_string(),
                lemma: "machine learning".to_string(),
                score: 0.85,
                count: 3,
                offsets: vec![(0, 2), (10, 12), (20, 22)],
                rank: 1,
            },
            crate::types::Phrase {
                text: "neural network".to_string(),
                lemma: "neural network".to_string(),
                score: 0.72,
                count: 2,
                offsets: vec![(5, 7)],
                rank: 2,
            },
        ]
    }

    #[test]
    fn test_phrase_set_from_phrases() {
        let phrases = sample_phrases();
        let mut pool = StringPool::new();
        let ps = PhraseSet::from_phrases(&phrases, &mut pool);

        assert_eq!(ps.len(), 2);
        assert!(!ps.is_empty());

        let e0 = &ps.entries()[0];
        assert!((e0.score - 0.85).abs() < 1e-15);
        assert_eq!(e0.count, 3);
        assert_eq!(e0.surface.as_deref(), Some("machine learning"));
        assert_eq!(e0.lemma_text.as_deref(), Some("machine learning"));
        // "machine learning" → 2 lemma tokens.
        assert_eq!(e0.lemma_ids.len(), 2);
        // 3 occurrence spans.
        assert_eq!(e0.spans.as_ref().unwrap().len(), 3);

        let e1 = &ps.entries()[1];
        assert!((e1.score - 0.72).abs() < 1e-15);
        assert_eq!(e1.count, 2);
    }

    #[test]
    fn test_phrase_set_empty() {
        let mut pool = StringPool::new();
        let ps = PhraseSet::from_phrases(&[], &mut pool);

        assert!(ps.is_empty());
        assert_eq!(ps.len(), 0);
    }

    #[test]
    fn test_phrase_set_ref_mirrors_owned() {
        let phrases = sample_phrases();
        let mut pool = StringPool::new();
        let ps = PhraseSet::from_phrases(&phrases, &mut pool);
        let r = ps.as_ref();

        assert_eq!(r.len(), ps.len());
        assert_eq!(r.entries().len(), ps.entries().len());
        assert!((r.entries()[0].score - 0.85).abs() < 1e-15);
    }

    #[test]
    fn test_phrase_set_no_offsets() {
        let phrases = vec![crate::types::Phrase::new("test", "test", 0.5, 1)];
        let mut pool = StringPool::new();
        let ps = PhraseSet::from_phrases(&phrases, &mut pool);

        // Phrase::new creates empty offsets → spans should be None.
        assert!(ps.entries()[0].spans.is_none());
    }

    #[test]
    fn test_phrase_entry_from_raw() {
        let entry = PhraseEntry {
            lemma_ids: vec![0, 1],
            score: 0.9,
            count: 5,
            surface: None,
            lemma_text: None,
            spans: None,
        };
        let ps = PhraseSet::from_entries(vec![entry]);

        assert_eq!(ps.len(), 1);
        assert!(ps.entries()[0].surface.is_none());
    }

    // ================================================================
    // FormattedResult tests
    // ================================================================

    #[test]
    fn test_formatted_result_new() {
        let phrases = sample_phrases();
        let fr = FormattedResult::new(phrases.clone(), true, 50);

        assert_eq!(fr.phrases.len(), 2);
        assert!(fr.converged);
        assert_eq!(fr.iterations, 50);
        assert!(fr.debug.is_none());
    }

    #[test]
    fn test_formatted_result_from_extraction() {
        let er = crate::phrase::extraction::ExtractionResult {
            phrases: sample_phrases(),
            converged: false,
            iterations: 100,
        };
        let fr = FormattedResult::from_extraction(&er);

        assert_eq!(fr.phrases.len(), 2);
        assert!(!fr.converged);
        assert_eq!(fr.iterations, 100);
        assert!(fr.debug.is_none());
    }

    #[test]
    fn test_formatted_result_with_debug() {
        let fr = FormattedResult::new(vec![], true, 10).with_debug(DebugPayload {
            node_scores: Some(vec![("machine|NOUN".to_string(), 0.5)]),
            graph_stats: Some(GraphStats {
                num_nodes: 10,
                num_edges: 25,
                is_transformed: false,
            }),
            stage_timings: None,
            residuals: None,
            convergence_summary: None,
            cluster_memberships: None,
        });

        let d = fr.debug.as_ref().unwrap();
        assert_eq!(d.node_scores.as_ref().unwrap().len(), 1);
        assert_eq!(d.graph_stats.as_ref().unwrap().num_nodes, 10);
        assert!(d.stage_timings.is_none());
    }

    // ================================================================
    // PipelineWorkspace tests
    // ================================================================

    #[test]
    fn test_workspace_new() {
        let ws = PipelineWorkspace::new();

        assert!(ws.edge_buf.is_empty());
        assert!(ws.score_buf.is_empty());
        assert!(ws.norm_buf.is_empty());
        assert!(ws.group_keys.is_empty());
    }

    #[test]
    fn test_workspace_with_capacity() {
        let ws = PipelineWorkspace::with_capacity(100, 50, 20);

        assert!(ws.edge_buf.capacity() >= 100);
        assert!(ws.score_buf.capacity() >= 50);
        assert!(ws.norm_buf.capacity() >= 50);
        assert!(ws.group_keys.capacity() >= 20);
    }

    #[test]
    fn test_workspace_clear_retains_capacity() {
        let mut ws = PipelineWorkspace::new();
        // Fill buffers.
        ws.edge_buf.extend_from_slice(&[(0, 1, 1.0), (1, 2, 2.0)]);
        ws.score_buf.extend_from_slice(&[0.5; 100]);
        ws.norm_buf.extend_from_slice(&[1.0; 100]);
        ws.group_keys.push("test".to_string());

        let cap_before = ws.capacity_bytes();
        assert!(cap_before > 0);

        ws.clear();

        assert!(ws.edge_buf.is_empty());
        assert!(ws.score_buf.is_empty());
        assert!(ws.norm_buf.is_empty());
        assert!(ws.group_keys.is_empty());
        // Capacity retained.
        assert_eq!(ws.capacity_bytes(), cap_before);
    }

    #[test]
    fn test_workspace_default() {
        let ws = PipelineWorkspace::default();
        assert!(ws.edge_buf.is_empty());
    }

    // ================================================================
    // ClusterAssignments tests
    // ================================================================

    #[test]
    fn test_cluster_assignments_empty() {
        let ca = ClusterAssignments::empty();
        assert!(ca.is_empty());
        assert_eq!(ca.num_clusters(), 0);
        assert_eq!(ca.num_candidates(), 0);
        assert!(ca.as_slice().is_empty());
    }

    #[test]
    fn test_cluster_assignments_from_cluster_vecs() {
        // 5 candidates, 3 clusters:
        //   cluster 0: [0, 2]
        //   cluster 1: [1, 4]
        //   cluster 2: [3]
        let clusters = vec![vec![0, 2], vec![1, 4], vec![3]];
        let ca = ClusterAssignments::from_cluster_vecs(&clusters, 5);

        assert_eq!(ca.num_clusters(), 3);
        assert_eq!(ca.num_candidates(), 5);
        assert!(!ca.is_empty());

        assert_eq!(ca.cluster_of(0), 0);
        assert_eq!(ca.cluster_of(1), 1);
        assert_eq!(ca.cluster_of(2), 0);
        assert_eq!(ca.cluster_of(3), 2);
        assert_eq!(ca.cluster_of(4), 1);
    }

    #[test]
    fn test_cluster_assignments_members_of() {
        let clusters = vec![vec![0, 3], vec![1, 2]];
        let ca = ClusterAssignments::from_cluster_vecs(&clusters, 4);

        let mut m0 = ca.members_of(0);
        m0.sort();
        assert_eq!(m0, vec![0, 3]);

        let mut m1 = ca.members_of(1);
        m1.sort();
        assert_eq!(m1, vec![1, 2]);

        // Non-existent cluster returns empty.
        assert!(ca.members_of(99).is_empty());
    }

    #[test]
    fn test_cluster_assignments_single_cluster() {
        let clusters = vec![vec![0, 1, 2]];
        let ca = ClusterAssignments::from_cluster_vecs(&clusters, 3);

        assert_eq!(ca.num_clusters(), 1);
        for i in 0..3 {
            assert_eq!(ca.cluster_of(i), 0);
        }
    }

    #[test]
    fn test_cluster_assignments_as_slice() {
        let clusters = vec![vec![0], vec![1], vec![2]];
        let ca = ClusterAssignments::from_cluster_vecs(&clusters, 3);
        assert_eq!(ca.as_slice(), &[0, 1, 2]);
    }

    // ================================================================
    // DebugPayload::build() tests
    // ================================================================

    /// Helper: build a Graph + RankOutput for debug payload tests.
    fn debug_test_graph_and_ranks() -> (Graph, RankOutput) {
        let builder = sample_graph_builder(); // 3 nodes: machine|NOUN, learning|NOUN, great|ADJ
        let graph = Graph::from_builder(&builder);
        let ranks = RankOutput::from_pagerank_result(&crate::pagerank::PageRankResult {
            scores: vec![0.5, 0.3, 0.2],
            iterations: 42,
            delta: 1e-7,
            converged: true,
        });
        (graph, ranks)
    }

    #[test]
    fn test_debug_build_none_returns_none() {
        let (graph, ranks) = debug_test_graph_and_ranks();
        assert!(DebugPayload::build(DebugLevel::None, &graph, &ranks, 50).is_none());
    }

    #[test]
    fn test_debug_build_stats_populates_graph_stats() {
        let (graph, ranks) = debug_test_graph_and_ranks();
        let payload = DebugPayload::build(DebugLevel::Stats, &graph, &ranks, 50).unwrap();

        let gs = payload
            .graph_stats
            .as_ref()
            .expect("graph_stats should be populated");
        assert_eq!(gs.num_nodes, 3);
        assert_eq!(gs.num_edges, 6); // 3 undirected = 6 directed
        assert!(!gs.is_transformed);
    }

    #[test]
    fn test_debug_build_stats_populates_convergence_summary() {
        let (graph, ranks) = debug_test_graph_and_ranks();
        let payload = DebugPayload::build(DebugLevel::Stats, &graph, &ranks, 50).unwrap();

        let cs = payload
            .convergence_summary
            .as_ref()
            .expect("convergence_summary should be populated");
        assert_eq!(cs.iterations, 42);
        assert!(cs.converged);
        assert!((cs.final_delta - 1e-7).abs() < 1e-15);
    }

    #[test]
    fn test_debug_build_stats_no_node_scores() {
        let (graph, ranks) = debug_test_graph_and_ranks();
        let payload = DebugPayload::build(DebugLevel::Stats, &graph, &ranks, 50).unwrap();

        assert!(
            payload.node_scores.is_none(),
            "Stats level should not include node_scores"
        );
        assert!(payload.residuals.is_none());
        assert!(payload.cluster_memberships.is_none());
    }

    #[test]
    fn test_debug_build_top_nodes_has_scores() {
        let (graph, ranks) = debug_test_graph_and_ranks();
        let payload = DebugPayload::build(DebugLevel::TopNodes, &graph, &ranks, 50).unwrap();

        let scores = payload
            .node_scores
            .as_ref()
            .expect("node_scores should be populated");
        assert_eq!(scores.len(), 3);
        // Should be sorted by score descending.
        assert_eq!(scores[0].0, "machine|NOUN");
        assert!((scores[0].1 - 0.5).abs() < 1e-10);
        assert_eq!(scores[1].0, "learning|NOUN");
        assert_eq!(scores[2].0, "great|ADJ");
        // Also includes stats (superset).
        assert!(payload.graph_stats.is_some());
        assert!(payload.convergence_summary.is_some());
    }

    #[test]
    fn test_debug_build_top_nodes_respects_top_k() {
        let (graph, ranks) = debug_test_graph_and_ranks();
        let payload = DebugPayload::build(DebugLevel::TopNodes, &graph, &ranks, 2).unwrap();

        let scores = payload.node_scores.as_ref().unwrap();
        assert_eq!(scores.len(), 2, "Should be truncated to top_k=2");
        assert_eq!(scores[0].0, "machine|NOUN"); // highest score
        assert_eq!(scores[1].0, "learning|NOUN");
    }

    #[test]
    fn test_debug_build_full_includes_residuals() {
        // Build with diagnostics enabled.
        let (graph, _) = debug_test_graph_and_ranks();
        let ranks = RankOutput::from_pagerank_result(&crate::pagerank::PageRankResult {
            scores: vec![0.5, 0.3, 0.2],
            iterations: 3,
            delta: 1e-7,
            converged: true,
        })
        .with_diagnostics(RankDiagnostics {
            residuals: vec![0.1, 0.01, 0.001],
        });

        let payload = DebugPayload::build(DebugLevel::Full, &graph, &ranks, 50).unwrap();

        let residuals = payload
            .residuals
            .as_ref()
            .expect("residuals should be populated at Full level");
        assert_eq!(residuals.len(), 3);
        assert!((residuals[0] - 0.1).abs() < 1e-10);
        // Also includes node_scores and stats (superset).
        assert!(payload.node_scores.is_some());
        assert!(payload.graph_stats.is_some());
    }

    #[test]
    fn test_debug_build_full_no_residuals_without_diagnostics() {
        let (graph, ranks) = debug_test_graph_and_ranks();
        let payload = DebugPayload::build(DebugLevel::Full, &graph, &ranks, 50).unwrap();

        // No diagnostics were set, so residuals should be None.
        assert!(payload.residuals.is_none());
    }

    #[test]
    fn test_debug_build_full_includes_cluster_memberships() {
        let (mut graph, ranks) = debug_test_graph_and_ranks();

        // Simulate a topic-family pipeline with cluster assignments.
        let clusters = vec![vec![0, 1], vec![2]]; // two clusters
        let assignments = ClusterAssignments::from_cluster_vecs(&clusters, 3);
        graph.set_cluster_assignments(assignments);

        let payload = DebugPayload::build(DebugLevel::Full, &graph, &ranks, 50).unwrap();

        let memberships = payload
            .cluster_memberships
            .as_ref()
            .expect("cluster_memberships should be populated for topic-family");
        assert_eq!(memberships.len(), 2);
        assert_eq!(memberships[0], vec![0, 1]);
        assert_eq!(memberships[1], vec![2]);
    }

    #[test]
    fn test_debug_build_full_no_clusters_when_not_topic_family() {
        let (graph, ranks) = debug_test_graph_and_ranks();
        let payload = DebugPayload::build(DebugLevel::Full, &graph, &ranks, 50).unwrap();

        // No cluster assignments → no cluster memberships.
        assert!(payload.cluster_memberships.is_none());
    }

    #[test]
    fn test_debug_build_node_scores_tiebreak_by_lemma() {
        // Two nodes with identical scores — should sort by lemma ascending.
        let mut builder = crate::graph::builder::GraphBuilder::new();
        let a = builder.get_or_create_node("zebra|NOUN");
        let b = builder.get_or_create_node("alpha|NOUN");
        builder.increment_edge(a, b, 1.0);
        let graph = Graph::from_builder(&builder);

        let ranks = RankOutput::from_pagerank_result(&crate::pagerank::PageRankResult {
            scores: vec![0.5, 0.5],
            iterations: 10,
            delta: 1e-7,
            converged: true,
        });

        let payload = DebugPayload::build(DebugLevel::TopNodes, &graph, &ranks, 50).unwrap();
        let scores = payload.node_scores.as_ref().unwrap();
        assert_eq!(
            scores[0].0, "alpha|NOUN",
            "Equal scores should sort by lemma ascending"
        );
        assert_eq!(scores[1].0, "zebra|NOUN");
    }

    // ================================================================
    // DebugLevel tests
    // ================================================================

    #[test]
    fn test_debug_level_default_is_none() {
        assert_eq!(DebugLevel::default(), DebugLevel::None);
    }

    #[test]
    fn test_debug_level_ordering() {
        assert!(DebugLevel::None < DebugLevel::Stats);
        assert!(DebugLevel::Stats < DebugLevel::TopNodes);
        assert!(DebugLevel::TopNodes < DebugLevel::Full);
    }

    #[test]
    fn test_debug_level_is_enabled() {
        assert!(!DebugLevel::None.is_enabled());
        assert!(DebugLevel::Stats.is_enabled());
        assert!(DebugLevel::TopNodes.is_enabled());
        assert!(DebugLevel::Full.is_enabled());
    }

    #[test]
    fn test_debug_level_includes_stats() {
        assert!(!DebugLevel::None.includes_stats());
        assert!(DebugLevel::Stats.includes_stats());
        assert!(DebugLevel::TopNodes.includes_stats());
        assert!(DebugLevel::Full.includes_stats());
    }

    #[test]
    fn test_debug_level_includes_node_scores() {
        assert!(!DebugLevel::None.includes_node_scores());
        assert!(!DebugLevel::Stats.includes_node_scores());
        assert!(DebugLevel::TopNodes.includes_node_scores());
        assert!(DebugLevel::Full.includes_node_scores());
    }

    #[test]
    fn test_debug_level_includes_full() {
        assert!(!DebugLevel::None.includes_full());
        assert!(!DebugLevel::Stats.includes_full());
        assert!(!DebugLevel::TopNodes.includes_full());
        assert!(DebugLevel::Full.includes_full());
    }

    #[test]
    fn test_debug_level_as_str() {
        assert_eq!(DebugLevel::None.as_str(), "none");
        assert_eq!(DebugLevel::Stats.as_str(), "stats");
        assert_eq!(DebugLevel::TopNodes.as_str(), "top_nodes");
        assert_eq!(DebugLevel::Full.as_str(), "full");
    }

    #[test]
    fn test_debug_level_from_str() {
        assert_eq!(DebugLevel::parse_str("none"), Some(DebugLevel::None));
        assert_eq!(DebugLevel::parse_str("stats"), Some(DebugLevel::Stats));
        assert_eq!(
            DebugLevel::parse_str("top_nodes"),
            Some(DebugLevel::TopNodes)
        );
        assert_eq!(
            DebugLevel::parse_str("topnodes"),
            Some(DebugLevel::TopNodes)
        );
        assert_eq!(DebugLevel::parse_str("full"), Some(DebugLevel::Full));
        // Case-insensitive.
        assert_eq!(DebugLevel::parse_str("STATS"), Some(DebugLevel::Stats));
        assert_eq!(
            DebugLevel::parse_str("Top_Nodes"),
            Some(DebugLevel::TopNodes)
        );
        // Unknown.
        assert_eq!(DebugLevel::parse_str("verbose"), None);
        assert_eq!(DebugLevel::parse_str(""), None);
    }

    #[test]
    fn test_debug_level_serde_roundtrip() {
        for level in [
            DebugLevel::None,
            DebugLevel::Stats,
            DebugLevel::TopNodes,
            DebugLevel::Full,
        ] {
            let json = serde_json::to_string(&level).unwrap();
            let parsed: DebugLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, level, "roundtrip failed for {:?}", level);
        }
    }

    #[test]
    fn test_debug_level_serde_values() {
        assert_eq!(
            serde_json::to_string(&DebugLevel::None).unwrap(),
            r#""none""#
        );
        assert_eq!(
            serde_json::to_string(&DebugLevel::Stats).unwrap(),
            r#""stats""#
        );
        assert_eq!(
            serde_json::to_string(&DebugLevel::TopNodes).unwrap(),
            r#""top_nodes""#
        );
        assert_eq!(
            serde_json::to_string(&DebugLevel::Full).unwrap(),
            r#""full""#
        );
    }

    #[test]
    fn test_debug_level_superset_property() {
        // Each level includes everything the previous levels include.
        let levels = [
            DebugLevel::None,
            DebugLevel::Stats,
            DebugLevel::TopNodes,
            DebugLevel::Full,
        ];
        for (i, &level) in levels.iter().enumerate() {
            // stats is available from level 1+
            assert_eq!(level.includes_stats(), i >= 1);
            // node scores from level 2+
            assert_eq!(level.includes_node_scores(), i >= 2);
            // full from level 3+
            assert_eq!(level.includes_full(), i >= 3);
        }
    }

    // ─── SentenceCandidate tests ──────────────────────────────────────

    #[test]
    fn test_sentence_candidates_basic() {
        use crate::types::{PosTag, Token};

        let mut tokens = vec![
            // Sentence 0: "Machine learning rocks"
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("rocks", "rock", PosTag::Verb, 17, 22, 0, 2),
            // Sentence 1: "Deep networks improve"
            Token::new("Deep", "deep", PosTag::Adjective, 24, 28, 1, 3),
            Token::new("networks", "network", PosTag::Noun, 29, 37, 1, 4),
            Token::new("improve", "improve", PosTag::Verb, 38, 45, 1, 5),
        ];
        // Mark none as stopwords — all lemmas should appear.
        let stream = TokenStream::from_tokens(&tokens);

        let cs = CandidateSet::from_sentence_boundaries(&stream);
        assert!(matches!(cs.kind(), CandidateKind::Sentences(_)));
        assert_eq!(cs.len(), 2);

        let sents = cs.sentences();

        // Sentence 0
        assert_eq!(sents[0].sentence_idx, 0);
        assert_eq!(sents[0].start_token, 0);
        assert_eq!(sents[0].end_token, 3);
        assert_eq!(sents[0].start_char, 0);
        assert_eq!(sents[0].end_char, 22);
        assert_eq!(sents[0].lemma_ids.len(), 3); // machine, learning, rock

        // Sentence 1
        assert_eq!(sents[1].sentence_idx, 1);
        assert_eq!(sents[1].start_token, 3);
        assert_eq!(sents[1].end_token, 6);
        assert_eq!(sents[1].start_char, 24);
        assert_eq!(sents[1].end_char, 45);
        assert_eq!(sents[1].lemma_ids.len(), 3); // deep, network, improve

        // token_len helper
        assert_eq!(sents[0].token_len(), 3);
        assert_eq!(sents[1].token_len(), 3);

        // Stopword filtering: mark "rocks" as stopword, rebuild
        tokens[2].is_stopword = true;
        let stream2 = TokenStream::from_tokens(&tokens);
        let cs2 = CandidateSet::from_sentence_boundaries(&stream2);
        let sents2 = cs2.sentences();
        assert_eq!(sents2[0].lemma_ids.len(), 2); // machine, learning (rock filtered)
    }

    #[test]
    fn test_sentence_candidates_empty() {
        let stream = TokenStream::from_tokens(&[]);
        let cs = CandidateSet::from_sentence_boundaries(&stream);
        assert_eq!(cs.len(), 0);
        assert!(cs.is_empty());
        assert!(cs.sentences().is_empty());
    }

    #[test]
    #[should_panic(expected = "called words() on non-word CandidateSet")]
    fn test_sentence_candidates_accessor_panics() {
        use crate::types::{PosTag, Token};
        let tokens = vec![Token::new("Hello", "hello", PosTag::Noun, 0, 5, 0, 0)];
        let stream = TokenStream::from_tokens(&tokens);
        let cs = CandidateSet::from_sentence_boundaries(&stream);
        let _ = cs.words(); // should panic
    }
}
