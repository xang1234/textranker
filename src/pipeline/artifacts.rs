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

/// Distinguishes the two candidate families.
///
/// Downstream stages (GraphBuilder, TeleportBuilder, etc.) can match on this
/// to select the appropriate processing strategy.
#[derive(Debug, Clone)]
pub enum CandidateKind {
    /// Word-level candidates (one per unique graph key).
    Words(Vec<WordCandidate>),
    /// Phrase-level candidates (one per noun chunk).
    Phrases(Vec<PhraseCandidate>),
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

            if !seen.contains_key(&key) {
                seen.insert(key, words.len());
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
    pub fn from_phrase_chunks(
        stream: &TokenStream,
        chunks: &[crate::types::ChunkSpan],
    ) -> Self {
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

    /// Access word candidates (panics if phrase-level).
    #[inline]
    pub fn words(&self) -> &[WordCandidate] {
        match &self.kind {
            CandidateKind::Words(w) => w,
            CandidateKind::Phrases(_) => panic!("called words() on phrase-level CandidateSet"),
        }
    }

    /// Access phrase candidates (panics if word-level).
    #[inline]
    pub fn phrases(&self) -> &[PhraseCandidate] {
        match &self.kind {
            CandidateKind::Phrases(p) => p,
            CandidateKind::Words(_) => panic!("called phrases() on word-level CandidateSet"),
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
            CandidateKind::Phrases(_) => panic!("called words() on phrase-level CandidateSetRef"),
        }
    }

    /// Access phrase candidates.
    #[inline]
    pub fn phrases(&self) -> &'a [PhraseCandidate] {
        match self.kind {
            CandidateKind::Phrases(p) => p,
            CandidateKind::Words(_) => panic!("called phrases() on word-level CandidateSetRef"),
        }
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
}

impl Graph {
    /// Build from an existing [`GraphBuilder`].
    ///
    /// This is the primary construction path — the existing graph builder
    /// infrastructure produces a `GraphBuilder`, and this method converts it
    /// to the pipeline artifact via [`CsrGraph::from_builder`].
    pub fn from_builder(builder: &crate::graph::builder::GraphBuilder) -> Self {
        Self {
            csr: crate::graph::csr::CsrGraph::from_builder(builder),
            transformed: false,
        }
    }

    /// Wrap a pre-existing [`CsrGraph`].
    pub fn from_csr(csr: crate::graph::csr::CsrGraph) -> Self {
        Self {
            csr,
            transformed: false,
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
    /// Construct from the existing `Vec<Phrase>` type.
    ///
    /// Interns lemma tokens via the provided pool and eagerly stores the
    /// surface/lemma strings (since they're already materialized in the
    /// legacy path).
    pub fn from_phrases(phrases: &[crate::types::Phrase], pool: &mut StringPool) -> Self {
        let entries = phrases
            .iter()
            .map(|p| {
                let lemma_ids: Vec<u32> = p
                    .lemma
                    .split_whitespace()
                    .map(|w| pool.intern(w))
                    .collect();
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
}

/// Optional debug information attached to a [`FormattedResult`].
///
/// Power users can request this via the `expose` config key.  Fields are
/// individually optional so callers pay only for what they ask for.
#[derive(Debug, Clone, Default)]
pub struct DebugPayload {
    /// Top-K node scores (node lemma → score).
    pub node_scores: Option<Vec<(String, f64)>>,
    /// Graph statistics.
    pub graph_stats: Option<GraphStats>,
    /// Per-stage timing in milliseconds.
    pub stage_timings: Option<Vec<(String, f64)>>,
    /// PageRank convergence residuals.
    pub residuals: Option<Vec<f64>>,
}

/// Summary statistics for the co-occurrence graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl FormattedResult {
    /// Construct from the existing `ExtractionResult`.
    pub fn from_extraction(er: &crate::phrase::extraction::ExtractionResult) -> Self {
        Self {
            phrases: er.phrases.clone(),
            converged: er.converged,
            iterations: er.iterations as u32,
            debug: None,
        }
    }

    /// Build directly.
    pub fn new(
        phrases: Vec<crate::types::Phrase>,
        converged: bool,
        iterations: u32,
    ) -> Self {
        Self {
            phrases,
            converged,
            iterations,
            debug: None,
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
    pub fn with_capacity(
        edge_cap: usize,
        node_cap: usize,
        phrase_cap: usize,
    ) -> Self {
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
        let cs_both = CandidateSet::from_word_tokens(
            &stream,
            &[PosTag::Adjective, PosTag::Adverb],
            true,
        );
        assert_eq!(cs_both.len(), 2);

        // Without POS in nodes: same lemma → deduplicated to one.
        let cs_no_pos = CandidateSet::from_word_tokens(
            &stream,
            &[PosTag::Adjective, PosTag::Adverb],
            false,
        );
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
        let great = words.iter().find(|w| {
            stream.pool().get(w.lemma_id) == Some("great")
        }).unwrap();
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

        let chunks = vec![
            crate::types::ChunkSpan {
                start_token: 0,
                end_token: 2,
                start_char: 0,
                end_char: 16,
                sentence_idx: 0,
            },
        ];

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
    #[should_panic(expected = "called phrases() on word-level")]
    fn test_words_panics_on_phrases_access() {
        let tokens = sample_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cs = CandidateSet::from_word_tokens(&stream, &[], true);
        let _ = cs.phrases(); // should panic
    }

    #[test]
    #[should_panic(expected = "called words() on phrase-level")]
    fn test_phrases_panics_on_words_access() {
        let stream = TokenStream::from_tokens(&[
            Token::new("a", "a", PosTag::Noun, 0, 1, 0, 0),
        ]);
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
            }),
            stage_timings: None,
            residuals: None,
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
}
