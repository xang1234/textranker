//! Stage trait definitions for the pipeline.
//!
//! Each trait represents one processing stage boundary. Implementations are
//! statically dispatched for performance; trait objects are available behind a
//! feature gate for dynamic composition.

use crate::pipeline::artifacts::{
    CandidateKind, CandidateSet, CandidateSetRef, Graph, PhraseCandidate, TeleportVector,
    TokenStream, TokenStreamRef, WordCandidate,
};
use crate::types::{ChunkSpan, PosTag, TextRankConfig};

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

// ============================================================================
// GraphBuilder — candidates + tokens to co-occurrence graph (stage 2)
// ============================================================================

/// Windowing strategy for co-occurrence graph construction.
///
/// Controls whether the sliding window respects sentence boundaries or spans
/// the entire document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowStrategy {
    /// Window slides within each sentence independently (default for
    /// BaseTextRank, PositionRank, BiasedTextRank).
    SentenceBounded,
    /// Window slides across the entire candidate sequence, ignoring sentence
    /// boundaries (used by SingleRank, TopicalPageRank).
    CrossSentence,
}

/// Edge weight policy for co-occurrence graph construction.
///
/// Controls whether repeated co-occurrences within the window accumulate
/// weight or produce binary (0/1) edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeWeightPolicy {
    /// Edge weight is 1.0 if any co-occurrence exists (default for
    /// BaseTextRank).
    Binary,
    /// Edge weight accumulates co-occurrence count (used by SingleRank).
    Count,
}

/// Builds a co-occurrence graph from tokens and pre-selected candidates.
///
/// This is the second processing stage, consuming the token stream and
/// candidate set produced by earlier stages and producing a [`Graph`]
/// artifact (CSR-backed) for downstream ranking.
///
/// Two axes of variation exist among the existing algorithm family:
///
/// - **Window strategy**: sentence-bounded (BaseTextRank) vs cross-sentence
///   (SingleRank / TopicalPageRank) — see [`WindowStrategy`].
/// - **Edge weight policy**: binary (BaseTextRank) vs count-accumulating
///   (SingleRank) — see [`EdgeWeightPolicy`].
///
/// The provided [`CooccurrenceGraphBuilder`] covers both families via
/// configuration. Custom implementations can override this trait for entirely
/// different graph construction strategies (e.g., topic-level graphs for
/// TopicRank).
///
/// # Contract
///
/// - **Input**: a borrowed [`TokenStreamRef`], a [`CandidateSetRef`]
///   (authority on which tokens are graph-eligible), and config.
/// - **Output**: a [`Graph`] wrapping a CSR-backed adjacency + weights.
/// - **Deterministic**: same input → same output (no internal randomness).
/// - **Candidate-driven**: only tokens matching an entry in the candidate
///   set appear as nodes. The graph builder does **not** re-filter by POS;
///   that responsibility belongs to the upstream [`CandidateSelector`].
pub trait GraphBuilder {
    /// Build a graph from the token stream and candidate set.
    fn build(
        &self,
        tokens: TokenStreamRef<'_>,
        candidates: CandidateSetRef<'_>,
        cfg: &TextRankConfig,
    ) -> Graph;
}

/// Windowed co-occurrence graph builder for the word-graph TextRank family.
///
/// This is the primary [`GraphBuilder`] implementation, covering BaseTextRank,
/// PositionRank, BiasedTextRank, SingleRank, and TopicalPageRank through the
/// [`WindowStrategy`] and [`EdgeWeightPolicy`] configuration axes.
///
/// # How it works
///
/// 1. Build a lookup set from the candidate words (fast `O(1)` membership).
/// 2. Walk the token stream in document order, collecting candidate
///    occurrences with their graph keys and sentence indices.
/// 3. Slide a window of size [`TextRankConfig::window_size`] over these
///    occurrences, creating undirected edges in the underlying
///    [`GraphBuilder`](crate::graph::builder::GraphBuilder) (the mutable
///    edge-accumulation struct).
/// 4. Convert to the pipeline [`Graph`] artifact (CSR-backed).
///
/// # Default
///
/// `Default` produces the BaseTextRank configuration: sentence-bounded
/// windowing with binary edge weights.
#[derive(Debug, Clone, Copy)]
pub struct CooccurrenceGraphBuilder {
    /// Windowing behavior (sentence-bounded vs cross-sentence).
    pub window_strategy: WindowStrategy,
    /// Edge weight policy (binary vs count-accumulating).
    pub edge_weight_policy: EdgeWeightPolicy,
}

impl Default for CooccurrenceGraphBuilder {
    fn default() -> Self {
        Self {
            window_strategy: WindowStrategy::SentenceBounded,
            edge_weight_policy: EdgeWeightPolicy::Binary,
        }
    }
}

impl CooccurrenceGraphBuilder {
    /// BaseTextRank configuration: sentence-bounded + binary.
    pub fn base_textrank() -> Self {
        Self::default()
    }

    /// SingleRank configuration: cross-sentence + count-accumulating.
    pub fn single_rank() -> Self {
        Self {
            window_strategy: WindowStrategy::CrossSentence,
            edge_weight_policy: EdgeWeightPolicy::Count,
        }
    }
}

impl GraphBuilder for CooccurrenceGraphBuilder {
    fn build(
        &self,
        tokens: TokenStreamRef<'_>,
        candidates: CandidateSetRef<'_>,
        cfg: &TextRankConfig,
    ) -> Graph {
        use rustc_hash::FxHashSet;

        // Only word-level candidates produce co-occurrence graphs.
        // Phrase-level candidates use topic-graph construction (TopicRank).
        let words = match candidates.kind() {
            CandidateKind::Words(w) => w,
            CandidateKind::Phrases(_) => {
                // Return an empty graph — phrase-family graph construction
                // is a different stage (future TopicRank GraphBuilder).
                let empty = crate::graph::builder::GraphBuilder::new();
                return Graph::from_builder(&empty);
            }
        };

        // Build a fast membership set from the candidate words.
        // Key: (lemma_id, optional POS discriminant).
        let valid_keys: FxHashSet<(u32, Option<PosTag>)> = words
            .iter()
            .map(|w| {
                if cfg.use_pos_in_nodes {
                    (w.lemma_id, Some(w.pos))
                } else {
                    (w.lemma_id, None)
                }
            })
            .collect();

        // Collect candidate token occurrences in document order with graph
        // keys and sentence indices for windowing.
        let mut occurrences: Vec<(u32, String)> = Vec::new(); // (sentence_idx, graph_key)

        for entry in tokens.tokens() {
            let key = if cfg.use_pos_in_nodes {
                (entry.lemma_id, Some(entry.pos))
            } else {
                (entry.lemma_id, None)
            };
            if valid_keys.contains(&key) {
                let graph_key = entry.graph_key(tokens.pool(), cfg.use_pos_in_nodes);
                occurrences.push((entry.sentence_idx, graph_key));
            }
        }

        // Build edges via the mutable GraphBuilder.
        let mut builder = crate::graph::builder::GraphBuilder::with_capacity(valid_keys.len());

        match self.window_strategy {
            WindowStrategy::SentenceBounded => {
                let mut i = 0;
                while i < occurrences.len() {
                    let sent_idx = occurrences[i].0;
                    let sent_start = i;
                    while i < occurrences.len() && occurrences[i].0 == sent_idx {
                        i += 1;
                    }
                    let sent_end = i;

                    for j in sent_start..sent_end {
                        let node_j = builder.get_or_create_node(&occurrences[j].1);
                        let window_end = std::cmp::min(j + cfg.window_size, sent_end);
                        for k in (j + 1)..window_end {
                            let node_k = builder.get_or_create_node(&occurrences[k].1);
                            match self.edge_weight_policy {
                                EdgeWeightPolicy::Binary => {
                                    builder.set_edge(node_j, node_k, 1.0);
                                }
                                EdgeWeightPolicy::Count => {
                                    builder.increment_edge(node_j, node_k, 1.0);
                                }
                            }
                        }
                    }
                }
            }
            WindowStrategy::CrossSentence => {
                for j in 0..occurrences.len() {
                    let node_j = builder.get_or_create_node(&occurrences[j].1);
                    let window_end = std::cmp::min(j + cfg.window_size, occurrences.len());
                    for k in (j + 1)..window_end {
                        let node_k = builder.get_or_create_node(&occurrences[k].1);
                        match self.edge_weight_policy {
                            EdgeWeightPolicy::Binary => {
                                builder.set_edge(node_j, node_k, 1.0);
                            }
                            EdgeWeightPolicy::Count => {
                                builder.increment_edge(node_j, node_k, 1.0);
                            }
                        }
                    }
                }
            }
        }

        Graph::from_builder(&builder)
    }
}

// ============================================================================
// GraphTransform — optional in-place graph modifications (stage 2a)
// ============================================================================

/// Optional in-place graph modification applied after initial construction.
///
/// Transforms mutate the [`Graph`] without rebuilding it — typically adjusting
/// edge weights or removing edges.  Multiple transforms compose as an ordered
/// `Vec<Box<dyn GraphTransform>>`; **order matters** (e.g., intra-cluster
/// edge removal must precede alpha-boost weighting in MultipartiteRank).
///
/// The provided [`NoopGraphTransform`] is the default for variants that don't
/// need post-construction modifications (BaseTextRank, PositionRank, etc.).
///
/// # Contract
///
/// - **Input**: a mutable [`Graph`], borrowed token stream and candidates,
///   and config.
/// - **Output**: none — the graph is mutated in place.
/// - **Side-effect**: calling [`Graph::csr_mut`] automatically sets the
///   `transformed` flag, which observers can inspect.
/// - **Idempotent**: a transform applied twice should produce the same graph
///   as applying it once (where semantically meaningful).
///
/// # Existing transform families (MultipartiteRank)
///
/// - **Intra-cluster edge removal**: removes edges between candidates that
///   belong to the same topic cluster, forming a k-partite graph.
/// - **Alpha-boost weighting**: boosts incoming edges to the first-occurring
///   variant in each topic cluster, encoding positional preference.
pub trait GraphTransform {
    /// Apply the transform to the graph in place.
    fn transform(
        &self,
        graph: &mut Graph,
        tokens: TokenStreamRef<'_>,
        candidates: CandidateSetRef<'_>,
        cfg: &TextRankConfig,
    );
}

/// No-op graph transform — the default for most pipeline configurations.
///
/// Passes the graph through unchanged, with zero overhead.  Used by
/// BaseTextRank, PositionRank, BiasedTextRank, SingleRank, and
/// TopicalPageRank (none of which modify the graph after construction).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopGraphTransform;

impl GraphTransform for NoopGraphTransform {
    #[inline]
    fn transform(
        &self,
        _graph: &mut Graph,
        _tokens: TokenStreamRef<'_>,
        _candidates: CandidateSetRef<'_>,
        _cfg: &TextRankConfig,
    ) {
        // Intentionally empty.
    }
}

// ============================================================================
// TeleportBuilder — optional personalization vector (stage 3a)
// ============================================================================

/// Builds a personalization (teleport) vector for PageRank.
///
/// The teleport vector controls where the random surfer jumps during the
/// teleportation step of PageRank.  It is the key differentiator between:
///
/// - **Standard PageRank** (uniform teleport): all nodes equally likely →
///   builder returns `None`.
/// - **PositionRank**: earlier-occurring tokens get higher teleport
///   probability.
/// - **BiasedTextRank**: user-specified focus terms get boosted teleport
///   probability.
/// - **TopicalPageRank**: per-lemma topic weights drive teleport distribution.
///
/// Separating teleport construction from ranking allows mixing and matching:
/// e.g., SingleRank graph construction + PositionRank teleport strategy.
///
/// # Contract
///
/// - **Input**: a borrowed [`TokenStreamRef`] (read-only), a
///   [`CandidateSetRef`] (authority on which tokens are graph-eligible),
///   and config.
/// - **Output**: `None` for uniform teleport (standard PageRank) or
///   `Some(TeleportVector)` for a personalized distribution.
/// - **Normalization**: when returning `Some`, the vector **must** be
///   normalized (entries sum to 1.0).  The [`Ranker`] stage does **not**
///   re-normalize.
/// - **Length**: the vector length must equal the number of graph nodes
///   (i.e., the number of unique candidates that became nodes).
/// - **Deterministic**: same input → same output (no internal randomness).
///
/// # Default implementation
///
/// [`UniformTeleportBuilder`] returns `None` (uniform teleport), making it
/// the correct default for BaseTextRank and SingleRank.
pub trait TeleportBuilder {
    /// Build a teleport vector from the token stream and candidates.
    ///
    /// Returns `None` for uniform teleport (standard PageRank) or
    /// `Some(TeleportVector)` for personalized PageRank.
    fn build(
        &self,
        tokens: TokenStreamRef<'_>,
        candidates: CandidateSetRef<'_>,
        cfg: &TextRankConfig,
    ) -> Option<TeleportVector>;
}

/// Uniform teleport builder — the default for standard PageRank variants.
///
/// Always returns `None`, signaling to the [`Ranker`] that it should use
/// uniform teleportation (equivalent to standard, non-personalized PageRank).
///
/// Used by BaseTextRank, SingleRank, and any pipeline configuration that
/// does not require personalized ranking.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformTeleportBuilder;

impl TeleportBuilder for UniformTeleportBuilder {
    #[inline]
    fn build(
        &self,
        _tokens: TokenStreamRef<'_>,
        _candidates: CandidateSetRef<'_>,
        _cfg: &TextRankConfig,
    ) -> Option<TeleportVector> {
        None
    }
}

// Future E3 stage traits will be added below:
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

    // ================================================================
    // GraphBuilder — CooccurrenceGraphBuilder tests
    // ================================================================

    /// Helper: build word candidates from a token stream using default config.
    fn word_candidates(stream: &TokenStream, cfg: &TextRankConfig) -> CandidateSet {
        WordNodeSelector.select(stream.as_ref(), cfg)
    }

    #[test]
    fn test_graph_builder_sentence_bounded_binary() {
        // BaseTextRank default: sentence-bounded + binary edges.
        let tokens = rich_tokens(); // 2 sentences, stopwords on "is"
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default(); // window_size = 3
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::base_textrank();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        // Should have nodes for: machine, learning, great, rust, fast (5 candidates).
        assert_eq!(graph.num_nodes(), 5);
        assert!(!graph.is_empty());
        // Edges exist within each sentence, not across.
        assert!(graph.num_edges() > 0);
    }

    #[test]
    fn test_graph_builder_cross_sentence_count() {
        // SingleRank config: cross-sentence + count-accumulating.
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::single_rank();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        assert_eq!(graph.num_nodes(), 5);
        // Cross-sentence: "great" (sent 0 last candidate) should connect to
        // "rust" (sent 1 first candidate) — which doesn't happen in
        // sentence-bounded mode.
        let great_id = graph.get_node_by_lemma("great|ADJ");
        let rust_id = graph.get_node_by_lemma("rust|PROPN");
        assert!(great_id.is_some());
        assert!(rust_id.is_some());

        let neighbors: Vec<u32> = graph
            .neighbors(great_id.unwrap())
            .map(|(id, _)| id)
            .collect();
        assert!(
            neighbors.contains(&rust_id.unwrap()),
            "Cross-sentence edge between 'great' and 'rust' should exist"
        );
    }

    #[test]
    fn test_graph_builder_sentence_bounded_no_cross_sentence() {
        // Sentence-bounded mode should NOT create cross-sentence edges.
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.window_size = 10; // Large window to stress the boundary.
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::base_textrank();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        let great_id = graph.get_node_by_lemma("great|ADJ").unwrap();
        let rust_id = graph.get_node_by_lemma("rust|PROPN").unwrap();

        let neighbors: Vec<u32> = graph.neighbors(great_id).map(|(id, _)| id).collect();
        assert!(
            !neighbors.contains(&rust_id),
            "Sentence-bounded should NOT create cross-sentence edge"
        );
    }

    #[test]
    fn test_graph_builder_empty_tokens() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        assert!(graph.is_empty());
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_graph_builder_empty_candidates() {
        // All tokens are stopwords → no candidates → empty graph.
        let mut tokens = vec![
            Token::new("the", "the", PosTag::Determiner, 0, 3, 0, 0),
            Token::new("is", "be", PosTag::Verb, 4, 6, 0, 1),
        ];
        tokens[0].is_stopword = true;
        tokens[1].is_stopword = true;

        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);
        assert!(cs.is_empty());

        let gb = CooccurrenceGraphBuilder::default();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        assert!(graph.is_empty());
    }

    #[test]
    fn test_graph_builder_binary_no_accumulation() {
        // Same pair co-occurs multiple times → binary should stay at weight 1.0.
        let tokens = vec![
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("machine", "machine", PosTag::Noun, 17, 24, 0, 2),
            Token::new("learning", "learning", PosTag::Noun, 25, 33, 0, 3),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.window_size = 2;
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder {
            window_strategy: WindowStrategy::SentenceBounded,
            edge_weight_policy: EdgeWeightPolicy::Binary,
        };
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        // All edge weights should be 1.0 (binary).
        let machine_id = graph.get_node_by_lemma("machine|NOUN").unwrap();
        for (_, weight) in graph.neighbors(machine_id) {
            assert!(
                (weight - 1.0).abs() < 1e-10,
                "Binary mode: expected weight 1.0, got {weight}"
            );
        }
    }

    #[test]
    fn test_graph_builder_count_accumulates() {
        // Same pair co-occurs multiple times → count should accumulate.
        let tokens = vec![
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("machine", "machine", PosTag::Noun, 17, 24, 0, 2),
            Token::new("learning", "learning", PosTag::Noun, 25, 33, 0, 3),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.window_size = 2;
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder {
            window_strategy: WindowStrategy::SentenceBounded,
            edge_weight_policy: EdgeWeightPolicy::Count,
        };
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        let machine_id = graph.get_node_by_lemma("machine|NOUN").unwrap();
        let learning_id = graph.get_node_by_lemma("learning|NOUN").unwrap();

        // With window_size=2, pairs are: (0,1), (1,2), (2,3) — all are
        // machine↔learning, so weight should be 3.0.
        let weight: f64 = graph
            .neighbors(machine_id)
            .find(|(id, _)| *id == learning_id)
            .map(|(_, w)| w)
            .unwrap_or(0.0);
        assert!(
            (weight - 3.0).abs() < 1e-10,
            "Count mode: expected weight 3.0, got {weight}"
        );
    }

    #[test]
    fn test_graph_builder_single_sentence() {
        let tokens = vec![
            Token::new("hello", "hello", PosTag::Noun, 0, 5, 0, 0),
            Token::new("world", "world", PosTag::Noun, 6, 11, 0, 1),
            Token::new("test", "test", PosTag::Noun, 12, 16, 0, 2),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.window_size = 2;
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        assert_eq!(graph.num_nodes(), 3);
        // Window of 2: (hello,world) and (world,test) → 2 undirected edges
        // = 4 directed entries in CSR.
        assert_eq!(graph.num_edges(), 4);
    }

    #[test]
    fn test_graph_builder_pos_in_nodes() {
        // Same lemma "fast" with different POS → distinct nodes when
        // use_pos_in_nodes is true.
        let tokens = vec![
            Token::new("fast", "fast", PosTag::Adjective, 0, 4, 0, 0),
            Token::new("fast", "fast", PosTag::Adverb, 5, 9, 0, 1),
            Token::new("run", "run", PosTag::Verb, 10, 13, 0, 2),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.include_pos = vec![PosTag::Adjective, PosTag::Adverb, PosTag::Verb];
        cfg.use_pos_in_nodes = true;
        cfg.window_size = 3;
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        // "fast|ADJ", "fast|ADV", "run|VERB" → 3 distinct nodes.
        assert_eq!(graph.num_nodes(), 3);
        assert!(graph.get_node_by_lemma("fast|ADJ").is_some());
        assert!(graph.get_node_by_lemma("fast|ADV").is_some());
        assert!(graph.get_node_by_lemma("run|VERB").is_some());
    }

    #[test]
    fn test_graph_builder_pos_not_in_nodes() {
        // Same lemma "fast" with different POS → single node when
        // use_pos_in_nodes is false.
        let tokens = vec![
            Token::new("fast", "fast", PosTag::Adjective, 0, 4, 0, 0),
            Token::new("fast", "fast", PosTag::Adverb, 5, 9, 0, 1),
            Token::new("run", "run", PosTag::Verb, 10, 13, 0, 2),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.include_pos = vec![PosTag::Adjective, PosTag::Adverb, PosTag::Verb];
        cfg.use_pos_in_nodes = false;
        cfg.window_size = 3;
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        // "fast" (merged) and "run" → 2 nodes.
        assert_eq!(graph.num_nodes(), 2);
        assert!(graph.get_node_by_lemma("fast").is_some());
        assert!(graph.get_node_by_lemma("run").is_some());
    }

    #[test]
    fn test_graph_builder_phrase_candidates_returns_empty() {
        // Phrase-level candidates should produce an empty graph from
        // CooccurrenceGraphBuilder (topic graphs are a separate stage).
        let tokens = vec![
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
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
        let cs = PhraseCandidateSelector::new(chunks).select(stream.as_ref(), &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        assert!(graph.is_empty());
    }

    #[test]
    fn test_graph_builder_as_trait_object() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb: Box<dyn GraphBuilder> = Box::new(CooccurrenceGraphBuilder::default());
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        assert_eq!(graph.num_nodes(), 5);
        assert!(graph.num_edges() > 0);
    }

    #[test]
    fn test_graph_builder_default_is_base_textrank() {
        let gb = CooccurrenceGraphBuilder::default();
        assert_eq!(gb.window_strategy, WindowStrategy::SentenceBounded);
        assert_eq!(gb.edge_weight_policy, EdgeWeightPolicy::Binary);
    }

    #[test]
    fn test_graph_builder_window_size_respected() {
        // Window size 2: only adjacent pairs get edges.
        let tokens = vec![
            Token::new("a", "a", PosTag::Noun, 0, 1, 0, 0),
            Token::new("b", "b", PosTag::Noun, 2, 3, 0, 1),
            Token::new("c", "c", PosTag::Noun, 4, 5, 0, 2),
            Token::new("d", "d", PosTag::Noun, 6, 7, 0, 3),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.window_size = 2;
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        assert_eq!(graph.num_nodes(), 4);
        // Window=2: (a,b), (b,c), (c,d) → 3 undirected edges → 6 directed.
        assert_eq!(graph.num_edges(), 6);

        // "a" should NOT connect to "c" (distance = 2, window = 2 means
        // only offset +1).
        let a_id = graph.get_node_by_lemma("a|NOUN").unwrap();
        let c_id = graph.get_node_by_lemma("c|NOUN").unwrap();
        let a_neighbors: Vec<u32> = graph.neighbors(a_id).map(|(id, _)| id).collect();
        assert!(
            !a_neighbors.contains(&c_id),
            "'a' should not connect to 'c' with window_size=2"
        );
    }

    #[test]
    fn test_graph_builder_cross_sentence_weight_accumulation() {
        // Cross-sentence + count: same pair across sentences accumulates.
        let tokens = vec![
            Token::new("machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("machine", "machine", PosTag::Noun, 17, 24, 1, 2),
            Token::new("learning", "learning", PosTag::Noun, 25, 33, 1, 3),
            Token::new("machine", "machine", PosTag::Noun, 34, 41, 2, 4),
            Token::new("learning", "learning", PosTag::Noun, 42, 50, 2, 5),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.window_size = 2;
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::single_rank();
        let graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        let machine_id = graph.get_node_by_lemma("machine|NOUN").unwrap();
        let learning_id = graph.get_node_by_lemma("learning|NOUN").unwrap();

        // Pairs: (0,1) (1,2) (2,3) (3,4) (4,5) = 5 machine↔learning hits.
        let weight: f64 = graph
            .neighbors(machine_id)
            .find(|(id, _)| *id == learning_id)
            .map(|(_, w)| w)
            .unwrap_or(0.0);
        assert!(
            (weight - 5.0).abs() < 1e-10,
            "Cross-sentence count: expected 5.0, got {weight}"
        );
    }

    // ================================================================
    // GraphTransform — NoopGraphTransform tests
    // ================================================================

    #[test]
    fn test_noop_graph_transform_preserves_graph() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let mut graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        let nodes_before = graph.num_nodes();
        let edges_before = graph.num_edges();
        assert!(!graph.is_transformed());

        NoopGraphTransform.transform(
            &mut graph,
            stream.as_ref(),
            cs.as_ref(),
            &cfg,
        );

        assert_eq!(graph.num_nodes(), nodes_before);
        assert_eq!(graph.num_edges(), edges_before);
        // NoopGraphTransform should NOT set the transformed flag.
        assert!(!graph.is_transformed());
    }

    #[test]
    fn test_noop_graph_transform_is_idempotent() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let mut graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        NoopGraphTransform.transform(&mut graph, stream.as_ref(), cs.as_ref(), &cfg);
        let snapshot_nodes = graph.num_nodes();
        let snapshot_edges = graph.num_edges();

        NoopGraphTransform.transform(&mut graph, stream.as_ref(), cs.as_ref(), &cfg);
        assert_eq!(graph.num_nodes(), snapshot_nodes);
        assert_eq!(graph.num_edges(), snapshot_edges);
    }

    #[test]
    fn test_noop_graph_transform_on_empty_graph() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let mut graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(graph.is_empty());

        NoopGraphTransform.transform(&mut graph, stream.as_ref(), cs.as_ref(), &cfg);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_custom_graph_transform_modifies_graph() {
        /// Test transform that removes all edges from the first node.
        struct RemoveFirstNodeEdges;

        impl GraphTransform for RemoveFirstNodeEdges {
            fn transform(
                &self,
                graph: &mut Graph,
                _tokens: TokenStreamRef<'_>,
                _candidates: CandidateSetRef<'_>,
                _cfg: &TextRankConfig,
            ) {
                if graph.num_nodes() == 0 {
                    return;
                }
                // Zero out weights for node 0's outgoing edges.
                let csr = graph.csr_mut();
                let start = csr.row_ptr[0] as usize;
                let end = csr.row_ptr[1] as usize;
                for i in start..end {
                    csr.weights[i] = 0.0;
                }
            }
        }

        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let mut graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(!graph.is_transformed());

        RemoveFirstNodeEdges.transform(
            &mut graph,
            stream.as_ref(),
            cs.as_ref(),
            &cfg,
        );

        // csr_mut() should have flipped the transformed flag.
        assert!(graph.is_transformed());

        // Node 0's outgoing edges should now have zero weight.
        let weights: Vec<f64> = graph.neighbors(0).map(|(_, w)| w).collect();
        for w in &weights {
            assert!(
                w.abs() < 1e-10,
                "Expected zero weight after transform, got {w}"
            );
        }
    }

    #[test]
    fn test_graph_transform_composition_order() {
        /// Doubles all edge weights.
        struct DoubleWeights;

        impl GraphTransform for DoubleWeights {
            fn transform(
                &self,
                graph: &mut Graph,
                _tokens: TokenStreamRef<'_>,
                _candidates: CandidateSetRef<'_>,
                _cfg: &TextRankConfig,
            ) {
                let csr = graph.csr_mut();
                for w in &mut csr.weights {
                    *w *= 2.0;
                }
            }
        }

        /// Adds 1.0 to all edge weights.
        struct AddOneToWeights;

        impl GraphTransform for AddOneToWeights {
            fn transform(
                &self,
                graph: &mut Graph,
                _tokens: TokenStreamRef<'_>,
                _candidates: CandidateSetRef<'_>,
                _cfg: &TextRankConfig,
            ) {
                let csr = graph.csr_mut();
                for w in &mut csr.weights {
                    *w += 1.0;
                }
            }
        }

        let tokens = vec![
            Token::new("hello", "hello", PosTag::Noun, 0, 5, 0, 0),
            Token::new("world", "world", PosTag::Noun, 6, 11, 0, 1),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.window_size = 2;
        let cs = word_candidates(&stream, &cfg);

        // Order A: double then add1 → 1.0 * 2 + 1 = 3.0
        let gb = CooccurrenceGraphBuilder::default();
        let mut graph_a = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        let transforms_a: Vec<Box<dyn GraphTransform>> = vec![
            Box::new(DoubleWeights),
            Box::new(AddOneToWeights),
        ];
        for t in &transforms_a {
            t.transform(&mut graph_a, stream.as_ref(), cs.as_ref(), &cfg);
        }

        let weight_a: f64 = graph_a.neighbors(0).map(|(_, w)| w).next().unwrap();
        assert!(
            (weight_a - 3.0).abs() < 1e-10,
            "double→add1: expected 3.0, got {weight_a}"
        );

        // Order B: add1 then double → (1.0 + 1) * 2 = 4.0
        let mut graph_b = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        let transforms_b: Vec<Box<dyn GraphTransform>> = vec![
            Box::new(AddOneToWeights),
            Box::new(DoubleWeights),
        ];
        for t in &transforms_b {
            t.transform(&mut graph_b, stream.as_ref(), cs.as_ref(), &cfg);
        }

        let weight_b: f64 = graph_b.neighbors(0).map(|(_, w)| w).next().unwrap();
        assert!(
            (weight_b - 4.0).abs() < 1e-10,
            "add1→double: expected 4.0, got {weight_b}"
        );

        // Confirm order matters.
        assert!(
            (weight_a - weight_b).abs() > 0.5,
            "Transform order should produce different results"
        );
    }

    #[test]
    fn test_graph_transform_as_trait_object() {
        let transform: Box<dyn GraphTransform> = Box::new(NoopGraphTransform);

        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder::default();
        let mut graph = gb.build(stream.as_ref(), cs.as_ref(), &cfg);

        transform.transform(&mut graph, stream.as_ref(), cs.as_ref(), &cfg);
        assert_eq!(graph.num_nodes(), 5);
    }

    #[test]
    fn test_noop_graph_transform_default() {
        let _t = NoopGraphTransform::default();
    }

    // ================================================================
    // TeleportBuilder — UniformTeleportBuilder tests
    // ================================================================

    #[test]
    fn test_uniform_teleport_returns_none() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let result = UniformTeleportBuilder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_uniform_teleport_empty_stream() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let result = UniformTeleportBuilder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_uniform_teleport_default() {
        let _tb = UniformTeleportBuilder::default();
    }

    #[test]
    fn test_teleport_builder_as_trait_object() {
        let builder: Box<dyn TeleportBuilder> = Box::new(UniformTeleportBuilder);

        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let result = builder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_custom_teleport_builder_returns_vector() {
        /// A test builder that assigns weight based on first_position.
        struct PositionTestBuilder;

        impl TeleportBuilder for PositionTestBuilder {
            fn build(
                &self,
                _tokens: TokenStreamRef<'_>,
                candidates: CandidateSetRef<'_>,
                _cfg: &TextRankConfig,
            ) -> Option<TeleportVector> {
                let words = match candidates.kind() {
                    CandidateKind::Words(w) => w,
                    CandidateKind::Phrases(_) => return None,
                };
                if words.is_empty() {
                    return None;
                }

                let mut tv = TeleportVector::zeros(words.len());
                for (i, w) in words.iter().enumerate() {
                    tv.set(i, 1.0 / (w.first_position as f64 + 1.0));
                }
                tv.normalize();
                Some(tv)
            }
        }

        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let result = PositionTestBuilder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_some());

        let tv = result.unwrap();
        assert_eq!(tv.len(), cs.len());
        assert!(tv.is_normalized(1e-10));

        // Earlier positions should have higher weight.
        // First candidate (position 0) should have highest weight.
        let max_val = tv.as_slice().iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((tv[0] - max_val).abs() < 1e-10);
    }

    #[test]
    fn test_custom_teleport_builder_focus_terms() {
        /// A test builder that boosts specific lemma IDs.
        struct FocusTestBuilder {
            focus_lemma_ids: Vec<u32>,
            bias: f64,
        }

        impl TeleportBuilder for FocusTestBuilder {
            fn build(
                &self,
                _tokens: TokenStreamRef<'_>,
                candidates: CandidateSetRef<'_>,
                _cfg: &TextRankConfig,
            ) -> Option<TeleportVector> {
                let words = match candidates.kind() {
                    CandidateKind::Words(w) => w,
                    CandidateKind::Phrases(_) => return None,
                };
                if words.is_empty() {
                    return None;
                }

                let mut tv = TeleportVector::new(vec![1.0; words.len()]);
                for (i, w) in words.iter().enumerate() {
                    if self.focus_lemma_ids.contains(&w.lemma_id) {
                        tv.set(i, self.bias);
                    }
                }
                tv.normalize();
                Some(tv)
            }
        }

        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        // Find lemma_id for "machine".
        let machine_lemma_id = cs.words().iter()
            .find(|w| stream.pool().get(w.lemma_id) == Some("machine"))
            .map(|w| w.lemma_id)
            .unwrap();

        let builder = FocusTestBuilder {
            focus_lemma_ids: vec![machine_lemma_id],
            bias: 10.0,
        };

        let result = builder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_some());

        let tv = result.unwrap();
        assert!(tv.is_normalized(1e-10));

        // The focused term should have higher probability than non-focused.
        let machine_idx = cs.words().iter()
            .position(|w| w.lemma_id == machine_lemma_id)
            .unwrap();
        let non_focus_idx = cs.words().iter()
            .position(|w| w.lemma_id != machine_lemma_id)
            .unwrap();
        assert!(
            tv[machine_idx] > tv[non_focus_idx],
            "Focus term should have higher teleport probability"
        );
    }

    #[test]
    fn test_custom_teleport_builder_as_trait_object() {
        /// Trivial builder returning a uniform vector for testing dynamic dispatch.
        struct UniformExplicitBuilder;

        impl TeleportBuilder for UniformExplicitBuilder {
            fn build(
                &self,
                _tokens: TokenStreamRef<'_>,
                candidates: CandidateSetRef<'_>,
                _cfg: &TextRankConfig,
            ) -> Option<TeleportVector> {
                let n = candidates.len();
                if n == 0 {
                    return None;
                }
                Some(TeleportVector::uniform(n))
            }
        }

        let builder: Box<dyn TeleportBuilder> = Box::new(UniformExplicitBuilder);

        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let result = builder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_some());

        let tv = result.unwrap();
        assert_eq!(tv.len(), 5);
        assert!(tv.is_normalized(1e-10));

        // All values should be equal (uniform).
        let expected = 1.0 / 5.0;
        for &v in tv.as_slice() {
            assert!((v - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_teleport_builder_with_phrase_candidates() {
        // TeleportBuilder should work with phrase candidates too
        // (though most implementations will return None for phrases).
        let tokens = vec![
            Token::new("big", "big", PosTag::Adjective, 0, 3, 0, 0),
            Token::new("cat", "cat", PosTag::Noun, 4, 7, 0, 1),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();

        let chunks = vec![ChunkSpan {
            start_token: 0, end_token: 2, start_char: 0, end_char: 7, sentence_idx: 0,
        }];
        let cs = PhraseCandidateSelector::new(chunks).select(stream.as_ref(), &cfg);

        let result = UniformTeleportBuilder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_none());
    }
}
