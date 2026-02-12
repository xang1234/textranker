//! Stage trait definitions for the pipeline.
//!
//! Each trait represents one processing stage boundary. Implementations are
//! statically dispatched for performance; trait objects are available behind a
//! feature gate for dynamic composition.

use crate::pipeline::artifacts::{
    CandidateKind, CandidateSet, CandidateSetRef, DebugPayload, FormattedResult, Graph,
    PhraseCandidate, PhraseSet, RankOutput, TeleportType, TeleportVector, TokenStream,
    TokenStreamRef, WordCandidate,
};
use crate::types::{ChunkSpan, PosTag, TextRankConfig};
use serde::{Deserialize, Serialize};

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
/// the entire document, and embeds the configurable window size.
///
/// # Serde
///
/// Serializes with an internally-tagged representation (`"type"` discriminator):
///
/// ```json
/// { "type": "sentence_bounded", "window_size": 3 }
/// { "type": "cross_sentence",   "window_size": 5 }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WindowStrategy {
    /// Window slides within each sentence independently (default for
    /// BaseTextRank, PositionRank, BiasedTextRank).
    SentenceBounded {
        /// Number of tokens ahead to consider for co-occurrence edges.
        window_size: usize,
    },
    /// Window slides across the entire candidate sequence, ignoring sentence
    /// boundaries (used by SingleRank, TopicalPageRank).
    CrossSentence {
        /// Number of tokens ahead to consider for co-occurrence edges.
        window_size: usize,
    },
}

/// Default window size used when constructing a `WindowStrategy` without an
/// explicit size (matches [`TextRankConfig::default().window_size`]).
pub const DEFAULT_WINDOW_SIZE: usize = 3;

impl Default for WindowStrategy {
    /// Sentence-bounded with the default window size (3).
    fn default() -> Self {
        Self::SentenceBounded {
            window_size: DEFAULT_WINDOW_SIZE,
        }
    }
}

impl WindowStrategy {
    /// Return the window size embedded in this strategy.
    pub fn window_size(&self) -> usize {
        match self {
            Self::SentenceBounded { window_size } | Self::CrossSentence { window_size } => {
                *window_size
            }
        }
    }

    /// Returns `true` if this is the sentence-bounded variant.
    pub fn is_sentence_bounded(&self) -> bool {
        matches!(self, Self::SentenceBounded { .. })
    }

    /// Returns `true` if this is the cross-sentence variant.
    pub fn is_cross_sentence(&self) -> bool {
        matches!(self, Self::CrossSentence { .. })
    }
}

/// Edge weight policy for co-occurrence graph construction.
///
/// Controls whether repeated co-occurrences within the window accumulate
/// weight or produce binary (0/1) edges.
///
/// # Serde
///
/// Serializes as a lowercase string:
///
/// ```json
/// "binary"
/// "count_accumulating"
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeWeightPolicy {
    /// Edge weight is 1.0 if any co-occurrence exists (default for
    /// BaseTextRank). Duplicate co-occurrences within the window are
    /// idempotent — the edge is set, not incremented.
    Binary,
    /// Edge weight accumulates (+= 1.0) for each co-occurrence within the
    /// window (used by SingleRank). Multiple hits between the same pair
    /// produce higher weights, capturing co-occurrence frequency.
    CountAccumulating,
}

impl Default for EdgeWeightPolicy {
    /// Binary is the default (BaseTextRank behavior).
    fn default() -> Self {
        Self::Binary
    }
}

impl EdgeWeightPolicy {
    /// Returns `true` if this is the binary (idempotent) policy.
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary)
    }

    /// Returns `true` if this is the count-accumulating policy.
    pub fn is_count_accumulating(&self) -> bool {
        matches!(self, Self::CountAccumulating)
    }
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
/// The provided [`WindowGraphBuilder`] covers both families via
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

/// Composable windowed graph builder for the word-graph TextRank family.
///
/// Combines a [`WindowStrategy`] (sentence-bounded vs cross-sentence, with
/// embedded window size) and an [`EdgeWeightPolicy`] (binary vs
/// count-accumulating) into a single parameterized [`GraphBuilder`].
///
/// This is the primary [`GraphBuilder`] implementation, covering BaseTextRank,
/// PositionRank, BiasedTextRank, SingleRank, and TopicalPageRank through its
/// two configuration axes.
///
/// # How it works
///
/// 1. Build a lookup set from the candidate words (fast `O(1)` membership).
/// 2. Walk the token stream in document order, collecting candidate
///    occurrences with their graph keys and sentence indices.
/// 3. Slide a window of size [`WindowStrategy::window_size()`] over these
///    occurrences, creating undirected edges in the underlying
///    [`GraphBuilder`](crate::graph::builder::GraphBuilder) (the mutable
///    edge-accumulation struct).
/// 4. Convert to the pipeline [`Graph`] artifact (CSR-backed).
///
/// # Presets
///
/// - [`WindowGraphBuilder::base_textrank()`] — sentence-bounded + binary
/// - [`WindowGraphBuilder::single_rank()`] — cross-sentence + count-accumulating
///
/// # Default
///
/// `Default` produces the BaseTextRank configuration: sentence-bounded
/// windowing with binary edge weights.
#[derive(Debug, Clone, Copy)]
pub struct WindowGraphBuilder {
    /// Windowing behavior (sentence-bounded vs cross-sentence).
    pub window_strategy: WindowStrategy,
    /// Edge weight policy (binary vs count-accumulating).
    pub edge_weight_policy: EdgeWeightPolicy,
}

/// Backward-compatible alias for [`WindowGraphBuilder`].
pub type CooccurrenceGraphBuilder = WindowGraphBuilder;

impl Default for WindowGraphBuilder {
    fn default() -> Self {
        Self {
            window_strategy: WindowStrategy::default(),
            edge_weight_policy: EdgeWeightPolicy::default(),
        }
    }
}

impl WindowGraphBuilder {
    /// BaseTextRank configuration: sentence-bounded + count-accumulating.
    ///
    /// This matches the library's default behavior (`use_edge_weights = true`
    /// in [`TextRankConfig`]). Co-occurrence counts accumulate as edge
    /// weights, even within sentence-bounded windows.
    ///
    /// For the paper-standard binary-edge variant, construct with
    /// [`EdgeWeightPolicy::Binary`] explicitly.
    pub fn base_textrank() -> Self {
        Self {
            window_strategy: WindowStrategy::default(),
            edge_weight_policy: EdgeWeightPolicy::CountAccumulating,
        }
    }

    /// SingleRank configuration: cross-sentence + count-accumulating.
    pub fn single_rank() -> Self {
        Self {
            window_strategy: WindowStrategy::CrossSentence {
                window_size: DEFAULT_WINDOW_SIZE,
            },
            edge_weight_policy: EdgeWeightPolicy::CountAccumulating,
        }
    }
}

impl GraphBuilder for WindowGraphBuilder {
    fn build(
        &self,
        tokens: TokenStreamRef<'_>,
        candidates: CandidateSetRef<'_>,
        cfg: &TextRankConfig,
    ) -> Graph {
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

        // Pre-compute graph key strings once per unique candidate.
        // This avoids repeated `format!("{}|{}", lemma, pos)` allocations
        // for every token occurrence.
        use rustc_hash::FxHashMap;
        let key_strings: FxHashMap<(u32, Option<PosTag>), String> = words
            .iter()
            .map(|w| {
                let lookup = if cfg.use_pos_in_nodes {
                    (w.lemma_id, Some(w.pos))
                } else {
                    (w.lemma_id, None)
                };
                let gk = w.graph_key(tokens.pool(), cfg.use_pos_in_nodes);
                (lookup, gk)
            })
            .collect();

        // Collect candidate token occurrences in document order with graph
        // keys and sentence indices for windowing.
        let mut occurrences: Vec<(u32, &str)> = Vec::new();

        for entry in tokens.tokens() {
            let key = if cfg.use_pos_in_nodes {
                (entry.lemma_id, Some(entry.pos))
            } else {
                (entry.lemma_id, None)
            };
            if let Some(gk) = key_strings.get(&key) {
                occurrences.push((entry.sentence_idx, gk.as_str()));
            }
        }

        // Build edges via the mutable GraphBuilder.
        let mut builder = crate::graph::builder::GraphBuilder::with_capacity(key_strings.len());

        let window_size = self.window_strategy.window_size();

        match self.window_strategy {
            WindowStrategy::SentenceBounded { .. } => {
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
                        let window_end = std::cmp::min(j + window_size, sent_end);
                        for k in (j + 1)..window_end {
                            let node_k = builder.get_or_create_node(&occurrences[k].1);
                            match self.edge_weight_policy {
                                EdgeWeightPolicy::Binary => {
                                    builder.set_edge(node_j, node_k, 1.0);
                                }
                                EdgeWeightPolicy::CountAccumulating => {
                                    builder.increment_edge(node_j, node_k, 1.0);
                                }
                            }
                        }
                    }
                }
            }
            WindowStrategy::CrossSentence { .. } => {
                for j in 0..occurrences.len() {
                    let node_j = builder.get_or_create_node(&occurrences[j].1);
                    let window_end = std::cmp::min(j + window_size, occurrences.len());
                    for k in (j + 1)..window_end {
                        let node_k = builder.get_or_create_node(&occurrences[k].1);
                        match self.edge_weight_policy {
                            EdgeWeightPolicy::Binary => {
                                builder.set_edge(node_j, node_k, 1.0);
                            }
                            EdgeWeightPolicy::CountAccumulating => {
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

/// Position-biased teleport builder for PositionRank.
///
/// Assigns teleport probability inversely proportional to each candidate's
/// first occurrence position in the document: `weight = 1 / (position + 1)`.
/// Earlier candidates receive higher teleport probability, biasing PageRank
/// towards terms that appear near the start of the text.
///
/// Returns `None` for phrase-level candidates or empty candidate sets
/// (falling back to uniform teleportation).
#[derive(Debug, Clone, Copy, Default)]
pub struct PositionTeleportBuilder;

impl TeleportBuilder for PositionTeleportBuilder {
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

        let mut tv = TeleportVector::zeros(words.len(), TeleportType::Position);
        for (i, w) in words.iter().enumerate() {
            tv.set(i, 1.0 / (w.first_position as f64 + 1.0));
        }
        tv.normalize();
        Some(tv)
    }
}

// ---------------------------------------------------------------------------
// FocusTermsTeleportBuilder — boosts focus-term nodes (BiasedTextRank)
// ---------------------------------------------------------------------------

/// Assigns higher teleport probability to candidates whose lemma matches one
/// of the specified focus terms. Non-focus candidates receive a small uniform
/// base weight (`1.0`), while focus candidates receive `bias_weight`.
///
/// The resulting vector is normalized so the values form a valid probability
/// distribution.
///
/// Returns `None` for phrase-level candidates or empty candidate sets.
#[derive(Debug, Clone)]
pub struct FocusTermsTeleportBuilder {
    /// Focus terms (lowercased / lemmatized).
    focus_terms: Vec<String>,
    /// Weight multiplier for focus-term nodes (typically >> 1.0).
    bias_weight: f64,
}

impl FocusTermsTeleportBuilder {
    /// Create a new builder with the given focus terms and bias weight.
    ///
    /// `focus_terms` should be lemmatized to match the candidate pool.
    /// `bias_weight` is the relative weight for focus nodes vs the base weight
    /// of `1.0` for non-focus nodes — e.g. `5.0` means focus nodes are 5x
    /// more likely as teleportation targets.
    pub fn new(focus_terms: Vec<String>, bias_weight: f64) -> Self {
        Self {
            focus_terms,
            bias_weight,
        }
    }
}

impl TeleportBuilder for FocusTermsTeleportBuilder {
    fn build(
        &self,
        tokens: TokenStreamRef<'_>,
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

        let pool = tokens.pool();
        let mut tv = TeleportVector::zeros(words.len(), TeleportType::Focus);

        for (i, w) in words.iter().enumerate() {
            let lemma = pool.get(w.lemma_id).unwrap_or("");
            let is_focus = self.focus_terms.iter().any(|ft| ft == lemma);
            tv.set(i, if is_focus { self.bias_weight } else { 1.0 });
        }

        tv.normalize();
        Some(tv)
    }
}

// ============================================================================
// Ranker — PageRank / Personalized PageRank execution (stage 3)
// ============================================================================

/// Executes PageRank (or a variant) on a graph, optionally using a
/// personalization (teleport) vector.
///
/// This is the core ranking stage: it takes the co-occurrence [`Graph`] built
/// by earlier stages, an optional [`TeleportVector`] produced by a
/// [`TeleportBuilder`], and config, and returns a [`RankOutput`] containing
/// per-node scores and convergence metadata.
///
/// The teleport vector determines the ranking flavor:
///
/// - `None` → standard PageRank (uniform teleportation).
/// - `Some(tv)` → Personalized PageRank, where the surfer jumps to node `i`
///   with probability `tv[i]` during teleportation.
///
/// # Contract
///
/// - **Input**: an immutable [`Graph`] reference, an optional
///   [`TeleportVector`] reference, and config.
/// - **Output**: a [`RankOutput`] with per-node scores, convergence flag,
///   iteration count, and final delta.
/// - **Config**: reads `damping`, `max_iterations`, and
///   `convergence_threshold` from [`TextRankConfig`].
/// - **Deterministic**: same graph + teleport + config → same output.
/// - **Scores**: normalized (sum to 1.0) and indexed by CSR node ID.
pub trait Ranker {
    /// Rank nodes in the graph.
    fn rank(
        &self,
        graph: &Graph,
        teleport: Option<&TeleportVector>,
        cfg: &TextRankConfig,
    ) -> RankOutput;
}

/// PageRank-based ranker — the default (and currently only) [`Ranker`]
/// implementation.
///
/// Handles both standard and personalized PageRank through a single struct.
/// When `teleport` is `None`, runs standard PageRank (uniform teleportation).
/// When `teleport` is `Some(tv)`, runs Personalized PageRank using `tv` as
/// the teleport distribution.
///
/// Config parameters (damping, max_iterations, convergence_threshold) are read
/// from [`TextRankConfig`] at call time, making this struct stateless and
/// zero-sized — ideal for static pipeline composition.
///
/// # Examples
///
/// ```ignore
/// let ranker = PageRankRanker;
/// // Standard PageRank:
/// let output = ranker.rank(&graph, None, &cfg);
/// // Personalized PageRank:
/// let output = ranker.rank(&graph, Some(&teleport_vec), &cfg);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct PageRankRanker;

impl Ranker for PageRankRanker {
    fn rank(
        &self,
        graph: &Graph,
        teleport: Option<&TeleportVector>,
        cfg: &TextRankConfig,
    ) -> RankOutput {
        let csr = graph.csr();

        let result = match teleport {
            None => {
                // Standard PageRank — uniform teleportation.
                crate::pagerank::standard::StandardPageRank {
                    damping: cfg.damping,
                    max_iterations: cfg.max_iterations,
                    threshold: cfg.convergence_threshold,
                }
                .run(csr)
            }
            Some(tv) => {
                // Personalized PageRank — use the provided teleport vector.
                crate::pagerank::personalized::PersonalizedPageRank::new()
                    .with_damping(cfg.damping)
                    .with_max_iterations(cfg.max_iterations)
                    .with_threshold(cfg.convergence_threshold)
                    .with_personalization(tv.as_slice().to_vec())
                    .run(csr)
            }
        };

        RankOutput::from_pagerank_result(&result)
    }
}

// ============================================================================
// ResultFormatter — phrases + metadata → public output (stage 5)
// ============================================================================

/// Formats internal pipeline artifacts into the public [`FormattedResult`].
///
/// This is the **formatting boundary** — the last stage of the pipeline.
/// Everything before this point uses interned IDs and internal types;
/// the formatter materializes strings, attaches convergence metadata, and
/// optionally appends debug information.
///
/// # Contract
///
/// - **Input**: a [`PhraseSet`] (scored phrases from the phrase builder),
///   a [`RankOutput`] (convergence metadata), an optional [`DebugPayload`],
///   and config.
/// - **Output**: a [`FormattedResult`] — the stable public contract exposed
///   to Python and JSON consumers.
/// - **Deterministic**: same input → same output.
///
/// # Implementations
///
/// - **[`StandardResultFormatter`]**: Preserves the existing `FormattedResult`
///   format exactly — converts `PhraseEntry` to `Phrase`, populates
///   convergence fields, and attaches debug payload when present.
pub trait ResultFormatter {
    /// Format pipeline artifacts into the public output.
    fn format(
        &self,
        phrases: &PhraseSet,
        ranks: &RankOutput,
        debug: Option<DebugPayload>,
        cfg: &TextRankConfig,
    ) -> FormattedResult;
}

/// Standard result formatter — the default for all pipeline configurations.
///
/// Converts [`PhraseEntry`] items into [`Phrase`] objects, attaches convergence
/// metadata from [`RankOutput`], and passes through the optional debug payload.
///
/// The output preserves today's public `FormattedResult` format exactly:
///
/// - Phrases are ordered by score descending (as produced by PhraseBuilder).
/// - `converged` and `iterations` come from PageRank output.
/// - Debug information is attached only when provided.
///
/// This is zero-sized because all configuration is read from [`TextRankConfig`]
/// at call time.
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardResultFormatter;

impl ResultFormatter for StandardResultFormatter {
    fn format(
        &self,
        phrases: &PhraseSet,
        ranks: &RankOutput,
        debug: Option<DebugPayload>,
        _cfg: &TextRankConfig,
    ) -> FormattedResult {
        use crate::types::Phrase;

        let formatted_phrases: Vec<Phrase> = phrases
            .entries()
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let text = entry.surface.clone().unwrap_or_default();
                let lemma = entry.lemma_text.clone().unwrap_or_default();
                let offsets = entry
                    .spans
                    .as_ref()
                    .map(|spans| {
                        spans
                            .iter()
                            .map(|&(s, e)| (s as usize, e as usize))
                            .collect()
                    })
                    .unwrap_or_default();

                Phrase {
                    text,
                    lemma,
                    score: entry.score,
                    count: entry.count as usize,
                    offsets,
                    rank: idx + 1, // 1-indexed rank
                }
            })
            .collect();

        let result = FormattedResult::new(
            formatted_phrases,
            ranks.converged(),
            ranks.iterations(),
        );

        match debug {
            Some(d) => result.with_debug(d),
            None => result,
        }
    }
}

// ============================================================================
// PhraseBuilder — token + rank → scored phrases (stage 4)
// ============================================================================

/// Builds scored phrases from tokens, candidates, graph, and PageRank output.
///
/// This is stage 4 of the pipeline — it takes the ranked graph and produces
/// a [`PhraseSet`] of scored phrases ready for formatting.  Different
/// implementations handle different phrase-building strategies:
///
/// - **[`ChunkPhraseBuilder`]**: Standard noun-chunking + score aggregation +
///   overlap resolution + grouping.  Used by BaseTextRank, PositionRank,
///   BiasedTextRank, SingleRank, and TopicalPageRank.
///
/// - **Topic representative selection** (future): TopicRank selects the
///   first-occurring phrase from each scored cluster.
///
/// - **Direct candidate extraction** (future): MultipartiteRank returns the
///   top-scoring candidates directly from the ranked candidate graph.
///
/// # Contract
///
/// - **Input**: a borrowed [`TokenStreamRef`] (read-only), a
///   [`CandidateSetRef`], the [`RankOutput`] from PageRank, an immutable
///   [`Graph`] reference (for node-key-to-score mapping), and config.
/// - **Output**: a [`PhraseSet`] containing scored, deduplicated phrases.
/// - **Deterministic**: same input → same output (no internal randomness).
/// - **Config-driven**: reads `min_phrase_length`, `max_phrase_length`,
///   `score_aggregation`, `phrase_grouping`, and `top_n` from
///   [`TextRankConfig`].
pub trait PhraseBuilder {
    /// Build scored phrases from ranked graph data.
    fn build(
        &self,
        tokens: TokenStreamRef<'_>,
        candidates: CandidateSetRef<'_>,
        ranks: &RankOutput,
        graph: &Graph,
        cfg: &TextRankConfig,
    ) -> PhraseSet;
}

/// Standard chunk-based phrase builder for the word-graph TextRank family.
///
/// Implements the canonical TextRank phrase extraction pipeline:
///
/// 1. **Noun chunking**: extract candidate phrases using the pattern
///    `(DET)? (ADJ)* (NOUN|PROPN)+`, respecting sentence boundaries and
///    stopword-based chunk breaks.
/// 2. **Chunk scoring**: aggregate PageRank scores of the constituent
///    tokens in each chunk using the configured [`ScoreAggregation`]
///    strategy (sum, mean, max, or RMS).
/// 3. **Overlap resolution**: greedily select the highest-scoring
///    non-overlapping chunks.
/// 4. **Variant grouping**: group surface-form variants by lemma or
///    scrubbed text (controlled by [`PhraseGrouping`]), selecting a
///    canonical form and aggregating counts/offsets.
/// 5. **Ranking**: sort by score descending and apply `top_n` limit.
///
/// This is the default phrase builder used by BaseTextRank, PositionRank,
/// BiasedTextRank, SingleRank, and TopicalPageRank.  It is zero-sized
/// because all configuration is read from [`TextRankConfig`] at call time.
///
/// # Implementation note
///
/// Currently delegates to the existing [`PhraseExtractor`] via a legacy
/// adapter bridge.  A native implementation working directly on
/// [`TokenEntry`] can be substituted later without changing the trait
/// contract.
///
/// [`ScoreAggregation`]: crate::types::ScoreAggregation
/// [`PhraseGrouping`]: crate::types::PhraseGrouping
/// [`PhraseExtractor`]: crate::phrase::extraction::PhraseExtractor
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkPhraseBuilder;

impl PhraseBuilder for ChunkPhraseBuilder {
    fn build(
        &self,
        tokens: TokenStreamRef<'_>,
        _candidates: CandidateSetRef<'_>,
        ranks: &RankOutput,
        graph: &Graph,
        cfg: &TextRankConfig,
    ) -> PhraseSet {
        use crate::phrase::extraction::PhraseExtractor;
        use crate::types::StringPool;

        // Bridge: convert pipeline artifacts to legacy types.
        let legacy_tokens = tokens.to_legacy_tokens();
        let pagerank_result = ranks.to_pagerank_result();

        // Delegate to the existing PhraseExtractor.
        let extractor = PhraseExtractor::with_config(cfg.clone());
        let phrases = extractor.extract(&legacy_tokens, graph.csr(), &pagerank_result);

        // Convert back to pipeline artifact.
        let mut pool = StringPool::new();
        PhraseSet::from_phrases(&phrases, &mut pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::artifacts::CandidateKind;
    use crate::pipeline::observer::NoopObserver;
    use crate::pipeline::runner::BaseTextRankPipeline;
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
    // GraphBuilder — WindowGraphBuilder tests
    // ================================================================

    /// Helper: build word candidates from a token stream using default config.
    fn word_candidates(stream: &TokenStream, cfg: &TextRankConfig) -> CandidateSet {
        WordNodeSelector.select(stream.as_ref(), cfg)
    }

    #[test]
    fn test_graph_builder_base_textrank_sentence_bounded() {
        // BaseTextRank: sentence-bounded + count-accumulating (library default).
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

        let gb = CooccurrenceGraphBuilder {
            window_strategy: WindowStrategy::SentenceBounded { window_size: 10 },
            edge_weight_policy: EdgeWeightPolicy::Binary,
        };
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
            window_strategy: WindowStrategy::SentenceBounded { window_size: 2 },
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
            window_strategy: WindowStrategy::SentenceBounded { window_size: 2 },
            edge_weight_policy: EdgeWeightPolicy::CountAccumulating,
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
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder {
            window_strategy: WindowStrategy::SentenceBounded { window_size: 2 },
            edge_weight_policy: EdgeWeightPolicy::Binary,
        };
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
        // WindowGraphBuilder (topic graphs are a separate stage).
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
    fn test_graph_builder_default_is_sentence_bounded_binary() {
        let gb = WindowGraphBuilder::default();
        assert_eq!(
            gb.window_strategy,
            WindowStrategy::SentenceBounded {
                window_size: DEFAULT_WINDOW_SIZE
            }
        );
        assert_eq!(gb.edge_weight_policy, EdgeWeightPolicy::Binary);
    }

    #[test]
    fn test_graph_builder_base_textrank_is_sentence_bounded_count() {
        // base_textrank() matches the library's default config
        // (use_edge_weights=true → CountAccumulating).
        let gb = WindowGraphBuilder::base_textrank();
        assert_eq!(
            gb.window_strategy,
            WindowStrategy::SentenceBounded {
                window_size: DEFAULT_WINDOW_SIZE
            }
        );
        assert_eq!(gb.edge_weight_policy, EdgeWeightPolicy::CountAccumulating);
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
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder {
            window_strategy: WindowStrategy::SentenceBounded { window_size: 2 },
            edge_weight_policy: EdgeWeightPolicy::Binary,
        };
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
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let gb = CooccurrenceGraphBuilder {
            window_strategy: WindowStrategy::CrossSentence { window_size: 2 },
            edge_weight_policy: EdgeWeightPolicy::CountAccumulating,
        };
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
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        // Order A: double then add1 → 1.0 * 2 + 1 = 3.0
        let gb = CooccurrenceGraphBuilder {
            window_strategy: WindowStrategy::SentenceBounded { window_size: 2 },
            edge_weight_policy: EdgeWeightPolicy::Binary,
        };
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

    // ================================================================
    // TeleportBuilder — PositionTeleportBuilder tests
    // ================================================================

    #[test]
    fn test_position_teleport_returns_normalized_vector() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let result = PositionTeleportBuilder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_some());

        let tv = result.unwrap();
        assert_eq!(tv.len(), cs.len());
        assert!(tv.is_normalized(1e-10));
        assert_eq!(tv.teleport_type(), TeleportType::Position);
    }

    #[test]
    fn test_position_teleport_earlier_positions_higher_weight() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let tv = PositionTeleportBuilder
            .build(stream.as_ref(), cs.as_ref(), &cfg)
            .unwrap();

        // The candidate with first_position == 0 should have the highest weight.
        let max_val = tv.as_slice().iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let first_candidate_pos = cs.words().iter().position(|w| w.first_position == 0);
        if let Some(idx) = first_candidate_pos {
            assert!(
                (tv[idx] - max_val).abs() < 1e-10,
                "Candidate at position 0 should have max weight"
            );
        }
    }

    #[test]
    fn test_position_teleport_empty_candidates() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let result = PositionTeleportBuilder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_position_teleport_phrase_candidates_returns_none() {
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

        let result = PositionTeleportBuilder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_position_teleport_as_trait_object() {
        let builder: Box<dyn TeleportBuilder> = Box::new(PositionTeleportBuilder);

        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let result = builder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_some());
        assert!(result.unwrap().is_normalized(1e-10));
    }

    #[test]
    fn test_position_teleport_default() {
        let _tb = PositionTeleportBuilder::default();
    }

    // ================================================================
    // TeleportBuilder — custom builder tests
    // ================================================================

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

                let mut tv = TeleportVector::zeros(words.len(), TeleportType::Position);
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

                let mut tv = TeleportVector::zeros(words.len(), TeleportType::Focus);
                for (i, w) in words.iter().enumerate() {
                    if self.focus_lemma_ids.contains(&w.lemma_id) {
                        tv.set(i, self.bias);
                    } else {
                        tv.set(i, 1.0);
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

    // ================================================================
    // PositionTeleportBuilder — edge cases
    // ================================================================

    #[test]
    fn test_position_teleport_single_candidate() {
        // A single candidate should produce a vector of [1.0].
        let tokens = vec![
            Token::new("Rust", "rust", PosTag::Noun, 0, 4, 0, 0),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);
        assert_eq!(cs.len(), 1);

        let tv = PositionTeleportBuilder
            .build(stream.as_ref(), cs.as_ref(), &cfg)
            .unwrap();
        assert_eq!(tv.len(), 1);
        assert!((tv[0] - 1.0).abs() < 1e-10);
        assert!(tv.is_normalized(1e-10));
    }

    #[test]
    fn test_position_teleport_all_same_position() {
        // All candidates at the same first_position → uniform after normalization.
        // We achieve this by having all tokens at position 0 with unique lemmas.
        let tokens = vec![
            Token::new("alpha", "alpha", PosTag::Noun, 0, 5, 0, 0),
            Token::new("beta", "beta", PosTag::Noun, 6, 10, 0, 0),
            Token::new("gamma", "gamma", PosTag::Noun, 11, 16, 0, 0),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        // Use default config but ensure all POS tags pass the filter.
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        // The WordNodeSelector deduplicates by graph key and records
        // first_position for each unique key. Since all three have
        // token_position=0, they all get first_position=0 in the
        // candidate set.
        let tv = PositionTeleportBuilder
            .build(stream.as_ref(), cs.as_ref(), &cfg)
            .unwrap();

        assert!(tv.is_normalized(1e-10));
        // All weights should be equal (same position → same raw weight).
        let expected = 1.0 / tv.len() as f64;
        for &v in tv.as_slice() {
            assert!(
                (v - expected).abs() < 1e-10,
                "Same-position candidates should produce uniform distribution, got {v} expected {expected}"
            );
        }
    }

    #[test]
    fn test_position_teleport_many_candidates_normalization() {
        // Verify normalization precision with many candidates.
        let tokens: Vec<Token> = (0usize..50)
            .map(|i| {
                let name = format!("word{i}");
                Token::new(&name, &name, PosTag::Noun, i * 10, i * 10 + 5, 0, i)
            })
            .collect();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let tv = PositionTeleportBuilder
            .build(stream.as_ref(), cs.as_ref(), &cfg)
            .unwrap();

        assert!(tv.is_normalized(1e-10));
        assert_eq!(tv.len(), cs.len());
        // All values should be positive.
        for &v in tv.as_slice() {
            assert!(v > 0.0, "All teleport values should be positive");
        }
    }

    // ================================================================
    // FocusTermsTeleportBuilder tests
    // ================================================================

    #[test]
    fn test_focus_teleport_returns_normalized_vector() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let builder = FocusTermsTeleportBuilder::new(
            vec!["machine".to_string()],
            10.0,
        );

        let result = builder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_some());

        let tv = result.unwrap();
        assert_eq!(tv.len(), cs.len());
        assert!(tv.is_normalized(1e-10));
        assert_eq!(tv.teleport_type(), TeleportType::Focus);
    }

    #[test]
    fn test_focus_teleport_boosts_focus_terms() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let builder = FocusTermsTeleportBuilder::new(
            vec!["machine".to_string()],
            10.0,
        );
        let tv = builder.build(stream.as_ref(), cs.as_ref(), &cfg).unwrap();

        // Find the index of "machine" among candidates.
        let machine_idx = cs.words().iter()
            .position(|w| stream.pool().get(w.lemma_id) == Some("machine"))
            .unwrap();
        // Find any non-focus candidate index.
        let other_idx = cs.words().iter()
            .position(|w| stream.pool().get(w.lemma_id) != Some("machine"))
            .unwrap();

        assert!(
            tv[machine_idx] > tv[other_idx],
            "Focus term 'machine' should have higher teleport probability"
        );
    }

    #[test]
    fn test_focus_teleport_multiple_focus_terms() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let builder = FocusTermsTeleportBuilder::new(
            vec!["machine".to_string(), "rust".to_string()],
            5.0,
        );
        let tv = builder.build(stream.as_ref(), cs.as_ref(), &cfg).unwrap();

        assert!(tv.is_normalized(1e-10));

        // Both focus terms should have the same probability (same bias weight).
        let machine_idx = cs.words().iter()
            .position(|w| stream.pool().get(w.lemma_id) == Some("machine"))
            .unwrap();
        let rust_idx = cs.words().iter()
            .position(|w| stream.pool().get(w.lemma_id) == Some("rust"))
            .unwrap();

        assert!(
            (tv[machine_idx] - tv[rust_idx]).abs() < 1e-10,
            "Both focus terms should have equal probability"
        );
    }

    #[test]
    fn test_focus_teleport_empty_candidates() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let builder = FocusTermsTeleportBuilder::new(
            vec!["machine".to_string()],
            10.0,
        );
        let result = builder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_focus_teleport_phrase_candidates_returns_none() {
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

        let builder = FocusTermsTeleportBuilder::new(
            vec!["big".to_string()],
            10.0,
        );
        let result = builder.build(stream.as_ref(), cs.as_ref(), &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_focus_teleport_no_matching_terms() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        // Focus on terms not present in the document.
        let builder = FocusTermsTeleportBuilder::new(
            vec!["nonexistent".to_string()],
            10.0,
        );
        let tv = builder.build(stream.as_ref(), cs.as_ref(), &cfg).unwrap();

        // All candidates get base weight 1.0 → effectively uniform.
        assert!(tv.is_normalized(1e-10));
        let expected = 1.0 / cs.len() as f64;
        for &v in tv.as_slice() {
            assert!(
                (v - expected).abs() < 1e-10,
                "No matches should produce uniform distribution"
            );
        }
    }

    #[test]
    fn test_focus_teleport_single_candidate_is_focus() {
        let tokens = vec![
            Token::new("Rust", "rust", PosTag::Noun, 0, 4, 0, 0),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);
        assert_eq!(cs.len(), 1);

        let builder = FocusTermsTeleportBuilder::new(vec!["rust".to_string()], 10.0);
        let tv = builder.build(stream.as_ref(), cs.as_ref(), &cfg).unwrap();

        // Single candidate → value must be 1.0 regardless of bias.
        assert_eq!(tv.len(), 1);
        assert!((tv[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_focus_teleport_single_candidate_not_focus() {
        let tokens = vec![
            Token::new("Rust", "rust", PosTag::Noun, 0, 4, 0, 0),
        ];
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let builder = FocusTermsTeleportBuilder::new(vec!["python".to_string()], 10.0);
        let tv = builder.build(stream.as_ref(), cs.as_ref(), &cfg).unwrap();

        // Single non-focus candidate → still 1.0 after normalization.
        assert_eq!(tv.len(), 1);
        assert!((tv[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_focus_teleport_empty_focus_terms() {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        // No focus terms → all get base weight → uniform.
        let builder = FocusTermsTeleportBuilder::new(vec![], 10.0);
        let tv = builder.build(stream.as_ref(), cs.as_ref(), &cfg).unwrap();

        assert!(tv.is_normalized(1e-10));
        let expected = 1.0 / cs.len() as f64;
        for &v in tv.as_slice() {
            assert!(
                (v - expected).abs() < 1e-10,
                "Empty focus terms should produce uniform distribution"
            );
        }
    }

    #[test]
    fn test_focus_teleport_many_candidates_normalization() {
        // Verify normalization precision with many candidates.
        let tokens: Vec<Token> = (0usize..50)
            .map(|i| {
                let name = format!("word{i}");
                Token::new(&name, &name, PosTag::Noun, i * 10, i * 10 + 5, 0, i)
            })
            .collect();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);

        let builder = FocusTermsTeleportBuilder::new(
            vec!["word0".to_string(), "word1".to_string()],
            20.0,
        );
        let tv = builder.build(stream.as_ref(), cs.as_ref(), &cfg).unwrap();

        assert!(tv.is_normalized(1e-10));
        assert_eq!(tv.len(), cs.len());
    }

    // ================================================================
    // Ranker — PageRankRanker tests
    // ================================================================

    /// Helper: build a graph from rich_tokens with default config.
    fn build_test_graph() -> (TokenStream, CandidateSet, Graph) {
        let tokens = rich_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = word_candidates(&stream, &cfg);
        let graph = CooccurrenceGraphBuilder::default().build(
            stream.as_ref(),
            cs.as_ref(),
            &cfg,
        );
        (stream, cs, graph)
    }

    #[test]
    fn test_pagerank_ranker_standard() {
        // Standard PageRank: teleport = None.
        let (_stream, _cs, graph) = build_test_graph();
        let cfg = TextRankConfig::default();

        let output = PageRankRanker.rank(&graph, None, &cfg);

        // Should produce scores for every node.
        assert_eq!(output.num_nodes(), graph.num_nodes());
        assert!(output.converged());
        assert!(output.iterations() > 0);
        assert!(output.final_delta() <= cfg.convergence_threshold);

        // Scores should sum to ~1.0.
        let sum: f64 = output.scores().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Scores should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_pagerank_ranker_personalized() {
        // Personalized PageRank with a non-uniform teleport vector.
        let (_stream, _cs, graph) = build_test_graph();
        let cfg = TextRankConfig::default();

        // Bias heavily towards the first candidate.
        let mut tv = TeleportVector::zeros(graph.num_nodes(), TeleportType::Focus);
        tv.set(0, 10.0);
        for i in 1..graph.num_nodes() {
            tv.set(i, 1.0);
        }
        tv.normalize();

        let output = PageRankRanker.rank(&graph, Some(&tv), &cfg);

        assert_eq!(output.num_nodes(), graph.num_nodes());
        assert!(output.converged());

        // Scores should sum to ~1.0.
        let sum: f64 = output.scores().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Scores should sum to 1.0, got {sum}"
        );

        // Compare with standard: biased node should score higher.
        let standard_output = PageRankRanker.rank(&graph, None, &cfg);
        assert!(
            output.score(0) > standard_output.score(0),
            "Biased node should score higher with personalization"
        );
    }

    #[test]
    fn test_pagerank_ranker_empty_graph() {
        let empty_builder = crate::graph::builder::GraphBuilder::new();
        let graph = Graph::from_builder(&empty_builder);
        let cfg = TextRankConfig::default();

        let output = PageRankRanker.rank(&graph, None, &cfg);

        assert_eq!(output.num_nodes(), 0);
        assert!(output.converged());
        assert_eq!(output.iterations(), 0);
    }

    #[test]
    fn test_pagerank_ranker_single_node() {
        // Single node, no edges (dangling).
        let mut builder = crate::graph::builder::GraphBuilder::new();
        builder.get_or_create_node("only|NOUN");
        let graph = Graph::from_builder(&builder);
        let cfg = TextRankConfig::default();

        let output = PageRankRanker.rank(&graph, None, &cfg);

        assert_eq!(output.num_nodes(), 1);
        // Single node should get all the score mass.
        assert!((output.score(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank_ranker_respects_damping() {
        let (_stream, _cs, graph) = build_test_graph();
        let mut cfg_low = TextRankConfig::default();
        cfg_low.damping = 0.5;

        let mut cfg_high = TextRankConfig::default();
        cfg_high.damping = 0.95;

        let output_low = PageRankRanker.rank(&graph, None, &cfg_low);
        let output_high = PageRankRanker.rank(&graph, None, &cfg_high);

        // Both should converge and have valid scores.
        assert!(output_low.converged());
        assert!(output_high.converged());

        // With lower damping (more teleportation), scores should be more
        // uniform. Measure via variance.
        let mean_low = 1.0 / output_low.num_nodes() as f64;
        let var_low: f64 = output_low
            .scores()
            .iter()
            .map(|s| (s - mean_low).powi(2))
            .sum::<f64>()
            / output_low.num_nodes() as f64;

        let mean_high = 1.0 / output_high.num_nodes() as f64;
        let var_high: f64 = output_high
            .scores()
            .iter()
            .map(|s| (s - mean_high).powi(2))
            .sum::<f64>()
            / output_high.num_nodes() as f64;

        assert!(
            var_low <= var_high + 1e-15,
            "Lower damping should produce more uniform scores: var_low={var_low}, var_high={var_high}"
        );
    }

    #[test]
    fn test_pagerank_ranker_respects_max_iterations() {
        // Build a star graph (asymmetric degree distribution) so the initial
        // uniform scores are far from stationary and won't converge quickly.
        let mut builder = crate::graph::builder::GraphBuilder::new();
        let hub = builder.get_or_create_node("hub");
        let leaves: Vec<u32> = (0..20)
            .map(|i| builder.get_or_create_node(&format!("leaf_{i}")))
            .collect();
        for &leaf in &leaves {
            builder.increment_edge(hub, leaf, 1.0);
        }
        let graph = Graph::from_builder(&builder);

        let mut cfg = TextRankConfig::default();
        cfg.max_iterations = 2;
        cfg.convergence_threshold = 1e-15; // Practically unreachable in 2 iters.

        let output = PageRankRanker.rank(&graph, None, &cfg);

        // Should stop at max_iterations.
        assert!(output.iterations() <= 2);
        assert!(!output.converged());
        // Should still produce valid scores.
        assert_eq!(output.num_nodes(), 21);
    }

    #[test]
    fn test_pagerank_ranker_uniform_teleport_matches_none() {
        // Passing a uniform TeleportVector should produce results very
        // close to passing None (standard PageRank).
        let (_stream, _cs, graph) = build_test_graph();
        let cfg = TextRankConfig::default();

        let output_none = PageRankRanker.rank(&graph, None, &cfg);
        let tv = TeleportVector::uniform(graph.num_nodes());
        let output_uniform = PageRankRanker.rank(&graph, Some(&tv), &cfg);

        for i in 0..graph.num_nodes() {
            let diff = (output_none.score(i as u32) - output_uniform.score(i as u32)).abs();
            assert!(
                diff < 0.01,
                "Node {i}: standard={}, uniform-PPR={}, diff={diff}",
                output_none.score(i as u32),
                output_uniform.score(i as u32),
            );
        }
    }

    #[test]
    fn test_pagerank_ranker_symmetric_graph() {
        // All nodes should have roughly equal scores in a fully connected graph.
        let mut builder = crate::graph::builder::GraphBuilder::new();
        let a = builder.get_or_create_node("a");
        let b = builder.get_or_create_node("b");
        let c = builder.get_or_create_node("c");
        builder.increment_edge(a, b, 1.0);
        builder.increment_edge(b, c, 1.0);
        builder.increment_edge(a, c, 1.0);
        let graph = Graph::from_builder(&builder);
        let cfg = TextRankConfig::default();

        let output = PageRankRanker.rank(&graph, None, &cfg);

        let expected = 1.0 / 3.0;
        for i in 0..3 {
            assert!(
                (output.score(i) - expected).abs() < 0.01,
                "Node {i}: expected ~{expected}, got {}",
                output.score(i),
            );
        }
    }

    #[test]
    fn test_pagerank_ranker_as_trait_object() {
        let ranker: Box<dyn Ranker> = Box::new(PageRankRanker);

        let (_stream, _cs, graph) = build_test_graph();
        let cfg = TextRankConfig::default();

        let output = ranker.rank(&graph, None, &cfg);

        assert_eq!(output.num_nodes(), graph.num_nodes());
        assert!(output.converged());
    }

    #[test]
    fn test_pagerank_ranker_default() {
        let _r = PageRankRanker::default();
    }

    #[test]
    fn test_pagerank_ranker_deterministic() {
        // Same input should produce identical output.
        let (_stream, _cs, graph) = build_test_graph();
        let cfg = TextRankConfig::default();

        let output1 = PageRankRanker.rank(&graph, None, &cfg);
        let output2 = PageRankRanker.rank(&graph, None, &cfg);

        assert_eq!(output1.scores(), output2.scores());
        assert_eq!(output1.iterations(), output2.iterations());
        assert_eq!(output1.converged(), output2.converged());
    }

    #[test]
    fn test_pagerank_ranker_personalized_deterministic() {
        let (_stream, _cs, graph) = build_test_graph();
        let cfg = TextRankConfig::default();

        let mut tv = TeleportVector::zeros(graph.num_nodes(), TeleportType::Focus);
        tv.set(0, 5.0);
        tv.set(1, 1.0);
        tv.normalize();

        let output1 = PageRankRanker.rank(&graph, Some(&tv), &cfg);
        let output2 = PageRankRanker.rank(&graph, Some(&tv), &cfg);

        assert_eq!(output1.scores(), output2.scores());
    }

    // ================================================================
    // PhraseBuilder — ChunkPhraseBuilder tests
    // ================================================================

    /// Helper: build a full pipeline from tokens through ranking,
    /// returning everything needed for PhraseBuilder testing.
    fn build_full_pipeline(tokens: &[Token]) -> (TokenStream, CandidateSet, Graph, RankOutput) {
        let stream = TokenStream::from_tokens(tokens);
        let cfg = TextRankConfig::default();
        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        let graph = CooccurrenceGraphBuilder::default().build(
            stream.as_ref(),
            cs.as_ref(),
            &cfg,
        );
        let ranks = PageRankRanker.rank(&graph, None, &cfg);
        (stream, cs, graph, ranks)
    }

    /// Richer token set for phrase builder testing.
    ///
    /// Two sentences with enough nouns to produce multi-word phrases:
    /// "Machine learning algorithms process data"
    /// "Deep learning models train fast"
    fn phrase_test_tokens() -> Vec<Token> {
        let mut tokens = vec![
            // Sentence 0: "Machine learning algorithms process data"
            Token::new("Machine", "machine", PosTag::Noun, 0, 7, 0, 0),
            Token::new("learning", "learning", PosTag::Noun, 8, 16, 0, 1),
            Token::new("algorithms", "algorithm", PosTag::Noun, 17, 27, 0, 2),
            Token::new("process", "process", PosTag::Verb, 28, 35, 0, 3),
            Token::new("data", "data", PosTag::Noun, 36, 40, 0, 4),
            // Sentence 1: "Deep learning models train fast"
            Token::new("Deep", "deep", PosTag::Adjective, 42, 46, 1, 5),
            Token::new("learning", "learning", PosTag::Noun, 47, 55, 1, 6),
            Token::new("models", "model", PosTag::Noun, 56, 62, 1, 7),
            Token::new("train", "train", PosTag::Verb, 63, 68, 1, 8),
            Token::new("fast", "fast", PosTag::Adverb, 69, 73, 1, 9),
        ];
        // Mark common verbs as stopwords (they are content words but won't
        // form noun phrase cores).
        tokens[3].is_stopword = false; // "process" verb
        tokens[8].is_stopword = false; // "train" verb
        tokens
    }

    #[test]
    fn test_chunk_phrase_builder_produces_phrases() {
        let tokens = phrase_test_tokens();
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let cfg = TextRankConfig::default();

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        // Should produce at least one phrase.
        assert!(
            !phrases.is_empty(),
            "ChunkPhraseBuilder should produce phrases from non-trivial input"
        );

        // All phrases should have positive scores.
        for entry in phrases.entries() {
            assert!(
                entry.score > 0.0,
                "Phrase score should be positive, got {}",
                entry.score
            );
            assert!(entry.count > 0, "Phrase count should be positive");
        }
    }

    #[test]
    fn test_chunk_phrase_builder_respects_top_n() {
        let tokens = phrase_test_tokens();
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.top_n = 1;

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        assert!(
            phrases.len() <= 1,
            "top_n=1 should produce at most 1 phrase, got {}",
            phrases.len()
        );
    }

    #[test]
    fn test_chunk_phrase_builder_empty_tokens() {
        let stream = TokenStream::from_tokens(&[]);
        let cfg = TextRankConfig::default();
        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        let graph = CooccurrenceGraphBuilder::default().build(
            stream.as_ref(),
            cs.as_ref(),
            &cfg,
        );
        let ranks = PageRankRanker.rank(&graph, None, &cfg);

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        assert!(phrases.is_empty());
    }

    #[test]
    fn test_chunk_phrase_builder_single_noun() {
        let tokens = vec![
            Token::new("Rust", "rust", PosTag::Noun, 0, 4, 0, 0),
        ];
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let cfg = TextRankConfig::default();

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        // Single noun should produce one single-word phrase.
        assert_eq!(phrases.len(), 1);
        let entry = &phrases.entries()[0];
        assert!(entry.score > 0.0);
    }

    #[test]
    fn test_chunk_phrase_builder_no_nouns() {
        // Only stopwords and verbs — no noun phrases possible.
        let mut tokens = vec![
            Token::new("is", "be", PosTag::Verb, 0, 2, 0, 0),
            Token::new("very", "very", PosTag::Adverb, 3, 7, 0, 1),
            Token::new("quickly", "quickly", PosTag::Adverb, 8, 15, 0, 2),
        ];
        tokens[0].is_stopword = true;

        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let cfg = TextRankConfig::default();

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        // No noun chunks → no phrases.
        assert!(phrases.is_empty());
    }

    #[test]
    fn test_chunk_phrase_builder_sorted_by_score() {
        let tokens = phrase_test_tokens();
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.top_n = 20; // Don't limit.

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        // Phrases should be in descending score order.
        for w in phrases.entries().windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "Phrases should be sorted by score descending: {} >= {}",
                w[0].score,
                w[1].score,
            );
        }
    }

    #[test]
    fn test_chunk_phrase_builder_surface_forms_materialized() {
        let tokens = phrase_test_tokens();
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let cfg = TextRankConfig::default();

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        // All phrase entries from ChunkPhraseBuilder (bridged via
        // PhraseSet::from_phrases) should have surface and lemma_text
        // materialized.
        for entry in phrases.entries() {
            assert!(
                entry.surface.is_some(),
                "Surface form should be materialized"
            );
            assert!(
                entry.lemma_text.is_some(),
                "Lemma text should be materialized"
            );
            assert!(
                !entry.surface.as_ref().unwrap().is_empty(),
                "Surface form should not be empty"
            );
        }
    }

    #[test]
    fn test_chunk_phrase_builder_min_phrase_length() {
        let tokens = phrase_test_tokens();
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.min_phrase_length = 2; // Only multi-word phrases.

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        // All phrases should have at least 2 lemma tokens.
        for entry in phrases.entries() {
            assert!(
                entry.lemma_ids.len() >= 2,
                "min_phrase_length=2 should filter single-word phrases, got len={}",
                entry.lemma_ids.len()
            );
        }
    }

    #[test]
    fn test_chunk_phrase_builder_max_phrase_length() {
        let tokens = phrase_test_tokens();
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let mut cfg = TextRankConfig::default();
        cfg.max_phrase_length = 1; // Only single-word phrases.

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        // All phrases should have at most 1 lemma token.
        for entry in phrases.entries() {
            assert!(
                entry.lemma_ids.len() <= 1,
                "max_phrase_length=1 should filter multi-word phrases, got len={}",
                entry.lemma_ids.len()
            );
        }
    }

    #[test]
    fn test_chunk_phrase_builder_deterministic() {
        let tokens = phrase_test_tokens();
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let cfg = TextRankConfig::default();

        let phrases1 = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );
        let phrases2 = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        assert_eq!(phrases1.len(), phrases2.len());
        for (a, b) in phrases1.entries().iter().zip(phrases2.entries()) {
            assert_eq!(a.surface, b.surface);
            assert!((a.score - b.score).abs() < 1e-15);
            assert_eq!(a.count, b.count);
        }
    }

    #[test]
    fn test_chunk_phrase_builder_as_trait_object() {
        let builder: Box<dyn PhraseBuilder> = Box::new(ChunkPhraseBuilder);

        let tokens = phrase_test_tokens();
        let (stream, cs, graph, ranks) = build_full_pipeline(&tokens);
        let cfg = TextRankConfig::default();

        let phrases = builder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_chunk_phrase_builder_default() {
        let _pb = ChunkPhraseBuilder::default();
    }

    #[test]
    fn test_chunk_phrase_builder_with_personalized_ranks() {
        // Verify that biasing PageRank towards a specific node changes
        // the phrase scores — demonstrating that PhraseBuilder correctly
        // propagates rank differences.
        let tokens = phrase_test_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        let graph = CooccurrenceGraphBuilder::default().build(
            stream.as_ref(),
            cs.as_ref(),
            &cfg,
        );

        // Standard ranks.
        let standard_ranks = PageRankRanker.rank(&graph, None, &cfg);
        let standard_phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &standard_ranks,
            &graph,
            &cfg,
        );

        // Biased ranks: heavily bias node 0.
        let mut tv = TeleportVector::zeros(graph.num_nodes(), TeleportType::Focus);
        if graph.num_nodes() > 0 {
            tv.set(0, 10.0);
            for i in 1..graph.num_nodes() {
                tv.set(i, 1.0);
            }
            tv.normalize();
        }
        let biased_ranks = PageRankRanker.rank(&graph, Some(&tv), &cfg);
        let biased_phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &biased_ranks,
            &graph,
            &cfg,
        );

        // Both should produce phrases.
        assert!(!standard_phrases.is_empty());
        assert!(!biased_phrases.is_empty());

        // The scores should differ (personalization changes the distribution).
        // At least one phrase should have a different score.
        let standard_scores: Vec<f64> = standard_phrases.entries().iter().map(|e| e.score).collect();
        let biased_scores: Vec<f64> = biased_phrases.entries().iter().map(|e| e.score).collect();
        let any_different = standard_scores.iter().zip(biased_scores.iter())
            .any(|(&s, &b)| (s - b).abs() > 1e-10);
        assert!(
            any_different || standard_scores.len() != biased_scores.len(),
            "Personalized ranks should produce different phrase scores"
        );
    }

    #[test]
    fn test_chunk_phrase_builder_cross_sentence_graph() {
        // Test with SingleRank-style cross-sentence graph to verify
        // PhraseBuilder works with different graph construction strategies.
        let tokens = phrase_test_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);

        // Cross-sentence graph (SingleRank-style).
        let graph = CooccurrenceGraphBuilder::single_rank().build(
            stream.as_ref(),
            cs.as_ref(),
            &cfg,
        );
        let ranks = PageRankRanker.rank(&graph, None, &cfg);

        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        assert!(
            !phrases.is_empty(),
            "ChunkPhraseBuilder should work with cross-sentence graphs"
        );
    }

    // ================================================================
    // ResultFormatter tests
    // ================================================================

    /// Build the full pipeline through to PhraseSet for formatter testing.
    fn build_phrases(tokens: &[Token]) -> (PhraseSet, RankOutput) {
        let (stream, cs, graph, ranks) = build_full_pipeline(tokens);
        let cfg = TextRankConfig::default();
        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );
        (phrases, ranks)
    }

    #[test]
    fn test_standard_formatter_produces_phrases() {
        let tokens = phrase_test_tokens();
        let (phrases, ranks) = build_phrases(&tokens);
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        assert!(
            !result.phrases.is_empty(),
            "StandardResultFormatter should produce phrases"
        );
    }

    #[test]
    fn test_standard_formatter_preserves_phrase_count() {
        let tokens = phrase_test_tokens();
        let (phrases, ranks) = build_phrases(&tokens);
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        assert_eq!(
            result.phrases.len(),
            phrases.len(),
            "Number of output phrases should match PhraseSet entries"
        );
    }

    #[test]
    fn test_standard_formatter_sets_convergence() {
        let tokens = phrase_test_tokens();
        let (phrases, ranks) = build_phrases(&tokens);
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        assert_eq!(result.converged, ranks.converged());
        assert_eq!(result.iterations, ranks.iterations());
    }

    #[test]
    fn test_standard_formatter_ranks_are_1_indexed() {
        let tokens = phrase_test_tokens();
        let (phrases, ranks) = build_phrases(&tokens);
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        for (i, phrase) in result.phrases.iter().enumerate() {
            assert_eq!(
                phrase.rank,
                i + 1,
                "Rank should be 1-indexed sequential"
            );
        }
    }

    #[test]
    fn test_standard_formatter_preserves_scores() {
        let tokens = phrase_test_tokens();
        let (phrases, ranks) = build_phrases(&tokens);
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        for (output, entry) in result.phrases.iter().zip(phrases.entries().iter()) {
            assert!(
                (output.score - entry.score).abs() < 1e-10,
                "Scores should be preserved exactly"
            );
        }
    }

    #[test]
    fn test_standard_formatter_preserves_surface_forms() {
        let tokens = phrase_test_tokens();
        let (phrases, ranks) = build_phrases(&tokens);
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        for (output, entry) in result.phrases.iter().zip(phrases.entries().iter()) {
            let expected_text = entry.surface.clone().unwrap_or_default();
            assert_eq!(
                output.text, expected_text,
                "Surface form should be preserved"
            );
        }
    }

    #[test]
    fn test_standard_formatter_no_debug_by_default() {
        let tokens = phrase_test_tokens();
        let (phrases, ranks) = build_phrases(&tokens);
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        assert!(
            result.debug.is_none(),
            "Debug payload should be None when not provided"
        );
    }

    #[test]
    fn test_standard_formatter_attaches_debug() {
        let tokens = phrase_test_tokens();
        let (phrases, ranks) = build_phrases(&tokens);
        let cfg = TextRankConfig::default();

        let debug = DebugPayload {
            node_scores: Some(vec![("machine|NOUN".to_string(), 0.5)]),
            graph_stats: Some(crate::pipeline::artifacts::GraphStats {
                num_nodes: 5,
                num_edges: 8,
            }),
            stage_timings: None,
            residuals: None,
        };

        let result =
            StandardResultFormatter.format(&phrases, &ranks, Some(debug), &cfg);

        assert!(result.debug.is_some(), "Debug payload should be attached");
        let d = result.debug.unwrap();
        assert!(d.node_scores.is_some());
        assert_eq!(d.node_scores.as_ref().unwrap().len(), 1);
        assert!(d.graph_stats.is_some());
        assert_eq!(d.graph_stats.as_ref().unwrap().num_nodes, 5);
    }

    #[test]
    fn test_standard_formatter_empty_phrases() {
        let ranks = RankOutput::from_pagerank_result(
            &crate::pagerank::PageRankResult {
                scores: vec![],
                iterations: 0,
                delta: 0.0,
                converged: true,
            },
        );
        let phrases = PhraseSet::from_entries(vec![]);
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        assert!(result.phrases.is_empty());
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_standard_formatter_preserves_offsets() {
        use crate::pipeline::artifacts::PhraseEntry;

        // Build a PhraseSet with explicit spans.
        let entry = PhraseEntry {
            lemma_ids: vec![0, 1],
            score: 1.5,
            count: 2,
            surface: Some("machine learning".to_string()),
            lemma_text: Some("machine learning".to_string()),
            spans: Some(vec![(0, 16), (42, 55)]),
        };
        let phrases = PhraseSet::from_entries(vec![entry]);
        let ranks = RankOutput::from_pagerank_result(
            &crate::pagerank::PageRankResult {
                scores: vec![0.5, 0.3],
                iterations: 50,
                delta: 1e-7,
                converged: true,
            },
        );
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        assert_eq!(result.phrases.len(), 1);
        let p = &result.phrases[0];
        assert_eq!(p.text, "machine learning");
        assert_eq!(p.count, 2);
        assert_eq!(p.offsets, vec![(0, 16), (42, 55)]);
        assert_eq!(p.rank, 1);
    }

    #[test]
    fn test_standard_formatter_handles_missing_surface() {
        use crate::pipeline::artifacts::PhraseEntry;

        // Entry with no surface/lemma strings (would happen for truly
        // interned-only entries in a future native PhraseBuilder).
        let entry = PhraseEntry {
            lemma_ids: vec![0],
            score: 0.8,
            count: 1,
            surface: None,
            lemma_text: None,
            spans: None,
        };
        let phrases = PhraseSet::from_entries(vec![entry]);
        let ranks = RankOutput::from_pagerank_result(
            &crate::pagerank::PageRankResult {
                scores: vec![0.8],
                iterations: 10,
                delta: 1e-6,
                converged: true,
            },
        );
        let cfg = TextRankConfig::default();

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        assert_eq!(result.phrases.len(), 1);
        // Missing surface/lemma default to empty string.
        assert_eq!(result.phrases[0].text, "");
        assert_eq!(result.phrases[0].lemma, "");
        assert!(result.phrases[0].offsets.is_empty());
    }

    #[test]
    fn test_standard_formatter_full_pipeline_roundtrip() {
        // End-to-end: tokens → candidates → graph → rank → phrases → format
        let tokens = phrase_test_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default();
        let cs = WordNodeSelector.select(stream.as_ref(), &cfg);
        let graph = CooccurrenceGraphBuilder::default().build(
            stream.as_ref(),
            cs.as_ref(),
            &cfg,
        );
        let ranks = PageRankRanker.rank(&graph, None, &cfg);
        let phrases = ChunkPhraseBuilder.build(
            stream.as_ref(),
            cs.as_ref(),
            &ranks,
            &graph,
            &cfg,
        );

        let result = StandardResultFormatter.format(&phrases, &ranks, None, &cfg);

        // Verify the full pipeline produces coherent output.
        assert!(!result.phrases.is_empty());
        assert!(result.converged);
        assert!(result.iterations > 0);
        // Phrases should have positive scores.
        for p in &result.phrases {
            assert!(p.score > 0.0, "All phrases should have positive scores");
            assert!(!p.text.is_empty(), "All phrases should have surface text");
            assert!(p.rank > 0, "Ranks should be 1-indexed");
        }
    }

    // ================================================================
    // Integration tests — teleport ordering stability
    // ================================================================

    /// Tokens where "early" word and "late" word have symmetric context,
    /// so the only differentiator is position-based teleport.
    fn position_integration_tokens() -> Vec<Token> {
        vec![
            // Sentence 0: "Alpha beta gamma"
            Token::new("Alpha", "alpha", PosTag::Noun, 0, 5, 0, 0),
            Token::new("beta", "beta", PosTag::Noun, 6, 10, 0, 1),
            Token::new("gamma", "gamma", PosTag::Noun, 11, 16, 0, 2),
            // Sentence 1: "Delta beta gamma"
            Token::new("Delta", "delta", PosTag::Noun, 18, 23, 1, 3),
            Token::new("beta", "beta", PosTag::Noun, 24, 28, 1, 4),
            Token::new("gamma", "gamma", PosTag::Noun, 29, 34, 1, 5),
        ]
    }

    #[test]
    fn test_position_pipeline_earlier_words_rank_higher() {
        use crate::pipeline::runner::PositionRankPipeline;

        let tokens = position_integration_tokens();
        let stream = TokenStream::from_tokens(&tokens);
        let cfg = TextRankConfig::default().with_top_n(10);
        let mut obs = NoopObserver;

        let pipeline = PositionRankPipeline::position_rank();
        let result = pipeline.run(stream, &cfg, &mut obs);

        assert!(result.converged);

        // "alpha" (position 0) should rank at or above "delta" (position 3)
        // because PositionRank biases towards earlier tokens.
        let alpha = result.phrases.iter().find(|p| p.lemma.contains("alpha"));
        let delta = result.phrases.iter().find(|p| p.lemma.contains("delta"));

        if let (Some(a), Some(d)) = (alpha, delta) {
            assert!(
                a.score >= d.score,
                "Earlier word 'alpha' (score={}) should score >= later word 'delta' (score={})",
                a.score, d.score
            );
        }
    }

    #[test]
    fn test_position_pipeline_deterministic_ordering() {
        use crate::pipeline::runner::PositionRankPipeline;

        let tokens = position_integration_tokens();
        let cfg = TextRankConfig::default().with_top_n(10);

        // Run twice and verify identical output.
        let run = || {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            PositionRankPipeline::position_rank().run(stream, &cfg, &mut obs)
        };

        let r1 = run();
        let r2 = run();

        assert_eq!(r1.phrases.len(), r2.phrases.len());
        for (p1, p2) in r1.phrases.iter().zip(r2.phrases.iter()) {
            assert_eq!(p1.lemma, p2.lemma, "Phrase ordering should be deterministic");
            assert!(
                (p1.score - p2.score).abs() < 1e-12,
                "Scores should be identical across runs"
            );
        }
    }

    #[test]
    fn test_focus_pipeline_boosts_target_term() {
        use crate::pipeline::runner::BiasedTextRankPipeline;

        let tokens = rich_tokens();
        let cfg = TextRankConfig::default().with_top_n(10);

        // Run with "rust" focused.
        let stream = TokenStream::from_tokens(&tokens);
        let mut obs = NoopObserver;
        let focused = BiasedTextRankPipeline::biased(
            vec!["rust".to_string()],
            20.0,
        ).run(stream, &cfg, &mut obs);

        // Run with no focus (uniform teleport = base textrank).
        let stream2 = TokenStream::from_tokens(&tokens);
        let mut obs2 = NoopObserver;
        let baseline = BaseTextRankPipeline::base_textrank()
            .run(stream2, &cfg, &mut obs2);

        // "rust" should have a higher score in the focused run.
        let rust_focused = focused.phrases.iter()
            .find(|p| p.lemma.contains("rust"))
            .map(|p| p.score);
        let rust_baseline = baseline.phrases.iter()
            .find(|p| p.lemma.contains("rust"))
            .map(|p| p.score);

        if let (Some(f), Some(b)) = (rust_focused, rust_baseline) {
            assert!(
                f >= b,
                "Focus term 'rust' should score higher with bias (focused={f}, baseline={b})"
            );
        }
    }

    #[test]
    fn test_focus_pipeline_deterministic_ordering() {
        use crate::pipeline::runner::BiasedTextRankPipeline;

        let tokens = rich_tokens();
        let cfg = TextRankConfig::default().with_top_n(10);

        let run = || {
            let stream = TokenStream::from_tokens(&tokens);
            let mut obs = NoopObserver;
            BiasedTextRankPipeline::biased(
                vec!["machine".to_string()],
                10.0,
            ).run(stream, &cfg, &mut obs)
        };

        let r1 = run();
        let r2 = run();

        assert_eq!(r1.phrases.len(), r2.phrases.len());
        for (p1, p2) in r1.phrases.iter().zip(r2.phrases.iter()) {
            assert_eq!(p1.lemma, p2.lemma, "Phrase ordering should be deterministic");
            assert!(
                (p1.score - p2.score).abs() < 1e-12,
                "Scores should be identical across runs"
            );
        }
    }

    // ================================================================
    // WindowStrategy — serde round-trip tests
    // ================================================================

    #[test]
    fn test_window_strategy_serde_sentence_bounded() {
        let ws = WindowStrategy::SentenceBounded { window_size: 3 };
        let json = serde_json::to_string(&ws).unwrap();
        assert!(json.contains("\"type\":\"sentence_bounded\""));
        assert!(json.contains("\"window_size\":3"));

        let deser: WindowStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, ws);
    }

    #[test]
    fn test_window_strategy_serde_cross_sentence() {
        let ws = WindowStrategy::CrossSentence { window_size: 5 };
        let json = serde_json::to_string(&ws).unwrap();
        assert!(json.contains("\"type\":\"cross_sentence\""));
        assert!(json.contains("\"window_size\":5"));

        let deser: WindowStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, ws);
    }

    #[test]
    fn test_window_strategy_serde_from_json_literal() {
        let json = r#"{"type":"sentence_bounded","window_size":7}"#;
        let ws: WindowStrategy = serde_json::from_str(json).unwrap();
        assert_eq!(ws, WindowStrategy::SentenceBounded { window_size: 7 });
        assert_eq!(ws.window_size(), 7);
        assert!(ws.is_sentence_bounded());
        assert!(!ws.is_cross_sentence());
    }

    #[test]
    fn test_window_strategy_default() {
        let ws = WindowStrategy::default();
        assert_eq!(ws.window_size(), DEFAULT_WINDOW_SIZE);
        assert!(ws.is_sentence_bounded());
    }

    #[test]
    fn test_window_strategy_accessors() {
        let sb = WindowStrategy::SentenceBounded { window_size: 4 };
        assert_eq!(sb.window_size(), 4);
        assert!(sb.is_sentence_bounded());
        assert!(!sb.is_cross_sentence());

        let cs = WindowStrategy::CrossSentence { window_size: 6 };
        assert_eq!(cs.window_size(), 6);
        assert!(!cs.is_sentence_bounded());
        assert!(cs.is_cross_sentence());
    }

    // ================================================================
    // EdgeWeightPolicy — serde round-trip tests
    // ================================================================

    #[test]
    fn test_edge_weight_policy_serde_binary() {
        let p = EdgeWeightPolicy::Binary;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"binary\"");

        let deser: EdgeWeightPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, p);
    }

    #[test]
    fn test_edge_weight_policy_serde_count_accumulating() {
        let p = EdgeWeightPolicy::CountAccumulating;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"count_accumulating\"");

        let deser: EdgeWeightPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, p);
    }

    #[test]
    fn test_edge_weight_policy_serde_from_json_literal() {
        let p: EdgeWeightPolicy = serde_json::from_str("\"binary\"").unwrap();
        assert_eq!(p, EdgeWeightPolicy::Binary);
        assert!(p.is_binary());

        let p2: EdgeWeightPolicy = serde_json::from_str("\"count_accumulating\"").unwrap();
        assert_eq!(p2, EdgeWeightPolicy::CountAccumulating);
        assert!(p2.is_count_accumulating());
    }

    #[test]
    fn test_edge_weight_policy_default() {
        let p = EdgeWeightPolicy::default();
        assert_eq!(p, EdgeWeightPolicy::Binary);
        assert!(p.is_binary());
        assert!(!p.is_count_accumulating());
    }
}
