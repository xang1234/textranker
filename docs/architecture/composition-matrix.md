# Variant Composition Matrix

Every algorithm variant in rapid_textrank is a **parameter choice**, not a separate code path. The `Pipeline` struct is generic over eight stage slots:

```rust
pub struct Pipeline<Pre, Sel, GB, GT, TB, Rnk, PB, Fmt> { .. }
```

Each variant is a type alias that fills these slots with concrete implementations. Stages that differ from the baseline are **bolded** in the tables below.

---

## Complete Matrix

| Stage | Trait | BaseTextRank | PositionRank | BiasedTextRank | SingleRank | TopicalPageRank | TopicRank | MultipartiteRank | SentenceRank |
|-------|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Pre` | `Preprocessor` | Noop | Noop | Noop | Noop | Noop | Noop | Noop | Noop |
| `Sel` | `CandidateSelector` | WordNode | WordNode | WordNode | WordNode | WordNode | **Phrase** | **Phrase** | **Sentence** |
| `GB` | `GraphBuilder` | Window | Window | Window | Window | Window | **TopicGraph** | **CandidateGraph** | **SentenceGraph** |
| `GT` | `GraphTransform` | Noop | Noop | Noop | Noop | Noop | Noop | **Multipartite** | Noop |
| `TB` | `TeleportBuilder` | Uniform | **Position** | **FocusTerms** | Uniform | **TopicWeights** | Uniform | Uniform | Uniform |
| `Rnk` | `Ranker` | PageRank | PageRank | PageRank | PageRank | PageRank | PageRank | PageRank | PageRank |
| `PB` | `PhraseBuilder` | Chunk | Chunk | Chunk | Chunk | Chunk | **TopicRep** | **MultipartitePhrase** | **SentencePhrase** |
| `Fmt` | `ResultFormatter` | Standard | Standard | Standard | Standard | Standard | Standard | Standard | **Sentence** |

Reading the table column-by-column shows exactly what makes each variant unique. Reading row-by-row shows which stages are shared.

---

## Rust Type Aliases

Each variant maps to a concrete type alias in `src/pipeline/runner.rs`:

```rust
// Word-graph family
type BaseTextRankPipeline     = Pipeline<Noop, WordNode, Window, Noop, Uniform,    PR, Chunk, Std>;
type PositionRankPipeline     = Pipeline<Noop, WordNode, Window, Noop, Position,   PR, Chunk, Std>;
type BiasedTextRankPipeline   = Pipeline<Noop, WordNode, Window, Noop, FocusTerms, PR, Chunk, Std>;
type SingleRankPipeline       = Pipeline<Noop, WordNode, Window, Noop, Uniform,    PR, Chunk, Std>;
type TopicalPageRankPipeline  = Pipeline<Noop, WordNode, Window, Noop, TopicWts,   PR, Chunk, Std>;

// Topic family
type TopicRankPipeline        = Pipeline<Noop, Phrase, TopicGB<HAC>, Noop, Uniform, PR, TopicRep, Std>;
type MultipartiteRankPipeline = Pipeline<Noop, Phrase, CandGB<HAC>,  MPT, Uniform, PR, MPPhrase, Std>;

// Extractive summarization (feature-gated: `sentence-rank`)
type SentenceRankPipeline     = Pipeline<Noop, SentSel, SentGB, Noop, Uniform, PR, SentPB, SentFmt>;
```

> Note: type names above are abbreviated for readability. See `runner.rs` for exact definitions.

Notice that `BaseTextRankPipeline` and `SingleRankPipeline` have **identical type signatures** — they differ only in the runtime configuration of `WindowGraphBuilder` (sentence-bounded vs. cross-sentence).

---

## Stage Implementation Details

### Candidate Selectors

| Implementation | Candidates Produced | Used By |
|---------------|-------------------|---------|
| `WordNodeSelector` | `CandidateKind::Words` — one entry per unique (lemma, POS) pair that passes POS and stopword filters | BaseTextRank, PositionRank, BiasedTextRank, SingleRank, TopicalPageRank |
| `PhraseCandidateSelector` | `CandidateKind::Phrases` — multi-word candidates from pre-supplied `ChunkSpan`s | TopicRank, MultipartiteRank |
| `SentenceCandidateSelector` | `CandidateKind::Sentences` — one candidate per sentence in the document | SentenceRank |

### Graph Builders

| Implementation | Graph Topology | Configuration Axes | Used By |
|---------------|---------------|-------------------|---------|
| `WindowGraphBuilder` | Co-occurrence within sliding window | `WindowStrategy` × `EdgeWeightPolicy` | All word-graph variants |
| `TopicGraphBuilder<C>` | Complete graph over cluster centroids | Embeds a `Clusterer` (e.g., `JaccardHacClusterer`) | TopicRank |
| `CandidateGraphBuilder<C>` | Complete graph over individual candidates | Embeds a `Clusterer` | MultipartiteRank |
| `SentenceGraphBuilder` | Jaccard similarity between token sets | `min_similarity` threshold (default: 0.0) | SentenceRank |

#### WindowGraphBuilder Configurations

The `WindowGraphBuilder` covers two configuration axes:

| Parameter | BaseTextRank / PositionRank / BiasedTextRank | SingleRank / TopicalPageRank |
|-----------|:---:|:---:|
| `WindowStrategy` | `SentenceBounded { window_size: 3 }` | `CrossSentence { window_size: 3 }` |
| `EdgeWeightPolicy` | `CountAccumulating` | `CountAccumulating` |

Both groups use count-accumulating edges by default. The paper-standard binary-edge variant can be constructed manually with `EdgeWeightPolicy::Binary`.

- **SentenceBounded**: Window resets at sentence boundaries. Co-occurrences never cross sentence breaks.
- **CrossSentence**: Window slides continuously over the entire candidate sequence, ignoring sentence boundaries.

### Graph Transforms

| Implementation | Effect | Used By |
|---------------|--------|---------|
| `NoopGraphTransform` | Nothing (zero-sized, compiled away) | All except MultipartiteRank |
| `MultipartiteTransform` | 1. Zeros intra-cluster edges (k-partite structure) 2. Boosts edges toward first-occurring variants (alpha=1.1) | MultipartiteRank |

The `MultipartiteTransform` combines two sub-operations (`IntraTopicEdgeRemover` + `AlphaBoostWeighter`) into a single pass for efficiency.

### Teleport Builders

| Implementation | Teleport Vector | Parameters | Used By |
|---------------|----------------|-----------|---------|
| `UniformTeleportBuilder` | `None` (standard PageRank, uniform jump) | — | BaseTextRank, SingleRank, TopicRank, MultipartiteRank, SentenceRank |
| `PositionTeleportBuilder` | `weight[i] = 1 / (position + 1)` | — | PositionRank |
| `FocusTermsTeleportBuilder` | Focus terms get `bias_weight`; others get `1.0` | `focus_terms: Vec<String>`, `bias_weight: f64` | BiasedTextRank |
| `TopicWeightsTeleportBuilder` | Per-word weights from external topic model | `topic_weights: HashMap<String, f64>`, `min_weight: f64` | TopicalPageRank |

All non-uniform teleport vectors are **normalized to sum to 1.0** before being passed to PageRank.

### Phrase Builders

| Implementation | Strategy | Used By |
|---------------|----------|---------|
| `ChunkPhraseBuilder` | Noun-phrase chunking: `(DET)? (ADJ)* (NOUN\|PROPN)+`. Scores via PageRank aggregation. Groups by lemma. | All word-graph variants |
| `TopicRepresentativeBuilder` | Selects first-occurring phrase from each top-scoring cluster | TopicRank |
| `MultipartitePhraseBuilder` | Selects highest-scoring phrase per lemma group | MultipartiteRank |
| `SentencePhraseBuilder` | Materializes full sentence text directly | SentenceRank |

### Result Formatters

| Implementation | Sort Order | Used By |
|---------------|-----------|---------|
| `StandardResultFormatter` | Score descending (deterministic mode: `stable_cmp` with tie-breaking) | All except SentenceRank |
| `SentenceFormatter` | Score descending (default) or document position ascending (`sort_by_position = true`) | SentenceRank |

---

## Clustering (Topic Family Only)

TopicRank and MultipartiteRank share a clustering sub-stage embedded in the graph builder:

| Parameter | TopicRank | MultipartiteRank |
|-----------|:---------:|:----------------:|
| Clusterer | `JaccardHacClusterer` | `JaccardHacClusterer` |
| Distance metric | Jaccard | Jaccard |
| Linkage | Average | Average |
| Similarity threshold | **0.25** | **0.26** |

The clustering result (`ClusterAssignments`) is attached to the `Graph` artifact and consumed by the phrase builder and (for MultipartiteRank) the graph transform.

---

## What This Means

1. **No variant-specific orchestration code.** The pipeline runner (`Pipeline::run_inner`) is a single generic function that executes the same eight stages for every variant. The variant is encoded entirely in which concrete types fill the generic slots.

2. **Adding a variant = choosing stages.** To create a new variant, pick an implementation for each stage and define a type alias. No new control flow is needed.

3. **Runtime vs. compile-time variation.** Some variants differ at the type level (e.g., `PositionTeleportBuilder` vs. `UniformTeleportBuilder`), while others differ only at runtime (e.g., `BaseTextRankPipeline` vs. `SingleRankPipeline` share the same types but configure `WindowGraphBuilder` differently). Both approaches are valid — type-level differences enable zero-cost abstraction, while runtime differences keep the type zoo small.

4. **Zero-cost defaults.** Stages like `NoopPreprocessor`, `NoopGraphTransform`, and `UniformTeleportBuilder` are zero-sized types. They add zero bytes to the pipeline struct and compile to no instructions.
