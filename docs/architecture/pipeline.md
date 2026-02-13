# Pipeline Architecture

This document describes the internal pipeline that powers every TextRank variant in rapid_textrank.

---

## Stage Diagram

Every extraction request flows through eight stages. Stages are statically composed — the compiler monomorphizes each variant into a unique concrete type with zero virtual-call overhead.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        Input: TokenStream                          │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 0 · PREPROCESS                                              │
  │  Trait: Preprocessor                                               │
  │  Default: NoopPreprocessor (zero-sized, compiled away)             │
  │  Mutates TokenStream in place                                      │
  │  Ownership: &mut TokenStream (borrowed, mutated)                   │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 1 · CANDIDATES                                              │
  │  Trait: CandidateSelector                                          │
  │  Impls: WordNodeSelector │ PhraseCandidateSelector                 │
  │         │ SentenceCandidateSelector                                │
  │  Input:  TokenStreamRef<'_> (borrowed)                             │
  │  Output: CandidateSet (owned)                                      │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 2 · GRAPH                                                   │
  │  Trait: GraphBuilder                                               │
  │  Impls: WindowGraphBuilder │ TopicGraphBuilder<C>                  │
  │         │ CandidateGraphBuilder<C> │ SentenceGraphBuilder          │
  │  Input:  TokenStreamRef<'_>, CandidateSetRef<'_> (borrowed)       │
  │  Output: Graph (owned, CSR format)                                 │
  │  Note: TopicGraphBuilder embeds a Clusterer (e.g. JaccardHAC)     │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 2a · GRAPH TRANSFORM (optional)                             │
  │  Trait: GraphTransform                                             │
  │  Default: NoopGraphTransform (zero-sized, compiled away)           │
  │  Impls: IntraTopicEdgeRemover │ AlphaBoostWeighter                 │
  │         │ MultipartiteTransform (combined)                         │
  │  Ownership: &mut Graph (borrowed, mutated in place)                │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 3a · TELEPORT                                               │
  │  Trait: TeleportBuilder                                            │
  │  Default: UniformTeleportBuilder → returns None (standard PR)      │
  │  Impls: PositionTeleportBuilder │ FocusTermsTeleportBuilder        │
  │         │ TopicWeightsTeleportBuilder                              │
  │  Input:  TokenStreamRef<'_>, CandidateSetRef<'_> (borrowed)       │
  │  Output: Option<TeleportVector> (owned)                            │
  │  Contract: returned vector MUST sum to 1.0                         │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 3 · RANK                                                    │
  │  Trait: Ranker                                                     │
  │  Impl: PageRankRanker                                              │
  │  Delegates to: StandardPageRank (if teleport = None)               │
  │                PersonalizedPageRank (if teleport = Some)           │
  │  Input:  &Graph, Option<&TeleportVector> (borrowed)                │
  │  Output: RankOutput (owned — scores + convergence metadata)        │
  │  Algorithm: Power iteration with damping, dangling-node handling   │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 4 · PHRASES                                                 │
  │  Trait: PhraseBuilder                                              │
  │  Impls: ChunkPhraseBuilder │ TopicRepresentativeBuilder            │
  │         │ MultipartitePhraseBuilder │ SentencePhraseBuilder        │
  │  Input:  all prior artifacts (borrowed)                            │
  │  Output: PhraseSet (owned)                                         │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stage 5 · FORMAT                                                  │
  │  Trait: ResultFormatter                                            │
  │  Impls: StandardResultFormatter │ SentenceFormatter                │
  │  Input:  &PhraseSet, &RankOutput, optional DebugPayload (borrowed) │
  │  Output: FormattedResult (owned — final public result)             │
  │  Responsibilities: sort, assign ranks, apply top_n, attach meta    │
  └───────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
                         FormattedResult
```

---

## Artifact Definitions

Each stage boundary has a well-defined artifact type. The ownership discipline keeps the hot path allocation-light: stages borrow whenever possible and only produce owned output when the data is new.

| Artifact | Defined in | Ownership | Description |
|----------|-----------|-----------|-------------|
| `TokenStream` | `pipeline/artifacts.rs` | **Owned** by pipeline runner | Interned token array with `StringPool`, CSR-style `sentence_offsets`. ~28–32 bytes per token. Passed by `&mut` to preprocess, then by `TokenStreamRef<'_>` to all later stages. |
| `TokenStreamRef<'_>` | `pipeline/artifacts.rs` | **Borrowed** view | Read-only reference into `TokenStream`. Zero-cost wrapper providing safe slice access. |
| `CandidateSet` | `pipeline/artifacts.rs` | **Owned** by runner | Contains `CandidateKind` enum: `Words(Vec<WordCandidate>)`, `Phrases(Vec<PhraseCandidate>)`, or `Sentences(Vec<SentenceCandidate>)`. |
| `CandidateSetRef<'_>` | `pipeline/artifacts.rs` | **Borrowed** view | Read-only reference into `CandidateSet`. Passed to graph, transform, teleport, and phrase stages. |
| `Graph` | `pipeline/artifacts.rs` | **Owned** by runner | Wraps a `CsrGraph` (compressed sparse row). Optional `ClusterAssignments` for topic variants. A `transformed` flag tracks whether transform has run. |
| `TeleportVector` | `pipeline/artifacts.rs` | **Owned** (optional) | `Vec<f64>` personalization vector. `None` means uniform teleport (standard PageRank). When present, must be normalized to sum to 1.0. |
| `RankOutput` | `pipeline/artifacts.rs` | **Owned** by runner | Per-node scores (`Vec<f64>`) plus convergence metadata (`converged: bool`, `iterations: usize`, `delta: f64`). |
| `PhraseSet` | `pipeline/artifacts.rs` | **Owned** by runner | Collection of `PhraseEntry` records (still using interned IDs — not yet materialized to strings). |
| `FormattedResult` | `pipeline/artifacts.rs` | **Owned** — returned to caller | Public result: materialized `Vec<Phrase>`, convergence info, optional debug payload. Strings are only materialized at this final stage. |

### Ownership Pattern Summary

```
Preprocess:  &mut TokenStream          (borrow + mutate)
Candidates:  TokenStreamRef → owned CandidateSet
Graph:       TokenStreamRef + CandidateSetRef → owned Graph
Transform:   &mut Graph                (borrow + mutate)
Teleport:    TokenStreamRef + CandidateSetRef → owned Option<TeleportVector>
Rank:        &Graph + Option<&TeleportVector> → owned RankOutput
Phrases:     all refs → owned PhraseSet
Format:      all refs → owned FormattedResult
```

String data remains interned (`StringPool` + `u32` IDs) through stages 0–4. Materialization to `String` only happens in the Format stage, keeping the hot path cache-friendly and allocation-light.

---

## Variant Composition Matrix

Each algorithm variant is a concrete composition of stages. The `Pipeline` struct is generic over all eight stage types and each variant is a type alias (e.g., `BaseTextRankPipeline`, `TopicRankPipeline`).

### Word-Graph Family

These variants use word-level candidates, co-occurrence graphs, and chunk-based phrase extraction. They differ only in graph configuration and teleport strategy.

| Stage | BaseTextRank | PositionRank | BiasedTextRank | SingleRank | TopicalPageRank |
|-------|-------------|-------------|---------------|------------|-----------------|
| Preprocess | Noop | Noop | Noop | Noop | Noop |
| Candidates | WordNode | WordNode | WordNode | WordNode | WordNode |
| Graph | Window (sentence-bounded, binary) | Window (sentence-bounded, binary) | Window (sentence-bounded, binary) | Window (**cross-sentence, count-accum**) | Window (**cross-sentence, count-accum**) |
| Transform | Noop | Noop | Noop | Noop | Noop |
| Teleport | Uniform | **Position** | **FocusTerms** | Uniform | **TopicWeights** |
| Rank | PageRank | PageRank | PageRank | PageRank | PageRank |
| Phrases | Chunk | Chunk | Chunk | Chunk | Chunk |
| Format | Standard | Standard | Standard | Standard | Standard |

**Key differences:**

- **BaseTextRank**: Baseline. Sentence-bounded window, binary edges, uniform teleport.
- **PositionRank**: Teleport biased by position — earlier words get higher teleport probability (`weight = 1/(position+1)`).
- **BiasedTextRank**: Teleport biased toward user-specified focus terms with configurable `bias_weight`.
- **SingleRank**: Window crosses sentence boundaries and accumulates co-occurrence counts as edge weights.
- **TopicalPageRank**: SingleRank's graph + per-word topic weights (e.g., from LDA) as teleport bias.

### Topic Family

These variants use phrase-level candidates, hierarchical agglomerative clustering (HAC), and cluster-aware graphs.

| Stage | TopicRank | MultipartiteRank |
|-------|-----------|------------------|
| Preprocess | Noop | Noop |
| Candidates | **PhraseCandidate** | **PhraseCandidate** |
| Clustering | **JaccardHAC (threshold=0.25)** | **JaccardHAC (threshold=0.26)** |
| Graph | **TopicGraph** (clusters as nodes) | **CandidateGraph** (candidates as nodes) |
| Transform | Noop | **Multipartite** (remove intra-cluster + alpha boost) |
| Teleport | Uniform | Uniform |
| Rank | PageRank | PageRank |
| Phrases | **TopicRepresentative** (first-occurring per cluster) | **Multipartite** (highest-score per lemma group) |
| Format | Standard | Standard |

**Key differences:**

- **TopicRank**: Clusters candidates into topics (HAC with Jaccard distance). Builds a graph where clusters are nodes. Picks the first-occurring phrase as cluster representative.
- **MultipartiteRank**: Same clustering, but keeps individual candidates as graph nodes. Removes intra-cluster edges (k-partite structure) and boosts edges toward first-occurring variants (`alpha=1.1` by default).

### Extractive Summarization

| Stage | SentenceRank |
|-------|-------------|
| Preprocess | Noop |
| Candidates | **SentenceCandidate** |
| Graph | **SentenceGraph** (Jaccard similarity between sentences) |
| Transform | Noop |
| Teleport | Uniform |
| Rank | PageRank |
| Phrases | **SentencePhrase** (materializes full sentence text) |
| Format | **SentenceFormatter** (optional position-based sort) |

Feature-gated behind `sentence-rank` (enabled by default). The `SentenceFormatter` supports two ordering modes: by score (default) or by document position (for readable summaries).

---

## Determinism Modes

The pipeline supports two execution modes via `TextRankConfig::determinism`:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `Default` | Fastest. HashMap iteration order is non-deterministic. Parallel reductions permitted. | Production latency-sensitive workloads |
| `Deterministic` | Reproducible. Sorts keys at graph construction, phrase grouping, and result formatting. | Research, testing, golden-test CI |

### Enforcement Points

1. **Graph construction** (`WindowGraphBuilder`): Deterministic mode sorts candidate keys before iterating.
2. **Phrase grouping** (`phrase/extraction.rs`): Deterministic mode sorts group keys lexicographically.
3. **Result sorting** (`StandardResultFormatter`): Deterministic mode uses `Phrase::stable_cmp()` with multi-key tie-breaking: score → position → length → lemma (float epsilon: `1e-10`).

Golden tests in `src/lib.rs` run each variant 3 times with identical input and assert bit-exact output, catching determinism regressions in CI.

---

## Validation Rules

### Build-Time Validation (`ValidationEngine`)

The `ValidationEngine` runs declarative rules against a `PipelineSpec` before construction. It never short-circuits — all errors are collected before reporting.

| Rule | Code | Description |
|------|------|-------------|
| `RankTeleportRule` | `invalid_combo` | Ensures ranking stage has compatible teleport configuration |
| `TopicGraphDepsRule` | `missing_stage` | Validates topic-family dependency chain: candidates → clustering → graph |
| `GraphTransformDepsRule` | `incompatible_modules` | Ensures graph transforms have the required graph structure |
| `RuntimeLimitsRule` | `limit_exceeded` | Validates numeric limits (max tokens, etc.) are within range |
| `UnknownFieldsRule` | `unknown_field` | Warns or errors on unrecognized fields (strict mode) |

Each finding is a `ValidationDiagnostic` with severity (error vs. warning), JSON-pointer path, and optional hint.

### Runtime Config Validation (`TextRankConfig::validate()`)

Called before pipeline execution. Validates:

| Field | Rule |
|-------|------|
| `damping` | Must be in `[0.0, 1.0]` |
| `max_iterations` | Must be `> 0` |
| `convergence_threshold` | Must be `> 0.0` |
| `window_size` | Must be `≥ 2` |
| `min_phrase_length` | Must be `> 0` |
| `max_phrase_length` | Must be `≥ min_phrase_length` |

---

## Error Codes

All pipeline errors carry an `ErrorCode` for programmatic handling. Codes are `#[non_exhaustive]` for forward compatibility and serialize to `snake_case` strings.

| Code | Serialized | When |
|------|-----------|------|
| `MissingStage` | `missing_stage` | Required pipeline stage absent from spec |
| `InvalidCombo` | `invalid_combo` | Conflicting configuration options |
| `ModuleUnavailable` | `module_unavailable` | Referenced module not available (e.g., feature-gated) |
| `LimitExceeded` | `limit_exceeded` | Numeric value out of allowed range |
| `UnknownField` | `unknown_field` | Unrecognized field in spec |
| `InvalidValue` | `invalid_value` | Field value invalid for type or context |
| `IncompatibleModules` | `incompatible_modules` | Selected modules cannot work together |
| `ValidationFailed` | `validation_failed` | General validation failure |
| `StageFailed` | `stage_failed` | Pipeline stage crashed during execution |
| `ConvergenceFailed` | `convergence_failed` | PageRank did not converge within allowed iterations |

### Error Types

| Type | Phase | Fields |
|------|-------|--------|
| `PipelineSpecError` | Build-time | `code`, `path` (JSON pointer), `message`, `hint` |
| `PipelineRuntimeError` | Execution | `code`, `path`, `stage`, `message`, `hint` |
| `TextRankError` | Legacy API | Enum variants: `EmptyInput`, `NoCandidates`, `ConvergenceFailure`, `InvalidConfig`, `Serialization`, `Internal` |

Note: PageRank convergence failure is **non-fatal** by design. Results are still returned with `RankOutput::converged = false`. Callers can check this flag and decide whether partial results are acceptable.

---

## Key Implementation Files

| File | Role |
|------|------|
| `src/pipeline/runner.rs` | Pipeline struct, type aliases, factory methods, stage orchestration |
| `src/pipeline/traits.rs` | All stage trait definitions and implementations (~2300 lines) |
| `src/pipeline/artifacts.rs` | Artifact types: `TokenStream`, `CandidateSet`, `Graph`, `RankOutput`, etc. |
| `src/pipeline/error_code.rs` | `ErrorCode` enum |
| `src/pipeline/errors.rs` | `PipelineSpecError`, `PipelineRuntimeError` |
| `src/pipeline/validation.rs` | `ValidationEngine` and rule implementations |
| `src/pipeline/spec.rs` | Declarative `PipelineSpec`, preset resolution |
| `src/pipeline/observer.rs` | `PipelineObserver` trait for debug/profiling hooks |
| `src/pagerank/standard.rs` | Standard PageRank (power iteration) |
| `src/pagerank/personalized.rs` | Personalized PageRank |
| `src/graph/csr.rs` | CSR graph representation |
| `src/phrase/extraction.rs` | Chunk-based phrase extraction, scoring, grouping |
| `src/types.rs` | Core public types: `Token`, `Phrase`, `TextRankConfig`, `DeterminismMode` |
| `src/variants/` | Per-variant structs and `extract_with_info()` entry points |

---

## Design Principles

1. **Static dispatch everywhere**: All stage traits are resolved at compile time. Zero-sized defaults (`NoopPreprocessor`, `NoopGraphTransform`, `UniformTeleportBuilder`) add zero bytes and zero cost.

2. **Borrow then own**: Stages receive borrowed views (`TokenStreamRef<'_>`, `CandidateSetRef<'_>`) and return owned artifacts. No unnecessary cloning on the hot path.

3. **Deferred materialization**: String data stays interned (`StringPool` + `u32` IDs) through the entire pipeline. Human-readable `String` values are only produced in the final Format stage.

4. **CSR for graphs**: Compressed Sparse Row format avoids per-edge allocations. Edge weights can be modified in place by transforms but new edges cannot be added (by design — the structure is frozen after construction).

5. **Workspace reuse**: `PipelineWorkspace` lets batch callers amortize PageRank buffer allocations across documents. `run_batch()` automates this pattern.
