# Production Hardening

rapid_textrank ships five features designed for operators running keyword extraction at scale. Together they form a layered defense: catch typos before they silently change behavior, preflight specs before committing compute, declare resource budgets, guarantee reproducible output, and negotiate feature support at runtime.

---

## Quick Reference

| Feature | Config Field | Default | Purpose |
|---------|-------------|---------|---------|
| [Strict Mode](#strict-mode) | `strict: bool` | `false` | Reject unknown fields |
| [Validate-Only](#validate-only-mode) | `validate_only: bool` | `false` | Preflight check without extraction |
| [Runtime Limits](#runtime-limits) | `runtime.max_*` | `None` | Declare resource budgets |
| [Deterministic Mode](#deterministic-mode) | `determinism: "deterministic"` | `"default"` | Reproducible output |
| [Capability Discovery](#capability-discovery) | `capabilities: bool` | `false` | Query supported features |

---

## Strict Mode

Strict mode controls whether unrecognized fields in a pipeline spec are errors or warnings.

### The Problem

By default, serde silently ignores unknown JSON fields. A typo like `"preest": "textrank"` (instead of `"preset"`) would be silently dropped — the spec parses successfully but behaves differently than intended.

### How It Works

Rather than using serde's `#[serde(deny_unknown_fields)]` (which hard-fails at deserialization), rapid_textrank uses `#[serde(flatten)]` with a `HashMap<String, Value>` to **capture** unknown fields. The `UnknownFieldsRule` validation rule then inspects these maps and emits diagnostics whose severity depends on the `strict` flag:

- **`strict: false`** (default) — unknown fields produce **warnings**. Validation succeeds.
- **`strict: true`** — unknown fields produce **errors**. Validation fails.

Unknown field detection covers three locations:

| Location | JSON Pointer Prefix |
|----------|-------------------|
| Top-level spec | `/` |
| Module set | `/modules/` |
| Runtime spec | `/runtime/` |

### Configuration

```json
{
  "v": 1,
  "strict": true,
  "preset": "textrank"
}
```

### Example: Catching a Typo

```json
{
  "v": 1,
  "strict": true,
  "preest": "textrank"
}
```

Validation response:

```json
{
  "valid": false,
  "diagnostics": [
    {
      "severity": "error",
      "code": "unknown_field",
      "path": "/preest",
      "message": "unrecognized field \"preest\"",
      "hint": "Check spelling or remove this field"
    }
  ]
}
```

Without `strict: true`, the same spec would produce a **warning** and `"valid": true` — the spec executes but `preest` is ignored.

### Recommendation

Always set `strict: true` in production. The default is `false` only for backward compatibility.

---

## Validate-Only Mode

Validate-only mode runs spec validation and returns a report **without** performing extraction. Use it as a preflight check in CI pipelines, deployment gates, or config editors.

### How It Works

Set `validate_only: true` on the input document. The system:

1. Resolves the preset (if any) and merges module overrides.
2. Runs all validation rules.
3. Returns a `ValidationResponse` immediately — no tokens are processed.

The `tokens` field is **not required** when `validate_only` is true.

### Configuration

```json
{
  "validate_only": true,
  "pipeline": {
    "v": 1,
    "strict": true,
    "preset": "position_rank",
    "runtime": { "max_threads": 4 }
  }
}
```

### Response

```json
{
  "valid": true,
  "diagnostics": []
}
```

Or, when validation fails:

```json
{
  "valid": false,
  "diagnostics": [
    {
      "severity": "error",
      "code": "missing_stage",
      "path": "/modules/teleport",
      "message": "personalized_pagerank requires a teleport module",
      "hint": "Add a teleport module: position, focus_terms, topic_weights, or uniform"
    }
  ]
}
```

### Requirements

A `pipeline` field **must** be present. If missing:

```
"validate_only requires a 'pipeline' field"
```

### Processing Priority

When multiple flags are set on a document, the system resolves them in this order:

```
capabilities (1st) → validate_only (2nd) → extraction (3rd)
```

If `capabilities: true` and `validate_only: true` are both set, the capabilities response wins.

### Validation Rules

The default `ValidationEngine` runs five rules. All rules run — the engine never short-circuits on the first error, so users see every problem at once.

| Rule | Checks | Error Code |
|------|--------|------------|
| `rank_teleport` | `personalized_pagerank` requires a teleport module | `missing_stage` |
| `topic_graph_deps` | `topic_graph` / `candidate_graph` require clustering + phrase_candidates | `missing_stage`, `invalid_combo` |
| `graph_transform_deps` | `remove_intra_cluster_edges` requires clustering | `missing_stage` |
| `runtime_limits` | Numeric limits must be > 0 when set | `limit_exceeded` |
| `unknown_fields` | Unrecognized fields (strict → error, non-strict → warning) | `unknown_field` |

### Custom Rules (Rust API)

Implement the `ValidationRule` trait to add project-specific checks:

```rust
use rapid_textrank::pipeline::validation::*;
use rapid_textrank::pipeline::errors::PipelineSpecError;
use rapid_textrank::pipeline::error_code::ErrorCode;

struct MaxDampingRule;

impl ValidationRule for MaxDampingRule {
    fn name(&self) -> &str { "max_damping" }

    fn validate(&self, spec: &PipelineSpecV1) -> Vec<ValidationDiagnostic> {
        // Your custom check here
        vec![]
    }
}

let mut engine = ValidationEngine::with_defaults();
engine.add_rule(Box::new(MaxDampingRule));
let report = engine.validate(&spec);
```

---

## Runtime Limits

The `RuntimeSpec` declares resource budgets for pipeline execution.

### Available Limits

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | `Option<usize>` | `None` | Maximum input token count |
| `max_nodes` | `Option<usize>` | `None` | Maximum graph node count |
| `max_edges` | `Option<usize>` | `None` | Maximum graph edge count |
| `max_threads` | `Option<usize>` | `None` | Maximum Rayon thread pool size |
| `single_thread` | `bool` | `false` | Force single-threaded execution |
| `max_debug_top_k` | `Option<usize>` | `None` | Limit debug node_scores output size |
| `deterministic` | `Option<bool>` | `None` | Request deterministic execution |

### Configuration

```json
{
  "v": 1,
  "preset": "textrank",
  "runtime": {
    "max_tokens": 200000,
    "max_nodes": 50000,
    "max_edges": 1000000,
    "max_threads": 4,
    "max_debug_top_k": 100
  }
}
```

### Validation

Setting any numeric limit to `0` produces a validation error:

```json
{
  "code": "limit_exceeded",
  "path": "/runtime/max_tokens",
  "message": "max_tokens must be greater than 0",
  "hint": "Remove max_tokens to disable the limit, or set it to a positive value"
}
```

Omitting a limit (`None`) disables it — no validation error.

### Threading Controls

Threading limits **are enforced** at runtime:

- `single_thread: true` — overrides `max_threads`, forces `num_threads(1)`.
- `max_threads: N` — creates a scoped Rayon thread pool with N threads.
- Both absent — uses the global Rayon pool (all logical cores).

Resolution order:

```
single_thread: true  →  effective_threads() = Some(1)
max_threads: N       →  effective_threads() = Some(N)
neither              →  effective_threads() = None (global pool)
```

The scoped pool is constructed via `RuntimeSpec::build_thread_pool()` and applied via `RuntimeSpec::scoped(|| { ... })`, ensuring all `par_iter()` calls within the closure use the constrained pool.

### Current Status: Resource Limits

> **Important:** `max_tokens`, `max_nodes`, and `max_edges` are **validated** (must be > 0 when set) but **not yet enforced** during pipeline execution. They serve as declarative annotations today — a pipeline with `max_tokens: 10000` will still process 100,000 tokens without error.
>
> Threading controls (`max_threads`, `single_thread`) **are** enforced.

---

## Deterministic Mode

Deterministic mode forces every pipeline stage to produce identical output across runs, at a potential throughput cost.

### The Problem

By default, the library uses the fastest available code path:

- Hash map iteration order varies across runs (Rust's random hash seed).
- Parallel reductions may accumulate floating-point values in different orders.
- Tie-breaking in ranking may not be stable.

This is fine for most use cases, but problematic when you need bit-exact reproducibility — for regression testing, audit trails, or reproducible research.

### How It Works

`DeterminismMode` is a two-variant enum:

```rust
#[derive(Default)]
pub enum DeterminismMode {
    #[default]
    Default,        // Fastest — non-deterministic iteration and reductions
    Deterministic,  // Stable — reproducible across runs and machines
}
```

When `Deterministic` is active, stages must:

- Use **sorted keys** instead of hash-map iteration order.
- Build CSR graphs in a **stable node order**.
- Apply **deterministic reductions** (no parallel non-deterministic sums).
- Use a **stable tie-breaker comparator** for final ranking (lexicographic by lemma).

### Configuration

**Via `TextRankConfig` (Rust / native Python):**

```python
from rapid_textrank import TextRankConfig

config = TextRankConfig(determinism="deterministic")
```

**Via `PipelineSpec` (JSON interface):**

```json
{
  "v": 1,
  "preset": "textrank",
  "runtime": {
    "deterministic": true
  }
}
```

**Via Rust API:**

```rust
use rapid_textrank::DeterminismMode;

let cfg = TextRankConfig::default();
// cfg.determinism = DeterminismMode::Deterministic;
```

### Affected Stages

| Stage | Default Behavior | Deterministic Behavior |
|-------|-----------------|----------------------|
| Graph build | `FxHashMap` iteration order | Sorted (lemma, POS) node order |
| Phrase grouping | Hash-map key order | Deterministic key ordering |
| Result formatting | Score descending, unstable ties | Score descending, ties broken by lemma ascending |

### Performance Impact

Deterministic mode adds sorting overhead proportional to the number of graph nodes and phrase groups. For typical documents (< 1000 unique candidates), the overhead is negligible. For very large documents (10K+ candidates), expect 5–15% throughput reduction.

### Serialization

| Mode | JSON | Serde |
|------|------|-------|
| Default | `"default"` | `DeterminismMode::Default` |
| Deterministic | `"deterministic"` | `DeterminismMode::Deterministic` |

---

## Capability Discovery

Capability discovery returns a static response describing what this build of rapid_textrank supports — versions, presets, and available modules per stage. Use it to negotiate feature support at runtime without attempting extraction.

### How It Works

Set `capabilities: true` on the input document. No `tokens`, `config`, or `pipeline` fields are needed.

### Configuration

```json
{
  "capabilities": true
}
```

### Response

```json
{
  "version": "0.7.0",
  "pipeline_spec_versions": [1],
  "presets": [
    "textrank",
    "position_rank",
    "biased_textrank",
    "single_rank",
    "topical_pagerank",
    "topic_rank",
    "multipartite_rank",
    "sentence_rank"
  ],
  "modules": {
    "preprocess": ["default"],
    "candidates": ["word_nodes", "phrase_candidates", "sentence_candidates"],
    "graph": ["cooccurrence_window", "topic_graph", "candidate_graph", "sentence_graph"],
    "graph_transforms": ["remove_intra_cluster_edges", "alpha_boost"],
    "teleport": ["uniform", "position", "focus_terms", "topic_weights"],
    "clustering": ["hac"],
    "rank": ["standard_pagerank", "personalized_pagerank"],
    "phrases": ["chunk_phrases", "sentence_phrases"],
    "format": ["standard_json", "sentence_json"]
  }
}
```

### Feature Gating

Some modules are conditionally compiled behind Cargo feature flags:

| Module | Feature Flag | Present When |
|--------|-------------|-------------|
| `sentence_candidates` | `sentence-rank` | Feature enabled (default) |
| `sentence_graph` | `sentence-rank` | Feature enabled (default) |
| `sentence_phrases` | `sentence-rank` | Feature enabled (default) |
| `sentence_json` | `sentence-rank` | Feature enabled (default) |
| `sentence_rank` (preset) | `sentence-rank` | Feature enabled (default) |

When `sentence-rank` is not compiled in, these entries are absent from the response.

### Use Cases

- **Client negotiation:** Check whether the server supports `sentence_rank` before sending a document with that preset.
- **Config UI generation:** Build a dropdown of available presets and module types from the response.
- **Version checking:** Compare the `version` field against known-good versions for deployment.

---

## Error Types

All production hardening features share two error types with a common structure.

### `PipelineSpecError` (Build-Time)

Found during spec validation, before execution begins.

```json
{
  "code": "missing_stage",
  "path": "/modules/teleport",
  "message": "personalized_pagerank requires a teleport module",
  "hint": "Add a teleport module: position, focus_terms, topic_weights, or uniform"
}
```

Display format: `[missing_stage] /modules/teleport: personalized_pagerank requires a teleport module`

### `PipelineRuntimeError` (Execution-Time)

Failures during pipeline execution.

```json
{
  "code": "convergence_failed",
  "path": "/modules/rank",
  "stage": "rank",
  "message": "PageRank did not converge after 100 iterations",
  "hint": "Increase max_iterations or relax the convergence threshold"
}
```

Display format: `[convergence_failed] /modules/rank (stage: rank): PageRank did not converge after 100 iterations`

### Error Codes

All error codes are `#[non_exhaustive]` — new codes may be added in future versions. Always include a wildcard arm when matching.

| Code | Serialized | Meaning |
|------|-----------|---------|
| `MissingStage` | `"missing_stage"` | A required pipeline stage is absent |
| `InvalidCombo` | `"invalid_combo"` | Configuration options conflict |
| `ModuleUnavailable` | `"module_unavailable"` | Requested module not available in this build |
| `LimitExceeded` | `"limit_exceeded"` | Numeric value out of allowed range |
| `UnknownField` | `"unknown_field"` | Unrecognized field name in spec |
| `InvalidValue` | `"invalid_value"` | Field value invalid for its context |
| `IncompatibleModules` | `"incompatible_modules"` | Selected modules cannot work together |
| `ValidationFailed` | `"validation_failed"` | General validation failure |
| `StageFailed` | `"stage_failed"` | Pipeline stage crashed during execution |
| `ConvergenceFailed` | `"convergence_failed"` | PageRank did not converge |

---

## Build-Validate-Run Lifecycle

The `SpecPipelineBuilder` integrates validation into a three-phase lifecycle:

```
Resolve → Validate → Build
```

1. **Resolve** — `resolve_spec(&spec)` merges the preset with module overrides to produce a fully-specified `PipelineSpecV1`.
2. **Validate** — `ValidationEngine::with_defaults().validate(&effective)` runs all rules. If any error is found, `build_from_spec` returns early with the first error.
3. **Build** — Each module spec is mapped to a concrete stage implementation, producing a `DynPipeline` ready to execute.

```rust
let pipeline = SpecPipelineBuilder::new()
    .with_chunks(chunks)
    .build_from_spec(&spec, &cfg)?;  // resolve + validate + build

let result = pipeline.run(stream, &cfg, &mut obs);
```

The validate-only JSON path (`validate_only: true`) performs steps 1–2 and returns the report, skipping step 3 entirely.

---

## Putting It All Together

A production-grade pipeline spec combining all hardening features:

```json
{
  "validate_only": false,
  "pipeline": {
    "v": 1,
    "strict": true,
    "preset": "position_rank",
    "runtime": {
      "max_tokens": 200000,
      "max_nodes": 50000,
      "max_edges": 1000000,
      "max_threads": 4,
      "deterministic": true,
      "max_debug_top_k": 50
    },
    "expose": {
      "graph_stats": true,
      "stage_timings": true
    }
  },
  "tokens": [...]
}
```

This spec:

1. **Strict mode** catches any typos in field names.
2. **Runtime limits** declare resource expectations (validated, not yet enforced).
3. **Threading** is constrained to 4 threads.
4. **Deterministic** output is guaranteed.
5. **Debug output** includes graph stats and stage timings for observability.

To preflight this spec before deploying:

```json
{
  "validate_only": true,
  "pipeline": {
    "v": 1,
    "strict": true,
    "preset": "position_rank",
    "runtime": {
      "max_tokens": 200000,
      "max_threads": 4,
      "deterministic": true
    }
  }
}
```

To check what the server supports:

```json
{
  "capabilities": true
}
```
