Use 'bd' for task tracking

## Build

This project uses PyO3 to bind Rust to Python. The local venv runs Python 3.14.

### Required environment variable

Always set this before any `cargo build`, `cargo test`, or `maturin develop`:

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

Without it, PyO3 refuses to build against Python 3.14 (newer than its compiled ABI).

### Running tests

Use `cargo test --lib` — **not** bare `cargo test`. Full `cargo test` attempts to link integration and benchmark test binaries as a Python dylib, which fails at link time with undefined `_Py*` symbols because the Python development library is not on the linker path. `--lib` runs only unit tests inside `src/` and avoids the dylib link entirely.

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --lib
```

### Known test failures

Two pre-existing tests in `python::json::tests` (`test_json_include_pos_filtering` and `test_json_include_pos_multiple_tags`) fail. These are not caused by recent changes — they are a known issue in the POS-filtering logic. Do not let them block your work.

### Feature flags

The `sentence-rank` feature (enabled by default) gates the SentenceRank extractive summarization module. To test without it:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --lib --no-default-features --features python
```

For complete test coverage across all features:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test --lib --all-features
```

### Pre-commit CI checks

CI runs these checks on every push. Run them locally before committing:

1. **Formatting** — `cargo fmt --all -- --check`. Always run `cargo fmt --all` after modifying Rust files.
2. **Benchmark compilation** — `cargo check --benches`. The `--lib` test flag skips benchmarks, so adding/changing struct fields can silently break `benches/benchmark.rs`. Always check it compiles.
3. **Exports manifest** — every public PyO3 class/function registered in `src/python/mod.rs` must also appear in `exports.toml` **and** `python/rapid_textrank/__init__.py` (both import and `__all__`). CI runs a bidirectional check (`test_all_rust_symbols_match_manifest`).

Quick local validation:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo fmt --all && cargo check --benches && cargo test --lib
```

### Python wheel build

```bash
source .venv/bin/activate
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```
