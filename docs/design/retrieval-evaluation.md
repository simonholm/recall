# Retrieval Evaluation

## Purpose

Recall needs a repeatable retrieval benchmark before ranking or search behavior
changes. The benchmark should answer one question:

> Given a real user question, does the current retrieval pipeline place the
> expected evidence high enough in the result list?

This is measurement infrastructure only. It must not improve retrieval, tune
scores, change adapter behavior, or add user-facing search features.

## Proposed Directory Structure

```text
tests/
  retrieval/
    cases.json
    fixtures/
      codex/
      git/
```

`tests/retrieval/cases.json` is the durable corpus of evaluation cases.
`tests/retrieval/fixtures/` should hold source fixtures when the runner needs
stable adapter input. Live local Codex sessions or the current Git checkout are
useful for manual smoke checks, but they are not a reproducible benchmark.

## Evaluation Case Format

Use JSON because the workspace already depends on `serde_json`, the format is
easy to extend, and it keeps the corpus language-agnostic if the runner later
moves outside Rust.

Each case contains:

- `id`: stable case identifier.
- `question`: the user-facing question.
- `expected`: one or more relevant source-qualified event references.
- `notes`: optional explanation of why those results are expected.

Expected results are source-qualified because Recall event ids are source-local.
Multiple expected entries are all required. An expected entry can set
`max_rank` to define the acceptable rank threshold and can list `alternatives`
when more than one event can satisfy the same requirement.

```json
{
  "id": "eventref-rationale",
  "question": "Why did I introduce EventRef?",
  "expected": [
    {
      "source": "codex",
      "id": "fixture-codex-eventref",
      "max_rank": 5,
      "why": "The Codex session discusses replacing embedded events with source-qualified references."
    },
    {
      "source": "git",
      "id": "fixture-git-eventref",
      "max_rank": 5,
      "why": "The commit records the landed core model change."
    }
  ],
  "notes": "Lexical retrieval should find durable terms such as eventref and introduce."
}
```

The first implementation should keep matching exact: a retrieved result matches
an expected result only when both `source` and `id` match.

## Evaluation Runner Design

The runner should:

1. Load `tests/retrieval/cases.json`.
2. Build a `Recall` instance with deterministic fixture-backed adapters.
3. For each case, execute the same retrieval path used by `recall ask` before
   prompt construction:
   - normalize the question into the ask search query,
   - call `Recall::search()`,
   - inspect expected result ranks from the ordered `SearchResult` list.
4. Print per-case output:
   - case id,
   - question,
   - normalized query,
   - retrieved result refs and scores,
   - rank of each expected result,
   - pass/fail.
5. Print summary metrics:
   - total cases,
   - Top-1 hit rate,
   - Top-3 hit rate,
   - Top-5 hit rate,
   - missing expected count.

Pass/fail should initially be Top-5. Top-1 and Top-3 are still reported so
future ranking changes can be compared without redefining the benchmark.

The output should be deterministic plain text for human review, with an optional
JSON output added later only if automated comparison needs it.

## Architecture Placement

The evaluation should live under top-level `tests/retrieval/` rather than inside
one adapter crate.

Reasons:

- Retrieval quality is a cross-source contract. `Recall::search()` combines
  adapter results, so the benchmark should not belong solely to `recall-codex`,
  `recall-git`, or `recall-cli`.
- The corpus is product evidence, not unit-test scaffolding. Keeping it at the
  workspace test boundary makes it visible when changing ranking behavior.
- The fixture-backed runner can exercise public crate APIs without changing the
  narrow `Adapter` trait.
- Future retrieval algorithms can be compared against the same cases by swapping
  only the retrieval implementation under test, not the expected-answer data.

Do not put this in `recall-cli` as a public `recall eval` command yet. That
would create a user-facing feature before the evaluation contract is stable.

## Approach Comparison

### Unit Tests

Unit tests are good for local invariants such as score sorting, query
normalization, or exact adapter parsing. They are too small for retrieval
quality because they usually test one function and one synthetic input at a
time.

Use unit tests to protect runner helpers, not as the benchmark itself.

### Integration Test

A Rust integration test can load the corpus, run deterministic adapters, and
fail when expected evidence drops below the required rank. This is the best
first implementation because it stays inside the existing Cargo workflow and
does not add a public CLI surface.

Recommended first runner:

```text
crates/core/tests/retrieval_eval.rs
```

If the runner needs real `recall-codex` or `recall-git` adapters, add a small
workspace-level eval crate later instead of adding cross-crate dev-dependencies
prematurely.

### Standalone Benchmark Binary

A standalone binary is useful once multiple retrieval strategies exist and the
project needs richer reports or non-failing comparisons. It is too early for
the first benchmark because it adds another executable surface and command
contract.

### Snapshot Tests

Snapshot tests are useful for stable report formatting. They are a poor primary
metric because retrieval quality should be measured by ranks and hit rates, not
large output diffs.

## Recommendation

Start with a corpus plus an integration-style evaluation test that reports ranks
and fails only on the agreed Top-5 threshold. Keep the runner small and direct:
load cases, run retrieval, compute ranks, print metrics.

When a second retrieval implementation exists, introduce a small runner
interface with one responsibility:

```rust
fn retrieve(question: &str) -> Result<Vec<SearchResult>, EvalError>
```

Do not add that abstraction until there are at least two concrete retrieval
implementations to compare.

## Recommended Implementation Order

1. Land `tests/retrieval/cases.json` with a small representative corpus.
2. Add deterministic fixture data for the sources used by those cases.
3. Add a Cargo integration test that loads the corpus and computes Top-1,
   Top-3, and Top-5 metrics.
4. Make the initial threshold explicit, preferably Top-5.
5. Run and record the baseline before changing ranking.
6. Only after the baseline exists, modify retrieval behavior in separate
   commits or tasks.
