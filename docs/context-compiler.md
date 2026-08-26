# Context Compiler

## Purpose

The Context Compiler transforms retrieved evidence into a compact,
deterministic project handoff.

Retrieval answers:

```text
What information is relevant?
```

The compiler answers:

```text
What information should be remembered?
```

This document describes the compiler only. It is not a general Recall
architecture document and should not define adapter, retrieval, ranking, or LLM
provider behavior.

## Mission

The compiler should preserve durable project state:

- current objective
- decisions
- milestones
- implementation status
- blockers
- planned next steps

The compiler should aggressively compress or discard transient transcript
detail:

- repeated debugging
- raw console output
- transcript history
- duplicated discussions
- superseded plans

The intended output is closer to a project handoff than a chronological replay
of retrieved sessions.

## Guiding Principles

Keep semantic inference as far to the right as possible.

Everything before optional summarization should remain deterministic,
benchmarkable, and reproducible. Categorization, retention priority,
deduplication, and project-state construction should be explainable from source
text and metadata without a model call.

Retrieval and compilation have separate responsibilities. Retrieval selects
candidate evidence. Compilation decides what retrieved evidence is worth
retaining in prompt context.

Introduce abstractions only after they solve demonstrated duplication or
complexity. The compiler should evolve through small, testable steps rather
than by adding speculative long-term types.

## Planned Pipeline

```text
Retrieved Events
    |
    v
Categorize
    |
    v
Retention Priority
    |
    v
Deterministic Deduplication
    |
    v
ProjectState
    |
    v
(Optional semantic summarization)
    |
    v
PromptBuilder
```

## Responsibilities

The compiler is responsible for:

- categorizing retrieved evidence into durable context signals
- deriving retention priority from those categories
- removing duplicate or redundant evidence with deterministic rules
- grouping retained evidence by project
- producing compact project handoff sections for prompt construction
- enforcing configurable context budgets by compressing lower-priority content
  before dropping high-priority content

The compiler is not responsible for:

- changing retrieval behavior
- ranking search results
- calling an LLM
- adapter-specific parsing beyond source-neutral event metadata
- provider-specific prompt or transport behavior

## Invariants

`Event` remains source-neutral. It represents evidence returned by adapters, not
compiler policy.

Compiler policy must not leak into adapters. Adapters should not need to know
whether text will later be retained as a decision, milestone, blocker, or next
step.

`CompiledEvent` is a compiler-internal representation. It may carry compiler
metadata such as categories, project keys, normalized text, and retained body
text, but it should not become part of the adapter contract.

Events may contain multiple independent signals. The initial categorization
shape should support `Vec<ContextCategory>` or an equivalent multi-category
representation.

Retention priority is derived from categories rather than stored as independent
state. Storing both creates duplicate state that can become inconsistent.

There is no dedicated `Noise` category. An event with no meaningful categories
is simply uncategorized and can be discarded or retained only as budget allows.

Semantic summarization is optional and occurs only after deterministic
project-state construction. A model may compress already-structured state, but
it should not be required to decide the basic compiler output.

Project grouping should prefer structural signals before text inference:
repository metadata, `cwd`, commit identifiers, timestamps, and source ids.

Ordering should be stable. Given the same retrieved evidence and compiler
options, the compiled context should be essentially the same across runs.

## Categories

Initial categories should be lightweight and deterministic:

- `Decision`
- `Milestone`
- `Todo`
- `Blocker`
- `Validation`
- `NextStep`

An event can have more than one category. For example, a single Codex session
may record an accepted design, a completed implementation milestone, and the
next planned step.

## Retention Priority

Retention priority should be computed from categories.

Highest priority:

- decisions
- blockers
- next steps

Medium priority:

- milestones
- implementation status
- validation attached to retained work

Lowest priority:

- repeated debugging
- raw console output
- long transcripts
- duplicated discussion
- uncategorized evidence

Budget handling should compress or omit lower-priority material before removing
high-priority project state.

## Retention Rule Precedence

Transient-line rejection runs before positive retention rules. A line rejected
as prompt scaffolding, a section header, process narration, or validation
chatter must not be rescued later because it also matches a question term,
category signal, or other high-value pattern.

Interrogative lines need a narrower rule than "drop every question." The
existing corpus contains legitimate question-shaped project evidence, including
architecture review questions about staged compiler phases, `ProjectState`, and
deferred abstractions such as `StateFact`. The rejection rule should therefore
target prompt/template scaffolding specifically: fixed handoff prompts such as
"What is the current objective?", generated section prompts such as "What has
recently been completed?", and generic retrieval examples such as "What did I
work on today?". Project-specific questions that mention concrete modules,
design choices, repositories, commits, or architecture terms should remain
eligible for retention.

## Deterministic Deduplication

Deduplication should begin with conservative structural rules:

- same source id
- same commit identifier
- same normalized retained text
- same project key, title, and timestamp
- explicit supersedes markers

Conservative fuzzy matching can be added after exact rules if repeated
duplicates remain. Semantic similarity and LLM-based merging are deferred.

## ProjectState

`ProjectState` is the first intended output shape that stops treating each
retrieved event as an independent prompt block.

The initial project handoff should use fixed sections:

- current objective
- recent milestones
- implementation status
- architectural decisions
- planned next step
- outstanding issues

The first implementation can populate these sections from categorized retained
events. It does not need a richer fact model unless repeated extraction or
normalization logic makes that worthwhile.

## Deferred Decisions

Timestamp caching depends on the eventual `EventRef` cost model. It should be
decided from measurement rather than principle.

Richer abstractions such as `StateFact` should only be introduced if repeated
normalization or extraction logic demonstrates a clear need across
categorization, prioritization, deduplication, and project-state construction.

Semantic summarization should remain optional until deterministic project-state
construction is useful on its own.

## Rejected Alternatives

### Single Category

Rejected because events frequently contain multiple independent signals. A
single category would require artificial event splitting or would lose useful
priority information.

### Storing Retention Priority

Rejected because priority is derived from categories. Storing both category and
priority creates duplicate state that can drift out of sync.

### Noise Category

Rejected because the absence of meaningful categories is a cleaner invariant
than introducing a category representing no useful context.

### Immediate StateFact

Deferred because it is a larger abstraction. It should be introduced only after
compiler stages show repeated extraction or normalization logic that a
dedicated fact type would simplify.

### Semantic Merging Before ProjectState

Rejected for the deterministic compiler path. Semantic merging should happen
only after reproducible categorization, priority selection, deduplication, and
project-state construction.

### Adapter-Level Categories

Rejected because compiler policy should not leak into source adapters. Adapters
provide source-neutral evidence; the compiler decides what that evidence means
for prompt context.

## Phase Roadmap

1. Categorize compiled evidence.
2. Add retention priority and budget-aware selection.
3. Add exact deterministic deduplication.
4. Build initial `ProjectState` output.
5. Add merge rules if demonstrated duplication remains.
6. Add optional semantic summarization.

Each phase should be independently testable and should preserve deterministic
behavior for the same retrieved evidence.
