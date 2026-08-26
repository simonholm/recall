# Retrieval Planner

## Problem

`recall ask` currently treats every question as a lexical search. The ask
command normalizes the user's question, passes the resulting text to the
registered adapters, retrieves matching events, and builds prompt context from
those events.

This works for questions where the important words are likely to appear in the
source evidence, such as:

- Why did I introduce EventRef?
- When did I implement the Git adapter?

Those questions contain durable terms like `eventref`, `implement`, `git`, and
`adapter`. If those terms appear in Git commit messages or Codex session text,
the current adapter searches can find useful evidence.

The same approach fails for natural questions such as:

- What did I work on today?
- Summarize today's recall work.
- What changed this week?
- What have I been working on recently?

These questions combine temporal intent, project scope, and summarization
intent. They are not asking for events containing the literal words `today`,
`summarize`, `work`, or `recently`; they are asking Recall to select an
appropriate slice of the timeline and summarize it.

The investigation of:

```text
recall ask "Summarize today's recall work."
```

found that the query normalized to:

```text
summarize todays recall work
```

The Codex adapter searched 157 events. The Git adapter searched 10 events. Both
adapters returned zero matches.

Both adapters currently require every normalized term to appear in the indexed
text before an event can match. That means words like:

- summarize
- today/todays
- work
- recently

incorrectly become mandatory lexical search terms.

## Root Cause

This is not an LLM problem.

This is a retrieval-planning problem.

The current pipeline has no distinction between:

- lexical search
- temporal search
- project-scoped search
- summarization requests

Every question is reduced to one search string before Recall decides what kind
of evidence should be retrieved.

## Goals

Introduce an ask-only Retrieval Planner that decides how retrieval should be
performed before calling adapters.

Do not weaken `Adapter::search()`.

Preserve current keyword behavior.

The planner should improve natural `recall ask` questions while keeping
existing keyword searches deterministic and source-local.

## Proposed Architecture

```text
Question
    |
    v
RetrievalPlanner
    |
    v
RetrievalPlan
    |
    v
Adapters
    |
    v
PromptBuilder
    |
    v
LLM
```

The Retrieval Planner should sit between the raw user question and adapter
retrieval. It should classify the question, preserve useful lexical terms, and
extract retrieval constraints such as time ranges and project scope.

The adapters should remain responsible for source-specific search, timeline,
and inspection behavior. The planner should decide which adapter capabilities to
use for a given ask request.

## RetrievalPlan

A future implementation could introduce a plan shape similar to:

```rust
struct RetrievalPlan {
    lexical_terms: Vec<String>,
    temporal_range: Option<TemporalRange>,
    project_scope: Option<ProjectScope>,
    intent: RetrievalIntent,
}
```

Possible intent values:

- `Search`
- `Timeline`
- `Summary`
- `RecentChanges`

These names are provisional. The important architectural point is that the
retrieval plan should represent user intent separately from literal search
terms.

## Examples

`Why did I introduce EventRef?`

Interpret as lexical search. The useful term is `eventref`, with supporting
terms such as `introduce`. Current adapter keyword search remains a good fit.

`What did I work on today?`

Interpret as a timeline request. The word `today` should become a temporal
range, and `work` should not be required as a literal indexed term.

`Summarize this week's recall work.`

Interpret as timeline plus project scope plus summary. `this week` should
become a temporal range, `recall` should constrain results to the current
project or matching project metadata, and `summarize` should set summary intent
rather than become a search term.

`What changed recently?`

Interpret as a recent timeline request. `recently` should map to a bounded
recency window, and `changed` should influence ranking or prompt framing without
requiring every selected event to contain the literal word.

## Incremental Implementation Plan

Phase 1:

- introduce `RetrievalPlanner`
- no adapter changes

Phase 2:

- detect temporal phrases
- call `Recall::timeline()`

Phase 3:

- infer current project scope
- filter timeline

Phase 4:

- combine timeline and lexical ranking

## Non-goals

Do not:

- add embeddings
- replace keyword search
- redesign adapters

## Future Extensions

Possible future capabilities include:

- richer temporal understanding
- project inference
- cross-repository queries
- benchmark-aware retrieval
- ChatGPT/Codex session integration
