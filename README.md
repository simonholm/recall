# Recall

Recall is an experimental source-agnostic engineering memory tool that
retrieves evidence from local development history before any LLM reasoning.

## Project Status

Recall is an ongoing experimental project that is functional end-to-end and
developed periodically as its retrieval and evidence-compilation approach is
evaluated against real development history.

The project is not considered stable. Data formats, behavior, and internal APIs
may change between revisions as the design is tested and simplified.

## Current Status

`recall ask` can retrieve and compile evidence from Git history and local Codex
sessions, including explicit calendar-date and single-day timeline questions.
Git evidence is preserved through compilation and reaches the generated prompt
as direct evidence.

The main area still under investigation is Codex evidence quality. Codex
sessions are currently flattened early, which requires `ContextCompiler` to
compensate with heuristics. Experiments with preserving more session structure
have produced cleaner intermediate evidence, and this remains part of the
project's ongoing evaluation rather than a settled design.

## Motivation

Engineering context is often fragmented across chats, commits, terminal
sessions, and local notes. Long context windows degrade over time, and replaying
entire histories is noisy. Recall is built around retrieving the relevant
evidence first, then using that evidence as explicit context for later
reasoning.

## Current Features

- Search across supported sources.
- Event inspection with source-qualified ids.
- Timeline aggregation across registered adapters.
- Git adapter for current-repository commit history.
- Codex session adapter for local Codex JSONL session files.
- Experimental `recall ask` command.
- Retrieval planning and prompt generation from compiled evidence.
- Explicit calendar-date planning for single-day timeline questions.
- Optional OpenRouter model calls when `OPENROUTER_API_KEY` is configured.

The default CLI currently registers the Git and Codex adapters.

## Architecture

Recall keeps source-specific parsing behind adapters and uses `recall-core` for
shared event, timeline, search result, and inspection types.

```text
Question
    ↓
RetrievalPlanner
    ↓
Search / Timeline
    ↓
Inspect
    ↓
ContextCompiler
    ↓
PromptBuilder
    ↓
Prompt or OpenRouter answer
```

`RetrievalPlanner` classifies the question, `ContextCompiler` reduces retrieved
events into focused evidence, and `PromptBuilder` formats that evidence for the
answer request. OpenRouter transport is kept behind the CLI boundary and is
used only when an API key is configured.

A prompt-formatting experiment on 2026-08-24 tested compact evidence rendering
against a representative real Recall prompt. Conservative grouping reduced the
prompt by about 1.7%, and a more aggressive row-oriented format reduced the
approximate token count by about 4.2%; evidence/project-state content dominated
the size, so the compact format was rejected for now because the savings did not
justify less readable, less self-contained evidence records. No TOON dependency
or rendering change was adopted.

The workspace is split into focused crates:

- `recall-core`: shared domain types and adapter interfaces.
- `recall-codex`: adapter for local Codex session files.
- `recall-git`: adapter for Git commit history.
- `recall-cli`: command-line entry point.

## Installation

Build the workspace:

```sh
cargo build
```

Install the CLI from the local checkout:

```sh
cargo install --path crates/cli
```

## Example Usage

Search for matching evidence:

```sh
recall search timeline
```

Inspect a Git event:

```sh
recall inspect git:<sha>
```

Show the aggregated timeline:

```sh
recall timeline
```

Build an experimental prompt from retrieved evidence:

```sh
recall ask "When did I implement timeline?"
```

`recall ask` plans retrieval, searches or reads the timeline from supported
sources, inspects matching events, compiles evidence, and builds a prompt. For
explicit single-day questions, it keeps all matching events for that day before
compilation. If `OPENROUTER_API_KEY` is set, it sends the prompt to OpenRouter
with that key. Otherwise, Recall loads a stored OpenRouter key from
`~/.local/share/recall/auth.json`. If neither credential exists, it prints the
prompt instead of sending it.

Store an OpenRouter key once for future SSH sessions:

```sh
recall auth login
recall auth status
```

Recall creates `~/.local/share/recall/auth.json` outside the repository with
user-only file permissions. `OPENROUTER_API_KEY` remains the highest-priority
override. Remove the stored credential with `recall auth logout`.

The `Evidence:` section in this output lists the compiled `EvidenceBlock`s
produced by `ContextCompiler` — the same evidence used to build the prompt —
not the raw retrieval/search results. See `--debug-search` below to inspect
the earlier, raw retrieval stage.

To keep a local audit trail of configured model requests, set
`RECALL_OUTBOUND_LOG_DIR` to a non-empty directory path. Immediately before
Recall sends the prompt to OpenRouter, it writes a local JSON record containing
the timestamp, selected model, original question, prompt byte length, and exact
outbound prompt. Audit files are created with user-only permissions on Unix. If
an audit record cannot be written, Recall fails closed and does not send the
model request. Recall keeps only the newest 20 Recall-created outbound audit
records in that directory.

Useful debug flags:

```sh
recall ask --debug-query "What did I work on today?"
recall ask --debug-search "What did I work on today?"
recall ask --debug-prompt "What did I work on today?"
recall ask --diagnostics "What did I work on today?"
```

## Privacy

Recall reads local development artifacts. The current adapters can read Git
history from the current repository and Codex session files from the local Codex
sessions directory.

Generated prompts and model requests may include metadata from local Codex
sessions and Git history, such as source ids, timestamps, paths, commit text,
and session content. Review generated prompts before sharing them publicly.

Outbound audit records written under `RECALL_OUTBOUND_LOG_DIR` contain exact
model prompts and can include private engineering history. Treat that directory
as sensitive local data and do not add it or generated audit records to the
repository.

## License

Licensed under either of

- Apache License, Version 2.0
- MIT license

at your option.
