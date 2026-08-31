//! Shared domain model for Recall.
//!
//! The core crate is intentionally source-agnostic. It defines the small set of
//! types and traits that source adapters use to expose development memory
//! without committing the project to a storage engine, parser, search strategy,
//! or external integration.

use chrono::{DateTime, Datelike, Duration, Local, NaiveDate};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::time::Instant;

/// Source-neutral metadata attached to events and search results.
///
/// Metadata is represented as ordered string pairs so early adapters can expose
/// useful source details without forcing the core model to know every possible
/// field in advance.
pub type Metadata = BTreeMap<String, String>;

/// Stable identifier for an event within one source.
///
/// Event identifiers are source-local. A Git commit hash, a Codex turn id, and a
/// shell history row can all be valid event ids, but they should only be treated
/// as globally meaningful together with their [`Source`].
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EventId(String);

impl EventId {
    /// Creates a source-local event identifier.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the identifier as source-provided text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for EventId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for EventId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Source-provided timestamp text.
///
/// Recall does not normalize timestamps yet because each future adapter may
/// expose different precision, timezone, or ordering guarantees. This wrapper
/// makes timestamp usage explicit without adding a time dependency too early.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Timestamp(String);

impl Timestamp {
    /// Creates a timestamp from source-provided text.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the timestamp exactly as Recall currently stores it.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Timestamp {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for Timestamp {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Origin family for development-memory evidence.
///
/// The named variants cover sources that Recall understands today. `Other`
/// preserves source-agnostic behavior for future or user-defined sources such as
/// shell history, Claude transcripts, issue exports, or local notes without
/// forcing the core event model to change first.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Source {
    /// Codex conversation or session evidence.
    Codex,
    /// Git history evidence.
    Git,
    /// Any source that does not yet have a first-class variant.
    Other(String),
}

impl Source {
    /// Returns a stable lowercase label for known sources and the stored label
    /// for custom sources.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Codex => "codex",
            Self::Git => "git",
            Self::Other(source) => source,
        }
    }
}

/// A single development-memory observation.
///
/// Events are source-agnostic records of evidence. They are not limited to Git
/// commits or Codex turns; future adapters can map shell history, Claude
/// transcripts, issue exports, or local notes into the same small shape. Richer
/// source-specific details belong in [`Event::metadata`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Event {
    /// Stable identifier within the source that produced this event.
    pub id: EventId,
    /// Source family that produced this event.
    pub source: Source,
    /// Optional timestamp supplied by the source.
    pub timestamp: Option<Timestamp>,
    /// Short human-readable event title.
    pub title: String,
    /// Longer source-neutral event description.
    pub description: String,
    /// Source-specific fields retained as explicit evidence.
    pub metadata: Metadata,
}

impl Event {
    /// Creates a minimal event with empty metadata and no timestamp.
    pub fn new(id: impl Into<EventId>, source: Source, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            source,
            timestamp: None,
            title: title.into(),
            description: String::new(),
            metadata: Metadata::new(),
        }
    }
}

/// Source-qualified reference to an event.
///
/// This is the smallest stable handle for cross-source results. It avoids
/// copying whole events into search output while still making the event's origin
/// explicit.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EventRef {
    /// Source family that owns the referenced event id.
    pub source: Source,
    /// Identifier local to [`EventRef::source`].
    pub id: EventId,
}

impl EventRef {
    /// Creates a source-qualified event reference.
    pub fn new(source: Source, id: impl Into<EventId>) -> Self {
        Self {
            source,
            id: id.into(),
        }
    }
}

/// Ordered events representing a source-neutral development timeline.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Timeline {
    /// Events in newest-first order, with untimestamped events last.
    pub events: Vec<Event>,
}

impl Timeline {
    /// Creates an empty timeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when the timeline contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// High-level retrieval strategy selected for an ask question.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetrievalPlan {
    /// Use source-local lexical search with the normalized subject query.
    Search { query: SearchQuery },
    /// Use newest project-owned events for resume/latest-state questions.
    ProjectLatest { query: String },
    /// Use timeline retrieval for a relative date range, optionally narrowed by explicit subject terms.
    Timeline { range: DateRange, query: String },
}

/// Structured lexical search query for ask retrieval.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchQuery {
    subject: String,
    intent: SearchIntent,
    intent_terms: Vec<String>,
}

impl SearchQuery {
    /// Creates a plain lexical search query.
    pub fn plain(query: impl Into<String>) -> Self {
        Self {
            subject: query.into(),
            intent: SearchIntent::Plain,
            intent_terms: Vec::new(),
        }
    }

    /// Returns the subject terms used for lexical retrieval.
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// Returns the evidence intent inferred from the question.
    pub fn intent(&self) -> SearchIntent {
        self.intent
    }

    /// Returns the terms that expressed the inferred evidence intent.
    pub fn intent_terms(&self) -> &[String] {
        &self.intent_terms
    }
}

/// Evidence intent inferred for an ask Search query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchIntent {
    /// No specific evidence intent was identified.
    Plain,
    /// The question asks for rationale or motivation.
    Rationale,
    /// The question asks whether something was discussed.
    Discussion,
    /// The question asks for completed or landed change evidence.
    CompletedChange,
}

impl SearchIntent {
    /// Returns a stable diagnostic label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Rationale => "rationale",
            Self::Discussion => "discussion",
            Self::CompletedChange => "completed-change",
        }
    }
}

/// Date range extracted from a question.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DateRange {
    /// Events from a specific calendar day.
    Day(NaiveDate),
    /// Events from today.
    Today,
    /// Events from yesterday.
    Yesterday,
    /// Events from the previous calendar week.
    LastWeek,
    /// Events from the last N days.
    LastDays(u32),
}

impl DateRange {
    /// Returns true when a source timestamp falls within this relative range.
    pub fn contains_timestamp(&self, timestamp: &Timestamp) -> bool {
        self.contains_timestamp_on(timestamp, Local::now().date_naive())
    }

    /// Returns true when a source timestamp falls within this range relative to
    /// the supplied local date.
    pub fn contains_timestamp_on(&self, timestamp: &Timestamp, today: NaiveDate) -> bool {
        parse_timestamp_date(timestamp)
            .map(|date| self.contains_date(date, today))
            .unwrap_or(false)
    }

    fn contains_date(&self, date: NaiveDate, today: NaiveDate) -> bool {
        match self {
            Self::Day(target) => date == *target,
            Self::Today => date == today,
            Self::Yesterday => date == today - Duration::days(1),
            Self::LastWeek => {
                let days_since_monday = today.weekday().num_days_from_monday() as i64;
                let this_week_start = today - Duration::days(days_since_monday);
                let last_week_start = this_week_start - Duration::days(7);
                date >= last_week_start && date < this_week_start
            }
            Self::LastDays(days) => {
                let start = today - Duration::days(days.saturating_sub(1) as i64);
                date >= start && date <= today
            }
        }
    }
}

/// Classifies ask questions before retrieval.
#[derive(Clone, Debug, Default)]
pub struct RetrievalPlanner;

impl RetrievalPlanner {
    /// Creates a retrieval planner.
    pub fn new() -> Self {
        Self
    }

    /// Builds a retrieval plan from a raw user question.
    pub fn plan(&self, question: &str) -> RetrievalPlan {
        if let Some(date) = explicit_date(question) {
            return RetrievalPlan::Timeline {
                range: DateRange::Day(date),
                query: normalize_temporal_subject_query_words(&normalize_question_words(question)),
            };
        }

        let normalized_words = normalize_question_words(question);
        if contains_word(&normalized_words, "today") || contains_word(&normalized_words, "todays") {
            return RetrievalPlan::Timeline {
                range: DateRange::Today,
                query: normalize_temporal_subject_query_words(&normalized_words),
            };
        }
        if contains_word(&normalized_words, "yesterday")
            || contains_word(&normalized_words, "yesterdays")
        {
            return RetrievalPlan::Timeline {
                range: DateRange::Yesterday,
                query: normalize_temporal_subject_query_words(&normalized_words),
            };
        }
        if contains_word(&normalized_words, "recently")
            || contains_word(&normalized_words, "recent")
        {
            return RetrievalPlan::Timeline {
                range: DateRange::LastDays(7),
                query: normalize_temporal_subject_query_words(&normalized_words),
            };
        }
        if contains_word_sequence(&normalized_words, &["last", "week"]) {
            return RetrievalPlan::Timeline {
                range: DateRange::LastWeek,
                query: normalize_temporal_subject_query_words(&normalized_words),
            };
        }
        if let Some(days) = last_days_range(&normalized_words) {
            return RetrievalPlan::Timeline {
                range: DateRange::LastDays(days),
                query: normalize_temporal_subject_query_words(&normalized_words),
            };
        }

        if has_project_latest_intent(&normalized_words) {
            return RetrievalPlan::ProjectLatest {
                query: normalize_project_latest_query_words(&normalized_words),
            };
        }

        RetrievalPlan::Search {
            query: search_query_from_words(&normalized_words),
        }
    }
}

/// Search output returned by an adapter.
///
/// Search results describe a match without embedding a full event. Future
/// ranking, indexing, or AI layers can use the source-qualified event reference
/// to retrieve the explicit evidence behind the snippet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchResult {
    /// Source-qualified event matched by the adapter.
    pub event: EventRef,
    /// Optional adapter-defined score.
    ///
    /// Scores are not normalized by the core crate. Consumers should treat them
    /// as hints from a specific source until a ranking layer exists.
    pub score: Option<u32>,
    /// Short source-provided match snippet.
    pub snippet: String,
    /// Additional source-specific evidence.
    pub metadata: Metadata,
    /// Internal retrieval explanation for debug and evaluation output.
    ///
    /// Diagnostics are not part of user-facing CLI output and must not
    /// participate in ranking or retrieval decisions.
    #[doc(hidden)]
    pub diagnostics: Metadata,
}

/// Request-local search match that keeps the materialized event with its result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchMatch {
    pub result: SearchResult,
    pub event: Event,
}

/// Number of retrieval items included in `recall ask` before context compilation.
pub const ASK_RESULT_LIMIT: usize = 8;

/// Events and ranked search results selected for an ask retrieval plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AskRetrieval {
    pub events: Vec<Event>,
    pub search_results: Vec<SearchResult>,
}

/// LLM-facing context compiled from a source event.
///
/// Evidence blocks deliberately keep only the fields needed to ground an answer.
/// The original [`Event`] remains the storage record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceBlock {
    /// Source family that produced the evidence.
    pub source: Source,
    /// Source-local identifier for citations.
    pub id: EventId,
    /// Optional timestamp supplied by the source.
    pub timestamp: Option<Timestamp>,
    /// Short human-readable evidence title.
    pub title: String,
    /// Concise deterministic evidence body.
    pub body: String,
}

/// Options controlling compiler-local context selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileOptions {
    /// Approximate maximum number of evidence characters retained.
    ///
    /// This is intentionally a simple character budget rather than a
    /// provider-specific token estimate. The compiler should remain
    /// deterministic and model-agnostic.
    pub character_budget: usize,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            character_budget: 12_000,
        }
    }
}

/// Compiles retrieved storage events into LLM-facing evidence.
///
/// The compiler is deterministic and local. It does not call an LLM, rank
/// results, summarize semantically, or change retrieval behavior.
#[derive(Clone, Debug, Default)]
pub struct ContextCompiler;

impl ContextCompiler {
    /// Creates a context compiler.
    pub fn new() -> Self {
        Self
    }

    /// Compiles retrieved events into concise evidence blocks.
    pub fn compile(&self, question: &str, events: &[Event]) -> Vec<EvidenceBlock> {
        self.compile_with_options(question, events, &CompileOptions::default())
    }

    /// Compiles timeline events as direct per-event evidence.
    pub fn compile_timeline(&self, question: &str, events: &[Event]) -> Vec<EvidenceBlock> {
        compile_events(question, events)
            .into_iter()
            .map(|event| event.evidence)
            .collect()
    }

    /// Compiles latest-project retrieval as direct evidence from the newest event.
    pub fn compile_project_latest(&self, question: &str, events: &[Event]) -> Vec<EvidenceBlock> {
        events
            .first()
            .map(|event| compile_project_latest_event(question, event))
            .into_iter()
            .collect()
    }

    /// Compiles retrieved events using explicit compiler options.
    pub fn compile_with_options(
        &self,
        question: &str,
        events: &[Event],
        options: &CompileOptions,
    ) -> Vec<EvidenceBlock> {
        if options.character_budget == 0 || events.is_empty() {
            return Vec::new();
        }

        let (direct_evidence, project_state_events) =
            split_direct_evidence(compile_events(question, events));
        let retained_project_state_events = select_compiled_events(
            deduplicate_compiled_events(project_state_events),
            options.character_budget,
        );
        project_states_from_events(retained_project_state_events)
            .into_iter()
            .map(ProjectState::into_evidence)
            .chain(direct_evidence)
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompiledEvent {
    event: EventRef,
    categories: Vec<ContextCategory>,
    metadata: Metadata,
    evidence: EvidenceBlock,
    order: usize,
}

impl CompiledEvent {
    /// Retention priority is derived from categories every time it is needed.
    /// For multi-category events, the strongest category wins.
    fn retention_priority(&self) -> RetentionPriority {
        self.categories
            .iter()
            .map(ContextCategory::retention_priority)
            .min()
            .unwrap_or(RetentionPriority::Uncategorized)
    }

    fn evidence_size(&self) -> usize {
        self.evidence.title.chars().count() + self.evidence.body.chars().count()
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ContextCategory {
    Objective,
    Decision,
    Milestone,
    Status,
    NextStep,
    Validation,
    Blocker,
    Todo,
}

impl ContextCategory {
    fn retention_priority(&self) -> RetentionPriority {
        match self {
            Self::Decision | Self::Blocker | Self::NextStep => RetentionPriority::Highest,
            Self::Objective | Self::Milestone => RetentionPriority::High,
            Self::Status | Self::Validation => RetentionPriority::Medium,
            Self::Todo => RetentionPriority::Low,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RetentionPriority {
    Highest,
    High,
    Medium,
    Low,
    Uncategorized,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum DedupKey {
    SourceId(Source, EventId),
    CommitSha(String),
    RetainedLines(Vec<String>),
    RepositoryTitleTimestamp {
        repository: String,
        title: String,
        timestamp: Timestamp,
    },
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProjectKey {
    kind: String,
    value: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ProjectSections {
    current_objective: Vec<String>,
    recent_milestones: Vec<String>,
    implementation_status: Vec<String>,
    architectural_decisions: Vec<String>,
    planned_next_steps: Vec<String>,
    outstanding_blockers: Vec<String>,
    validation: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProjectState {
    key: ProjectKey,
    source: Source,
    id: EventId,
    timestamp: Option<Timestamp>,
    order: usize,
    sections: ProjectSections,
    seen_entries: BTreeSet<String>,
}

impl ProjectState {
    fn new(key: ProjectKey, event: &CompiledEvent) -> Self {
        Self {
            id: EventId::new(format!("project:{}", key.value)),
            key,
            source: event.evidence.source.clone(),
            timestamp: event.evidence.timestamp.clone(),
            order: event.order,
            sections: ProjectSections::default(),
            seen_entries: BTreeSet::new(),
        }
    }

    fn add_event(&mut self, event: &CompiledEvent) {
        self.timestamp =
            newest_optional_timestamp(self.timestamp.clone(), event.evidence.timestamp.clone());
        self.order = self.order.min(event.order);

        for line in project_state_candidate_lines(event) {
            let categories = categorize_line(&line);
            for category in categories {
                self.add_entry(category, &line);
            }
        }
    }

    fn add_entry(&mut self, category: ContextCategory, line: &str) {
        let entry = normalize_state_entry(line);
        if entry.is_empty() || !self.seen_entries.insert(normalize_dedup_text(&entry)) {
            return;
        }

        match category {
            ContextCategory::Objective => self.sections.current_objective.push(entry),
            ContextCategory::Milestone => self.sections.recent_milestones.push(entry),
            ContextCategory::Status | ContextCategory::Todo => {
                self.sections.implementation_status.push(entry);
            }
            ContextCategory::Decision => self.sections.architectural_decisions.push(entry),
            ContextCategory::NextStep => self.sections.planned_next_steps.push(entry),
            ContextCategory::Blocker => self.sections.outstanding_blockers.push(entry),
            ContextCategory::Validation => self.sections.validation.push(entry),
        }
    }

    fn into_evidence(self) -> EvidenceBlock {
        let body = render_project_state_body(&self);
        EvidenceBlock {
            source: self.source,
            id: self.id,
            timestamp: self.timestamp,
            title: format!("Project: {}", self.key.value),
            body,
        }
    }
}

/// Builds a prompt from compiled evidence.
///
/// This helper only formats source-neutral [`EvidenceBlock`] values. It does
/// not search, inspect, call models, rank results, or know about the CLI.
#[derive(Clone, Debug, Default)]
pub struct PromptBuilder;

impl PromptBuilder {
    /// Creates a prompt builder.
    pub fn new() -> Self {
        Self
    }

    /// Builds an answer prompt from a user question and compiled evidence.
    pub fn build(&self, question: &str, evidence: &[EvidenceBlock]) -> String {
        self.build_with_clock(question, evidence, Local::now())
    }

    fn build_with_clock(
        &self,
        question: &str,
        evidence: &[EvidenceBlock],
        now: DateTime<Local>,
    ) -> String {
        let mut prompt = String::new();
        prompt.push_str("You are answering questions about my engineering history.\n\n");
        prompt.push_str("Current date: ");
        prompt.push_str(&now.format("%Y-%m-%d").to_string());
        prompt.push('\n');
        prompt.push_str("Current local time: ");
        prompt.push_str(&now.format("%H:%M %Z").to_string());
        prompt.push_str("\n\n");
        prompt.push_str("Interpret relative temporal expressions such as \"today\", \"yesterday\", \"this week\", and \"last week\" relative to this date.\n\n");
        prompt.push_str("Use ONLY the supplied context.\n\n");
        prompt.push_str("If the context does not contain enough information,\n");
        prompt.push_str("say so.\n\n");
        prompt.push_str("For time-period questions, phrase the answer as what the supplied evidence shows, not as a complete account of the period. Do not present inferred follow-ups or unresolved items as facts; label them as inference when mentioned.\n\n");
        prompt.push_str("When evidence describes an evolving or contradictory state, distinguish intermediate findings from the latest known state. If later timestamped evidence supersedes earlier state claims, make that final state clear while still preserving the chronology. Describe superseded earlier conclusions as what appeared true, was believed, or was concluded at that point, not as objective final facts.\n\n");
        prompt.push_str("Cite source ids when referring to events.\n\n");
        prompt.push_str("Question:\n\n");
        prompt.push_str(question);
        prompt.push_str("\n\n");
        prompt.push_str("Context:\n");

        for block in evidence {
            prompt.push_str("\n=== Evidence ===\n");
            prompt.push_str("Source: ");
            prompt.push_str(block.source.as_str());
            prompt.push('\n');
            prompt.push_str("Id: ");
            prompt.push_str(block.id.as_str());
            prompt.push('\n');
            prompt.push_str("Timestamp: ");
            if let Some(timestamp) = &block.timestamp {
                prompt.push_str(timestamp.as_str());
            }
            prompt.push('\n');
            prompt.push_str("Title: ");
            prompt.push_str(&block.title);
            prompt.push('\n');
            prompt.push_str("Evidence:\n");
            prompt.push_str(&block.body);
            prompt.push('\n');
        }

        prompt
    }
}

fn compile_events(question: &str, events: &[Event]) -> Vec<CompiledEvent> {
    events
        .iter()
        .enumerate()
        .map(|(order, event)| compile_event(question, event, order))
        .collect()
}

fn deduplicate_compiled_events(events: Vec<CompiledEvent>) -> Vec<CompiledEvent> {
    let mut deduplicated: Vec<CompiledEvent> = Vec::new();
    let mut key_indexes = BTreeMap::new();

    for event in events {
        let keys = dedup_keys(&event);
        if let Some(index) = keys.iter().find_map(|key| key_indexes.get(key).copied()) {
            merge_compiled_event(&mut deduplicated[index], event);
            for key in dedup_keys(&deduplicated[index]) {
                key_indexes.insert(key, index);
            }
        } else {
            let index = deduplicated.len();
            for key in keys {
                key_indexes.insert(key, index);
            }
            deduplicated.push(event);
        }
    }

    deduplicated
}

fn dedup_keys(event: &CompiledEvent) -> Vec<DedupKey> {
    let mut keys = vec![DedupKey::SourceId(
        event.event.source.clone(),
        event.event.id.clone(),
    )];

    if let Some(sha) = event.metadata.get("sha").map(String::as_str).map(str::trim) {
        if !sha.is_empty() {
            keys.push(DedupKey::CommitSha(sha.to_ascii_lowercase()));
        }
    }

    let retained_lines = normalized_retained_lines(&event.evidence.body);
    if !retained_lines.is_empty() {
        keys.push(DedupKey::RetainedLines(retained_lines));
    }

    if let (Some(repository), Some(timestamp)) = (
        repository_key(&event.metadata),
        event.evidence.timestamp.clone(),
    ) {
        let title = normalize_dedup_text(&event.evidence.title);
        if !title.is_empty() {
            keys.push(DedupKey::RepositoryTitleTimestamp {
                repository,
                title,
                timestamp,
            });
        }
    }

    keys
}

fn repository_key(metadata: &Metadata) -> Option<String> {
    ["repo", "repository", "cwd"]
        .iter()
        .filter_map(|key| metadata.get(*key))
        .map(|value| normalize_dedup_text(value))
        .find(|value| !value.is_empty())
}

fn normalized_retained_lines(body: &str) -> Vec<String> {
    body.lines()
        .map(normalize_dedup_text)
        .filter(|line| !line.is_empty())
        .collect()
}

fn normalize_dedup_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn merge_compiled_event(target: &mut CompiledEvent, duplicate: CompiledEvent) {
    merge_categories(&mut target.categories, duplicate.categories);
    merge_evidence_body(&mut target.evidence.body, &duplicate.evidence.body);
    refresh_retained_line_count(&mut target.metadata, &target.evidence.body);
}

fn merge_categories(target: &mut Vec<ContextCategory>, duplicate: Vec<ContextCategory>) {
    let mut categories = target.iter().cloned().collect::<BTreeSet<_>>();
    categories.extend(duplicate);
    *target = categories.into_iter().collect();
}

fn merge_evidence_body(target: &mut String, duplicate: &str) {
    let mut seen = normalized_retained_lines(target)
        .into_iter()
        .collect::<BTreeSet<_>>();
    for line in duplicate
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if seen.insert(normalize_dedup_text(line)) {
            if !target.is_empty() {
                target.push('\n');
            }
            target.push_str(line);
        }
    }
}

fn split_direct_evidence(events: Vec<CompiledEvent>) -> (Vec<EvidenceBlock>, Vec<CompiledEvent>) {
    let mut direct_evidence = Vec::new();
    let mut project_state_events = Vec::new();

    for event in events {
        if event.event.source == Source::Git {
            direct_evidence.push(event.evidence);
        } else {
            project_state_events.push(event);
        }
    }

    (direct_evidence, project_state_events)
}

fn compile_event(question: &str, event: &Event, order: usize) -> CompiledEvent {
    let mut lines = Vec::new();
    push_metadata_lines(event, &mut lines);
    if event.source == Source::Git {
        push_git_description_lines(&event.description, &mut lines);
    } else {
        push_selected_description_lines(question, &event.description, &mut lines);
    }

    let categories = categorize_event(event, &lines);
    let metadata = compiler_metadata(event, &lines);
    let evidence = EvidenceBlock {
        source: event.source.clone(),
        id: event.id.clone(),
        timestamp: event.timestamp.clone(),
        title: compile_title(event),
        body: lines.join("\n"),
    };

    CompiledEvent {
        event: EventRef::new(event.source.clone(), event.id.clone()),
        categories,
        metadata,
        evidence,
        order,
    }
}

fn compile_project_latest_event(question: &str, event: &Event) -> EvidenceBlock {
    let mut lines = Vec::new();
    push_metadata_lines(event, &mut lines);
    push_project_latest_description_lines(question, &event.description, &mut lines);

    EvidenceBlock {
        source: event.source.clone(),
        id: event.id.clone(),
        timestamp: event.timestamp.clone(),
        title: compile_title(event),
        body: lines.join("\n"),
    }
}

fn select_compiled_events(
    events: Vec<CompiledEvent>,
    character_budget: usize,
) -> Vec<CompiledEvent> {
    if character_budget == 0 || events.is_empty() {
        return Vec::new();
    }

    let mut candidates = events;
    candidates.sort_by(compare_for_retention);

    let mut selected = Vec::new();
    let mut used = 0usize;

    for event in candidates {
        let size = event.evidence_size();
        if selected.is_empty() || used.saturating_add(size) <= character_budget {
            used = used.saturating_add(size);
            selected.push(event);
        }
    }

    selected
}

fn project_states_from_events(events: Vec<CompiledEvent>) -> Vec<ProjectState> {
    let mut projects = BTreeMap::new();

    for event in events {
        let key = project_key(&event);
        let state = projects
            .entry(key.clone())
            .or_insert_with(|| ProjectState::new(key, &event));
        state.add_event(&event);
    }

    // BTreeMap gives stable project ordering by structural project key. Within a
    // project, section entries retain the already-selected compiler event order:
    // retention priority, timestamp recency, then original input order.
    projects.into_values().collect()
}

/// Project grouping is structural and deterministic.
///
/// The grouping order is intentionally conservative: explicit repository
/// metadata first, then repository-root/cwd-style paths, then commit SHA, then a
/// source-qualified event id fallback. The final fallback avoids semantic text
/// inference while still producing a stable project state for otherwise
/// ungroupable evidence.
fn project_key(event: &CompiledEvent) -> ProjectKey {
    for key in ["repo", "repository", "repository_root", "repo_root"] {
        if let Some(value) = normalized_metadata_value(&event.metadata, key) {
            return ProjectKey {
                kind: "repository".to_string(),
                value,
            };
        }
    }

    for key in ["cwd", "workspace"] {
        if let Some(value) = normalized_metadata_value(&event.metadata, key) {
            return ProjectKey {
                kind: "cwd".to_string(),
                value,
            };
        }
    }

    if let Some(value) = normalized_metadata_value(&event.metadata, "sha") {
        return ProjectKey {
            kind: "sha".to_string(),
            value,
        };
    }

    ProjectKey {
        kind: event.event.source.as_str().to_string(),
        value: format!(
            "{}:{}",
            event.event.source.as_str(),
            event.event.id.as_str()
        ),
    }
}

fn normalized_metadata_value(metadata: &Metadata, key: &str) -> Option<String> {
    metadata
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(normalize_project_key_value)
}

fn normalize_project_key_value(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn newest_optional_timestamp(
    left: Option<Timestamp>,
    right: Option<Timestamp>,
) -> Option<Timestamp> {
    match (left, right) {
        (Some(left), Some(right)) => Some(std::cmp::max(left, right)),
        (Some(timestamp), None) | (None, Some(timestamp)) => Some(timestamp),
        (None, None) => None,
    }
}

fn project_state_candidate_lines(event: &CompiledEvent) -> Vec<String> {
    std::iter::once(event.evidence.title.as_str())
        .chain(event.evidence.body.lines())
        .map(str::trim)
        .filter(|line| !line.is_empty() && !is_project_metadata_line(line))
        .map(str::to_string)
        .collect()
}

fn is_project_metadata_line(line: &str) -> bool {
    line.starts_with("cwd:")
        || line.starts_with("repo:")
        || line.starts_with("repository:")
        || line.starts_with("repository_root:")
        || line.starts_with("repo_root:")
        || line.starts_with("workspace:")
        || line.starts_with("sha:")
}

fn categorize_line(line: &str) -> Vec<ContextCategory> {
    let text = line.to_lowercase();
    let mut categories = Vec::new();
    push_category_if(
        &mut categories,
        ContextCategory::Objective,
        has_objective_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Decision,
        has_decision_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Milestone,
        has_milestone_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Status,
        has_status_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::NextStep,
        has_next_step_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Validation,
        has_validation_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Blocker,
        has_blocker_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Todo,
        has_todo_signal(&text),
    );
    let is_uncategorized_conclusion = categories.is_empty() && is_assistant_conclusion_line(line);
    push_category_if(
        &mut categories,
        ContextCategory::Status,
        is_uncategorized_conclusion,
    );
    categories
}

fn normalize_state_entry(line: &str) -> String {
    line.trim()
        .trim_start_matches("- ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_project_state_body(state: &ProjectState) -> String {
    let mut body = String::new();
    body.push_str("Project key: ");
    body.push_str(&state.key.kind);
    body.push(':');
    body.push(' ');
    body.push_str(&state.key.value);
    push_project_section(
        &mut body,
        "Current objective",
        &state.sections.current_objective,
    );
    push_project_section(
        &mut body,
        "Recent milestones",
        &state.sections.recent_milestones,
    );
    push_project_section(
        &mut body,
        "Implementation status",
        &state.sections.implementation_status,
    );
    push_project_section(
        &mut body,
        "Architectural decisions",
        &state.sections.architectural_decisions,
    );
    push_project_section(
        &mut body,
        "Planned next step",
        &state.sections.planned_next_steps,
    );
    push_project_section(
        &mut body,
        "Outstanding blockers",
        &state.sections.outstanding_blockers,
    );
    push_project_section(&mut body, "Validation", &state.sections.validation);
    body
}

fn push_project_section(body: &mut String, heading: &str, entries: &[String]) {
    if entries.is_empty() {
        return;
    }

    body.push('\n');
    body.push_str(heading);
    body.push_str(":\n");
    for entry in entries {
        body.push_str("- ");
        body.push_str(entry);
        body.push('\n');
    }
    if body.ends_with('\n') {
        body.pop();
    }
}

/// Retention selection is stable and deterministic:
/// priority first, newest timestamp next, then original compiler order.
fn compare_for_retention(left: &CompiledEvent, right: &CompiledEvent) -> std::cmp::Ordering {
    left.retention_priority()
        .cmp(&right.retention_priority())
        .then_with(|| {
            compare_optional_timestamps_desc(&left.evidence.timestamp, &right.evidence.timestamp)
        })
        .then_with(|| left.order.cmp(&right.order))
}

fn compare_optional_timestamps_desc(
    left: &Option<Timestamp>,
    right: &Option<Timestamp>,
) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.cmp(left),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn compiler_metadata(event: &Event, lines: &[String]) -> Metadata {
    let mut metadata = Metadata::from([("retained_lines".to_string(), lines.len().to_string())]);
    for key in ["cwd", "repo", "repository", "sha"] {
        if let Some(value) = event.metadata.get(key).map(String::as_str).map(str::trim) {
            if !value.is_empty() {
                metadata.insert(key.to_string(), value.to_string());
            }
        }
    }
    metadata
}

fn refresh_retained_line_count(metadata: &mut Metadata, body: &str) {
    metadata.insert(
        "retained_lines".to_string(),
        body.lines().count().to_string(),
    );
}

fn categorize_event(event: &Event, retained_lines: &[String]) -> Vec<ContextCategory> {
    let text = categorization_text(event, retained_lines);
    let mut categories = Vec::new();
    push_category_if(
        &mut categories,
        ContextCategory::Objective,
        has_objective_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Decision,
        has_decision_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Milestone,
        has_milestone_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Status,
        has_status_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::NextStep,
        has_next_step_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Validation,
        has_validation_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Blocker,
        has_blocker_signal(&text),
    );
    push_category_if(
        &mut categories,
        ContextCategory::Todo,
        has_todo_signal(&text),
    );
    categories
}

fn categorization_text(event: &Event, retained_lines: &[String]) -> String {
    let mut text = String::new();
    text.push_str(&event.title);
    push_categorizable_description_text(&mut text, &event.description);
    for line in retained_lines {
        text.push('\n');
        text.push_str(line);
    }
    text.to_lowercase()
}

fn push_categorizable_description_text(text: &mut String, description: &str) {
    let mut in_skipped_block = false;

    for raw_line in description.lines() {
        let line = raw_line.trim();
        if starts_skipped_block(line) {
            in_skipped_block = true;
            continue;
        }
        if in_skipped_block {
            if ends_skipped_block(line) {
                in_skipped_block = false;
            }
            continue;
        }
        if is_noise_line(line) {
            continue;
        }
        text.push('\n');
        text.push_str(line);
    }
}

fn push_category_if(
    categories: &mut Vec<ContextCategory>,
    category: ContextCategory,
    condition: bool,
) {
    if condition {
        categories.push(category);
    }
}

fn has_objective_signal(text: &str) -> bool {
    contains_any(
        text,
        &[
            "current objective",
            "objective:",
            "goal:",
            "the goal is",
            "the objective is",
        ],
    )
}

fn has_decision_signal(text: &str) -> bool {
    contains_any(
        text,
        &[
            "decision:",
            "decided",
            "accepted design",
            "rejected because",
            "deferred because",
            "architectural decision",
        ],
    )
}

fn has_milestone_signal(text: &str) -> bool {
    contains_any(
        text,
        &[
            "implemented",
            "completed",
            "added ",
            "committed as",
            "final state:",
            "changed:",
            "feat(",
            "fix(",
            "docs(",
            "chore(",
            "refactor(",
            "test(",
        ],
    )
}

fn has_status_signal(text: &str) -> bool {
    contains_any(
        text,
        &[
            "current implementation status",
            "implementation status",
            "status:",
            "final state:",
            "current state",
        ],
    )
}

fn has_next_step_signal(text: &str) -> bool {
    contains_any(
        text,
        &[
            "next step",
            "planned next step",
            "next planned",
            "follow-up",
            "follow up",
            "future work",
        ],
    )
}

fn has_validation_signal(text: &str) -> bool {
    contains_any(
        text,
        &[
            "validation:",
            "validation passed",
            "cargo test",
            "cargo fmt",
            "git diff --check",
            "tests passed",
        ],
    )
}

fn has_blocker_signal(text: &str) -> bool {
    contains_any(
        text,
        &[
            "blocker",
            "blocked",
            "failed because",
            "unable to",
            "cannot proceed",
            "unresolved",
            "open question",
        ],
    )
}

fn has_todo_signal(text: &str) -> bool {
    contains_any(text, &["todo", "to do", "fixme"])
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn compile_title(event: &Event) -> String {
    let title = event.title.trim();
    if is_noise_line(title) {
        format!("{} event", event.source.as_str())
    } else {
        title.to_string()
    }
}

fn push_metadata_lines(event: &Event, lines: &mut Vec<String>) {
    for key in ["cwd", "repo", "repository", "sha"] {
        if let Some(value) = event.metadata.get(key).map(String::as_str).map(str::trim) {
            if !value.is_empty() {
                lines.push(format!("{key}: {value}"));
            }
        }
    }
}

fn push_selected_description_lines(question: &str, description: &str, lines: &mut Vec<String>) {
    let question_terms = normalize_question_words(question);
    let mut in_skipped_block = false;

    for raw_line in description.lines() {
        let line = raw_line.trim();
        if starts_skipped_block(line) {
            in_skipped_block = true;
            continue;
        }
        if in_skipped_block {
            if ends_skipped_block(line) {
                in_skipped_block = false;
            }
            continue;
        }
        if is_prompt_scaffolding_line(line.trim_start_matches("- "))
            || is_noise_line(line)
            || is_project_latest_citation_line(line)
            || is_duplicate_line(lines, line)
        {
            continue;
        }
        if is_high_value_line(line, &question_terms) {
            lines.push(line.to_string());
        }
    }
}

fn push_project_latest_description_lines(
    question: &str,
    description: &str,
    lines: &mut Vec<String>,
) {
    let question_terms = normalize_question_words(question);
    let mut in_skipped_block = false;
    let mut keep_fenced_block = false;
    let mut keep_next_fenced_block = false;
    let mut keep_list = false;

    for raw_line in description.lines() {
        let line = raw_line.trim();
        if starts_skipped_block(line) {
            in_skipped_block = true;
            continue;
        }
        if in_skipped_block {
            if ends_skipped_block(line) {
                in_skipped_block = false;
            }
            continue;
        }
        if is_prompt_scaffolding_line(line.trim_start_matches("- "))
            || is_noise_line(line)
            || is_project_latest_citation_line(line)
            || is_duplicate_line(lines, line)
        {
            continue;
        }

        if keep_fenced_block {
            lines.push(line.to_string());
            if line == "```" {
                keep_fenced_block = false;
                keep_next_fenced_block = false;
            }
            continue;
        }
        if keep_next_fenced_block && line.starts_with("```") {
            lines.push(line.to_string());
            if line != "```" {
                keep_fenced_block = true;
            }
            continue;
        }
        if keep_list {
            if line.starts_with("- ") {
                lines.push(line.to_string());
                continue;
            }
            keep_list = false;
        }

        if is_high_value_line(line, &question_terms) || is_project_latest_state_line(line) {
            lines.push(line.to_string());
            if keeps_following_fenced_block(line) {
                keep_next_fenced_block = true;
            }
            if keeps_following_list(line) {
                keep_list = true;
            }
            if line.starts_with("```") && line != "```" {
                keep_fenced_block = true;
            }
        }
    }
}

fn is_project_latest_state_line(line: &str) -> bool {
    let text = line.to_ascii_lowercase();
    line.starts_with("Commit:")
        || line.starts_with("Message:")
        || line.starts_with("Final git status:")
        || line.starts_with("No commit was made.")
        || line.starts_with("No push was performed")
        || line.starts_with("Do not redo")
        || line.starts_with("Do not push")
        || line.starts_with("Do not modify")
        || line.starts_with("Report the commit hash")
        || text.contains("ssh connection died")
        || text.contains("not committed")
        || text.contains("not pushed")
        || text.contains("uncommitted")
        || text.contains("unpushed")
        || text.contains("ahead ")
        || text.contains("working tree")
}

fn keeps_following_fenced_block(line: &str) -> bool {
    line.starts_with("Final git status:")
}

fn keeps_following_list(line: &str) -> bool {
    line.starts_with("Validation passed:")
}

fn is_project_latest_citation_line(line: &str) -> bool {
    line == "<oai-mem-citation>"
        || line == "</oai-mem-citation>"
        || line == "<citation_entries>"
        || line == "</citation_entries>"
        || line == "<rollout_ids>"
        || line == "</rollout_ids>"
        || line.starts_with("MEMORY.md:")
        || line.starts_with("rollout_summaries/")
}

fn push_git_description_lines(description: &str, lines: &mut Vec<String>) {
    for raw_line in description.lines() {
        let line = raw_line.trim();
        if !is_noise_line(line) && !is_duplicate_line(lines, line) {
            lines.push(line.to_string());
        }
    }
}

/// Fixed prompt/template lines produced by Recall itself (handoff prompts,
/// generated section headers, generic retrieval examples). These must be
/// rejected before `is_high_value_line` runs so they cannot be rescued by a
/// question-term or context-signal match. See "Retention Rule Precedence" in
/// docs/context-compiler.md.
fn is_prompt_scaffolding_line(line: &str) -> bool {
    matches!(
        line,
        "What is the current objective?"
            | "What changed:"
            | "What has recently been completed?"
            | "What architectural decisions have been made?"
            | "What blockers remain?"
            | "What is the next planned step?"
            | "Current objective"
            | "Recent milestones"
            | "Implementation status"
            | "Architectural decisions"
            | "Planned next step"
            | "Outstanding blockers"
            | "Validation:"
            | "What did I work on today?"
            | "Summarize today's recall work."
    )
}

fn starts_skipped_block(line: &str) -> bool {
    matches!(
        line,
        "<environment_context>"
            | "<permissions instructions>"
            | "<skills_instructions>"
            | "<apps_instructions>"
            | "<plugins_instructions>"
            | "========= MEMORY_SUMMARY BEGINS ========="
    )
}

fn ends_skipped_block(line: &str) -> bool {
    matches!(
        line,
        "</environment_context>"
            | "</permissions instructions>"
            | "</skills_instructions>"
            | "</apps_instructions>"
            | "</plugins_instructions>"
            | "========= MEMORY_SUMMARY ENDS ========="
    )
}

fn is_noise_line(line: &str) -> bool {
    line.is_empty()
        || starts_skipped_block(line)
        || ends_skipped_block(line)
        || line.starts_with("Knowledge cutoff:")
        || line.starts_with("Current date:")
        || line.starts_with("You are an AI assistant")
        || line.starts_with("You are Codex")
        || line.starts_with("Decision boundary:")
        || line.starts_with("Memory layout")
        || line.starts_with("Quick memory pass")
        || line.starts_with("Quick-pass budget")
        || line.starts_with("Memory citation")
        || line.starts_with("Updating memories")
        || line.starts_with("- /home/simon/.codex")
        || line.starts_with("- Skip memory")
        || line.starts_with("- Hard skip")
        || line.starts_with("- Use memory")
        || line.starts_with("- the query mentions")
        || line.starts_with("- Keep memory lookup")
        || line.starts_with("- `rollout_ids`")
        || line.starts_with("- Each update must")
        || line.starts_with("# Tools")
        || line.starts_with("# Valid channels:")
        || line.starts_with("# Task Group:")
        || line.starts_with("Use SSH when creating or changing Git remotes.")
        || line.starts_with("Do not create manual backup copies")
        || line.starts_with("Make requested edits directly")
        || line.starts_with("Keep changes focused")
        || line.starts_with("For debugging, review, or design questions")
        || line.starts_with("If a change would be destructive")
        || line.contains("Filesystem sandboxing defines which files can be read or written")
        || line.contains("available skills")
        || line.contains("AGENTS")
}

fn is_duplicate_line(lines: &[String], line: &str) -> bool {
    lines.iter().any(|existing| existing == line)
}

fn is_high_value_line(line: &str, question_terms: &[String]) -> bool {
    is_commit_line(line)
        || is_context_signal_line(line)
        || is_tool_result_line(line)
        || is_assistant_conclusion_line(line)
        || is_user_question_line(line)
        || matches_question_term(line, question_terms)
}

fn is_commit_line(line: &str) -> bool {
    line.starts_with("feat(")
        || line.starts_with("fix(")
        || line.starts_with("docs(")
        || line.starts_with("chore(")
        || line.starts_with("refactor(")
        || line.starts_with("test(")
        || line.starts_with("Validation:")
        || line.starts_with("- cargo ")
        || line.starts_with("- git ")
}

fn is_context_signal_line(line: &str) -> bool {
    let line = line.to_ascii_lowercase();
    contains_any(
        &line,
        &[
            "current objective",
            "objective:",
            "goal:",
            "decision:",
            "next step",
            "planned next step",
            "blocker",
            "blocked",
            "implementation status",
            "status:",
            "todo",
            "fixme",
        ],
    )
}

fn is_tool_result_line(line: &str) -> bool {
    line.starts_with("Committed as ")
        || line.starts_with("Final state:")
        || line.starts_with("Validation passed:")
        || line.starts_with("Implemented ")
        || line.starts_with("Changed:")
        || line.starts_with("No commit was made.")
}

fn is_assistant_conclusion_line(line: &str) -> bool {
    line.starts_with("The ")
        || line.starts_with("This ")
        || line.starts_with("I found ")
        || line.starts_with("I removed ")
        || line.starts_with("I’m ")
        || line.starts_with("I'm ")
}

fn is_user_question_line(line: &str) -> bool {
    line.ends_with('?') || line.starts_with("Please ") || line.starts_with("The goal ")
}

fn matches_question_term(line: &str, question_terms: &[String]) -> bool {
    let line_words = normalize_question_words(line);
    question_terms
        .iter()
        .filter(|term| term.len() > 3)
        .any(|term| line_words.iter().any(|word| word == term))
}

/// Error returned by a source adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterError {
    /// Source family that produced the error.
    pub source: Source,
    /// Human-readable error detail.
    pub message: String,
}

impl AdapterError {
    /// Creates an adapter error for a source.
    pub fn new(source: Source, message: impl Into<String>) -> Self {
        Self {
            source,
            message: message.into(),
        }
    }
}

impl fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} adapter: {}",
            self.source.as_str(),
            self.message
        )
    }
}

impl Error for AdapterError {}

/// Result returned by source adapters and Recall orchestration.
pub type AdapterResult<T> = Result<T, AdapterError>;

const ASK_STOP_WORDS: &[&str] = &[
    "when",
    "what",
    "why",
    "where",
    "who",
    "how",
    "did",
    "do",
    "does",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "behave",
    "has",
    "have",
    "had",
    "should",
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "my",
    "i",
    "me",
    "summarize",
    "summary",
    "today",
    "todays",
    "work",
    "leave",
    "off",
    "with",
];
const ASK_EMPTY_QUERY_FALLBACK_STOP_WORDS: &[&str] = &[
    "when",
    "what",
    "why",
    "where",
    "who",
    "how",
    "did",
    "do",
    "does",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "behave",
    "has",
    "have",
    "had",
    "should",
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "my",
    "i",
    "me",
    "summarize",
    "summary",
    "today",
    "todays",
];
const TEMPORAL_SUBJECT_STOP_WORDS: &[&str] = &[
    "evidence",
    "show",
    "shows",
    "shown",
    "actually",
    "activity",
    "that",
    "s",
    "completed",
    "complete",
    "finished",
    "finish",
    "implemented",
    "implement",
    "work",
    "works",
    "working",
    "accomplish",
    "accomplished",
    "changed",
    "happened",
    "recent",
    "recently",
    "last",
    "days",
    "day",
    "week",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

fn normalize_question_words(question: &str) -> Vec<String> {
    let mut normalized = String::new();
    for character in question.chars() {
        if character.is_alphanumeric() || character.is_whitespace() {
            normalized.extend(character.to_lowercase());
        } else {
            normalized.push(' ');
        }
    }

    normalized.split_whitespace().map(str::to_string).collect()
}

fn normalize_ask_query_words(words: &[String]) -> String {
    let normalized = words
        .iter()
        .map(String::as_str)
        .filter(|word| !ASK_STOP_WORDS.contains(word))
        .collect::<Vec<_>>();
    if !normalized.is_empty() {
        return normalized.join(" ");
    }

    let fallback = words
        .iter()
        .map(String::as_str)
        .filter(|word| !ASK_EMPTY_QUERY_FALLBACK_STOP_WORDS.contains(word))
        .collect::<Vec<_>>();
    if !fallback.is_empty() {
        return fallback.join(" ");
    }

    words
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ")
}

fn search_query_from_words(words: &[String]) -> SearchQuery {
    let intent = search_intent(words);
    if intent == SearchIntent::Plain {
        return SearchQuery::plain(normalize_ask_query_words(words));
    }

    let intent_terms = search_intent_terms(words, intent);
    let intent_term_refs = intent_terms.iter().map(String::as_str).collect::<Vec<_>>();
    let subject_terms = words
        .iter()
        .map(String::as_str)
        .filter(|word| !ASK_STOP_WORDS.contains(word) && !intent_term_refs.contains(word))
        .collect::<Vec<_>>();

    let subject = if subject_terms.is_empty() {
        normalize_ask_query_words(words)
    } else {
        subject_terms.join(" ")
    };

    SearchQuery {
        subject,
        intent,
        intent_terms,
    }
}

fn search_intent(words: &[String]) -> SearchIntent {
    if contains_word(words, "why") {
        return SearchIntent::Rationale;
    }
    if contains_word_sequence(words, &["did", "i", "discuss"])
        || contains_word_sequence(words, &["what", "did", "i", "discuss"])
        || contains_word_sequence(words, &["what", "we", "discussed"])
    {
        return SearchIntent::Discussion;
    }
    if contains_word(words, "landed")
        || contains_word_sequence(words, &["when", "did", "i", "implement"])
    {
        return SearchIntent::CompletedChange;
    }

    SearchIntent::Plain
}

fn search_intent_terms(words: &[String], intent: SearchIntent) -> Vec<String> {
    words
        .iter()
        .filter(|word| match intent {
            SearchIntent::Plain => false,
            SearchIntent::Rationale => {
                word.as_str() == "why"
                    || (word.as_str() == "introduce"
                        && contains_word_sequence(words, &["why", "did", "i", "introduce"]))
            }
            SearchIntent::Discussion => {
                matches!(word.as_str(), "about" | "discuss" | "discussed")
            }
            SearchIntent::CompletedChange if contains_word(words, "landed") => {
                matches!(word.as_str(), "actually" | "change" | "landed")
            }
            SearchIntent::CompletedChange => word.as_str() == "implement",
        })
        .cloned()
        .collect()
}

fn normalize_project_latest_query_words(words: &[String]) -> String {
    let normalized = words
        .iter()
        .map(String::as_str)
        .filter(|word| {
            !ASK_STOP_WORDS.contains(word) && !["resume", "left", "doing", "last"].contains(word)
        })
        .collect::<Vec<_>>();
    if !normalized.is_empty() {
        return normalized.join(" ");
    }

    normalize_ask_query_words(words)
}

fn normalize_temporal_subject_query_words(words: &[String]) -> String {
    words
        .iter()
        .map(String::as_str)
        .filter(|word| {
            !ASK_STOP_WORDS.contains(word)
                && !ASK_EMPTY_QUERY_FALLBACK_STOP_WORDS.contains(word)
                && !TEMPORAL_SUBJECT_STOP_WORDS.contains(word)
                && word.parse::<u32>().is_err()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_word(words: &[String], needle: &str) -> bool {
    words.iter().any(|word| word == needle)
}

fn contains_word_sequence(words: &[String], sequence: &[&str]) -> bool {
    words.windows(sequence.len()).any(|window| {
        window
            .iter()
            .map(String::as_str)
            .eq(sequence.iter().copied())
    })
}

fn has_project_latest_intent(words: &[String]) -> bool {
    contains_word(words, "resume")
        || contains_word_sequence(words, &["leave", "off"])
        || contains_word_sequence(words, &["left", "off"])
        || contains_word_sequence(words, &["doing", "last"])
        || contains_word_sequence(words, &["last", "doing"])
}

fn last_days_range(words: &[String]) -> Option<u32> {
    words
        .windows(3)
        .find(|window| window[0] == "last" && window[2] == "days")
        .and_then(|window| window[1].parse::<u32>().ok())
        .filter(|days| *days > 0)
}

fn explicit_date(question: &str) -> Option<NaiveDate> {
    explicit_iso_date(question)
        .or_else(|| explicit_month_name_date(&normalize_question_words(question)))
}

fn explicit_iso_date(question: &str) -> Option<NaiveDate> {
    question
        .as_bytes()
        .windows(10)
        .filter_map(|window| std::str::from_utf8(window).ok())
        .find_map(|candidate| NaiveDate::parse_from_str(candidate, "%Y-%m-%d").ok())
}

fn explicit_month_name_date(words: &[String]) -> Option<NaiveDate> {
    words.windows(3).find_map(|window| {
        let month = month_number(&window[0])?;
        let day = window[1].parse::<u32>().ok()?;
        let year = window[2].parse::<i32>().ok()?;
        NaiveDate::from_ymd_opt(year, month, day)
    })
}

fn month_number(month: &str) -> Option<u32> {
    match month {
        "january" => Some(1),
        "february" => Some(2),
        "march" => Some(3),
        "april" => Some(4),
        "may" => Some(5),
        "june" => Some(6),
        "july" => Some(7),
        "august" => Some(8),
        "september" => Some(9),
        "october" => Some(10),
        "november" => Some(11),
        "december" => Some(12),
        _ => None,
    }
}

fn parse_timestamp_date(timestamp: &Timestamp) -> Option<NaiveDate> {
    DateTime::parse_from_rfc3339(timestamp.as_str())
        .ok()
        .map(|timestamp| timestamp.date_naive())
        .or_else(|| {
            timestamp
                .as_str()
                .get(..10)
                .and_then(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d").ok())
        })
}

/// Common interface implemented by source adapters.
///
/// The adapter trait is deliberately narrow. It describes what Recall can ask a
/// source for without prescribing parsing, indexing, persistence, or transport.
pub trait Adapter {
    /// Source family exposed by this adapter.
    ///
    /// The value is used to qualify event ids and search results. Adapters for
    /// user-defined or experimental sources can return [`Source::Other`].
    fn source(&self) -> Source;

    /// Searches this source for development-memory evidence.
    ///
    /// Placeholder adapters may return an empty result set until real source
    /// extraction exists. Implementations should not fabricate events just to
    /// make search output non-empty.
    fn search(&self, query: &str) -> AdapterResult<Vec<SearchResult>>;

    /// Searches this source and returns each retained result with its event.
    ///
    /// The default preserves the existing search-then-inspect behavior.
    fn search_events(&self, query: &str) -> AdapterResult<Vec<SearchMatch>> {
        self.search(query)?
            .into_iter()
            .map(|result| {
                let event = self.inspect(&result.event.id)?.ok_or_else(|| {
                    AdapterError::new(
                        result.event.source.clone(),
                        format!(
                            "event not found: {}:{}",
                            result.event.source.as_str(),
                            result.event.id.as_str()
                        ),
                    )
                })?;
                Ok(SearchMatch { result, event })
            })
            .collect()
    }

    /// Returns a source-local timeline.
    ///
    /// Empty timelines are valid for placeholders and unavailable sources.
    fn timeline(&self) -> AdapterResult<Timeline>;

    /// Inspects one adapter-local identifier and returns the matching event
    /// when the source can provide it.
    ///
    /// The id is local to [`Adapter::source`]. Cross-source callers should pair
    /// the id with the adapter source before storing or comparing it.
    fn inspect(&self, id: &EventId) -> AdapterResult<Option<Event>>;
}

/// Source-agnostic registry of enabled adapters.
///
/// `Recall` owns the adapter set and provides the narrow dispatch surface that
/// callers use before parsing, indexing, ranking, or storage layers exist.
#[derive(Default)]
pub struct Recall {
    adapters: Vec<Box<dyn Adapter>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterCallTiming {
    pub source: Source,
    pub elapsed_ms: u64,
    pub item_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchDiagnostics {
    pub adapter_searches: Vec<AdapterCallTiming>,
    pub sort_ms: u64,
    pub total_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchWithDiagnostics {
    pub results: Vec<SearchMatch>,
    pub diagnostics: SearchDiagnostics,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimelineDiagnostics {
    pub adapter_timelines: Vec<AdapterCallTiming>,
    pub sort_ms: u64,
    pub total_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimelineWithDiagnostics {
    pub timeline: Timeline,
    pub diagnostics: TimelineDiagnostics,
}

impl Recall {
    /// Creates an empty adapter registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an adapter for future dispatch.
    pub fn register<A>(&mut self, adapter: A)
    where
        A: Adapter + 'static,
    {
        self.adapters.push(Box::new(adapter));
    }

    /// Returns the number of registered adapters.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Searches all registered adapters and returns results by descending score.
    pub fn search(&self, query: &str) -> AdapterResult<Vec<SearchResult>> {
        let mut results = self
            .adapters
            .iter()
            .map(|adapter| adapter.search(query))
            .try_fold(Vec::new(), |mut results, adapter_results| {
                results.extend(adapter_results?);
                Ok(results)
            })?;

        apply_project_ownership_boost_to_results(query, &mut results);
        sort_search_results(&mut results);

        Ok(results)
    }

    /// Searches all registered adapters and keeps each result's event.
    pub fn search_events(&self, query: &str) -> AdapterResult<Vec<SearchMatch>> {
        let mut results = self
            .adapters
            .iter()
            .map(|adapter| adapter.search_events(query))
            .try_fold(Vec::new(), |mut results, adapter_results| {
                results.extend(adapter_results?);
                Ok(results)
            })?;

        apply_project_ownership_boost_to_matches(query, &mut results);
        sort_search_matches(&mut results);

        Ok(results)
    }

    /// Searches all registered adapters and records opt-in diagnostic timings.
    pub fn search_with_diagnostics(&self, query: &str) -> AdapterResult<SearchWithDiagnostics> {
        let total_started = Instant::now();
        let mut results = Vec::new();
        let mut adapter_searches = Vec::new();
        for adapter in &self.adapters {
            let source = adapter.source();
            let search_started = Instant::now();
            let adapter_results = adapter.search_events(query)?;
            let elapsed_ms = elapsed_ms(search_started);
            let item_count = adapter_results.len();
            results.extend(adapter_results);
            adapter_searches.push(AdapterCallTiming {
                source,
                elapsed_ms,
                item_count,
            });
        }

        let sort_started = Instant::now();
        apply_project_ownership_boost_to_matches(query, &mut results);
        sort_search_matches(&mut results);
        let sort_ms = elapsed_ms(sort_started);
        let total_ms = elapsed_ms(total_started);

        Ok(SearchWithDiagnostics {
            results,
            diagnostics: SearchDiagnostics {
                adapter_searches,
                sort_ms,
                total_ms,
            },
        })
    }

    /// Returns a combined timeline from all registered adapters.
    pub fn timeline(&self) -> AdapterResult<Timeline> {
        let mut events = self
            .adapters
            .iter()
            .map(|adapter| adapter.timeline())
            .try_fold(Vec::new(), |mut events, timeline| {
                events.extend(timeline?.events);
                Ok(events)
            })?;

        sort_timeline_events(&mut events);

        Ok(Timeline { events })
    }

    /// Returns a combined timeline and records opt-in diagnostic timings.
    pub fn timeline_with_diagnostics(&self) -> AdapterResult<TimelineWithDiagnostics> {
        let total_started = Instant::now();
        let mut events = Vec::new();
        let mut adapter_timelines = Vec::new();
        for adapter in &self.adapters {
            let source = adapter.source();
            let timeline_started = Instant::now();
            let timeline = adapter.timeline()?;
            let elapsed_ms = elapsed_ms(timeline_started);
            let item_count = timeline.events.len();
            events.extend(timeline.events);
            adapter_timelines.push(AdapterCallTiming {
                source,
                elapsed_ms,
                item_count,
            });
        }

        let sort_started = Instant::now();
        sort_timeline_events(&mut events);
        let sort_ms = elapsed_ms(sort_started);
        let total_ms = elapsed_ms(total_started);

        Ok(TimelineWithDiagnostics {
            timeline: Timeline { events },
            diagnostics: TimelineDiagnostics {
                adapter_timelines,
                sort_ms,
                total_ms,
            },
        })
    }

    /// Inspects a source-qualified event by routing to the owning adapter.
    pub fn inspect(&self, event: &EventRef) -> AdapterResult<Option<Event>> {
        for adapter in &self.adapters {
            if adapter.source() == event.source {
                return adapter.inspect(&event.id);
            }
        }

        Ok(None)
    }

    /// Retrieves the events used by `recall ask` for a planned question.
    pub fn ask_retrieval(&self, plan: &RetrievalPlan) -> AdapterResult<AskRetrieval> {
        self.ask_retrieval_on(plan, Local::now().date_naive())
    }

    /// Retrieves ask evidence using an explicit local date for relative ranges.
    pub fn ask_retrieval_on(
        &self,
        plan: &RetrievalPlan,
        today: NaiveDate,
    ) -> AdapterResult<AskRetrieval> {
        match plan {
            RetrievalPlan::Search { query } => {
                let mut search_matches = self
                    .search_events(query.subject())?
                    .into_iter()
                    .take(ASK_RESULT_LIMIT)
                    .collect::<Vec<_>>();
                annotate_search_matches(query, &mut search_matches);
                let (search_results, events) = split_search_matches(search_matches);
                Ok(AskRetrieval {
                    events,
                    search_results,
                })
            }
            RetrievalPlan::ProjectLatest { query } => Ok(AskRetrieval {
                events: project_latest_events(self.timeline()?.events, query)
                    .into_iter()
                    .take(ASK_RESULT_LIMIT)
                    .collect(),
                search_results: Vec::new(),
            }),
            RetrievalPlan::Timeline { range, query } => {
                let events = timeline_events(self.timeline()?.events, range, query, today);
                Ok(AskRetrieval {
                    events,
                    search_results: Vec::new(),
                })
            }
        }
    }
}

/// Compiles ask retrieval events with the same plan-specific compiler branch as the CLI.
pub fn compile_ask_evidence(
    plan: &RetrievalPlan,
    question: &str,
    events: &[Event],
) -> Vec<EvidenceBlock> {
    let compiler = ContextCompiler::new();
    match plan {
        RetrievalPlan::Search { .. } => compiler.compile(question, events),
        RetrievalPlan::ProjectLatest { .. } => compiler.compile_project_latest(question, events),
        RetrievalPlan::Timeline { .. } => compiler.compile_timeline(question, events),
    }
}

fn split_search_matches(search_matches: Vec<SearchMatch>) -> (Vec<SearchResult>, Vec<Event>) {
    search_matches
        .into_iter()
        .map(|search_match| (search_match.result, search_match.event))
        .unzip()
}

fn annotate_search_matches(query: &SearchQuery, matches: &mut [SearchMatch]) {
    for search_match in matches {
        annotate_search_result(query, &mut search_match.result);
    }
}

/// Adds ask Search query diagnostics to search results.
pub fn annotate_search_results(query: &SearchQuery, results: &mut [SearchResult]) {
    for result in results {
        annotate_search_result(query, result);
    }
}

fn annotate_search_result(query: &SearchQuery, result: &mut SearchResult) {
    result
        .diagnostics
        .insert("search_subject".to_string(), query.subject().to_string());
    result.diagnostics.insert(
        "search_intent".to_string(),
        query.intent().as_str().to_string(),
    );
    if !query.intent_terms().is_empty() {
        result.diagnostics.insert(
            "search_intent_terms".to_string(),
            query.intent_terms().join(" "),
        );
    }
}

fn project_latest_events(events: Vec<Event>, query: &str) -> Vec<Event> {
    events
        .into_iter()
        .filter(|event| project_metadata_matches_query_text(&event.metadata, query))
        .collect()
}

fn event_matches_query(event: &Event, query: &str) -> bool {
    let query_terms = normalized_query_terms(query);
    if query_terms.is_empty() {
        return true;
    }

    let mut haystack = String::new();
    haystack.push_str(&event.title);
    haystack.push(' ');
    haystack.push_str(&event.description);
    for (key, value) in &event.metadata {
        haystack.push(' ');
        haystack.push_str(key);
        haystack.push(' ');
        haystack.push_str(value);
    }
    let haystack_terms = normalized_query_terms(&haystack);

    query_terms.iter().all(|term| {
        haystack_terms
            .iter()
            .any(|haystack_term| haystack_term == term)
    })
}

fn normalized_query_terms(text: &str) -> Vec<String> {
    let mut normalized = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() || character.is_whitespace() {
            normalized.extend(character.to_lowercase());
        } else {
            normalized.push(' ');
        }
    }

    normalized.split_whitespace().map(str::to_string).collect()
}

fn timeline_events(
    events: Vec<Event>,
    range: &DateRange,
    query: &str,
    today: NaiveDate,
) -> Vec<Event> {
    let in_range = events
        .into_iter()
        .filter(|event| {
            event
                .timestamp
                .as_ref()
                .is_some_and(|timestamp| range.contains_timestamp_on(timestamp, today))
        })
        .collect::<Vec<_>>();

    let narrowed = in_range
        .iter()
        .filter(|event| event_matches_query(event, query))
        .cloned()
        .collect::<Vec<_>>();
    // A subject term absent from the recognized vocabulary (see
    // TEMPORAL_SUBJECT_STOP_WORDS) must not silently empty an otherwise
    // populated range: fall back to the unnarrowed range in that case.
    let events = if narrowed.is_empty() {
        in_range
    } else {
        narrowed
    };

    match range {
        DateRange::Day(_) => events,
        _ => events.into_iter().take(ASK_RESULT_LIMIT).collect(),
    }
}

const PROJECT_OWNERSHIP_SCORE_BOOST: u32 = 1_000;

fn apply_project_ownership_boost_to_results(query: &str, results: &mut [SearchResult]) {
    let query_terms = normalize_question_words(query);
    for result in results {
        if project_metadata_matches_query(&result.metadata, &query_terms) {
            boost_search_result(result);
        }
    }
}

fn apply_project_ownership_boost_to_matches(query: &str, results: &mut [SearchMatch]) {
    let query_terms = normalize_question_words(query);
    for search_match in results {
        if project_metadata_matches_query(&search_match.event.metadata, &query_terms)
            || project_metadata_matches_query(&search_match.result.metadata, &query_terms)
        {
            boost_search_result(&mut search_match.result);
        }
    }
}

fn boost_search_result(result: &mut SearchResult) {
    let boosted_score = result
        .score
        .unwrap_or(0)
        .saturating_add(PROJECT_OWNERSHIP_SCORE_BOOST);
    result.score = Some(boosted_score);
    result
        .diagnostics
        .insert("score".to_string(), boosted_score.to_string());
    result.diagnostics.insert(
        "project_metadata_boost".to_string(),
        PROJECT_OWNERSHIP_SCORE_BOOST.to_string(),
    );
}

pub fn project_metadata_matches_query_text(metadata: &Metadata, query: &str) -> bool {
    let query_terms = normalize_question_words(query);
    project_metadata_matches_query(metadata, &query_terms)
}

fn project_metadata_matches_query(metadata: &Metadata, query_terms: &[String]) -> bool {
    if query_terms.is_empty() {
        return false;
    }

    [
        "cwd",
        "repo",
        "repository",
        "repository_root",
        "repo_root",
        "workspace",
    ]
    .iter()
    .filter_map(|key| metadata.get(*key))
    .any(|value| project_name_matches_query(value, query_terms))
}

fn project_name_matches_query(value: &str, query_terms: &[String]) -> bool {
    let name = project_name(value);
    let project_terms = normalize_question_words(&name);
    !project_terms.is_empty()
        && project_terms
            .iter()
            .all(|term| query_terms.iter().any(|query_term| query_term == term))
}

fn project_name(value: &str) -> String {
    value
        .trim()
        .trim_end_matches('/')
        .rsplit('/')
        .find(|part| !part.is_empty())
        .unwrap_or(value)
        .to_string()
}

fn sort_search_matches(results: &mut [SearchMatch]) {
    results.sort_by(|left, right| compare_search_results(&left.result, &right.result));
}

fn sort_search_results(results: &mut [SearchResult]) {
    results.sort_by(compare_search_results);
}

fn compare_search_results(left: &SearchResult, right: &SearchResult) -> std::cmp::Ordering {
    right
        .score
        .unwrap_or(0)
        .cmp(&left.score.unwrap_or(0))
        .then_with(|| left.event.source.cmp(&right.event.source))
        .then_with(|| left.event.id.cmp(&right.event.id))
        .then_with(|| left.snippet.cmp(&right.snippet))
}

fn sort_timeline_events(events: &mut [Event]) {
    events.sort_by(
        |left: &Event, right: &Event| match (&left.timestamp, &right.timestamp) {
            (Some(left), Some(right)) => right.cmp(left),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        },
    );
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct StaticAdapter {
        source: Source,
        event: Event,
        score: Option<u32>,
    }

    impl StaticAdapter {
        fn new(source: Source, id: &str, title: &str) -> Self {
            Self {
                source: source.clone(),
                event: Event::new(id, source, title),
                score: None,
            }
        }

        fn new_with_event(source: Source, event: Event) -> Self {
            Self {
                source,
                event,
                score: None,
            }
        }

        fn with_timestamp(mut self, timestamp: &str) -> Self {
            self.event.timestamp = Some(Timestamp::new(timestamp));
            self
        }

        fn with_score(mut self, score: u32) -> Self {
            self.score = Some(score);
            self
        }

        fn with_metadata(mut self, key: &str, value: &str) -> Self {
            self.event
                .metadata
                .insert(key.to_string(), value.to_string());
            self
        }
    }

    impl Adapter for StaticAdapter {
        fn source(&self) -> Source {
            self.source.clone()
        }

        fn search(&self, query: &str) -> AdapterResult<Vec<SearchResult>> {
            Ok(vec![SearchResult {
                event: EventRef::new(self.source.clone(), self.event.id.clone()),
                score: self.score,
                snippet: query.to_string(),
                metadata: self.event.metadata.clone(),
                diagnostics: Metadata::new(),
            }])
        }

        fn timeline(&self) -> AdapterResult<Timeline> {
            Ok(Timeline {
                events: vec![self.event.clone()],
            })
        }

        fn inspect(&self, id: &EventId) -> AdapterResult<Option<Event>> {
            Ok((id == &self.event.id).then(|| self.event.clone()))
        }
    }

    #[test]
    fn adapter_error_includes_source() {
        let error = AdapterError::new(Source::Codex, "missing sessions directory");

        assert_eq!(
            error.to_string(),
            "codex adapter: missing sessions directory"
        );
    }

    #[test]
    fn event_new_sets_minimal_fields() {
        let event = Event::new("one", Source::Other("test".to_string()), "Created scaffold");

        assert_eq!(event.id.as_str(), "one");
        assert_eq!(event.source.as_str(), "test");
        assert_eq!(event.title, "Created scaffold");
        assert!(event.timestamp.is_none());
        assert!(event.description.is_empty());
        assert!(event.metadata.is_empty());
    }

    #[test]
    fn event_ref_pairs_source_and_id() {
        let event_ref = EventRef::new(Source::Codex, "turn-1");

        assert_eq!(event_ref.source, Source::Codex);
        assert_eq!(event_ref.id.as_str(), "turn-1");
    }

    #[test]
    fn source_other_preserves_custom_label() {
        let source = Source::Other("shell".to_string());

        assert_eq!(source.as_str(), "shell");
    }

    #[test]
    fn prompt_builder_formats_question_and_events() {
        let evidence = EvidenceBlock {
            source: Source::Codex,
            id: EventId::new("session-1"),
            timestamp: Some(Timestamp::new("2026-07-20T13:00:00Z")),
            title: "Implement ask".to_string(),
            body: "Built a prompt from retrieved evidence.".to_string(),
        };

        let prompt = PromptBuilder::new().build("What happened?", &[evidence]);

        assert!(prompt.starts_with("You are answering questions about my engineering history."));
        assert!(prompt.contains("Use ONLY the supplied context."));
        assert!(prompt.contains("Question:\n\nWhat happened?\n\nContext:\n"));
        assert!(prompt.contains("=== Evidence ==="));
        assert!(prompt.contains("Source: codex"));
        assert!(prompt.contains("Id: session-1"));
        assert!(prompt.contains("Timestamp: 2026-07-20T13:00:00Z"));
        assert!(prompt.contains("Title: Implement ask"));
        assert!(prompt.contains("Evidence:\nBuilt a prompt from retrieved evidence."));
    }

    #[test]
    fn prompt_builder_includes_current_date_field() {
        let before = Local::now().format("%Y-%m-%d").to_string();
        let prompt = PromptBuilder::new().build("What did I work on today?", &[]);
        let after = Local::now().format("%Y-%m-%d").to_string();

        assert!(
            prompt.contains(&format!("Current date: {before}"))
                || prompt.contains(&format!("Current date: {after}"))
        );
        assert!(prompt.contains("Current local time: "));
    }

    #[test]
    fn prompt_builder_includes_temporal_grounding_instruction() {
        let prompt = PromptBuilder::new().build("What happened yesterday?", &[]);

        assert!(prompt.contains(
            "Interpret relative temporal expressions such as \"today\", \"yesterday\", \"this week\", and \"last week\" relative to this date."
        ));
    }

    #[test]
    fn prompt_builder_instructs_evidence_bounded_time_period_answers() {
        let prompt = PromptBuilder::new().build("What did I work on this week?", &[]);

        assert!(prompt.contains(
            "For time-period questions, phrase the answer as what the supplied evidence shows, not as a complete account of the period."
        ));
        assert!(prompt.contains(
            "Do not present inferred follow-ups or unresolved items as facts; label them as inference when mentioned."
        ));
    }

    #[test]
    fn prompt_builder_instructs_latest_timeline_state_to_supersede_intermediate_findings() {
        let earlier = EvidenceBlock {
            source: Source::Codex,
            id: EventId::new("session-earlier"),
            timestamp: Some(Timestamp::new("2026-08-28T09:00:00Z")),
            title: "Investigate build failure".to_string(),
            body: "Initial diagnosis: the flaky build failure was permanently resolved."
                .to_string(),
        };
        let later = EvidenceBlock {
            source: Source::Codex,
            id: EventId::new("session-later"),
            timestamp: Some(Timestamp::new("2026-08-28T11:30:00Z")),
            title: "Revisit build failure".to_string(),
            body: "Later evidence showed the earlier diagnosis was incomplete.".to_string(),
        };

        let prompt = PromptBuilder::new().build("What did I do today?", &[later, earlier]);

        assert!(prompt.contains(
            "When evidence describes an evolving or contradictory state, distinguish intermediate findings from the latest known state."
        ));
        assert!(prompt.contains(
            "If later timestamped evidence supersedes earlier state claims, make that final state clear while still preserving the chronology."
        ));
        assert!(prompt.contains(
            "Describe superseded earlier conclusions as what appeared true, was believed, or was concluded at that point, not as objective final facts."
        ));
        assert!(
            prompt.find("Id: session-later").unwrap() < prompt.find("Id: session-earlier").unwrap()
        );
    }

    #[test]
    fn date_range_uses_timestamp_calendar_date_without_utc_shift() {
        let range = DateRange::Day(NaiveDate::from_ymd_opt(2026, 8, 24).unwrap());
        let timestamp = Timestamp::new("2026-08-24T00:30:00+02:00");

        assert!(range.contains_timestamp(&timestamp));
    }

    #[test]
    fn retrieval_planner_keeps_keyword_questions_as_search() {
        let plan = RetrievalPlanner::new().plan("Why did I introduce EventRef?");

        assert_eq!(
            plan,
            RetrievalPlan::Search {
                query: SearchQuery {
                    subject: "eventref".to_string(),
                    intent: SearchIntent::Rationale,
                    intent_terms: vec!["why".to_string(), "introduce".to_string()]
                }
            }
        );
    }

    #[test]
    fn retrieval_planner_splits_completed_change_search_subject_and_intent() {
        let plan = RetrievalPlanner::new().plan("What EventRef change actually landed?");

        assert_eq!(
            plan,
            RetrievalPlan::Search {
                query: SearchQuery {
                    subject: "eventref".to_string(),
                    intent: SearchIntent::CompletedChange,
                    intent_terms: vec![
                        "change".to_string(),
                        "actually".to_string(),
                        "landed".to_string()
                    ]
                }
            }
        );
    }

    #[test]
    fn retrieval_planner_splits_discussion_search_subject_and_intent() {
        let plan = RetrievalPlanner::new().plan("Did I discuss EventRef?");

        assert_eq!(
            plan,
            RetrievalPlan::Search {
                query: SearchQuery {
                    subject: "eventref".to_string(),
                    intent: SearchIntent::Discussion,
                    intent_terms: vec!["discuss".to_string()]
                }
            }
        );
    }

    #[test]
    fn retrieval_planner_keeps_intent_like_words_when_they_are_subject_terms() {
        for (question, query) in [
            (
                "Why did the introduce command fail?",
                SearchQuery {
                    subject: "introduce command fail".to_string(),
                    intent: SearchIntent::Rationale,
                    intent_terms: vec!["why".to_string()],
                },
            ),
            (
                "What is the discussion parser?",
                SearchQuery::plain("discussion parser"),
            ),
            (
                "What changed in the change detector?",
                SearchQuery::plain("changed change detector"),
            ),
        ] {
            let plan = RetrievalPlanner::new().plan(question);

            assert_eq!(plan, RetrievalPlan::Search { query }, "{question}");
        }
    }

    #[test]
    fn retrieval_planner_preserves_implemented_search_subject() {
        let plan = RetrievalPlanner::new().plan("When did I implement the Git adapter?");

        assert_eq!(
            plan,
            RetrievalPlan::Search {
                query: SearchQuery {
                    subject: "git adapter".to_string(),
                    intent: SearchIntent::CompletedChange,
                    intent_terms: vec!["implement".to_string()]
                }
            }
        );
    }

    #[test]
    fn retrieval_planner_removes_leave_off_conversational_terms() {
        let plan = RetrievalPlanner::new().plan("Where did I leave off with disk-agent?");

        assert_eq!(
            plan,
            RetrievalPlan::ProjectLatest {
                query: "disk agent".to_string()
            }
        );
    }

    #[test]
    fn retrieval_planner_classifies_resume_project_questions_as_latest_state() {
        let plan = RetrievalPlanner::new().plan("Resume disk-agent.");

        assert_eq!(
            plan,
            RetrievalPlan::ProjectLatest {
                query: "disk agent".to_string()
            }
        );
    }

    #[test]
    fn retrieval_planner_classifies_doing_last_project_questions_as_latest_state() {
        let plan = RetrievalPlanner::new().plan("What was I doing last with disk-agent?");

        assert_eq!(
            plan,
            RetrievalPlan::ProjectLatest {
                query: "disk agent".to_string()
            }
        );
    }

    #[test]
    fn retrieval_planner_keeps_ordinary_project_questions_as_search() {
        let plan = RetrievalPlanner::new().plan("What is disk-agent?");

        assert_eq!(
            plan,
            RetrievalPlan::Search {
                query: SearchQuery::plain("disk agent")
            }
        );
    }

    #[test]
    fn retrieval_planner_classifies_today_questions_as_timeline() {
        let plan = RetrievalPlanner::new().plan("Summarize today's recall work.");

        assert_eq!(
            plan,
            RetrievalPlan::Timeline {
                range: DateRange::Today,
                query: "recall".to_string()
            }
        );
    }

    #[test]
    fn retrieval_planner_preserves_subject_terms_for_today_evidence_questions() {
        let plan = RetrievalPlanner::new()
            .plan("What evidence shows that I actually completed disk-guard today?");

        assert_eq!(
            plan,
            RetrievalPlan::Timeline {
                range: DateRange::Today,
                query: "disk guard".to_string()
            }
        );
    }

    #[test]
    fn retrieval_planner_keeps_generic_today_questions_broad() {
        for question in [
            "What did I do today?",
            "Summarize my activity today.",
            "What did I accomplish today?",
        ] {
            let plan = RetrievalPlanner::new().plan(question);

            assert_eq!(
                plan,
                RetrievalPlan::Timeline {
                    range: DateRange::Today,
                    query: String::new()
                },
                "{question}"
            );
        }
    }

    #[test]
    fn retrieval_planner_preserves_other_named_entities_for_today_questions() {
        for (question, query) in [
            ("What did I do with disk-guard today?", "disk guard"),
            ("What did I finish in Recall today?", "recall"),
            ("Did I finish recall-indexer today?", "recall indexer"),
        ] {
            let plan = RetrievalPlanner::new().plan(question);

            assert_eq!(
                plan,
                RetrievalPlan::Timeline {
                    range: DateRange::Today,
                    query: query.to_string()
                },
                "{question}"
            );
        }
    }

    #[test]
    fn retrieval_planner_classifies_relative_day_ranges_as_timeline() {
        let plan = RetrievalPlanner::new().plan("What changed in the last 3 days?");

        assert_eq!(
            plan,
            RetrievalPlan::Timeline {
                range: DateRange::LastDays(3),
                query: String::new()
            }
        );
    }

    #[test]
    fn retrieval_planner_classifies_last_week_questions_as_timeline() {
        let plan = RetrievalPlanner::new().plan("What happened last week?");

        assert_eq!(
            plan,
            RetrievalPlan::Timeline {
                range: DateRange::LastWeek,
                query: String::new()
            }
        );
    }

    #[test]
    fn retrieval_planner_classifies_month_name_dates_as_single_day_timeline() {
        let plan = RetrievalPlanner::new().plan("What did I work on in Recall on August 5, 2026?");

        assert_eq!(
            plan,
            RetrievalPlan::Timeline {
                range: DateRange::Day(NaiveDate::from_ymd_opt(2026, 8, 5).unwrap()),
                query: "recall".to_string()
            }
        );
    }

    #[test]
    fn retrieval_planner_classifies_iso_dates_as_single_day_timeline() {
        let plan = RetrievalPlanner::new().plan("What did I work on in Recall on 2026-08-05?");

        assert_eq!(
            plan,
            RetrievalPlan::Timeline {
                range: DateRange::Day(NaiveDate::from_ymd_opt(2026, 8, 5).unwrap()),
                query: "recall".to_string()
            }
        );
    }

    #[test]
    fn date_range_matches_expected_calendar_dates() {
        let today = NaiveDate::from_ymd_opt(2026, 8, 5).unwrap();

        assert!(DateRange::Day(today).contains_date(today, today));
        assert!(!DateRange::Day(today)
            .contains_date(NaiveDate::from_ymd_opt(2026, 8, 4).unwrap(), today));

        assert!(DateRange::Today.contains_date(today, today));
        assert!(
            !DateRange::Today.contains_date(NaiveDate::from_ymd_opt(2026, 8, 4).unwrap(), today)
        );

        assert!(
            DateRange::Yesterday.contains_date(NaiveDate::from_ymd_opt(2026, 8, 4).unwrap(), today)
        );
        assert!(!DateRange::Yesterday.contains_date(today, today));

        assert!(DateRange::LastDays(3)
            .contains_date(NaiveDate::from_ymd_opt(2026, 8, 3).unwrap(), today));
        assert!(!DateRange::LastDays(3)
            .contains_date(NaiveDate::from_ymd_opt(2026, 8, 2).unwrap(), today));

        assert!(
            DateRange::LastWeek.contains_date(NaiveDate::from_ymd_opt(2026, 7, 27).unwrap(), today)
        );
        assert!(
            DateRange::LastWeek.contains_date(NaiveDate::from_ymd_opt(2026, 8, 2).unwrap(), today)
        );
        assert!(
            !DateRange::LastWeek.contains_date(NaiveDate::from_ymd_opt(2026, 8, 3).unwrap(), today)
        );
    }

    #[test]
    fn context_compiler_is_deterministic() {
        let mut event = Event::new("session-1", Source::Codex, "Ask milestone");
        event.description = "Implemented ask routing.\nCommitted as `abc123`.".to_string();

        let compiler = ContextCompiler::new();

        assert_eq!(
            compiler.compile("What did I work on today?", &[event.clone()]),
            compiler.compile("What did I work on today?", &[event])
        );
    }

    #[test]
    fn context_compiler_categorization_is_deterministic() {
        let mut event = Event::new("session-1", Source::Codex, "Decision record");
        event.description = concat!(
            "Goal: keep compiler behavior deterministic.\n",
            "Decision: categorize before retention.\n",
            "Validation:\n",
            "- cargo test"
        )
        .to_string();

        let first = compile_events("What changed?", &[event.clone()]);
        let second = compile_events("What changed?", &[event]);

        assert_eq!(first, second);
        assert_eq!(
            first[0].categories,
            vec![
                ContextCategory::Objective,
                ContextCategory::Decision,
                ContextCategory::Validation,
            ]
        );
    }

    #[test]
    fn context_compiler_supports_multi_category_events() {
        let mut event = Event::new("session-1", Source::Codex, "Compiler phase 1");
        event.description = concat!(
            "The objective is to categorize compiled evidence.\n",
            "Decision: keep Event source-neutral.\n",
            "Implemented category assignment.\n",
            "Current implementation status: tests are being added.\n",
            "Next step: add retention priority.\n",
            "Blocker: none known.\n",
            "TODO: benchmark prompt size later.\n",
            "Validation: cargo test"
        )
        .to_string();

        let compiled = compile_events("What is the compiler status?", &[event]).remove(0);

        assert_eq!(
            compiled.categories,
            vec![
                ContextCategory::Objective,
                ContextCategory::Decision,
                ContextCategory::Milestone,
                ContextCategory::Status,
                ContextCategory::NextStep,
                ContextCategory::Validation,
                ContextCategory::Blocker,
                ContextCategory::Todo,
            ]
        );
        assert_eq!(compiled.event.source, Source::Codex);
        assert_eq!(compiled.event.id.as_str(), "session-1");
        assert!(compiled.metadata.contains_key("retained_lines"));
    }

    #[test]
    fn context_compiler_leaves_events_without_signals_uncategorized() {
        let mut event = Event::new("session-1", Source::Codex, "Casual discussion");
        event.description = "Talked through naming options and read nearby files.".to_string();

        let compiled = compile_events("What happened?", &[event]).remove(0);

        assert!(compiled.categories.is_empty());
    }

    #[test]
    fn context_compiler_preserves_input_order_for_compiled_events() {
        let mut first = Event::new("first", Source::Codex, "Decision");
        first.description = "Decision: keep adapters unchanged.".to_string();
        let mut second = Event::new("second", Source::Git, "feat(core): add category layer");
        second.description = "Implemented compiler categorization.".to_string();

        let compiled = compile_events("What changed?", &[first, second]);

        let refs = compiled
            .iter()
            .map(|event| event.event.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(refs, vec!["first", "second"]);
    }

    #[test]
    fn context_compiler_categorizes_without_adapter_changes() {
        let mut recall = Recall::new();
        let mut event = Event::new("event-1", Source::Other("test".to_string()), "Decision");
        event.description = "Decision: compiler policy stays out of adapters.".to_string();
        recall.register(StaticAdapter::new_with_event(
            Source::Other("test".to_string()),
            event,
        ));

        let events = recall.timeline().unwrap().events;
        let compiled = compile_events("What was decided?", &events);

        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].categories, vec![ContextCategory::Decision]);
    }

    #[test]
    fn context_compiler_derives_priority_from_categories() {
        let mut decision = Event::new("decision", Source::Codex, "Decision: keep compiler local");
        decision.description = "No output behavior change.".to_string();
        let mut milestone = Event::new("milestone", Source::Codex, "Implemented phase 2");
        milestone.description = "No retention edge cases found.".to_string();
        let mut todo = Event::new("todo", Source::Codex, "TODO: tune budget later");
        todo.description = "No semantic summarization yet.".to_string();

        let compiled = compile_events("What changed?", &[decision, milestone, todo]);

        assert_eq!(
            compiled
                .iter()
                .map(CompiledEvent::retention_priority)
                .collect::<Vec<_>>(),
            vec![
                RetentionPriority::Highest,
                RetentionPriority::High,
                RetentionPriority::Low,
            ]
        );
    }

    #[test]
    fn context_compiler_treats_added_lines_as_milestones() {
        assert_eq!(
            categorize_line("Added `EventRef`: source-qualified handles."),
            vec![ContextCategory::Milestone]
        );
        assert_eq!(
            categorize_line("- Added `EventRef`: source-qualified handles."),
            vec![ContextCategory::Milestone]
        );
    }

    #[test]
    fn context_compiler_uses_strongest_priority_for_multi_category_events() {
        let mut event = Event::new("event-1", Source::Codex, "Milestone");
        event.description = concat!(
            "Implemented category assignment.\n",
            "Decision: derive retention priority from categories.\n",
            "Validation: cargo test"
        )
        .to_string();

        let compiled = compile_events("What changed?", &[event]).remove(0);

        assert_eq!(compiled.retention_priority(), RetentionPriority::Highest);
    }

    #[test]
    fn context_compiler_treats_uncategorized_events_as_lowest_priority() {
        let mut event = Event::new("event-1", Source::Codex, "Casual discussion");
        event.description = "Read nearby files and talked through naming.".to_string();

        let compiled = compile_events("What happened?", &[event]).remove(0);

        assert_eq!(
            compiled.retention_priority(),
            RetentionPriority::Uncategorized
        );
    }

    #[test]
    fn context_compiler_budget_selection_prefers_higher_priority_events() {
        let mut low = Event::new("low", Source::Codex, "TODO: revisit phrasing");
        low.description = "TODO: tune this later.".to_string();
        let mut high = Event::new("high", Source::Codex, "Decision: keep Event source-neutral");
        high.description = "Decision: compiler policy stays internal.".to_string();

        let evidence = ContextCompiler::new().compile_with_options(
            "What should survive?",
            &[low, high],
            &CompileOptions {
                character_budget: 20,
            },
        );

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id.as_str(), "project:codex:high");
        assert!(evidence[0]
            .body
            .contains("Decision: keep Event source-neutral"));
    }

    #[test]
    fn context_compiler_budget_selection_uses_recency_within_equal_priority() {
        let mut older = Event::new("older", Source::Codex, "Decision: earlier");
        older.timestamp = Some(Timestamp::new("2026-08-01T10:00:00Z"));
        older.description = "Decision: earlier choice.".to_string();
        let mut newer = Event::new("newer", Source::Codex, "Decision: later");
        newer.timestamp = Some(Timestamp::new("2026-08-02T10:00:00Z"));
        newer.description = "Decision: later choice.".to_string();

        let evidence = ContextCompiler::new().compile_with_options(
            "What should survive?",
            &[older, newer],
            &CompileOptions {
                character_budget: 20,
            },
        );

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id.as_str(), "project:codex:newer");
        assert!(evidence[0].body.contains("Decision: later"));
    }

    #[test]
    fn context_compiler_preserves_input_order_for_equal_priority_without_timestamps() {
        let mut first = Event::new("first", Source::Codex, "Decision: first");
        first.description = "Decision: first.".to_string();
        let mut second = Event::new("second", Source::Codex, "Decision: second");
        second.description = "Decision: second.".to_string();

        let evidence = ContextCompiler::new().compile_with_options(
            "What changed?",
            &[first, second],
            &CompileOptions {
                character_budget: 1_000,
            },
        );

        assert_eq!(
            evidence
                .iter()
                .map(|block| block.id.as_str())
                .collect::<Vec<_>>(),
            vec!["project:codex:first", "project:codex:second"]
        );
    }

    #[test]
    fn context_compiler_budget_zero_omits_all_evidence() {
        let mut event = Event::new("event-1", Source::Codex, "Decision: keep deterministic");
        event.description = "Decision: no model call.".to_string();

        let evidence = ContextCompiler::new().compile_with_options(
            "What changed?",
            &[event],
            &CompileOptions {
                character_budget: 0,
            },
        );

        assert!(evidence.is_empty());
    }

    #[test]
    fn context_compiler_budget_includes_best_event_even_if_it_exceeds_budget() {
        let mut event = Event::new(
            "event-1",
            Source::Codex,
            "Decision: keep at least one useful item",
        );
        event.description =
            "Decision: avoid empty context when useful evidence exists.".to_string();

        let evidence = ContextCompiler::new().compile_with_options(
            "What changed?",
            &[event],
            &CompileOptions {
                character_budget: 1,
            },
        );

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id.as_str(), "project:codex:event-1");
    }

    #[test]
    fn context_compiler_deduplicates_identical_retained_lines() {
        let mut first = Event::new("session-1", Source::Codex, "Compiler milestone");
        first.description = "Implemented deterministic deduplication.".to_string();
        let mut second = Event::new("session-2", Source::Codex, "Compiler milestone");
        second.description = "Implemented deterministic deduplication.".to_string();

        let evidence = ContextCompiler::new().compile("What changed?", &[first, second]);

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id.as_str(), "project:codex:session-1");
        assert!(evidence[0]
            .body
            .contains("Implemented deterministic deduplication."));
    }

    #[test]
    fn context_compiler_deduplication_unions_categories() {
        let mut milestone = Event::new("session-1", Source::Codex, "Compiler milestone");
        milestone.description = "Implemented deterministic deduplication.".to_string();
        let mut decision = Event::new("session-1", Source::Codex, "Compiler decision");
        decision.description = "Decision: merge duplicate categories by union.".to_string();

        let deduplicated =
            deduplicate_compiled_events(compile_events("What changed?", &[milestone, decision]));

        assert_eq!(deduplicated.len(), 1);
        assert_eq!(
            deduplicated[0].categories,
            vec![ContextCategory::Decision, ContextCategory::Milestone]
        );
    }

    #[test]
    fn context_compiler_deduplicated_event_uses_strongest_category_priority() {
        let mut milestone = Event::new("session-1", Source::Codex, "Compiler milestone");
        milestone.description = "Implemented deterministic deduplication.".to_string();
        let mut next_step = Event::new("session-1", Source::Codex, "Compiler follow-up");
        next_step.description = "Next step: add project-state tests later.".to_string();

        let deduplicated =
            deduplicate_compiled_events(compile_events("What changed?", &[milestone, next_step]));

        assert_eq!(
            deduplicated[0].retention_priority(),
            RetentionPriority::Highest
        );
    }

    #[test]
    fn context_compiler_preserves_git_direct_evidence_separately_from_project_state() {
        let mut decision = Event::new("commit-event-1", Source::Codex, "Dedup phase");
        decision
            .metadata
            .insert("sha".to_string(), "abc123".to_string());
        decision.description = "Decision: use exact structural dedup keys.".to_string();
        let mut next_step = Event::new("commit-event-2", Source::Git, "Dedup phase");
        next_step
            .metadata
            .insert("sha".to_string(), "ABC123".to_string());
        next_step.description = "Next step: evaluate ProjectState separately.".to_string();

        let evidence = ContextCompiler::new().compile("What changed?", &[decision, next_step]);

        assert_eq!(evidence.len(), 2);
        assert!(evidence[0]
            .body
            .contains("Decision: use exact structural dedup keys."));
        assert_eq!(evidence[1].source, Source::Git);
        assert_eq!(evidence[1].id.as_str(), "commit-event-2");
        assert!(evidence[1]
            .body
            .contains("Next step: evaluate ProjectState separately."));
    }

    #[test]
    fn context_compiler_deduplicates_by_repository_title_and_timestamp() {
        let mut first = Event::new("event-1", Source::Codex, "Dedup phase");
        first.timestamp = Some(Timestamp::new("2026-08-04T10:00:00Z"));
        first
            .metadata
            .insert("repo".to_string(), "recall".to_string());
        first.description = "Decision: preserve categories.".to_string();
        let mut second = Event::new("event-2", Source::Codex, "  dedup   phase  ");
        second.timestamp = Some(Timestamp::new("2026-08-04T10:00:00Z"));
        second
            .metadata
            .insert("repository".to_string(), " Recall ".to_string());
        second.description = "Next step: keep ProjectState out of phase 3.".to_string();

        let evidence = ContextCompiler::new().compile("What changed?", &[first, second]);

        assert_eq!(evidence.len(), 1);
        assert!(evidence[0].body.contains("Decision: preserve categories."));
        assert!(evidence[0]
            .body
            .contains("Next step: keep ProjectState out of phase 3."));
    }

    #[test]
    fn context_compiler_deduplication_order_is_stable() {
        let mut first = Event::new("first", Source::Codex, "Decision: first");
        first.description = "Decision: first survives.".to_string();
        let mut duplicate = Event::new("first", Source::Codex, "Decision: first duplicate");
        duplicate.description = "Next step: merged into first.".to_string();
        let mut second = Event::new("second", Source::Codex, "Decision: second");
        second.description = "Decision: second survives.".to_string();

        let compiler = ContextCompiler::new();
        let first_run = compiler.compile(
            "What changed?",
            &[first.clone(), duplicate.clone(), second.clone()],
        );
        let second_run = compiler.compile("What changed?", &[first, duplicate, second]);

        assert_eq!(first_run, second_run);
        assert_eq!(
            first_run
                .iter()
                .map(|block| block.id.as_str())
                .collect::<Vec<_>>(),
            vec!["project:codex:first", "project:codex:second"]
        );
    }

    #[test]
    fn context_compiler_deduplication_reduces_prompt_fixture_without_losing_state() {
        let mut first = Event::new("session-1", Source::Codex, "Phase 3 dedup");
        first.description = concat!(
            "Decision: use deterministic deduplication only.\n",
            "Next step: keep ProjectState deferred."
        )
        .to_string();
        let mut duplicate = Event::new("session-1", Source::Codex, "Phase 3 duplicate");
        duplicate.description = concat!(
            "Decision: use deterministic deduplication only.\n",
            "Next step: keep ProjectState deferred.\n",
            "Validation: cargo test"
        )
        .to_string();
        let mut other = Event::new("session-2", Source::Git, "feat(core): add dedup");
        other.description = "Implemented deterministic deduplication.".to_string();

        let compiled = compile_events(
            "What changed in the context compiler?",
            &[first, duplicate, other],
        );
        let deduplicated = deduplicate_compiled_events(compiled.clone());

        assert!(deduplicated.len() < compiled.len());
        let evidence = ContextCompiler::new().compile(
            "What changed in the context compiler?",
            &[
                {
                    let mut event = Event::new("session-1", Source::Codex, "Phase 3 dedup");
                    event.description = concat!(
                        "Decision: use deterministic deduplication only.\n",
                        "Next step: keep ProjectState deferred."
                    )
                    .to_string();
                    event
                },
                {
                    let mut event = Event::new("session-1", Source::Codex, "Phase 3 duplicate");
                    event.description = concat!(
                        "Decision: use deterministic deduplication only.\n",
                        "Next step: keep ProjectState deferred.\n",
                        "Validation: cargo test"
                    )
                    .to_string();
                    event
                },
                {
                    let mut event = Event::new("session-2", Source::Git, "feat(core): add dedup");
                    event.description = "Implemented deterministic deduplication.".to_string();
                    event
                },
            ],
        );

        assert_eq!(evidence.len(), 2);
        assert!(evidence[0]
            .body
            .contains("Decision: use deterministic deduplication only."));
        assert!(evidence[0]
            .body
            .contains("Next step: keep ProjectState deferred."));
        assert!(evidence[0].body.contains("Validation: cargo test"));
    }

    #[test]
    fn context_compiler_project_state_keeps_git_events_as_direct_evidence() {
        let mut objective = Event::new("event-1", Source::Codex, "Recall planning");
        objective
            .metadata
            .insert("repo".to_string(), "recall".to_string());
        objective.description = concat!(
            "Objective: construct deterministic project state.\n",
            "Decision: keep grouping structural."
        )
        .to_string();
        let mut implementation = Event::new("event-2", Source::Git, "feat(core): project state");
        implementation
            .metadata
            .insert("repository".to_string(), "recall".to_string());
        implementation.description = concat!(
            "Implemented ProjectState construction.\n",
            "Next step: add semantic summarization later.\n",
            "Validation: cargo test"
        )
        .to_string();

        let evidence =
            ContextCompiler::new().compile("What is recall state?", &[objective, implementation]);

        assert_eq!(evidence.len(), 2);
        assert_eq!(evidence[0].title, "Project: recall");
        assert!(evidence[0].body.contains("Current objective:"));
        assert!(evidence[0]
            .body
            .contains("Objective: construct deterministic project state."));
        assert!(evidence[0].body.contains("Architectural decisions:"));
        assert!(evidence[0]
            .body
            .contains("Decision: keep grouping structural."));
        assert_eq!(evidence[1].source, Source::Git);
        assert_eq!(evidence[1].id.as_str(), "event-2");
        assert_eq!(evidence[1].title, "feat(core): project state");
        assert!(evidence[1].body.contains("repository: recall"));
        assert!(evidence[1]
            .body
            .contains("Implemented ProjectState construction."));
        assert!(evidence[1]
            .body
            .contains("Next step: add semantic summarization later."));
        assert!(evidence[1].body.contains("Validation: cargo test"));
    }

    #[test]
    fn context_compiler_project_state_keeps_multiple_projects_separated() {
        let mut recall = Event::new("recall-1", Source::Codex, "Decision: Recall state");
        recall
            .metadata
            .insert("repo".to_string(), "recall".to_string());
        recall.description = "Decision: build ProjectState internally.".to_string();
        let mut disk_agent = Event::new("disk-1", Source::Codex, "Decision: Disk agent state");
        disk_agent
            .metadata
            .insert("repo".to_string(), "disk-agent".to_string());
        disk_agent.description = "Decision: keep diagnostics read-only.".to_string();

        let evidence =
            ContextCompiler::new().compile("What decisions exist?", &[recall, disk_agent]);

        assert_eq!(evidence.len(), 2);
        assert_eq!(evidence[0].title, "Project: disk-agent");
        assert_eq!(evidence[1].title, "Project: recall");
        assert!(evidence[0]
            .body
            .contains("Decision: keep diagnostics read-only."));
        assert!(evidence[1]
            .body
            .contains("Decision: build ProjectState internally."));
    }

    #[test]
    fn context_compiler_project_latest_keeps_newest_event_direct() {
        let mut latest = Event::new("latest", Source::Codex, "Resume disk-agent");
        latest.timestamp = Some(Timestamp::new("2026-08-29T07:15:50.861Z"));
        latest.metadata.insert(
            "cwd".to_string(),
            "/home/simon/labs/repos/disk-agent".to_string(),
        );
        latest.description = concat!(
            "Commit: `ca46c169efa06389002a0157da5857a22f564bc7`\n",
            "Validation passed:\n",
            "- `cargo test`\n",
            "Final git status:\n",
            "```text\n",
            "## main...origin/main [ahead 1]\n",
            "```\n",
            "No push was performed."
        )
        .to_string();
        let mut older = Event::new("older", Source::Codex, "Implemented Podman attribution");
        older.timestamp = Some(Timestamp::new("2026-08-15T09:48:49.847Z"));
        older.metadata.insert(
            "cwd".to_string(),
            "/home/simon/labs/repos/disk-agent".to_string(),
        );
        older.description = "Implemented the minimal Podman attribution change.".to_string();

        let evidence = ContextCompiler::new()
            .compile_project_latest("Where did I leave off with disk-agent?", &[latest, older]);

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id.as_str(), "latest");
        assert_eq!(evidence[0].title, "Resume disk-agent");
        assert!(evidence[0]
            .body
            .contains("ca46c169efa06389002a0157da5857a22f564bc7"));
        assert!(evidence[0].body.contains("`cargo test`"));
        assert!(evidence[0].body.contains("## main...origin/main [ahead 1]"));
        assert!(!evidence[0].body.contains("Podman"));
    }

    #[test]
    fn context_compiler_project_state_preserves_added_question_term_evidence() {
        let mut event = Event::new("event-1", Source::Codex, "Architecture review");
        event
            .metadata
            .insert("cwd".to_string(), "/home/simon/labs/repos".to_string());
        event.description =
            "- Added `EventRef`: search results can point to source-qualified events without embedding full events."
                .to_string();

        let evidence = ContextCompiler::new()
            .compile("Why did I introduce EventRef?", &[event])
            .remove(0);

        assert!(evidence.body.contains(
            "Added `EventRef`: search results can point to source-qualified events without embedding full events."
        ));
    }

    #[test]
    fn context_compiler_project_state_preserves_blockers() {
        let mut event = Event::new("event-1", Source::Codex, "Recall blocker");
        event.metadata.insert(
            "cwd".to_string(),
            "/home/simon/labs/repos/recall".to_string(),
        );
        event.description = concat!(
            "Blocker: fixture coverage was missing.\n",
            "Next step: add blockers to ProjectState tests."
        )
        .to_string();

        let evidence = ContextCompiler::new().compile("What is blocked?", &[event]);

        assert_eq!(evidence.len(), 1);
        assert!(evidence[0].body.contains("Outstanding blockers:"));
        assert!(evidence[0]
            .body
            .contains("Blocker: fixture coverage was missing."));
        assert!(evidence[0].body.contains("Planned next step:"));
    }

    #[test]
    fn context_compiler_project_state_ordering_is_deterministic() {
        let mut second = Event::new("second", Source::Codex, "Decision: second");
        second
            .metadata
            .insert("repo".to_string(), "beta".to_string());
        second.description = "Decision: beta decision.".to_string();
        let mut first = Event::new("first", Source::Codex, "Decision: first");
        first
            .metadata
            .insert("repo".to_string(), "alpha".to_string());
        first.description = "Decision: alpha decision.".to_string();

        let compiler = ContextCompiler::new();
        let first_run = compiler.compile("What changed?", &[second.clone(), first.clone()]);
        let second_run = compiler.compile("What changed?", &[second, first]);

        assert_eq!(first_run, second_run);
        assert_eq!(
            first_run
                .iter()
                .map(|block| block.title.as_str())
                .collect::<Vec<_>>(),
            vec!["Project: alpha", "Project: beta"]
        );
    }

    #[test]
    fn context_compiler_project_state_keeps_git_duplicates_as_direct_evidence() {
        let mut first = Event::new("event-1", Source::Codex, "Recall decision");
        first
            .metadata
            .insert("repo".to_string(), "recall".to_string());
        first.description = "Decision: preserve deterministic state entries.".to_string();
        let mut duplicate = Event::new("event-2", Source::Git, "Recall decision");
        duplicate
            .metadata
            .insert("repo".to_string(), "recall".to_string());
        duplicate.description = "Decision: preserve deterministic state entries.".to_string();

        let evidence = ContextCompiler::new().compile("What was decided?", &[first, duplicate]);

        assert_eq!(evidence.len(), 2);
        assert_eq!(
            evidence[0]
                .body
                .matches("Decision: preserve deterministic state entries.")
                .count(),
            1
        );
        assert_eq!(evidence[1].source, Source::Git);
        assert_eq!(
            evidence[1]
                .body
                .matches("Decision: preserve deterministic state entries.")
                .count(),
            1
        );
    }

    #[test]
    fn context_compiler_excludes_environment_context() {
        let mut event = Event::new("session-1", Source::Codex, "<environment_context>");
        event.description = concat!(
            "<environment_context>\n",
            "cwd=/home/simon/labs/repos/recall\n",
            "filesystem permissions and tool instructions\n",
            "</environment_context>\n",
            "Implemented the typed retrieval planner.\n",
            "Committed as `6b2d000`."
        )
        .to_string();

        let evidence = ContextCompiler::new()
            .compile("What did I work on today?", &[event])
            .remove(0);

        assert_eq!(evidence.title, "Project: codex:session-1");
        assert!(!evidence.body.contains("filesystem permissions"));
        assert!(!evidence.body.contains("environment_context"));
        assert!(evidence
            .body
            .contains("Implemented the typed retrieval planner."));
        assert!(evidence.body.contains("Committed as `6b2d000`."));
    }

    #[test]
    fn context_compiler_excludes_prompt_scaffolding_lines() {
        let mut event = Event::new("session-2", Source::Codex, "handoff prompt");
        event.description = concat!(
            "What is the current objective?\n",
            "- What is the current objective?\n",
            "Current objective\n",
            "The objective is to complete the transition to a deterministic project-state compiler.\n",
            "What blockers remain?\n",
            "- What blockers remain?\n",
            "Outstanding blockers\n",
            "What did I work on today?\n",
        )
        .to_string();

        let evidence = ContextCompiler::new()
            .compile("What did I work on today?", &[event])
            .remove(0);

        assert!(!evidence.body.contains("What is the current objective?"));
        assert!(!evidence.body.contains("What blockers remain?"));
        assert!(!evidence.body.contains("What did I work on today?"));
        assert!(evidence
            .body
            .contains("The objective is to complete the transition to a deterministic project-state compiler."));
    }

    #[test]
    fn context_compiler_preserves_git_commit_direct_evidence() {
        let mut event = Event::new(
            "commit-1",
            Source::Git,
            "feat(retrieval): introduce typed retrieval planner",
        );
        event.timestamp = Some(Timestamp::new("2026-08-03T13:00:00Z"));
        event.metadata.insert(
            "cwd".to_string(),
            "/home/simon/labs/repos/recall".to_string(),
        );
        event
            .metadata
            .insert("sha".to_string(), "abcdef".to_string());
        event.description = "Validation:\n- cargo test\n- git diff --check".to_string();

        let evidence = ContextCompiler::new()
            .compile("Why did retrieval change?", &[event])
            .remove(0);

        assert_eq!(evidence.source, Source::Git);
        assert_eq!(evidence.id.as_str(), "commit-1");
        assert_eq!(
            evidence.timestamp,
            Some(Timestamp::new("2026-08-03T13:00:00Z"))
        );
        assert_eq!(
            evidence.title,
            "feat(retrieval): introduce typed retrieval planner"
        );
        assert!(!evidence.body.contains("Project key: sha:"));
        assert!(evidence.body.contains("sha: abcdef"));
        assert!(evidence.body.contains("Validation:"));
        assert!(evidence.body.contains("- cargo test"));
        assert!(evidence.body.contains("- git diff --check"));
    }

    #[test]
    fn context_compiler_produces_smaller_evidence_than_events() {
        let mut event = Event::new("session-1", Source::Codex, "Context compiler");
        let noisy_context = (0..100)
            .map(|index| format!("repeated shell prompt and environment line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        event.description = format!(
            "<environment_context>\n{noisy_context}\n</environment_context>\nThe prompt now uses compiled evidence."
        );
        let original_size = event.description.chars().count();

        let evidence = ContextCompiler::new()
            .compile("What did the prompt use?", &[event])
            .remove(0);

        assert!(evidence
            .body
            .contains("The prompt now uses compiled evidence."));
        assert!(evidence.body.chars().count() < original_size / 2);
    }

    #[test]
    fn prompt_builder_handles_missing_timestamps() {
        let evidence = EvidenceBlock {
            source: Source::Other("notes".to_string()),
            id: EventId::new("event-1"),
            timestamp: None,
            title: "No timestamp".to_string(),
            body: String::new(),
        };

        let prompt = PromptBuilder::new().build("Why?", &[evidence]);

        assert!(prompt.contains("Timestamp: \n"));
    }

    #[test]
    fn timeline_new_is_empty() {
        let timeline = Timeline::new();

        assert!(timeline.is_empty());
    }

    #[test]
    fn recall_registers_and_dispatches_to_adapters() {
        let mut recall = Recall::new();
        recall.register(StaticAdapter::new(Source::Codex, "codex-1", "Codex turn"));
        recall.register(StaticAdapter::new(Source::Git, "git-1", "Git commit"));

        assert_eq!(recall.adapter_count(), 2);
        assert_eq!(recall.search("why").unwrap().len(), 2);

        let timeline = recall.timeline().unwrap();
        assert_eq!(timeline.events.len(), 2);
        assert_eq!(timeline.events[0].title, "Codex turn");
        assert_eq!(timeline.events[1].title, "Git commit");

        let event = recall
            .inspect(&EventRef::new(Source::Git, "git-1"))
            .unwrap();
        assert_eq!(event.map(|event| event.source), Some(Source::Git));
    }

    #[test]
    fn recall_search_sorts_combined_results_by_score() {
        let mut recall = Recall::new();
        recall.register(StaticAdapter::new(Source::Codex, "early", "Early").with_score(10));
        recall.register(StaticAdapter::new(Source::Git, "later", "Later").with_score(90));

        let results = recall.search("query").unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].event, EventRef::new(Source::Git, "later"));
        assert_eq!(results[1].event, EventRef::new(Source::Codex, "early"));
        assert!(results.iter().all(|result| result.diagnostics.is_empty()));
    }

    #[test]
    fn recall_search_events_prefers_matching_project_metadata() {
        let mut recall = Recall::new();
        recall.register(
            StaticAdapter::new(
                Source::Codex,
                "misleading-recall",
                "Misleading recall session",
            )
            .with_score(200)
            .with_metadata("cwd", "/home/simon/labs/repos/recall"),
        );
        recall.register(
            StaticAdapter::new(Source::Codex, "disk-agent", "Disk agent project session")
                .with_score(20)
                .with_metadata("cwd", "/home/simon/labs/repos/disk-agent"),
        );

        let plan = RetrievalPlanner::new().plan("What is disk-agent?");
        let RetrievalPlan::Search { query } = plan else {
            panic!("expected search plan");
        };
        let results = recall.search_events(query.subject()).unwrap();

        assert_eq!(query.subject(), "disk agent");
        assert_eq!(results[0].event.id.as_str(), "disk-agent");
        assert_eq!(
            results[0]
                .result
                .diagnostics
                .get("project_metadata_boost")
                .map(String::as_str),
            Some("1000")
        );
        assert_eq!(results[1].event.id.as_str(), "misleading-recall");
    }

    #[test]
    fn recall_ask_search_uses_subject_query_and_preserves_intent_diagnostics() {
        let mut recall = Recall::new();
        recall.register(StaticAdapter::new(
            Source::Git,
            "eventref",
            "Refine EventRef core model",
        ));

        let plan = RetrievalPlanner::new().plan("What EventRef change actually landed?");
        let retrieval = recall.ask_retrieval(&plan).unwrap();

        assert_eq!(
            plan,
            RetrievalPlan::Search {
                query: SearchQuery {
                    subject: "eventref".to_string(),
                    intent: SearchIntent::CompletedChange,
                    intent_terms: vec![
                        "change".to_string(),
                        "actually".to_string(),
                        "landed".to_string()
                    ]
                }
            }
        );
        assert_eq!(retrieval.search_results.len(), 1);
        assert_eq!(retrieval.search_results[0].snippet, "eventref");
        assert_eq!(
            retrieval.search_results[0]
                .diagnostics
                .get("search_subject")
                .map(String::as_str),
            Some("eventref")
        );
        assert_eq!(
            retrieval.search_results[0]
                .diagnostics
                .get("search_intent")
                .map(String::as_str),
            Some("completed-change")
        );
        assert_eq!(
            retrieval.search_results[0]
                .diagnostics
                .get("search_intent_terms")
                .map(String::as_str),
            Some("change actually landed")
        );
    }

    #[test]
    fn ask_retrieval_timeline_falls_back_to_full_range_when_subject_terms_match_nothing() {
        let mut event = Event::new("today-event", Source::Codex, "Deployed the release");
        event.timestamp = Some(Timestamp::new("2026-08-31T09:00:00Z"));
        event.description = "Shipped the release to production.".to_string();
        let mut recall = Recall::new();
        recall.register(StaticAdapter::new_with_event(Source::Codex, event));

        let plan = RetrievalPlan::Timeline {
            range: DateRange::Today,
            query: "wrap up".to_string(),
        };
        let today = NaiveDate::from_ymd_opt(2026, 8, 31).unwrap();

        let retrieval = recall.ask_retrieval_on(&plan, today).unwrap();

        assert_eq!(retrieval.events.len(), 1);
        assert_eq!(retrieval.events[0].id.as_str(), "today-event");
    }

    #[test]
    fn recall_timeline_sorts_newest_first_with_untimestamped_events_last() {
        let mut recall = Recall::new();
        recall.register(
            StaticAdapter::new(Source::Codex, "codex-old", "Older Codex event")
                .with_timestamp("2026-07-19T10:00:00Z"),
        );
        recall.register(StaticAdapter::new(
            Source::Other("notes".to_string()),
            "notes-1",
            "Untimestamped note",
        ));
        recall.register(
            StaticAdapter::new(Source::Git, "git-new", "Newer Git event")
                .with_timestamp("2026-07-20T10:00:00Z"),
        );

        let timeline = recall.timeline().unwrap();

        let titles: Vec<_> = timeline
            .events
            .iter()
            .map(|event| event.title.as_str())
            .collect();
        assert_eq!(
            titles,
            vec!["Newer Git event", "Older Codex event", "Untimestamped note"]
        );
    }

    #[test]
    fn recall_inspect_routes_by_source() {
        let mut recall = Recall::new();
        recall.register(StaticAdapter::new(Source::Codex, "shared", "Codex event"));
        recall.register(StaticAdapter::new(Source::Git, "shared", "Git event"));

        let event = recall
            .inspect(&EventRef::new(Source::Git, "shared"))
            .unwrap()
            .unwrap();

        assert_eq!(event.source, Source::Git);
        assert_eq!(event.title, "Git event");
    }
}
