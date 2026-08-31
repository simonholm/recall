use chrono::NaiveDate;
use recall_core::{
    compile_ask_evidence, Adapter, AdapterResult, AskRetrieval, Event, EventId, EventRef,
    EvidenceBlock, Metadata, Recall, RetrievalPlan, RetrievalPlanner, SearchResult, Source,
    Timeline, Timestamp,
};
use serde_json::Value;

const CASES_JSON: &str = include_str!("../../../tests/retrieval/cases.json");
const DEFAULT_MAX_RANK: usize = 5;
// Fixture events are dated relative to this corpus anchor, not wall-clock time.
const EVAL_TODAY: &str = "2026-08-03";

#[test]
fn retrieval_eval_report_is_reproducible() {
    let cases = load_cases(CASES_JSON).expect("retrieval corpus should be valid");
    let first = run_evaluation(&cases, false).expect("retrieval evaluation should run");
    let second = run_evaluation(&cases, false).expect("retrieval evaluation should run twice");

    assert_eq!(first, second);
    assert!(first.contains("Summary\n-------"));

    let verbose = std::env::var_os("RECALL_RETRIEVAL_EVAL_VERBOSE").is_some();
    let report = if verbose {
        run_evaluation(&cases, true).expect("verbose retrieval evaluation should run")
    } else {
        first
    };

    println!("{report}");
}

#[test]
fn retrieval_eval_verbose_report_includes_failed_ranked_results() {
    let cases = load_cases(CASES_JSON).expect("retrieval corpus should be valid");
    let report = run_evaluation(&cases, true).expect("retrieval evaluation should run");

    assert!(report.contains("ranked results:"));
    assert!(report.contains("metadata    :"));
    assert!(report.contains("diagnostics :"));
    assert!(report.contains("matched     :"));
}

fn run_evaluation(cases: &[EvalCase], verbose: bool) -> Result<String, String> {
    let recall = fixture_recall();
    let today = NaiveDate::parse_from_str(EVAL_TODAY, "%Y-%m-%d").map_err(|e| e.to_string())?;
    let mut reports = Vec::new();

    for case in cases {
        let plan = RetrievalPlanner::new().plan(&case.question);
        let retrieval = recall
            .ask_retrieval_on(&plan, today)
            .map_err(|error| error.to_string())?;
        let evidence = compile_ask_evidence(&plan, &case.question, &retrieval.events);
        reports.push(evaluate_case(case, &plan, retrieval, evidence));
    }

    Ok(format_report(&reports, verbose))
}

fn evaluate_case(
    case: &EvalCase,
    plan: &RetrievalPlan,
    retrieval: AskRetrieval,
    evidence: Vec<EvidenceBlock>,
) -> CaseReport {
    let evaluation = EvaluationOutput::from_plan(plan, &retrieval.search_results, &evidence);
    let expected = case
        .expected
        .iter()
        .map(|expected| {
            let matched = std::iter::once(&expected.event)
                .chain(expected.alternatives.iter())
                .find_map(|candidate| {
                    evaluation
                        .events
                        .iter()
                        .position(|event| event == candidate)
                        .map(|index| MatchedResult {
                            event: candidate.clone(),
                            position: index + 1,
                        })
                });

            let status = match matched {
                Some(matched) if matched.position <= expected.max_rank => ExpectedStatus::Found {
                    event: matched.event,
                    position: matched.position,
                },
                Some(matched) => ExpectedStatus::BelowRank {
                    event: matched.event,
                    position: matched.position,
                    max_rank: expected.max_rank,
                },
                None => ExpectedStatus::Missing,
            };

            ExpectedReport {
                expected: expected.event.clone(),
                max_rank: expected.max_rank,
                status,
            }
        })
        .collect::<Vec<_>>();

    let passed = expected
        .iter()
        .all(|expected| matches!(expected.status, ExpectedStatus::Found { .. }));

    CaseReport {
        id: case.id.clone(),
        question: case.question.clone(),
        plan: format_retrieval_plan(plan),
        evaluation,
        retrieved_events: retrieval.events,
        evidence,
        expected,
        passed,
    }
}

fn format_report(reports: &[CaseReport], verbose: bool) -> String {
    let mut output = String::new();

    for report in reports {
        output.push_str(if report.passed { "PASS  " } else { "FAIL  " });
        output.push_str(&report.id);
        output.push('\n');
        output.push_str("  question: ");
        output.push_str(&report.question);
        output.push('\n');
        output.push_str("  plan: ");
        output.push_str(&report.plan);
        output.push('\n');

        for expected in &report.expected {
            output.push_str("  expected: ");
            output.push_str(&format_event_ref(&expected.expected));
            output.push_str(" <= ");
            output.push_str(report.evaluation.position_label);
            output.push(' ');
            output.push_str(&expected.max_rank.to_string());
            output.push('\n');
            output.push_str("  ");
            output.push_str(&format_expected_status(
                &expected.status,
                report.evaluation.position_label,
            ));
            output.push('\n');
        }

        if verbose {
            if !report.evaluation.search_results.is_empty()
                || matches!(report.evaluation.kind, EvaluationKind::Search)
            {
                output.push_str("  ranked results:\n");
                if report.evaluation.search_results.is_empty() {
                    output.push_str("    (none)\n");
                } else {
                    for (index, result) in report.evaluation.search_results.iter().enumerate() {
                        append_verbose_result(&mut output, index + 1, result);
                    }
                }
            } else {
                output.push_str("  retrieved events:\n");
                if report.retrieved_events.is_empty() {
                    output.push_str("    (none)\n");
                } else {
                    for (index, event) in report.retrieved_events.iter().enumerate() {
                        append_verbose_event(&mut output, index + 1, event);
                    }
                }

                output.push_str("  compiled evidence:\n");
                if report.evidence.is_empty() {
                    output.push_str("    (none)\n");
                } else {
                    for (index, evidence) in report.evidence.iter().enumerate() {
                        append_verbose_evidence(&mut output, index + 1, evidence);
                    }
                }
            }
        }

        output.push('\n');
    }

    let summary = Summary::from_reports(reports);
    output.push_str("Summary\n");
    output.push_str("-------\n");
    output.push_str("Queries : ");
    output.push_str(&summary.queries.to_string());
    output.push('\n');
    output.push_str("Search  : ");
    output.push_str(&summary.search_queries.to_string());
    output.push('\n');
    output.push_str("Evidence: ");
    output.push_str(&summary.evidence_queries.to_string());
    output.push('\n');
    output.push_str("Search Top-1   : ");
    output.push_str(&summary.top_1.to_string());
    output.push('\n');
    output.push_str("Search Top-3   : ");
    output.push_str(&summary.top_3.to_string());
    output.push('\n');
    output.push_str("Search Top-5   : ");
    output.push_str(&summary.top_5.to_string());
    output.push('\n');
    output.push_str("Misses  : ");
    output.push_str(&summary.misses.to_string());
    output.push('\n');

    output
}

fn format_expected_status(status: &ExpectedStatus, position_label: &str) -> String {
    match status {
        ExpectedStatus::Found { event, position } => {
            format!(
                "found: {} {position_label} {position}",
                format_event_ref(event)
            )
        }
        ExpectedStatus::BelowRank {
            event,
            position,
            max_rank,
        } => format!(
            "found below required {position_label}: {} {position_label} {position}, required <= {max_rank}",
            format_event_ref(event)
        ),
        ExpectedStatus::Missing => "result: not found".to_string(),
    }
}

fn format_metadata(metadata: &Metadata) -> String {
    if metadata.is_empty() {
        return "{}".to_string();
    }

    let pairs = metadata
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ");

    format!("{{{pairs}}}")
}

fn append_verbose_result(output: &mut String, rank: usize, result: &SearchResult) {
    output.push_str("    Rank ");
    output.push_str(&rank.to_string());
    output.push('\n');
    output.push_str("      source      : ");
    output.push_str(result.event.source.as_str());
    output.push('\n');
    output.push_str("      id          : ");
    output.push_str(result.event.id.as_str());
    output.push('\n');
    output.push_str("      score       : ");
    output.push_str(
        &result
            .score
            .map(|score| score.to_string())
            .unwrap_or_else(|| "none".to_string()),
    );
    output.push('\n');
    output.push_str("      matched     : ");
    output.push_str(diagnostic_value(result, "matched_fields"));
    output.push('\n');
    output.push_str("      terms       : ");
    output.push_str(diagnostic_value(result, "matched_terms"));
    output.push('\n');
    output.push_str("      components  : ");
    output.push_str(diagnostic_value(result, "score_components"));
    output.push('\n');
    output.push_str("      timestamp   : ");
    output.push_str(diagnostic_value(result, "timestamp"));
    output.push('\n');
    output.push_str("      metadata    : ");
    output.push_str(&format_metadata(&result.metadata));
    output.push('\n');
    output.push_str("      diagnostics : ");
    output.push_str(&format_metadata(&result.diagnostics));
    output.push('\n');
}

fn append_verbose_event(output: &mut String, position: usize, event: &Event) {
    output.push_str("    Event ");
    output.push_str(&position.to_string());
    output.push('\n');
    output.push_str("      source      : ");
    output.push_str(event.source.as_str());
    output.push('\n');
    output.push_str("      id          : ");
    output.push_str(event.id.as_str());
    output.push('\n');
    output.push_str("      timestamp   : ");
    output.push_str(
        event
            .timestamp
            .as_ref()
            .map(Timestamp::as_str)
            .unwrap_or("none"),
    );
    output.push('\n');
    output.push_str("      title       : ");
    output.push_str(&event.title);
    output.push('\n');
    output.push_str("      metadata    : ");
    output.push_str(&format_metadata(&event.metadata));
    output.push('\n');
}

fn append_verbose_evidence(output: &mut String, position: usize, evidence: &EvidenceBlock) {
    output.push_str("    Evidence ");
    output.push_str(&position.to_string());
    output.push('\n');
    output.push_str("      source      : ");
    output.push_str(evidence.source.as_str());
    output.push('\n');
    output.push_str("      id          : ");
    output.push_str(evidence.id.as_str());
    output.push('\n');
    output.push_str("      timestamp   : ");
    output.push_str(
        evidence
            .timestamp
            .as_ref()
            .map(Timestamp::as_str)
            .unwrap_or("none"),
    );
    output.push('\n');
    output.push_str("      title       : ");
    output.push_str(&evidence.title);
    output.push('\n');
}

fn diagnostic_value<'a>(result: &'a SearchResult, key: &str) -> &'a str {
    result
        .diagnostics
        .get(key)
        .map(String::as_str)
        .unwrap_or("n/a")
}

fn format_event_ref(event: &EventRef) -> String {
    format!("{}:{}", event.source.as_str(), event.id.as_str())
}

fn format_retrieval_plan(plan: &RetrievalPlan) -> String {
    match plan {
        RetrievalPlan::Search { query } => format!("search {query}"),
        RetrievalPlan::ProjectLatest { query } => format!("project latest {query}"),
        RetrievalPlan::Timeline { range, query } => {
            if query.is_empty() {
                format!("timeline {}", format_date_range(range))
            } else {
                format!("timeline {} {query}", format_date_range(range))
            }
        }
    }
}

fn format_date_range(range: &recall_core::DateRange) -> String {
    match range {
        recall_core::DateRange::Day(date) => date.to_string(),
        recall_core::DateRange::Today => "today".to_string(),
        recall_core::DateRange::Yesterday => "yesterday".to_string(),
        recall_core::DateRange::LastWeek => "last week".to_string(),
        recall_core::DateRange::LastDays(days) => format!("last {days} days"),
    }
}

fn load_cases(json: &str) -> Result<Vec<EvalCase>, String> {
    let value: Value = serde_json::from_str(json).map_err(|error| error.to_string())?;
    let schema_version = value
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing schema_version".to_string())?;
    if schema_version != 1 {
        return Err(format!("unsupported schema_version: {schema_version}"));
    }

    value
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing cases".to_string())?
        .iter()
        .map(parse_case)
        .collect()
}

fn parse_case(value: &Value) -> Result<EvalCase, String> {
    let id = required_string(value, "id")?;
    let question = required_string(value, "question")?;
    let expected = value
        .get("expected")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{id}: missing expected"))?
        .iter()
        .map(|expected| parse_expected(&id, expected))
        .collect::<Result<Vec<_>, _>>()?;

    if expected.is_empty() {
        return Err(format!("{id}: expected must not be empty"));
    }

    Ok(EvalCase {
        id,
        question,
        expected,
    })
}

fn parse_expected(case_id: &str, value: &Value) -> Result<Expected, String> {
    let source = required_string(value, "source")?;
    let id = required_string(value, "id")?;
    let max_rank = value
        .get("max_rank")
        .and_then(Value::as_u64)
        .map(|rank| rank as usize)
        .unwrap_or(DEFAULT_MAX_RANK);

    if max_rank == 0 {
        return Err(format!("{case_id}: max_rank must be greater than zero"));
    }

    let alternatives = value
        .get("alternatives")
        .and_then(Value::as_array)
        .map(|alternatives| {
            alternatives
                .iter()
                .map(|alternative| parse_match_ref(case_id, alternative))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    Ok(Expected {
        event: EventRef::new(parse_source(&source), id),
        max_rank,
        alternatives,
    })
}

fn parse_match_ref(case_id: &str, value: &Value) -> Result<EventRef, String> {
    let source = required_string(value, "source")?;
    let id = required_string(value, "id")?;

    if source.is_empty() || id.is_empty() {
        return Err(format!(
            "{case_id}: expected source and id must be non-empty"
        ));
    }

    Ok(EventRef::new(parse_source(&source), id))
}

fn required_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("missing {key}"))
}

fn parse_source(source: &str) -> Source {
    match source {
        "codex" => Source::Codex,
        "git" => Source::Git,
        other => Source::Other(other.to_string()),
    }
}

fn fixture_recall() -> Recall {
    let mut recall = Recall::new();
    recall.register(FixtureAdapter::new(Source::Codex, codex_events()));
    recall.register(FixtureAdapter::new(Source::Git, git_events()));
    recall
}

fn codex_events() -> Vec<Event> {
    vec![
        event(
            Source::Codex,
            "fixture-codex-eventref",
            "Introduce EventRef",
            "Discussed why recall should replace embedded search events with source-qualified eventref handles.",
            "2026-07-17T12:00:00Z",
            &[("fixture", "codex-eventref")],
        ),
        event(
            Source::Codex,
            "fixture-codex-git-adapter",
            "Implement Git adapter",
            "Implemented the git adapter by mapping commit history into recall events.",
            "2026-07-18T12:00:00Z",
            &[("fixture", "codex-git-adapter")],
        ),
        event(
            Source::Codex,
            "fixture-codex-inspect-missing",
            "Inspect missing event id",
            "Defined recall inspect behavior for a missing event id as a clear event not found error.",
            "2026-07-19T12:00:00Z",
            &[("fixture", "codex-inspect-missing")],
        ),
        event(
            Source::Codex,
            "fixture-codex-recall-today",
            "Recall project work",
            "Worked on the recall project evaluation benchmark.",
            "2026-08-03T12:00:00Z",
            &[("fixture", "codex-recall-today")],
        ),
        event(
            Source::Codex,
            "fixture-codex-recent-work",
            "Recent engineering work",
            "Captured a recent fixture window for recall evaluation.",
            "2026-08-02T12:00:00Z",
            &[("fixture", "codex-recent-work")],
        ),
        event(
            Source::Codex,
            "fixture-codex-distractor",
            "Prompt formatting",
            "Updated prompt formatting tests without changing retrieval ranking.",
            "2026-07-20T12:00:00Z",
            &[("fixture", "codex-distractor")],
        ),
        event(
            Source::Codex,
            "fixture-codex-recall-mentions-disk-agent",
            "Recall retrieval investigation",
            "Where did I leave off with disk agent? This recall session mentions disk-agent while debugging retrieval.",
            "2026-08-29T12:00:00Z",
            &[
                ("fixture", "codex-recall-mentions-disk-agent"),
                ("cwd", "/home/simon/labs/repos/recall"),
            ],
        ),
        event(
            Source::Codex,
            "fixture-codex-disk-agent-owned",
            "Disk-agent diagnostics",
            "Disk-agent work focused on read-only diagnostics and Cargo target reporting.",
            "2026-08-29T13:00:00Z",
            &[
                ("fixture", "codex-disk-agent-owned"),
                ("cwd", "/home/simon/labs/repos/disk-agent"),
            ],
        ),
    ]
}

fn git_events() -> Vec<Event> {
    vec![
        event(
            Source::Git,
            "fixture-git-eventref",
            "Refine EventRef core model",
            "Introduce source-qualified eventref values for search results.",
            "2026-07-17T13:00:00Z",
            &[("fixture", "git-eventref")],
        ),
        event(
            Source::Git,
            "fixture-git-adapter",
            "Implement Git adapter",
            "Add git commit history retrieval to recall.",
            "2026-07-18T13:00:00Z",
            &[("fixture", "git-adapter")],
        ),
        event(
            Source::Git,
            "fixture-git-recall-today",
            "Add recall evaluation corpus",
            "Create retrieval benchmark cases for recall.",
            "2026-08-03T13:00:00Z",
            &[("fixture", "git-recall-today")],
        ),
    ]
}

fn event(
    source: Source,
    id: &str,
    title: &str,
    description: &str,
    timestamp: &str,
    metadata: &[(&str, &str)],
) -> Event {
    let mut event = Event::new(id, source, title);
    event.description = description.to_string();
    event.timestamp = Some(Timestamp::new(timestamp));
    event.metadata = metadata
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    event
}

#[derive(Clone, Debug)]
struct FixtureAdapter {
    source: Source,
    events: Vec<Event>,
}

impl FixtureAdapter {
    fn new(source: Source, events: Vec<Event>) -> Self {
        Self { source, events }
    }
}

impl Adapter for FixtureAdapter {
    fn source(&self) -> Source {
        self.source.clone()
    }

    fn search(&self, query: &str) -> AdapterResult<Vec<SearchResult>> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let query_lower = query.to_lowercase();
        let terms = query_lower.split_whitespace().collect::<Vec<_>>();
        let mut results = self
            .events
            .iter()
            .filter_map(|event| search_event(event, &query_lower, &terms))
            .collect::<Vec<_>>();

        results.sort_by(|left, right| {
            right
                .score
                .unwrap_or(0)
                .cmp(&left.score.unwrap_or(0))
                .then_with(|| left.event.id.cmp(&right.event.id))
        });

        Ok(results)
    }

    fn timeline(&self) -> AdapterResult<Timeline> {
        Ok(Timeline {
            events: self.events.clone(),
        })
    }

    fn inspect(&self, id: &EventId) -> AdapterResult<Option<Event>> {
        Ok(self.events.iter().find(|event| event.id == *id).cloned())
    }
}

fn search_event(event: &Event, query_lower: &str, terms: &[&str]) -> Option<SearchResult> {
    let score = score_event(&event.title, &event.description, query_lower, terms)?;
    Some(SearchResult {
        event: EventRef::new(event.source.clone(), event.id.clone()),
        score: Some(score.total),
        snippet: snippet(&event.title, &event.description, query_lower, terms),
        metadata: event.metadata.clone(),
        diagnostics: search_diagnostics(event, query_lower, terms, &score),
    })
}

fn score_event(
    title: &str,
    description: &str,
    query_lower: &str,
    terms: &[&str],
) -> Option<ScoreBreakdown> {
    let title_lower = title.to_lowercase();
    let description_lower = description.to_lowercase();
    let haystack = format!("{title_lower}\n{description_lower}");

    if !terms.iter().all(|term| haystack.contains(term)) {
        return None;
    }

    let mut score = (terms.len() as u32) * 10;
    let term_count_score = score;
    let mut title_phrase_score = 0;
    let mut description_phrase_score = 0;
    if title_lower.contains(query_lower) {
        title_phrase_score = 100;
        score += title_phrase_score;
    } else if description_lower.contains(query_lower) {
        description_phrase_score = 50;
        score += description_phrase_score;
    }

    let mut title_term_score = 0;
    let mut description_term_score = 0;
    for term in terms {
        if title_lower.contains(term) {
            title_term_score += 5;
            score += 5;
        }
        if description_lower.contains(term) {
            description_term_score += 1;
            score += 1;
        }
    }

    Some(ScoreBreakdown::new(
        score,
        term_count_score,
        title_phrase_score,
        description_phrase_score,
        title_term_score,
        description_term_score,
    ))
}

fn search_diagnostics(
    event: &Event,
    query_lower: &str,
    terms: &[&str],
    score: &ScoreBreakdown,
) -> Metadata {
    let title_lower = event.title.to_lowercase();
    let description_lower = event.description.to_lowercase();
    let mut fields = Vec::new();
    if title_lower.contains(query_lower) || terms.iter().any(|term| title_lower.contains(term)) {
        fields.push("title");
    }
    if description_lower.contains(query_lower)
        || terms.iter().any(|term| description_lower.contains(term))
    {
        fields.push("description");
    }
    let matched_terms = terms
        .iter()
        .copied()
        .filter(|term| title_lower.contains(term) || description_lower.contains(term))
        .collect::<Vec<_>>()
        .join(", ");

    let mut diagnostics = Metadata::new();
    diagnostics.insert("adapter".to_string(), event.source.as_str().to_string());
    diagnostics.insert("source_id".to_string(), event.id.as_str().to_string());
    diagnostics.insert("score".to_string(), score.total.to_string());
    diagnostics.insert("matched_fields".to_string(), fields.join(", "));
    diagnostics.insert("matched_terms".to_string(), matched_terms);
    diagnostics.insert(
        "score_components".to_string(),
        format!(
            "term_count={}, title_phrase={}, description_phrase={}, title_terms={}, description_terms={}",
            score.term_count_score,
            score.title_phrase_score,
            score.description_phrase_score,
            score.title_term_score,
            score.description_term_score
        ),
    );
    if let Some(timestamp) = &event.timestamp {
        diagnostics.insert("timestamp".to_string(), timestamp.as_str().to_string());
    }
    diagnostics
}

#[derive(Debug)]
struct ScoreBreakdown {
    total: u32,
    term_count_score: u32,
    title_phrase_score: u32,
    description_phrase_score: u32,
    title_term_score: u32,
    description_term_score: u32,
}

impl ScoreBreakdown {
    fn new(
        total: u32,
        term_count_score: u32,
        title_phrase_score: u32,
        description_phrase_score: u32,
        title_term_score: u32,
        description_term_score: u32,
    ) -> Self {
        Self {
            total,
            term_count_score,
            title_phrase_score,
            description_phrase_score,
            title_term_score,
            description_term_score,
        }
    }
}

fn snippet(title: &str, description: &str, query_lower: &str, terms: &[&str]) -> String {
    for line in std::iter::once(title).chain(description.lines()) {
        let line_lower = line.to_lowercase();
        if line_lower.contains(query_lower) || terms.iter().all(|term| line_lower.contains(term)) {
            return line.to_string();
        }
    }

    title.to_string()
}

#[derive(Debug)]
struct EvalCase {
    id: String,
    question: String,
    expected: Vec<Expected>,
}

#[derive(Debug)]
struct Expected {
    event: EventRef,
    max_rank: usize,
    alternatives: Vec<EventRef>,
}

#[derive(Debug, Eq, PartialEq)]
struct MatchedResult {
    event: EventRef,
    position: usize,
}

#[derive(Debug)]
struct CaseReport {
    id: String,
    question: String,
    plan: String,
    evaluation: EvaluationOutput,
    retrieved_events: Vec<Event>,
    evidence: Vec<EvidenceBlock>,
    expected: Vec<ExpectedReport>,
    passed: bool,
}

#[derive(Debug)]
struct EvaluationOutput {
    kind: EvaluationKind,
    position_label: &'static str,
    events: Vec<EventRef>,
    search_results: Vec<SearchResult>,
}

impl EvaluationOutput {
    fn from_plan(
        plan: &RetrievalPlan,
        search_results: &[SearchResult],
        evidence: &[EvidenceBlock],
    ) -> Self {
        match plan {
            RetrievalPlan::Search { .. } => Self {
                kind: EvaluationKind::Search,
                position_label: "rank",
                events: search_results
                    .iter()
                    .map(|result| result.event.clone())
                    .collect(),
                search_results: search_results.to_vec(),
            },
            RetrievalPlan::ProjectLatest { .. } | RetrievalPlan::Timeline { .. } => Self {
                kind: EvaluationKind::CompiledEvidence,
                position_label: "evidence position",
                events: evidence
                    .iter()
                    .map(|evidence| EventRef::new(evidence.source.clone(), evidence.id.clone()))
                    .collect(),
                search_results: Vec::new(),
            },
        }
    }
}

#[derive(Debug)]
enum EvaluationKind {
    Search,
    CompiledEvidence,
}

#[derive(Debug)]
struct ExpectedReport {
    expected: EventRef,
    max_rank: usize,
    status: ExpectedStatus,
}

#[derive(Debug)]
enum ExpectedStatus {
    Found {
        event: EventRef,
        position: usize,
    },
    BelowRank {
        event: EventRef,
        position: usize,
        max_rank: usize,
    },
    Missing,
}

#[derive(Debug)]
struct Summary {
    queries: usize,
    search_queries: usize,
    evidence_queries: usize,
    top_1: usize,
    top_3: usize,
    top_5: usize,
    misses: usize,
}

impl Summary {
    fn from_reports(reports: &[CaseReport]) -> Self {
        let mut top_1 = 0;
        let mut top_3 = 0;
        let mut top_5 = 0;
        let mut search_queries = 0;
        let mut evidence_queries = 0;
        let mut misses = 0;

        for report in reports {
            match report.evaluation.kind {
                EvaluationKind::Search => search_queries += 1,
                EvaluationKind::CompiledEvidence => evidence_queries += 1,
            }

            let best_rank = report
                .expected
                .iter()
                .filter_map(|expected| match expected.status {
                    ExpectedStatus::Found { position, .. }
                    | ExpectedStatus::BelowRank { position, .. } => Some(position),
                    ExpectedStatus::Missing => None,
                })
                .min();

            if matches!(report.evaluation.kind, EvaluationKind::Search) {
                match best_rank {
                    Some(rank) if rank <= 1 => {
                        top_1 += 1;
                        top_3 += 1;
                        top_5 += 1;
                    }
                    Some(rank) if rank <= 3 => {
                        top_3 += 1;
                        top_5 += 1;
                    }
                    Some(rank) if rank <= 5 => {
                        top_5 += 1;
                    }
                    Some(_) => {}
                    None => {}
                }
            }

            misses += report
                .expected
                .iter()
                .filter(|expected| matches!(expected.status, ExpectedStatus::Missing))
                .count();
        }

        Self {
            queries: reports.len(),
            search_queries,
            evidence_queries,
            top_1,
            top_3,
            top_5,
            misses,
        }
    }
}
