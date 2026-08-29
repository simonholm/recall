use super::{
    ask_retrieval, compile_evidence, default_recall, format_event_ref,
    openrouter::OpenRouterConfig, send_configured_prompt_with_audit,
};
use chrono::{DateTime, FixedOffset};
use recall_core::{
    Adapter, AdapterResult, Event, EventId, EventRef, Metadata, PromptBuilder, Recall,
    RetrievalPlan, RetrievalPlanner, SearchResult, Source, Timeline, Timestamp,
};
use serde_json::Value;
use std::fmt::Write;
use std::io;

const ANSWER_CASES_JSON: &str = include_str!("../../../tests/answer/cases.json");

#[derive(Clone, Debug, Eq, PartialEq)]
struct AnswerEvalCase {
    id: String,
    question: String,
    as_of: Option<DateTime<FixedOffset>>,
    expected_facts: Vec<ExpectedFact>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedFact {
    name: String,
    phrases: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AskDiagnosticObservation {
    case_id: String,
    question: String,
    prompt: String,
    answer: Option<String>,
    answer_error: Option<String>,
    facts: Vec<FactObservation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FactObservation {
    name: String,
    evidence_present: bool,
    answer_present: Option<bool>,
    classification: FactClassification,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FactClassification {
    Success,
    SynthesisMiss,
    RetrievalCompilerMiss,
    SuspiciousUnsupported,
    AnswerNotRun,
}

fn load_answer_eval_cases(json: &str) -> Result<Vec<AnswerEvalCase>, String> {
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
        .map(parse_answer_eval_case)
        .collect()
}

fn parse_answer_eval_case(value: &Value) -> Result<AnswerEvalCase, String> {
    let id = required_json_string(value, "id")?;
    let question = required_json_string(value, "question")?;
    let as_of =
        optional_rfc3339_timestamp(value, "as_of").map_err(|error| format!("{id}: {error}"))?;
    let expected_facts = value
        .get("expected_facts")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{id}: missing expected_facts"))?
        .iter()
        .map(|fact| parse_expected_fact(&id, fact))
        .collect::<Result<Vec<_>, _>>()?;

    if expected_facts.is_empty() {
        return Err(format!("{id}: expected_facts must not be empty"));
    }

    Ok(AnswerEvalCase {
        id,
        question,
        as_of,
        expected_facts,
    })
}

fn parse_expected_fact(case_id: &str, value: &Value) -> Result<ExpectedFact, String> {
    let name = required_json_string(value, "name")?;
    let phrases = value
        .get("phrases")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{case_id}: {name}: missing phrases"))?
        .iter()
        .map(|phrase| {
            phrase
                .as_str()
                .filter(|phrase| !phrase.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| format!("{case_id}: {name}: phrases must be non-empty strings"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if phrases.is_empty() {
        return Err(format!("{case_id}: {name}: phrases must not be empty"));
    }

    Ok(ExpectedFact { name, phrases })
}

fn required_json_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("missing {key}"))
}

fn optional_rfc3339_timestamp(
    value: &Value,
    key: &str,
) -> Result<Option<DateTime<FixedOffset>>, String> {
    value
        .get(key)
        .map(|timestamp| {
            timestamp
                .as_str()
                .filter(|timestamp| !timestamp.trim().is_empty())
                .ok_or_else(|| format!("{key} must be a non-empty RFC3339 string"))
                .and_then(|timestamp| {
                    DateTime::parse_from_rfc3339(timestamp)
                        .map_err(|error| format!("{key} must be RFC3339: {error}"))
                })
        })
        .transpose()
}

fn run_answer_diagnostics(
    recall: &Recall,
    cases: &[AnswerEvalCase],
    answerer: impl Fn(&str) -> Result<Option<String>, String>,
) -> Result<Vec<AskDiagnosticObservation>, String> {
    cases
        .iter()
        .map(|case| observe_answer_case(recall, case, &answerer))
        .collect()
}

fn run_live_answer_diagnostics(
    recall: &Recall,
    cases: &[AnswerEvalCase],
    mut progress: impl io::Write,
    mut answerer: impl FnMut(&str) -> Result<Option<String>, String>,
) -> Result<Vec<AskDiagnosticObservation>, String> {
    let mut observations = Vec::new();

    for case in cases {
        let prompt = build_actual_ask_prompt(recall, case)?;
        writeln!(progress, "Running answer diagnostic: {}", case.id)
            .map_err(|error| error.to_string())?;
        writeln!(progress, "  {}", case.question).map_err(|error| error.to_string())?;
        progress.flush().map_err(|error| error.to_string())?;

        observations.push(observe_live_answer_case(case, prompt, &mut answerer));
    }

    Ok(observations)
}

fn observe_answer_case(
    recall: &Recall,
    case: &AnswerEvalCase,
    answerer: &impl Fn(&str) -> Result<Option<String>, String>,
) -> Result<AskDiagnosticObservation, String> {
    let prompt = build_actual_ask_prompt(recall, case)?;
    let answer = answerer(&prompt)?;
    let facts = case
        .expected_facts
        .iter()
        .map(|fact| observe_fact(fact, &prompt, answer.as_deref()))
        .collect();

    Ok(AskDiagnosticObservation {
        case_id: case.id.clone(),
        question: case.question.clone(),
        prompt,
        answer,
        answer_error: None,
        facts,
    })
}

fn observe_live_answer_case(
    case: &AnswerEvalCase,
    prompt: String,
    answerer: &mut impl FnMut(&str) -> Result<Option<String>, String>,
) -> AskDiagnosticObservation {
    let (answer, answer_error) = match answerer(&prompt) {
        Ok(answer) => (answer, None),
        Err(error) => (None, Some(error)),
    };
    let facts = case
        .expected_facts
        .iter()
        .map(|fact| observe_fact(fact, &prompt, answer.as_deref()))
        .collect();

    AskDiagnosticObservation {
        case_id: case.id.clone(),
        question: case.question.clone(),
        prompt,
        answer,
        answer_error,
        facts,
    }
}

fn build_actual_ask_prompt(recall: &Recall, case: &AnswerEvalCase) -> Result<String, String> {
    let plan = RetrievalPlanner::new().plan(&case.question);
    let events = if let Some(as_of) = &case.as_of {
        diagnostic_events_as_of(recall, &plan, as_of)?
    } else {
        ask_retrieval(recall, &plan)?.events
    };
    let evidence = compile_evidence(&plan, &case.question, &events);
    Ok(PromptBuilder::new().build(&case.question, &evidence))
}

fn diagnostic_events_as_of(
    recall: &Recall,
    plan: &RetrievalPlan,
    as_of: &DateTime<FixedOffset>,
) -> Result<Vec<Event>, String> {
    match plan {
        RetrievalPlan::Search { query } => diagnostic_search_events_as_of(recall, query, as_of),
        RetrievalPlan::ProjectLatest { query } => {
            let events = recall
                .timeline()
                .map_err(|error| error.to_string())?
                .events
                .into_iter()
                .filter(|event| {
                    event.timestamp.as_ref().is_some_and(|timestamp| {
                        timestamp_is_at_or_before(timestamp, as_of)
                            && recall_core::project_metadata_matches_query_text(
                                &event.metadata,
                                query,
                            )
                    })
                })
                .take(super::ASK_RESULT_LIMIT)
                .collect();
            Ok(events)
        }
        RetrievalPlan::Timeline { range } => {
            let events = recall
                .timeline()
                .map_err(|error| error.to_string())?
                .events
                .into_iter()
                .filter(|event| {
                    event.timestamp.as_ref().is_some_and(|timestamp| {
                        range.contains_timestamp(timestamp)
                            && timestamp_is_at_or_before(timestamp, as_of)
                    })
                });

            match range {
                recall_core::DateRange::Day(_) => Ok(events.collect()),
                _ => Ok(events.take(super::ASK_RESULT_LIMIT).collect()),
            }
        }
    }
}

fn diagnostic_search_events_as_of(
    recall: &Recall,
    query: &str,
    as_of: &DateTime<FixedOffset>,
) -> Result<Vec<Event>, String> {
    let mut events = Vec::new();
    for result in recall.search(query).map_err(|error| error.to_string())? {
        let event = recall
            .inspect(&result.event)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("event not found: {}", format_event_ref(&result.event)))?;
        if event
            .timestamp
            .as_ref()
            .is_some_and(|timestamp| timestamp_is_at_or_before(timestamp, as_of))
        {
            events.push(event);
            if events.len() == super::ASK_RESULT_LIMIT {
                break;
            }
        }
    }

    Ok(events)
}

fn timestamp_is_at_or_before(timestamp: &Timestamp, as_of: &DateTime<FixedOffset>) -> bool {
    DateTime::parse_from_rfc3339(timestamp.as_str())
        .map(|timestamp| timestamp <= *as_of)
        .unwrap_or(false)
}

fn observe_fact(fact: &ExpectedFact, prompt: &str, answer: Option<&str>) -> FactObservation {
    let evidence_present = contains_any_phrase(prompt, &fact.phrases);
    let answer_present = answer.map(|answer| contains_any_phrase(answer, &fact.phrases));
    let classification = classify_fact(evidence_present, answer_present);

    FactObservation {
        name: fact.name.clone(),
        evidence_present,
        answer_present,
        classification,
    }
}

fn contains_any_phrase(haystack: &str, phrases: &[String]) -> bool {
    let haystack = haystack.to_lowercase();
    phrases
        .iter()
        .map(|phrase| phrase.to_lowercase())
        .any(|phrase| haystack.contains(&phrase))
}

fn classify_fact(evidence_present: bool, answer_present: Option<bool>) -> FactClassification {
    match (evidence_present, answer_present) {
        (_, None) => FactClassification::AnswerNotRun,
        (true, Some(true)) => FactClassification::Success,
        (true, Some(false)) => FactClassification::SynthesisMiss,
        (false, Some(false)) => FactClassification::RetrievalCompilerMiss,
        (false, Some(true)) => FactClassification::SuspiciousUnsupported,
    }
}

fn format_answer_diagnostic_report(observations: &[AskDiagnosticObservation]) -> String {
    let mut output = String::new();

    for observation in observations {
        writeln!(output, "Question: {}", observation.case_id).unwrap();
        writeln!(output, "  {}", observation.question).unwrap();
        if let Some(error) = &observation.answer_error {
            writeln!(output, "  Error: {error}").unwrap();
        }
        for fact in &observation.facts {
            writeln!(
                output,
                "  - {}: evidence={} answer={} {}",
                fact.name,
                yes_no(fact.evidence_present),
                option_yes_no(fact.answer_present),
                format_fact_classification(fact.classification)
            )
            .unwrap();
        }
        writeln!(output).unwrap();
    }

    let summary = AnswerDiagnosticSummary::from_observations(observations);
    writeln!(output, "Summary").unwrap();
    writeln!(output, "-------").unwrap();
    writeln!(output, "Questions              : {}", observations.len()).unwrap();
    writeln!(output, "Facts                  : {}", summary.facts).unwrap();
    writeln!(output, "Success                : {}", summary.success).unwrap();
    writeln!(
        output,
        "Synthesis misses       : {}",
        summary.synthesis_misses
    )
    .unwrap();
    writeln!(
        output,
        "Retrieval/compiler miss: {}",
        summary.retrieval_compiler_misses
    )
    .unwrap();
    writeln!(
        output,
        "Suspicious unsupported : {}",
        summary.suspicious_unsupported
    )
    .unwrap();
    writeln!(
        output,
        "Answer not run         : {}",
        summary.answer_not_run
    )
    .unwrap();
    writeln!(output, "Answer errors          : {}", summary.answer_errors).unwrap();

    output
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn option_yes_no(value: Option<bool>) -> &'static str {
    value.map(yes_no).unwrap_or("n/a")
}

fn format_fact_classification(classification: FactClassification) -> &'static str {
    match classification {
        FactClassification::Success => "success",
        FactClassification::SynthesisMiss => "synthesis miss",
        FactClassification::RetrievalCompilerMiss => "retrieval/compiler miss",
        FactClassification::SuspiciousUnsupported => "suspicious/unsupported",
        FactClassification::AnswerNotRun => "answer not run",
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AnswerDiagnosticSummary {
    facts: usize,
    success: usize,
    synthesis_misses: usize,
    retrieval_compiler_misses: usize,
    suspicious_unsupported: usize,
    answer_not_run: usize,
    answer_errors: usize,
}

impl AnswerDiagnosticSummary {
    fn from_observations(observations: &[AskDiagnosticObservation]) -> Self {
        let mut summary = Self::default();

        for fact in observations
            .iter()
            .flat_map(|observation| observation.facts.iter())
        {
            summary.facts += 1;
            match fact.classification {
                FactClassification::Success => summary.success += 1,
                FactClassification::SynthesisMiss => summary.synthesis_misses += 1,
                FactClassification::RetrievalCompilerMiss => summary.retrieval_compiler_misses += 1,
                FactClassification::SuspiciousUnsupported => summary.suspicious_unsupported += 1,
                FactClassification::AnswerNotRun => summary.answer_not_run += 1,
            }
        }
        summary.answer_errors = observations
            .iter()
            .filter(|observation| observation.answer_error.is_some())
            .count();

        summary
    }
}

#[derive(Debug)]
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
        Ok(self
            .events
            .iter()
            .filter(|event| event.title.contains(query) || event.description.contains(query))
            .map(|event| SearchResult {
                event: EventRef::new(event.source.clone(), event.id.clone()),
                score: None,
                snippet: event.title.clone(),
                metadata: Metadata::new(),
                diagnostics: Metadata::new(),
            })
            .collect())
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

fn answer_eval_fixture_recall() -> Recall {
    let mut recall = Recall::new();
    let source = Source::Other("test".to_string());
    let mut event = Event::new("event-1", source.clone(), "introduce eventref");
    event.timestamp = Some(Timestamp::new("2026-07-20T12:00:00Z"));
    event
        .metadata
        .insert("cwd".to_string(), "/repo/recall".to_string());
    event.description =
        "Decision: use source-qualified EventRef handles for search results.".to_string();
    recall.register(FixtureAdapter::new(source, vec![event]));
    recall
}

#[test]
fn answer_eval_cases_load_from_separate_corpus() {
    let cases = load_answer_eval_cases(ANSWER_CASES_JSON).unwrap();

    assert_eq!(cases.len(), 5);
    assert_eq!(cases[0].id, "eventref-rationale");
    assert_eq!(
        cases[0].as_of.unwrap().to_rfc3339(),
        "2026-07-20T12:51:00+00:00"
    );
    assert_eq!(
        cases[0].expected_facts[0].name,
        "EventRef is source-qualified"
    );
    assert_eq!(
        cases[0].expected_facts[0].phrases,
        vec![
            "source-qualified eventref".to_string(),
            "source-qualified events".to_string(),
            "source-qualified references".to_string()
        ]
    );
    assert_eq!(cases[2].id, "retrieval-planner-rationale");
    assert_eq!(
        cases[2].as_of.unwrap().to_rfc3339(),
        "2026-08-02T15:31:00+00:00"
    );
    assert_eq!(cases[3].id, "context-compiler-project-handoff");
    assert_eq!(
        cases[3].as_of.unwrap().to_rfc3339(),
        "2026-08-04T10:39:00+00:00"
    );
    assert_eq!(cases[4].id, "august-26-clean-slate-handover");
    assert_eq!(
        cases[4].as_of.unwrap().to_rfc3339(),
        "2026-08-26T23:59:59+00:00"
    );
}

#[test]
fn answer_eval_case_parses_optional_as_of_rfc3339_timestamp() {
    let cases = load_answer_eval_cases(
        r#"{
          "schema_version": 1,
          "cases": [
            {
              "id": "with-cutoff",
              "question": "Why did I introduce EventRef?",
              "as_of": "2026-07-20T12:51:00Z",
              "expected_facts": [
                {
                  "name": "EventRef is source-qualified",
                  "phrases": ["source-qualified eventref"]
                }
              ]
            }
          ]
        }"#,
    )
    .unwrap();

    assert_eq!(
        cases[0].as_of.unwrap().to_rfc3339(),
        "2026-07-20T12:51:00+00:00"
    );
}

#[test]
fn answer_eval_case_rejects_non_rfc3339_as_of_timestamp() {
    let error = load_answer_eval_cases(
        r#"{
          "schema_version": 1,
          "cases": [
            {
              "id": "bad-cutoff",
              "question": "Why did I introduce EventRef?",
              "as_of": "2026-07-20",
              "expected_facts": [
                {
                  "name": "EventRef is source-qualified",
                  "phrases": ["source-qualified eventref"]
                }
              ]
            }
          ]
        }"#,
    )
    .unwrap_err();

    assert!(error.contains("bad-cutoff: as_of must be RFC3339"));
}

#[test]
fn phrase_matching_is_case_insensitive_and_accepts_alternatives() {
    let phrases = vec![
        "missing phrase".to_string(),
        "source-qualified EventRef".to_string(),
    ];

    assert!(contains_any_phrase(
        "Decision: use SOURCE-QUALIFIED eventref handles.",
        &phrases
    ));
}

#[test]
fn fact_classification_distinguishes_all_answer_states() {
    assert_eq!(classify_fact(true, Some(true)), FactClassification::Success);
    assert_eq!(
        classify_fact(true, Some(false)),
        FactClassification::SynthesisMiss
    );
    assert_eq!(
        classify_fact(false, Some(false)),
        FactClassification::RetrievalCompilerMiss
    );
    assert_eq!(
        classify_fact(false, Some(true)),
        FactClassification::SuspiciousUnsupported
    );
    assert_eq!(classify_fact(true, None), FactClassification::AnswerNotRun);
}

#[test]
fn answer_diagnostic_runner_uses_actual_ask_prompt_path() {
    let recall = answer_eval_fixture_recall();
    let cases = vec![AnswerEvalCase {
        id: "eventref".to_string(),
        question: "Why did I introduce EventRef?".to_string(),
        as_of: None,
        expected_facts: vec![ExpectedFact {
            name: "EventRef is source-qualified".to_string(),
            phrases: vec!["source-qualified eventref".to_string()],
        }],
    }];

    let observations = run_answer_diagnostics(&recall, &cases, |prompt| {
        assert!(prompt.starts_with("You are answering questions"));
        Ok(Some(
            "You introduced source-qualified EventRef handles.".to_string(),
        ))
    })
    .unwrap();

    assert_eq!(observations.len(), 1);
    assert!(observations[0]
        .prompt
        .contains("Use ONLY the supplied context."));
    assert_eq!(
        observations[0].facts[0].classification,
        FactClassification::Success
    );
}

#[test]
fn answer_diagnostic_as_of_excludes_newer_search_events_and_retains_historical_events() {
    let recall = answer_eval_recall_with_ranked_events();
    let cases = vec![AnswerEvalCase {
        id: "eventref".to_string(),
        question: "Why did I introduce EventRef?".to_string(),
        as_of: Some(DateTime::parse_from_rfc3339("2026-07-20T12:51:00Z").unwrap()),
        expected_facts: vec![ExpectedFact {
            name: "EventRef is source-qualified".to_string(),
            phrases: vec!["source-qualified eventref".to_string()],
        }],
    }];

    let observations = run_answer_diagnostics(&recall, &cases, |_| Ok(None)).unwrap();
    let prompt = &observations[0].prompt;

    assert!(prompt.contains("source-qualified EventRef"));
    assert!(prompt.contains("historical retained marker"));
    assert!(!prompt.contains("newer diagnostic marker"));
    assert!(observations[0].facts[0].evidence_present);
    assert_eq!(
        observations[0].facts[0].classification,
        FactClassification::AnswerNotRun
    );
}

#[test]
fn answer_diagnostic_as_of_is_applied_before_search_result_limit() {
    let recall = answer_eval_recall_with_ranked_events();
    let case = AnswerEvalCase {
        id: "eventref".to_string(),
        question: "Why did I introduce EventRef?".to_string(),
        as_of: Some(DateTime::parse_from_rfc3339("2026-07-20T12:51:00Z").unwrap()),
        expected_facts: vec![ExpectedFact {
            name: "EventRef is source-qualified".to_string(),
            phrases: vec!["source-qualified eventref".to_string()],
        }],
    };

    let prompt = build_actual_ask_prompt(&recall, &case).unwrap();

    assert!(prompt.contains("historical retained marker"));
}

#[test]
fn answer_diagnostic_without_as_of_preserves_existing_search_limit_behavior() {
    let recall = answer_eval_recall_with_ranked_events();
    let case = AnswerEvalCase {
        id: "eventref".to_string(),
        question: "Why did I introduce EventRef?".to_string(),
        as_of: None,
        expected_facts: vec![ExpectedFact {
            name: "EventRef is source-qualified".to_string(),
            phrases: vec!["source-qualified eventref".to_string()],
        }],
    };

    let prompt = build_actual_ask_prompt(&recall, &case).unwrap();

    assert!(prompt.contains("newer diagnostic marker"));
    assert!(!prompt.contains("historical retained marker"));
}

#[test]
fn answer_diagnostic_as_of_excludes_newer_timeline_events() {
    let mut recall = Recall::new();
    let source = Source::Other("test".to_string());
    recall.register(FixtureAdapter::new(
        source.clone(),
        vec![
            diagnostic_event(
                &source,
                "timeline-old",
                "timeline event",
                "Decision: Historical timeline retained marker.",
                "2026-07-20T12:00:00Z",
            ),
            diagnostic_event(
                &source,
                "timeline-new",
                "timeline event",
                "Decision: Newer timeline diagnostic marker.",
                "2026-07-20T13:00:00Z",
            ),
        ],
    ));
    let case = AnswerEvalCase {
        id: "timeline".to_string(),
        question: "What changed in the last 9999 days?".to_string(),
        as_of: Some(DateTime::parse_from_rfc3339("2026-07-20T12:51:00Z").unwrap()),
        expected_facts: vec![ExpectedFact {
            name: "historical timeline".to_string(),
            phrases: vec!["historical timeline retained marker".to_string()],
        }],
    };

    let prompt = build_actual_ask_prompt(&recall, &case).unwrap();

    assert!(prompt.contains("Historical timeline retained marker."));
    assert!(!prompt.contains("Newer timeline diagnostic marker."));
}

#[test]
fn answer_diagnostic_report_is_compact_and_summarizes_classes() {
    let observations = vec![AskDiagnosticObservation {
        case_id: "case-1".to_string(),
        question: "What happened?".to_string(),
        prompt: "prompt".to_string(),
        answer: Some("answer".to_string()),
        answer_error: None,
        facts: vec![
            FactObservation {
                name: "present".to_string(),
                evidence_present: true,
                answer_present: Some(true),
                classification: FactClassification::Success,
            },
            FactObservation {
                name: "missed".to_string(),
                evidence_present: true,
                answer_present: Some(false),
                classification: FactClassification::SynthesisMiss,
            },
        ],
    }];

    let report = format_answer_diagnostic_report(&observations);

    assert!(report.contains("Question: case-1"));
    assert!(report.contains("- present: evidence=yes answer=yes success"));
    assert!(report.contains("- missed: evidence=yes answer=no synthesis miss"));
    assert!(report.contains("Questions              : 1"));
    assert!(report.contains("Facts                  : 2"));
    assert!(report.contains("Success                : 1"));
    assert!(report.contains("Synthesis misses       : 1"));
    assert!(report.contains("Answer errors          : 0"));
}

#[test]
fn live_answer_diagnostic_prints_progress_before_answerer_invocation() {
    use std::cell::RefCell;
    use std::rc::Rc;

    struct SharedProgress {
        output: Rc<RefCell<Vec<u8>>>,
    }

    impl io::Write for SharedProgress {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.output.borrow_mut().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let recall = answer_eval_fixture_recall();
    let cases = vec![AnswerEvalCase {
        id: "eventref".to_string(),
        question: "Why did I introduce EventRef?".to_string(),
        as_of: None,
        expected_facts: vec![ExpectedFact {
            name: "EventRef is source-qualified".to_string(),
            phrases: vec!["source-qualified eventref".to_string()],
        }],
    }];
    let progress_output = Rc::new(RefCell::new(Vec::new()));
    let progress = SharedProgress {
        output: Rc::clone(&progress_output),
    };

    let observations = run_live_answer_diagnostics(&recall, &cases, progress, |_| {
        let output = String::from_utf8(progress_output.borrow().clone()).unwrap();
        assert!(output.contains("Running answer diagnostic: eventref"));
        assert!(output.contains("  Why did I introduce EventRef?"));
        Ok(Some(
            "You introduced source-qualified EventRef handles.".to_string(),
        ))
    })
    .unwrap();

    assert_eq!(observations.len(), 1);
}

#[test]
fn live_answer_diagnostic_continues_after_answerer_error() {
    let recall = answer_eval_fixture_recall();
    let cases = vec![
        AnswerEvalCase {
            id: "first".to_string(),
            question: "Why did I introduce EventRef?".to_string(),
            as_of: None,
            expected_facts: vec![ExpectedFact {
                name: "EventRef is source-qualified".to_string(),
                phrases: vec!["source-qualified eventref".to_string()],
            }],
        },
        AnswerEvalCase {
            id: "second".to_string(),
            question: "Why did I introduce EventRef?".to_string(),
            as_of: None,
            expected_facts: vec![ExpectedFact {
                name: "EventRef is source-qualified".to_string(),
                phrases: vec!["source-qualified eventref".to_string()],
            }],
        },
    ];
    let mut calls = 0;

    let observations = run_live_answer_diagnostics(&recall, &cases, Vec::new(), |_| {
        calls += 1;
        if calls == 1 {
            Err("openrouter request timed out: receive response".to_string())
        } else {
            Ok(Some(
                "You introduced source-qualified EventRef handles.".to_string(),
            ))
        }
    })
    .unwrap();

    assert_eq!(calls, 2);
    assert_eq!(observations.len(), 2);
    assert_eq!(
        observations[0].answer_error.as_deref(),
        Some("openrouter request timed out: receive response")
    );
    assert_eq!(observations[1].answer_error, None);
    assert_eq!(
        observations[1].facts[0].classification,
        FactClassification::Success
    );
}

#[test]
fn live_answer_diagnostic_represents_provider_error_as_answer_not_run() {
    let recall = answer_eval_fixture_recall();
    let cases = vec![AnswerEvalCase {
        id: "timeout".to_string(),
        question: "Why did I introduce EventRef?".to_string(),
        as_of: None,
        expected_facts: vec![ExpectedFact {
            name: "EventRef is source-qualified".to_string(),
            phrases: vec!["source-qualified eventref".to_string()],
        }],
    }];

    let observations = run_live_answer_diagnostics(&recall, &cases, Vec::new(), |_| {
        Err("openrouter request timed out: receive response".to_string())
    })
    .unwrap();
    let report = format_answer_diagnostic_report(&observations);

    assert_eq!(
        observations[0].facts[0].classification,
        FactClassification::AnswerNotRun
    );
    assert!(report.contains("Error: openrouter request timed out: receive response"));
    assert!(
        report.contains("- EventRef is source-qualified: evidence=yes answer=n/a answer not run")
    );
    assert!(report.contains("Synthesis misses       : 0"));
    assert!(report.contains("Retrieval/compiler miss: 0"));
    assert!(report.contains("Answer errors          : 1"));
}

#[test]
fn deterministic_answer_diagnostic_still_returns_answerer_errors() {
    let recall = answer_eval_fixture_recall();
    let cases = vec![AnswerEvalCase {
        id: "eventref".to_string(),
        question: "Why did I introduce EventRef?".to_string(),
        as_of: None,
        expected_facts: vec![ExpectedFact {
            name: "EventRef is source-qualified".to_string(),
            phrases: vec!["source-qualified eventref".to_string()],
        }],
    }];

    let error =
        run_answer_diagnostics(
            &recall,
            &cases,
            |_| Err("deterministic failure".to_string()),
        )
        .unwrap_err();

    assert_eq!(error, "deterministic failure");
}

#[test]
#[ignore = "calls the configured answer model and may send local Recall evidence"]
fn answer_diagnostic_report_can_call_configured_model() {
    let config = OpenRouterConfig::from_env();
    assert!(
        config.is_configured(),
        "OPENROUTER_API_KEY is required for the live answer diagnostic"
    );
    let recall = default_recall();
    let cases = load_answer_eval_cases(ANSWER_CASES_JSON).unwrap();
    let observations = run_live_answer_diagnostics(&recall, &cases, io::stdout(), |prompt| {
        send_configured_prompt_with_audit(&config, "answer diagnostic", prompt, false, None)
            .map(|response| Some(response.answer))
    })
    .unwrap();

    println!("{}", format_answer_diagnostic_report(&observations));
}

fn answer_eval_recall_with_ranked_events() -> Recall {
    let mut recall = Recall::new();
    let source = Source::Other("test".to_string());
    let mut events = Vec::new();
    for index in 0..9 {
        events.push(diagnostic_event(
            &source,
            &format!("event-{index:02}-new"),
            "introduce eventref",
            "Decision: newer diagnostic marker for introduce eventref.",
            "2026-08-26T15:00:00Z",
        ));
    }
    events.push(diagnostic_event(
        &source,
        "event-09-old",
        "introduce eventref",
        "Decision: historical retained marker: use source-qualified EventRef handles.",
        "2026-07-20T12:00:00Z",
    ));
    recall.register(FixtureAdapter::new(source, events));
    recall
}

fn diagnostic_event(
    source: &Source,
    id: &str,
    title: &str,
    description: &str,
    timestamp: &str,
) -> Event {
    let mut event = Event::new(id, source.clone(), title);
    event.description = description.to_string();
    event.timestamp = Some(Timestamp::new(timestamp));
    event
}
