//! Codex source adapter.
//!
//! This adapter reads local Codex session JSONL files and maps each session
//! file to one source-neutral Recall event. It intentionally does no indexing,
//! ranking, caching, or cross-source correlation.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use recall_core::{
    Adapter, AdapterError, AdapterResult, Event, EventId, EventRef, Metadata, SearchMatch,
    SearchResult, Source, Timeline, Timestamp,
};
use serde_json::Value;

/// Adapter for local Codex session JSONL files.
#[derive(Clone, Debug)]
pub struct CodexAdapter {
    sessions_dir: PathBuf,
}

impl Default for CodexAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexAdapter {
    /// Creates a Codex adapter pointed at `$HOME/.codex/sessions`.
    pub fn new() -> Self {
        let sessions_dir = env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".codex")
            .join("sessions");

        Self { sessions_dir }
    }

    /// Creates a Codex adapter pointed at an explicit sessions directory.
    pub fn with_sessions_dir(sessions_dir: impl Into<PathBuf>) -> Self {
        Self {
            sessions_dir: sessions_dir.into(),
        }
    }

    /// Reads Codex session files and returns one event per readable session.
    pub fn scan(&self) -> AdapterResult<Vec<Event>> {
        if !self.sessions_dir.exists() {
            return Err(self.error(format!(
                "sessions directory does not exist: {}",
                self.sessions_dir.display()
            )));
        }

        if !self.sessions_dir.is_dir() {
            return Err(self.error(format!(
                "sessions path is not a directory: {}",
                self.sessions_dir.display()
            )));
        }

        let mut files = Vec::new();
        collect_session_files(&self.sessions_dir, &mut files).map_err(|error| {
            self.error(format!(
                "failed to read sessions directory {}: {error}",
                self.sessions_dir.display()
            ))
        })?;

        files.sort();

        let mut events = Vec::new();

        for file in files {
            if let Some(event) = self.read_session_file(&file)? {
                events.push(event);
            }
        }

        Ok(events)
    }

    fn read_session_file(&self, path: &Path) -> AdapterResult<Option<Event>> {
        let text = fs::read_to_string(path).map_err(|error| {
            self.error(format!(
                "failed to read session file {}: {error}",
                path.display()
            ))
        })?;

        let mut builder = SessionBuilder::new(path);

        for (line_index, line) in text.lines().enumerate() {
            match serde_json::from_str::<Value>(line) {
                Ok(value) => builder.record(&value),
                Err(error) => builder.record_skipped_line(line_index + 1, &error),
            }
        }

        Ok(builder.finish())
    }

    fn error(&self, message: impl Into<String>) -> AdapterError {
        AdapterError::new(Source::Codex, message)
    }
}

impl Adapter for CodexAdapter {
    fn source(&self) -> Source {
        Source::Codex
    }

    fn search(&self, query: &str) -> AdapterResult<Vec<SearchResult>> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let query_lower = query.to_lowercase();
        let terms: Vec<_> = query_lower.split_whitespace().collect();

        let mut results: Vec<_> = self
            .scan()?
            .into_iter()
            .filter_map(|event| search_event(event, &query_lower, &terms))
            .collect();
        results.sort_by(|left, right| {
            right
                .score
                .unwrap_or(0)
                .cmp(&left.score.unwrap_or(0))
                .then_with(|| left.event.id.cmp(&right.event.id))
        });

        Ok(results)
    }

    fn search_events(&self, query: &str) -> AdapterResult<Vec<SearchMatch>> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let query_lower = query.to_lowercase();
        let terms: Vec<_> = query_lower.split_whitespace().collect();

        let mut results: Vec<_> = self
            .scan()?
            .into_iter()
            .filter_map(|event| {
                search_event(event.clone(), &query_lower, &terms)
                    .map(|result| SearchMatch { result, event })
            })
            .collect();
        results.sort_by(|left, right| {
            right
                .result
                .score
                .unwrap_or(0)
                .cmp(&left.result.score.unwrap_or(0))
                .then_with(|| left.result.event.id.cmp(&right.result.event.id))
        });

        Ok(results)
    }

    fn timeline(&self) -> AdapterResult<Timeline> {
        Ok(Timeline {
            events: self.scan()?,
        })
    }

    fn inspect(&self, id: &EventId) -> AdapterResult<Option<Event>> {
        Ok(self.scan()?.into_iter().find(|event| event.id == *id))
    }
}

#[derive(Debug)]
struct SessionBuilder {
    path: PathBuf,
    id: Option<String>,
    timestamp: Option<String>,
    title: Option<String>,
    description: String,
    metadata: Metadata,
}

impl SessionBuilder {
    fn new(path: &Path) -> Self {
        let mut metadata = Metadata::new();
        metadata.insert("path".to_string(), path.display().to_string());

        Self {
            path: path.to_path_buf(),
            id: None,
            timestamp: None,
            title: None,
            description: String::new(),
            metadata,
        }
    }

    fn record(&mut self, value: &Value) {
        match value.get("type").and_then(Value::as_str) {
            Some("session_meta") => self.record_session_meta(value),
            Some("response_item") => self.record_response_item(value),
            _ => {}
        }
    }

    fn record_skipped_line(&mut self, line_number: usize, error: &serde_json::Error) {
        let count = self
            .metadata
            .get("skipped_records")
            .and_then(|count| count.parse::<usize>().ok())
            .unwrap_or(0)
            + 1;

        self.metadata
            .insert("skipped_records".to_string(), count.to_string());
        let diagnostic = format!("{} line {}: {error}", self.path.display(), line_number);
        self.metadata
            .entry("skipped_record_diagnostic".to_string())
            .and_modify(|existing| {
                existing.push_str("; ");
                existing.push_str(&diagnostic);
            })
            .or_insert(diagnostic);
    }

    fn record_session_meta(&mut self, value: &Value) {
        self.timestamp = value
            .pointer("/payload/timestamp")
            .and_then(Value::as_str)
            .or_else(|| value.get("timestamp").and_then(Value::as_str))
            .map(str::to_string)
            .or_else(|| self.timestamp.take());

        self.id = value
            .pointer("/payload/id")
            .and_then(Value::as_str)
            .or_else(|| value.pointer("/payload/session_id").and_then(Value::as_str))
            .map(str::to_string)
            .or_else(|| self.id.take());

        for key in ["cwd", "originator", "cli_version", "model_provider"] {
            if let Some(value) = value
                .pointer(&format!("/payload/{key}"))
                .and_then(Value::as_str)
            {
                self.metadata.insert(key.to_string(), value.to_string());
            }
        }
    }

    fn record_response_item(&mut self, value: &Value) {
        if value.pointer("/payload/type").and_then(Value::as_str) != Some("message") {
            return;
        }
        if value.pointer("/payload/phase").and_then(Value::as_str) == Some("commentary") {
            return;
        }

        let role = value.pointer("/payload/role").and_then(Value::as_str);
        if role == Some("developer") {
            return;
        }

        for text in message_texts(value) {
            if self.title.is_none() && role == Some("user") {
                self.title = title_candidate(&text).map(str::to_string);
            }

            let text = strip_scaffolding_blocks(&text);
            if text.is_empty() {
                continue;
            }

            if !self.description.is_empty() {
                self.description.push('\n');
            }
            self.description.push_str(&text);
        }
    }

    fn finish(self) -> Option<Event> {
        if self.description.is_empty() && self.id.is_none() {
            return None;
        }

        let id = self.id.unwrap_or_else(|| fallback_id(&self.path));
        let mut event = Event::new(
            EventId::new(id),
            Source::Codex,
            self.title.unwrap_or_else(|| fallback_title(&self.path)),
        );
        event.description = self.description;
        event.metadata = self.metadata;
        event.timestamp = self.timestamp.map(Timestamp::new);

        Some(event)
    }
}

fn collect_session_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_session_files(&path, files)?;
        } else if path.extension() == Some(OsStr::new("jsonl")) {
            files.push(path);
        }
    }

    Ok(())
}

fn message_texts(value: &Value) -> Vec<String> {
    value
        .pointer("/payload/content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|content| {
            content
                .get("text")
                .and_then(Value::as_str)
                .or_else(|| content.get("input_text").and_then(Value::as_str))
                .or_else(|| content.get("output_text").and_then(Value::as_str))
        })
        .map(str::to_string)
        .collect()
}

fn strip_scaffolding_blocks(text: &str) -> String {
    let mut retained = Vec::new();
    let mut skipped_block_end = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if skipped_block_end.is_some_and(|end| trimmed != end) {
            continue;
        }
        if skipped_block_end.is_some_and(|end| trimmed == end) {
            skipped_block_end = None;
            continue;
        }
        if let Some(end) = scaffolding_block_end(trimmed) {
            skipped_block_end = Some(end);
            continue;
        }

        retained.push(line);
    }

    retained.join("\n").trim().to_string()
}

fn search_event(event: Event, query_lower: &str, terms: &[&str]) -> Option<SearchResult> {
    let score = score_event(&event.title, &event.description, query_lower, terms)?;
    let diagnostics = search_diagnostics(&event, query_lower, terms, &score);

    Some(SearchResult {
        event: EventRef::new(event.source, event.id),
        score: Some(score.total),
        snippet: snippet(&event.title, &event.description, query_lower, terms),
        metadata: event.metadata,
        diagnostics,
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

    Some(ScoreBreakdown {
        total: score,
        term_count_score,
        title_phrase_score,
        description_phrase_score,
        title_term_score,
        description_term_score,
    })
}

fn search_diagnostics(
    event: &Event,
    query_lower: &str,
    terms: &[&str],
    score: &ScoreBreakdown,
) -> Metadata {
    let title_lower = event.title.to_lowercase();
    let description_lower = event.description.to_lowercase();
    let matched_fields = matched_fields(&title_lower, &description_lower, query_lower, terms);
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
    diagnostics.insert("matched_fields".to_string(), matched_fields);
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

fn matched_fields(
    title_lower: &str,
    description_lower: &str,
    query_lower: &str,
    terms: &[&str],
) -> String {
    let mut fields = Vec::new();
    if title_lower.contains(query_lower) || terms.iter().any(|term| title_lower.contains(term)) {
        fields.push("title");
    }
    if description_lower.contains(query_lower)
        || terms.iter().any(|term| description_lower.contains(term))
    {
        fields.push("description");
    }
    fields.join(", ")
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

fn snippet(title: &str, description: &str, query_lower: &str, terms: &[&str]) -> String {
    for line in description.lines() {
        if line.to_lowercase().contains(query_lower) {
            return first_chars(line, 160);
        }
    }

    for line in description.lines() {
        let line_lower = line.to_lowercase();
        if terms.iter().any(|term| line_lower.contains(term)) {
            return first_chars(line, 160);
        }
    }

    first_chars(title, 160)
}

fn title_candidate(text: &str) -> Option<&str> {
    let mut skipped_block_end = None;

    for line in text.lines().map(str::trim) {
        if line.is_empty() {
            continue;
        }
        if skipped_block_end.is_some_and(|end| line != end) {
            continue;
        }
        if skipped_block_end.is_some_and(|end| line == end) {
            skipped_block_end = None;
            continue;
        }
        if let Some(end) = scaffolding_block_end(line) {
            skipped_block_end = Some(end);
            continue;
        }
        if is_wrapper_tag(line) {
            continue;
        }

        return Some(line);
    }

    None
}

fn scaffolding_block_end(line: &str) -> Option<&'static str> {
    match line {
        "<environment_context>" => Some("</environment_context>"),
        "<permissions instructions>" => Some("</permissions instructions>"),
        "<skills_instructions>" => Some("</skills_instructions>"),
        "<apps_instructions>" => Some("</apps_instructions>"),
        "<plugins_instructions>" => Some("</plugins_instructions>"),
        _ => None,
    }
}

fn is_wrapper_tag(line: &str) -> bool {
    line.starts_with('<') && line.ends_with('>') && !line.contains(char::is_whitespace)
}

fn first_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn fallback_id(path: &Path) -> String {
    path.file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("unknown-session")
        .to_string()
}

fn fallback_title(path: &Path) -> String {
    format!("Codex session {}", fallback_id(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn scan_maps_session_jsonl_to_event() {
        let sessions_dir = temp_sessions_dir("scan");
        write_session(
            &sessions_dir,
            "2026/07/17/session.jsonl",
            r#"{"timestamp":"2026-07-17T10:00:00Z","type":"session_meta","payload":{"id":"session-1","timestamp":"2026-07-17T09:59:00Z","cwd":"/repo","cli_version":"0.1.0"}}"#,
        );
        append_session(
            &sessions_dir,
            "2026/07/17/session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Find the registry milestone"}]}}"#,
        );
        append_session(
            &sessions_dir,
            "2026/07/17/session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"The registry is implemented."}]}}"#,
        );

        let events = CodexAdapter::with_sessions_dir(&sessions_dir)
            .scan()
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_str(), "session-1");
        assert_eq!(events[0].source, Source::Codex);
        assert_eq!(
            events[0].timestamp.as_ref().map(Timestamp::as_str),
            Some("2026-07-17T09:59:00Z")
        );
        assert_eq!(events[0].title, "Find the registry milestone");
        assert!(events[0].description.contains("registry is implemented"));
        assert_eq!(
            events[0].metadata.get("cwd").map(String::as_str),
            Some("/repo")
        );
    }

    #[test]
    fn scan_uses_first_task_line_after_environment_context_as_title_and_description() {
        let sessions_dir = temp_sessions_dir("environment-title");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-environment-title"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>\n  <cwd>/repo</cwd>\n  <shell>zsh</shell>\n</environment_context>\n\nImplement robust Codex title selection"}]}}"#,
        );

        let events = CodexAdapter::with_sessions_dir(&sessions_dir)
            .scan()
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].title, "Implement robust Codex title selection");
        assert_eq!(
            events[0].description,
            "Implement robust Codex title selection"
        );
    }

    #[test]
    fn scan_skips_multiple_scaffolding_blocks_before_title_and_description() {
        let sessions_dir = temp_sessions_dir("scaffolding-title");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-scaffolding-title"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"\n<permissions instructions>\nRead-only sandbox details\n</permissions instructions>\n<skills_instructions>\nAvailable skills\n</skills_instructions>\n\nReview Recall retrieval output"}]}}"#,
        );

        let events = CodexAdapter::with_sessions_dir(&sessions_dir)
            .scan()
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].title, "Review Recall retrieval output");
        assert_eq!(events[0].description, "Review Recall retrieval output");
    }

    #[test]
    fn search_matches_scanned_events() {
        let sessions_dir = temp_sessions_dir("search");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-2"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Search Codex transcripts"}]}}"#,
        );

        let adapter = CodexAdapter::with_sessions_dir(&sessions_dir);
        let results = adapter.search("transcripts").unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].event, EventRef::new(Source::Codex, "session-2"));
        assert!(results[0].score.is_some());
        assert!(results[0].snippet.contains("transcripts"));
        assert_eq!(
            results[0].diagnostics.get("adapter").map(String::as_str),
            Some("codex")
        );
        assert_eq!(
            results[0].diagnostics.get("source_id").map(String::as_str),
            Some("session-2")
        );
        assert_eq!(
            results[0]
                .diagnostics
                .get("matched_terms")
                .map(String::as_str),
            Some("transcripts")
        );
        assert!(results[0]
            .diagnostics
            .get("matched_fields")
            .is_some_and(|fields| fields.contains("description")));
        assert_eq!(
            results[0].diagnostics.get("score").map(String::as_str),
            results[0].score.map(|score| score.to_string()).as_deref()
        );
    }

    #[test]
    fn scan_skips_commentary_phase_messages() {
        let sessions_dir = temp_sessions_dir("commentary-phase");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-commentary"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Evaluate Codex commentary filtering"}]}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"I am checking files and running tests."}]}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"Commentary filtering is implemented."}]}}"#,
        );

        let events = CodexAdapter::with_sessions_dir(&sessions_dir)
            .scan()
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].title, "Evaluate Codex commentary filtering");
        assert!(events[0]
            .description
            .contains("Evaluate Codex commentary filtering"));
        assert!(events[0]
            .description
            .contains("Commentary filtering is implemented."));
        assert!(!events[0].description.contains("checking files"));
        assert!(!events[0].description.contains("running tests"));
    }

    #[test]
    fn scan_skips_developer_role_messages() {
        let sessions_dir = temp_sessions_dir("developer-role");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-developer"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"Hidden instruction payload about unrelated memory"}]}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Inspect substantive Codex content"}]}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"Substantive assistant content remains."}]}}"#,
        );

        let events = CodexAdapter::with_sessions_dir(&sessions_dir)
            .scan()
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].title, "Inspect substantive Codex content");
        assert!(events[0]
            .description
            .contains("Inspect substantive Codex content"));
        assert!(events[0]
            .description
            .contains("Substantive assistant content remains."));
        assert!(!events[0].description.contains("Hidden instruction payload"));
        assert!(!events[0].description.contains("unrelated memory"));
    }

    #[test]
    fn search_ignores_query_terms_only_present_in_developer_messages() {
        let sessions_dir = temp_sessions_dir("developer-search");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-developer-search"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"Needle only appears in developer instructions"}]}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Write normal session content"}]}}"#,
        );

        let results = CodexAdapter::with_sessions_dir(&sessions_dir)
            .search("needle developer instructions")
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn search_matches_non_contiguous_query_terms() {
        let sessions_dir = temp_sessions_dir("term-search");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-terms"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Implement Codex retrieval"}]}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"The adapter now reads session evidence."}]}}"#,
        );

        let results = CodexAdapter::with_sessions_dir(&sessions_dir)
            .search("implement codex adapter")
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].event,
            EventRef::new(Source::Codex, "session-terms")
        );
        assert!(results[0].score.unwrap() > 0);
    }

    #[test]
    fn timeline_returns_scanned_events() {
        let sessions_dir = temp_sessions_dir("timeline");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-6","timestamp":"2026-07-20T10:00:00Z"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Build timeline output"}]}}"#,
        );

        let timeline = CodexAdapter::with_sessions_dir(&sessions_dir)
            .timeline()
            .unwrap();

        assert_eq!(timeline.events.len(), 1);
        assert_eq!(timeline.events[0].id.as_str(), "session-6");
        assert_eq!(timeline.events[0].title, "Build timeline output");
    }

    #[test]
    fn inspect_returns_matching_scanned_event() {
        let sessions_dir = temp_sessions_dir("inspect");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-4","timestamp":"2026-07-18T12:00:00Z"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Inspect this Codex event"}]}}"#,
        );

        let event = CodexAdapter::with_sessions_dir(&sessions_dir)
            .inspect(&EventId::new("session-4"))
            .unwrap()
            .unwrap();

        assert_eq!(event.id.as_str(), "session-4");
        assert_eq!(event.source, Source::Codex);
        assert_eq!(event.title, "Inspect this Codex event");
    }

    #[test]
    fn inspect_returns_none_for_missing_event() {
        let sessions_dir = temp_sessions_dir("inspect-missing");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-5"}}"#,
        );

        let event = CodexAdapter::with_sessions_dir(&sessions_dir)
            .inspect(&EventId::new("missing"))
            .unwrap();

        assert!(event.is_none());
    }

    #[test]
    fn missing_sessions_dir_returns_error() {
        let sessions_dir = temp_sessions_dir("missing").join("does-not-exist");
        let error = CodexAdapter::with_sessions_dir(&sessions_dir)
            .scan()
            .unwrap_err();

        assert_eq!(error.source, Source::Codex);
        assert!(error.message.contains("does not exist"));
    }

    #[test]
    fn malformed_record_does_not_discard_later_valid_session_records() {
        let sessions_dir = temp_sessions_dir("malformed-record");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-malformed"}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Ingest valid records before malformed lines"}]}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"This assistant record is truncated"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"A replacement assistant record after the malformed line is retained."}]}}"#,
        );

        let events = CodexAdapter::with_sessions_dir(&sessions_dir)
            .scan()
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_str(), "session-malformed");
        assert_eq!(
            events[0].title,
            "Ingest valid records before malformed lines"
        );
        assert!(events[0].description.contains("valid records before"));
        assert!(events[0]
            .description
            .contains("replacement assistant record after the malformed line"));
        assert!(!events[0]
            .description
            .contains("This assistant record is truncated"));
        assert_eq!(
            events[0]
                .metadata
                .get("skipped_records")
                .map(String::as_str),
            Some("1")
        );
        assert!(events[0]
            .metadata
            .get("skipped_record_diagnostic")
            .is_some_and(|diagnostic| diagnostic.contains("session.jsonl line 3")));
    }

    #[test]
    fn skipped_record_diagnostic_retains_multiple_malformed_line_numbers() {
        let sessions_dir = temp_sessions_dir("multiple-malformed-records");
        write_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"session_meta","payload":{"id":"session-multiple-malformed"}}"#,
        );
        append_session(&sessions_dir, "session.jsonl", "not json");
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Continue after first malformed record"}]}}"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"second malformed"#,
        );
        append_session(
            &sessions_dir,
            "session.jsonl",
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Continue after second malformed record."}]}}"#,
        );

        let events = CodexAdapter::with_sessions_dir(&sessions_dir)
            .scan()
            .unwrap();
        let diagnostic = events[0].metadata.get("skipped_record_diagnostic").unwrap();

        assert_eq!(
            events[0]
                .metadata
                .get("skipped_records")
                .map(String::as_str),
            Some("2")
        );
        assert!(diagnostic.contains("session.jsonl line 2"));
        assert!(diagnostic.contains("session.jsonl line 4"));
        assert!(events[0].description.contains("Continue after first"));
        assert!(events[0].description.contains("Continue after second"));
    }

    fn temp_sessions_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("recall-codex-{label}-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_session(root: &Path, relative: &str, line: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = File::create(path).unwrap();
        writeln!(file, "{line}").unwrap();
    }

    fn append_session(root: &Path, relative: &str, line: &str) {
        let path = root.join(relative);
        let mut file = fs::OpenOptions::new().append(true).open(path).unwrap();
        writeln!(file, "{line}").unwrap();
    }
}
