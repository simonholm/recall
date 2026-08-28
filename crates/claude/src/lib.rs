//! Claude Code source adapter.
//!
//! This adapter reads local Claude Code project JSONL files and maps each
//! session file to one source-neutral Recall event. It intentionally ignores
//! `/export` text transcripts and reads only Claude's native session storage.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use recall_core::{
    Adapter, AdapterError, AdapterResult, Event, EventId, EventRef, Metadata, SearchResult, Source,
    Timeline, Timestamp,
};
use serde_json::Value;

/// Adapter for local Claude Code session JSONL files.
#[derive(Clone, Debug)]
pub struct ClaudeAdapter {
    projects_dir: PathBuf,
}

impl Default for ClaudeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeAdapter {
    /// Creates a Claude adapter pointed at `$HOME/.claude/projects`.
    pub fn new() -> Self {
        let projects_dir = env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".claude")
            .join("projects");

        Self { projects_dir }
    }

    /// Creates a Claude adapter pointed at an explicit projects directory.
    pub fn with_projects_dir(projects_dir: impl Into<PathBuf>) -> Self {
        Self {
            projects_dir: projects_dir.into(),
        }
    }

    /// Reads Claude session files and returns one event per readable session.
    pub fn scan(&self) -> AdapterResult<Vec<Event>> {
        if !self.projects_dir.exists() {
            return Ok(Vec::new());
        }

        if !self.projects_dir.is_dir() {
            return Err(self.error(format!(
                "projects path is not a directory: {}",
                self.projects_dir.display()
            )));
        }

        let mut files = Vec::new();
        collect_session_files(&self.projects_dir, &mut files).map_err(|error| {
            self.error(format!(
                "failed to read projects directory {}: {error}",
                self.projects_dir.display()
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
            let value = serde_json::from_str::<Value>(line).map_err(|error| {
                self.error(format!(
                    "failed to parse session file {} line {}: {error}",
                    path.display(),
                    line_index + 1
                ))
            })?;

            builder.record(&value);
        }

        Ok(builder.finish())
    }

    fn error(&self, message: impl Into<String>) -> AdapterError {
        AdapterError::new(claude_source(), message)
    }
}

impl Adapter for ClaudeAdapter {
    fn source(&self) -> Source {
        claude_source()
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
        let record_type = value.get("type").and_then(Value::as_str);
        if !matches!(record_type, Some("user" | "assistant")) {
            return;
        }

        let Some(role @ ("user" | "assistant")) =
            value.pointer("/message/role").and_then(Value::as_str)
        else {
            return;
        };

        let texts = message_texts(value);
        if texts.is_empty() {
            return;
        }

        self.id = value
            .get("sessionId")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.id.take());
        self.timestamp = value
            .get("timestamp")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.timestamp.take());

        for key in ["cwd", "entrypoint", "version", "gitBranch"] {
            if let Some(value) = value.get(key).and_then(Value::as_str) {
                self.metadata.insert(key.to_string(), value.to_string());
            }
        }

        for text in texts {
            if self.title.is_none() && role == "user" {
                self.title = title_candidate(&text).map(str::to_string);
            }

            if !self.description.is_empty() {
                self.description.push('\n');
            }
            self.description.push_str(role);
            self.description.push_str(": ");
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
            claude_source(),
            self.title.unwrap_or_else(|| fallback_title(&self.path)),
        );
        event.description = self.description;
        event.metadata = self.metadata;
        event.timestamp = self.timestamp.map(Timestamp::new);

        Some(event)
    }
}

fn claude_source() -> Source {
    Source::Other("claude".to_string())
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
    let Some(content) = value.pointer("/message/content") else {
        return Vec::new();
    };

    if let Some(text) = content.as_str() {
        let text = text.trim();
        return (!text.is_empty())
            .then(|| text.to_string())
            .into_iter()
            .collect();
    }

    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|content| {
            if content.get("type").and_then(Value::as_str) != Some("text") {
                return None;
            }
            content.get("text").and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
        .collect()
}

fn search_event(event: Event, query_lower: &str, terms: &[&str]) -> Option<SearchResult> {
    let score = score_event(&event.title, &event.description, query_lower, terms)?;

    Some(SearchResult {
        event: EventRef::new(event.source, event.id),
        score: Some(score),
        snippet: snippet(&event.title, &event.description, query_lower, terms),
        metadata: event.metadata,
        diagnostics: Metadata::new(),
    })
}

fn score_event(title: &str, description: &str, query_lower: &str, terms: &[&str]) -> Option<u32> {
    let title_lower = title.to_lowercase();
    let description_lower = description.to_lowercase();
    let haystack = format!("{title_lower}\n{description_lower}");

    if !terms.iter().all(|term| haystack.contains(term)) {
        return None;
    }

    let mut score = (terms.len() as u32) * 10;
    if title_lower.contains(query_lower) {
        score += 100;
    } else if description_lower.contains(query_lower) {
        score += 50;
    }

    for term in terms {
        if title_lower.contains(term) {
            score += 5;
        }
        if description_lower.contains(term) {
            score += 1;
        }
    }

    Some(score)
}

fn snippet(title: &str, description: &str, query_lower: &str, terms: &[&str]) -> String {
    if title.to_lowercase().contains(query_lower) {
        return title.to_string();
    }

    description
        .lines()
        .find(|line| {
            let line_lower = line.to_lowercase();
            line_lower.contains(query_lower) || terms.iter().any(|term| line_lower.contains(term))
        })
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .unwrap_or(title)
        .to_string()
}

fn title_candidate(text: &str) -> Option<&str> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(truncate_title)
}

fn truncate_title(line: &str) -> &str {
    const MAX_CHARS: usize = 120;

    if line.chars().count() <= MAX_CHARS {
        return line;
    }

    line.char_indices()
        .nth(MAX_CHARS)
        .map(|(index, _)| &line[..index])
        .unwrap_or(line)
}

fn fallback_id(path: &Path) -> String {
    path.file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("unknown")
        .to_string()
}

fn fallback_title(path: &Path) -> String {
    format!("Claude session {}", fallback_id(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn scan_parses_representative_claude_session_records() {
        let projects_dir = temp_projects_dir("scan");
        write_session(
            &projects_dir,
            "project/session-1.jsonl",
            [
                r#"{"type":"mode","mode":"normal","sessionId":"session-1"}"#,
                r#"{"type":"user","message":{"role":"user","content":"Investigate Claude /export behavior"},"uuid":"user-1","timestamp":"2026-08-28T13:13:40.822Z","cwd":"/repo","sessionId":"session-1","entrypoint":"cli","version":"2.1.250","gitBranch":"main"}"#,
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Claude Code does not support configuring the default /export directory."},{"type":"tool_use","name":"Bash","input":{"command":"ignored"}}]},"uuid":"assistant-1","timestamp":"2026-08-28T13:19:06.658Z","cwd":"/repo","sessionId":"session-1"}"#,
                r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"ignored command output"}]},"uuid":"tool-result","timestamp":"2026-08-28T13:20:00Z","cwd":"/repo","sessionId":"session-1"}"#,
            ]
            .join("\n")
                + "\n",
        );

        let events = ClaudeAdapter::with_projects_dir(&projects_dir)
            .scan()
            .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, Source::Other("claude".to_string()));
        assert_eq!(events[0].id.as_str(), "session-1");
        assert_eq!(events[0].title, "Investigate Claude /export behavior");
        assert_eq!(
            events[0].timestamp.as_ref().map(Timestamp::as_str),
            Some("2026-08-28T13:19:06.658Z")
        );
        assert_eq!(events[0].metadata.get("cwd"), Some(&"/repo".to_string()));
        assert!(events[0]
            .description
            .contains("user: Investigate Claude /export behavior"));
        assert!(events[0]
            .description
            .contains("assistant: Claude Code does not support configuring"));
        assert!(!events[0].description.contains("ignored command output"));
        assert!(!events[0].description.contains("tool_use"));
    }

    #[test]
    fn search_finds_claude_export_session() {
        let projects_dir = temp_projects_dir("search");
        write_session(
            &projects_dir,
            "project/session-2.jsonl",
            r#"{"type":"user","message":{"role":"user","content":"What did I learn today about Claude Code's /export command?"},"timestamp":"2026-08-28T14:00:00Z","sessionId":"session-2"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Bare /export saves to the current working directory unless a path is entered."}]},"timestamp":"2026-08-28T14:01:00Z","sessionId":"session-2"}
"#,
        );

        let results = ClaudeAdapter::with_projects_dir(&projects_dir)
            .search("Claude Code /export command")
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].event,
            EventRef::new(Source::Other("claude".to_string()), "session-2")
        );
    }

    #[test]
    fn missing_projects_dir_returns_empty_events() {
        let projects_dir = temp_projects_dir("missing").join("does-not-exist");
        let events = ClaudeAdapter::with_projects_dir(projects_dir)
            .scan()
            .unwrap();

        assert!(events.is_empty());
    }

    fn temp_projects_dir(label: &str) -> PathBuf {
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("recall-claude-{label}-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_session(projects_dir: &Path, relative_path: &str, content: impl AsRef<str>) {
        let path = projects_dir.join(relative_path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content.as_ref()).unwrap();
    }
}
