//! Git source adapter.
//!
//! This adapter reads commit history from configured repositories through the
//! `git` executable. It intentionally does no indexing, caching, ranking, or
//! repository discovery beyond the configured working directories.

use std::path::PathBuf;
use std::process::Command;

use recall_core::{
    Adapter, AdapterError, AdapterResult, Event, EventId, EventRef, Metadata, SearchResult, Source,
    Timeline, Timestamp,
};

/// Adapter for the current Git repository.
#[derive(Clone, Debug)]
pub struct GitAdapter {
    repo_dirs: Vec<PathBuf>,
}

impl Default for GitAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GitAdapter {
    /// Creates a Git adapter pointed at the current working directory.
    pub fn new() -> Self {
        Self {
            repo_dirs: vec![PathBuf::from(".")],
        }
    }

    /// Creates a Git adapter pointed at an explicit repository directory.
    pub fn with_repo_dir(repo_dir: impl Into<PathBuf>) -> Self {
        Self {
            repo_dirs: vec![repo_dir.into()],
        }
    }

    /// Creates a Git adapter pointed at explicit repository directories.
    pub fn with_repo_dirs(repo_dirs: impl IntoIterator<Item = PathBuf>) -> Self {
        let repo_dirs = repo_dirs.into_iter().collect();
        Self { repo_dirs }
    }

    /// Reads commits from the configured repository.
    pub fn scan(&self) -> AdapterResult<Vec<Event>> {
        let mut events = Vec::new();
        for repo_dir in &self.repo_dirs {
            let output = match self.git(
                repo_dir,
                ["log", "--format=%H%x1f%cI%x1f%s%x1f%b%x1e", "--no-color"],
            ) {
                Ok(output) => output,
                Err(error) if is_unborn_head_error(&error.message) => continue,
                Err(error) => return Err(error),
            };
            events.extend(parse_events(&output, repo_dir));
        }

        Ok(events)
    }

    fn inspect_commit(&self, id: &EventId) -> AdapterResult<Event> {
        if id.as_str().trim().is_empty() {
            return Err(self.error("commit SHA is empty"));
        }

        let mut missing_errors = Vec::new();
        for repo_dir in &self.repo_dirs {
            let output = self.git(
                repo_dir,
                [
                    "show",
                    "--quiet",
                    "--format=%H%x1f%cI%x1f%s%x1f%b",
                    "--no-color",
                    id.as_str(),
                ],
            );
            match output {
                Ok(output) => {
                    return parse_event(output.trim_end_matches('\n'), repo_dir)
                        .ok_or_else(|| self.error(format!("commit not found: {}", id.as_str())));
                }
                Err(error) => missing_errors.push(error.message),
            }
        }

        let detail = missing_errors
            .into_iter()
            .next()
            .unwrap_or_else(|| format!("commit not found: {}", id.as_str()));
        Err(self.error(detail))
    }

    fn git<const N: usize>(&self, repo_dir: &PathBuf, args: [&str; N]) -> AdapterResult<String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(repo_dir)
            .output()
            .map_err(|error| self.error(format!("failed to run git: {error}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr.trim();
            let message = if detail.is_empty() {
                format!("git exited with status {}", output.status)
            } else {
                detail.to_string()
            };
            return Err(self.error(message));
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    fn error(&self, message: impl Into<String>) -> AdapterError {
        AdapterError::new(Source::Git, message)
    }
}

impl Adapter for GitAdapter {
    fn source(&self) -> Source {
        Source::Git
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
        let mut events = self.scan()?;
        events.sort_by(|left, right| right.timestamp.cmp(&left.timestamp));

        Ok(Timeline { events })
    }

    fn inspect(&self, id: &EventId) -> AdapterResult<Option<Event>> {
        Ok(Some(self.inspect_commit(id)?))
    }
}

fn parse_events(output: &str, repo_dir: &PathBuf) -> Vec<Event> {
    output
        .split('\x1e')
        .filter_map(|record| parse_event(record.trim_matches('\n'), repo_dir))
        .collect()
}

fn parse_event(record: &str, repo_dir: &PathBuf) -> Option<Event> {
    if record.is_empty() {
        return None;
    }

    let mut fields = record.splitn(4, '\x1f');
    let sha = fields.next()?;
    let timestamp = fields.next()?;
    let subject = fields.next()?;
    let body = fields.next().unwrap_or("");

    if sha.is_empty() {
        return None;
    }

    let mut event = Event::new(EventId::new(sha), Source::Git, subject);
    event.timestamp = (!timestamp.is_empty()).then(|| Timestamp::new(timestamp));
    event.description = body.trim_end().to_string();
    event.metadata = Metadata::from([
        ("repository".to_string(), repo_dir.display().to_string()),
        ("sha".to_string(), sha.to_string()),
    ]);

    Some(event)
}

fn is_unborn_head_error(message: &str) -> bool {
    message.contains("does not have any commits yet")
        || (message.contains("your current branch")
            && message.contains("does not have any commits"))
        || message.contains("Needed a single revision")
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

fn first_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn scan_maps_commits_to_events() {
        let repo = temp_repo("scan");
        commit(
            &repo,
            "first.txt",
            "first",
            "Initial commit",
            "Set up project.",
        );

        let events = GitAdapter::with_repo_dir(&repo).scan().unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, Source::Git);
        assert_eq!(events[0].title, "Initial commit");
        assert_eq!(events[0].description, "Set up project.");
        assert!(events[0].timestamp.is_some());
        assert_eq!(
            events[0].metadata.get("sha").map(String::as_str),
            Some(events[0].id.as_str())
        );
    }

    #[test]
    fn search_matches_subject_and_body() {
        let repo = temp_repo("search");
        let sha = commit(
            &repo,
            "timeline.txt",
            "timeline",
            "Implement timeline",
            "Wire Git into Recall search.",
        );

        let results = GitAdapter::with_repo_dir(&repo)
            .search("recall search")
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].event, EventRef::new(Source::Git, sha.clone()));
        assert!(results[0].score.is_some());
        assert_eq!(results[0].snippet, "Wire Git into Recall search.");
        assert_eq!(
            results[0].diagnostics.get("adapter").map(String::as_str),
            Some("git")
        );
        assert_eq!(
            results[0].diagnostics.get("source_id").map(String::as_str),
            Some(sha.as_str())
        );
        assert_eq!(
            results[0]
                .diagnostics
                .get("matched_terms")
                .map(String::as_str),
            Some("recall, search")
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
    fn search_matches_non_contiguous_query_terms() {
        let repo = temp_repo("term-search");
        let sha = commit(
            &repo,
            "adapter.txt",
            "adapter",
            "Implement current repository adapter",
            "Read Git history through the git executable.",
        );

        let results = GitAdapter::with_repo_dir(&repo)
            .search("implement git adapter")
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].event, EventRef::new(Source::Git, sha));
        assert!(results[0].score.unwrap() > 0);
    }

    #[test]
    fn timeline_returns_commits_newest_first() {
        let repo = temp_repo("timeline");
        commit_with_date(
            &repo,
            "old.txt",
            "old",
            "Older commit",
            "",
            "2026-07-19T10:00:00Z",
        );
        commit_with_date(
            &repo,
            "new.txt",
            "new",
            "Newer commit",
            "",
            "2026-07-20T10:00:00Z",
        );

        let timeline = GitAdapter::with_repo_dir(&repo).timeline().unwrap();

        let titles: Vec<_> = timeline
            .events
            .iter()
            .map(|event| event.title.as_str())
            .collect();
        assert_eq!(titles, vec!["Newer commit", "Older commit"]);
    }

    #[test]
    fn timeline_can_read_multiple_repositories() {
        let first_repo = temp_repo("timeline-multi-first");
        let second_repo = temp_repo("timeline-multi-second");
        let first_sha = commit_with_date(
            &first_repo,
            "first.txt",
            "first",
            "First repo commit",
            "",
            "2026-07-20T10:00:00Z",
        );
        let second_sha = commit_with_date(
            &second_repo,
            "second.txt",
            "second",
            "Second repo commit",
            "",
            "2026-07-20T11:00:00Z",
        );

        let timeline = GitAdapter::with_repo_dirs([first_repo.clone(), second_repo.clone()])
            .timeline()
            .unwrap();

        assert_eq!(
            timeline
                .events
                .iter()
                .map(|event| event.id.as_str())
                .collect::<Vec<_>>(),
            vec![second_sha.as_str(), first_sha.as_str()]
        );
        let first_repo = first_repo.display().to_string();
        let second_repo = second_repo.display().to_string();
        assert_eq!(
            timeline.events[0]
                .metadata
                .get("repository")
                .map(String::as_str),
            Some(second_repo.as_str())
        );
        assert_eq!(
            timeline.events[1]
                .metadata
                .get("repository")
                .map(String::as_str),
            Some(first_repo.as_str())
        );
    }

    #[test]
    fn timeline_skips_empty_repositories_without_aborting_other_repositories() {
        let committed_repo = temp_repo("timeline-committed");
        let empty_repo = temp_repo("timeline-empty");
        let sha = commit_with_date(
            &committed_repo,
            "work.txt",
            "work",
            "Implement multi-repo timeline retrieval",
            "The empty sibling repository should not abort Git timeline retrieval.",
            "2026-08-31T09:00:00Z",
        );

        let timeline = GitAdapter::with_repo_dirs([committed_repo.clone(), empty_repo])
            .timeline()
            .unwrap();

        assert_eq!(timeline.events.len(), 1);
        assert_eq!(timeline.events[0].id.as_str(), sha);
        let committed_repo = committed_repo.display().to_string();
        assert_eq!(
            timeline.events[0]
                .metadata
                .get("repository")
                .map(String::as_str),
            Some(committed_repo.as_str())
        );
    }

    #[test]
    fn timeline_does_not_suppress_non_history_git_failures() {
        let committed_repo = temp_repo("timeline-committed-before-failure");
        commit_with_date(
            &committed_repo,
            "work.txt",
            "work",
            "Implement multi-repo timeline retrieval",
            "",
            "2026-08-31T09:00:00Z",
        );
        let non_repo = std::env::temp_dir().join(format!(
            "recall-git-non-repo-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&non_repo).unwrap();

        let error = GitAdapter::with_repo_dirs([committed_repo, non_repo])
            .timeline()
            .unwrap_err();

        assert_eq!(error.source, Source::Git);
        assert!(!is_unborn_head_error(&error.message));
    }

    #[test]
    fn inspect_checks_each_configured_repository() {
        let first_repo = temp_repo("inspect-multi-first");
        let second_repo = temp_repo("inspect-multi-second");
        commit(&first_repo, "first.txt", "first", "First repo commit", "");
        let second_sha = commit(
            &second_repo,
            "second.txt",
            "second",
            "Second repo commit",
            "",
        );

        let event = GitAdapter::with_repo_dirs([first_repo, second_repo])
            .inspect(&EventId::new(second_sha))
            .unwrap()
            .unwrap();

        assert_eq!(event.title, "Second repo commit");
    }

    #[test]
    fn inspect_returns_complete_commit_event() {
        let repo = temp_repo("inspect");
        let sha = commit(
            &repo,
            "inspect.txt",
            "inspect",
            "Inspect commit",
            "Return the commit body.",
        );

        let event = GitAdapter::with_repo_dir(&repo)
            .inspect(&EventId::new(sha))
            .unwrap()
            .unwrap();

        assert_eq!(event.source, Source::Git);
        assert_eq!(event.title, "Inspect commit");
        assert_eq!(event.description, "Return the commit body.");
    }

    #[test]
    fn inspect_returns_error_for_missing_commit() {
        let repo = temp_repo("inspect-missing");
        commit(&repo, "one.txt", "one", "One commit", "");

        let error = GitAdapter::with_repo_dir(&repo)
            .inspect(&EventId::new("0000000000000000000000000000000000000000"))
            .unwrap_err();

        assert_eq!(error.source, Source::Git);
        assert!(error.message.contains("bad object") || error.message.contains("unknown"));
    }

    fn temp_repo(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let repo = std::env::temp_dir().join(format!("recall-git-{label}-{unique}"));
        fs::create_dir_all(&repo).unwrap();
        run_git(&repo, ["init"]);
        run_git(&repo, ["config", "user.name", "Recall Tests"]);
        run_git(&repo, ["config", "user.email", "recall@example.test"]);
        repo
    }

    fn commit(
        repo: &std::path::Path,
        file: &str,
        contents: &str,
        subject: &str,
        body: &str,
    ) -> String {
        commit_with_date(repo, file, contents, subject, body, "2026-07-20T12:00:00Z")
    }

    fn commit_with_date(
        repo: &std::path::Path,
        file: &str,
        contents: &str,
        subject: &str,
        body: &str,
        date: &str,
    ) -> String {
        fs::write(repo.join(file), contents).unwrap();
        run_git(repo, ["add", file]);

        let mut message = subject.to_string();
        if !body.is_empty() {
            message.push_str("\n\n");
            message.push_str(body);
        }

        let output = Command::new("git")
            .args(["commit", "--date", date, "-m", &message])
            .env("GIT_AUTHOR_DATE", date)
            .env("GIT_COMMITTER_DATE", date)
            .current_dir(repo)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(repo)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn run_git<const N: usize>(repo: &std::path::Path, args: [&str; N]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(repo)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
