//! Command-line interface for Recall.
//!
//! Commands are intentionally present but unimplemented while the workspace is
//! still architecture scaffolding.

#[cfg(test)]
mod answer_diagnostic;
mod openrouter;
mod outbound_audit;

use clap::{Parser, Subcommand};
use openrouter::{AuthStatus, LlmDiagnostics, OpenRouterConfig};
use outbound_audit::OutboundAuditConfig;
use recall_codex::CodexAdapter;
use recall_core::{
    ContextCompiler, DateRange, Event, EventId, EventRef, EvidenceBlock, PromptBuilder, Recall,
    RetrievalPlan, RetrievalPlanner, SearchResult, Source, Timeline,
};
use recall_git::GitAdapter;
use std::fmt::Write;
use std::fs::File;
use std::io::{self, BufRead, Write as IoWrite};
use std::process::ExitCode;

const ASK_RESULT_LIMIT: usize = 8;

/// Local development memory CLI.
#[derive(Debug, Parser)]
#[command(name = "recall")]
#[command(about = "Local development memory system")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Recall command set.
#[derive(Debug, Subcommand)]
enum Command {
    /// Manage OpenRouter authentication.
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    /// Build an experimental answer prompt from retrieved memory.
    Ask {
        /// Append OpenRouter response diagnostics when available.
        #[arg(long)]
        diagnostics: bool,
        /// Print original and normalized query debug information.
        #[arg(long)]
        debug_query: bool,
        /// Print retrieved search results and retrieval diagnostics.
        #[arg(long)]
        debug_search: bool,
        /// Print the final prompt sent to OpenRouter before requesting an answer.
        #[arg(long)]
        debug_prompt: bool,
        /// Natural-language question to ask.
        question: Vec<String>,
    },
    /// Search development memory sources.
    Search {
        /// Text to search for.
        query: Vec<String>,
    },
    /// Show a development timeline.
    Timeline,
    /// Inspect a specific memory item.
    Inspect {
        /// Source-qualified event id, such as codex:<id>.
        event_id: String,
    },
}

#[derive(Debug, Subcommand)]
enum AuthCommand {
    /// Store an OpenRouter API key for future Recall runs.
    Login,
    /// Show whether OpenRouter authentication is configured.
    Status,
    /// Remove Recall's stored OpenRouter API key.
    Logout,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let recall = default_recall();
    match run(cli, &recall) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn default_recall() -> Recall {
    let mut recall = Recall::new();
    recall.register(CodexAdapter::new());
    recall.register(GitAdapter::new());
    recall
}

fn run(cli: Cli, recall: &Recall) -> Result<(), String> {
    match cli.command {
        Command::Auth { command } => match command {
            AuthCommand::Login => auth_login(),
            AuthCommand::Status => auth_status(),
            AuthCommand::Logout => auth_logout(),
        }?,
        Command::Ask {
            diagnostics,
            debug_query,
            debug_search,
            debug_prompt,
            question,
        } => {
            if question.is_empty() {
                return Err("ask requires a question".to_string());
            }

            let question = question.join(" ");
            print!(
                "{}",
                ask_output(
                    recall,
                    &question,
                    &OpenRouterConfig::from_env(),
                    diagnostics,
                    DebugOptions {
                        query: debug_query,
                        search: debug_search,
                        prompt: debug_prompt,
                    },
                )?
            );
        }
        Command::Search { query } => {
            if query.is_empty() {
                println!("Not implemented");
            } else {
                print_search_results(
                    &recall
                        .search(&query.join(" "))
                        .map_err(|error| error.to_string())?,
                );
            }
        }
        Command::Timeline => {
            print_timeline(&recall.timeline().map_err(|error| error.to_string())?);
        }
        Command::Inspect { event_id } => {
            let event_ref = parse_event_ref(&event_id)?;
            let event = recall
                .inspect(&event_ref)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| format!("event not found: {event_id}"))?;
            print_event(&event);
        }
    }

    Ok(())
}

fn auth_login() -> Result<(), String> {
    let api_key = read_secret("OpenRouter API key: ")?;
    let path = openrouter::save_api_key_from_env_home(&api_key)?;
    println!("OpenRouter credential stored at {}", path.display());
    Ok(())
}

fn auth_status() -> Result<(), String> {
    print!(
        "{}",
        format_auth_status(openrouter::auth_status_from_env())?
    );
    Ok(())
}

fn format_auth_status(status: AuthStatus) -> Result<String, String> {
    match status {
        AuthStatus::EnvironmentOverride => {
            Ok("OpenRouter authentication: configured from OPENROUTER_API_KEY\n".to_string())
        }
        AuthStatus::Stored { path } => {
            Ok(format!(
                "OpenRouter authentication: configured at {}",
                path.display()
            ))
            .map(|line| format!("{line}\n"))
        }
        AuthStatus::NotConfigured { path } => {
            Ok(format!(
                "OpenRouter authentication: not configured; run `recall auth login` to store a credential at {}",
                path.display()
            ))
            .map(|line| format!("{line}\n"))
        }
        AuthStatus::Error(error) => Err(error),
    }
}

fn auth_logout() -> Result<(), String> {
    match openrouter::delete_api_key_from_env_home()? {
        Some(path) => println!("OpenRouter credential removed from {}", path.display()),
        None => println!("OpenRouter credential was not stored by Recall"),
    }
    Ok(())
}

fn read_secret(prompt: &str) -> Result<String, String> {
    #[cfg(unix)]
    {
        read_secret_from_tty(prompt)
    }
    #[cfg(not(unix))]
    {
        print!("{prompt}");
        io::stdout()
            .flush()
            .map_err(|error| format!("failed to write prompt: {error}"))?;
        let mut secret = String::new();
        io::stdin()
            .read_line(&mut secret)
            .map_err(|error| format!("failed to read OpenRouter API key: {error}"))?;
        Ok(secret.trim().to_string())
    }
}

#[cfg(unix)]
fn read_secret_from_tty(prompt: &str) -> Result<String, String> {
    use std::process::{Command, Stdio};

    let mut tty = File::options()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .map_err(|error| format!("failed to open terminal: {error}"))?;
    tty.write_all(prompt.as_bytes())
        .and_then(|_| tty.flush())
        .map_err(|error| format!("failed to write prompt: {error}"))?;

    let tty_for_stty = tty
        .try_clone()
        .map_err(|error| format!("failed to configure terminal: {error}"))?;
    let status = Command::new("stty")
        .arg("-echo")
        .stdin(Stdio::from(tty_for_stty))
        .status()
        .map_err(|error| format!("failed to disable terminal echo: {error}"))?;
    if !status.success() {
        return Err("failed to disable terminal echo".to_string());
    }

    let mut secret = String::new();
    let read_result = io::BufReader::new(
        tty.try_clone()
            .map_err(|error| format!("failed to read terminal: {error}"))?,
    )
    .read_line(&mut secret);

    let tty_for_stty = tty
        .try_clone()
        .map_err(|error| format!("failed to restore terminal echo: {error}"))?;
    let restore_status = Command::new("stty")
        .arg("echo")
        .stdin(Stdio::from(tty_for_stty))
        .status()
        .map_err(|error| format!("failed to restore terminal echo: {error}"))?;
    writeln!(tty).map_err(|error| format!("failed to write prompt: {error}"))?;
    if !restore_status.success() {
        return Err("failed to restore terminal echo".to_string());
    }

    read_result.map_err(|error| format!("failed to read OpenRouter API key: {error}"))?;
    Ok(secret.trim().to_string())
}

#[derive(Clone, Copy, Debug, Default)]
struct DebugOptions {
    query: bool,
    search: bool,
    prompt: bool,
}

fn ask_output(
    recall: &Recall,
    question: &str,
    openrouter_config: &OpenRouterConfig,
    include_diagnostics: bool,
    debug_options: DebugOptions,
) -> Result<String, String> {
    ask_output_with_audit(
        recall,
        question,
        openrouter_config,
        include_diagnostics,
        debug_options,
        OutboundAuditConfig::from_env(),
    )
}

fn ask_output_with_audit(
    recall: &Recall,
    question: &str,
    openrouter_config: &OpenRouterConfig,
    include_diagnostics: bool,
    debug_options: DebugOptions,
    audit_config: Option<OutboundAuditConfig>,
) -> Result<String, String> {
    let plan = RetrievalPlanner::new().plan(question);
    let retrieval = ask_retrieval(recall, &plan)?;
    let evidence = compile_evidence(&plan, question, &retrieval.events);
    let prompt = PromptBuilder::new().build(question, &evidence);
    let configuration = format_configuration_output(openrouter_config);

    eprint!(
        "{}",
        format_debug_output(
            debug_options,
            question,
            &plan,
            &retrieval.search_results,
            &retrieval.events,
            &prompt
        )
    );

    if let Some(error) = openrouter_config.credential_error() {
        return Err(error.to_string());
    }

    if !openrouter_config.is_configured() {
        return Ok(format!(
            "{configuration}\n{}",
            format_prompt_only_output(&format_retrieval_plan(&plan), &prompt)
        ));
    }

    let answer = send_configured_prompt_with_audit(
        openrouter_config,
        question,
        &prompt,
        include_diagnostics,
        audit_config,
    )?;
    let mut answer_output =
        format_answer_output(&format_retrieval_plan(&plan), &evidence, &answer.answer);
    if include_diagnostics {
        append_diagnostics_output(&mut answer_output, answer.diagnostics.as_ref());
    }

    Ok(format!("{configuration}\n{answer_output}"))
}

fn send_configured_prompt_with_audit(
    openrouter_config: &OpenRouterConfig,
    question: &str,
    prompt: &str,
    include_diagnostics: bool,
    audit_config: Option<OutboundAuditConfig>,
) -> Result<openrouter::LlmResponse, String> {
    send_configured_prompt_with_audit_and_sender(
        openrouter_config,
        question,
        prompt,
        include_diagnostics,
        audit_config,
        |config, prompt, include_diagnostics| {
            openrouter::send_prompt(config, prompt, include_diagnostics)
                .map_err(|error| error.to_string())
        },
    )
}

fn send_configured_prompt_with_audit_and_sender(
    openrouter_config: &OpenRouterConfig,
    question: &str,
    prompt: &str,
    include_diagnostics: bool,
    audit_config: Option<OutboundAuditConfig>,
    sender: impl FnOnce(&OpenRouterConfig, &str, bool) -> Result<openrouter::LlmResponse, String>,
) -> Result<openrouter::LlmResponse, String> {
    if let Some(audit_config) = audit_config {
        outbound_audit::write_outbound_prompt(
            &audit_config,
            openrouter_config.model(),
            question,
            prompt,
        )
        .map_err(|error| error.to_string())?;
    }

    sender(openrouter_config, prompt, include_diagnostics)
}

fn format_debug_output(
    debug_options: DebugOptions,
    question: &str,
    plan: &RetrievalPlan,
    search_results: &[SearchResult],
    events: &[Event],
    prompt: &str,
) -> String {
    let mut output = String::new();
    if debug_options.query {
        output.push_str(&format_debug_query(question, plan));
    }
    if debug_options.search {
        output.push_str(&format_debug_retrieval(plan, search_results, events));
    }
    if debug_options.prompt {
        output.push_str(prompt);
        if !prompt.ends_with('\n') {
            output.push('\n');
        }
    }

    output
}

fn format_debug_query(question: &str, plan: &RetrievalPlan) -> String {
    format!(
        "Original query:\n{question}\n\nRetrieval plan:\n{}\n",
        format_retrieval_plan(plan)
    )
}

fn format_debug_retrieval(
    plan: &RetrievalPlan,
    search_results: &[SearchResult],
    events: &[Event],
) -> String {
    match plan {
        RetrievalPlan::Search { .. } => format_debug_search(search_results),
        RetrievalPlan::Timeline { .. } => format_debug_timeline(events),
    }
}

fn format_debug_search(results: &[SearchResult]) -> String {
    let mut output = String::new();
    writeln!(output, "Search results:").unwrap();
    if results.is_empty() {
        writeln!(output, "(none)").unwrap();
        return output;
    }

    for result in results {
        writeln!(output, "- source: {}", result.event.source.as_str()).unwrap();
        writeln!(output, "  identifier: {}", result.event.id.as_str()).unwrap();
        match result.score {
            Some(score) => writeln!(output, "  score: {score}").unwrap(),
            None => writeln!(output, "  score: none").unwrap(),
        }
        writeln!(output, "  snippet: {}", result.snippet).unwrap();
        if !result.diagnostics.is_empty() {
            writeln!(output, "  diagnostics:").unwrap();
            for (key, value) in &result.diagnostics {
                writeln!(output, "    {key}: {value}").unwrap();
            }
        }
    }

    output
}

fn format_debug_timeline(events: &[Event]) -> String {
    let mut output = String::new();
    writeln!(output, "Timeline results:").unwrap();
    if events.is_empty() {
        writeln!(output, "(none)").unwrap();
        return output;
    }

    for event in events {
        writeln!(output, "- source: {}", event.source.as_str()).unwrap();
        writeln!(output, "  identifier: {}", event.id.as_str()).unwrap();
        match &event.timestamp {
            Some(timestamp) => writeln!(output, "  timestamp: {}", timestamp.as_str()).unwrap(),
            None => writeln!(output, "  timestamp: none").unwrap(),
        }
        writeln!(output, "  title: {}", event.title).unwrap();
    }

    output
}

fn format_configuration_output(openrouter_config: &OpenRouterConfig) -> String {
    let api_key = if openrouter_config.is_configured() {
        "yes"
    } else {
        "no"
    };

    let mut output = String::new();
    writeln!(output, "Configuration:").unwrap();
    writeln!(output, "  Model: {}", openrouter_config.model()).unwrap();
    if !openrouter_config.uses_default_endpoint() {
        writeln!(
            output,
            "  OpenRouter base URL: {}",
            openrouter_config.endpoint()
        )
        .unwrap();
    }
    writeln!(output, "  API key: {api_key}").unwrap();

    output
}

fn format_prompt_only_output(search_query: &str, prompt: &str) -> String {
    format!("Search query:\n{search_query}\n\n{prompt}")
}

fn format_answer_output(search_query: &str, evidence: &[EvidenceBlock], answer: &str) -> String {
    let mut output = String::new();
    writeln!(output, "Search query:").unwrap();
    writeln!(output, "{search_query}").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "Evidence:").unwrap();
    if evidence.is_empty() {
        writeln!(output, "(none)").unwrap();
    } else {
        for block in evidence {
            writeln!(
                output,
                "- {}:{} {}",
                block.source.as_str(),
                block.id.as_str(),
                block.title
            )
            .unwrap();
        }
    }
    writeln!(output).unwrap();
    writeln!(output, "Answer:").unwrap();
    write!(output, "{answer}").unwrap();
    if !answer.ends_with('\n') {
        writeln!(output).unwrap();
    }

    output
}

fn append_diagnostics_output(output: &mut String, diagnostics: Option<&LlmDiagnostics>) {
    let Some(diagnostics) = diagnostics else {
        return;
    };

    let mut diagnostics_output = String::new();
    if let Some(model) = &diagnostics.model {
        writeln!(diagnostics_output, "  Model: {model}").unwrap();
    }
    if let Some(provider) = &diagnostics.provider {
        writeln!(diagnostics_output, "  Provider: {provider}").unwrap();
    }
    let mut estimated_cost = None;
    if let Some(usage) = &diagnostics.usage {
        if let Some(prompt_tokens) = usage.prompt_tokens {
            writeln!(diagnostics_output, "  Prompt tokens: {prompt_tokens}").unwrap();
        }
        if let Some(completion_tokens) = usage.completion_tokens {
            writeln!(
                diagnostics_output,
                "  Completion tokens: {completion_tokens}"
            )
            .unwrap();
        }
        if let Some(cached_prompt_tokens) = usage.cached_prompt_tokens {
            writeln!(
                diagnostics_output,
                "  Cached prompt tokens: {cached_prompt_tokens}"
            )
            .unwrap();
        }
        if let Some(reasoning_tokens) = usage.reasoning_tokens {
            writeln!(diagnostics_output, "  Reasoning tokens: {reasoning_tokens}").unwrap();
        }
        if let Some(total_tokens) = usage.total_tokens {
            writeln!(diagnostics_output, "  Total tokens: {total_tokens}").unwrap();
        }
        estimated_cost = usage.estimated_cost;
    }
    if let Some(latency_ms) = diagnostics.latency_ms {
        writeln!(diagnostics_output, "  Latency: {latency_ms} ms").unwrap();
    }
    if let Some(transport) = &diagnostics.transport {
        writeln!(
            diagnostics_output,
            "  Request creation: {} ms",
            transport.request_creation_ms
        )
        .unwrap();
        writeln!(
            diagnostics_output,
            "  Upload to response headers: {} ms",
            transport.upload_to_headers_ms
        )
        .unwrap();
        match transport.first_body_byte_ms {
            Some(first_body_byte_ms) => {
                writeln!(
                    diagnostics_output,
                    "  First body byte: {first_body_byte_ms} ms"
                )
                .unwrap();
            }
            None => writeln!(diagnostics_output, "  First body byte: none").unwrap(),
        }
        writeln!(
            diagnostics_output,
            "  Body completion: {} ms",
            transport.body_completion_ms
        )
        .unwrap();
        writeln!(
            diagnostics_output,
            "  Total request: {} ms",
            transport.total_request_ms
        )
        .unwrap();
        writeln!(
            diagnostics_output,
            "  Response body bytes: {}",
            transport.response_body_bytes
        )
        .unwrap();
    }
    if let Some(estimated_cost) = estimated_cost {
        writeln!(diagnostics_output, "  Cost: {estimated_cost}").unwrap();
    }

    if !diagnostics_output.is_empty() {
        writeln!(output).unwrap();
        writeln!(output, "Diagnostics:").unwrap();
        output.push_str(&diagnostics_output);
    }
}

fn ask_search_results(recall: &Recall, search_query: &str) -> Result<Vec<SearchResult>, String> {
    Ok(recall
        .search(search_query)
        .map_err(|error| error.to_string())?
        .into_iter()
        .take(ASK_RESULT_LIMIT)
        .collect())
}

struct AskRetrieval {
    events: Vec<Event>,
    search_results: Vec<SearchResult>,
}

fn ask_retrieval(recall: &Recall, plan: &RetrievalPlan) -> Result<AskRetrieval, String> {
    match plan {
        RetrievalPlan::Search { query } => {
            let search_results = ask_search_results(recall, query)?;
            let events = inspect_search_results(recall, &search_results)?;
            Ok(AskRetrieval {
                events,
                search_results,
            })
        }
        RetrievalPlan::Timeline { range } => Ok(AskRetrieval {
            events: ask_timeline_events(recall, range)?,
            search_results: Vec::new(),
        }),
    }
}

fn compile_evidence(plan: &RetrievalPlan, question: &str, events: &[Event]) -> Vec<EvidenceBlock> {
    let compiler = ContextCompiler::new();
    match plan {
        RetrievalPlan::Search { .. } => compiler.compile(question, events),
        RetrievalPlan::Timeline { .. } => compiler.compile_timeline(question, events),
    }
}

fn ask_timeline_events(recall: &Recall, range: &DateRange) -> Result<Vec<Event>, String> {
    let events: Vec<_> = recall
        .timeline()
        .map_err(|error| error.to_string())?
        .events
        .into_iter()
        .filter(|event| {
            event
                .timestamp
                .as_ref()
                .is_some_and(|timestamp| range.contains_timestamp(timestamp))
        })
        .collect();

    match range {
        DateRange::Day(_) => Ok(events),
        _ => Ok(events.into_iter().take(ASK_RESULT_LIMIT).collect()),
    }
}

#[cfg(test)]
fn normalize_ask_query(question: &str) -> String {
    match RetrievalPlanner::new().plan(question) {
        RetrievalPlan::Search { query } => query,
        RetrievalPlan::Timeline { range } => format_date_range(&range),
    }
}

fn format_retrieval_plan(plan: &RetrievalPlan) -> String {
    match plan {
        RetrievalPlan::Search { query } => query.clone(),
        RetrievalPlan::Timeline { range } => format!("timeline {}", format_date_range(range)),
    }
}

fn format_date_range(range: &DateRange) -> String {
    match range {
        DateRange::Day(date) => date.to_string(),
        DateRange::Today => "today".to_string(),
        DateRange::Yesterday => "yesterday".to_string(),
        DateRange::LastWeek => "last week".to_string(),
        DateRange::LastDays(days) => format!("last {days} days"),
    }
}

fn inspect_search_results(
    recall: &Recall,
    search_results: &[SearchResult],
) -> Result<Vec<Event>, String> {
    search_results
        .iter()
        .map(|result| inspect_result_event(recall, &result.event))
        .collect()
}

#[cfg(test)]
fn ask_events(recall: &Recall, search_query: &str) -> Result<Vec<Event>, String> {
    recall
        .search(search_query)
        .map_err(|error| error.to_string())?
        .into_iter()
        .take(ASK_RESULT_LIMIT)
        .map(|result| inspect_result_event(recall, &result.event))
        .collect()
}

fn inspect_result_event(recall: &Recall, event_ref: &EventRef) -> Result<Event, String> {
    recall
        .inspect(event_ref)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("event not found: {}", format_event_ref(event_ref)))
}

fn format_event_ref(event_ref: &EventRef) -> String {
    format!("{}:{}", event_ref.source.as_str(), event_ref.id.as_str())
}

fn parse_event_ref(value: &str) -> Result<EventRef, String> {
    let Some((source, id)) = value.split_once(':') else {
        return Err(format!(
            "invalid event id: {value}; expected source:id, such as codex:<id>"
        ));
    };

    if source.is_empty() || id.is_empty() {
        return Err(format!(
            "invalid event id: {value}; expected source:id, such as codex:<id>"
        ));
    }

    let source = match source {
        "codex" => Source::Codex,
        "git" => Source::Git,
        other => Source::Other(other.to_string()),
    };

    Ok(EventRef::new(source, EventId::new(id)))
}

fn print_search_results(results: &[SearchResult]) {
    if results.is_empty() {
        println!("No results");
        return;
    }

    for result in results {
        println!(
            "{}:{} {}",
            result.event.source.as_str(),
            result.event.id.as_str(),
            result.snippet
        );
    }
}

fn print_timeline(timeline: &Timeline) {
    print!("{}", format_timeline(timeline));
}

fn format_timeline(timeline: &Timeline) -> String {
    if timeline.events.is_empty() {
        return "No events\n".to_string();
    }

    let mut output = String::new();
    for event in &timeline.events {
        writeln!(
            output,
            "{:<10}  {:<5}  {}",
            format_timeline_timestamp(event),
            event.source.as_str(),
            event.title
        )
        .unwrap();
    }

    output
}

fn format_timeline_timestamp(event: &Event) -> &str {
    event
        .timestamp
        .as_ref()
        .map(|timestamp| {
            timestamp
                .as_str()
                .split('T')
                .next()
                .unwrap_or(timestamp.as_str())
        })
        .unwrap_or("")
}

fn print_event(event: &Event) {
    print!("{}", format_event(event));
}

fn format_event(event: &Event) -> String {
    let mut output = String::new();
    writeln!(output, "Source: {}", event.source.as_str()).unwrap();
    writeln!(output, "ID: {}", event.id.as_str()).unwrap();
    writeln!(output, "Title: {}", event.title).unwrap();

    if let Some(timestamp) = &event.timestamp {
        writeln!(output, "Timestamp: {}", timestamp.as_str()).unwrap();
    } else {
        writeln!(output, "Timestamp:").unwrap();
    }

    writeln!(output).unwrap();
    writeln!(output, "Description:").unwrap();
    if event.description.is_empty() {
        writeln!(output).unwrap();
    } else {
        writeln!(output, "{}", event.description).unwrap();
    }

    writeln!(output).unwrap();
    writeln!(output, "Metadata:").unwrap();
    if event.metadata.is_empty() {
        writeln!(output, "(none)").unwrap();
    } else {
        for (key, value) in &event.metadata {
            writeln!(output, "{key}: {value}").unwrap();
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use recall_core::{Adapter, AdapterResult, Metadata, Timestamp};
    use std::cell::Cell;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug)]
    struct TestAdapter {
        events: Vec<Event>,
    }

    impl TestAdapter {
        fn new(events: Vec<Event>) -> Self {
            Self { events }
        }
    }

    impl Adapter for TestAdapter {
        fn source(&self) -> Source {
            Source::Other("test".to_string())
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

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn ask_debug_prompt_flag_defaults_to_false() {
        let cli = Cli::try_parse_from(["recall", "ask", "What happened?"]).unwrap();

        let Command::Ask {
            debug_query,
            debug_search,
            debug_prompt,
            ..
        } = cli.command
        else {
            panic!("expected ask command");
        };
        assert!(!debug_query);
        assert!(!debug_search);
        assert!(!debug_prompt);
    }

    #[test]
    fn ask_debug_flags_are_accepted() {
        let cli = Cli::try_parse_from([
            "recall",
            "ask",
            "--debug-query",
            "--debug-search",
            "--debug-prompt",
            "What happened?",
        ])
        .unwrap();

        let Command::Ask {
            debug_query,
            debug_search,
            debug_prompt,
            ..
        } = cli.command
        else {
            panic!("expected ask command");
        };
        assert!(debug_query);
        assert!(debug_search);
        assert!(debug_prompt);
    }

    #[test]
    fn auth_subcommands_are_accepted() {
        for subcommand in ["login", "status", "logout"] {
            let cli = Cli::try_parse_from(["recall", "auth", subcommand]).unwrap();
            let Command::Auth { command } = cli.command else {
                panic!("expected auth command");
            };
            match (subcommand, command) {
                ("login", AuthCommand::Login)
                | ("status", AuthCommand::Status)
                | ("logout", AuthCommand::Logout) => {}
                _ => panic!("unexpected auth subcommand"),
            }
        }
    }

    #[test]
    fn auth_status_output_does_not_expose_secret_values() {
        let output = format_auth_status(AuthStatus::Stored {
            path: "/tmp/recall-auth-test/auth.json".into(),
        })
        .unwrap();

        assert_eq!(
            output,
            "OpenRouter authentication: configured at /tmp/recall-auth-test/auth.json\n"
        );
        assert!(!output.contains("secret-key-value"));
        assert!(!output.contains("Bearer"));
    }

    #[test]
    fn default_registry_registers_current_adapters() {
        let recall = default_recall();

        assert_eq!(recall.adapter_count(), 2);
    }

    #[test]
    fn print_search_results_accepts_empty_results() {
        print_search_results(&[]);
    }

    #[test]
    fn normalize_ask_query_removes_question_words_and_punctuation() {
        assert_eq!(
            normalize_ask_query("When did I implement timeline?"),
            "implement timeline"
        );
    }

    #[test]
    fn normalize_ask_query_treats_punctuation_as_word_boundaries() {
        assert_eq!(
            normalize_ask_query("Atuin/zsh keybinding"),
            "atuin zsh keybinding"
        );
        assert_eq!(
            normalize_ask_query("Atuin/Mosh scrolling"),
            "atuin mosh scrolling"
        );
        assert_eq!(
            normalize_ask_query("Mosh/tmux scrolling"),
            "mosh tmux scrolling"
        );
        assert_eq!(normalize_ask_query("Atuin up-arrow"), "atuin up arrow");
    }

    #[test]
    fn normalize_ask_query_preserves_technical_terms_as_lowercase_words() {
        assert_eq!(
            normalize_ask_query("Why did I introduce EventRef?"),
            "introduce eventref"
        );
    }

    #[test]
    fn normalize_ask_query_keeps_meaningful_non_stop_words() {
        assert_eq!(
            normalize_ask_query("How was inspect implemented?"),
            "inspect implemented"
        );
    }

    #[test]
    fn normalize_ask_query_removes_control_and_temporal_words() {
        assert_eq!(
            normalize_ask_query("How should recall inspect behave when an event id is missing?"),
            "recall inspect event id missing"
        );
        assert_eq!(
            normalize_ask_query("Summarize today's recall work."),
            "today"
        );
    }

    #[test]
    fn normalize_ask_query_falls_back_before_returning_empty_query() {
        assert_eq!(normalize_ask_query("What did I work on today?"), "today");
        assert!(!normalize_ask_query("What did I work on today?").is_empty());
    }

    #[test]
    fn ask_events_fetches_full_events_from_search_results() {
        let mut recall = Recall::new();
        let mut event = Event::new(
            "event-1",
            Source::Other("test".to_string()),
            "ask milestone",
        );
        event.description = "Detailed event body".to_string();
        recall.register(TestAdapter::new(vec![event]));

        let events = ask_events(&recall, "ask").unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].title, "ask milestone");
        assert_eq!(events[0].description, "Detailed event body");
    }

    #[test]
    fn ask_events_limits_context_to_eight_events() {
        let mut recall = Recall::new();
        let events: Vec<_> = (0..9)
            .map(|index| {
                Event::new(
                    format!("event-{index}"),
                    Source::Other("test".to_string()),
                    "Ask milestone",
                )
            })
            .collect();
        recall.register(TestAdapter::new(events));

        let events = ask_events(&recall, "Ask").unwrap();

        assert_eq!(events.len(), ASK_RESULT_LIMIT);
    }

    #[test]
    fn ask_timeline_events_filters_by_date_range() {
        let mut recall = Recall::new();
        let mut included = Event::new("included", Source::Other("test".to_string()), "Included");
        included.timestamp = Some(Timestamp::new("2026-08-04T12:00:00Z"));
        let mut excluded = Event::new("excluded", Source::Other("test".to_string()), "Excluded");
        excluded.timestamp = Some(Timestamp::new("1900-01-01T12:00:00Z"));
        recall.register(TestAdapter::new(vec![included, excluded]));

        let events = ask_timeline_events(&recall, &DateRange::LastDays(30_000)).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_str(), "included");
    }

    #[test]
    fn ask_single_day_timeline_keeps_all_matching_mixed_source_events() {
        let mut recall = Recall::new();
        let sources = [
            Source::Git,
            Source::Codex,
            Source::Codex,
            Source::Codex,
            Source::Git,
            Source::Codex,
            Source::Git,
            Source::Git,
            Source::Git,
            Source::Git,
        ];
        let events: Vec<_> = sources
            .into_iter()
            .enumerate()
            .map(|(index, source)| {
                let mut event =
                    Event::new(format!("event-{index}"), source, format!("Event {index}"));
                event.timestamp = Some(Timestamp::new(format!(
                    "2026-08-05T12:{:02}:00Z",
                    59 - index
                )));
                event
            })
            .collect();
        recall.register(TestAdapter::new(events));

        let plan = RetrievalPlanner::new().plan("What did I work on in Recall on August 5, 2026?");
        let RetrievalPlan::Timeline { range } = plan else {
            panic!("expected explicit date to use timeline retrieval");
        };
        let events = ask_timeline_events(&recall, &range).unwrap();

        assert_eq!(events.len(), 10);
        assert_eq!(events[8].source, Source::Git);
        assert_eq!(events[8].id.as_str(), "event-8");
        assert_eq!(events[9].source, Source::Git);
        assert_eq!(events[9].id.as_str(), "event-9");
    }

    #[test]
    fn timeline_evidence_keeps_same_project_codex_sessions_distinct() {
        let mut first = Event::new("session-1", Source::Codex, "Add WebMCP experiment");
        first.timestamp = Some(Timestamp::new("2026-08-26T12:34:39Z"));
        first
            .metadata
            .insert("cwd".to_string(), "/repo/site".to_string());
        first.description = "Implemented minimal WebMCP page.".to_string();
        let mut second = Event::new("session-2", Source::Codex, "Prepare WebMCP for Pages");
        second.timestamp = Some(Timestamp::new("2026-08-26T12:43:53Z"));
        second
            .metadata
            .insert("cwd".to_string(), "/repo/site".to_string());
        second.description = "Implemented GitHub Pages preparation.".to_string();
        let plan = RetrievalPlan::Timeline {
            range: DateRange::Day(chrono::NaiveDate::from_ymd_opt(2026, 8, 26).unwrap()),
        };
        let events = vec![first, second];

        let evidence = compile_evidence(&plan, "What did I work on today?", &events);
        let prompt = PromptBuilder::new().build("What did I work on today?", &evidence);
        let displayed = format_answer_output("timeline today", &evidence, "answer");

        assert_eq!(evidence.len(), 2);
        assert_eq!(evidence[0].id.as_str(), "session-1");
        assert_eq!(evidence[1].id.as_str(), "session-2");
        assert!(evidence[0].body.contains("cwd: /repo/site"));
        assert!(evidence[0]
            .body
            .contains("Implemented minimal WebMCP page."));
        assert!(evidence[1]
            .body
            .contains("Implemented GitHub Pages preparation."));
        assert!(prompt.contains("Id: session-1"));
        assert!(prompt.contains("Id: session-2"));
        assert!(displayed.contains("- codex:session-1 Add WebMCP experiment"));
        assert!(displayed.contains("- codex:session-2 Prepare WebMCP for Pages"));
    }

    #[test]
    fn search_evidence_still_uses_project_state_compiler() {
        let mut first = Event::new("session-1", Source::Codex, "Compiler milestone");
        first
            .metadata
            .insert("cwd".to_string(), "/repo/recall".to_string());
        first.description = "Implemented deterministic deduplication.".to_string();
        let mut second = Event::new("session-2", Source::Codex, "Compiler follow-up");
        second
            .metadata
            .insert("cwd".to_string(), "/repo/recall".to_string());
        second.description = "Next step: keep ProjectState for search.".to_string();
        let plan = RetrievalPlan::Search {
            query: "compiler".to_string(),
        };

        let evidence = compile_evidence(&plan, "What changed?", &[first, second]);

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id.as_str(), "project:/repo/recall");
        assert_eq!(evidence[0].title, "Project: /repo/recall");
    }

    #[test]
    fn ask_output_without_api_key_prints_configuration_then_prompt() {
        let mut recall = Recall::new();
        let mut event = Event::new(
            "event-1",
            Source::Other("test".to_string()),
            "ask milestone",
        );
        event.description = "Detailed event body".to_string();
        recall.register(TestAdapter::new(vec![event]));

        let output = ask_output(
            &recall,
            "When did I ask?",
            &OpenRouterConfig::without_api_key_for_tests(),
            false,
            DebugOptions::default(),
        )
        .unwrap();
        let events = ask_events(&recall, "ask").unwrap();
        let evidence = ContextCompiler::new().compile("When did I ask?", &events);
        let prompt = PromptBuilder::new().build("When did I ask?", &evidence);

        assert_eq!(
            output,
            format!(
                "Configuration:\n  Model: deepseek/deepseek-v4-flash-0731\n  API key: no\n\nSearch query:\nask\n\n{prompt}"
            )
        );
    }

    #[test]
    fn configured_send_with_disabled_audit_creates_no_audit_file() {
        let log_dir = temp_log_dir("disabled");
        let config = OpenRouterConfig::for_tests(
            Some("secret-key-value".to_string()),
            "test-model".to_string(),
            "https://example.test/chat/completions".to_string(),
            5,
        );

        let answer = send_configured_prompt_with_audit_and_sender(
            &config,
            "Question",
            "Prompt",
            false,
            None,
            |_, prompt, include_diagnostics| {
                assert_eq!(prompt, "Prompt");
                assert!(!include_diagnostics);
                Ok(openrouter::LlmResponse {
                    answer: "Answer".to_string(),
                    diagnostics: None,
                })
            },
        )
        .unwrap();

        assert_eq!(answer.answer, "Answer");
        assert!(!log_dir.exists());
    }

    #[test]
    fn prompt_only_without_api_key_ignores_enabled_outbound_audit() {
        let log_dir = temp_log_dir("prompt-only");
        let audit_config = OutboundAuditConfig::from_dir(&log_dir).unwrap();
        let mut recall = Recall::new();
        recall.register(TestAdapter::new(vec![Event::new(
            "event-1",
            Source::Other("test".to_string()),
            "ask milestone",
        )]));

        let output = ask_output_with_audit(
            &recall,
            "When did I ask?",
            &OpenRouterConfig::without_api_key_for_tests(),
            false,
            DebugOptions::default(),
            Some(audit_config),
        )
        .unwrap();

        assert!(output.contains("API key: no"));
        assert!(!log_dir.exists());
    }

    #[test]
    fn audit_write_failure_prevents_provider_request() {
        let blocking_file = temp_log_dir("blocked");
        fs::write(&blocking_file, "not a directory").unwrap();
        let audit_config = OutboundAuditConfig::from_dir(&blocking_file).unwrap();
        let config = OpenRouterConfig::for_tests(
            Some("secret-key-value".to_string()),
            "test-model".to_string(),
            "https://example.test/chat/completions".to_string(),
            5,
        );
        let called = Cell::new(false);

        let error = send_configured_prompt_with_audit_and_sender(
            &config,
            "Question",
            "Prompt",
            false,
            Some(audit_config),
            |_, _, _| {
                called.set(true);
                Ok(openrouter::LlmResponse {
                    answer: "Answer".to_string(),
                    diagnostics: None,
                })
            },
        )
        .expect_err("audit failure should stop before provider send");

        assert!(error.contains("outbound audit failed"));
        assert!(!called.get());
        fs::remove_file(blocking_file).unwrap();
    }

    #[test]
    fn format_configuration_output_omits_default_endpoint_and_missing_api_key() {
        let config = OpenRouterConfig::without_api_key_for_tests();

        let output = format_configuration_output(&config);

        assert_eq!(
            output,
            "Configuration:\n  Model: deepseek/deepseek-v4-flash-0731\n  API key: no\n"
        );
    }

    #[test]
    fn format_configuration_output_shows_custom_endpoint_and_configured_api_key() {
        let config = OpenRouterConfig::for_tests(
            Some("secret-key-value".to_string()),
            "test-model".to_string(),
            "https://example.test/chat/completions".to_string(),
            5,
        );

        let output = format_configuration_output(&config);

        assert_eq!(
            output,
            "Configuration:\n  Model: test-model\n  OpenRouter base URL: https://example.test/chat/completions\n  API key: yes\n"
        );
        assert!(!output.contains("secret-key-value"));
    }

    #[test]
    fn format_answer_output_prints_compact_evidence() {
        let block = EvidenceBlock {
            source: Source::Other("test".to_string()),
            id: EventId::new("event-1"),
            timestamp: None,
            title: "ask milestone".to_string(),
            body: "ask milestone body".to_string(),
        };

        let output = format_answer_output("ask", &[block], "Answer with test:event-1");

        assert_eq!(
            output,
            "Search query:\nask\n\nEvidence:\n- test:event-1 ask milestone\n\nAnswer:\nAnswer with test:event-1\n"
        );
    }

    #[test]
    fn format_answer_output_reflects_compiled_evidence_not_raw_events() {
        let mut milestone = Event::new("session-1", Source::Codex, "Compiler milestone");
        milestone.description = "Implemented deterministic deduplication.".to_string();
        let mut decision = Event::new("session-1", Source::Codex, "Compiler decision");
        decision.description = "Decision: merge duplicate categories by union.".to_string();
        let events = vec![milestone, decision];

        let evidence = ContextCompiler::new().compile("What changed?", &events);
        let output = format_answer_output("ask", &evidence, "answer");

        let evidence_section = output
            .split("Evidence:\n")
            .nth(1)
            .and_then(|rest| rest.split("\n\n").next())
            .unwrap();
        let displayed_lines = evidence_section.lines().count();

        assert_ne!(evidence.len(), events.len());
        assert_eq!(displayed_lines, evidence.len());
    }

    #[test]
    fn append_diagnostics_output_prints_available_fields_in_stable_order() {
        let diagnostics = LlmDiagnostics {
            model: Some("openai/gpt-4o-mini".to_string()),
            provider: Some("OpenAI".to_string()),
            latency_ms: Some(345),
            usage: Some(openrouter::TokenUsage {
                prompt_tokens: Some(100),
                completion_tokens: Some(20),
                reasoning_tokens: Some(5),
                cached_prompt_tokens: Some(80),
                total_tokens: Some(120),
                estimated_cost: Some(0.0012),
            }),
            transport: Some(openrouter::TransportDiagnostics {
                request_creation_ms: 1,
                upload_to_headers_ms: 2,
                first_body_byte_ms: Some(3),
                body_completion_ms: 4,
                total_request_ms: 5,
                response_body_bytes: 6,
            }),
        };
        let mut output = "Answer:\nDone\n".to_string();

        append_diagnostics_output(&mut output, Some(&diagnostics));

        assert_eq!(
            output,
            "Answer:\nDone\n\nDiagnostics:\n  Model: openai/gpt-4o-mini\n  Provider: OpenAI\n  Prompt tokens: 100\n  Completion tokens: 20\n  Cached prompt tokens: 80\n  Reasoning tokens: 5\n  Total tokens: 120\n  Latency: 345 ms\n  Request creation: 1 ms\n  Upload to response headers: 2 ms\n  First body byte: 3 ms\n  Body completion: 4 ms\n  Total request: 5 ms\n  Response body bytes: 6\n  Cost: 0.0012\n"
        );
    }

    #[test]
    fn format_debug_query_prints_only_original_and_normalized_queries() {
        let output = format_debug_query(
            "When did I ask?",
            &RetrievalPlan::Search {
                query: "ask".to_string(),
            },
        );

        assert_eq!(
            output,
            "Original query:\nWhen did I ask?\n\nRetrieval plan:\nask\n"
        );
        assert!(!output.contains("Search results"));
        assert!(!output.contains("Configuration"));
    }

    #[test]
    fn format_debug_search_prints_results_and_diagnostics() {
        let mut diagnostics = Metadata::new();
        diagnostics.insert("matched_terms".to_string(), "ask milestone".to_string());
        let results = vec![SearchResult {
            event: EventRef::new(Source::Other("test".to_string()), "event-1"),
            score: Some(42),
            snippet: "ask milestone".to_string(),
            metadata: Metadata::new(),
            diagnostics,
        }];

        let output = format_debug_search(&results);

        assert_eq!(
            output,
            "Search results:\n- source: test\n  identifier: event-1\n  score: 42\n  snippet: ask milestone\n  diagnostics:\n    matched_terms: ask milestone\n"
        );
    }

    #[test]
    fn format_debug_retrieval_prints_timeline_results_for_timeline_plan() {
        let mut event = Event::new(
            "event-1",
            Source::Other("test".to_string()),
            "ask milestone",
        );
        event.timestamp = Some(Timestamp::new("2026-08-04T12:00:00Z"));

        let output = format_debug_retrieval(
            &RetrievalPlan::Timeline {
                range: DateRange::Yesterday,
            },
            &[],
            &[event],
        );

        assert_eq!(
            output,
            "Timeline results:\n- source: test\n  identifier: event-1\n  timestamp: 2026-08-04T12:00:00Z\n  title: ask milestone\n"
        );
    }

    #[test]
    fn format_debug_retrieval_prints_empty_timeline_results() {
        let output = format_debug_retrieval(
            &RetrievalPlan::Timeline {
                range: DateRange::Yesterday,
            },
            &[],
            &[],
        );

        assert_eq!(output, "Timeline results:\n(none)\n");
    }

    #[test]
    fn format_debug_prompt_is_raw_prompt_body() {
        let prompt = "You are answering questions about my engineering history.\n\nQuestion:\n\nWhat happened?\n";

        let output = format_debug_output(
            DebugOptions {
                prompt: true,
                ..DebugOptions::default()
            },
            "What happened?",
            &RetrievalPlan::Search {
                query: "happened".to_string(),
            },
            &[],
            &[],
            prompt,
        );

        assert_eq!(output, prompt);
        assert!(output.starts_with("You are answering questions"));
        assert!(!output.contains("===== Messages ====="));
        assert!(!output.contains("Configuration:"));
    }

    #[test]
    fn format_timeline_prints_concise_rows() {
        let mut event = Event::new("session-1", Source::Codex, "Event title...");
        event.timestamp = Some(Timestamp::new("2026-07-20T12:34:56Z"));
        let timeline = Timeline {
            events: vec![event],
        };

        let output = format_timeline(&timeline);

        assert_eq!(output, "2026-07-20  codex  Event title...\n");
    }

    #[test]
    fn format_timeline_handles_empty_timeline() {
        let output = format_timeline(&Timeline::new());

        assert_eq!(output, "No events\n");
    }

    #[test]
    fn parse_event_ref_accepts_source_qualified_ids() {
        let event_ref = parse_event_ref("codex:session-1").unwrap();

        assert_eq!(event_ref.source, Source::Codex);
        assert_eq!(event_ref.id.as_str(), "session-1");
    }

    #[test]
    fn parse_event_ref_rejects_unqualified_ids() {
        let error = parse_event_ref("session-1").unwrap_err();

        assert!(error.contains("expected source:id"));
    }

    #[test]
    fn format_event_includes_all_event_fields() {
        let mut event = Event::new("session-1", Source::Codex, "Inspect milestone");
        event.timestamp = Some(Timestamp::new("2026-07-18T12:00:00Z"));
        event.description = "Implemented inspect.".to_string();
        event.metadata = Metadata::from([("path".to_string(), "/tmp/session.jsonl".to_string())]);

        let output = format_event(&event);

        assert!(output.contains("Source: codex"));
        assert!(output.contains("ID: session-1"));
        assert!(output.contains("Title: Inspect milestone"));
        assert!(output.contains("Timestamp: 2026-07-18T12:00:00Z"));
        assert!(output.contains("Implemented inspect."));
        assert!(output.contains("path: /tmp/session.jsonl"));
    }

    fn temp_log_dir(name: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("recall-cli-audit-{name}-{unique}"))
    }
}
