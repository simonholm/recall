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
use recall_claude::ClaudeAdapter;
use recall_codex::CodexAdapter;
#[cfg(test)]
use recall_core::ContextCompiler;
use recall_core::{
    annotate_search_results, compile_ask_evidence, project_metadata_matches_query_text,
    AdapterCallTiming, DateRange, Event, EventId, EventRef, EvidenceBlock, PromptBuilder, Recall,
    RetrievalPlan, RetrievalPlanner, SearchDiagnostics, SearchMatch, SearchQuery, SearchResult,
    Source, Timeline, TimelineDiagnostics, ASK_RESULT_LIMIT,
};
use recall_git::GitAdapter;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::{self, BufRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode};
use std::time::Instant;

const CLI_VERSION: &str = cli_version(
    option_env!("RECALL_LONG_VERSION"),
    env!("CARGO_PKG_VERSION"),
);

const fn cli_version(
    long_version: Option<&'static str>,
    package_version: &'static str,
) -> &'static str {
    match long_version {
        Some(version) => version,
        None => package_version,
    }
}

/// Local development memory CLI.
#[derive(Debug, Parser)]
#[command(name = "recall")]
#[command(about = "Local development memory system")]
#[command(version = CLI_VERSION)]
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
        /// Print a Codex Desktop deep link for the compiled prompt instead of calling OpenRouter.
        #[arg(long)]
        codex: bool,
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
    recall.register(ClaudeAdapter::new());
    recall.register(default_git_adapter());
    recall
}

fn default_git_adapter() -> GitAdapter {
    let repo_dirs = git_repository_dirs_for_current_working_dir();
    if repo_dirs.is_empty() {
        GitAdapter::new()
    } else {
        GitAdapter::with_repo_dirs(repo_dirs)
    }
}

fn git_repository_dirs_for_current_working_dir() -> Vec<PathBuf> {
    std::env::current_dir()
        .ok()
        .map(|current_dir| git_repository_dirs_for_current_dir(&current_dir))
        .unwrap_or_default()
}

fn git_repository_dirs_for_current_dir(current_dir: &Path) -> Vec<PathBuf> {
    let Some(root) = git_repository_root(current_dir) else {
        return Vec::new();
    };
    let Some(parent) = root.parent() else {
        return vec![root];
    };
    let Ok(entries) = fs::read_dir(parent) else {
        return vec![root];
    };

    let mut repo_dirs = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join(".git").exists())
        .collect::<Vec<_>>();
    repo_dirs.sort();
    repo_dirs.dedup();

    if repo_dirs.is_empty() {
        vec![root]
    } else {
        repo_dirs
    }
}

fn git_repository_root(current_dir: &Path) -> Option<PathBuf> {
    let output = ProcessCommand::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(current_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let root = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!root.is_empty()).then(|| PathBuf::from(root))
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
            codex,
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
                    codex,
                    None,
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct AskTimings {
    retrieval_planning_ms: u64,
    retrieval: Option<RetrievalTimings>,
    context_compilation_ms: u64,
    prompt_construction_ms: u64,
    openrouter_request_start_ms: Option<u64>,
    total_ms: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct RetrievalTimings {
    planning_ms: u64,
    search: Option<SearchRetrievalTimings>,
    timeline: Option<TimelineRetrievalTimings>,
    inspect_total_ms: u64,
    inspections: Vec<InspectTiming>,
    total_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SearchRetrievalTimings {
    adapter_searches: Vec<AdapterCallTiming>,
    sort_ms: u64,
    total_ms: u64,
    limit_ms: u64,
    returned_results: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TimelineRetrievalTimings {
    adapter_timelines: Vec<AdapterCallTiming>,
    sort_ms: u64,
    total_ms: u64,
    filter_limit_ms: u64,
    returned_events: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InspectTiming {
    event_ref: EventRef,
    elapsed_ms: u64,
    found: bool,
}

fn ask_output(
    recall: &Recall,
    question: &str,
    openrouter_config: &OpenRouterConfig,
    include_diagnostics: bool,
    debug_options: DebugOptions,
    codex: bool,
    codex_workspace_path: Option<&Path>,
) -> Result<String, String> {
    ask_output_with_audit(
        recall,
        question,
        openrouter_config,
        include_diagnostics,
        debug_options,
        OutboundAuditConfig::from_env(),
        codex,
        codex_workspace_path,
    )
}

fn ask_output_with_audit(
    recall: &Recall,
    question: &str,
    openrouter_config: &OpenRouterConfig,
    include_diagnostics: bool,
    debug_options: DebugOptions,
    audit_config: Option<OutboundAuditConfig>,
    codex: bool,
    codex_workspace_path: Option<&Path>,
) -> Result<String, String> {
    let ask_started = Instant::now();
    let mut timings = AskTimings::default();

    let retrieval_started = Instant::now();
    let plan = RetrievalPlanner::new().plan(question);
    let planning_ms = elapsed_ms(retrieval_started);
    let (retrieval, retrieval_timings) = if include_diagnostics {
        let retrieval = ask_retrieval_with_timings(recall, &plan, planning_ms)?;
        let timings = retrieval.timings.clone();
        (retrieval.into_retrieval(), Some(timings))
    } else {
        (ask_retrieval(recall, &plan)?, None)
    };
    timings.retrieval_planning_ms = elapsed_ms(retrieval_started);
    timings.retrieval = retrieval_timings;

    let compilation_started = Instant::now();
    let evidence = compile_evidence(&plan, question, &retrieval.events);
    timings.context_compilation_ms = elapsed_ms(compilation_started);

    let prompt_started = Instant::now();
    let prompt = PromptBuilder::new().build(question, &evidence);
    timings.prompt_construction_ms = elapsed_ms(prompt_started);

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

    if codex {
        let link = codex_new_thread_deep_link(codex_workspace_path, &prompt);
        return Ok(format_codex_deep_link_output(&link));
    }

    if let Some(error) = openrouter_config.credential_error() {
        return Err(error.to_string());
    }

    timings.total_ms = elapsed_ms(ask_started);
    if !openrouter_config.is_configured() {
        let mut output = format!(
            "{configuration}\n{}",
            format_prompt_only_output(&format_retrieval_plan(&plan), &prompt)
        );
        if include_diagnostics {
            append_ask_timings_output(&mut output, &timings);
        }
        return Ok(output);
    }

    timings.openrouter_request_start_ms = Some(elapsed_ms(ask_started));
    let answer = send_configured_prompt_with_audit(
        openrouter_config,
        question,
        &prompt,
        include_diagnostics,
        audit_config,
    )?;
    timings.total_ms = elapsed_ms(ask_started);
    let mut answer_output =
        format_answer_output(&format_retrieval_plan(&plan), &evidence, &answer.answer);
    if include_diagnostics {
        append_diagnostics_output(&mut answer_output, answer.diagnostics.as_ref());
        append_ask_timings_output(&mut answer_output, &timings);
    }

    Ok(format!("{configuration}\n{answer_output}"))
}

fn codex_new_thread_deep_link(workspace_path: Option<&Path>, prompt: &str) -> String {
    let mut serializer = form_urlencoded::Serializer::new(String::new());
    if let Some(path) = workspace_path {
        serializer.append_pair("path", &path.display().to_string());
    }
    serializer.append_pair("prompt", prompt);
    let query = serializer.finish();
    format!("codex://threads/new?{query}")
}

fn format_codex_deep_link_output(link: &str) -> String {
    format!("Codex deep link:\n{link}\n")
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
        RetrievalPlan::ProjectLatest { .. } => format_debug_timeline(events),
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

fn append_ask_timings_output(output: &mut String, timings: &AskTimings) {
    writeln!(output).unwrap();
    writeln!(output, "Ask timings:").unwrap();
    writeln!(
        output,
        "  Retrieval/planning: {} ms",
        timings.retrieval_planning_ms
    )
    .unwrap();
    if let Some(retrieval) = &timings.retrieval {
        writeln!(output, "  Retrieval/planning detail:").unwrap();
        writeln!(output, "    Planning: {} ms", retrieval.planning_ms).unwrap();
        if let Some(search) = &retrieval.search {
            writeln!(output, "    Search total: {} ms", search.total_ms).unwrap();
            for adapter in &search.adapter_searches {
                writeln!(
                    output,
                    "    Search adapter {}: {} ms ({} results)",
                    adapter.source.as_str(),
                    adapter.elapsed_ms,
                    adapter.item_count
                )
                .unwrap();
            }
            writeln!(output, "    Search sort: {} ms", search.sort_ms).unwrap();
            writeln!(
                output,
                "    Search result limit: {} ms ({} retained)",
                search.limit_ms, search.returned_results
            )
            .unwrap();
        }
        if let Some(timeline) = &retrieval.timeline {
            writeln!(output, "    Timeline total: {} ms", timeline.total_ms).unwrap();
            for adapter in &timeline.adapter_timelines {
                writeln!(
                    output,
                    "    Timeline adapter {}: {} ms ({} events)",
                    adapter.source.as_str(),
                    adapter.elapsed_ms,
                    adapter.item_count
                )
                .unwrap();
            }
            writeln!(output, "    Timeline sort: {} ms", timeline.sort_ms).unwrap();
            writeln!(
                output,
                "    Timeline filter/limit: {} ms ({} retained)",
                timeline.filter_limit_ms, timeline.returned_events
            )
            .unwrap();
        }
        writeln!(
            output,
            "    Inspect total: {} ms ({} events)",
            retrieval.inspect_total_ms,
            retrieval.inspections.len()
        )
        .unwrap();
        for inspection in &retrieval.inspections {
            let status = if inspection.found { "found" } else { "missing" };
            writeln!(
                output,
                "    Inspect {}: {} ms ({status})",
                format_event_ref(&inspection.event_ref),
                inspection.elapsed_ms
            )
            .unwrap();
        }
        writeln!(
            output,
            "    Retrieval function total: {} ms",
            retrieval.total_ms
        )
        .unwrap();
    }
    writeln!(
        output,
        "  Context compilation: {} ms",
        timings.context_compilation_ms
    )
    .unwrap();
    writeln!(
        output,
        "  Prompt construction: {} ms",
        timings.prompt_construction_ms
    )
    .unwrap();
    match timings.openrouter_request_start_ms {
        Some(openrouter_request_start_ms) => {
            writeln!(
                output,
                "  OpenRouter request start: {openrouter_request_start_ms} ms"
            )
            .unwrap();
        }
        None => writeln!(output, "  OpenRouter request start: none").unwrap(),
    }
    writeln!(output, "  Total ask: {} ms", timings.total_ms).unwrap();
}

struct AskRetrieval {
    events: Vec<Event>,
    search_results: Vec<SearchResult>,
}

struct AskRetrievalWithTimings {
    events: Vec<Event>,
    search_results: Vec<SearchResult>,
    timings: RetrievalTimings,
}

impl AskRetrievalWithTimings {
    fn into_retrieval(self) -> AskRetrieval {
        AskRetrieval {
            events: self.events,
            search_results: self.search_results,
        }
    }
}

fn ask_retrieval(recall: &Recall, plan: &RetrievalPlan) -> Result<AskRetrieval, String> {
    let retrieval = recall
        .ask_retrieval(plan)
        .map_err(|error| error.to_string())?;
    Ok(AskRetrieval {
        events: retrieval.events,
        search_results: retrieval.search_results,
    })
}

fn ask_retrieval_with_timings(
    recall: &Recall,
    plan: &RetrievalPlan,
    planning_ms: u64,
) -> Result<AskRetrievalWithTimings, String> {
    let total_started = Instant::now();
    match plan {
        RetrievalPlan::Search { query } => {
            let search = recall
                .search_with_diagnostics(query.subject())
                .map_err(|error| error.to_string())?;
            let SearchDiagnostics {
                adapter_searches,
                sort_ms,
                total_ms: search_total_ms,
            } = search.diagnostics;

            let limit_started = Instant::now();
            let search_matches: Vec<_> =
                search.results.into_iter().take(ASK_RESULT_LIMIT).collect();
            let limit_ms = elapsed_ms(limit_started);

            let returned_results = search_matches.len();
            let (search_results, events) = split_search_matches(search_matches);
            let mut search_results = search_results;
            annotate_search_results(query, &mut search_results);
            let inspect_total_ms = 0;
            let total_ms = elapsed_ms(total_started);

            Ok(AskRetrievalWithTimings {
                events,
                search_results,
                timings: RetrievalTimings {
                    planning_ms,
                    search: Some(SearchRetrievalTimings {
                        adapter_searches,
                        sort_ms,
                        total_ms: search_total_ms,
                        limit_ms,
                        returned_results,
                    }),
                    timeline: None,
                    inspect_total_ms,
                    inspections: Vec::new(),
                    total_ms,
                },
            })
        }
        RetrievalPlan::ProjectLatest { query } => {
            let timeline = recall
                .timeline_with_diagnostics()
                .map_err(|error| error.to_string())?;
            let TimelineDiagnostics {
                adapter_timelines,
                sort_ms,
                total_ms: timeline_total_ms,
            } = timeline.diagnostics;

            let filter_started = Instant::now();
            let events: Vec<_> = project_latest_events(timeline.timeline.events, query)
                .into_iter()
                .take(ASK_RESULT_LIMIT)
                .collect();
            let filter_limit_ms = elapsed_ms(filter_started);
            let returned_events = events.len();
            let inspect_total_ms = 0;
            let total_ms = elapsed_ms(total_started);

            Ok(AskRetrievalWithTimings {
                events,
                search_results: Vec::new(),
                timings: RetrievalTimings {
                    planning_ms,
                    search: None,
                    timeline: Some(TimelineRetrievalTimings {
                        adapter_timelines,
                        sort_ms,
                        total_ms: timeline_total_ms,
                        filter_limit_ms,
                        returned_events,
                    }),
                    inspect_total_ms,
                    inspections: Vec::new(),
                    total_ms,
                },
            })
        }
        RetrievalPlan::Timeline { range, query } => {
            let timeline = recall
                .timeline_with_diagnostics()
                .map_err(|error| error.to_string())?;
            let TimelineDiagnostics {
                adapter_timelines,
                sort_ms,
                total_ms,
            } = timeline.diagnostics;

            let filter_started = Instant::now();
            let events: Vec<_> = timeline
                .timeline
                .events
                .into_iter()
                .filter(|event| {
                    event
                        .timestamp
                        .as_ref()
                        .is_some_and(|timestamp| range.contains_timestamp(timestamp))
                })
                .filter(|event| event_matches_query(event, query))
                .collect();
            let events = match range {
                DateRange::Day(_) => events,
                _ => events.into_iter().take(ASK_RESULT_LIMIT).collect(),
            };
            let filter_limit_ms = elapsed_ms(filter_started);
            let returned_events = events.len();

            Ok(AskRetrievalWithTimings {
                events,
                search_results: Vec::new(),
                timings: RetrievalTimings {
                    planning_ms,
                    search: None,
                    timeline: Some(TimelineRetrievalTimings {
                        adapter_timelines,
                        sort_ms,
                        total_ms,
                        filter_limit_ms,
                        returned_events,
                    }),
                    inspect_total_ms: 0,
                    inspections: Vec::new(),
                    total_ms: elapsed_ms(total_started),
                },
            })
        }
    }
}

fn split_search_matches(search_matches: Vec<SearchMatch>) -> (Vec<SearchResult>, Vec<Event>) {
    search_matches
        .into_iter()
        .map(|search_match| (search_match.result, search_match.event))
        .unzip()
}

fn compile_evidence(plan: &RetrievalPlan, question: &str, events: &[Event]) -> Vec<EvidenceBlock> {
    compile_ask_evidence(plan, question, events)
}

fn project_latest_events(events: Vec<Event>, query: &str) -> Vec<Event> {
    events
        .into_iter()
        .filter(|event| project_metadata_matches_query_text(&event.metadata, query))
        .collect()
}

#[cfg(test)]
fn ask_timeline_events(
    recall: &Recall,
    range: &DateRange,
    query: &str,
) -> Result<Vec<Event>, String> {
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
        .filter(|event| event_matches_query(event, query))
        .collect();

    match range {
        DateRange::Day(_) => Ok(events),
        _ => Ok(events.into_iter().take(ASK_RESULT_LIMIT).collect()),
    }
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

#[cfg(test)]
fn normalize_ask_query(question: &str) -> String {
    match RetrievalPlanner::new().plan(question) {
        RetrievalPlan::Search { query } => query.subject().to_string(),
        RetrievalPlan::ProjectLatest { query } => query,
        RetrievalPlan::Timeline { range, query } => {
            format_timeline_query(&range, &query).unwrap_or_else(|| format_date_range(&range))
        }
    }
}

fn format_retrieval_plan(plan: &RetrievalPlan) -> String {
    match plan {
        RetrievalPlan::Search { query } => format_search_query(query),
        RetrievalPlan::ProjectLatest { query } => format!("project latest {query}"),
        RetrievalPlan::Timeline { range, query } => format_timeline_query(range, query)
            .map(|query| format!("timeline {query}"))
            .unwrap_or_else(|| format!("timeline {}", format_date_range(range))),
    }
}

fn format_search_query(query: &SearchQuery) -> String {
    let mut formatted = query.subject().to_string();
    if query.intent() != recall_core::SearchIntent::Plain {
        write!(formatted, " intent:{}", query.intent().as_str()).unwrap();
    }
    if !query.intent_terms().is_empty() {
        write!(
            formatted,
            " intent_terms:{}",
            query.intent_terms().join(" ")
        )
        .unwrap();
    }
    formatted
}

fn format_timeline_query(range: &DateRange, query: &str) -> Option<String> {
    if query.is_empty() {
        None
    } else {
        Some(format!("{} {query}", format_date_range(range)))
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

#[cfg(test)]
fn inspect_result_event(recall: &Recall, event_ref: &EventRef) -> Result<Event, String> {
    recall
        .inspect(event_ref)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("event not found: {}", format_event_ref(event_ref)))
}

fn format_event_ref(event_ref: &EventRef) -> String {
    format!("{}:{}", event_ref.source.as_str(), event_ref.id.as_str())
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
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
    use chrono::NaiveDate;
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

    #[derive(Debug)]
    struct SearchEventsOnlyAdapter {
        event: Event,
    }

    impl Adapter for SearchEventsOnlyAdapter {
        fn source(&self) -> Source {
            Source::Other("test".to_string())
        }

        fn search(&self, _query: &str) -> AdapterResult<Vec<SearchResult>> {
            Ok(Vec::new())
        }

        fn search_events(&self, _query: &str) -> AdapterResult<Vec<SearchMatch>> {
            Ok(vec![SearchMatch {
                result: SearchResult {
                    event: EventRef::new(self.event.source.clone(), self.event.id.clone()),
                    score: Some(1),
                    snippet: self.event.title.clone(),
                    metadata: Metadata::new(),
                    diagnostics: Metadata::new(),
                },
                event: self.event.clone(),
            }])
        }

        fn timeline(&self) -> AdapterResult<Timeline> {
            Ok(Timeline { events: Vec::new() })
        }

        fn inspect(&self, _id: &EventId) -> AdapterResult<Option<Event>> {
            Err(recall_core::AdapterError::new(
                Source::Other("test".to_string()),
                "inspect should not be called",
            ))
        }
    }

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn clap_version_uses_package_version() {
        let error = Cli::command()
            .try_get_matches_from(["recall", "--version"])
            .unwrap_err();

        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(error
            .to_string()
            .starts_with(&format!("recall {}", env!("CARGO_PKG_VERSION"))));
    }

    #[test]
    fn cli_version_uses_build_metadata_when_available() {
        assert_eq!(
            cli_version(Some("0.1.0 (922eafb-dirty)"), "0.1.0"),
            "0.1.0 (922eafb-dirty)"
        );
        assert_eq!(cli_version(None, "0.1.0"), "0.1.0");
    }

    #[test]
    fn ask_debug_prompt_flag_defaults_to_false() {
        let cli = Cli::try_parse_from(["recall", "ask", "What happened?"]).unwrap();

        let Command::Ask {
            debug_query,
            debug_search,
            debug_prompt,
            codex,
            ..
        } = cli.command
        else {
            panic!("expected ask command");
        };
        assert!(!debug_query);
        assert!(!debug_search);
        assert!(!debug_prompt);
        assert!(!codex);
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
            codex,
            ..
        } = cli.command
        else {
            panic!("expected ask command");
        };
        assert!(debug_query);
        assert!(debug_search);
        assert!(debug_prompt);
        assert!(!codex);
    }

    #[test]
    fn ask_codex_flag_is_accepted() {
        let cli = Cli::try_parse_from(["recall", "ask", "What happened?", "--codex"]).unwrap();

        let Command::Ask { codex, .. } = cli.command else {
            panic!("expected ask command");
        };
        assert!(codex);
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

        assert_eq!(recall.adapter_count(), 3);
    }

    #[test]
    fn print_search_results_accepts_empty_results() {
        print_search_results(&[]);
    }

    #[test]
    fn normalize_ask_query_removes_question_words_and_punctuation() {
        assert_eq!(
            normalize_ask_query("When did I implement timeline?"),
            "timeline"
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
            "eventref"
        );
        assert_eq!(
            normalize_ask_query("What evidence shows that I actually completed disk-guard today?"),
            "today disk guard"
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
            "today recall"
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

        let events = ask_timeline_events(&recall, &DateRange::LastDays(30_000), "").unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_str(), "included");
    }

    #[test]
    fn ask_timeline_events_filters_by_explicit_subject_terms() {
        let mut recall = Recall::new();
        let mut disk_guard = Event::new(
            "disk-guard",
            Source::Other("test".to_string()),
            "Completed disk-guard",
        );
        disk_guard.timestamp = Some(Timestamp::new("2026-08-30T12:00:00Z"));
        disk_guard.description = "Finished the disk-guard release checks.".to_string();
        let mut unrelated = Event::new(
            "unrelated",
            Source::Other("test".to_string()),
            "Completed unrelated work",
        );
        unrelated.timestamp = Some(Timestamp::new("2026-08-30T13:00:00Z"));
        unrelated.description = "Finished a different task today.".to_string();
        recall.register(TestAdapter::new(vec![disk_guard, unrelated]));

        let events =
            ask_timeline_events(&recall, &DateRange::LastDays(30_000), "disk guard").unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_str(), "disk-guard");
    }

    #[test]
    fn ask_project_latest_prefers_newest_same_basename_project_event() {
        let mut old = Event::new(
            "old-rust-rewrite",
            Source::Codex,
            "disk-agent: Evaluate and propose a Rust migration",
        );
        old.timestamp = Some(Timestamp::new("2026-06-29T19:44:14.479Z"));
        old.description = "The disk-agent project is being rewritten in Rust.".to_string();
        old.metadata
            .insert("cwd".to_string(), "/home/simon/disk-agent".to_string());

        let mut current = Event::new(
            "current-left-off",
            Source::Codex,
            "Resume the interrupted disk-agent session from the current working tree.",
        );
        current.timestamp = Some(Timestamp::new("2026-08-29T07:15:50.861Z"));
        current.description =
            "Committed the intended interrupted work locally; no push was performed.".to_string();
        current.metadata.insert(
            "cwd".to_string(),
            "/home/simon/labs/repos/disk-agent".to_string(),
        );

        let mut unrelated_newer = Event::new(
            "unrelated-newer",
            Source::Codex,
            "Mention disk-agent from Recall",
        );
        unrelated_newer.timestamp = Some(Timestamp::new("2026-08-30T07:15:50.861Z"));
        unrelated_newer.metadata.insert(
            "cwd".to_string(),
            "/home/simon/labs/repos/recall".to_string(),
        );

        let mut recall = Recall::new();
        recall.register(TestAdapter::new(vec![old, current, unrelated_newer]));

        let plan = RetrievalPlanner::new().plan("Where did I leave off with disk-agent?");
        let retrieval = ask_retrieval(&recall, &plan).unwrap();

        assert_eq!(
            plan,
            RetrievalPlan::ProjectLatest {
                query: "disk agent".to_string()
            }
        );
        assert_eq!(
            retrieval
                .events
                .iter()
                .map(|event| event.id.as_str())
                .collect::<Vec<_>>(),
            vec!["current-left-off", "old-rust-rewrite"]
        );
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
                let mut event = Event::new(
                    format!("event-{index}"),
                    source,
                    format!("Recall event {index}"),
                );
                event.timestamp = Some(Timestamp::new(format!(
                    "2026-08-05T12:{:02}:00Z",
                    59 - index
                )));
                event
            })
            .collect();
        recall.register(TestAdapter::new(events));

        let plan = RetrievalPlanner::new().plan("What did I work on in Recall on August 5, 2026?");
        let RetrievalPlan::Timeline { range, query } = plan else {
            panic!("expected explicit date to use timeline retrieval");
        };
        let events = ask_timeline_events(&recall, &range, &query).unwrap();

        assert_eq!(events.len(), 10);
        assert_eq!(events[8].source, Source::Git);
        assert_eq!(events[8].id.as_str(), "event-8");
        assert_eq!(events[9].source, Source::Git);
        assert_eq!(events[9].id.as_str(), "event-9");
    }

    #[test]
    fn default_git_discovery_is_stable_across_sibling_repository_cwds() {
        let parent = temp_log_dir("git-discovery");
        let recall_repo = parent.join("recall");
        let reel_repo = parent.join("reel2ocr");
        init_git_repo(&recall_repo);
        init_git_repo(&reel_repo);

        let from_recall = git_repository_dirs_for_current_dir(&recall_repo);
        let from_reel = git_repository_dirs_for_current_dir(&reel_repo);

        assert_eq!(from_recall, from_reel);
        assert_eq!(from_recall, vec![recall_repo, reel_repo]);
    }

    #[test]
    fn broad_timeline_ask_keeps_git_evidence_across_sibling_repository_cwds() {
        let parent = temp_log_dir("git-timeline");
        let recall_repo = parent.join("recall");
        let reel_repo = parent.join("reel2ocr");
        init_git_repo(&recall_repo);
        init_git_repo(&reel_repo);
        let git_sha = commit_git_repo(
            &recall_repo,
            "fix.txt",
            "fix",
            "fix: preserve temporal subject terms",
            "Validated broad temporal Recall retrieval.",
            "2026-08-31T09:00:00Z",
        );
        commit_git_repo(
            &reel_repo,
            "ocr.txt",
            "ocr",
            "docs: update OCR notes",
            "",
            "2026-08-30T09:00:00Z",
        );

        let mut codex_event = Event::new(
            "01a0565a-544b-71d0-9f98-8e8ed322d2b7",
            Source::Codex,
            "Finish the current Recall work.",
        );
        codex_event.timestamp = Some(Timestamp::new("2026-08-31T10:00:00Z"));

        let recall_from_recall =
            broad_timeline_recall(&recall_repo, TestAdapter::new(vec![codex_event.clone()]));
        let recall_from_reel =
            broad_timeline_recall(&reel_repo, TestAdapter::new(vec![codex_event]));
        let range = DateRange::Day(NaiveDate::from_ymd_opt(2026, 8, 31).unwrap());

        let from_recall = ask_timeline_events(&recall_from_recall, &range, "").unwrap();
        let from_reel = ask_timeline_events(&recall_from_reel, &range, "").unwrap();

        for events in [&from_recall, &from_reel] {
            assert!(events
                .iter()
                .any(|event| event.source == Source::Git && event.id.as_str() == git_sha));
            assert!(events.iter().any(|event| event.source == Source::Codex
                && event.id.as_str() == "01a0565a-544b-71d0-9f98-8e8ed322d2b7"));
        }
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
            query: String::new(),
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
    fn answer_output_evidence_lines_match_prompt_evidence_blocks() {
        let block = EvidenceBlock {
            source: Source::Codex,
            id: EventId::new("session-1"),
            timestamp: Some(Timestamp::new("2026-08-31T10:00:00Z")),
            title: "Finish the current Recall work.".to_string(),
            body: "Implemented disk-guard retrieval notes.\nPrompt-only temporal guidance."
                .to_string(),
        };

        let displayed = format_answer_output("timeline today", &[block.clone()], "answer");
        let prompt = PromptBuilder::new().build("What did I do today?", &[block]);

        assert!(displayed.contains("- codex:session-1 Finish the current Recall work."));
        assert!(prompt.contains("Id: session-1"));
        assert!(prompt.contains("Implemented disk-guard retrieval notes."));
        assert!(prompt.contains("Prompt-only temporal guidance."));
        assert!(!displayed.contains("disk-guard"));
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
            query: SearchQuery::plain("compiler"),
        };

        let evidence = compile_evidence(&plan, "What changed?", &[first, second]);

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id.as_str(), "project:/repo/recall");
        assert_eq!(evidence[0].title, "Project: /repo/recall");
    }

    #[test]
    fn project_latest_evidence_uses_newest_resumable_event_without_older_project_history() {
        let mut latest = Event::new(
            "latest-disk-agent",
            Source::Codex,
            "Resume the interrupted disk-agent session from the current working tree.",
        );
        latest.timestamp = Some(Timestamp::new("2026-08-29T07:15:50.861Z"));
        latest.metadata.insert(
            "cwd".to_string(),
            "/home/simon/labs/repos/disk-agent".to_string(),
        );
        latest.description = concat!(
            "Resume the interrupted disk-agent session from the current working tree.\n\n",
            "The previous session completed the implementation and validation, but the SSH connection died while attempting to stage/commit. Do not redo or expand the implementation.\n\n",
            "Committed the intended interrupted work locally.\n\n",
            "Commit: `ca46c169efa06389002a0157da5857a22f564bc7`\n",
            "Message: `Improve disk investigation diagnostics`\n\n",
            "Validation passed:\n",
            "- `cargo fmt --check`\n",
            "- `git diff --check`\n",
            "- `cargo test` with 57 total tests passing plus doc-tests\n\n",
            "Final git status:\n",
            "```text\n",
            "## main...origin/main [ahead 1]\n",
            "```\n\n",
            "No push was performed, and the working tree has no remaining diff.\n\n",
            "<oai-mem-citation>\n",
            "<citation_entries>\n",
            "MEMORY.md:2151-2155|note=[disk-agent prior validation context]\n",
            "</citation_entries>\n",
            "</oai-mem-citation>"
        )
        .to_string();

        let mut older_podman = Event::new(
            "older-podman",
            Source::Codex,
            "The new Podman attribution is not rendering in a real run.",
        );
        older_podman.timestamp = Some(Timestamp::new("2026-08-15T09:56:14.271Z"));
        older_podman.metadata.insert(
            "cwd".to_string(),
            "/home/simon/labs/repos/disk-agent".to_string(),
        );
        older_podman.description = concat!(
            "The new Podman attribution is not rendering in a real run.\n",
            "The real Podman output reveals the parsing bug.\n",
            "Implemented the scoped fix in src/podman.rs: container attribution now reads Size.rwSize."
        )
        .to_string();

        let mut recall = Recall::new();
        recall.register(TestAdapter::new(vec![older_podman, latest]));

        let plan = RetrievalPlanner::new().plan("Where did I leave off with disk-agent?");
        let retrieval = ask_retrieval(&recall, &plan).unwrap();
        let evidence = compile_evidence(
            &plan,
            "Where did I leave off with disk-agent?",
            &retrieval.events,
        );

        assert_eq!(
            retrieval
                .events
                .iter()
                .map(|event| event.id.as_str())
                .collect::<Vec<_>>(),
            vec!["latest-disk-agent", "older-podman"]
        );
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id.as_str(), "latest-disk-agent");
        assert_eq!(
            evidence[0].title,
            "Resume the interrupted disk-agent session from the current working tree."
        );
        assert!(evidence[0]
            .body
            .contains("ca46c169efa06389002a0157da5857a22f564bc7"));
        assert!(evidence[0].body.contains("Validation passed:"));
        assert!(evidence[0].body.contains("`cargo test`"));
        assert!(evidence[0].body.contains("Final git status:"));
        assert!(evidence[0].body.contains("## main...origin/main [ahead 1]"));
        assert!(evidence[0].body.contains("No push was performed"));
        assert!(!evidence[0].body.contains("Podman"));
        assert!(!evidence[0].body.contains("Size.rwSize"));
        assert!(!evidence[0].body.contains("MEMORY.md"));
        assert!(!evidence[0].body.contains("oai-mem-citation"));
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
            false,
            None,
        )
        .unwrap();
        let events = ask_events(&recall, "ask").unwrap();
        let evidence = ContextCompiler::new().compile("When did I ask?", &events);
        let prompt = PromptBuilder::new().build("When did I ask?", &evidence);

        assert_eq!(
            output,
            format!(
                "Configuration:\n  Model: google/gemini-2.5-flash-lite\n  API key: no\n\nSearch query:\nask\n\n{prompt}"
            )
        );
    }

    #[test]
    fn ask_output_uses_search_events_without_reinspecting_results() {
        let mut recall = Recall::new();
        let mut event = Event::new(
            "event-1",
            Source::Other("test".to_string()),
            "ask milestone",
        );
        event.description = "Implemented search_events reuse.".to_string();
        recall.register(SearchEventsOnlyAdapter { event });

        let output = ask_output(
            &recall,
            "When did I ask?",
            &OpenRouterConfig::without_api_key_for_tests(),
            false,
            DebugOptions::default(),
            false,
            None,
        )
        .unwrap();

        assert!(output.contains("Implemented search_events reuse."));
        assert!(!output.contains("inspect should not be called"));
    }

    #[test]
    fn codex_new_thread_deep_link_encodes_path_and_prompt() {
        let link = codex_new_thread_deep_link(
            Some(Path::new("/tmp/recall workspace/#1")),
            "Line one\nLine two & three+four",
        );

        assert_eq!(
            link,
            "codex://threads/new?path=%2Ftmp%2Frecall+workspace%2F%231&prompt=Line+one%0ALine+two+%26+three%2Bfour"
        );
        assert_eq!(
            decoded_deep_link_query(&link),
            vec![
                ("path".to_string(), "/tmp/recall workspace/#1".to_string()),
                (
                    "prompt".to_string(),
                    "Line one\nLine two & three+four".to_string(),
                ),
            ]
        );
    }

    #[test]
    fn codex_new_thread_deep_link_can_omit_path() {
        let link = codex_new_thread_deep_link(None, "Question only");

        assert_eq!(link, "codex://threads/new?prompt=Question+only");
        assert_eq!(
            decoded_deep_link_query(&link),
            vec![("prompt".to_string(), "Question only".to_string())]
        );
    }

    #[test]
    fn codex_ask_output_uses_compiled_prompt_without_openrouter_configuration() {
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
            true,
            None,
        )
        .unwrap();
        let link = output
            .strip_prefix("Codex deep link:\n")
            .and_then(|output| output.strip_suffix('\n'))
            .expect("output should contain only a usable deep link");
        let query = decoded_deep_link_query(link);
        let prompt = query
            .iter()
            .find_map(|(key, value)| (key == "prompt").then_some(value))
            .expect("prompt query parameter");
        let plan = RetrievalPlanner::new().plan("When did I ask?");
        let retrieval = ask_retrieval(&recall, &plan).unwrap();
        let evidence = compile_evidence(&plan, "When did I ask?", &retrieval.events);
        let expected_prompt = PromptBuilder::new().build("When did I ask?", &evidence);

        assert!(!query.iter().any(|(key, _)| key == "path"));
        assert_eq!(prompt, &expected_prompt);
        assert!(!output.contains("Configuration:"));
        assert!(!output.contains("API key"));
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
            false,
            None,
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
            "Configuration:\n  Model: google/gemini-2.5-flash-lite\n  API key: no\n"
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
    fn append_ask_timings_output_prints_available_fields_in_stable_order() {
        let timings = AskTimings {
            retrieval_planning_ms: 10,
            retrieval: Some(RetrievalTimings {
                planning_ms: 1,
                search: Some(SearchRetrievalTimings {
                    adapter_searches: vec![AdapterCallTiming {
                        source: Source::Codex,
                        elapsed_ms: 2,
                        item_count: 3,
                    }],
                    sort_ms: 4,
                    total_ms: 5,
                    limit_ms: 6,
                    returned_results: 7,
                }),
                timeline: None,
                inspect_total_ms: 8,
                inspections: vec![InspectTiming {
                    event_ref: EventRef::new(Source::Codex, "event-1"),
                    elapsed_ms: 9,
                    found: true,
                }],
                total_ms: 10,
            }),
            context_compilation_ms: 20,
            prompt_construction_ms: 30,
            openrouter_request_start_ms: Some(40),
            total_ms: 50,
        };
        let mut output = "Answer:\nDone\n".to_string();

        append_ask_timings_output(&mut output, &timings);

        assert_eq!(
            output,
            "Answer:\nDone\n\nAsk timings:\n  Retrieval/planning: 10 ms\n  Retrieval/planning detail:\n    Planning: 1 ms\n    Search total: 5 ms\n    Search adapter codex: 2 ms (3 results)\n    Search sort: 4 ms\n    Search result limit: 6 ms (7 retained)\n    Inspect total: 8 ms (1 events)\n    Inspect codex:event-1: 9 ms (found)\n    Retrieval function total: 10 ms\n  Context compilation: 20 ms\n  Prompt construction: 30 ms\n  OpenRouter request start: 40 ms\n  Total ask: 50 ms\n"
        );
    }

    #[test]
    fn format_debug_query_prints_only_original_and_normalized_queries() {
        let output = format_debug_query(
            "When did I ask?",
            &RetrievalPlan::Search {
                query: SearchQuery::plain("ask"),
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
                query: String::new(),
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
                query: String::new(),
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
                query: SearchQuery::plain("happened"),
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

    fn broad_timeline_recall(git_cwd: &Path, codex_adapter: TestAdapter) -> Recall {
        let mut recall = Recall::new();
        recall.register(codex_adapter);
        recall.register(GitAdapter::with_repo_dirs(
            git_repository_dirs_for_current_dir(git_cwd),
        ));
        recall
    }

    fn init_git_repo(repo: &Path) {
        fs::create_dir_all(repo).unwrap();
        run_git(repo, ["init"]);
        run_git(repo, ["config", "user.name", "Recall Tests"]);
        run_git(repo, ["config", "user.email", "recall@example.test"]);
    }

    fn commit_git_repo(
        repo: &Path,
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

        let output = ProcessCommand::new("git")
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

        let output = ProcessCommand::new("git")
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

    fn run_git<const N: usize>(repo: &Path, args: [&str; N]) {
        let output = ProcessCommand::new("git")
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

    fn decoded_deep_link_query(link: &str) -> Vec<(String, String)> {
        let query = link
            .strip_prefix("codex://threads/new?")
            .expect("new Codex thread deep link");
        form_urlencoded::parse(query.as_bytes())
            .into_owned()
            .collect()
    }
}
