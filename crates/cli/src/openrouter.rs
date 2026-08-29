//! OpenRouter client and shared LLM response types.
//!
//! `LlmResponse` is the canonical response type used internally. Individual
//! providers translate their native responses into this common representation so
//! callers do not depend on provider-specific response shapes. This abstraction
//! is intentionally small, but it leaves room for additional providers and local
//! inference engines such as OpenAI, Anthropic, Gemini, Ollama, and llama.cpp.

use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

pub(crate) const DEFAULT_MODEL: &str = "google/gemini-2.5-flash-lite";
pub(crate) const DEFAULT_ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 30;
pub(crate) const DEFAULT_RESPONSE_HEADER_TIMEOUT_SECS: u64 = 120;
const DEFAULT_AUTH_PATH: &str = ".local/share/recall/auth.json";
const MAX_RESPONSE_BODY_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OpenRouterConfig {
    api_key: Option<String>,
    credential_error: Option<String>,
    model: String,
    endpoint: String,
    transport: TransportConfig,
}

impl OpenRouterConfig {
    pub(crate) fn from_env() -> Self {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        Self::from_lookup_and_home(|key| std::env::var(key).ok(), home.as_deref())
    }

    #[cfg(test)]
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        Self::from_lookup_and_home(&mut lookup, None)
    }

    fn from_lookup_and_home(
        mut lookup: impl FnMut(&str) -> Option<String>,
        home: Option<&Path>,
    ) -> Self {
        let explicit_api_key = lookup("OPENROUTER_API_KEY").and_then(non_empty);
        let (api_key, credential_error) = match explicit_api_key {
            Some(api_key) => (Some(api_key), None),
            None => match home.map(load_stored_api_key) {
                Some(Ok(api_key)) => (api_key, None),
                Some(Err(error)) => (None, Some(error)),
                None => (None, None),
            },
        };
        let model = lookup("RECALL_OPENROUTER_MODEL")
            .and_then(non_empty)
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let endpoint = lookup("RECALL_OPENROUTER_ENDPOINT")
            .and_then(non_empty)
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
        let transport = TransportConfig::from_lookup(&mut lookup);

        Self {
            api_key,
            credential_error,
            model,
            endpoint,
            transport,
        }
    }

    pub(crate) fn is_configured(&self) -> bool {
        self.api_key.is_some()
    }

    pub(crate) fn credential_error(&self) -> Option<&str> {
        self.credential_error.as_deref()
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(crate) fn uses_default_endpoint(&self) -> bool {
        self.endpoint == DEFAULT_ENDPOINT
    }

    #[cfg(test)]
    pub(crate) fn without_api_key_for_tests() -> Self {
        Self::from_lookup(|_| None)
    }

    #[cfg(test)]
    pub(crate) fn for_tests(
        api_key: Option<String>,
        model: String,
        endpoint: String,
        timeout_secs: u64,
    ) -> Self {
        Self {
            api_key,
            credential_error: None,
            model,
            endpoint,
            transport: TransportConfig::with_all_timeouts(timeout_secs),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum AuthStatus {
    EnvironmentOverride,
    Stored { path: PathBuf },
    NotConfigured { path: PathBuf },
    Error(String),
}

pub(crate) fn auth_status_from_env() -> AuthStatus {
    if std::env::var("OPENROUTER_API_KEY")
        .ok()
        .and_then(non_empty)
        .is_some()
    {
        return AuthStatus::EnvironmentOverride;
    }

    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return AuthStatus::Error("HOME is not set".to_string());
    };
    auth_status_from_home(&home)
}

#[cfg(test)]
fn auth_status_from_lookup_and_home(
    mut lookup: impl FnMut(&str) -> Option<String>,
    home: &Path,
) -> AuthStatus {
    if lookup("OPENROUTER_API_KEY").and_then(non_empty).is_some() {
        return AuthStatus::EnvironmentOverride;
    }

    auth_status_from_home(home)
}

fn auth_status_from_home(home: &Path) -> AuthStatus {
    let path = auth_path(home);
    match load_auth_record(&path) {
        Ok(Some(_)) => AuthStatus::Stored { path },
        Ok(None) => AuthStatus::NotConfigured { path },
        Err(error) => AuthStatus::Error(error),
    }
}

pub(crate) fn save_api_key_from_env_home(api_key: &str) -> Result<PathBuf, String> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Err("HOME is not set".to_string());
    };
    save_api_key(&home, api_key)
}

pub(crate) fn delete_api_key_from_env_home() -> Result<Option<PathBuf>, String> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Err("HOME is not set".to_string());
    };
    delete_api_key(&home)
}

fn non_empty(value: String) -> Option<String> {
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn auth_path(home: &Path) -> PathBuf {
    home.join(DEFAULT_AUTH_PATH)
}

fn load_stored_api_key(home: &Path) -> Result<Option<String>, String> {
    load_auth_record(&auth_path(home))
}

fn load_auth_record(path: &Path) -> Result<Option<String>, String> {
    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "openrouter credential file could not be read: {}; {error}",
                path.display()
            ));
        }
    };
    if !credential_permissions_are_private(&metadata) {
        return Err(format!(
            "openrouter credential file has insecure permissions: {}; expected mode 0600",
            path.display()
        ));
    }

    let contents = fs::read_to_string(&path).map_err(|error| {
        format!(
            "openrouter credential file could not be read: {}; {error}",
            path.display()
        )
    })?;

    parse_auth_json(&contents)
        .ok_or_else(|| {
            format!(
                "openrouter credential is not configured in {}; expected openrouter_api_key",
                path.display()
            )
        })
        .map(Some)
}

fn save_api_key(home: &Path, api_key: &str) -> Result<PathBuf, String> {
    let api_key = non_empty(api_key.to_string())
        .ok_or_else(|| "openrouter API key cannot be empty".to_string())?;
    let path = auth_path(home);
    let directory = path
        .parent()
        .ok_or_else(|| format!("invalid auth path: {}", path.display()))?;
    fs::create_dir_all(directory).map_err(|error| {
        format!(
            "openrouter credential directory could not be created: {}; {error}",
            directory.display()
        )
    })?;
    set_directory_private(directory)?;

    let contents = serde_json::to_string_pretty(&json!({ "openrouter_api_key": api_key }))
        .map_err(|error| format!("openrouter credential could not be encoded: {error}"))?;
    write_private_file(&path, format!("{contents}\n").as_bytes())?;
    Ok(path)
}

fn delete_api_key(home: &Path) -> Result<Option<PathBuf>, String> {
    let path = auth_path(home);
    match fs::remove_file(&path) {
        Ok(()) => Ok(Some(path)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "openrouter credential file could not be removed: {}; {error}",
            path.display()
        )),
    }
}

fn parse_auth_json(contents: &str) -> Option<String> {
    serde_json::from_str::<Value>(contents)
        .ok()?
        .get("openrouter_api_key")
        .and_then(Value::as_str)
        .map(str::to_string)
        .and_then(non_empty)
}

#[cfg(unix)]
fn set_directory_private(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)
        .map_err(|error| format!("could not inspect {}; {error}", path.display()))?
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions)
        .map_err(|error| format!("could not set permissions on {}; {error}", path.display()))
}

#[cfg(not(unix))]
fn set_directory_private(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn write_private_file(path: &Path, contents: &[u8]) -> Result<(), String> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let directory = path
        .parent()
        .ok_or_else(|| format!("invalid auth path: {}", path.display()))?;
    let temp_path = directory.join(format!(
        ".auth.json.tmp-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&temp_path)
        .map_err(|error| {
            format!(
                "openrouter credential file could not be written: {}; {error}",
                temp_path.display()
            )
        })?;
    if let Err(error) = file.write_all(contents) {
        let _ = fs::remove_file(&temp_path);
        return Err(format!(
            "openrouter credential file could not be written: {}; {error}",
            temp_path.display()
        ));
    }
    if let Err(error) = file.sync_all() {
        let _ = fs::remove_file(&temp_path);
        return Err(format!(
            "openrouter credential file could not be written: {}; {error}",
            temp_path.display()
        ));
    }

    let mut permissions = file
        .metadata()
        .map_err(|error| format!("could not inspect {}; {error}", temp_path.display()))?
        .permissions();
    permissions.set_mode(0o600);
    if let Err(error) = file.set_permissions(permissions) {
        let _ = fs::remove_file(&temp_path);
        return Err(format!(
            "could not set permissions on {}; {error}",
            temp_path.display()
        ));
    }
    drop(file);

    fs::rename(&temp_path, path).map_err(|error| {
        let _ = fs::remove_file(&temp_path);
        format!(
            "openrouter credential file could not be written: {}; {error}",
            path.display()
        )
    })
}

#[cfg(not(unix))]
fn write_private_file(path: &Path, contents: &[u8]) -> Result<(), String> {
    fs::write(path, contents).map_err(|error| {
        format!(
            "openrouter credential file could not be written: {}; {error}",
            path.display()
        )
    })
}

#[cfg(unix)]
fn credential_permissions_are_private(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;

    metadata.permissions().mode() & 0o777 == 0o600
}

#[cfg(not(unix))]
fn credential_permissions_are_private(_metadata: &fs::Metadata) -> bool {
    true
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum OpenRouterError {
    MissingApiKey,
    Http(String),
    Request(String),
    Timeout(String),
    Authentication(String),
    RateLimit {
        message: String,
        retry_after: Option<String>,
    },
    Status {
        code: u16,
        message: String,
        retry_after: Option<String>,
    },
    Provider(String),
    MalformedResponse(String),
}

impl fmt::Display for OpenRouterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingApiKey => write!(formatter, "openrouter API key is not configured"),
            Self::Http(message) => write!(formatter, "openrouter HTTP error: {message}"),
            Self::Request(message) => write!(formatter, "openrouter request failed: {message}"),
            Self::Timeout(stage) => write!(formatter, "openrouter request timed out: {stage}"),
            Self::Authentication(message) => {
                write!(formatter, "openrouter authentication failed: {message}")
            }
            Self::RateLimit {
                message,
                retry_after,
            } => {
                write!(formatter, "openrouter rate limited: {message}")?;
                if let Some(retry_after) = retry_after {
                    write!(formatter, "; retry after {retry_after}")?;
                }
                Ok(())
            }
            Self::Status {
                code,
                message,
                retry_after,
            } => {
                write!(formatter, "openrouter returned HTTP {code}: {message}")?;
                if let Some(retry_after) = retry_after {
                    write!(formatter, "; retry after {retry_after}")?;
                }
                Ok(())
            }
            Self::Provider(message) => write!(formatter, "openrouter returned an error: {message}"),
            Self::MalformedResponse(message) => {
                write!(
                    formatter,
                    "openrouter returned malformed response: {message}"
                )
            }
        }
    }
}

impl std::error::Error for OpenRouterError {}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LlmResponse {
    pub(crate) answer: String,
    pub(crate) diagnostics: Option<LlmDiagnostics>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LlmDiagnostics {
    pub(crate) model: Option<String>,
    pub(crate) provider: Option<String>,
    pub(crate) latency_ms: Option<u64>,
    pub(crate) usage: Option<TokenUsage>,
    pub(crate) transport: Option<TransportDiagnostics>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TokenUsage {
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
    pub(crate) reasoning_tokens: Option<u64>,
    pub(crate) cached_prompt_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
    pub(crate) estimated_cost: Option<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TransportDiagnostics {
    pub(crate) request_creation_ms: u64,
    pub(crate) upload_to_headers_ms: u64,
    pub(crate) first_body_byte_ms: Option<u64>,
    pub(crate) body_completion_ms: u64,
    pub(crate) total_request_ms: u64,
    pub(crate) response_body_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TransportConfig {
    connect_timeout: Duration,
    request_write_timeout: Duration,
    response_header_timeout: Duration,
    response_body_timeout: Duration,
}

impl TransportConfig {
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        let legacy_timeout = lookup_timeout(&mut lookup, "RECALL_OPENROUTER_TIMEOUT_SECS");
        let default_timeout = legacy_timeout.unwrap_or(DEFAULT_TIMEOUT_SECS);
        let default_response_header_timeout =
            legacy_timeout.unwrap_or(DEFAULT_RESPONSE_HEADER_TIMEOUT_SECS);
        Self {
            connect_timeout: Duration::from_secs(
                lookup_timeout(&mut lookup, "RECALL_OPENROUTER_CONNECT_TIMEOUT_SECS")
                    .unwrap_or(default_timeout),
            ),
            request_write_timeout: Duration::from_secs(
                lookup_timeout(&mut lookup, "RECALL_OPENROUTER_REQUEST_WRITE_TIMEOUT_SECS")
                    .unwrap_or(default_timeout),
            ),
            response_header_timeout: Duration::from_secs(
                lookup_timeout(
                    &mut lookup,
                    "RECALL_OPENROUTER_RESPONSE_HEADER_TIMEOUT_SECS",
                )
                .unwrap_or(default_response_header_timeout),
            ),
            response_body_timeout: Duration::from_secs(
                lookup_timeout(&mut lookup, "RECALL_OPENROUTER_RESPONSE_BODY_TIMEOUT_SECS")
                    .unwrap_or(default_timeout),
            ),
        }
    }

    #[cfg(test)]
    fn with_all_timeouts(seconds: u64) -> Self {
        let timeout = Duration::from_secs(seconds);
        Self {
            connect_timeout: timeout,
            request_write_timeout: timeout,
            response_header_timeout: timeout,
            response_body_timeout: timeout,
        }
    }
}

fn lookup_timeout(lookup: &mut impl FnMut(&str) -> Option<String>, key: &str) -> Option<u64> {
    lookup(key)
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TransportMode {
    Buffered,
    #[allow(dead_code)]
    Streaming,
}

impl TransportMode {
    fn request_streaming(self) -> bool {
        matches!(self, Self::Streaming)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TransportRequest {
    endpoint: String,
    api_key: String,
    body: Value,
    include_router_metadata: bool,
    mode: TransportMode,
    collect_diagnostics: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TransportResponse {
    status: u16,
    retry_after: Option<String>,
    body: String,
    diagnostics: Option<TransportDiagnostics>,
}

#[derive(Clone, Debug)]
struct Transport {
    config: TransportConfig,
}

impl Transport {
    fn new(config: TransportConfig) -> Self {
        Self { config }
    }

    fn send(&self, request: TransportRequest) -> Result<TransportResponse, OpenRouterError> {
        match request.mode {
            TransportMode::Buffered => self.send_buffered(request),
            TransportMode::Streaming => self.send_streaming(request),
        }
    }

    fn send_buffered(
        &self,
        request: TransportRequest,
    ) -> Result<TransportResponse, OpenRouterError> {
        let total_started = Instant::now();
        let request_started = Instant::now();
        let agent_config = ureq::Agent::config_builder()
            .timeout_connect(Some(self.config.connect_timeout))
            .timeout_send_request(Some(self.config.request_write_timeout))
            .timeout_send_body(Some(self.config.request_write_timeout))
            .timeout_recv_response(Some(self.config.response_header_timeout))
            .timeout_recv_body(Some(self.config.response_body_timeout))
            .http_status_as_error(false)
            .build();
        let agent: ureq::Agent = agent_config.into();
        let request_creation_ms = elapsed_ms(request_started);
        let http_request = agent
            .post(&request.endpoint)
            .header("Authorization", format!("Bearer {}", request.api_key))
            .header("Content-Type", "application/json");
        let http_request = if request.include_router_metadata {
            http_request.header("X-OpenRouter-Metadata", "enabled")
        } else {
            http_request
        };

        let upload_started = Instant::now();
        let mut response = http_request
            .send_json(&request.body)
            .map_err(map_ureq_error)?;
        let upload_to_headers_ms = elapsed_ms(upload_started);
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("Retry-After")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);

        let body_started = Instant::now();
        let (body, first_body_byte_ms) = read_response_body(response.body_mut())?;
        let body_completion_ms = elapsed_ms(body_started);
        let response_body_bytes = body.len();
        let diagnostics = request.collect_diagnostics.then(|| TransportDiagnostics {
            request_creation_ms,
            upload_to_headers_ms,
            first_body_byte_ms,
            body_completion_ms,
            total_request_ms: elapsed_ms(total_started),
            response_body_bytes,
        });

        Ok(TransportResponse {
            status,
            retry_after,
            body,
            diagnostics,
        })
    }

    fn send_streaming(
        &self,
        _request: TransportRequest,
    ) -> Result<TransportResponse, OpenRouterError> {
        Err(OpenRouterError::Request(
            "streaming transport is not enabled".to_string(),
        ))
    }
}

fn read_response_body(body: &mut ureq::Body) -> Result<(String, Option<u64>), OpenRouterError> {
    let mut reader = body.with_config().limit(MAX_RESPONSE_BODY_BYTES).reader();
    let first_started = Instant::now();
    let mut bytes = Vec::new();
    let mut first = [0_u8; 1];
    let first_body_byte_ms = match reader.read(&mut first) {
        Ok(0) => None,
        Ok(count) => {
            bytes.extend_from_slice(&first[..count]);
            Some(elapsed_ms(first_started))
        }
        Err(error) => return Err(map_ureq_error(error.into())),
    };
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| map_ureq_error(error.into()))?;
    String::from_utf8(bytes)
        .map(|body| (body, first_body_byte_ms))
        .map_err(|error| OpenRouterError::MalformedResponse(format!("invalid UTF-8: {error}")))
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[derive(Clone, Debug)]
struct OpenRouterClient {
    config: OpenRouterConfig,
    transport: Transport,
}

impl OpenRouterClient {
    fn new(config: &OpenRouterConfig) -> Self {
        Self {
            config: config.clone(),
            transport: Transport::new(config.transport),
        }
    }

    fn send_prompt(
        &self,
        prompt: &str,
        include_diagnostics: bool,
    ) -> Result<LlmResponse, OpenRouterError> {
        let api_key = self
            .config
            .api_key
            .clone()
            .ok_or(OpenRouterError::MissingApiKey)?;
        let mode = TransportMode::Buffered;
        let request_body = build_request_body(&self.config.model, prompt, mode);
        let response = self.transport.send(TransportRequest {
            endpoint: self.config.endpoint.clone(),
            api_key,
            body: request_body,
            include_router_metadata: include_diagnostics,
            mode,
            collect_diagnostics: include_diagnostics,
        })?;

        parse_response(
            response.status,
            response.retry_after.as_deref(),
            &response.body,
            response.diagnostics,
        )
    }
}

pub(crate) fn build_request_body(model: &str, prompt: &str, mode: TransportMode) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
    });
    if mode.request_streaming() {
        body["stream"] = json!(true);
    }
    body
}

pub(crate) fn send_prompt(
    config: &OpenRouterConfig,
    prompt: &str,
    include_diagnostics: bool,
) -> Result<LlmResponse, OpenRouterError> {
    OpenRouterClient::new(config).send_prompt(prompt, include_diagnostics)
}

fn map_ureq_error(error: ureq::Error) -> OpenRouterError {
    match error {
        ureq::Error::Timeout(timeout) => OpenRouterError::Timeout(timeout.to_string()),
        ureq::Error::Http(error) => OpenRouterError::Http(error.to_string()),
        ureq::Error::StatusCode(code) => OpenRouterError::Status {
            code,
            message: status_default_message(code).to_string(),
            retry_after: None,
        },
        other => OpenRouterError::Request(other.to_string()),
    }
}

fn parse_response(
    status: u16,
    retry_after: Option<&str>,
    body: &str,
    transport_diagnostics: Option<TransportDiagnostics>,
) -> Result<LlmResponse, OpenRouterError> {
    let value = serde_json::from_str::<Value>(body).map_err(|error| {
        if (200..300).contains(&status) {
            OpenRouterError::MalformedResponse(format!("invalid JSON: {error}"))
        } else {
            OpenRouterError::Status {
                code: status,
                message: format!("{}; response was not JSON", status_default_message(status)),
                retry_after: retry_after.map(str::to_string),
            }
        }
    })?;

    if !(200..300).contains(&status) {
        return Err(status_error(
            status,
            error_message(&value).unwrap_or_else(|| status_default_message(status).to_string()),
            retry_after,
        ));
    }

    if let Some(message) = error_message(&value) {
        return Err(OpenRouterError::Provider(message));
    }

    value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(|content| LlmResponse {
            answer: content.to_string(),
            diagnostics: parse_diagnostics(&value, transport_diagnostics),
        })
        .ok_or_else(|| {
            OpenRouterError::MalformedResponse("missing choices[0].message.content".to_string())
        })
}

fn parse_diagnostics(
    value: &Value,
    transport_diagnostics: Option<TransportDiagnostics>,
) -> Option<LlmDiagnostics> {
    let usage = value.get("usage");
    let metadata = value.get("openrouter_metadata");
    let selected_endpoint = metadata
        .and_then(|metadata| metadata.pointer("/endpoints/available"))
        .and_then(Value::as_array)
        .and_then(|endpoints| {
            endpoints
                .iter()
                .find(|endpoint| endpoint.get("selected").and_then(Value::as_bool) == Some(true))
        });
    let successful_attempt = metadata
        .and_then(|metadata| metadata.get("attempts"))
        .and_then(Value::as_array)
        .and_then(|attempts| {
            attempts.iter().find(|attempt| {
                attempt
                    .get("status")
                    .and_then(Value::as_u64)
                    .is_some_and(|status| (200..300).contains(&status))
            })
        });

    let diagnostics = LlmDiagnostics {
        model: selected_endpoint
            .and_then(|endpoint| string_field(endpoint, "model"))
            .or_else(|| successful_attempt.and_then(|attempt| string_field(attempt, "model")))
            .or_else(|| string_field(value, "model")),
        provider: selected_endpoint
            .and_then(|endpoint| string_field(endpoint, "provider"))
            .or_else(|| successful_attempt.and_then(|attempt| string_field(attempt, "provider"))),
        latency_ms: metadata.and_then(parse_latency_ms),
        usage: usage.and_then(parse_usage),
        transport: transport_diagnostics,
    };

    diagnostics.has_fields().then_some(diagnostics)
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let usage = TokenUsage {
        prompt_tokens: u64_field(usage, "prompt_tokens"),
        completion_tokens: u64_field(usage, "completion_tokens"),
        reasoning_tokens: usage
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64),
        cached_prompt_tokens: usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64),
        total_tokens: u64_field(usage, "total_tokens"),
        estimated_cost: f64_field(usage, "cost"),
    };

    usage.has_fields().then_some(usage)
}

impl LlmDiagnostics {
    fn has_fields(&self) -> bool {
        self.model.is_some()
            || self.provider.is_some()
            || self.latency_ms.is_some()
            || self.usage.is_some()
            || self.transport.is_some()
    }
}

impl TokenUsage {
    fn has_fields(&self) -> bool {
        self.prompt_tokens.is_some()
            || self.completion_tokens.is_some()
            || self.reasoning_tokens.is_some()
            || self.cached_prompt_tokens.is_some()
            || self.total_tokens.is_some()
            || self.estimated_cost.is_some()
    }
}

fn parse_latency_ms(metadata: &Value) -> Option<u64> {
    ["latency", "latency_ms", "duration_ms", "elapsed_ms"]
        .into_iter()
        .find_map(|field| u64_field(metadata, field))
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_string)
}

fn u64_field(value: &Value, field: &str) -> Option<u64> {
    value.get(field).and_then(Value::as_u64)
}

fn f64_field(value: &Value, field: &str) -> Option<f64> {
    match value.get(field)? {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse::<f64>().ok(),
        _ => None,
    }
}

fn status_error(status: u16, message: String, retry_after: Option<&str>) -> OpenRouterError {
    match status {
        401 => OpenRouterError::Authentication(message),
        429 => OpenRouterError::RateLimit {
            message,
            retry_after: retry_after.map(str::to_string),
        },
        _ => OpenRouterError::Status {
            code: status,
            message,
            retry_after: retry_after.map(str::to_string),
        },
    }
}

fn error_message(value: &Value) -> Option<String> {
    value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| value.get("error").and_then(Value::as_str))
        .map(str::to_string)
}

fn status_default_message(status: u16) -> &'static str {
    match status {
        400 => "bad request",
        401 => "invalid or missing API key",
        402 => "insufficient credits",
        403 => "forbidden",
        408 => "upstream request timed out",
        429 => "rate limited",
        502 => "model or provider unavailable",
        503 => "service unavailable",
        _ => "request failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn default_model_literal_requires_intentional_test_update() {
        assert_eq!(DEFAULT_MODEL, "google/gemini-2.5-flash-lite");
    }

    #[test]
    fn config_loads_defaults_without_api_key() {
        let config = OpenRouterConfig::from_lookup(|_| None);

        assert_eq!(config.api_key, None);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(
            config.transport,
            TransportConfig {
                connect_timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
                request_write_timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
                response_header_timeout: Duration::from_secs(DEFAULT_RESPONSE_HEADER_TIMEOUT_SECS),
                response_body_timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            }
        );
        assert!(!config.is_configured());
    }

    #[test]
    fn config_legacy_timeout_remains_fallback_for_all_transport_stages() {
        let config = OpenRouterConfig::from_lookup(|key| match key {
            "RECALL_OPENROUTER_TIMEOUT_SECS" => Some("5".to_string()),
            _ => None,
        });

        assert_eq!(config.transport, TransportConfig::with_all_timeouts(5));
    }

    #[test]
    fn config_trims_values_and_parses_transport_timeouts() {
        let vars = BTreeMap::from([
            ("OPENROUTER_API_KEY", " key "),
            ("RECALL_OPENROUTER_MODEL", " model "),
            ("RECALL_OPENROUTER_ENDPOINT", " http://localhost "),
            ("RECALL_OPENROUTER_TIMEOUT_SECS", "5"),
            ("RECALL_OPENROUTER_CONNECT_TIMEOUT_SECS", "2"),
            ("RECALL_OPENROUTER_REQUEST_WRITE_TIMEOUT_SECS", "3"),
            ("RECALL_OPENROUTER_RESPONSE_HEADER_TIMEOUT_SECS", "4"),
            ("RECALL_OPENROUTER_RESPONSE_BODY_TIMEOUT_SECS", "6"),
        ]);
        let config =
            OpenRouterConfig::from_lookup(|key| vars.get(key).map(|value| value.to_string()));

        assert_eq!(config.api_key.as_deref(), Some("key"));
        assert_eq!(config.model, "model");
        assert_eq!(config.endpoint, "http://localhost");
        assert_eq!(
            config.transport,
            TransportConfig {
                connect_timeout: Duration::from_secs(2),
                request_write_timeout: Duration::from_secs(3),
                response_header_timeout: Duration::from_secs(4),
                response_body_timeout: Duration::from_secs(6),
            }
        );
        assert!(config.is_configured());
    }

    #[test]
    fn config_loads_stored_api_key_when_environment_is_absent() {
        let home = temp_home("stored-api-key");
        write_stored_key(&home, " file-key ");

        let config = OpenRouterConfig::from_lookup_and_home(|_| None, Some(&home));

        assert_eq!(config.api_key.as_deref(), Some("file-key"));
        assert!(config.is_configured());
    }

    #[test]
    fn config_ignores_model_like_fields_in_stored_auth_json() {
        let home = temp_home("stored-model-fields");
        write_auth_json(
            &home,
            r#"{
  "openrouter_api_key": "file-key",
  "openrouter_model": "stored/openrouter-model",
  "model": "stored/model"
}
"#,
        );

        let default_config = OpenRouterConfig::from_lookup_and_home(|_| None, Some(&home));
        let override_config = OpenRouterConfig::from_lookup_and_home(
            |key| (key == "RECALL_OPENROUTER_MODEL").then(|| "env/model".to_string()),
            Some(&home),
        );

        assert_eq!(default_config.api_key.as_deref(), Some("file-key"));
        assert_eq!(default_config.model, DEFAULT_MODEL);
        assert_eq!(override_config.api_key.as_deref(), Some("file-key"));
        assert_eq!(override_config.model, "env/model");
    }

    #[test]
    fn config_environment_api_key_overrides_stored_api_key() {
        let home = temp_home("stored-api-key-override");
        write_stored_key(&home, "file-key");

        let config = OpenRouterConfig::from_lookup_and_home(
            |key| (key == "OPENROUTER_API_KEY").then(|| "env-key".to_string()),
            Some(&home),
        );

        assert_eq!(config.api_key.as_deref(), Some("env-key"));
    }

    #[test]
    fn config_reports_stored_api_key_with_insecure_permissions() {
        let home = temp_home("stored-api-key-insecure");
        let path = write_stored_key(&home, "file-key");
        set_mode(&path, 0o644);

        let config = OpenRouterConfig::from_lookup_and_home(|_| None, Some(&home));

        assert_eq!(config.api_key, None);
        assert!(!config.is_configured());
        assert!(config
            .credential_error()
            .unwrap()
            .contains("expected mode 0600"));
    }

    #[test]
    fn config_reports_malformed_stored_api_key() {
        let missing_home = temp_home("stored-api-key-missing");
        let missing_config = OpenRouterConfig::from_lookup_and_home(|_| None, Some(&missing_home));
        assert_eq!(missing_config.api_key, None);
        assert_eq!(missing_config.credential_error(), None);

        let malformed_home = temp_home("stored-api-key-malformed");
        write_auth_json(&malformed_home, "{}\n");
        let malformed_config =
            OpenRouterConfig::from_lookup_and_home(|_| None, Some(&malformed_home));
        assert_eq!(malformed_config.api_key, None);
        assert!(malformed_config
            .credential_error()
            .unwrap()
            .contains("expected openrouter_api_key"));
    }

    #[test]
    fn auth_json_parser_loads_key_without_exposing_it() {
        assert_eq!(
            parse_auth_json(r#"{"openrouter_api_key":"stored-key"}"#).as_deref(),
            Some("stored-key")
        );
        assert_eq!(parse_auth_json(r#"{"openrouter_api_key":""}"#), None);
        assert_eq!(parse_auth_json("not json"), None);
    }

    #[test]
    fn auth_status_reports_environment_precedence_without_reading_stored_key() {
        let home = temp_home("status-env-override");
        write_stored_key(&home, "file-key");

        let status = auth_status_from_lookup_and_home(
            |key| (key == "OPENROUTER_API_KEY").then(|| "env-key".to_string()),
            &home,
        );

        assert_eq!(status, AuthStatus::EnvironmentOverride);
    }

    #[test]
    fn auth_status_reports_stored_missing_and_malformed_credentials() {
        let stored_home = temp_home("status-stored");
        let stored_path = write_stored_key(&stored_home, "file-key");
        assert_eq!(
            auth_status_from_lookup_and_home(|_| None, &stored_home),
            AuthStatus::Stored { path: stored_path }
        );

        let missing_home = temp_home("status-missing");
        assert_eq!(
            auth_status_from_lookup_and_home(|_| None, &missing_home),
            AuthStatus::NotConfigured {
                path: auth_path(&missing_home)
            }
        );

        let malformed_home = temp_home("status-malformed");
        write_auth_json(&malformed_home, "not json");
        let status = auth_status_from_lookup_and_home(|_| None, &malformed_home);
        assert!(
            matches!(status, AuthStatus::Error(error) if error.contains("expected openrouter_api_key"))
        );
    }

    #[test]
    fn save_api_key_writes_private_user_state_file() {
        let home = temp_home("save-private");

        let path = save_api_key(&home, " stored-key ").unwrap();

        let record = serde_json::from_str::<Value>(&fs::read_to_string(&path).unwrap()).unwrap();
        let object = record.as_object().unwrap();
        assert_eq!(object.len(), 1);
        assert_eq!(
            object.get("openrouter_api_key").and_then(Value::as_str),
            Some("stored-key")
        );
        assert!(!object.contains_key("openrouter_model"));
        assert!(!object.contains_key("model"));
        assert_eq!(path, auth_path(&home));
        assert_eq!(
            load_auth_record(&path).unwrap().as_deref(),
            Some("stored-key")
        );
        assert!(credential_permissions_are_private(
            &fs::metadata(&path).unwrap()
        ));
        assert_eq!(file_mode(&path), 0o600);
        assert_eq!(file_mode(path.parent().unwrap()), 0o700);
    }

    #[test]
    fn save_api_key_replaces_existing_file_atomically_and_privately() {
        let home = temp_home("save-replace");
        let path = write_stored_key(&home, "old-key");
        set_mode(&path, 0o644);

        let saved_path = save_api_key(&home, "new-key").unwrap();

        assert_eq!(saved_path, path);
        assert_eq!(load_auth_record(&path).unwrap().as_deref(), Some("new-key"));
        assert_eq!(file_mode(&path), 0o600);
    }

    #[test]
    fn delete_api_key_removes_stored_credential_only_when_present() {
        let home = temp_home("delete-key");
        let path = write_stored_key(&home, "file-key");

        assert_eq!(delete_api_key(&home).unwrap(), Some(path.clone()));
        assert!(!path.exists());
        assert_eq!(delete_api_key(&home).unwrap(), None);
    }

    #[test]
    fn config_reports_whether_endpoint_is_default() {
        let default_config = OpenRouterConfig::from_lookup(|_| None);
        let custom_config = OpenRouterConfig::from_lookup(|key| match key {
            "RECALL_OPENROUTER_ENDPOINT" => Some("https://example.test/openrouter".to_string()),
            _ => None,
        });

        assert!(default_config.uses_default_endpoint());
        assert!(!custom_config.uses_default_endpoint());
        assert_eq!(custom_config.endpoint(), "https://example.test/openrouter");
    }

    #[test]
    fn request_body_uses_prompt_as_user_message() {
        let body = build_request_body("test-model", "Prompt text", TransportMode::Buffered);

        assert_eq!(body["model"], "test-model");
        assert_eq!(body["temperature"], json!(0.2));
        assert_eq!(body.get("stream"), None);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Prompt text");
    }

    #[test]
    fn streaming_request_body_enables_streaming() {
        let body = build_request_body("test-model", "Prompt text", TransportMode::Streaming);

        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn parses_successful_response_content() {
        let answer = parse_response(
            200,
            None,
            r#"{"choices":[{"message":{"content":"Answer with git:abc"}}]}"#,
            None,
        )
        .unwrap();

        assert_eq!(answer.answer, "Answer with git:abc");
        assert_eq!(answer.diagnostics, None);
    }

    #[test]
    fn parses_response_metadata_when_present() {
        let answer = parse_response(
            200,
            None,
            r#"{
                "model": "requested/model",
                "choices": [{"message": {"content": "Answer"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {
                        "cached_tokens": 80
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 5
                    },
                    "total_tokens": 120,
                    "cost": 0.0012
                },
                "openrouter_metadata": {
                    "latency_ms": 345,
                    "endpoints": {
                        "available": [
                            {
                                "provider": "OpenAI",
                                "model": "openai/gpt-4o-mini",
                                "selected": true
                            }
                        ]
                    }
                }
            }"#,
            None,
        )
        .unwrap();

        assert_eq!(answer.answer, "Answer");
        assert_eq!(
            answer.diagnostics,
            Some(LlmDiagnostics {
                model: Some("openai/gpt-4o-mini".to_string()),
                provider: Some("OpenAI".to_string()),
                latency_ms: Some(345),
                usage: Some(TokenUsage {
                    prompt_tokens: Some(100),
                    completion_tokens: Some(20),
                    reasoning_tokens: Some(5),
                    cached_prompt_tokens: Some(80),
                    total_tokens: Some(120),
                    estimated_cost: Some(0.0012),
                }),
                transport: None,
            })
        );
    }

    #[test]
    fn parses_partial_response_metadata() {
        let answer = parse_response(
            200,
            None,
            r#"{
                "model": "fallback/model",
                "choices": [{"message": {"content": "Answer"}}],
                "usage": {
                    "prompt_tokens": 11,
                    "total_tokens": 15
                },
                "openrouter_metadata": {
                    "attempts": [
                        {
                            "provider": "Anthropic",
                            "model": "anthropic/claude-sonnet",
                            "status": 200
                        }
                    ]
                }
            }"#,
            None,
        )
        .unwrap();

        assert_eq!(
            answer.diagnostics,
            Some(LlmDiagnostics {
                model: Some("anthropic/claude-sonnet".to_string()),
                provider: Some("Anthropic".to_string()),
                latency_ms: None,
                usage: Some(TokenUsage {
                    prompt_tokens: Some(11),
                    completion_tokens: None,
                    reasoning_tokens: None,
                    cached_prompt_tokens: None,
                    total_tokens: Some(15),
                    estimated_cost: None,
                }),
                transport: None,
            })
        );
    }

    #[test]
    fn parses_json_error_body_for_status() {
        let error = parse_response(
            402,
            None,
            r#"{"error":{"message":"No credits remaining"}}"#,
            None,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OpenRouterError::Status {
                code: 402,
                message: "No credits remaining".to_string(),
                retry_after: None,
            }
        );
    }

    #[test]
    fn parses_authentication_error() {
        let error = parse_response(
            401,
            None,
            r#"{"error":{"message":"Invalid API key"}}"#,
            None,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OpenRouterError::Authentication("Invalid API key".to_string())
        );
        assert_eq!(
            error.to_string(),
            "openrouter authentication failed: Invalid API key"
        );
    }

    #[test]
    fn parses_rate_limit_retry_after() {
        let error = parse_response(
            429,
            Some("10"),
            r#"{"error":{"message":"Slow down"}}"#,
            None,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OpenRouterError::RateLimit {
                message: "Slow down".to_string(),
                retry_after: Some("10".to_string()),
            }
        );
        assert_eq!(
            error.to_string(),
            "openrouter rate limited: Slow down; retry after 10"
        );
    }

    #[test]
    fn parses_top_level_error_on_success_status() {
        let error = parse_response(
            200,
            None,
            r#"{"error":{"message":"Provider failed"}}"#,
            None,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OpenRouterError::Provider("Provider failed".to_string())
        );
    }

    #[test]
    fn rejects_malformed_success_response() {
        let error = parse_response(200, None, r#"{"choices":[]}"#, None).unwrap_err();

        assert_eq!(
            error,
            OpenRouterError::MalformedResponse("missing choices[0].message.content".to_string(),)
        );
    }

    #[test]
    fn rejects_invalid_json_success_response() {
        let error = parse_response(200, None, "not json", None).unwrap_err();

        assert!(matches!(error, OpenRouterError::MalformedResponse(_)));
        assert!(error.to_string().contains("invalid JSON"));
    }

    #[test]
    fn sends_prompt_to_local_http_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (sender, receiver) = mpsc::channel();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            sender.send(request).unwrap();
            let response_body = r#"{"choices":[{"message":{"content":"server answer"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let config = OpenRouterConfig {
            api_key: Some("test-key".to_string()),
            credential_error: None,
            model: "test-model".to_string(),
            endpoint,
            transport: TransportConfig::with_all_timeouts(5),
        };

        let answer = send_prompt(&config, "Prompt text", false).unwrap();
        handle.join().unwrap();
        let request = receiver.recv().unwrap();

        assert_eq!(answer.answer, "server answer");
        let request_lower = request.to_lowercase();
        assert!(request_lower.contains("authorization: bearer test-key"));
        assert!(request_lower.contains("content-type: application/json"));
        assert!(!request_lower.contains("x-openrouter-metadata"));
        let request_body: Value =
            serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(request_body["model"], "test-model");
        assert_eq!(request_body["messages"][0]["content"], "Prompt text");
        assert_eq!(request_body.get("stream"), None);
    }

    #[test]
    fn send_prompt_requests_router_metadata_when_diagnostics_are_enabled() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (sender, receiver) = mpsc::channel();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            sender.send(request).unwrap();
            let response_body = r#"{"choices":[{"message":{"content":"server answer"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let config = OpenRouterConfig {
            api_key: Some("test-key".to_string()),
            credential_error: None,
            model: "test-model".to_string(),
            endpoint,
            transport: TransportConfig::with_all_timeouts(5),
        };

        let answer = send_prompt(&config, "Prompt text", true).unwrap();
        handle.join().unwrap();
        let request = receiver.recv().unwrap();

        assert_eq!(answer.answer, "server answer");
        assert!(answer.diagnostics.unwrap().transport.is_some());
        assert!(request
            .to_lowercase()
            .contains("x-openrouter-metadata: enabled"));
    }

    #[test]
    fn send_prompt_times_out_waiting_for_response_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (sender, receiver) = mpsc::channel();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            sender.send(request).unwrap();
            thread::sleep(Duration::from_millis(200));
            let response_body = r#"{"choices":[{"message":{"content":"late answer"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(response.as_bytes());
        });

        let config = OpenRouterConfig {
            api_key: Some("test-key".to_string()),
            credential_error: None,
            model: "test-model".to_string(),
            endpoint,
            transport: TransportConfig {
                connect_timeout: Duration::from_secs(5),
                request_write_timeout: Duration::from_secs(5),
                response_header_timeout: Duration::from_millis(50),
                response_body_timeout: Duration::from_secs(5),
            },
        };

        let error = send_prompt(&config, "Prompt text", false).unwrap_err();
        handle.join().unwrap();
        let request = receiver.recv().unwrap();

        assert!(request.contains("Prompt text"));
        assert_eq!(
            error,
            OpenRouterError::Timeout("receive response".to_string())
        );
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buffer = [0; 1024];
        let mut header_end = None;

        while header_end.is_none() {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&buffer[..count]);
            header_end = find_header_end(&bytes);
        }

        let header_end = header_end.unwrap();
        let headers = String::from_utf8_lossy(&bytes[..header_end]).into_owned();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("Content-Length")
                    .then_some(value.trim())
            })
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let chunked = headers.lines().any(|line| {
            let Some((name, value)) = line.split_once(':') else {
                return false;
            };
            name.eq_ignore_ascii_case("Transfer-Encoding")
                && value.trim().eq_ignore_ascii_case("chunked")
        });

        if chunked {
            while !bytes.ends_with(b"0\r\n\r\n") {
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
            }

            let body = decode_chunked_body(&bytes[header_end + 4..]);
            return format!("{headers}\r\n\r\n{}", String::from_utf8_lossy(&body));
        } else {
            while bytes.len() < header_end + 4 + content_length {
                let count = stream.read(&mut buffer).unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn find_header_end(bytes: &[u8]) -> Option<usize> {
        bytes.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn decode_chunked_body(bytes: &[u8]) -> Vec<u8> {
        let mut decoded = Vec::new();
        let mut index = 0;

        loop {
            let size_end = bytes[index..]
                .windows(2)
                .position(|window| window == b"\r\n")
                .map(|position| index + position)
                .unwrap();
            let size_text = std::str::from_utf8(&bytes[index..size_end]).unwrap();
            let size = usize::from_str_radix(size_text, 16).unwrap();
            if size == 0 {
                break;
            }

            let chunk_start = size_end + 2;
            let chunk_end = chunk_start + size;
            decoded.extend_from_slice(&bytes[chunk_start..chunk_end]);
            index = chunk_end + 2;
        }

        decoded
    }

    fn temp_home(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("recall-openrouter-{label}-{unique}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_stored_key(home: &Path, key: &str) -> PathBuf {
        write_auth_json(home, &format!("{{\"openrouter_api_key\":\"{key}\"}}\n"))
    }

    fn write_auth_json(home: &Path, contents: &str) -> PathBuf {
        let path = home.join(DEFAULT_AUTH_PATH);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, contents).unwrap();
        set_mode(&path, 0o600);
        path
    }

    #[cfg(unix)]
    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(mode);
        fs::set_permissions(path, permissions).unwrap();
    }

    #[cfg(not(unix))]
    fn set_mode(_path: &Path, _mode: u32) {}

    #[cfg(unix)]
    fn file_mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;

        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[cfg(not(unix))]
    fn file_mode(_path: &Path) -> u32 {
        0o600
    }
}
