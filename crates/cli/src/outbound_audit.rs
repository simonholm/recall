//! Local outbound prompt audit records for configured model requests.

use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde_json::json;

const ENV_LOG_DIR: &str = "RECALL_OUTBOUND_LOG_DIR";
const FILE_PREFIX: &str = "recall-outbound-";
const FILE_SUFFIX: &str = ".json";
const RETAIN_RECORDS: usize = 20;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OutboundAuditConfig {
    log_dir: PathBuf,
}

impl OutboundAuditConfig {
    pub(crate) fn from_env() -> Option<Self> {
        std::env::var(ENV_LOG_DIR)
            .ok()
            .and_then(|value| Self::from_dir(value.trim()))
    }

    pub(crate) fn from_dir(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();
        (!path.as_os_str().is_empty()).then(|| Self {
            log_dir: path.to_path_buf(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct OutboundAuditError {
    message: String,
}

impl fmt::Display for OutboundAuditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "outbound audit failed: {}", self.message)
    }
}

impl std::error::Error for OutboundAuditError {}

pub(crate) fn write_outbound_prompt(
    config: &OutboundAuditConfig,
    model: &str,
    question: &str,
    prompt: &str,
) -> Result<PathBuf, OutboundAuditError> {
    let timestamp = Utc::now();
    let timestamp_text = timestamp.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
    let record = json!({
        "timestamp": timestamp_text,
        "selected_model": model,
        "original_question": question,
        "prompt_byte_length": prompt.len(),
        "outbound_prompt": prompt,
    });
    let bytes = serde_json::to_vec_pretty(&record).map_err(|error| OutboundAuditError {
        message: format!("could not encode audit record: {error}"),
    })?;

    fs::create_dir_all(&config.log_dir).map_err(|error| OutboundAuditError {
        message: format!(
            "could not create audit log directory {}: {error}",
            config.log_dir.display()
        ),
    })?;

    let path = unique_record_path(&config.log_dir, &timestamp_text);
    let mut file = create_private_file(&path).map_err(|error| OutboundAuditError {
        message: format!("could not create audit record {}: {error}", path.display()),
    })?;
    file.write_all(&bytes).map_err(|error| OutboundAuditError {
        message: format!("could not write audit record {}: {error}", path.display()),
    })?;
    file.write_all(b"\n").map_err(|error| OutboundAuditError {
        message: format!("could not finish audit record {}: {error}", path.display()),
    })?;
    file.sync_all().map_err(|error| OutboundAuditError {
        message: format!("could not sync audit record {}: {error}", path.display()),
    })?;

    retain_newest_records(&config.log_dir, RETAIN_RECORDS)?;

    Ok(path)
}

fn unique_record_path(log_dir: &Path, timestamp: &str) -> PathBuf {
    let process_id = std::process::id();
    let timestamp = timestamp.replace([':', '.'], "-");
    log_dir.join(format!(
        "{FILE_PREFIX}{timestamp}-{process_id}{FILE_SUFFIX}"
    ))
}

#[cfg(unix)]
fn create_private_file(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn create_private_file(path: &Path) -> std::io::Result<std::fs::File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

fn retain_newest_records(log_dir: &Path, retain: usize) -> Result<(), OutboundAuditError> {
    let mut records = fs::read_dir(log_dir)
        .map_err(|error| OutboundAuditError {
            message: format!(
                "could not read audit log directory {}: {error}",
                log_dir.display()
            ),
        })?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let file_name = entry.file_name();
            let file_name = file_name.to_str()?;
            (file_name.starts_with(FILE_PREFIX) && file_name.ends_with(FILE_SUFFIX))
                .then(|| entry.path())
        })
        .collect::<Vec<_>>();

    if records.len() <= retain {
        return Ok(());
    }

    records.sort();
    let remove_count = records.len() - retain;
    for path in records.into_iter().take(remove_count) {
        fs::remove_file(&path).map_err(|error| OutboundAuditError {
            message: format!(
                "could not remove old audit record {}: {error}",
                path.display()
            ),
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn records_exact_prompt_and_metadata() {
        let log_dir = temp_log_dir("exact");
        let config = OutboundAuditConfig::from_dir(&log_dir).unwrap();
        let prompt = "First line\n\nSecond line with \"quotes\" and trailing newline\n";

        let path = write_outbound_prompt(&config, "test/model", "What changed?", prompt).unwrap();

        let record = read_record(&path);
        assert_eq!(record["selected_model"], "test/model");
        assert_eq!(record["original_question"], "What changed?");
        assert_eq!(record["prompt_byte_length"], prompt.len());
        assert_eq!(record["outbound_prompt"], prompt);
        assert!(record["timestamp"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));

        fs::remove_dir_all(log_dir).unwrap();
    }

    #[test]
    fn record_omits_api_key_authorization_headers_and_environment() {
        let log_dir = temp_log_dir("privacy");
        let config = OutboundAuditConfig::from_dir(&log_dir).unwrap();

        let path = write_outbound_prompt(&config, "test/model", "Question", "Prompt").unwrap();

        let raw = fs::read_to_string(path).unwrap();
        assert!(!raw.contains("OPENROUTER_API_KEY"));
        assert!(!raw.contains("Authorization"));
        assert!(!raw.contains("Bearer"));
        assert!(!raw.contains("test-secret"));

        fs::remove_dir_all(log_dir).unwrap();
    }

    #[test]
    fn empty_log_dir_disables_auditing() {
        assert_eq!(OutboundAuditConfig::from_dir(""), None);
    }

    #[cfg(unix)]
    #[test]
    fn creates_audit_file_with_user_only_permissions() {
        let log_dir = temp_log_dir("permissions");
        let config = OutboundAuditConfig::from_dir(&log_dir).unwrap();

        let path = write_outbound_prompt(&config, "test/model", "Question", "Prompt").unwrap();

        let mode = fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);

        fs::remove_dir_all(log_dir).unwrap();
    }

    #[test]
    fn write_failure_is_reported() {
        let blocking_file = temp_log_dir("not-a-directory");
        fs::write(&blocking_file, "not a directory").unwrap();
        let config = OutboundAuditConfig::from_dir(&blocking_file).unwrap();

        let error = write_outbound_prompt(&config, "test/model", "Question", "Prompt")
            .expect_err("audit should fail when log dir is a file");

        assert!(error.to_string().contains("outbound audit failed"));
        fs::remove_file(blocking_file).unwrap();
    }

    #[test]
    fn retains_only_newest_twenty_recall_audit_records() {
        let log_dir = temp_log_dir("retention");
        let config = OutboundAuditConfig::from_dir(&log_dir).unwrap();
        fs::create_dir_all(&log_dir).unwrap();

        for index in 0..25 {
            fs::write(
                log_dir.join(format!(
                    "{FILE_PREFIX}2000-01-01T00-00-{index:02}Z-1{FILE_SUFFIX}"
                )),
                "{}\n",
            )
            .unwrap();
        }
        fs::write(log_dir.join("unrelated.json"), "{}\n").unwrap();

        write_outbound_prompt(&config, "test/model", "Question", "Prompt").unwrap();

        let recall_records = fs::read_dir(&log_dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_name().to_str().is_some_and(|name| {
                    name.starts_with(FILE_PREFIX) && name.ends_with(FILE_SUFFIX)
                })
            })
            .count();
        assert_eq!(recall_records, RETAIN_RECORDS);
        assert!(log_dir.join("unrelated.json").exists());

        fs::remove_dir_all(log_dir).unwrap();
    }

    fn read_record(path: &Path) -> serde_json::Value {
        let raw = fs::read_to_string(path).unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    fn temp_log_dir(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("recall-outbound-audit-{name}-{unique}"))
    }
}
