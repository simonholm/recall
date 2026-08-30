use std::fs;
use std::path::Path;
use std::process::Command;

const BUILD_RELEVANT_PATHS: &[&str] = &["Cargo.toml", "Cargo.lock", "crates"];

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.join("../..");

    for path in BUILD_RELEVANT_PATHS {
        println!("cargo:rerun-if-changed={}", repo_root.join(path).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join(".git/HEAD").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join(".git/index").display()
    );
    if let Some(head_ref) = current_head_ref(&repo_root.join(".git/HEAD")) {
        println!(
            "cargo:rerun-if-changed={}",
            repo_root.join(".git").join(head_ref).display()
        );
    }

    let Some(revision) = git_output(&repo_root, &["rev-parse", "--short=7", "HEAD"]) else {
        return;
    };
    if !revision.is_empty() {
        let dirty = is_build_relevant_worktree_dirty(&repo_root);
        let revision = if dirty {
            format!("{revision}-dirty")
        } else {
            revision
        };
        println!(
            "cargo:rustc-env=RECALL_LONG_VERSION={} ({revision})",
            env!("CARGO_PKG_VERSION")
        );
    }
}

fn is_build_relevant_worktree_dirty(repo_root: &Path) -> bool {
    let mut diff_args = vec!["diff", "--quiet", "HEAD", "--"];
    diff_args.extend(BUILD_RELEVANT_PATHS);
    if !git_status_success(repo_root, &diff_args) {
        return true;
    }

    let mut ls_files_args = vec!["ls-files", "--others", "--exclude-standard", "--"];
    ls_files_args.extend(BUILD_RELEVANT_PATHS);
    match git_output(repo_root, &ls_files_args) {
        Some(output) => !output.is_empty(),
        None => true,
    }
}

fn git_output(repo_root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_status_success(repo_root: &Path, args: &[&str]) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .status()
        .is_ok_and(|status| status.success())
}

fn current_head_ref(head_path: &Path) -> Option<String> {
    let head = fs::read_to_string(head_path).ok()?;
    head.strip_prefix("ref: ")
        .map(|reference| reference.trim().to_string())
}
