use std::path::Path;
use std::process::Command;

fn valid_commit(value: &str) -> bool {
    (7..=40).contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn git_commit(workspace: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(workspace)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?;
    let commit = commit.trim();
    valid_commit(commit).then(|| commit.to_owned())
}

fn main() {
    println!("cargo:rerun-if-env-changed=ALLPAKA_GIT_COMMIT");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/packed-refs");

    let explicit = std::env::var("ALLPAKA_GIT_COMMIT")
        .ok()
        .filter(|value| valid_commit(value));
    let commit = explicit
        .or_else(|| git_commit(Path::new("../..")))
        .unwrap_or_else(|| "unknown".to_owned());

    println!("cargo:rustc-env=ALLPAKA_GIT_COMMIT={commit}");
}
