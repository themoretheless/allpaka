//! Editable agent harness: the files around the model that shape a turn.
//!
//! A harness is a flat map from a relative path to text (the Harness² "tree").
//! Four always-on files are composed into the system prompt; skills are named
//! there and loaded on demand; subagents run as delegated turns; scripts run
//! through the command plugin; `hooks.json` adds reminders inside the tool
//! loop. An empty tree changes nothing, so a project without a harness keeps
//! today's prompt byte for byte.
//!
//! Revisions are immutable and content addressed under
//! `<data>/harness/revisions/<sha256>.json`; `harness/active.state` maps each
//! project to the revision its turns use.

use crate::{error, ApiResult, App};
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub type Tree = BTreeMap<String, String>;

pub const SYSTEM: &str = "system.md";
pub const GUARDRAILS: &str = "guardrails.md";
pub const MEMORY: &str = "memory.md";
pub const TOOL_GUIDANCE: &str = "tool_guidance.md";
pub const HOOKS: &str = "hooks.json";
pub const ALWAYS_ON: [&str; 4] = [SYSTEM, GUARDRAILS, MEMORY, TOOL_GUIDANCE];
/// Combined size of the always-on files, the cost paid on every request.
pub const ALWAYS_ON_CAP: usize = 24_000;
const MAX_FILE_BYTES: usize = 64_000;
const MAX_FILES: usize = 128;
const MAX_TREE_BYTES: usize = 512_000;
const MAX_HOOKS: usize = 16;

/// What a path is on the harness surface. Anything else is rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    AlwaysOn,
    Skill(String),
    SkillScript(String),
    Subagent(String),
    Script,
    Hooks,
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}
fn valid_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 80
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

pub fn classify(path: &str) -> Option<Kind> {
    if ALWAYS_ON.contains(&path) {
        return Some(Kind::AlwaysOn);
    }
    if path == HOOKS {
        return Some(Kind::Hooks);
    }
    let parts: Vec<&str> = path.split('/').collect();
    match parts.as_slice() {
        ["skills", name, "SKILL.md"] if valid_name(name) => Some(Kind::Skill((*name).into())),
        ["skills", name, "scripts", file] if valid_name(name) && is_script(file) => {
            Some(Kind::SkillScript((*name).into()))
        }
        ["subagents", name, "AGENT.md"] if valid_name(name) => Some(Kind::Subagent((*name).into())),
        ["scripts", file] if is_script(file) => Some(Kind::Script),
        _ => None,
    }
}
fn is_script(file: &str) -> bool {
    valid_file_name(file) && (file.ends_with(".sh") || file.ends_with(".py"))
}

/// Split `---` YAML-style frontmatter of flat `key: value` lines from a body.
pub fn frontmatter(text: &str) -> (BTreeMap<String, String>, &str) {
    let mut fields = BTreeMap::new();
    let Some(rest) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (fields, text);
    };
    let Some(end) = rest.find("\n---") else {
        return (fields, text);
    };
    for line in rest[..end].lines() {
        if let Some((key, value)) = line.split_once(':') {
            let value = value.trim().trim_matches('"').trim_matches('\'');
            fields.insert(key.trim().to_string(), value.to_string());
        }
    }
    let body = &rest[end + 4..];
    let body = body.strip_prefix('\n').unwrap_or(body);
    (
        fields,
        body.trim_start_matches('\r').trim_start_matches('\n'),
    )
}

/// Text that is only HTML comments and whitespace renders to nothing, so
/// placeholder files cost no prompt.
pub fn meaningful(text: &str) -> &str {
    let mut rest = text.trim();
    while let Some(after) = rest.strip_prefix("<!--") {
        match after.find("-->") {
            Some(end) => rest = after[end + 3..].trim(),
            None => return "",
        }
    }
    if rest.is_empty() {
        ""
    } else {
        text.trim()
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub description: String,
}

pub fn skills(tree: &Tree) -> Vec<Entry> {
    tree.iter()
        .filter_map(|(path, text)| match classify(path) {
            Some(Kind::Skill(name)) => Some(Entry {
                description: frontmatter(text)
                    .0
                    .remove("description")
                    .unwrap_or_default(),
                name,
            }),
            _ => None,
        })
        .collect()
}
#[derive(Clone, Debug, Serialize)]
pub struct Subagent {
    pub name: String,
    pub description: String,
    pub writes: bool,
    pub prompt: String,
}
pub fn subagents(tree: &Tree) -> Vec<Subagent> {
    tree.iter()
        .filter_map(|(path, text)| match classify(path) {
            Some(Kind::Subagent(name)) => {
                let (mut fields, body) = frontmatter(text);
                Some(Subagent {
                    name,
                    description: fields.remove("description").unwrap_or_default(),
                    writes: fields.get("writes").is_some_and(|v| v == "true"),
                    prompt: body.to_string(),
                })
            }
            _ => None,
        })
        .collect()
}
/// Runnable scripts with the first comment or docstring line as a summary.
pub fn scripts(tree: &Tree) -> Vec<Entry> {
    tree.iter()
        .filter(|(path, _)| matches!(classify(path), Some(Kind::Script | Kind::SkillScript(_))))
        .map(|(path, text)| Entry {
            name: path.clone(),
            description: text
                .lines()
                .map(|l| {
                    l.trim()
                        .trim_start_matches("#!")
                        .trim_start_matches('#')
                        .trim_matches('"')
                        .trim()
                })
                .find(|l| !l.is_empty() && !l.starts_with("/usr/") && !l.starts_with("/bin/"))
                .unwrap_or("")
                .chars()
                .take(160)
                .collect(),
        })
        .collect()
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Hook {
    /// Tool name, or a prefix ending in `*`. Empty matches every tool.
    #[serde(default)]
    pub matcher: String,
    /// Only when the raw JSON arguments contain this text.
    #[serde(default)]
    pub args_contains: Option<String>,
    /// post_tool, stop: text added for the model.
    #[serde(default)]
    pub inject: Option<String>,
    /// pre_tool: a non-zero exit blocks the call with its output as the
    /// reason. post_tool, stop: stdout is added for the model.
    #[serde(default)]
    pub command: Option<String>,
    /// pre_tool: refuse the call with this reason.
    #[serde(default)]
    pub block: Option<String>,
    /// post_tool: keep at most this many characters of the tool result.
    #[serde(default)]
    pub max_output: Option<usize>,
    /// post_tool: a program that reads the tool result on stdin and whose
    /// stdout replaces it (code acting in the loop, like a Harness² plugin).
    #[serde(default)]
    pub transform: Option<String>,
}
/// Hooks act inside the tool loop, where prose cannot be skipped:
/// `pre_tool` before a call runs, `post_tool` on its result, `stop` once
/// before the agent finishes.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Hooks {
    #[serde(default)]
    pub pre_tool: Vec<Hook>,
    #[serde(default)]
    pub post_tool: Vec<Hook>,
    #[serde(default)]
    pub stop: Vec<Hook>,
}
impl Hook {
    pub fn matches(&self, tool: &str, args: &str) -> bool {
        let name = match self.matcher.strip_suffix('*') {
            _ if self.matcher.is_empty() => true,
            Some(prefix) => tool.starts_with(prefix),
            None => tool == self.matcher,
        };
        name && self
            .args_contains
            .as_deref()
            .is_none_or(|text| args.contains(text))
    }
}
fn present(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|s| !s.trim().is_empty())
}
pub fn hooks(tree: &Tree) -> Result<Hooks> {
    let text = match tree.get(HOOKS) {
        None => return Ok(Hooks::default()),
        Some(text) if text.trim().is_empty() => return Ok(Hooks::default()),
        Some(text) => text,
    };
    let hooks: Hooks = serde_json::from_value(crate::strict_json::parse(text)?)
        .context("hooks.json must be {\"pre_tool\":[…],\"post_tool\":[…],\"stop\":[…]}")?;
    let events: [(&str, &[Hook], &[&str]); 3] = [
        ("pre_tool", &hooks.pre_tool, &["block", "command"]),
        (
            "post_tool",
            &hooks.post_tool,
            &["inject", "command", "max_output", "transform"],
        ),
        ("stop", &hooks.stop, &["inject", "command"]),
    ];
    for (event, list, allowed) in events {
        for hook in list {
            let set: Vec<&str> = [
                ("inject", present(&hook.inject)),
                ("command", present(&hook.command)),
                ("block", present(&hook.block)),
                ("max_output", hook.max_output.is_some()),
                ("transform", present(&hook.transform)),
            ]
            .into_iter()
            .filter(|(_, on)| *on)
            .map(|(name, _)| name)
            .collect();
            if set.len() != 1 || !allowed.contains(&set[0]) {
                bail!(
                    "Each {event} hook needs exactly one of {}",
                    allowed.join(", ")
                );
            }
            if hook
                .max_output
                .is_some_and(|n| !(200..=131072).contains(&n))
            {
                bail!("max_output must be between 200 and 131072");
            }
            if hook.matcher.len() > 80
                || hook
                    .args_contains
                    .as_ref()
                    .is_some_and(|s| s.is_empty() || s.len() > 200)
                || hook.inject.as_ref().is_some_and(|s| s.len() > 2000)
                || hook.block.as_ref().is_some_and(|s| s.len() > 2000)
                || hook.command.as_ref().is_some_and(|s| s.len() > 1000)
                || hook.transform.as_ref().is_some_and(|s| s.len() > 1000)
            {
                bail!("Hook text exceeds bounds");
            }
        }
    }
    if hooks.pre_tool.len() + hooks.post_tool.len() + hooks.stop.len() > MAX_HOOKS {
        bail!("At most {MAX_HOOKS} hooks");
    }
    Ok(hooks)
}

/// Keep the head and the tail of a tool result, as a post_tool hook asks.
pub fn trim_output(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit * 3 / 4).collect();
    let tail: String = text.chars().skip(count - limit / 4).collect();
    format!(
        "{head}\n… [harness hook trimmed {} characters] …\n{tail}",
        count - limit
    )
}

fn always_on_text(tree: &Tree) -> String {
    ALWAYS_ON
        .iter()
        .filter_map(|f| tree.get(*f))
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n")
}
pub fn always_on_chars(tree: &Tree) -> usize {
    ALWAYS_ON
        .iter()
        .filter_map(|f| tree.get(*f))
        .map(|t| meaningful(t).chars().count())
        .sum()
}

/// Append a trigger line to the guardrails for every skill that the
/// always-on text never names; an unnamed skill would never be loaded.
pub fn autotrigger(tree: &Tree) -> Tree {
    let mut out = tree.clone();
    let text = always_on_text(tree);
    let mut lines = vec![];
    for skill in skills(tree) {
        if !text.contains(&format!("`{}`", skill.name)) {
            let when = skill.description.trim_end_matches('.');
            lines.push(format!("- When {when}, load the `{}` skill.", skill.name));
        }
    }
    if !lines.is_empty() {
        let guard = out.entry(GUARDRAILS.into()).or_default();
        if !guard.is_empty() && !guard.ends_with('\n') {
            guard.push('\n');
        }
        guard.push_str(&lines.join("\n"));
        guard.push('\n');
    }
    out
}

/// Every reason the tree cannot be used. Empty means valid.
pub fn validate(tree: &Tree) -> Vec<String> {
    let mut errors = vec![];
    if tree.len() > MAX_FILES {
        errors.push(format!("At most {MAX_FILES} files"));
    }
    if tree.iter().map(|(p, t)| p.len() + t.len()).sum::<usize>() > MAX_TREE_BYTES {
        errors.push(format!("Harness exceeds {MAX_TREE_BYTES} bytes"));
    }
    for (path, text) in tree {
        if classify(path).is_none() {
            errors.push(format!("{path}: not a harness path (always-on files, skills/<name>/SKILL.md, skills/<name>/scripts/*, subagents/<name>/AGENT.md, scripts/*.sh|py, hooks.json)"));
        }
        if text.len() > MAX_FILE_BYTES {
            errors.push(format!("{path}: exceeds {MAX_FILE_BYTES} bytes"));
        }
        if text.contains('\0') {
            errors.push(format!("{path}: contains NUL"));
        }
    }
    let cost = always_on_chars(tree);
    if cost > ALWAYS_ON_CAP {
        errors.push(format!(
            "Always-on text is {cost} characters; the cap is {ALWAYS_ON_CAP}. Move procedure into skills."
        ));
    }
    let named = always_on_text(tree);
    for (path, text) in tree {
        match classify(path) {
            Some(Kind::Skill(dir)) => {
                let fields = frontmatter(text).0;
                if fields.get("name") != Some(&dir) {
                    errors.push(format!("{path}: frontmatter name must be `{dir}`"));
                }
                if fields
                    .get("description")
                    .is_none_or(|d| d.trim().is_empty())
                {
                    errors.push(format!("{path}: frontmatter needs a description"));
                }
                if !named.contains(&format!("`{dir}`")) {
                    errors.push(format!(
                        "{path}: no always-on text names `{dir}`, so it would never be loaded"
                    ));
                }
            }
            Some(Kind::SkillScript(dir))
                if !tree.contains_key(&format!("skills/{dir}/SKILL.md")) =>
            {
                errors.push(format!("{path}: script of a missing skill"));
            }
            Some(Kind::Subagent(_)) => {
                let (fields, body) = frontmatter(text);
                if fields
                    .get("description")
                    .is_none_or(|d| d.trim().is_empty())
                {
                    errors.push(format!("{path}: frontmatter needs a description"));
                }
                if fields
                    .get("writes")
                    .is_some_and(|w| w != "true" && w != "false")
                {
                    errors.push(format!("{path}: writes must be true or false"));
                }
                if body.trim().is_empty() {
                    errors.push(format!("{path}: subagent prompt is empty"));
                }
            }
            _ => {}
        }
    }
    if let Err(e) = hooks(tree) {
        errors.push(format!("{HOOKS}: {e:#}"));
    }
    // A reference to a script that is not shipped sends the agent to a dead end.
    for (path, text) in tree {
        if !matches!(classify(path), Some(Kind::AlwaysOn | Kind::Skill(_))) {
            continue;
        }
        for reference in script_references(text) {
            if !tree.contains_key(&reference) {
                errors.push(format!(
                    "{path}: references {reference}, which is not in the harness"
                ));
            }
        }
    }
    errors
}
fn script_references(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for word in text.split(|c: char| c.is_whitespace() || matches!(c, '`' | '(' | ')' | '"' | '\''))
    {
        let word = word.trim_end_matches(['.', ',', ';', ':']);
        if (word.starts_with("scripts/")
            || (word.starts_with("skills/") && word.contains("/scripts/")))
            && (word.ends_with(".sh") || word.ends_with(".py"))
        {
            out.insert(word.to_string());
        }
    }
    out
}

/// The harness part of the system prompt. Empty for an empty or
/// placeholder-only tree. `dynamic` lists skills and subagents; executors
/// without `load_skill`/`delegate` (Swarm members) leave it off.
pub fn compose_system(tree: &Tree, dynamic: bool, scripts_runnable: bool) -> String {
    let mut parts = vec![];
    let section = |file: &str, title: Option<&str>| -> Option<String> {
        let text = meaningful(tree.get(file)?);
        (!text.is_empty()).then(|| match title {
            Some(title) => format!("# {title}\n{text}"),
            None => text.to_string(),
        })
    };
    parts.extend(section(SYSTEM, None));
    parts.extend(section(GUARDRAILS, Some("Operating guardrails")));
    parts.extend(section(TOOL_GUIDANCE, Some("Tool guidance")));
    parts.extend(section(MEMORY, Some("Long-term memory")));
    if dynamic {
        let skills = skills(tree);
        if !skills.is_empty() {
            parts.push(format!(
                "# Available skills (load the full text with load_skill before following one)\n{}",
                skills
                    .iter()
                    .map(|s| format!("- {}: {}", s.name, s.description))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        let agents = subagents(tree);
        if !agents.is_empty() {
            parts.push(format!(
                "# Available subagents (hand a self-contained task to one with delegate)\n{}",
                agents
                    .iter()
                    .map(|a| format!("- {}: {}", a.name, a.description))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }
    if scripts_runnable {
        let scripts = scripts(tree);
        if !scripts.is_empty() {
            parts.push(format!(
                "# Helper scripts (run with run_script)\n{}",
                scripts
                    .iter()
                    .map(|s| format!("- {}: {}", s.name, s.description))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }
    parts.join("\n\n")
}

pub fn fingerprint(tree: &Tree) -> String {
    let mut hasher = Sha256::new();
    for (path, text) in tree {
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update((text.len() as u64).to_le_bytes());
        hasher.update(text.as_bytes());
    }
    crate::evaluation::hex_sha256(hasher)
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Diff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub modified: Vec<String>,
    /// Unified-style text for display and as proposer evidence.
    pub text: String,
}
pub fn diff(from: &Tree, to: &Tree) -> Diff {
    let mut out = Diff::default();
    let paths: BTreeSet<&String> = from.keys().chain(to.keys()).collect();
    for path in paths {
        let (a, b) = (from.get(path), to.get(path));
        match (a, b) {
            (None, Some(b)) => {
                out.added.push(path.clone());
                out.text.push_str(&format!("+++ {path} (added)\n"));
                for line in b.lines() {
                    out.text.push_str(&format!("+{line}\n"));
                }
            }
            (Some(a), None) => {
                out.removed.push(path.clone());
                out.text.push_str(&format!("--- {path} (removed)\n"));
                for line in a.lines() {
                    out.text.push_str(&format!("-{line}\n"));
                }
            }
            (Some(a), Some(b)) if a != b => {
                out.modified.push(path.clone());
                out.text.push_str(&format!("*** {path}\n"));
                out.text.push_str(&line_diff(a, b));
            }
            _ => {}
        }
    }
    out
}
/// LCS line diff; falls back to replace-all when the table would be large.
pub(crate) fn line_diff(a: &str, b: &str) -> String {
    let a: Vec<&str> = a.lines().collect();
    let b: Vec<&str> = b.lines().collect();
    let mut out = String::new();
    if a.len().saturating_mul(b.len()) > 4_000_000 {
        a.iter().for_each(|l| out.push_str(&format!("-{l}\n")));
        b.iter().for_each(|l| out.push_str(&format!("+{l}\n")));
        return out;
    }
    let mut table = vec![vec![0u32; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            table[i][j] = if a[i] == b[j] {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            out.push_str(&format!(" {}\n", a[i]));
            i += 1;
            j += 1;
        } else if i < a.len() && (j == b.len() || table[i + 1][j] >= table[i][j + 1]) {
            out.push_str(&format!("-{}\n", a[i]));
            i += 1;
        } else {
            out.push_str(&format!("+{}\n", b[j]));
            j += 1;
        }
    }
    out
}

/// Mean pairwise line-Jaccard similarity: 1.0 means the trees collapsed into
/// one harness, lower means the candidates explored different edits.
pub fn similarity(trees: &[&Tree]) -> Option<f64> {
    let sets: Vec<BTreeSet<String>> = trees
        .iter()
        .map(|tree| {
            tree.iter()
                .flat_map(|(p, t)| t.lines().map(move |l| format!("{p}\0{}", l.trim())))
                .filter(|l| !l.ends_with('\0'))
                .collect()
        })
        .collect();
    let mut total = 0.0;
    let mut pairs = 0;
    for i in 0..sets.len() {
        for j in i + 1..sets.len() {
            let union = sets[i].union(&sets[j]).count();
            let shared = sets[i].intersection(&sets[j]).count();
            total += if union == 0 {
                1.0
            } else {
                shared as f64 / union as f64
            };
            pairs += 1;
        }
    }
    (pairs > 0).then(|| total / pairs as f64)
}

/// The body of one skill and the scripts it ships, for the load_skill tool.
pub fn load_skill(tree: &Tree, name: &str) -> Result<Value> {
    let text = tree
        .get(&format!("skills/{name}/SKILL.md"))
        .with_context(|| {
            format!(
                "Unknown skill `{name}`; available: {}",
                skills(tree)
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let prefix = format!("skills/{name}/scripts/");
    Ok(json!({
        "skill": name,
        "instructions": frontmatter(text).1,
        "scripts": tree.keys().filter(|p| p.starts_with(&prefix)).collect::<Vec<_>>(),
    }))
}

pub fn tool_schemas(tree: &Tree, dynamic: bool, scripts_runnable: bool) -> Vec<Value> {
    let schema = |name: &str, description: &str, properties: Value, required: Value| json!({"type":"function","function":{"name":name,"description":description,"parameters":{"type":"object","properties":properties,"required":required,"additionalProperties":false}}});
    let mut list = vec![];
    if dynamic && !skills(tree).is_empty() {
        list.push(schema(
            "load_skill",
            "Load the full instructions of one harness skill listed under Available skills.",
            json!({"name":{"type":"string"}}),
            json!(["name"]),
        ));
    }
    if dynamic && !subagents(tree).is_empty() {
        list.push(schema(
            "delegate",
            "Hand a self-contained task to one harness subagent. It runs in its own context with your tools (writes only if it is allowed to) and returns its final answer. It cannot see this conversation, so include everything it needs.",
            json!({"agent":{"type":"string"},"task":{"type":"string","maxLength":16000}}),
            json!(["agent","task"]),
        ));
    }
    if scripts_runnable && !scripts(tree).is_empty() {
        list.push(schema(
            "run_script",
            "Run one harness helper script from the project root. Returns stdout, stderr and exit code.",
            json!({"path":{"type":"string"},"args":{"type":"array","maxItems":32,"items":{"type":"string","maxLength":4000}},"timeout":{"type":"integer","minimum":1,"maximum":120}}),
            json!(["path"]),
        ));
    }
    list
}

/// Write the script set of a tree to `<data>/harness/run/<sha>/` once and
/// return the absolute path of `script`.
pub fn materialize_script(
    data: &std::path::Path,
    tree: &Tree,
    script: &str,
) -> Result<std::path::PathBuf> {
    if !matches!(classify(script), Some(Kind::Script | Kind::SkillScript(_))) {
        bail!("Unknown harness script `{script}`");
    }
    let text = tree
        .get(script)
        .with_context(|| format!("Unknown harness script `{script}`"))?;
    let directory = data.join("harness/run").join(fingerprint(tree));
    let path = directory.join(script);
    if !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap())?;
        let temp = path.with_extension(format!("{}.tmp", crate::evaluation::new_id()));
        std::fs::write(&temp, text)?;
        std::fs::rename(temp, &path)?;
    }
    Ok(path)
}
pub fn script_command(path: &std::path::Path, args: &[String]) -> String {
    let quote = |s: &str| format!("'{}'", s.replace('\'', "'\\''"));
    let interpreter = if path.extension().is_some_and(|e| e == "py") {
        "python3"
    } else {
        "/bin/sh"
    };
    let mut command = format!("{interpreter} {}", quote(&path.to_string_lossy()));
    for arg in args {
        command.push(' ');
        command.push_str(&quote(arg));
    }
    command
}

// ---- storage ---------------------------------------------------------------

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Revision {
    pub id: String,
    pub project_id: String,
    pub tree: Tree,
    #[serde(default)]
    pub origin: Option<String>,
    #[serde(default)]
    pub note: String,
    pub created_ms: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Active {
    project_id: String,
    revision: String,
}

fn valid_hash(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn revisions_dir(data: &std::path::Path) -> std::path::PathBuf {
    data.join("harness/revisions")
}
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Store a tree as a revision. The ID covers the project and the tree, so
/// saving the same tree twice returns the existing revision.
pub fn save_revision(
    data: &std::path::Path,
    project_id: &str,
    tree: &Tree,
    origin: Option<String>,
    note: &str,
) -> Result<Revision> {
    let errors = validate(tree);
    if !errors.is_empty() {
        bail!("{}", errors.join("\n"));
    }
    if note.len() > 4000 {
        bail!("Revision note exceeds 4000 bytes");
    }
    if origin.as_deref().is_some_and(|o| !valid_hash(o)) {
        bail!("Invalid origin revision");
    }
    let mut hasher = Sha256::new();
    hasher.update(project_id.as_bytes());
    hasher.update([0]);
    hasher.update(fingerprint(tree).as_bytes());
    let id = crate::evaluation::hex_sha256(hasher);
    let directory = revisions_dir(data);
    std::fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{id}.json"));
    if path.exists() {
        return load_revision(data, &id);
    }
    let revision = Revision {
        id,
        project_id: project_id.into(),
        tree: tree.clone(),
        origin,
        note: note.into(),
        created_ms: now_ms(),
    };
    match crate::evaluation::commit_new(&path, &revision) {
        Ok(()) => Ok(revision),
        Err(_) if path.exists() => load_revision(data, &revision.id),
        Err(e) => Err(e),
    }
}
pub fn load_revision(data: &std::path::Path, id: &str) -> Result<Revision> {
    if !valid_hash(id) {
        bail!("Invalid harness revision");
    }
    let bytes = std::fs::read(revisions_dir(data).join(format!("{id}.json")))
        .context("Harness revision not found")?;
    let revision: Revision = serde_json::from_slice(&bytes).context("Invalid harness revision")?;
    if revision.id != id {
        bail!("Harness revision does not match its name");
    }
    Ok(revision)
}
pub fn list_revisions(data: &std::path::Path, project_id: &str) -> Result<Vec<Revision>> {
    let mut list = vec![];
    let Ok(entries) = std::fs::read_dir(revisions_dir(data)) else {
        return Ok(list);
    };
    for entry in entries {
        let name = entry?.file_name().to_string_lossy().to_string();
        if let Some(id) = name.strip_suffix(".json") {
            if let Ok(revision) = load_revision(data, id) {
                if revision.project_id == project_id {
                    list.push(revision);
                }
            }
        }
    }
    list.sort_by_key(|r| std::cmp::Reverse(r.created_ms));
    list.truncate(200);
    Ok(list)
}
fn load_active(data: &std::path::Path) -> Result<Vec<Active>> {
    crate::state_file::load_list(
        data.join("harness").as_path(),
        "active",
        "harness",
        |a: &Active| {
            if valid_hash(&a.revision) {
                Ok(())
            } else {
                bail!("Invalid harness revision")
            }
        },
    )
}
pub fn active_revision(data: &std::path::Path, project_id: &str) -> Result<Option<Revision>> {
    match load_active(data)?
        .into_iter()
        .find(|a| a.project_id == project_id)
    {
        Some(active) => load_revision(data, &active.revision).map(Some),
        None => Ok(None),
    }
}
/// The tree a turn of this project runs with; empty when none is active.
pub fn active_tree(data: &std::path::Path, project_id: &str) -> Result<Tree> {
    Ok(active_revision(data, project_id)?
        .map(|r| r.tree)
        .unwrap_or_default())
}
pub fn activate(data: &std::path::Path, project_id: &str, revision: Option<&str>) -> Result<()> {
    if let Some(id) = revision {
        if load_revision(data, id)?.project_id != project_id {
            bail!("Revision belongs to another project");
        }
    }
    std::fs::create_dir_all(data.join("harness"))?;
    let mut list = load_active(data)?;
    list.retain(|a| a.project_id != project_id);
    if let Some(id) = revision {
        list.push(Active {
            project_id: project_id.into(),
            revision: id.into(),
        });
    }
    crate::state_file::save_list(data.join("harness").as_path(), "active", &list)
}

// ---- HTTP ------------------------------------------------------------------

fn bad(e: impl ToString) -> (StatusCode, Json<Value>) {
    error(StatusCode::BAD_REQUEST, e)
}
fn known_project(app: &App, id: &str) -> std::result::Result<(), (StatusCode, Json<Value>)> {
    if app.projects.lock().unwrap().iter().any(|p| p.id == id) {
        Ok(())
    } else {
        Err(error(StatusCode::NOT_FOUND, "Project not found"))
    }
}
fn summary(revision: &Revision) -> Value {
    json!({
        "id": revision.id,
        "origin": revision.origin,
        "note": revision.note,
        "created_ms": revision.created_ms,
        "files": revision.tree.len(),
        "skills": skills(&revision.tree).len(),
        "subagents": subagents(&revision.tree).len(),
        "always_on_chars": always_on_chars(&revision.tree),
    })
}
pub fn describe(tree: &Tree) -> Value {
    json!({
        "tree": tree,
        "errors": validate(tree),
        "always_on_chars": always_on_chars(tree),
        "always_on_cap": ALWAYS_ON_CAP,
        "skills": skills(tree),
        "subagents": subagents(tree).iter().map(|a| json!({"name":a.name,"description":a.description,"writes":a.writes})).collect::<Vec<_>>(),
        "scripts": scripts(tree),
        "preview": compose_system(tree, true, true),
    })
}

#[derive(Deserialize)]
pub(crate) struct ProjectQuery {
    project_id: String,
}
pub(crate) async fn get_api(
    State(app): State<App>,
    Query(q): Query<ProjectQuery>,
) -> ApiResult<Value> {
    known_project(&app, &q.project_id)?;
    let active = active_revision(&app.data, &q.project_id).map_err(bad)?;
    let revisions = list_revisions(&app.data, &q.project_id).map_err(bad)?;
    let tree = active.as_ref().map(|r| r.tree.clone()).unwrap_or_default();
    Ok(Json(json!({
        "project_id": q.project_id,
        "active": active.as_ref().map(|r| r.id.clone()),
        "current": describe(&tree),
        "revisions": revisions.iter().map(summary).collect::<Vec<_>>(),
    })))
}
pub(crate) async fn revision_api(
    State(app): State<App>,
    Path(id): Path<String>,
) -> ApiResult<Value> {
    let revision = load_revision(&app.data, &id).map_err(|e| error(StatusCode::NOT_FOUND, e))?;
    let parent = revision
        .origin
        .as_deref()
        .and_then(|o| load_revision(&app.data, o).ok())
        .map(|r| r.tree)
        .unwrap_or_default();
    Ok(Json(json!({
        "revision": summary(&revision),
        "project_id": revision.project_id,
        "current": describe(&revision.tree),
        "diff_from_origin": diff(&parent, &revision.tree),
    })))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SaveRequest {
    project_id: String,
    tree: Tree,
    #[serde(default)]
    note: String,
    #[serde(default)]
    activate: bool,
}
pub(crate) async fn save_api(
    State(app): State<App>,
    Json(request): Json<SaveRequest>,
) -> ApiResult<Value> {
    known_project(&app, &request.project_id)?;
    let origin = active_revision(&app.data, &request.project_id)
        .map_err(bad)?
        .map(|r| r.id);
    let tree = autotrigger(&request.tree);
    let revision =
        save_revision(&app.data, &request.project_id, &tree, origin, &request.note).map_err(bad)?;
    if request.activate {
        activate(&app.data, &request.project_id, Some(&revision.id)).map_err(bad)?;
    }
    Ok(Json(
        json!({"revision": summary(&revision), "current": describe(&revision.tree)}),
    ))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ActivateRequest {
    project_id: String,
    revision: Option<String>,
}
pub(crate) async fn activate_api(
    State(app): State<App>,
    Json(request): Json<ActivateRequest>,
) -> ApiResult<Value> {
    known_project(&app, &request.project_id)?;
    activate(&app.data, &request.project_id, request.revision.as_deref()).map_err(bad)?;
    Ok(Json(json!({"active": request.revision})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImportSkill {
    project_id: String,
    text: String,
    #[serde(default)]
    activate: bool,
}
/// Turn a `SKILL.md` (the format of `plugins/*.SKILL.md`) into a harness
/// skill on top of the active tree.
pub(crate) async fn import_skill_api(
    State(app): State<App>,
    Json(request): Json<ImportSkill>,
) -> ApiResult<Value> {
    known_project(&app, &request.project_id)?;
    let fields = frontmatter(&request.text).0;
    let name = fields
        .get("name")
        .filter(|n| valid_name(n))
        .ok_or_else(|| bad("SKILL.md needs frontmatter name of lowercase letters, digits, - or _"))?
        .clone();
    let active = active_revision(&app.data, &request.project_id).map_err(bad)?;
    let mut tree = active.as_ref().map(|r| r.tree.clone()).unwrap_or_default();
    tree.insert(format!("skills/{name}/SKILL.md"), request.text.clone());
    let tree = autotrigger(&tree);
    let revision = save_revision(
        &app.data,
        &request.project_id,
        &tree,
        active.map(|r| r.id),
        &format!("import skill {name}"),
    )
    .map_err(bad)?;
    if request.activate {
        activate(&app.data, &request.project_id, Some(&revision.id)).map_err(bad)?;
    }
    Ok(Json(
        json!({"revision": summary(&revision), "current": describe(&revision.tree)}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(files: &[(&str, &str)]) -> Tree {
        files
            .iter()
            .map(|(p, t)| (p.to_string(), t.to_string()))
            .collect()
    }
    const SKILL: &str = "---\nname: cite\ndescription: the answer quotes a source file\n---\nQuote the exact line and path.\n";

    #[test]
    fn empty_and_placeholder_trees_compose_nothing() {
        assert_eq!(compose_system(&Tree::new(), true, true), "");
        let t = tree(&[(SYSTEM, "<!-- placeholder -->\n"), (MEMORY, "  ")]);
        assert_eq!(compose_system(&t, true, true), "");
        assert!(validate(&t).is_empty());
    }

    #[test]
    fn compose_orders_sections_and_lists_dynamic_components() {
        let t = autotrigger(&tree(&[
            (SYSTEM, "Be exact."),
            (MEMORY, "In this workspace tests run with cargo."),
            ("skills/cite/SKILL.md", SKILL),
            (
                "subagents/review/AGENT.md",
                "---\ndescription: second opinion\n---\nReview critically.",
            ),
            ("scripts/count.py", "\"\"\"Count lines.\"\"\"\nprint(1)\n"),
        ]));
        let text = compose_system(&t, true, true);
        let order = [
            "Be exact.",
            "# Operating guardrails",
            "load the `cite` skill",
            "# Long-term memory",
            "# Available skills",
            "- cite:",
            "# Available subagents",
            "- review: second opinion",
            "- scripts/count.py: Count lines.",
        ];
        let mut at = 0;
        for needle in order {
            let found = text[at..]
                .find(needle)
                .unwrap_or_else(|| panic!("{needle} missing or out of order in\n{text}"));
            at += found;
        }
        let swarm = compose_system(&t, false, false);
        assert!(!swarm.contains("Available skills") && !swarm.contains("Helper scripts"));
        assert!(validate(&t).is_empty(), "{:?}", validate(&t));
    }

    #[test]
    fn validation_reports_each_broken_rule() {
        let t = tree(&[
            ("skills/cite/SKILL.md", "---\nname: other\n---\nbody"),
            ("subagents/x/AGENT.md", "---\nwrites: maybe\n---\n"),
            ("notes.txt", "x"),
            (GUARDRAILS, "run scripts/missing.py first"),
            (
                HOOKS,
                "{\"post_tool\":[{\"inject\":\"a\",\"command\":\"b\"}]}",
            ),
            (SYSTEM, &"x".repeat(ALWAYS_ON_CAP + 1)),
        ]);
        let errors = validate(&t).join("\n");
        for expected in [
            "name must be `cite`",
            "needs a description",
            "never be loaded",
            "notes.txt: not a harness path",
            "scripts/missing.py",
            "exactly one of",
            "cap is 24000",
            "subagent prompt is empty",
        ] {
            assert!(errors.contains(expected), "missing {expected} in\n{errors}");
        }
    }

    #[test]
    fn autotrigger_only_adds_missing_triggers() {
        let t = autotrigger(&tree(&[("skills/cite/SKILL.md", SKILL)]));
        assert_eq!(
            t[GUARDRAILS],
            "- When the answer quotes a source file, load the `cite` skill.\n"
        );
        assert_eq!(autotrigger(&t), t);
    }

    #[test]
    fn frontmatter_and_skill_loading() {
        let (fields, body) = frontmatter(SKILL);
        assert_eq!(fields["name"], "cite");
        assert_eq!(body, "Quote the exact line and path.\n");
        let t = tree(&[
            ("skills/cite/SKILL.md", SKILL),
            ("skills/cite/scripts/find.sh", "# find\n"),
        ]);
        let loaded = load_skill(&t, "cite").unwrap();
        assert_eq!(loaded["scripts"], json!(["skills/cite/scripts/find.sh"]));
        assert!(load_skill(&t, "nope")
            .unwrap_err()
            .to_string()
            .contains("available: cite"));
    }

    #[test]
    fn hooks_match_names_prefixes_and_arguments() {
        let t = tree(&[(
            HOOKS,
            r#"{"pre_tool":[{"matcher":"write_file","args_contains":"Cargo.lock","block":"Never edit the lockfile."}],"post_tool":[{"matcher":"write_file","inject":"Reopen the file."},{"matcher":"mcp_*","inject":"Cite the tool."},{"matcher":"read_file","max_output":200}],"stop":[{"inject":"Check the deliverable."}]}"#,
        )]);
        let h = hooks(&t).unwrap();
        assert!(h.pre_tool[0].matches("write_file", r#"{"path":"ws/Cargo.lock"}"#));
        assert!(!h.pre_tool[0].matches("write_file", r#"{"path":"ws/src/a.rs"}"#));
        assert!(
            h.post_tool[0].matches("write_file", "{}")
                && !h.post_tool[0].matches("edit_file", "{}")
        );
        assert!(h.post_tool[1].matches("mcp_rag_search", "{}"));
        assert!(h.stop[0].matches("anything", ""));
        assert!(validate(&t).is_empty());
        for bad in [
            "{\"pre\":[]}",
            r#"{"pre_tool":[{"inject":"x"}]}"#,
            r#"{"post_tool":[{"block":"x"}]}"#,
            r#"{"pre_tool":[{"transform":"cat"}]}"#,
            r#"{"post_tool":[{"inject":"x","max_output":500}]}"#,
            r#"{"post_tool":[{"max_output":10}]}"#,
            r#"{"stop":[{"max_output":500}]}"#,
        ] {
            assert!(hooks(&tree(&[(HOOKS, bad)])).is_err(), "{bad}");
        }
        let long = "x".repeat(1000);
        let trimmed = trim_output(&long, 200);
        assert!(
            trimmed.starts_with(&"x".repeat(150))
                && trimmed.contains("trimmed 800 characters")
                && trimmed.ends_with(&"x".repeat(50))
        );
        assert_eq!(trim_output("short", 200), "short");
    }

    #[test]
    fn diff_and_similarity() {
        let a = tree(&[(SYSTEM, "one\ntwo\n"), (MEMORY, "m")]);
        let b = tree(&[(SYSTEM, "one\nthree\n"), ("skills/cite/SKILL.md", SKILL)]);
        let d = diff(&a, &b);
        assert_eq!(d.added, vec!["skills/cite/SKILL.md"]);
        assert_eq!(d.removed, vec![MEMORY]);
        assert_eq!(d.modified, vec![SYSTEM]);
        assert!(d.text.contains(" one\n-two\n+three\n"), "{}", d.text);
        assert_eq!(similarity(&[&a, &a]), Some(1.0));
        assert!(similarity(&[&a, &b]).unwrap() < 0.5);
        assert_eq!(similarity(&[&a]), None);
    }

    #[test]
    fn revisions_are_content_addressed_and_activated_per_project() {
        let dir =
            std::env::temp_dir().join(format!("allpaka-harness-{}", crate::evaluation::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t = tree(&[(SYSTEM, "Be exact.")]);
        let first = save_revision(&dir, "p", &t, None, "first").unwrap();
        let again = save_revision(&dir, "p", &t, None, "again").unwrap();
        assert_eq!(first.id, again.id);
        assert_eq!(again.note, "first");
        assert!(save_revision(&dir, "p", &tree(&[("bad", "x")]), None, "").is_err());
        assert!(active_tree(&dir, "p").unwrap().is_empty());
        activate(&dir, "p", Some(&first.id)).unwrap();
        assert_eq!(active_tree(&dir, "p").unwrap(), t);
        assert!(active_tree(&dir, "q").unwrap().is_empty());
        assert!(activate(&dir, "q", Some(&first.id)).is_err());
        activate(&dir, "p", None).unwrap();
        assert!(active_tree(&dir, "p").unwrap().is_empty());
        assert_eq!(list_revisions(&dir, "p").unwrap().len(), 1);
        let script =
            materialize_script(&dir, &tree(&[("scripts/a.sh", "echo hi")]), "scripts/a.sh")
                .unwrap();
        assert_eq!(std::fs::read_to_string(&script).unwrap(), "echo hi");
        assert!(script_command(&script, &["it's".into()]).ends_with("'it'\\''s'"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
