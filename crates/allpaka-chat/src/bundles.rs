//! Named runtime profiles, portable `.allpaka-bundle.json` export/import, and hot harness/plugin reload.

use crate::harness::{self, Tree};
use crate::sandbox::{ApprovalPolicy, SandboxMode};
use crate::workflow::{self, WorkflowSpec};
use crate::{bad_request, state_file, storage, ApiResult, App, Mode, Settings};
use anyhow::{bail, Context, Result};
use axum::extract::{Path as AxumPath, Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;

const MAX_PROFILES: usize = 64;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ProfileSpec {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub mode: Mode,
    #[serde(default)]
    pub sandbox_mode: SandboxMode,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
    #[serde(default)]
    pub allow_writes: bool,
    #[serde(default = "default_profile_steps")]
    pub max_steps: usize,
    #[serde(default)]
    pub team_mode: bool,
    #[serde(default)]
    pub builtin: bool,
}

fn default_profile_steps() -> usize {
    12
}

pub(crate) fn builtin_profiles() -> Vec<ProfileSpec> {
    vec![
        ProfileSpec {
            id: "default".into(),
            name: "Default".into(),
            description: "Balanced workspace-write Auto profile".into(),
            mode: Mode::Auto,
            sandbox_mode: SandboxMode::WorkspaceWrite,
            approval_policy: ApprovalPolicy::Commands,
            allow_writes: false,
            max_steps: 12,
            team_mode: false,
            builtin: true,
        },
        ProfileSpec {
            id: "coding-safe".into(),
            name: "Coding Safe".into(),
            description: "Writable sandboxed Auto session with approval gate".into(),
            mode: Mode::Auto,
            sandbox_mode: SandboxMode::WorkspaceWrite,
            approval_policy: ApprovalPolicy::WritesAndCommands,
            allow_writes: true,
            max_steps: 20,
            team_mode: false,
            builtin: true,
        },
        ProfileSpec {
            id: "autonomous-goal".into(),
            name: "Autonomous Goal".into(),
            description: "Multi-step Goal execution with Agent Team tools enabled".into(),
            mode: Mode::Goal,
            sandbox_mode: SandboxMode::WorkspaceWrite,
            approval_policy: ApprovalPolicy::Never,
            allow_writes: true,
            max_steps: 30,
            team_mode: true,
            builtin: true,
        },
        ProfileSpec {
            id: "research".into(),
            name: "Research".into(),
            description: "Read-only Plan exploration profile".into(),
            mode: Mode::Plan,
            sandbox_mode: SandboxMode::ReadOnly,
            approval_policy: ApprovalPolicy::Never,
            allow_writes: false,
            max_steps: 12,
            team_mode: false,
            builtin: true,
        },
    ]
}

pub(crate) fn validate_profile(p: &ProfileSpec) -> Result<()> {
    if !storage::valid_id(&p.id) {
        bail!("Invalid profile id `{}`", p.id);
    }
    if p.name.trim().is_empty() || p.name.chars().count() > 120 {
        bail!("Profile name must be 1-120 characters");
    }
    if !(1..=50).contains(&p.max_steps) {
        bail!("Profile max_steps must be 1-50");
    }
    Ok(())
}

pub(crate) fn load_custom_profiles(app: &App) -> Result<Vec<ProfileSpec>> {
    state_file::load_list(app.data.as_path(), "profiles", "profile", validate_profile)
}

pub(crate) fn all_profiles(app: &App) -> Result<Vec<ProfileSpec>> {
    let mut merged = builtin_profiles();
    let custom = load_custom_profiles(app)?;
    for mut p in custom {
        p.builtin = false;
        if let Some(slot) = merged.iter_mut().find(|b| b.id == p.id && !b.builtin) {
            *slot = p;
        } else if !merged.iter().any(|b| b.id == p.id) {
            merged.push(p);
        }
    }
    Ok(merged)
}

pub(crate) fn resolve_profile(app: &App, id: &str) -> Result<ProfileSpec> {
    all_profiles(app)?
        .into_iter()
        .find(|p| p.id == id)
        .with_context(|| format!("Unknown profile `{id}`"))
}

pub(crate) fn apply_profile(profile: &ProfileSpec, settings: &mut Settings) {
    settings.mode = profile.mode;
    settings.sandbox_mode = profile.sandbox_mode;
    settings.approval_policy = profile.approval_policy;
    settings.allow_writes = profile.allow_writes;
    settings.max_steps = profile.max_steps;
    settings.team_mode = profile.team_mode;
    settings.permission_preset = Some(profile.id.clone());
}

#[derive(Deserialize)]
pub(crate) struct ProjectQuery {
    #[serde(default)]
    pub project_id: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct BundleImportRequest {
    pub version: u32,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub harness: Tree,
    #[serde(default)]
    pub workflows: Vec<WorkflowSpec>,
    #[serde(default)]
    pub profiles: Vec<ProfileSpec>,
}

pub(crate) fn export_bundle_value(app: &App, project_id: &str) -> Result<Value> {
    let tree = harness::active_tree(&app.data, project_id)?;
    let workflows = workflow::load_workflows(app)?;
    let profiles = load_custom_profiles(app)?;
    Ok(json!({
        "version": 1,
        "format": "allpaka-bundle-v1",
        "project_id": project_id,
        "exported_at_ms": storage::now_ms(),
        "harness": tree,
        "workflows": workflows,
        "profiles": profiles,
    }))
}

pub(crate) fn import_bundle_req(app: &App, req: BundleImportRequest) -> Result<Value> {
    if req.version != 1 {
        bail!("Unsupported bundle version `{}` (expected 1)", req.version);
    }
    let project_id = req.project_id.unwrap_or_else(|| "default".into());
    if !storage::valid_id(&project_id) {
        bail!("Invalid project_id `{project_id}`");
    }
    let project_exists = app
        .projects
        .lock()
        .unwrap()
        .iter()
        .any(|p| p.id == project_id);
    if !project_exists {
        bail!("Unknown project `{project_id}`");
    }

    for wf in &req.workflows {
        workflow::validate_workflow(wf)?;
    }
    for pr in &req.profiles {
        validate_profile(pr)?;
    }

    let mut revision_id = None;
    if !req.harness.is_empty() {
        let origin = harness::active_revision(&app.data, &project_id)?.map(|r| r.id);
        let rev = harness::save_revision(
            &app.data,
            &project_id,
            &req.harness,
            origin,
            "Imported portable bundle",
        )?;
        harness::activate(&app.data, &project_id, Some(&rev.id))?;
        revision_id = Some(rev.id);
    }

    if !req.workflows.is_empty() {
        let mut existing = workflow::load_workflows(app)?;
        let incoming_ids: BTreeSet<String> = req.workflows.iter().map(|w| w.id.clone()).collect();
        existing.retain(|w| !incoming_ids.contains(&w.id));
        existing.extend(req.workflows.clone());
        state_file::save_list(app.data.as_path(), "workflows", &existing)?;
    }

    if !req.profiles.is_empty() {
        let builtin_ids: BTreeSet<String> = builtin_profiles().into_iter().map(|p| p.id).collect();
        let mut custom = load_custom_profiles(app)?;
        for mut pr in req.profiles.clone() {
            if builtin_ids.contains(&pr.id) {
                continue;
            }
            pr.builtin = false;
            custom.retain(|c| c.id != pr.id);
            custom.push(pr);
        }
        if custom.len() > MAX_PROFILES {
            bail!("Custom profiles exceed {MAX_PROFILES}-item limit");
        }
        state_file::save_list(app.data.as_path(), "profiles", &custom)?;
    }

    Ok(json!({
        "imported": true,
        "project_id": project_id,
        "harness_revision": revision_id,
        "workflows_imported": req.workflows.len(),
        "profiles_imported": req.profiles.len(),
    }))
}

pub(crate) async fn list_profiles_api(State(app): State<App>) -> ApiResult<Value> {
    let profiles = all_profiles(&app).map_err(crate::internal)?;
    Ok(Json(json!({ "profiles": profiles })))
}

pub(crate) async fn save_profile_api(
    State(app): State<App>,
    Json(mut profile): Json<ProfileSpec>,
) -> ApiResult<Value> {
    if builtin_profiles().iter().any(|b| b.id == profile.id) {
        return Err(bad_request("Built-in profiles cannot be overwritten"));
    }
    profile.builtin = false;
    validate_profile(&profile).map_err(bad_request)?;
    let mut custom = load_custom_profiles(&app).map_err(crate::internal)?;
    custom.retain(|p| p.id != profile.id);
    if custom.len() >= MAX_PROFILES {
        return Err(bad_request(format!(
            "Maximum {MAX_PROFILES} custom profiles supported"
        )));
    }
    custom.push(profile.clone());
    state_file::save_list(app.data.as_path(), "profiles", &custom).map_err(crate::internal)?;
    Ok(Json(json!({ "profile": profile })))
}

pub(crate) async fn delete_profile_api(
    State(app): State<App>,
    AxumPath(pid): AxumPath<String>,
) -> ApiResult<Value> {
    if builtin_profiles().iter().any(|b| b.id == pid) {
        return Err(bad_request("Built-in profiles cannot be deleted"));
    }
    let mut custom = load_custom_profiles(&app).map_err(crate::internal)?;
    let before = custom.len();
    custom.retain(|p| p.id != pid);
    if custom.len() == before {
        return Err(crate::not_found("Profile not found"));
    }
    state_file::save_list(app.data.as_path(), "profiles", &custom).map_err(crate::internal)?;
    Ok(Json(json!({ "deleted": pid })))
}

pub(crate) async fn export_bundle_api(
    State(app): State<App>,
    Query(q): Query<ProjectQuery>,
) -> ApiResult<Value> {
    let pid = q.project_id.unwrap_or_else(|| "default".into());
    export_bundle_value(&app, &pid)
        .map(Json)
        .map_err(bad_request)
}

pub(crate) async fn import_bundle_api(
    State(app): State<App>,
    Json(req): Json<BundleImportRequest>,
) -> ApiResult<Value> {
    import_bundle_req(&app, req).map(Json).map_err(bad_request)
}

pub(crate) async fn reload_harness_api(
    State(app): State<App>,
    body: Option<Json<ProjectQuery>>,
) -> ApiResult<Value> {
    let pid = body
        .and_then(|Json(q)| q.project_id)
        .unwrap_or_else(|| "default".into());
    let tree = harness::active_tree(&app.data, &pid).map_err(crate::internal)?;
    crate::plugins::reload_all(&app);
    Ok(Json(json!({
        "reloaded": true,
        "project_id": pid,
        "harness_files": tree.len(),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Project, Root};
    use crate::ext::test_app;

    #[test]
    fn bundle_export_import_roundtrip_and_builtin_profile() {
        let root_dir =
            std::env::temp_dir().join(format!("allpaka-bundle-test-{}", storage::now_ms()));
        let ws = root_dir.join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let ws = ws.canonicalize().unwrap();
        let data = root_dir.join("data");
        let app = test_app(
            &data,
            &ws,
            vec![Project {
                id: "default".into(),
                name: "default".into(),
                roots: vec![Root {
                    alias: "workspace".into(),
                    path: ws.clone(),
                    writable: true,
                    repository: false,
                    ssh: None,
                }],
                instructions: String::new(),
            }],
        );

        assert_eq!(resolve_profile(&app, "coding-safe").unwrap().max_steps, 20);

        let mut tree = Tree::new();
        tree.insert("system.md".into(), "You are a strict verifier.".into());
        let bundle_req = BundleImportRequest {
            version: 1,
            project_id: Some("default".into()),
            harness: tree,
            workflows: Vec::new(),
            profiles: vec![ProfileSpec {
                id: "custom-fast".into(),
                name: "Custom Fast".into(),
                description: "Fast".into(),
                mode: Mode::Auto,
                sandbox_mode: SandboxMode::ReadOnly,
                approval_policy: ApprovalPolicy::Never,
                allow_writes: false,
                max_steps: 6,
                team_mode: false,
                builtin: false,
            }],
        };
        let res = import_bundle_req(&app, bundle_req).unwrap();
        assert_eq!(res["imported"], true);
        assert!(res["harness_revision"].as_str().is_some());

        let exported = export_bundle_value(&app, "default").unwrap();
        assert_eq!(
            exported["harness"]["system.md"],
            "You are a strict verifier."
        );
        assert_eq!(exported["profiles"][0]["id"], "custom-fast");
        let _ = std::fs::remove_dir_all(&root_dir);
    }
}
