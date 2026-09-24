//! RAG plugin plumbing: the one connected knowledge source, its prompt
//! injections, and the periodic maintenance pass.
//!
//! The registry lookup here is the single definition of "which plugin is the
//! RAG one" -- `turn()` and the maintenance loop both go through it.

use crate::mcp;
use crate::App;
use serde_json::{json, Value};
use std::sync::Arc;

fn rag_client(app: &App, rag_id: &str) -> Option<Arc<mcp::Client>> {
    app.plugins.read().unwrap().get(rag_id)?.client.clone()
}
pub(crate) async fn retrieve_rag_wakeup(app: &App, rag_id: &str) -> Option<String> {
    let client = rag_client(app, rag_id)?;
    let value = client.call("wake_up", &json!({})).await.ok()?;
    let text = match &value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    Some(text.chars().take(1500).collect())
}
pub(crate) async fn retrieve_rag_context(app: &App, rag_id: &str, query: &str) -> Option<String> {
    let client = rag_client(app, rag_id)?;
    if let Ok(value) = client
        .call("search_wiki", &json!({"query": query, "top_k": 6, "mode": "vec"}))
        .await
    {
        if let Some(hits) = value.as_array() {
            let lines: Vec<String> = hits
                .iter()
                .filter_map(|hit| {
                    let title = hit["document_title"].as_str().unwrap_or("");
                    let uri = hit["document_uri"].as_str().unwrap_or("");
                    let content = hit["content"].as_str().unwrap_or("");
                    if content.is_empty() {
                        return None;
                    }
                    let snippet: String = content.chars().take(600).collect();
                    Some(format!("- {title} ({uri})\n  {snippet}"))
                })
                .collect();
            if !lines.is_empty() {
                return Some(lines.join("\n"));
            }
        }
    }
    if let Ok(value) = client
        .call("query_with_index", &json!({"query": query, "top_k": 6}))
        .await
    {
        if let Some(matches) = value["matches"].as_array() {
            let lines: Vec<String> = matches
                .iter()
                .filter_map(|m| {
                    let title = m["entry"]["title"].as_str().unwrap_or("");
                    let slug = m["entry"]["slug"].as_str().unwrap_or("");
                    let summary = m["entry"]["summary"].as_str().unwrap_or("");
                    if summary.is_empty() {
                        return None;
                    }
                    Some(format!("- {title} (wiki://{slug}): {summary}"))
                })
                .collect();
            if !lines.is_empty() {
                return Some(lines.join("\n"));
            }
        }
    }
    None
}
pub(crate) async fn auto_file_answer(app: &App, rag_id: &str, title: &str, body: &str) {
    let Some(client) = rag_client(app, rag_id) else { return };
    let _ = client
        .call(
            "file_answer",
            &json!({"title": title, "body": body, "agent": "allpaka-studio"}),
        )
        .await;
}
pub(crate) fn connected_rag_id(app: &App) -> Option<String> {
    let registry = app.plugins.read().unwrap();
    registry
        .iter()
        .find(|(id, state)| {
            state.client.is_some()
                && (id.to_lowercase().contains("rag")
                    || state.config.name.to_lowercase().contains("rag"))
        })
        .map(|(id, _)| id.clone())
}
pub(crate) fn spawn_rag_maintenance(app: App) {
    let enabled = std::env::var("ALLPAKA_RAG_AUTO_MAINTENANCE")
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    if !enabled {
        return;
    }
    let interval_secs = std::env::var("ALLPAKA_RAG_AUTO_MAINTENANCE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(86400);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
            run_rag_maintenance(&app).await;
        }
    });
}
async fn run_rag_maintenance(app: &App) {
    let Some(rag_id) = connected_rag_id(app) else { return };
    let Some(client) = rag_client(app, &rag_id) else { return };
    eprintln!("rag maintenance: analyze_corpus");
    let analysis = client.call("analyze_corpus", &json!({})).await.ok();
    let plan = match client
        .call(
            "plan_maintenance",
            &json!({"force_heuristic": true, "analysis": analysis}),
        )
        .await
    {
        Ok(value) => value,
        Err(err) => {
            eprintln!("rag maintenance: plan failed: {err:#}");
            return;
        }
    };
    let actions = plan["actions"].clone();
    if actions.is_null() {
        eprintln!("rag maintenance: no actions");
        return;
    }
    let apply = std::env::var("ALLPAKA_RAG_AUTO_MAINTENANCE_APPLY")
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    if apply {
        let _ = client
            .call(
                "apply_maintenance_plan",
                &json!({"actions": actions, "dry_run": false}),
            )
            .await;
        let _ = client
            .call("maintain_refresh", &json!({"dry_run": false}))
            .await;
    } else {
        let _ = client
            .call(
                "apply_maintenance_plan",
                &json!({"actions": actions, "dry_run": true}),
            )
            .await;
        eprintln!(
            "rag maintenance: dry-run only; set ALLPAKA_RAG_AUTO_MAINTENANCE_APPLY=true to apply"
        );
    }
}

