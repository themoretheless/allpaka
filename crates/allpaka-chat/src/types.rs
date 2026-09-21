use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Chat,
    Plan,
    Auto,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Settings {
    #[serde(default = "default_project")]
    pub project_id: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub mode: Mode,
    #[serde(default = "default_steps")]
    pub max_steps: usize,
    #[serde(default = "default_output_tokens")]
    pub max_output_tokens: u32,
    #[serde(default)]
    pub allow_writes: bool,
    #[serde(default = "default_auto_compact")]
    pub auto_compact: bool,
    #[serde(default = "default_compact_threshold")]
    pub compact_threshold: usize,
}
fn default_auto_compact() -> bool {
    true
}
fn default_compact_threshold() -> usize {
    24000
}
fn default_project() -> String {
    "default".into()
}
fn default_output_tokens() -> u32 {
    8192
}
fn default_steps() -> usize {
    12
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Message {
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub incomplete_tool_calls: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_details: Vec<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<crate::context::Image>,
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}
impl Message {
    pub fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Pending {
    #[serde(default)]
    pub images: Vec<crate::context::Image>,
    pub text: String,
    pub settings: Settings,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlanItem {
    pub title: String,
    pub status: String,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HistoryFolder {
    #[default]
    Active,
    Archived,
    Trash,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Session {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<BranchOrigin>,
    #[serde(default)]
    pub compaction: Option<Compaction>,
    pub title: String,
    #[serde(default)]
    pub folder: HistoryFolder,
    pub messages: Vec<Message>,
    pub settings: Settings,
    pub status: String,
    pub queue: Vec<Pending>,
    pub steering: Vec<String>,
    pub plan: Vec<PlanItem>,
    pub error: Option<String>,
    #[serde(default)]
    pub notice: Option<String>,
    pub step: usize,
    pub usage: Value,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Compaction {
    pub summary: String,
    pub through: usize,
    pub provider: String,
    pub model: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BranchOrigin {
    pub session_id: String,
    pub message_count: usize,
}
impl Session {
    pub fn new(id: String, settings: Settings) -> Self {
        Self {
            id,
            parent: None,
            compaction: None,
            title: "Новый чат".into(),
            folder: HistoryFolder::Active,
            messages: vec![],
            settings,
            status: "idle".into(),
            queue: vec![],
            steering: vec![],
            plan: vec![],
            error: None,
            notice: None,
            step: 0,
            usage: Value::Null,
        }
    }
    // Every emitted tool call must have a result, including interrupted batches.
    pub fn close_pending_tools(&mut self) {
        let calls: Vec<String> = self
            .messages
            .iter()
            .flat_map(|m| m.tool_calls.iter())
            .filter_map(|c| c["id"].as_str().map(str::to_owned))
            .collect();
        for id in calls {
            if !self
                .messages
                .iter()
                .any(|m| m.tool_call_id.as_ref() == Some(&id))
            {
                self.messages.push(Message {
                    role: "tool".into(),
                    content: "Tool cancelled before completion.".into(),
                    tool_call_id: Some(id),
                    ..Message::default()
                });
            }
        }
    }
}
