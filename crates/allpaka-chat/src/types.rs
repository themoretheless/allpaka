use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Chat,
    Plan,
    Auto,
    Goal,
    Swarm,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HistoryFolder {
    #[default]
    Active,
    Archived,
    Trash,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verbosity {
    Brief,
    #[default]
    Normal,
    Detailed,
    Maximum,
}
impl Verbosity {
    pub fn instruction(self) -> &'static str {
        match self {
            Self::Brief => "Response detail: brief. Give the essential answer without unnecessary explanation.",
            Self::Normal => "Response detail: normal. Include the explanation needed to understand the answer.",
            Self::Detailed => "Response detail: detailed. Explain relevant steps, reasons and examples.",
            Self::Maximum => "Response detail: maximum. Give a thorough structured answer with relevant reasoning, examples, alternatives and limitations. Avoid repetition and filler.",
        }
    }
}

/// One swarm participant: its own persona, provider and model.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SwarmMember {
    /// Short name used for citations and the UI, e.g. `safety`.
    #[serde(default)]
    pub label: String,
    /// Persona and angle this member is responsible for.
    #[serde(default)]
    pub role: String,
    pub provider: String,
    pub model: String,
}

/// One agent's report inside a swarm turn. Stored with the assistant message
/// so the transcript stays a single message with expandable per-agent sections.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SwarmReport {
    pub label: String,
    pub provider: String,
    pub model: String,
    #[serde(default = "default_swarm_rounds")]
    pub round: u8,
    /// queued | running | done | error | cancelled
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Swarm settings: who investigates in parallel, how many waves, who merges.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SwarmConfig {
    #[serde(default)]
    pub members: Vec<SwarmMember>,
    /// Waves of member passes. Wave 2 sees the other members' wave-1 reports.
    #[serde(default = "default_swarm_rounds")]
    pub rounds: u8,
    /// Empty means "use the session provider/model" for the merge pass.
    #[serde(default)]
    pub synthesis_provider: String,
    #[serde(default)]
    pub synthesis_model: String,
    /// One adversarial pass over the merged draft before it is shown.
    #[serde(default)]
    pub critic: bool,
    /// Read-only tool-loop steps allowed per member per wave.
    #[serde(default = "default_swarm_steps")]
    pub max_steps_per_member: usize,
    /// Per-member report budget handed to the synthesizer, in bytes.
    #[serde(default = "default_swarm_report_bytes")]
    pub report_bytes: usize,
}
impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            members: vec![],
            rounds: default_swarm_rounds(),
            synthesis_provider: String::new(),
            synthesis_model: String::new(),
            critic: false,
            max_steps_per_member: default_swarm_steps(),
            report_bytes: default_swarm_report_bytes(),
        }
    }
}
impl SwarmConfig {
    /// Provider requests one swarm turn costs: members × waves + merge (+critic).
    pub fn request_count(&self) -> usize {
        self.members.len() * self.rounds.max(1) as usize + 1 + usize::from(self.critic)
    }
}
fn default_swarm_rounds() -> u8 {
    1
}
fn default_swarm_steps() -> usize {
    2
}
fn default_swarm_report_bytes() -> usize {
    6000
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Settings {
    #[serde(default)]
    pub verbosity: Verbosity,
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
    #[serde(default)]
    pub swarm: SwarmConfig,
    #[serde(default = "default_auto_compact")]
    pub auto_compact: bool,
    #[serde(default = "default_compact_threshold")]
    pub compact_threshold: usize,
    #[serde(default)]
    pub json_mode: bool,
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
    /// Per-agent reports of a Swarm-mode turn. Never sent to a provider.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub swarm: Vec<SwarmReport>,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Session {
    pub id: String,
    #[serde(default)]
    pub folder: HistoryFolder,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<BranchOrigin>,
    #[serde(default)]
    pub compaction: Option<Compaction>,
    pub title: String,
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
            folder: HistoryFolder::default(),
            parent: None,
            compaction: None,
            title: "Новый чат".into(),
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
