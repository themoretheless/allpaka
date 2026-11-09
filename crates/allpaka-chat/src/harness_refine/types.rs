use super::*;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What the algorithm needs from the outside world. Studio runs real turns
/// and model calls; tests script both.
pub trait Engine: Send + Sync {
    /// Execute the task once under `tree`.
    fn rollout<'a>(&'a self, label: &'a str, tree: &'a Tree) -> BoxFuture<'a, Result<RunView>>;
    /// One model call without tools; returns the reply text.
    fn call<'a>(
        &'a self,
        label: &'a str,
        system: &'a str,
        user: &'a str,
    ) -> BoxFuture<'a, Result<String>>;
    /// An agent session that edits `current` (its `harness` root) while
    /// reading `files` (its read-only `context` root). Returns the edited
    /// tree and the agent's notes.
    fn edit<'a>(
        &'a self,
        _label: &'a str,
        _system: &'a str,
        _mission: &'a str,
        _files: &'a BTreeMap<String, String>,
        _current: &'a Tree,
    ) -> BoxFuture<'a, Result<(Tree, String)>> {
        Box::pin(async { bail!("This engine cannot run agentic proposers") })
    }
    /// A read-only agent session over `files` (as `harness_evidence`) and
    /// the project's own roots; returns its final answer. Engines without
    /// agents get the files inlined into one call.
    fn inspect<'a>(
        &'a self,
        label: &'a str,
        system: &'a str,
        mission: &'a str,
        files: &'a BTreeMap<String, String>,
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let inlined: String = files
                .iter()
                .map(|(p, t)| format!("=== {p} ===\n{t}\n"))
                .collect();
            self.call(label, system, &format!("{mission}\n\n{inlined}"))
                .await
        })
    }
}

/// One execution. It never carries a score: `delivered` only says that the
/// run finished with an answer.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RunView {
    pub label: String,
    /// Content hash of the harness tree, retained separately from the cache key.
    pub fingerprint: String,
    pub delivered: bool,
    pub outcome: String,
    pub answer: String,
    /// Diff of the files the run changed against the task workspace snapshot.
    pub changes: String,
    pub trajectory: String,
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub workspace_fingerprint: Option<String>,
    #[serde(default)]
    pub trace_id: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub usage: Option<Value>,
    #[serde(default)]
    pub error_class: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Stage {
    pub label: String,
    pub dimension: String,
    pub tree: Tree,
    pub notes: String,
    /// The proposal was rejected twice and the inherited tree was kept.
    pub fell_back: bool,
    #[serde(default)]
    pub error: Option<String>,
    /// Hash of everything the stage was asked; a resumed job reuses the
    /// saved stage only when its inputs are unchanged.
    #[serde(default)]
    pub key: String,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Recursion {
    #[default]
    Parallel,
    Sequential,
}

pub struct Plan<'a> {
    pub engine: &'a dyn Engine,
    pub dir: PathBuf,
    pub k: usize,
    pub mode: Recursion,
    pub task: String,
    pub base: Tree,
    /// Execution context shared by every rollout of this task. Deliberately
    /// excludes `k` and recursion mode, which do not change a single rollout.
    pub cache_namespace: String,
    /// Content identity of the frozen writable roots, recorded even when the
    /// executor fails before it can return its own RunView.
    pub workspace_fingerprint: String,
    /// Library entries of earlier tasks, already rendered.
    pub library: String,
    pub width: usize,
    /// Proposers and composers run as agents editing files (as in Harness²)
    /// instead of answering with one JSON object.
    pub agentic: bool,
    /// Optional rubric for reporting only: the method never sees scores.
    pub rubric: Option<String>,
    /// Fresh Base/Champion pairs after selection. Zero disables validation.
    pub validation_runs: usize,
    /// Expected output from a dataset sample: given to the rubric judge
    /// only, never to proposers, composers, selectors or comparators.
    pub reference: Option<String>,
    pub progress: &'a (dyn Fn(String) + Send + Sync),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Outcome {
    pub mode: Recursion,
    pub k: usize,
    pub chosen: String,
    /// `final` when a composed harness was chosen, `base` when every final
    /// failed to deliver and the unmodified harness stays.
    pub chosen_kind: String,
    pub selection_why: String,
    pub tree: Tree,
    pub finals: Vec<String>,
    pub similarity: Option<f64>,
    pub fallbacks: Vec<String>,
    pub rollouts: usize,
    pub stages: usize,
    pub runs: Vec<RunSummary>,
    pub stage_notes: BTreeMap<String, String>,
    pub reflection: Option<Value>,
    /// Judge scores by rollout label, when a rubric was given. Reported
    /// after selection and never fed back into it.
    #[serde(default)]
    pub scores: BTreeMap<String, Option<f64>>,
    /// Independent post-selection validation. Missing only in legacy outcomes.
    #[serde(default)]
    pub validation: Option<ValidationReport>,
    #[serde(default)]
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSummary {
    pub label: String,
    pub delivered: bool,
    pub outcome: String,
    #[serde(default)]
    pub trace_id: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub error_class: Option<String>,
    /// Fresh validation score, when a rubric was supplied. Search scores stay
    /// in `Outcome::scores` and are never shown to the comparator.
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

pub(super) fn summarize(run: &RunView) -> RunSummary {
    RunSummary {
        label: run.label.clone(),
        delivered: run.delivered,
        outcome: run.outcome.clone(),
        trace_id: run.trace_id.clone(),
        duration_ms: run.duration_ms,
        steps: run.steps,
        error_class: run.error_class.clone(),
        score: None,
        warnings: run.warnings.clone(),
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    NotRun,
    Unchanged,
    ImprovementObserved,
    NoObservedDifference,
    RegressionObserved,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PairVerdict {
    Base,
    Champion,
    Tie,
    Inconclusive,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationPair {
    pub index: usize,
    pub base: RunSummary,
    pub champion: RunSummary,
    /// `None` means the comparator failed and can be retried on resume.
    pub verdict: Option<PairVerdict>,
    pub why: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationReport {
    pub status: ValidationStatus,
    pub champion: String,
    pub base_tree: String,
    pub champion_tree: String,
    pub requested_runs: usize,
    pub rollout_count: usize,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    pub pairs: Vec<ValidationPair>,
    pub reason: String,
    pub warnings: Vec<String>,
}
