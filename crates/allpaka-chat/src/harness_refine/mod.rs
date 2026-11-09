//! Task-level recursion: Propose -> Probe -> Compose (after Harness²).
//!
//! A job takes the last task of a session and refines the project harness on
//! it without any ground truth:
//!
//! 1. `k` base rollouts run the task under the current harness.
//! 2. Proposers edit the harness on two dimensions, INV (how the agent works)
//!    and DOM (what a practitioner knows); every candidate is probed once.
//! 3. A composer merges, adopts or keeps from the behavioral contrasts.
//!
//! Parallel mode probes `k` INV and `k` DOM candidates and composes `k`
//! finals from the shared pool; sequential mode carries one checkpoint
//! through `k` rounds. Either way the budget is 4k rollouts and 3k proposer
//! stages. A blind selector picks among the finals by deliverable alone, the
//! reflector writes what each edit did, and the chosen tree is stored as a
//! revision that the user adopts explicitly.
//!
//! Rollouts run in headless sessions on copies of the project's writable
//! roots, so a probe never touches the real workspace. Every rollout and
//! stage is saved under the job directory, and a resumed job reuses them.

use crate::{
    context, error, harness, harness::Tree, harness_library, harness_prompts as prompts, types::*,
    ApiResult, App,
};
use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
};


mod engine;
mod http;
mod store;
mod types;
mod validation;
mod workspace;
#[cfg(test)]
mod tests;

pub use self::engine::*;
pub(crate) use self::http::*;
pub(crate) use self::store::*;
pub use self::types::*;
use self::validation::*;
pub use self::workspace::*;
