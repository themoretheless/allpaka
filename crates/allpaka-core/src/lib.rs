//! Core types and planning logic for splitting a language model across
//! several machines.
//!
//! This crate is deliberately pure: no I/O, no processes, no sockets. That
//! makes the cost model testable, which matters because the whole point of the
//! project is to answer "is splitting worth it here?" with numbers rather than
//! with hope.

mod cost;
pub mod fabric;
pub mod fleet;
pub mod link;
pub mod model;
pub mod node;
pub mod plan;
pub mod presets;
pub mod replicate;
pub mod speculation;

pub use fabric::Fabric;
pub use fleet::{fleet, FleetPlan, Placement};
pub use replicate::{replicate, Replica, ReplicaPlan};
pub use speculation::{SpeculativeCost, Speculation};
pub use link::Link;
pub use model::Model;
pub use node::{Backend, Node};
pub use plan::{gib, plan, Plan, PlanRequest, Stage, Verdict};
