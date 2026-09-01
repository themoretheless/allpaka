//! Immutable acceleration policy.
//!
//! Environment compatibility is confined to construction. Execution code
//! reads one policy snapshot, so effective settings cannot change depending
//! on which kernel happened to initialise its own `OnceLock` first.

use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct RuntimePolicy {
    pub normflag: bool,
    pub attention_split: Option<usize>,
    pub decode_serial: bool,
    pub prefill_defer: bool,
    pub prefill_one_buffer: bool,
    pub gpu_route: bool,
    pub mm_pipeline: bool,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self {
            normflag: true,
            attention_split: None,
            decode_serial: false,
            prefill_defer: true,
            prefill_one_buffer: true,
            gpu_route: true,
            mm_pipeline: true,
        }
    }
}

impl RuntimePolicy {
    pub fn from_env() -> Self {
        let mut policy = Self::default();
        policy.normflag = !std::env::var("ALLPAKA_NORMFLAG").is_ok_and(|v| v == "0");
        policy.attention_split = std::env::var("ALLPAKA_ATTN_SPLIT")
            .ok()
            .and_then(|v| v.parse().ok());
        policy.decode_serial = std::env::var("ALLPAKA_DECODE_SERIAL").is_ok_and(|v| v == "1");
        policy.prefill_defer = std::env::var("ALLPAKA_PF_DEFER").map_or(true, |v| v != "0");
        policy.prefill_one_buffer =
            std::env::var("ALLPAKA_PF_ONEBUF").map_or(true, |v| v != "0");
        policy.gpu_route = std::env::var("ALLPAKA_GPU_ROUTE").map_or(true, |v| v != "0");
        policy.mm_pipeline = std::env::var("ALLPAKA_MM_PIPE").map_or(true, |v| v != "0");
        policy
    }
}

static POLICY: OnceLock<RuntimePolicy> = OnceLock::new();

pub fn get() -> &'static RuntimePolicy {
    POLICY.get_or_init(RuntimePolicy::from_env)
}

pub fn install(policy: RuntimePolicy) -> Result<(), RuntimePolicy> {
    POLICY.set(policy)
}

#[cfg(test)]
mod tests {
    use super::RuntimePolicy;

    #[test]
    fn production_defaults_are_explicit() {
        let p = RuntimePolicy::default();
        assert!(p.normflag && p.prefill_defer && p.prefill_one_buffer);
        assert!(p.gpu_route && p.mm_pipeline);
        assert!(!p.decode_serial);
        assert_eq!(p.attention_split, None);
    }
}
