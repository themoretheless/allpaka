//! Named runtime profiles and their fully resolved backend policy.

use crate::runtime::RuntimePolicy;
use std::collections::BTreeMap;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeProfile {
    Auto,
    Safe,
    Balanced,
    MaxPerformance,
    Deterministic,
}

impl RuntimeProfile {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Safe => "safe",
            Self::Balanced => "balanced",
            Self::MaxPerformance => "max-performance",
            Self::Deterministic => "deterministic",
        }
    }

    pub fn resolve(self) -> ResolvedRuntime {
        let policy = match self {
            Self::Auto | Self::Balanced => RuntimePolicy::default(),
            Self::Safe => RuntimePolicy {
                normflag: false,
                attention_split: None,
                decode_serial: true,
                prefill_defer: false,
                prefill_one_buffer: false,
                gpu_route: true,
                mm_pipeline: false,
            },
            Self::MaxPerformance => RuntimePolicy {
                normflag: true,
                attention_split: None,
                decode_serial: false,
                prefill_defer: true,
                prefill_one_buffer: true,
                gpu_route: true,
                mm_pipeline: true,
            },
            Self::Deterministic => RuntimePolicy {
                normflag: false,
                attention_split: None,
                decode_serial: true,
                prefill_defer: false,
                prefill_one_buffer: true,
                gpu_route: true,
                mm_pipeline: false,
            },
        };
        ResolvedRuntime {
            profile: self,
            policy,
            overrides: BTreeMap::new(),
        }
    }
}

impl FromStr for RuntimeProfile {
    type Err = UnknownRuntimeProfile;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "safe" => Ok(Self::Safe),
            "balanced" => Ok(Self::Balanced),
            "max-performance" | "max_performance" | "max" => Ok(Self::MaxPerformance),
            "deterministic" => Ok(Self::Deterministic),
            _ => Err(UnknownRuntimeProfile(value.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnknownRuntimeProfile(pub String);

impl std::fmt::Display for UnknownRuntimeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown runtime profile {:?}; expected auto, safe, balanced, max-performance, or deterministic",
            self.0
        )
    }
}

impl std::error::Error for UnknownRuntimeProfile {}

#[derive(Debug, Clone)]
pub struct ResolvedRuntime {
    pub profile: RuntimeProfile,
    pub policy: RuntimePolicy,
    pub overrides: BTreeMap<String, String>,
}

impl ResolvedRuntime {
    pub fn override_bool(mut self, name: &'static str, value: bool) -> Self {
        match name {
            "normflag" => self.policy.normflag = value,
            "decode_serial" => self.policy.decode_serial = value,
            "prefill_defer" => self.policy.prefill_defer = value,
            "prefill_one_buffer" => self.policy.prefill_one_buffer = value,
            "gpu_route" => self.policy.gpu_route = value,
            "mm_pipeline" => self.policy.mm_pipeline = value,
            _ => return self,
        }
        self.overrides.insert(name.to_string(), value.to_string());
        self
    }

    pub fn override_attention_split(mut self, value: Option<usize>) -> Self {
        self.policy.attention_split = value;
        self.overrides
            .insert("attention_split".to_string(), format!("{value:?}"));
        self
    }

    pub fn report_lines(&self) -> Vec<String> {
        let p = &self.policy;
        vec![
            format!("profile={}", self.profile.name()),
            format!("normflag={}", p.normflag),
            format!("attention_split={:?}", p.attention_split),
            format!("decode_serial={}", p.decode_serial),
            format!("prefill_defer={}", p.prefill_defer),
            format!("prefill_one_buffer={}", p.prefill_one_buffer),
            format!("gpu_route={}", p.gpu_route),
            format!("mm_pipeline={}", p.mm_pipeline),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeProfile;

    #[test]
    fn safe_and_max_performance_resolve_differently() {
        let safe = RuntimeProfile::Safe.resolve();
        let fast = RuntimeProfile::MaxPerformance.resolve();
        assert!(safe.policy.decode_serial);
        assert!(!fast.policy.decode_serial);
        assert!(!safe.policy.mm_pipeline);
        assert!(fast.policy.mm_pipeline);
    }

    #[test]
    fn resolved_report_includes_overrides() {
        let resolved = RuntimeProfile::Auto
            .resolve()
            .override_bool("decode_serial", true);
        assert_eq!(resolved.overrides["decode_serial"], "true");
        assert!(resolved
            .report_lines()
            .iter()
            .any(|line| line == "decode_serial=true"));
    }
}
