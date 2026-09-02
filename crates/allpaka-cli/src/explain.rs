//! Explainable execution plans shared by CLI diagnostics and benchmarks.

use allpaka_backend::capability::{BackendCapabilities, Feature};
use allpaka_backend::profile::ResolvedRuntime;
use allpaka_model::requirements::ModelRequirements;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorCoverage {
    pub tensor_type: String,
    pub tensors: usize,
    pub bytes: u64,
    pub kernel: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanStep {
    pub operation: String,
    pub implementation: String,
    pub supported: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub architecture: String,
    pub runtime_profile: String,
    pub supported: bool,
    pub steps: Vec<PlanStep>,
    pub tensors: Vec<TensorCoverage>,
    pub resolved_runtime: Vec<String>,
}

impl ExecutionPlan {
    pub fn resolve(
        requirements: &ModelRequirements,
        backend: &BackendCapabilities,
        runtime: &ResolvedRuntime,
        tensors: Vec<TensorCoverage>,
    ) -> Self {
        let coverage = backend.coverage(&requirements.required);
        let mut steps = requirements
            .required
            .iter()
            .copied()
            .map(|feature| feature_step(feature, backend.supports(feature)))
            .collect::<Vec<_>>();
        for tensor in &tensors {
            if tensor.kernel.is_none() {
                steps.push(PlanStep {
                    operation: format!("tensor-type:{}", tensor.tensor_type),
                    implementation: "none".into(),
                    supported: false,
                    reason: Some(format!(
                        "{} tensor(s), {} bytes have no accelerator kernel",
                        tensor.tensors, tensor.bytes
                    )),
                });
            }
        }
        let tensors_supported = tensors.iter().all(|tensor| tensor.kernel.is_some());
        Self {
            architecture: requirements.architecture.clone(),
            runtime_profile: runtime.profile.name().to_string(),
            supported: coverage.is_supported() && tensors_supported,
            steps,
            tensors,
            resolved_runtime: runtime.report_lines(),
        }
    }

    pub fn render_text(&self) -> String {
        let mut lines = vec![
            format!("architecture: {}", self.architecture),
            format!("profile: {}", self.runtime_profile),
            format!("supported: {}", self.supported),
        ];
        for step in &self.steps {
            let status = if step.supported { "yes" } else { "no" };
            let reason = step
                .reason
                .as_deref()
                .map(|reason| format!(" ({reason})"))
                .unwrap_or_default();
            lines.push(format!(
                "{:20} {:24} supported={status}{reason}",
                step.operation, step.implementation
            ));
        }
        lines.push("resolved runtime:".into());
        lines.extend(
            self.resolved_runtime
                .iter()
                .map(|setting| format!("  {setting}")),
        );
        lines.join("\n")
    }
}

pub fn run(path: &Path, profile: &str, json: bool) -> anyhow::Result<()> {
    let file = allpaka_gguf::GgufFile::open(path)?;
    let requirements = ModelRequirements::for_architecture(file.architecture())
        .ok_or_else(|| anyhow::anyhow!("no model requirements registered for {}", file.architecture()))?;
    let profile: allpaka_backend::profile::RuntimeProfile = profile.parse()?;
    let mut census: BTreeMap<String, (usize, u64, bool)> = BTreeMap::new();
    for tensor in file.tensors() {
        let tensor_type = format!("{:?}", tensor.ggml_type);
        let bytes = tensor.byte_size()?;
        let has_kernel = !tensor_type.starts_with("Other(");
        let entry = census.entry(tensor_type).or_insert((0, 0, has_kernel));
        entry.0 += 1;
        entry.1 += bytes;
        entry.2 &= has_kernel;
    }
    let tensors = census
        .into_iter()
        .map(|(tensor_type, (tensors, bytes, has_kernel))| TensorCoverage {
            kernel: has_kernel.then(|| format!("metal.{}", tensor_type.to_ascii_lowercase())),
            tensor_type,
            tensors,
            bytes,
        })
        .collect();
    let plan = ExecutionPlan::resolve(
        &requirements,
        &BackendCapabilities::current_metal(),
        &profile.resolve(),
        tensors,
    );
    if json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        println!("{}", plan.render_text());
    }
    Ok(())
}

fn feature_step(feature: Feature, supported: bool) -> PlanStep {
    let (operation, implementation) = match feature {
        Feature::DenseFfn => ("ffn", "metal.dense"),
        Feature::SparseMoe => ("experts", "metal.moe.grouped"),
        Feature::GqaAttention => ("attention", "metal.gqa"),
        Feature::QkNorm => ("qk-norm", "metal.norm"),
        Feature::StandardKv => ("kv", "metal.standard-kv"),
        Feature::GatedDeltaNet => ("gdn", "metal.gdn"),
        Feature::MlaAttention => ("attention", "metal.mla"),
        Feature::F16Kv => ("kv-format", "f16"),
        Feature::PagedKv => ("kv-layout", "paged"),
    };
    PlanStep {
        operation: operation.into(),
        implementation: implementation.into(),
        supported,
        reason: (!supported).then(|| format!("missing backend capability {feature:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionPlan, TensorCoverage};
    use allpaka_backend::capability::BackendCapabilities;
    use allpaka_backend::profile::RuntimeProfile;
    use allpaka_model::requirements::ModelRequirements;

    #[test]
    fn explain_plan_names_the_exact_missing_capability() {
        let requirements = ModelRequirements::for_architecture("deepseek_mla").unwrap();
        let plan = ExecutionPlan::resolve(
            &requirements,
            &BackendCapabilities::current_metal(),
            &RuntimeProfile::Safe.resolve(),
            vec![TensorCoverage {
                tensor_type: "Q4_K".into(),
                tensors: 10,
                bytes: 1024,
                kernel: Some("metal.q4k".into()),
            }],
        );
        assert!(!plan.supported);
        assert!(plan.render_text().contains("MlaAttention"));
        assert!(plan.render_text().contains("decode_serial=true"));
    }
}
