//! GGUF weight loading, stage slicing, and BF16-to-F32 router weight conversion.

use super::{
    gdn::GdnLayer,
    moe::{Ffn, SharedFfn},
    mtp::MtpLayer,
    AttnBias, AttnWeights, Layer, Model,
};
use crate::{
    config::{Config, MoeConfig},
    error::ModelError,
};
use allpaka_backend::{ops, quantmat::QuantMat};
use allpaka_gguf::GgufFile;
use anyhow::Result;

impl<'a> Model<'a> {
    pub fn load(f: &'a GgufFile) -> Result<Self> {
        let count = Config::from_gguf(f)?.n_layers as usize;
        Self::load_range(f, 0, count, false)
    }

    /// Load only a stage's tensors. Unused layers and endpoint tensors are
    /// never decoded. Mapping a file does not make its pages resident.
    pub fn load_stage(f: &'a GgufFile, first: usize, end: usize) -> Result<Self> {
        Self::load_range(f, first, end, true)
    }

    pub(crate) fn embedding(&self) -> Result<&QuantMat<'a>> {
        self.embd
            .as_ref()
            .ok_or_else(|| ModelError::MissingStageEndpoint("embedding").into())
    }

    pub(crate) fn output_head(&self) -> Result<&QuantMat<'a>> {
        self.output
            .as_ref()
            .ok_or_else(|| ModelError::MissingStageEndpoint("output head").into())
    }

    pub(crate) fn load_range(
        f: &'a GgufFile,
        first: usize,
        end: usize,
        stage: bool,
    ) -> Result<Self> {
        let config = Config::from_gguf(f)?;
        if first >= end || end > config.n_layers as usize {
            return Err(ModelError::InvalidStageRange { first, end }.into());
        }
        let capability = config.capability();
        if capability.support == crate::config::ModelSupport::Unsupported {
            return Err(ModelError::UnsupportedArchitecture {
                architecture: config.architecture.to_string(),
                reason: capability.reason,
            }
            .into());
        }
        // Offer the weights to the GPU; on a machine without Metal this is a
        // no-op and every matmul stays on the CPU reference. Split files get
        // one GPU window set per part.
        if !stage {
            for m in f.mappings() {
                allpaka_backend::gpu::attach(m);
            }
        } else {
            let selected = f.selected_mappings(|tensor| {
                if let Some(rest) = tensor.name.strip_prefix("blk.") {
                    return rest
                        .split('.')
                        .next()
                        .and_then(|n| n.parse::<usize>().ok())
                        .is_some_and(|index| index >= first && index < end);
                }
                (first == 0 && tensor.name == "token_embd.weight")
                    || (end == config.n_layers as usize
                        && (tensor.name == "output_norm.weight"
                            || tensor.name == "output.weight"
                            || (tensor.name == "token_embd.weight"
                                && f.tensor("output.weight").is_none())))
            })?;
            for mapping in selected {
                allpaka_backend::gpu::attach(mapping);
            }
        }
        let hidden = config.hidden as usize;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();

        let mut owned_weights: Vec<Box<[f32]>> = Vec::new();
        let mut layers = Vec::with_capacity(end - first);
        for i in first as u32..end as u32 {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            // GLM calls the pre-FFN norm "post_attention_norm".
            let ffn_norm = if f.tensor(&name("ffn_norm")).is_some() {
                name("ffn_norm")
            } else {
                name("post_attention_norm")
            };
            let bias = if config.has_attn_bias {
                Some(AttnBias {
                    q: bias_vec_raw(f, &format!("blk.{i}.attn_q.bias"), q_dim)?,
                    k: bias_vec_raw(f, &format!("blk.{i}.attn_k.bias"), kv_dim)?,
                    v: bias_vec_raw(f, &format!("blk.{i}.attn_v.bias"), kv_dim)?,
                })
            } else {
                None
            };
            // qwen35moe: layers (i+1) % full_attention_interval != 0 are gated
            // delta net; the rest are full attention.
            let is_gdn = config.ssm.is_some()
                && config.full_attention_interval > 0
                && (i + 1) % config.full_attention_interval != 0;
            let (q_norm, k_norm) = load_qk_norms(f, &name, &config, !is_gdn)?;
            layers.push(Layer {
                attn_norm: norm_vec(f, &name("attn_norm"), hidden)?,
                attn_norm_raw: norm_raw(f, &name("attn_norm")),
                ffn_norm_raw: norm_raw(f, &ffn_norm),
                attn: if is_gdn {
                    None
                } else {
                    Some(load_attn_weights(f, &name, &config)?)
                },
                gdn: if is_gdn {
                    Some(load_gdn_layer(f, i, &name, &config)?)
                } else {
                    None
                },
                bias,
                q_norm,
                k_norm,
                ffn_norm: norm_vec(f, &ffn_norm, hidden)?,
                ffn: load_block_ffn(f, &name, &config, Some(i), &mut owned_weights)?,
            });
        }

        // A model without a separate output head ties it to the embedding.
        let output = if end != config.n_layers as usize {
            None
        } else if f.tensor("output.weight").is_some() {
            Some(qmat(f, "output.weight", config.vocab as usize, hidden)?)
        } else {
            Some(qmat(f, "token_embd.weight", config.vocab as usize, hidden)?)
        };

        // The MTP speculative block (blk.n_layers.*), when present: a
        // full-attention layer plus the nextn input/output glue.
        let mtp = if config.nextn && !stage {
            let i = config.n_layers;
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            let attn = load_attn_weights(f, &name, &config)?;
            let (q_norm, k_norm) = load_qk_norms(f, &name, &config, true)?;
            let ffn = load_block_ffn(f, &name, &config, None, &mut owned_weights)?;
            Some(MtpLayer {
                attn_norm: norm_vec(f, &name("attn_norm"), hidden)?,
                attn,
                q_norm,
                k_norm,
                ffn_norm: norm_vec(f, &name("post_attention_norm"), hidden)?,
                ffn,
                eh_proj: qmat(f, &name("nextn.eh_proj"), hidden, 2 * hidden)?,
                enorm: norm_vec(f, &name("nextn.enorm"), hidden)?,
                hnorm: norm_vec(f, &name("nextn.hnorm"), hidden)?,
                shared_head_norm: match f.tensor(&name("nextn.shared_head_norm")) {
                    Some(_) => norm_vec(f, &name("nextn.shared_head_norm"), hidden)?,
                    None => norm_vec(f, "output_norm.weight", hidden)?,
                },
            })
        } else {
            None
        };

        let model = Model {
            embd: if first == 0 {
                Some(qmat(f, "token_embd.weight", config.vocab as usize, hidden)?)
            } else {
                None
            },
            layer_start: first,
            output_norm: if end == config.n_layers as usize {
                norm_vec(f, "output_norm.weight", hidden)?
            } else {
                Vec::new()
            },
            output_norm_raw: if end == config.n_layers as usize {
                norm_raw(f, "output_norm.weight")
            } else {
                &[]
            },
            layers,
            mtp,
            output,
            rope_inv_freq: ops::rope_inv_freq(config.rope_dim as usize, config.rope_freq_base),
            config,
            _owned_weights: owned_weights,
        };
        // Device-side RoPE freqs: decode skips per-token sin/cos H2D.
        let _ = allpaka_backend::gpu::set_rope_inv_freq(&model.rope_inv_freq);
        Ok(model)
    }
}

/// Load full-attention projections (`wq`, `wk`, `wv`, `wo`, plus `gate_in_q`)
/// shared between trunk attention blocks and the MTP (`nextn`) speculative block.
fn load_attn_weights<'a>(
    f: &'a GgufFile,
    name: &dyn Fn(&str) -> String,
    config: &Config,
) -> Result<AttnWeights<'a>> {
    let hidden = config.hidden as usize;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    // qwen35moe fuses the output gate into the q projection
    // (second half of the rows per head).
    let gate_in_q = f
        .tensor(&name("attn_q"))
        .is_some_and(|t| t.dims.len() == 2 && t.dims[1] == 2 * q_dim as u64);
    Ok(AttnWeights {
        wq: qmat(
            f,
            &name("attn_q"),
            if gate_in_q { 2 * q_dim } else { q_dim },
            hidden,
        )?,
        wk: qmat(f, &name("attn_k"), kv_dim, hidden)?,
        wv: qmat(f, &name("attn_v"), kv_dim, hidden)?,
        wo: qmat(f, &name("attn_output"), hidden, q_dim)?,
        gate_in_q,
    })
}

type QkNorms = (Option<Vec<f32>>, Option<Vec<f32>>);

/// Load optional per-head Q/K RMSNorm vectors when the block uses full attention.
fn load_qk_norms(
    f: &GgufFile,
    name: &dyn Fn(&str) -> String,
    config: &Config,
    is_full_attention: bool,
) -> Result<QkNorms> {
    // Only full-attention layers carry q/k norms (qwen35moe's GDN layers have
    // none), hence the `is_full_attention` guard.
    if config.has_qk_norm && is_full_attention {
        let hd = config.head_dim as usize;
        Ok((
            Some(norm_vec(f, &name("attn_q_norm"), hd)?),
            Some(norm_vec(f, &name("attn_k_norm"), hd)?),
        ))
    } else {
        Ok((None, None))
    }
}

/// Load either a dense SwiGLU FFN or a routed MoE FFN for a transformer block.
/// When `trunk_layer_idx` is `Some(i)` and `i < moe.leading_dense` (GLM), the
/// block stays dense even on an MoE model; the MTP block passes `None`.
fn load_block_ffn<'a>(
    f: &'a GgufFile,
    name: &dyn Fn(&str) -> String,
    config: &Config,
    trunk_layer_idx: Option<u32>,
    owned_weights: &mut Vec<Box<[f32]>>,
) -> Result<Ffn<'a>> {
    let hidden = config.hidden as usize;
    let ffn = config.ffn_hidden as usize;
    match &config.moe {
        // GLM's leading blocks are plain dense FFNs before routing starts.
        Some(moe) if trunk_layer_idx.is_some_and(|i| i < moe.leading_dense) => Ok(Ffn::Dense {
            w_gate: qmat(f, &name("ffn_gate"), ffn, hidden)?,
            w_up: qmat(f, &name("ffn_up"), ffn, hidden)?,
            w_down: qmat(f, &name("ffn_down"), hidden, ffn)?,
        }),
        None => Ok(Ffn::Dense {
            w_gate: qmat(f, &name("ffn_gate"), ffn, hidden)?,
            w_up: qmat(f, &name("ffn_up"), ffn, hidden)?,
            w_down: qmat(f, &name("ffn_down"), hidden, ffn)?,
        }),
        Some(moe) => moe_ffn(f, name, moe, hidden, owned_weights),
    }
}

fn load_gdn_layer<'a>(
    f: &'a GgufFile,
    layer_idx: u32,
    name: &dyn Fn(&str) -> String,
    config: &Config,
) -> Result<GdnLayer<'a>> {
    let hidden = config.hidden as usize;
    let s = config.ssm.as_ref().expect("is_gdn implies ssm");
    let value_dim = (s.dt_rank * s.d_state) as usize;
    let channels = s.d_inner as usize + 2 * (s.n_group * s.d_state) as usize;
    let d_conv = s.d_conv as usize;
    let conv1d = norm_vec(f, &name("ssm_conv1d"), d_conv * channels)?;
    let mut conv1d_t = vec![0.0f32; d_conv * channels];
    for ch in 0..channels {
        for j in 0..d_conv {
            conv1d_t[j * channels + ch] = conv1d[ch * d_conv + j];
        }
    }
    Ok(GdnLayer {
        wqkv: qmat(f, &name("attn_qkv"), channels, hidden)?,
        zgate: qmat(f, &name("attn_gate"), value_dim, hidden)?,
        alpha: qmat(f, &name("ssm_alpha"), s.dt_rank as usize, hidden)?,
        beta: qmat(f, &name("ssm_beta"), s.dt_rank as usize, hidden)?,
        conv1d_t,
        conv1d_raw: norm_raw(f, &name("ssm_conv1d")),
        // ssm_a is the rare tensor with no `.weight` suffix.
        a: norm_vec(f, &format!("blk.{layer_idx}.ssm_a"), s.dt_rank as usize)?,
        dt: norm_vec(
            f,
            &format!("blk.{layer_idx}.ssm_dt.bias"),
            s.dt_rank as usize,
        )?,
        ssm_norm: norm_vec(f, &name("ssm_norm"), s.d_state as usize)?,
        ssm_out: qmat(f, &name("ssm_out"), hidden, value_dim)?,
    })
}

/// The stacked-expert weights of one MoE layer.
/// A router-width projection that may be BF16 (ggml id 30, the qwen35moe
/// MTP block's router and shared-expert gate): BF16 tensors are converted
/// to 4-byte-aligned F32 at load, attached to the GPU, and stored in
/// `Model::_owned_weights` so their lifetime is tied to `Model` without leaking.
fn qmat_f32ish<'a>(
    f: &'a GgufFile,
    name: &str,
    n_out: usize,
    n_in: usize,
    owned_weights: &mut Vec<Box<[f32]>>,
) -> Result<QuantMat<'a>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    if t.ggml_type != allpaka_gguf::GgmlType::BF16 {
        return qmat(f, name, n_out, n_in);
    }
    let f32s = f.dequant(t)?.into_boxed_slice();
    // SAFETY: `f32s` is a heap-allocated `Box<[f32]>` stored in `Model::_owned_weights`,
    // which is the last field of `Model<'a>` and is dropped only after all `QuantMat<'a>`
    // fields in `Model<'a>` have been dropped. Moving the outer `Vec<Box<[f32]>>` never
    // relocates the inner `[f32]` heap allocation.
    let bytes: &'a [u8] =
        unsafe { std::slice::from_raw_parts(f32s.as_ptr() as *const u8, f32s.len() * 4) };
    allpaka_backend::gpu::attach(bytes);
    owned_weights.push(f32s);
    QuantMat::new(bytes, allpaka_gguf::GgmlType::F32, n_out, n_in)
}

fn moe_ffn<'a>(
    f: &'a GgufFile,
    name: &dyn Fn(&str) -> String,
    moe: &MoeConfig,
    hidden: usize,
    owned_weights: &mut Vec<Box<[f32]>>,
) -> Result<Ffn<'a>> {
    let n_expert = moe.n_expert as usize;
    let expert_ffn = moe.expert_ffn as usize;
    // GLM's always-on shared expert: `n_shared` expert-width FFNs stacked
    // into one dense triple (`ffn_gate_shexp` and friends).
    let shared = if moe.n_shared > 0 {
        let shared_ffn = expert_ffn * moe.n_shared as usize;
        Some(SharedFfn {
            gate: qmat(f, &name("ffn_gate_shexp"), shared_ffn, hidden)?,
            up: qmat(f, &name("ffn_up_shexp"), shared_ffn, hidden)?,
            down: qmat(f, &name("ffn_down_shexp"), hidden, shared_ffn)?,
            // qwen35moe gates the shared expert's output per token; the gate
            // tensor is 1-D [hidden] (dot product to a scalar).
            gate_out: match f.tensor(&name("ffn_gate_inp_shexp")) {
                Some(t) if t.ggml_type == allpaka_gguf::GgmlType::BF16 => Some(qmat_f32ish(
                    f,
                    &name("ffn_gate_inp_shexp"),
                    1,
                    hidden,
                    owned_weights,
                )?),
                Some(t) => Some(QuantMat::new(f.data(t)?, t.ggml_type, 1, hidden)?),
                None => None,
            },
        })
    } else {
        None
    };
    // GLM's sigmoid router adds a learned bias to the logits.
    let router_bias_name = name("exp_probs_b").replace(".weight", ".bias");
    let router_bias = if f.tensor(&router_bias_name).is_some() {
        Some(bias_vec_raw(f, &router_bias_name, n_expert)?)
    } else {
        None
    };
    Ok(Ffn::Moe {
        router: qmat_f32ish(f, &name("ffn_gate_inp"), n_expert, hidden, owned_weights)?,
        router_bias,
        gate_exps: qmat3(f, &name("ffn_gate_exps"), expert_ffn, hidden, n_expert)?,
        up_exps: qmat3(f, &name("ffn_up_exps"), expert_ffn, hidden, n_expert)?,
        down_exps: qmat3(f, &name("ffn_down_exps"), hidden, expert_ffn, n_expert)?,
        shared,
        gating: moe.gating,
        weights_norm: moe.weights_norm,
        weights_scale: moe.weights_scale,
        expert_ffn,
        hidden,
        n_used: moe.n_used as usize,
    })
}

/// A projection bias vector: dequantised for the CPU path, raw F32 bytes for
/// GPU binds (empty raw when the tensor is not F32, which then declines the
/// whole-token GPU path).
fn bias_vec_raw<'a>(f: &'a GgufFile, tensor: &str, len: usize) -> Result<(Vec<f32>, &'a [u8])> {
    Ok((norm_vec(f, tensor, len)?, norm_raw(f, tensor)))
}

/// Add a per-row bias to a row-major buffer whose row length is `bias.len()`.
pub(crate) fn add_bias_rows(rows: &mut [f32], bias: &[f32]) {
    for row in rows.chunks_exact_mut(bias.len()) {
        ops::add_assign_slice(row, bias);
    }
}

/// A stacked-expert tensor `[n_in, n_out, n_expert]`, opened as one tall
/// matrix of `n_out * n_expert` rows; experts are addressed by row bands.
fn qmat3<'a>(
    f: &'a GgufFile,
    name: &str,
    n_out: usize,
    n_in: usize,
    n_expert: usize,
) -> Result<QuantMat<'a>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    if t.dims.len() != 3
        || t.dims[0] != n_in as u64
        || t.dims[1] != n_out as u64
        || t.dims[2] != n_expert as u64
    {
        return Err(ModelError::ShapeMismatch {
            tensor: name.into(),
            actual: t.dims.clone(),
            expected: format!("[{n_in}, {n_out}, {n_expert}]"),
        }
        .into());
    }
    QuantMat::new(f.data(t)?, t.ggml_type, n_out * n_expert, n_in)
}

/// A weight matrix by name, with its GGUF shape checked against what the
/// graph expects. A silently transposed tensor produces plausible garbage;
/// a named shape error produces a fix.
fn qmat<'a>(f: &'a GgufFile, name: &str, n_out: usize, n_in: usize) -> Result<QuantMat<'a>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    if t.dims.len() != 2 || t.dims[0] != n_in as u64 || t.dims[1] != n_out as u64 {
        return Err(ModelError::ShapeMismatch {
            tensor: name.into(),
            actual: t.dims.clone(),
            expected: format!("[{n_in}, {n_out}]"),
        }
        .into());
    }
    QuantMat::new(f.data(t)?, t.ggml_type, n_out, n_in)
}

/// A norm weight vector, dequantised once at load.
/// The raw bytes of an F32 norm tensor inside the mmap, or empty. The GPU
/// binds these directly; anything but F32 falls back to per-layer paths.
fn norm_raw<'a>(f: &'a GgufFile, name: &str) -> &'a [u8] {
    f.tensor(name)
        .filter(|t| t.ggml_type == allpaka_gguf::GgmlType::F32)
        .and_then(|t| f.data(t).ok())
        .unwrap_or(&[])
}

fn norm_vec(f: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
    if t.elements() != len as u64 {
        return Err(ModelError::ElementCountMismatch {
            tensor: name.into(),
            actual: t.elements(),
            expected: len,
        }
        .into());
    }
    Ok(f.dequant(t)?)
}
