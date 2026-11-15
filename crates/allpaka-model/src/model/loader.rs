//! GGUF weight loading, stage slicing, and BF16-to-F32 router weight conversion.

use super::moe::RoutingPolicy;
use super::Mixer;
use super::{
    gdn::GdnLayer,
    moe::{Ffn, SharedFfn},
    mtp::MtpLayer,
    AttnBias, AttnWeights, Layer, Model,
};
use crate::config::{Config, MoeConfig};
use crate::error::{bail, Context, Result};
use allpaka_backend::{ops, quantmat::QuantMat};
use allpaka_gguf::GgufFile;

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
        self.embd.as_ref().context("this stage has no embedding")
    }

    pub(crate) fn output_head(&self) -> Result<&QuantMat<'a>> {
        self.output
            .as_ref()
            .context("this stage has no output head")
    }

    pub(crate) fn load_range(
        f: &'a GgufFile,
        first: usize,
        end: usize,
        stage: bool,
    ) -> Result<Self> {
        let mut config = Config::from_gguf(f)?;
        if first >= end || end > config.n_layers as usize {
            bail!("invalid stage layer range {first}..{end}");
        }
        let capability = config.capability();
        if capability.support == crate::config::ModelSupport::Unsupported {
            bail!(
                "architecture {} is unsupported: {}",
                config.architecture,
                capability.reason
            );
        }
        // Offer the weights to the GPU; on a machine without Metal this is a
        // no-op and every matmul stays on the CPU reference. Split files get
        // one GPU window set per part.
        let mut attached: Vec<&[u8]> = Vec::new();
        if !stage {
            for m in f.mappings() {
                if allpaka_backend::gpu::attach(m) {
                    attached.push(m);
                }
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
                if allpaka_backend::gpu::attach(mapping) {
                    attached.push(mapping);
                }
            }
        }
        let hidden = config.hidden as usize;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let ffn = config.ffn_hidden as usize;

        let mut owned_weights: Vec<Box<[f32]>> = Vec::new();
        // Llama and Mistral rotate adjacent pairs (NORM); every fast path and
        // GPU kernel rotates the two halves of a head (NEOX). Reordering the
        // rows of Wq and Wk turns one into the other without changing a single
        // q.k product, so those models are loaded in NEOX order and from here
        // on are NEOX models. `ALLPAKA_ROPE_NORM=1` keeps the file's order.
        let to_neox = neox_order(&config);
        if to_neox.is_some() {
            config.rope_style = crate::config::RopeStyle::Neox;
        }
        let to_neox = to_neox.as_deref();
        let mut layers = Vec::with_capacity(end - first);
        for i in first as u32..end as u32 {
            let name = |part: &str| format!("blk.{i}.{part}.weight");
            // GLM calls the pre-FFN norm "post_attention_norm" (Gemma has
            // both: there it is the post-attention sandwich norm).
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
            layers.push(Layer {
                attn_norm: norm_vec(f, &name("attn_norm"), hidden)?,
                attn_norm_raw: norm_raw(f, &name("attn_norm")),
                ffn_norm_raw: norm_raw(f, &ffn_norm),
                mixer: if is_gdn {
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
                    Mixer::Gdn(GdnLayer {
                        wqkv: qmat(f, &name("attn_qkv"), channels, hidden)?,
                        zgate: qmat(f, &name("attn_gate"), value_dim, hidden)?,
                        alpha: qmat(f, &name("ssm_alpha"), s.dt_rank as usize, hidden)?,
                        beta: qmat(f, &name("ssm_beta"), s.dt_rank as usize, hidden)?,
                        conv1d_t,
                        conv1d_raw: norm_raw(f, &name("ssm_conv1d")),
                        // ssm_a is the rare tensor with no `.weight` suffix.
                        a: norm_vec(f, &format!("blk.{i}.ssm_a"), s.dt_rank as usize)?,
                        dt: norm_vec(f, &format!("blk.{i}.ssm_dt.bias"), s.dt_rank as usize)?,
                        ssm_norm: norm_vec(f, &name("ssm_norm"), s.d_state as usize)?,
                        ssm_out: qmat(f, &name("ssm_out"), hidden, value_dim)?,
                    })
                } else {
                    Mixer::Attention(attn_weights(
                        f,
                        &name,
                        &config,
                        to_neox,
                        &mut owned_weights,
                    )?)
                },
                bias,
                // Only full-attention layers carry q/k norms (qwen35moe's
                // GDN layers have none), hence the !is_gdn guard.
                q_norm: qk_norm(f, &name("attn_q_norm"), &config, !is_gdn)?,
                k_norm: qk_norm(f, &name("attn_k_norm"), &config, !is_gdn)?,
                ffn_norm: norm_vec(f, &ffn_norm, hidden)?,
                ffn: match &config.moe {
                    // GLM's leading blocks are plain dense FFNs before the
                    // routing starts.
                    Some(moe) if i >= moe.leading_dense => {
                        moe_ffn(f, &name, moe, hidden, &mut owned_weights)?
                    }
                    _ => dense_ffn(f, &name, ffn, hidden)?,
                },
                post_attn_norm: match &config.gemma {
                    Some(_) => Some(norm_vec(f, &name("post_attention_norm"), hidden)?),
                    None => None,
                },
                post_ffn_norm: match &config.gemma {
                    Some(_) => Some(norm_vec(f, &name("post_ffw_norm"), hidden)?),
                    None => None,
                },
                post_attn_norm_raw: match &config.gemma {
                    Some(_) => norm_raw(f, &name("post_attention_norm")),
                    None => &[],
                },
                post_ffn_norm_raw: match &config.gemma {
                    Some(_) => norm_raw(f, &name("post_ffw_norm")),
                    None => &[],
                },
                window: config
                    .gemma
                    .as_ref()
                    .filter(|g| g.is_local(i as usize))
                    .map(|g| g.swa_window as usize),
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
            let ffn = match &config.moe {
                Some(moe) => moe_ffn(f, &name, moe, hidden, &mut owned_weights)?,
                None => dense_ffn(f, &name, ffn, hidden)?,
            };
            Some(MtpLayer {
                attn_norm: norm_vec(f, &name("attn_norm"), hidden)?,
                attn: attn_weights(f, &name, &config, to_neox, &mut owned_weights)?,
                q_norm: qk_norm(f, &name("attn_q_norm"), &config, true)?,
                k_norm: qk_norm(f, &name("attn_k_norm"), &config, true)?,
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

        let token_path_decline = super::decode::token_path_decline(&layers);
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
            rope_inv_freq: rope_inv_freq_with_factors(f, &config)?,
            rope_inv_freq_swa: match &config.gemma {
                Some(g) => ops::rope_inv_freq(config.rope_dim as usize, g.swa_rope_base),
                None => Vec::new(),
            },
            embed_scale: if config.gemma.is_some() {
                (config.hidden as f32).sqrt()
            } else {
                1.0
            },
            token_path_decline,
            config,
            attached,
            _owned_weights: owned_weights,
        };
        // Device-side RoPE freqs: decode skips per-token sin/cos H2D.
        let _ = allpaka_backend::gpu::set_rope_inv_freq(&model.rope_inv_freq);
        let _ = allpaka_backend::gpu::set_rope_inv_freq_swa(&model.rope_inv_freq_swa);
        // Gemma's activations overflow the f16 prefill GEMM.
        let _ = allpaka_backend::gpu::set_quant_gemm(model.config.gemma.is_some());
        Ok(model)
    }
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
    let t = f.tensor(name).with_context(|| format!("missing {name}"))?;
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
    Ok(QuantMat::new(
        bytes,
        allpaka_gguf::GgmlType::F32,
        n_out,
        n_in,
    )?)
}

/// q/k/v/o projections of one full-attention block. qwen35moe fuses the
/// output gate into the q projection (second half of the rows per head).
fn attn_weights<'a>(
    f: &'a GgufFile,
    name: &dyn Fn(&str) -> String,
    config: &Config,
    to_neox: Option<&[usize]>,
    owned_weights: &mut Vec<Box<[f32]>>,
) -> Result<AttnWeights<'a>> {
    let hidden = config.hidden as usize;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let gate_in_q = f
        .tensor(&name("attn_q"))
        .is_some_and(|t| t.dims.len() == 2 && t.dims[1] == 2 * q_dim as u64);
    let mut wq = qmat(
        f,
        &name("attn_q"),
        if gate_in_q { 2 * q_dim } else { q_dim },
        hidden,
    )?;
    let mut wk = qmat(f, &name("attn_k"), kv_dim, hidden)?;
    if let Some(order) = to_neox {
        if gate_in_q {
            bail!("NORM-style RoPE with a gate fused into attn_q is not supported");
        }
        wq = reorder_head_rows(&wq, order, owned_weights)?;
        wk = reorder_head_rows(&wk, order, owned_weights)?;
    }
    Ok(AttnWeights {
        wq,
        wk,
        wv: qmat(f, &name("attn_v"), kv_dim, hidden)?,
        wo: qmat(f, &name("attn_output"), hidden, q_dim)?,
        gate_in_q,
    })
}

/// For a NORM-style model that can be loaded in NEOX order: where each row of
/// a head comes from. Row `i` of the new head is old row `2i`, row
/// `i + rope_dim/2` is old row `2i + 1`, and rows past `rope_dim` stay put, so
/// the pair NORM rotates as `(2i, 2i+1)` is the pair NEOX rotates as
/// `(i, i + rope_dim/2)`, by the same angle.
///
/// `None` keeps the file's order: the model is NEOX already, the user asked for
/// it, or it has per-head q/k norms or projection biases, which would need the
/// same reordering and no NORM-style architecture has.
fn neox_order(config: &Config) -> Option<Vec<usize>> {
    if !matches!(config.rope_style, crate::config::RopeStyle::Norm)
        || config.has_qk_norm
        || config.has_attn_bias
        || std::env::var_os("ALLPAKA_ROPE_NORM").is_some_and(|v| v == "1")
    {
        return None;
    }
    let head_dim = config.head_dim as usize;
    let rope_dim = (config.rope_dim as usize).min(head_dim);
    let half = rope_dim / 2;
    Some(
        (0..head_dim)
            .map(|new| {
                if new < half {
                    2 * new
                } else if new < rope_dim {
                    2 * (new - half) + 1
                } else {
                    new
                }
            })
            .collect(),
    )
}

/// A copy of `m` with the rows of every head taken in `order` (new row -> old
/// row within the head). Rows are whole quantised blocks, so this moves bytes
/// and changes no value. The copy lives in `owned_weights` and is offered to
/// the GPU like any other weight.
fn reorder_head_rows<'a>(
    m: &QuantMat<'a>,
    order: &[usize],
    owned_weights: &mut Vec<Box<[f32]>>,
) -> Result<QuantMat<'a>> {
    let (ty, bytes) = m.raw();
    let head_dim = order.len();
    if head_dim == 0 || m.n_out % head_dim != 0 || bytes.len() % m.n_out != 0 {
        bail!("cannot reorder {} rows in heads of {head_dim}", m.n_out);
    }
    let row = bytes.len() / m.n_out;
    // f32 storage keeps the 4-byte alignment the mmap gives quantised blocks.
    let mut owned = vec![0f32; bytes.len().div_ceil(4)].into_boxed_slice();
    // SAFETY: `owned` holds at least `bytes.len()` initialised bytes; u8 has no
    // alignment or validity requirement, and nothing else borrows `owned` here.
    let out = unsafe { std::slice::from_raw_parts_mut(owned.as_mut_ptr() as *mut u8, bytes.len()) };
    for (head, rows) in out.chunks_exact_mut(head_dim * row).enumerate() {
        for (new, &old) in order.iter().enumerate() {
            let src = (head * head_dim + old) * row;
            rows[new * row..(new + 1) * row].copy_from_slice(&bytes[src..src + row]);
        }
    }
    // SAFETY: as in `qmat_f32ish`: the boxed slice goes into `Model::_owned_weights`,
    // which outlives every `QuantMat<'a>` of the model and whose heap allocation
    // does not move when the outer `Vec` does.
    let shared: &'a [u8] =
        unsafe { std::slice::from_raw_parts(owned.as_ptr() as *const u8, bytes.len()) };
    allpaka_backend::gpu::attach(shared);
    owned_weights.push(owned);
    Ok(QuantMat::new(shared, ty, m.n_out, m.n_in)?)
}

/// A per-head q or k norm, when the model has them and this block uses them.
fn qk_norm(f: &GgufFile, tensor: &str, config: &Config, used: bool) -> Result<Option<Vec<f32>>> {
    if config.has_qk_norm && used {
        Ok(Some(norm_vec(f, tensor, config.head_dim as usize)?))
    } else {
        Ok(None)
    }
}

/// The gate/up/down SwiGLU FFN of a dense block.
fn dense_ffn<'a>(
    f: &'a GgufFile,
    name: &dyn Fn(&str) -> String,
    ffn: usize,
    hidden: usize,
) -> Result<Ffn<'a>> {
    Ok(Ffn::Dense {
        w_gate: qmat(f, &name("ffn_gate"), ffn, hidden)?,
        w_up: qmat(f, &name("ffn_up"), ffn, hidden)?,
        w_down: qmat(f, &name("ffn_down"), hidden, ffn)?,
        gelu: f.architecture() == "gemma3",
    })
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
        routing: RoutingPolicy {
            gating: moe.gating,
            weights_norm: moe.weights_norm,
            weights_scale: moe.weights_scale,
            n_used: moe.n_used as usize,
        },
        expert_ffn,
        hidden,
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
        .with_context(|| format!("GGUF has no tensor {name:?}"))?;
    if t.dims.len() != 3
        || t.dims[0] != n_in as u64
        || t.dims[1] != n_out as u64
        || t.dims[2] != n_expert as u64
    {
        bail!(
            "tensor {name:?} has shape {:?}, expected [{n_in}, {n_out}, {n_expert}]",
            t.dims
        );
    }
    Ok(QuantMat::new(
        f.data(t)?,
        t.ggml_type,
        n_out * n_expert,
        n_in,
    )?)
}

/// A weight matrix by name, with its GGUF shape checked against what the
/// graph expects. A silently transposed tensor produces plausible garbage;
/// a named shape error produces a fix.
fn qmat<'a>(f: &'a GgufFile, name: &str, n_out: usize, n_in: usize) -> Result<QuantMat<'a>> {
    let t = f
        .tensor(name)
        .with_context(|| format!("GGUF has no tensor {name:?}"))?;
    if t.dims.len() != 2 || t.dims[0] != n_in as u64 || t.dims[1] != n_out as u64 {
        bail!(
            "tensor {name:?} has shape {:?}, expected [{n_in}, {n_out}]",
            t.dims
        );
    }
    Ok(QuantMat::new(f.data(t)?, t.ggml_type, n_out, n_in)?)
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

/// RoPE inverse frequencies, divided by the per-pair factors in
/// `rope_freqs.weight` when the file carries them (Llama 3.1+ "llama3" rope
/// scaling), as llama.cpp's `freq_factors` do: theta_i = pos * inv_freq_i / f_i.
/// Pair index `i` is the same in NORM and NEOX order, so this holds after the
/// loader's Wq/Wk permutation too.
fn rope_inv_freq_with_factors(f: &GgufFile, config: &Config) -> Result<Vec<f32>> {
    let mut inv = ops::rope_inv_freq(config.rope_dim as usize, config.rope_freq_base);
    if f.tensor("rope_freqs.weight").is_some()
        && std::env::var_os("ALLPAKA_NO_ROPE_FREQS").is_none()
    {
        let factors = norm_vec(f, "rope_freqs.weight", inv.len())?;
        for (v, factor) in inv.iter_mut().zip(factors) {
            *v /= factor;
        }
    }
    // Gemma 3's global layers: linear scaling divides every frequency.
    if let Some(g) = &config.gemma {
        for v in inv.iter_mut() {
            *v /= g.rope_scale;
        }
    }
    Ok(inv)
}

fn norm_vec(f: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = f
        .tensor(name)
        .with_context(|| format!("GGUF has no tensor {name:?}"))?;
    if t.elements() != len as u64 {
        bail!(
            "tensor {name:?} has {} elements, expected {len}",
            t.elements()
        );
    }
    Ok(f.dequant(t)?)
}
