//! Chat template detection, prompt formatting, and stop token sets.

use allpaka_model::Tokenizer;
use anyhow::{bail, Context, Result};

/// The chat template families this server can format. Decided by which
/// special tokens the vocabulary actually contains.
pub(crate) enum Template {
    /// `<|im_start|>role\n...<|im_end|>\n` - Qwen and much of the ecosystem.
    /// `force_think` mirrors the GGUF chat template of the thinking Qwen
    /// generations: the assistant turn is primed with a literal `<think>\n`
    /// (llama.cpp renders the same prefix from the jinja template).
    ChatMl {
        im_start: u32,
        im_end: u32,
        think: Option<u32>,
        force_think: bool,
    },
    /// `<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>` - Llama 3.
    Llama3 {
        bos: u32,
        start_header: u32,
        end_header: u32,
        eot: u32,
    },
    /// `[gMASK]<sop><|user|>\n...<|assistant|>\n` - GLM-4.x (ChatGLM).
    Glm {
        gmask: Option<u32>,
        sop: Option<u32>,
        system: u32,
        user: u32,
        assistant: u32,
    },
}

impl Template {
    pub(crate) fn detect(tok: &Tokenizer) -> Result<Self> {
        if let (Some(im_start), Some(im_end)) =
            (tok.piece_id("<|im_start|>"), tok.piece_id("<|im_end|>"))
        {
            return Ok(Template::ChatMl {
                im_start,
                im_end,
                think: tok.piece_id("<think>"),
                force_think: false,
            });
        }
        if let (Some(start_header), Some(end_header), Some(eot)) = (
            tok.piece_id("<|start_header_id|>"),
            tok.piece_id("<|end_header_id|>"),
            tok.piece_id("<|eot_id|>"),
        ) {
            return Ok(Template::Llama3 {
                bos: tok.bos.context("llama3 template needs a BOS token")?,
                start_header,
                end_header,
                eot,
            });
        }
        if let (Some(system), Some(user), Some(assistant)) = (
            tok.piece_id("<|system|>"),
            tok.piece_id("<|user|>"),
            tok.piece_id("<|assistant|>"),
        ) {
            return Ok(Template::Glm {
                gmask: tok.piece_id("[gMASK]"),
                sop: tok.piece_id("<sop>"),
                system,
                user,
                assistant,
            });
        }
        bail!("the vocabulary has neither ChatML nor Llama 3 nor GLM special tokens");
    }

    /// The literal prefix the assistant turn was primed with (`<think>\n`
    /// for the thinking Qwens, empty otherwise). The reply text gets it back
    /// so clients see the complete `<think>...</think>` block.
    pub(crate) fn think_prefix(&self) -> &'static str {
        match self {
            Template::ChatMl {
                force_think: true, ..
            } => "<think>\n",
            _ => "",
        }
    }

    /// Format a conversation into token ids, ending where the assistant
    /// starts writing.
    pub(crate) fn prompt(
        &self,
        tok: &Tokenizer,
        messages: &[(String, String)],
    ) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        match self {
            Template::ChatMl {
                im_start,
                im_end,
                think,
                force_think,
            } => {
                let nl = tok.encode("\n")?;
                for (role, content) in messages {
                    ids.push(*im_start);
                    ids.extend(tok.encode(&format!("{role}\n{content}"))?);
                    ids.push(*im_end);
                    ids.extend_from_slice(&nl);
                }
                ids.push(*im_start);
                ids.extend(tok.encode("assistant\n")?);
                if *force_think {
                    // The special token by id, exactly as the jinja template
                    // renders it - plain BPE would split the literal text.
                    match think {
                        Some(t) => {
                            ids.push(*t);
                            ids.extend_from_slice(&nl);
                        }
                        None => ids.extend(tok.encode("<think>\n")?),
                    }
                }
            }
            Template::Llama3 {
                bos,
                start_header,
                end_header,
                eot,
            } => {
                ids.push(*bos);
                for (role, content) in messages {
                    ids.push(*start_header);
                    ids.extend(tok.encode(role)?);
                    ids.push(*end_header);
                    ids.extend(tok.encode(&format!("\n\n{content}"))?);
                    ids.push(*eot);
                }
                ids.push(*start_header);
                ids.extend(tok.encode("assistant")?);
                ids.push(*end_header);
                ids.extend(tok.encode("\n\n")?);
            }
            Template::Glm {
                gmask,
                sop,
                system,
                user,
                assistant,
            } => {
                if let Some(g) = gmask {
                    ids.push(*g);
                }
                if let Some(s) = sop {
                    ids.push(*s);
                }
                for (role, content) in messages {
                    let role_id = match role.as_str() {
                        "system" => *system,
                        "assistant" => *assistant,
                        _ => *user,
                    };
                    ids.push(role_id);
                    ids.extend(tok.encode(&format!("\n{content}"))?);
                }
                ids.push(*assistant);
                ids.extend(tok.encode("\n")?);
            }
        }
        Ok(ids)
    }

    /// Tokens that end an assistant turn.
    pub(crate) fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        let mut stops = Vec::new();
        match self {
            Template::ChatMl { im_end, .. } => stops.push(*im_end),
            Template::Llama3 { eot, .. } => stops.push(*eot),
            Template::Glm { user, .. } => stops.push(*user),
        }
        if let Some(eos) = tok.eos {
            stops.push(eos);
        }
        stops
    }
}
