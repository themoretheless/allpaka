//! The Llama-family forward pass, parameterised by GGUF metadata.
//!
//! Qwen3 (dense), Llama 3.x and Mistral all run through the same graph; the
//! genuine differences - RoPE pairing style and Qwen's QK norms - are config
//! switches decided once at load. Mixtures of experts and DeepSeek's MLA are
//! the two known extensions and both have their seams reserved: MoE hangs off
//! the FFN block, MLA lives entirely behind the [`kv`] interface.
//!
//! Correctness story, in order of strength: unit invariants here (causality,
//! cache-vs-recompute equality), then logit comparison against llama.cpp on a
//! real model, which is the acceptance test for the whole engine milestone.

pub mod config;
pub mod kv;
pub mod model;
pub mod pipeline;
pub mod profile;
pub mod prefix_cache;
pub mod requirements;
pub mod speculate;
pub mod tokenizer;

pub use config::{Config, ModelCapability, ModelSupport, RopeStyle};
pub use kv::KvCache;
pub use model::{set_trace_hook, Model, Session};
pub use tokenizer::Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;
    use allpaka_gguf::GgufFile;
    use std::io::Write;

    /// Deterministic pseudo-random weights: an LCG, so the test model is the
    /// same on every run and platform.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to roughly [-0.1, 0.1]: small weights keep activations tame
            // through two layers without normal init math.
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
        }
    }

    struct FileBuilder {
        kvs: Vec<u8>,
        kv_count: u64,
        tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    }

    impl FileBuilder {
        fn new() -> Self {
            Self {
                kvs: Vec::new(),
                kv_count: 0,
                tensors: Vec::new(),
            }
        }

        fn key(&mut self, k: &str) {
            self.kvs.extend_from_slice(&(k.len() as u64).to_le_bytes());
            self.kvs.extend_from_slice(k.as_bytes());
        }

        fn str_kv(mut self, k: &str, v: &str) -> Self {
            self.key(k);
            self.kvs.extend_from_slice(&8u32.to_le_bytes());
            self.kvs.extend_from_slice(&(v.len() as u64).to_le_bytes());
            self.kvs.extend_from_slice(v.as_bytes());
            self.kv_count += 1;
            self
        }

        fn u32_kv(mut self, k: &str, v: u32) -> Self {
            self.key(k);
            self.kvs.extend_from_slice(&4u32.to_le_bytes());
            self.kvs.extend_from_slice(&v.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn f32_kv(mut self, k: &str, v: f32) -> Self {
            self.key(k);
            self.kvs.extend_from_slice(&6u32.to_le_bytes());
            self.kvs.extend_from_slice(&v.to_le_bytes());
            self.kv_count += 1;
            self
        }

        /// dims in GGUF order: dims[0] is the contiguous (input) dimension.
        fn tensor(mut self, name: &str, dims: &[u64], data: Vec<f32>) -> Self {
            assert_eq!(dims.iter().product::<u64>(), data.len() as u64);
            let mut raw = Vec::with_capacity(data.len() * 4);
            for v in data {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            self.tensors.push((name.into(), dims.to_vec(), 0u32, raw));
            self
        }

        fn tensor_q8_0(mut self, name: &str, dims: &[u64], data: Vec<f32>) -> Self {
            assert_eq!(dims.iter().product::<u64>(), data.len() as u64);
            assert_eq!(data.len() % 32, 0);
            let mut raw = Vec::with_capacity(data.len() / 32 * 34);
            let scale_f16 = 0x211Fu16.to_le_bytes(); // ~0.01 in f16
            for chunk in data.chunks_exact(32) {
                raw.extend_from_slice(&scale_f16);
                for &v in chunk {
                    let q = (v * 100.0).round().clamp(-127.0, 127.0) as i8;
                    raw.push(q as u8);
                }
            }
            self.tensors.push((name.into(), dims.to_vec(), 8u32, raw));
            self
        }

        fn build(self) -> Vec<u8> {
            let mut info = Vec::new();
            let mut data = Vec::new();
            for (name, dims, ty, raw) in &self.tensors {
                // Each tensor starts 32-byte aligned inside the data section.
                while data.len() % 32 != 0 {
                    data.push(0u8);
                }
                info.extend_from_slice(&(name.len() as u64).to_le_bytes());
                info.extend_from_slice(name.as_bytes());
                info.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    info.extend_from_slice(&d.to_le_bytes());
                }
                info.extend_from_slice(&ty.to_le_bytes());
                info.extend_from_slice(&(data.len() as u64).to_le_bytes());
                data.extend_from_slice(raw);
            }

            let mut out = Vec::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&self.kv_count.to_le_bytes());
            out.extend_from_slice(&self.kvs);
            out.extend_from_slice(&info);
            while out.len() % 32 != 0 {
                out.push(0);
            }
            out.extend_from_slice(&data);
            out
        }
    }

    const HIDDEN: u64 = 8;
    const HEADS: u64 = 2;
    const KV_HEADS: u64 = 1;
    const HEAD_DIM: u64 = 4;
    const FFN: u64 = 16;
    const VOCAB: u64 = 11;
    const LAYERS: u64 = 2;

    /// A complete 2-layer llama-shaped model with deterministic weights.
    fn tiny_model(arch: &str, qk_norm: bool) -> Vec<u8> {
        let mut rng = Rng(42);
        let mut mat = |rows: u64, cols: u64| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.next_f32()).collect()
        };
        let q_dim = HEADS * HEAD_DIM;
        let kv_dim = KV_HEADS * HEAD_DIM;

        let mut b = FileBuilder::new()
            .str_kv("general.architecture", arch)
            .u32_kv(&format!("{arch}.block_count"), LAYERS as u32)
            .u32_kv(&format!("{arch}.embedding_length"), HIDDEN as u32)
            .u32_kv(&format!("{arch}.attention.head_count"), HEADS as u32)
            .u32_kv(&format!("{arch}.attention.head_count_kv"), KV_HEADS as u32)
            .u32_kv(&format!("{arch}.attention.key_length"), HEAD_DIM as u32)
            .u32_kv(&format!("{arch}.feed_forward_length"), FFN as u32)
            .f32_kv(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5)
            .f32_kv(&format!("{arch}.rope.freq_base"), 10000.0)
            .tensor("token_embd.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN))
            .tensor("output_norm.weight", &[HIDDEN], vec![1.0; HIDDEN as usize])
            .tensor("output.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN));

        for i in 0..LAYERS {
            let n = |p: &str| format!("blk.{i}.{p}.weight");
            b = b
                .tensor(&n("attn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
                .tensor(&n("attn_q"), &[HIDDEN, q_dim], mat(q_dim, HIDDEN))
                .tensor(&n("attn_k"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
                .tensor(&n("attn_v"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
                .tensor(&n("attn_output"), &[q_dim, HIDDEN], mat(HIDDEN, q_dim))
                .tensor(&n("ffn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
                .tensor(&n("ffn_gate"), &[HIDDEN, FFN], mat(FFN, HIDDEN))
                .tensor(&n("ffn_up"), &[HIDDEN, FFN], mat(FFN, HIDDEN))
                .tensor(&n("ffn_down"), &[FFN, HIDDEN], mat(HIDDEN, FFN));
            if qk_norm {
                b = b
                    .tensor(&n("attn_q_norm"), &[HEAD_DIM], vec![1.0; HEAD_DIM as usize])
                    .tensor(&n("attn_k_norm"), &[HEAD_DIM], vec![1.0; HEAD_DIM as usize]);
            }
        }
        b.build()
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("allpaka-model-{name}.gguf"));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    fn logits_for(f: &GgufFile, tokens: &[u32]) -> Vec<Vec<f32>> {
        let model = Model::load(f).unwrap();
        let mut s = model.new_session(16);
        tokens
            .iter()
            .map(|&t| model.forward(t, &mut s).unwrap())
            .collect()
    }

    #[test]
    fn selective_gpu_mappings_cover_chosen_tensor_without_whole_file() {
        let bytes = FileBuilder::new()
            .str_kv("general.architecture", "llama")
            .tensor("unused-before", &[40000], vec![0.0; 40000])
            .tensor("chosen", &[32], vec![0.5; 32])
            .tensor("unused-after", &[40000], vec![0.0; 40000])
            .build();
        let path = write_temp("selected-mapping", &bytes);
        let file = GgufFile::open(&path).unwrap();
        let mappings = file.selected_mappings(|tensor| tensor.name == "chosen").unwrap();
        assert_eq!(mappings.len(), 1);
        assert!(mappings[0].len() <= 32768);
        assert!(mappings[0].len() < file.mapping().len() / 2);
        let data = file.data(file.tensor("chosen").unwrap()).unwrap();
        let start = mappings[0].as_ptr() as usize;
        assert!(data.as_ptr() as usize >= start);
        assert!(data.as_ptr() as usize + data.len() <= start + mappings[0].len());
        assert_eq!((start - file.mapping().as_ptr() as usize) % 16384, 0);
        assert!(file.selected_mappings(|_| false).unwrap().is_empty());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn stage_loader_does_not_require_other_layers_or_endpoint_tensors() {
        let original = tiny_model("llama", false);
        // Rename tensors without changing GGUF offsets. The head must load
        // even when the other layer and output head are absent from metadata.
        let mut bytes = original.clone();
        for index in 0..bytes.len() - 5 {
            if &bytes[index..index + 5] == b"blk.1" && bytes.get(index + 6..index + 9) == Some(b"ffn") { bytes[index + 4] = b'9'; }
        }
        for index in 0..bytes.len() - 6 {
            if &bytes[index..index + 6] == b"output" && (index == 0 || bytes[index - 1] != b'_') { bytes[index] = b'X'; }
        }
        let path = write_temp("head-only", &bytes);
        let file = GgufFile::open(&path).unwrap();
        assert!(Model::load(&file).is_err());
        let head = Model::load_stage(&file, 0, 1).unwrap();
        let mut session = head.new_session(16);
        let h = head.embed_token(3).unwrap();
        assert!(head.forward_layers(&h, &mut session, 0, 1).is_ok());
        assert!(head.finish_hidden(&h).is_err());
        assert!(head.forward_layers(&h, &mut session, 1, 2).is_err());
        std::fs::remove_file(path).ok();

        let mut bytes = original;
        for index in 0..bytes.len() - 5 {
            if &bytes[index..index + 5] == b"blk.0" && bytes.get(index + 6..index + 9) == Some(b"ffn") { bytes[index + 4] = b'9'; }
        }
        let path = write_temp("tail-only", &bytes);
        let file = GgufFile::open(&path).unwrap();
        let tail = Model::load_stage(&file, 1, 2).unwrap();
        assert!(tail.embed_token(3).is_err());
        let mut session = tail.new_session(16);
        let h = tail.forward_layers(&vec![0.01; HIDDEN as usize], &mut session, 1, 2).unwrap();
        assert!(tail.finish_hidden(&h).is_ok());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn tcp_pipeline_matches_local_model_and_rejects_wrong_key() {
        use std::net::TcpListener;
        let path = write_temp("tcp-pipeline", &tiny_model("llama", false));
        let mut handles = Vec::new();
        let mut addresses = Vec::new();
        for first in 0..2 {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            addresses.push(listener.local_addr().unwrap().to_string());
            let path = path.clone();
            handles.push(std::thread::spawn(move || {
                let file = GgufFile::open(&path).unwrap();
                let model = Model::load_stage(&file, first, first + 1).unwrap();
                let info = pipeline::StageInfo {
                    model_id: "tiny-llama-v1".into(),
                    fingerprint: allpaka_gguf::fingerprint(&path).unwrap(), first, end: first + 1,
                    layers: 2, hidden: HIDDEN as usize, vocab: VOCAB as u32,
                };
                // The head first receives a deliberately unauthorised connection.
                if first == 0 {
                    let (stream, _) = listener.accept().unwrap();
                    assert!(pipeline::serve_connection(stream, &model, info.clone(), "test-key", 16).is_err());
                }
                let (stream, _) = listener.accept().unwrap();
                let _ = pipeline::serve_connection(stream, &model, info, "test-key", 16);
            }));
        }
        assert!(pipeline::Pipeline::connect(&addresses, "wrong-key", 16).is_err());
        let mut remote = pipeline::Pipeline::connect(&addresses, "test-key", 16).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let model = Model::load(&file).unwrap();
        let mut session = model.new_session(16);
        for token in [3, 1, 4, 2] {
            let expected = model.forward(token, &mut session).unwrap();
            let actual = remote.forward(token).unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-5, "{a} != {b}");
            }
        }
        drop(remote);
        for handle in handles { handle.join().unwrap(); }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn pipeline_stages_match_whole_model_across_tokens() {
        let path = write_temp("pipeline-stages", &tiny_model("llama", false));
        let file = GgufFile::open(&path).unwrap();
        let model = Model::load(&file).unwrap();
        let mut whole = model.new_session(16);
        let mut head = model.new_session(16);
        let mut tail = model.new_session(16);
        for token in [3, 1, 4, 2] {
            let expected = model.forward(token, &mut whole).unwrap();
            let h = model.embed_token(token).unwrap();
            let h = model.forward_layers(&h, &mut head, 0, 1).unwrap();
            let h = model.forward_layers(&h, &mut tail, 1, model.config.n_layers as usize).unwrap();
            let actual = model.finish_hidden(&h).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-5, "{a} != {b}");
            }
            assert_eq!(head.pos(), whole.pos());
            assert_eq!(tail.pos(), whole.pos());
        }
        let pos = head.pos();
        assert!(model.forward_layers(&[], &mut head, 0, 1).is_err());
        assert!(model.forward_layers(&vec![0.0; HIDDEN as usize], &mut head, 1, 1).is_err());
        assert_eq!(head.pos(), pos);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_config_reads_what_the_file_says() {
        let path = write_temp("config", &tiny_model("llama", false));
        let f = GgufFile::open(&path).unwrap();
        let c = Config::from_gguf(&f).unwrap();
        assert_eq!(c.n_layers, 2);
        assert_eq!(c.vocab, VOCAB as u32);
        assert_eq!(c.head_dim, HEAD_DIM as u32);
        assert_eq!(c.rope_style, RopeStyle::Norm);
        assert!(!c.has_qk_norm);
        assert!((c.rms_eps - 1e-5).abs() < 1e-12);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn qwen_shaped_files_flip_the_two_switches() {
        let path = write_temp("qwen", &tiny_model("qwen3", true));
        let f = GgufFile::open(&path).unwrap();
        let c = Config::from_gguf(&f).unwrap();
        assert_eq!(c.rope_style, RopeStyle::Neox);
        assert!(c.has_qk_norm);
        // And the forward pass runs with the extra norms in place.
        let out = logits_for(&f, &[1, 2, 3]);
        assert!(out[2].iter().all(|v| v.is_finite()));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn logits_are_finite_deterministic_and_vocab_sized() {
        let path = write_temp("determinism", &tiny_model("llama", false));
        let f = GgufFile::open(&path).unwrap();
        let a = logits_for(&f, &[3, 1, 4]);
        let b = logits_for(&f, &[3, 1, 4]);
        assert_eq!(a, b);
        assert_eq!(a[2].len(), VOCAB as usize);
        assert!(a[2].iter().all(|v| v.is_finite()));
        std::fs::remove_file(path).ok();
    }

    /// The current token must see earlier tokens: attention is the only
    /// mechanism connecting them, so this fails if the cache or the scores
    /// are wired wrong.
    #[test]
    fn changing_an_earlier_token_changes_the_current_logits() {
        let path = write_temp("attends", &tiny_model("llama", false));
        let f = GgufFile::open(&path).unwrap();
        let a = logits_for(&f, &[3, 1]);
        let b = logits_for(&f, &[5, 1]);
        assert_ne!(a[1], b[1], "the second token ignored the first");
        std::fs::remove_file(path).ok();
    }

    /// And only earlier ones: two sessions sharing a prefix must agree on the
    /// prefix's logits no matter what they feed afterwards.
    #[test]
    fn logits_depend_only_on_the_prefix() {
        let path = write_temp("causal", &tiny_model("llama", false));
        let f = GgufFile::open(&path).unwrap();
        let a = logits_for(&f, &[3, 1, 4]);
        let b = logits_for(&f, &[3, 1, 9]);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[1], b[1]);
        assert_ne!(a[2], b[2]);
        std::fs::remove_file(path).ok();
    }

    /// A model without output.weight ties the head to the embedding and still
    /// produces logits.
    #[test]
    fn a_tied_output_head_falls_back_to_the_embedding() {
        let bytes = tiny_model("llama", false);
        let path = write_temp("tied-src", &bytes);
        // Rebuild without the output tensor by name surgery: cheaper to just
        // build a fresh file lacking it.
        std::fs::remove_file(&path).ok();
        let mut rng = Rng(7);
        let mut mat = |rows: u64, cols: u64| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.next_f32()).collect()
        };
        let q_dim = HEADS * HEAD_DIM;
        let kv_dim = KV_HEADS * HEAD_DIM;
        let mut b = FileBuilder::new()
            .str_kv("general.architecture", "llama")
            .u32_kv("llama.block_count", 1)
            .u32_kv("llama.embedding_length", HIDDEN as u32)
            .u32_kv("llama.attention.head_count", HEADS as u32)
            .u32_kv("llama.attention.head_count_kv", KV_HEADS as u32)
            .u32_kv("llama.attention.key_length", HEAD_DIM as u32)
            .u32_kv("llama.feed_forward_length", FFN as u32)
            .tensor("token_embd.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN))
            .tensor("output_norm.weight", &[HIDDEN], vec![1.0; HIDDEN as usize]);
        let n = |p: &str| format!("blk.0.{p}.weight");
        b = b
            .tensor(&n("attn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
            .tensor(&n("attn_q"), &[HIDDEN, q_dim], mat(q_dim, HIDDEN))
            .tensor(&n("attn_k"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
            .tensor(&n("attn_v"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
            .tensor(&n("attn_output"), &[q_dim, HIDDEN], mat(HIDDEN, q_dim))
            .tensor(&n("ffn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
            .tensor(&n("ffn_gate"), &[HIDDEN, FFN], mat(FFN, HIDDEN))
            .tensor(&n("ffn_up"), &[HIDDEN, FFN], mat(FFN, HIDDEN))
            .tensor(&n("ffn_down"), &[FFN, HIDDEN], mat(HIDDEN, FFN));
        let path = write_temp("tied", &b.build());
        let f = GgufFile::open(&path).unwrap();
        let out = logits_for(&f, &[2]);
        assert_eq!(out[0].len(), VOCAB as usize);
        assert!(out[0].iter().all(|v| v.is_finite()));
        std::fs::remove_file(path).ok();
    }

    /// Rolling back and replaying must reproduce the original logits exactly:
    /// truncation is what chat prefix-reuse stands on.
    #[test]
    fn truncate_then_replay_matches_the_original_computation() {
        let path = write_temp("truncate", &tiny_model("llama", false));
        let f = GgufFile::open(&path).unwrap();
        let model = Model::load(&f).unwrap();

        let mut s = model.new_session(16);
        model.forward(3, &mut s).unwrap();
        let after_two = model.forward(1, &mut s).unwrap();
        let after_three = model.forward(4, &mut s).unwrap();

        // Roll back to just the first token and replay a different second
        // token, then the original one again.
        s.truncate(1);
        let replay_two = model.forward(1, &mut s).unwrap();
        assert_eq!(replay_two, after_two);
        let replay_three = model.forward(4, &mut s).unwrap();
        assert_eq!(replay_three, after_three);
        std::fs::remove_file(path).ok();
    }

    /// Batched prefill is an optimisation, not a different computation: the
    /// last-token logits must match feeding the same tokens one at a time,
    /// and the cache it leaves behind must continue identically.
    #[test]
    fn forward_batch_matches_sequential_forward() {
        for (arch, qk) in [("llama", false), ("qwen3", true)] {
            let path = write_temp(&format!("batch-{arch}"), &tiny_model(arch, qk));
            let f = GgufFile::open(&path).unwrap();
            let model = Model::load(&f).unwrap();

            let seq = logits_for(&f, &[3, 1, 4, 1, 5]);

            let mut s = model.new_session(16);
            let batched = model.forward_batch(&[3, 1, 4, 1, 5], &mut s).unwrap();
            let diff = batched
                .iter()
                .zip(&seq[4])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(diff < 1e-4, "{arch}: batched logits diverge by {diff}");

            // And decoding continues from the batched cache identically.
            let next_from_batch = model.forward(9, &mut s).unwrap();
            let mut s2 = model.new_session(16);
            for &t in &[3u32, 1, 4, 1, 5] {
                model.forward(t, &mut s2).unwrap();
            }
            let next_from_seq = model.forward(9, &mut s2).unwrap();
            let diff = next_from_batch
                .iter()
                .zip(&next_from_seq)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(diff < 1e-4, "{arch}: post-batch decode diverges by {diff}");
            std::fs::remove_file(path).ok();
        }
    }

    /// The MoE path routes per token inside a batch too.
    #[test]
    fn forward_batch_matches_sequential_on_a_moe() {
        let path = write_temp("batch-moe", &tiny_moe(None, 0));
        let f = GgufFile::open(&path).unwrap();
        let model = Model::load(&f).unwrap();

        let seq = logits_for(&f, &[3, 1, 4]);
        let mut s = model.new_session(16);
        let batched = model.forward_batch(&[3, 1, 4], &mut s).unwrap();
        let diff = batched
            .iter()
            .zip(&seq[2])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-4, "moe batched logits diverge by {diff}");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn an_out_of_vocab_token_is_an_error() {
        let path = write_temp("oov", &tiny_model("llama", false));
        let f = GgufFile::open(&path).unwrap();
        let model = Model::load(&f).unwrap();
        let mut s = model.new_session(4);
        assert!(model.forward(VOCAB as u32, &mut s).is_err());
        std::fs::remove_file(path).ok();
    }

    const EXPERTS: u64 = 4;
    const USED: u64 = 2;
    const EXPERT_FFN: u64 = 8;

    /// A 1-layer qwen3moe-shaped model. `poison` fills one expert's weights
    /// with huge values, to prove routing either avoids or hits it.
    fn tiny_moe(poisoned_expert: Option<usize>, router_bias_expert: usize) -> Vec<u8> {
        let mut rng = Rng(1234);
        let q_dim = HEADS * HEAD_DIM;
        let kv_dim = KV_HEADS * HEAD_DIM;
        let arch = "qwen3moe";

        // Stacked expert tensors: expert-major, each [n_out, n_in] row-major.
        let mut stack = |n_out: u64, n_in: u64, poison: Option<usize>| -> Vec<f32> {
            let mut v = Vec::with_capacity((EXPERTS * n_out * n_in) as usize);
            for e in 0..EXPERTS {
                for _ in 0..n_out * n_in {
                    let w = rng.next_f32();
                    v.push(if Some(e as usize) == poison { 1e6 } else { w });
                }
            }
            v
        };
        // A router that strongly prefers one expert regardless of input: its
        // row is large-positive on every channel... but sign depends on x, so
        // instead make every OTHER row zero. Zero rows tie; softmax then puts
        // the biased row on top for any x where its dot is positive, and the
        // test drives it with both signs via a symmetric second expert.
        let mut router = vec![0f32; (EXPERTS * HIDDEN) as usize];
        for c in 0..HIDDEN as usize {
            // Biased expert row +w, its mirror row -w: exactly one of the two
            // beats the zero rows for any nonzero x, and both beat ties only
            // through softmax order. Top-2 therefore always contains the pair.
            router[router_bias_expert * HIDDEN as usize + c] = 1.0;
            let mirror = (router_bias_expert + 1) % EXPERTS as usize;
            router[mirror * HIDDEN as usize + c] = -1.0;
        }

        // Materialise the expert stacks before `mat` also borrows the rng.
        let gate_stack = stack(EXPERT_FFN, HIDDEN, poisoned_expert);
        let up_stack = stack(EXPERT_FFN, HIDDEN, poisoned_expert);
        let down_stack = stack(HIDDEN, EXPERT_FFN, poisoned_expert);
        drop(stack);

        let mut mat = |rows: u64, cols: u64| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.next_f32()).collect()
        };
        let mut b = FileBuilder::new()
            .str_kv("general.architecture", arch)
            .u32_kv(&format!("{arch}.block_count"), 1)
            .u32_kv(&format!("{arch}.embedding_length"), HIDDEN as u32)
            .u32_kv(&format!("{arch}.attention.head_count"), HEADS as u32)
            .u32_kv(&format!("{arch}.attention.head_count_kv"), KV_HEADS as u32)
            .u32_kv(&format!("{arch}.attention.key_length"), HEAD_DIM as u32)
            .u32_kv(&format!("{arch}.feed_forward_length"), FFN as u32)
            .u32_kv(&format!("{arch}.expert_count"), EXPERTS as u32)
            .u32_kv(&format!("{arch}.expert_used_count"), USED as u32)
            .u32_kv(
                &format!("{arch}.expert_feed_forward_length"),
                EXPERT_FFN as u32,
            )
            .tensor("token_embd.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN))
            .tensor("output_norm.weight", &[HIDDEN], vec![1.0; HIDDEN as usize])
            .tensor("output.weight", &[HIDDEN, VOCAB], mat(VOCAB, HIDDEN));
        let n = |p: &str| format!("blk.0.{p}.weight");
        b = b
            .tensor(&n("attn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
            .tensor(&n("attn_q"), &[HIDDEN, q_dim], mat(q_dim, HIDDEN))
            .tensor(&n("attn_k"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
            .tensor(&n("attn_v"), &[HIDDEN, kv_dim], mat(kv_dim, HIDDEN))
            .tensor(&n("attn_output"), &[q_dim, HIDDEN], mat(HIDDEN, q_dim))
            .tensor(&n("ffn_norm"), &[HIDDEN], vec![1.0; HIDDEN as usize])
            .tensor(&n("ffn_gate_inp"), &[HIDDEN, EXPERTS], router)
            .tensor(
                &n("ffn_gate_exps"),
                &[HIDDEN, EXPERT_FFN, EXPERTS],
                gate_stack,
            )
            .tensor(&n("ffn_up_exps"), &[HIDDEN, EXPERT_FFN, EXPERTS], up_stack)
            .tensor(
                &n("ffn_down_exps"),
                &[EXPERT_FFN, HIDDEN, EXPERTS],
                down_stack,
            );
        b.build()
    }

    #[test]
    fn route_picks_the_top_k_and_renormalises() {
        let picked = model::route(&[0.1, 0.5, 0.15, 0.25], 2);
        assert_eq!(picked.len(), 2);
        assert_eq!(picked[0].0, 1);
        assert_eq!(picked[1].0, 3);
        // 0.5 and 0.25 renormalised to sum 1.
        assert!((picked[0].1 - 2.0 / 3.0).abs() < 1e-6);
        assert!((picked[1].1 - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn a_moe_model_produces_finite_deterministic_logits() {
        let path = write_temp("moe-runs", &tiny_moe(None, 0));
        let f = GgufFile::open(&path).unwrap();
        let c = Config::from_gguf(&f).unwrap();
        let moe = c.moe.as_ref().expect("config should see the experts");
        assert_eq!(moe.n_expert, EXPERTS as u32);
        assert_eq!(moe.n_used, USED as u32);

        let a = logits_for(&f, &[3, 1, 4]);
        let b = logits_for(&f, &[3, 1, 4]);
        assert_eq!(a, b);
        assert!(a[2].iter().all(|v| v.is_finite()));
        std::fs::remove_file(path).ok();
    }

    /// The point of routing: experts the router does not pick must not touch
    /// the output. Poisoning an unpicked expert with 1e6 weights changes
    /// nothing; poisoning a picked one blows the logits up visibly.
    #[test]
    fn unrouted_experts_do_not_influence_the_output() {
        let clean = write_temp("moe-clean", &tiny_moe(None, 0));
        // Experts 0 and 1 form the always-picked pair (router rows +-1);
        // experts 2 and 3 have zero router rows and lose every top-2.
        let poison_unused = write_temp("moe-poison-unused", &tiny_moe(Some(3), 0));
        let poison_used = write_temp("moe-poison-used", &tiny_moe(Some(0), 0));

        let a = logits_for(&GgufFile::open(&clean).unwrap(), &[3, 1]);
        let b = logits_for(&GgufFile::open(&poison_unused).unwrap(), &[3, 1]);
        let c = logits_for(&GgufFile::open(&poison_used).unwrap(), &[3, 1]);

        assert_eq!(a, b, "an unrouted expert leaked into the output");
        assert_ne!(a, c, "a routed expert was ignored");
        for path in [clean, poison_unused, poison_used] {
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn gpu_greedy_chain_matches_step_by_step_on_128_head_dim() {
        let mut rng = Rng(12345);
        let mut mat = |rows: u64, cols: u64| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.next_f32()).collect()
        };
        let hidden = 128u64;
        let heads = 2u64;
        let kv_heads = 1u64;
        let head_dim = 128u64;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let ffn = 128u64;
        let vocab = 32u64;
        let layers = 2u64;

        let mut b = FileBuilder::new()
            .str_kv("general.architecture", "qwen3")
            .u32_kv("qwen3.block_count", layers as u32)
            .u32_kv("qwen3.embedding_length", hidden as u32)
            .u32_kv("qwen3.attention.head_count", heads as u32)
            .u32_kv("qwen3.attention.head_count_kv", kv_heads as u32)
            .u32_kv("qwen3.attention.key_length", head_dim as u32)
            .u32_kv("qwen3.feed_forward_length", ffn as u32)
            .f32_kv("qwen3.attention.layer_norm_rms_epsilon", 1e-5)
            .f32_kv("qwen3.rope.freq_base", 10000.0)
            .tensor_q8_0("token_embd.weight", &[hidden, vocab], mat(vocab, hidden))
            .tensor("output_norm.weight", &[hidden], vec![1.0; hidden as usize])
            .tensor_q8_0("output.weight", &[hidden, vocab], mat(vocab, hidden));

        for i in 0..layers {
            let n = |p: &str| format!("blk.{i}.{p}.weight");
            b = b
                .tensor(&n("attn_norm"), &[hidden], vec![1.0; hidden as usize])
                .tensor_q8_0(&n("attn_q"), &[hidden, q_dim], mat(q_dim, hidden))
                .tensor_q8_0(&n("attn_k"), &[hidden, kv_dim], mat(kv_dim, hidden))
                .tensor_q8_0(&n("attn_v"), &[hidden, kv_dim], mat(kv_dim, hidden))
                .tensor_q8_0(&n("attn_output"), &[q_dim, hidden], mat(hidden, q_dim))
                .tensor(&n("ffn_norm"), &[hidden], vec![1.0; hidden as usize])
                .tensor_q8_0(&n("ffn_gate"), &[hidden, ffn], mat(ffn, hidden))
                .tensor_q8_0(&n("ffn_up"), &[hidden, ffn], mat(ffn, hidden))
                .tensor_q8_0(&n("ffn_down"), &[ffn, hidden], mat(hidden, ffn));
        }
        let mut raw = b.build();
        while raw.len() % 16384 != 0 {
            raw.push(0);
        }
        let path = write_temp("greedy-chain-128", &raw);
        let f = GgufFile::open(&path).unwrap();
        let model = Model::load(&f).unwrap();
        let prompt = [1u32, 5, 12];

        std::env::set_var("ALLPAKA_GREEDY_CHAIN", "0");
        let mut s1 = model.new_session(32);
        let logits = model.forward_batch(&prompt, &mut s1).unwrap();
        let seed = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        let ref_tokens = model.forward_greedy_n(seed, &mut s1, 4).unwrap();

        std::env::set_var("ALLPAKA_GREEDY_CHAIN", "1");
        let mut s2 = model.new_session(32);
        let _ = model.forward_batch(&prompt, &mut s2).unwrap();
        let gpu_before = allpaka_backend::gpu::stats();
        let before = allpaka_backend::gpu::decode_path_stats();
        let chain_tokens = model.forward_greedy_n(seed, &mut s2, 4).unwrap();
        let after = allpaka_backend::gpu::decode_path_stats();
        let gpu_after = allpaka_backend::gpu::stats();
        let waits = gpu_after.0 - gpu_before.0;

        assert_eq!(ref_tokens, chain_tokens, "greedy chain diverged from step-by-step");
        #[cfg(target_os = "macos")]
        {
            assert!(
                allpaka_backend::gpu::is_attached(),
                "Metal GPU should be attached and shader library compiled cleanly"
            );
            assert_eq!(after.attempts - before.attempts, 4);
            assert_eq!(after.successes - before.successes, 4);
            assert_eq!(after.declines - before.declines, 0);
            assert_eq!(waits, 2, "expected 1 wait for seed + 1 wait for 3-token continuation chain");
        }
        std::fs::remove_file(path).ok();
    }
}
