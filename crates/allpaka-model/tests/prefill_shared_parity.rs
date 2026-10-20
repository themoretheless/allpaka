//! Run explicitly with ALLPAKA_TEST_MODEL=/path/to/model.gguf.
#![cfg(target_os = "macos")]
use allpaka_backend::gpu;
use allpaka_gguf::GgufFile;
use allpaka_model::Model;

fn compare(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    let worst = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "max logit/hidden difference {worst}");
}

#[test]
#[ignore = "requires an actual GGUF model and Metal"]
fn shared_prefill_matches_copying_for_all_public_batch_outputs() {
    let path = std::env::var("ALLPAKA_TEST_MODEL").expect("ALLPAKA_TEST_MODEL is required");
    let file = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let model = Model::load(&file).unwrap();
    assert!(gpu::is_attached());
    let before = gpu::shared_activation_stats();
    for m in [16, 33, 512] {
        let tokens: Vec<u32> = (0..m)
            .map(|i| (i as u32 * 733 + 17) % model.config.vocab.min(30000))
            .collect();
        for mode in [0, 1, 2, 3] {
            let mut results = Vec::new();
            for enabled in ["0", "1"] {
                std::env::set_var("ALLPAKA_PREFILL_ZERO_COPY", enabled);
                let mut session = model.new_session(m + 1);
                let output = match mode {
                    0 => model.forward_batch(&tokens, &mut session).unwrap(),
                    1 => model.forward_batch_hidden(&tokens, &mut session).unwrap(),
                    2 => model.forward_batch_full(&tokens, &mut session).unwrap(),
                    _ => {
                        let (mut logits, hidden) =
                            model.forward_batch_full_hn(&tokens, &mut session).unwrap();
                        logits.extend(hidden);
                        logits
                    }
                };
                assert_eq!(session.pos(), m);
                results.push(output);
            }
            compare(&results[0], &results[1]);
        }
    }
    std::env::remove_var("ALLPAKA_PREFILL_ZERO_COPY");
    let after = gpu::shared_activation_stats();
    assert!(
        after.outputs > before.outputs,
        "fused prefill must exercise shared output"
    );
    assert!(
        after.projection_bytes > before.projection_bytes,
        "head must consume shared input"
    );
}
