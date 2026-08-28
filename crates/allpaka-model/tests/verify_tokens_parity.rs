//! The one-buffer GPU verify (`verify_tokens`, encode_verify_tokens) against
//! the CPU-reference batch path (`forward_batch_full_hn`): two fresh sessions
//! through the same prefill and the same 4-token batch must agree on row 0's
//! argmax (later rows run the batch path's half-precision tile matmuls, so
//! their argmax is a printed signal only). Hidden rows are printed as a
//! regression signal. The decode-exact contract is the second test.

#![cfg(target_os = "macos")]

use allpaka_gguf::GgufFile;
use allpaka_model::Model;

const MODEL: &str = "../../models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf";

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[test]
fn gpu_verify_matches_the_batch_path() {
    if !std::path::Path::new(MODEL).is_file() {
        eprintln!("SKIP: {MODEL} not present (CI runners have no models)");
        return;
    }
    let f = GgufFile::open(std::path::Path::new(MODEL)).expect("Qwen3.6 model present");
    let model = Model::load(&f).unwrap();
    let prompt = [33963u32, 728, 264, 2716, 1103, 314, 2250, 5839, 7736, 13];
    let batch = [271u32, 198, 20052, 5044];

    // Reference: fresh session, CPU batch path.
    let mut s1 = model.new_session_mtp(64);
    let _ = model.forward_batch_hidden(&prompt, &mut s1).unwrap();
    let hidden = model.config.hidden as usize;
    let (logits_ref, hidden_ref) = model.forward_batch_full_hn(&batch, &mut s1).unwrap();
    let vocab = model.config.vocab as usize;

    // GPU verify: fresh session, same prefill, same batch.
    let mut s2 = model.new_session_mtp(64);
    model.forward_batch_hidden(&prompt, &mut s2).unwrap();
    let (argmax_gpu, hidden_gpu) = model
        .verify_tokens(&batch, &mut s2)
        .unwrap()
        .expect("verify_tokens engages on this machine");

    assert_eq!(argmax_gpu.len(), batch.len());
    for (i, &a) in argmax_gpu.iter().enumerate() {
        let want = argmax(&logits_ref[i * vocab..(i + 1) * vocab]);
        // Only row 0 is asserted: the batch reference's tile-matmul kernels
        // stage weights/activations as half past row 0, so later rows'
        // argmax legitimately differs from the f32 decode kernels' on
        // near-ties. Row-exact decode parity is the second test below.
        if i == 0 {
            assert_eq!(a, want, "argmax diverged at row {i}");
        } else if a != want {
            eprintln!("row {i}: argmax {a} vs batch-path {want} (half-mm reference)");
        }
        let r_ref = &hidden_ref[i * hidden..(i + 1) * hidden];
        let r_gpu = &hidden_gpu[i * hidden..(i + 1) * hidden];
        let mut diffs: Vec<(usize, f32)> = r_ref
            .iter()
            .zip(r_gpu)
            .enumerate()
            .map(|(j, (a, b))| (j, (a - b).abs()))
            .collect();
        diffs.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!(
            "row {i}: top diffs {:?}",
            diffs
                .iter()
                .take(8)
                .map(|(j, d)| format!("{j}:{d:.4}(ref {:.3} gpu {:.3})", r_ref[*j], r_gpu[*j]))
                .collect::<Vec<_>>()
        );
        let big = diffs.iter().filter(|(_, d)| *d > 0.05).count();
        let worst = diffs[0].1;
        eprintln!("row {i}: channels >0.05: {big}/2048, max diff {worst:.4}");
        // The reference is the CPU/batch path: beyond row 0 its tile-matmul
        // kernels stage weights and activations as HALF, so on hidden values
        // of magnitude ~30 the gap reaches ~5 - the hidden numbers above are
        // a printed regression signal, not an assertion. The argmax equality
        // is the hard check here; decode-exactness is the second test below.
    }
}

/// Teacher-forced row check at a CLEAN state (no rollback involved): decode
/// a plain greedy stream with `forward` (the one-buffer GPU decode path),
/// then replay the same tokens through `verify_tokens` from a fresh session.
/// Every verify row's argmax must equal the decode token at that position -
/// for row 0 in particular, since that is where the bench's k=4 run flips a
/// 0.27-gap argmax. Run with both m=5 and m=3 to catch any m-dependence in
/// the batched kernels.
#[test]
fn verify_rows_match_greedy_decode() {
    if !std::path::Path::new(MODEL).is_file() {
        eprintln!("SKIP: {MODEL} not present (CI runners have no models)");
        return;
    }
    let f = GgufFile::open(std::path::Path::new(MODEL)).expect("Qwen3.6 model present");
    let model = Model::load(&f).unwrap();
    let prompt = [33963u32, 728, 264, 2716, 1103, 314, 2250, 5839, 7736, 13];
    let hidden = model.config.hidden as usize;

    // Plain greedy stream, 8 tokens, GPU decode path.
    let mut sa = model.new_session_mtp(64);
    let xs = model.forward_batch_hidden(&prompt, &mut sa).unwrap();
    let mut logits = model
        .lm_head(&model.output_normed(&xs[(prompt.len() - 1) * hidden..]))
        .unwrap();
    let mut stream = Vec::new();
    for _ in 0..8 {
        let t = argmax(&logits);
        stream.push(t);
        logits = model.forward(t, &mut sa).unwrap();
    }
    eprintln!("stream: {stream:?}");

    for m in [5usize, 3, 2] {
        let mut sb = model.new_session_mtp(64);
        model.forward_batch_hidden(&prompt, &mut sb).unwrap();
        let batch = &stream[..m];
        let (rows, _hidden) = model
            .verify_tokens(batch, &mut sb)
            .unwrap()
            .expect("verify_tokens engages on this machine");
        eprintln!("m={m}: batch {batch:?} rows {rows:?} want {:?}", &stream[1..m]);
        for (r, &got) in rows.iter().enumerate() {
            let want = stream[r + 1];
            assert_eq!(got, want, "m={m} row {r}: verify {got} != decode {want}");
        }
    }
}
