//! Decode rate of the three ways a caller can drive greedy decode: one token
//! at a time reading logits (`forward`), one token at a time reading the
//! on-device argmax (`forward_greedy`), and a chain (`forward_greedy_n`).
//! `serve` uses the first two, `allpaka bench` the third; they should agree
//! to within launch overhead, and the argmax path must not be the slowest.
//!
//! Run explicitly: ALLPAKA_TEST_MODEL=/path/to/model.gguf cargo test --release
//! -p allpaka-model --features cuda --test cuda_decode_paths_rate -- --ignored --nocapture
#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use allpaka_gguf::GgufFile;
use allpaka_model::Model;
use std::time::Instant;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

#[test]
#[ignore = "requires an actual GGUF model and a CUDA GPU"]
fn the_three_greedy_paths_agree_and_run_at_comparable_rates() {
    let path = std::env::var("ALLPAKA_TEST_MODEL").expect("ALLPAKA_TEST_MODEL is required");
    let file = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let model = Model::load(&file).unwrap();
    let prompt: Vec<u32> = (0..200u32).map(|i| (i * 733 + 17) % 30000).collect();
    let n = 128;

    let mut rates = Vec::new();
    let mut streams = Vec::new();
    for (name, round) in [
        ("forward", 0),
        ("forward_greedy", 1),
        ("forward_greedy_n", 2),
    ]
    .iter()
    .cycle()
    .take(6)
    {
        let mut s = model.new_session(prompt.len() + n + 2);
        let first = argmax(&model.forward_batch(&prompt, &mut s).unwrap());
        let started = Instant::now();
        let out: Vec<u32> = match round {
            0 => {
                let mut t = first;
                (0..n)
                    .map(|_| {
                        t = argmax(&model.forward(t, &mut s).unwrap());
                        t
                    })
                    .collect()
            }
            1 => {
                let mut t = first;
                (0..n)
                    .map(|_| {
                        t = model.forward_greedy(t, &mut s).unwrap();
                        t
                    })
                    .collect()
            }
            _ => model.forward_greedy_n(first, &mut s, n).unwrap(),
        };
        let rate = n as f64 / started.elapsed().as_secs_f64();
        eprintln!("{name:>16}: {rate:7.1} tok/s");
        rates.push((*name, rate));
        streams.push(out);
    }
    for stream in &streams[1..] {
        assert_eq!(stream, &streams[0], "the paths decoded different streams");
    }
    // Second pass of each path, after everything is warm.
    let rate = |name: &str| rates.iter().rev().find(|(n, _)| *n == name).unwrap().1;
    assert!(
        rate("forward_greedy") >= rate("forward") * 0.95,
        "the argmax path is slower than reading the logits"
    );
}
