use std::time::Instant;

#[test]
fn speculation_microbench_100_rounds() {
    const ITERATIONS: usize = 100;
    const LOOPS: usize = 200_000;

    let mut expected_ns = Vec::with_capacity(ITERATIONS);
    let mut argmax_ns = Vec::with_capacity(ITERATIONS);

    for round in 0..ITERATIONS {
        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for i in 0..LOOPS {
            let a = 0.72 + ((i % 17) as f64 * 0.013);
            let s = allpaka_core::Speculation {
                draft_weight_bytes: 1 << 20,
                draft_tokens: (i % 32) as u32,
                acceptance_rate: a.clamp(0.0, 1.0),
            };
            acc += s.expected_accepted();
        }
        let expected_ms = t0.elapsed();
        expected_ns.push(expected_ms.as_nanos() as u128);

        let logits: Vec<f32> = (0..512)
            .map(|i| ((i % 23) as f32) * 0.125 + (round as f32) * 0.01)
            .collect();
        let t1 = Instant::now();
        for _ in 0..LOOPS {
            let _ = allpaka_core::Speculation {
                draft_weight_bytes: 1,
                draft_tokens: 8,
                acceptance_rate: 0.8,
            }
            .verify_batch();
            let _ = argmax_manual(&logits);
        }
        let argmax_ms = t1.elapsed();
        argmax_ns.push(argmax_ms.as_nanos() as u128);

        if round == 0 {
            assert!(acc > 0.0, "microbench should produce valid numeric output");
        }
    }

    let expected_avg_ns = expected_ns.iter().sum::<u128>() as f64 / expected_ns.len() as f64;
    let argmax_avg_ns = argmax_ns.iter().sum::<u128>() as f64 / argmax_ns.len() as f64;

    println!(
        "speculation.expected_accepted avg: {:.2} ns/iter, 100 rounds",
        expected_avg_ns / LOOPS as f64
    );
    println!(
        "speculation.argmax avg: {:.2} ns/iter, 100 rounds",
        argmax_avg_ns / LOOPS as f64
    );
}

#[inline(always)]
fn argmax_manual(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_value = logits[0];
    for (idx, &value) in logits.iter().enumerate().skip(1) {
        if value > best_value {
            best_value = value;
            best_idx = idx;
        }
    }
    best_idx as u32
}
