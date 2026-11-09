//! Zero-allocation token sampler with on-the-fly repeat penalty and a 256-entry min-heap.

/// How many most recent tokens the repeat penalty looks at.
pub(crate) const PENALTY_WINDOW: usize = 256;

/// A deliberately small sampler: greedy at temperature 0, otherwise
/// repetition penalty, temperature scaling and nucleus (top-p) truncation.
pub(crate) struct Sampler {
    temperature: f64,
    top_p: f64,
    /// llama.cpp-style repeat penalty over the recent window: positive
    /// logits of recent tokens are divided by it, negative multiplied.
    /// Without it a heavily quantised model happily locks into "либо-либо-
    /// либо" forever; 1.1 breaks such loops without visibly changing prose.
    repeat_penalty: f64,
    state: u64,
}

impl Sampler {
    pub(crate) fn new(temperature: f64, top_p: f64, repeat_penalty: f64) -> Self {
        let seed = if temperature > 0.0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5eed)
                | 1
        } else {
            0x5eed
        };
        Self {
            temperature,
            top_p,
            repeat_penalty,
            state: seed,
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Candidates that survive the top-k preselect. Nucleus sampling never
    /// needs a total order of the vocabulary, and a full 152k-element sort
    /// costs ~3 ms per token, serial with the forward pass.
    pub(crate) const TOP_K: usize = 256;

    /// Sample the next token. `recent` is the context so far; its tail feeds
    /// the repeat penalty without cloning the vocabulary logits vector.
    pub(crate) fn pick(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        let rp = self.repeat_penalty as f32;
        let mut window_buf = [0u32; PENALTY_WINDOW];
        let mut pen_len = 0usize;
        if self.repeat_penalty != 1.0 && !recent.is_empty() {
            let start = recent.len().saturating_sub(PENALTY_WINDOW);
            let tail = &recent[start..];
            window_buf[..tail.len()].copy_from_slice(tail);
            let window = &mut window_buf[..tail.len()];
            window.sort_unstable();
            let mut prev = u32::MAX;
            for i in 0..window.len() {
                let t = window[i];
                if t != prev {
                    window[pen_len] = t;
                    pen_len += 1;
                    prev = t;
                }
            }
        }
        let window = &window_buf[..pen_len];
        let mut pen_idx = 0usize;
        let mut next_pen = if pen_len > 0 {
            window[0] as usize
        } else {
            usize::MAX
        };

        if self.temperature <= 0.0 {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &raw_v) in logits.iter().enumerate() {
                let v = if i == next_pen {
                    pen_idx += 1;
                    next_pen = if pen_idx < pen_len {
                        window[pen_idx] as usize
                    } else {
                        usize::MAX
                    };
                    if raw_v > 0.0 {
                        raw_v / rp
                    } else {
                        raw_v * rp
                    }
                } else {
                    raw_v
                };
                if i == 0 || !v.total_cmp(&best_val).is_lt() {
                    best_val = v;
                    best_idx = i;
                }
            }
            return best_idx as u32;
        }

        let k = Self::TOP_K.min(logits.len());
        let mut top_buf = [(0usize, 0.0f32); Self::TOP_K];
        let top = &mut top_buf[..k];
        for (i, &raw_v) in logits[..k].iter().enumerate() {
            let v = if i == next_pen {
                pen_idx += 1;
                next_pen = if pen_idx < pen_len {
                    window[pen_idx] as usize
                } else {
                    usize::MAX
                };
                if raw_v > 0.0 {
                    raw_v / rp
                } else {
                    raw_v * rp
                }
            } else {
                raw_v
            };
            top[i] = (i, v);
        }

        // Build a min-heap over `top[0..k]` in O(k) so `top[0]` holds the minimum.
        for root in (0..k / 2).rev() {
            sift_down(top, root);
        }

        for (i, &raw_v) in logits.iter().enumerate().skip(k) {
            let v = if i == next_pen {
                pen_idx += 1;
                next_pen = if pen_idx < pen_len {
                    window[pen_idx] as usize
                } else {
                    usize::MAX
                };
                if raw_v > 0.0 {
                    raw_v / rp
                } else {
                    raw_v * rp
                }
            } else {
                raw_v
            };
            if v.total_cmp(&top[0].1).is_gt() {
                top[0] = (i, v);
                sift_down(top, 0);
            }
        }
        top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let inv_temp = 1.0 / self.temperature;
        let max_scaled = top[0].1 as f64 * inv_temp;
        let mut scaled_buf = [0.0f64; Self::TOP_K];
        let scaled = &mut scaled_buf[..k];
        let mut sum = 0.0f64;
        for (dst, &(_, v)) in scaled.iter_mut().zip(top.iter()) {
            let e = ((v as f64) * inv_temp - max_scaled).exp();
            *dst = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum;
        for p in scaled.iter_mut() {
            *p *= inv_sum;
        }

        // Nucleus: keep the smallest prefix whose mass reaches top_p.
        let mut kept = 0usize;
        let mut mass = 0.0f64;
        for &p in scaled.iter() {
            kept += 1;
            mass += p;
            if mass >= self.top_p {
                break;
            }
        }
        let mut r = self.next_f64() * mass;
        for (&(i, _), &p) in top[..kept].iter().zip(scaled.iter()) {
            r -= p;
            if r <= 0.0 {
                return i as u32;
            }
        }
        top[kept - 1].0 as u32
    }
}

#[inline]
fn sift_down(heap: &mut [(usize, f32)], mut pos: usize) {
    let len = heap.len();
    let item = heap[pos];
    loop {
        let left = 2 * pos + 1;
        if left >= len {
            break;
        }
        let right = left + 1;
        let smallest = if right < len && heap[right].1.total_cmp(&heap[left].1).is_lt() {
            right
        } else {
            left
        };
        if !heap[smallest].1.total_cmp(&item.1).is_lt() {
            break;
        }
        heap[pos] = heap[smallest];
        pos = smallest;
    }
    heap[pos] = item;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_pick_is_argmax() {
        let mut s = Sampler::new(0.0, 0.95, 1.0);
        let mut logits = vec![0.0f32; 1000];
        logits[421] = 3.5;
        assert_eq!(s.pick(&logits, &[]), 421);
    }

    #[test]
    fn sampled_pick_follows_the_dominant_token() {
        let mut s = Sampler::new(0.7, 0.95, 1.0);
        let mut logits = vec![0.0f32; 152_000];
        logits[123] = 20.0;
        for _ in 0..50 {
            assert_eq!(s.pick(&logits, &[]), 123);
        }
    }

    #[test]
    fn sampled_pick_stays_within_the_top_candidates() {
        let mut s = Sampler::new(1.0, 0.9, 1.0);
        let mut logits = vec![-10.0f32; 152_000];
        for v in logits.iter_mut().take(512) {
            *v = 1.0;
        }
        for _ in 0..100 {
            let t = s.pick(&logits, &[]) as usize;
            assert!(t < 512, "picked {t} from outside the plateau");
        }
    }

    #[test]
    fn repeat_penalty_demotes_a_recently_emitted_token() {
        let mut s = Sampler::new(0.0, 0.95, 1.1);
        let mut logits = vec![0.0f32; 1000];
        logits[7] = 3.0;
        logits[8] = 2.9;
        assert_eq!(s.pick(&logits, &[]), 7);
        assert_eq!(s.pick(&logits, &[7]), 8);
        let mut neg = vec![-1.0f32; 10];
        neg[3] = -0.5;
        neg[4] = -0.52;
        assert_eq!(s.pick(&neg, &[3]), 4);
    }

    #[test]
    fn repeat_penalty_applies_once_per_distinct_token() {
        let mut s = Sampler::new(0.0, 0.95, 1.3);
        let mut logits = vec![0.0f32; 100];
        logits[5] = 2.0;
        logits[6] = 1.6;
        assert_eq!(s.pick(&logits, &[5, 5, 5, 5, 5]), 6);
    }
}
