//! Grouped MoE feed-forward for prefill: all routed experts of a chunk with
//! optional shared expert and on-device combine.

use super::*;

pub fn ffn_batch_grouped(req: &GroupedFfnReq) -> Option<Vec<f32>> {
    // Dense fused prefill: FFN stays on-device from pf_hs.
    if req.fused.is_some() && req.route.is_none() && req.shared.is_none() && req.n_expert <= 1 {
        if let Some(out) = try_dense_ffn_pf(req) {
            return Some(out);
        }
    }

    // GPU routing: logits live in y_arena from the fused attn block; rebuild
    // groups/tok/CSR on the host from pf_hs + those logits, then fall through.
    let owned = if let Some(route) = &req.route {
        let (hs, logits) = with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            let m = req.fused.as_ref()?.m;
            let n_expert = req.n_expert;
            let hs = gpu
                .stream
                .clone_dtoh(&gpu.pf_hs.slice(0..m * req.hidden))
                .ok()?;
            // Router was run at the end of prefill_attn_block into y_arena.
            let logits = gpu
                .stream
                .clone_dtoh(&gpu.y_arena.slice(0..m * n_expert))
                .ok()?;
            Some((hs, logits))
        })?;
        let m = req.fused.as_ref()?.m;
        let mut groups: Vec<Vec<(usize, f32)>> = vec![Vec::new(); req.n_expert];
        for i in 0..m {
            let row = &logits[i * req.n_expert..(i + 1) * req.n_expert];
            let mut scored: Vec<(f32, usize)> = row
                .iter()
                .enumerate()
                .map(|(e, &v)| {
                    let mut s = if route.sigmoid {
                        1.0 / (1.0 + (-v).exp())
                    } else {
                        v
                    };
                    if let Some(b) = route.bias {
                        s += b.get(e).copied().unwrap_or(0.0);
                    }
                    (s, e)
                })
                .collect();
            if !route.sigmoid {
                let mx = scored.iter().map(|s| s.0).fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scored {
                    s.0 = ((s.0 - mx) * route.scale).exp();
                    sum += s.0;
                }
                for s in &mut scored {
                    s.0 /= sum.max(1e-30);
                }
            } else if route.scale != 1.0 {
                for s in &mut scored {
                    s.0 *= route.scale;
                }
            }
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut picks: Vec<(usize, f32)> = scored
                .into_iter()
                .take(route.n_used)
                .map(|(w, e)| (e, w))
                .collect();
            if route.norm {
                let wsum: f32 = picks.iter().map(|p| p.1).sum();
                if wsum > 0.0 {
                    for p in &mut picks {
                        p.1 /= wsum;
                    }
                }
            }
            for (e, w) in picks {
                groups[e].push((i, w));
            }
        }
        let used: Vec<usize> = (0..req.n_expert)
            .filter(|&e| !groups[e].is_empty())
            .collect();
        let mut table = Vec::with_capacity(used.len());
        let mut tok = Vec::new();
        let mut row0 = 0u32;
        for &e in &used {
            let rows = groups[e].len() as u32;
            table.push([e as u32, row0, rows]);
            for &(i, _) in &groups[e] {
                tok.push(i as u32);
            }
            row0 += rows;
        }
        let total_rows = row0 as usize;
        let mut hits: Vec<Vec<(u32, f32)>> = vec![Vec::new(); m];
        for (gi, &e) in used.iter().enumerate() {
            let r0 = table[gi][1];
            for (ri, &(i, weight)) in groups[e].iter().enumerate() {
                hits[i].push((r0 + ri as u32, weight));
            }
        }
        if req.shared.is_some() {
            for (i, h) in hits.iter_mut().enumerate() {
                h.push(((total_rows + i) as u32, 1.0));
            }
        }
        let mut tok_off = Vec::with_capacity(m + 1);
        let mut hit_row = Vec::new();
        let mut hit_w = Vec::new();
        tok_off.push(0u32);
        for h in &hits {
            for &(r, w) in h {
                hit_row.push(r);
                hit_w.push(w);
            }
            tok_off.push(hit_row.len() as u32);
        }
        Some((hs, table, tok, total_rows, tok_off, hit_row, hit_w, m))
    } else {
        None
    };

    let (groups, tok, total_rows, fused_comb, act_src) =
        if let Some((hs, ref table, ref tok, total_rows, ref tok_off, ref hit_row, ref hit_w, m)) =
            owned
        {
            (
                table.as_slice(),
                tok.as_slice(),
                total_rows,
                Some(GroupedCombine {
                    tok_off,
                    hit_row,
                    hit_w,
                    m,
                    next_norm: None,
                }),
                Some(hs),
            )
        } else {
            if req.groups.is_empty() || req.tok.len() != req.total_rows {
                return None;
            }
            (req.groups, req.tok, req.total_rows, None, None)
        };
    let fused = fused_comb.as_ref().or(req.fused.as_ref());

    if req.shared.is_some() && fused.is_none() {
        return None;
    }

    // Fused prefill leaves normed activations in pf_hs; `x` is often empty.
    // Cache `pf_hs` on the host once on dUMA so the shared-expert pass below
    // does not perform a second PCIe D2H transfer of the same activation block.
    let mut act_src = act_src;
    if act_src.is_none() && fused.is_some() && req.x.is_empty() {
        let m = fused.as_ref()?.m;
        act_src = Some(with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            gpu.stream
                .clone_dtoh(&gpu.pf_hs.slice(0..m * req.hidden))
                .ok()
        })?);
    }

    let mut gathered = vec![0f32; total_rows * req.hidden];
    if let Some(hs) = act_src.as_ref() {
        for (r, &t) in tok.iter().enumerate() {
            let s = (t as usize) * req.hidden;
            if s + req.hidden > hs.len() {
                return None;
            }
            gathered[r * req.hidden..(r + 1) * req.hidden].copy_from_slice(&hs[s..s + req.hidden]);
        }
    } else {
        let m_src = req.x.len() / req.hidden.max(1);
        for (r, &t) in tok.iter().enumerate() {
            if (t as usize) >= m_src {
                return None;
            }
            let s = (t as usize) * req.hidden;
            gathered[r * req.hidden..(r + 1) * req.hidden]
                .copy_from_slice(&req.x[s..s + req.hidden]);
        }
    }

    let gate_rb = req.gate.1.len() / req.n_expert.max(1);
    let up_rb = req.up.1.len() / req.n_expert.max(1);
    let down_rb = req.down.1.len() / req.n_expert.max(1);

    let mut gate_out = vec![0f32; total_rows * req.ffn];
    let mut up_out = vec![0f32; total_rows * req.ffn];
    for g in groups {
        let (eid, row0, rows) = (g[0] as usize, g[1] as usize, g[2] as usize);
        if eid >= req.n_expert.max(1) || row0 + rows > total_rows {
            return None;
        }
        let xg = &gathered[row0 * req.hidden..(row0 + rows) * req.hidden];
        let go = with_gpu(|gpu| {
            run_matvec(
                gpu,
                req.gate.0,
                &req.gate.1[eid * gate_rb..(eid + 1) * gate_rb],
                req.hidden,
                req.ffn,
                xg,
                rows,
            )
        })?;
        let uo = with_gpu(|gpu| {
            run_matvec(
                gpu,
                req.up.0,
                &req.up.1[eid * up_rb..(eid + 1) * up_rb],
                req.hidden,
                req.ffn,
                xg,
                rows,
            )
        })?;
        gate_out[row0 * req.ffn..(row0 + rows) * req.ffn].copy_from_slice(&go);
        up_out[row0 * req.ffn..(row0 + rows) * req.ffn].copy_from_slice(&uo);
    }
    crate::ops::swiglu(&mut gate_out, &up_out);

    let mut down_out = vec![0f32; total_rows * req.hidden];
    for g in groups {
        let (eid, row0, rows) = (g[0] as usize, g[1] as usize, g[2] as usize);
        let ag = &gate_out[row0 * req.ffn..(row0 + rows) * req.ffn];
        let d = with_gpu(|gpu| {
            run_matvec(
                gpu,
                req.down.0,
                &req.down.1[eid * down_rb..(eid + 1) * down_rb],
                req.ffn,
                req.hidden,
                ag,
                rows,
            )
        })?;
        down_out[row0 * req.hidden..(row0 + rows) * req.hidden].copy_from_slice(&d);
    }

    if let (Some(sh), Some(comb)) = (&req.shared, fused) {
        let hs_owned;
        let hs: &[f32] = if let Some(hs) = act_src.as_deref() {
            hs
        } else if !req.x.is_empty() && req.x.len() >= comb.m * req.hidden {
            &req.x[..comb.m * req.hidden]
        } else {
            hs_owned = with_gpu(|gpu| {
                if !gpu.pf_active {
                    return None;
                }
                gpu.stream
                    .clone_dtoh(&gpu.pf_hs.slice(0..comb.m * req.hidden))
                    .ok()
            })?;
            &hs_owned
        };
        let g =
            with_gpu(|gpu| run_matvec(gpu, sh.gate.0, sh.gate.1, req.hidden, sh.ffn, hs, comb.m))?;
        let u = with_gpu(|gpu| run_matvec(gpu, sh.up.0, sh.up.1, req.hidden, sh.ffn, hs, comb.m))?;
        let mut act = g;
        crate::ops::swiglu(&mut act, &u);
        let d = with_gpu(|gpu| {
            run_matvec(gpu, sh.down.0, sh.down.1, sh.ffn, req.hidden, &act, comb.m)
        })?;
        down_out.extend_from_slice(&d);
    }

    if let Some(comb) = fused {
        with_gpu(|gpu| {
            if !gpu.pf_active {
                return None;
            }
            gpu.ensure_arenas(down_out.len() * 4, 4)?;
            gpu.stream
                .memcpy_htod(&down_out, &mut gpu.y_arena.slice_mut(0..down_out.len()))
                .ok()?;
            let mut tok_off = unsafe { gpu.stream.alloc::<u32>(comb.tok_off.len()) }.ok()?;
            let mut hit_row = unsafe { gpu.stream.alloc::<u32>(comb.hit_row.len()) }.ok()?;
            let mut hit_w = unsafe { gpu.stream.alloc::<f32>(comb.hit_w.len()) }.ok()?;
            gpu.stream.memcpy_htod(comb.tok_off, &mut tok_off).ok()?;
            gpu.stream.memcpy_htod(comb.hit_row, &mut hit_row).ok()?;
            gpu.stream.memcpy_htod(comb.hit_w, &mut hit_w).ok()?;
            let f = gpu.func_owned("moe_combine_csr")?;
            let m_u = comb.m as u32;
            let hidden_u = req.hidden as u32;
            let cfg = LaunchConfig {
                grid_dim: (hidden_u.div_ceil(256), m_u, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let downs = gpu.y_arena.slice(0..down_out.len());
            let mut xs = gpu.pf_x.slice_mut(0..comb.m * req.hidden);
            unsafe {
                gpu.stream
                    .launch_builder_pdl(&f)
                    .arg(&mut xs)
                    .arg(&downs)
                    .arg(&tok_off)
                    .arg(&hit_row)
                    .arg(&hit_w)
                    .arg(&m_u)
                    .arg(&hidden_u)
                    .launch_pdl(cfg)
            }
            .ok()?;
            gpu.sync()?;
            Some(Vec::new())
        })
    } else {
        Some(down_out)
    }
}
