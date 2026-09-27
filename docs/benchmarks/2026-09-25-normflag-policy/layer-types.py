#!/usr/bin/env python3
"""Which layers can each decode fold reach? Read the GGUF header, no GPU.

Every dispatch delta measured on qwen3-30b-a3b-Q4_K_M came out as 24 per token
(7680 over TG=320) or 48 per token (15360), never 1-per-layer-of-48. The
explanation is in the file's own tensor table: llama.cpp's Q4_K_M rule keeps
`.attn_v` and `.ffn_down_exps` at Q6_K in only some layers, and the encoder
gates two different folds on exactly those two tensors' kernels:

  * `refs.normflag` (metal.rs) needs q, k AND v to be `matvec_q4_k_mv`; the
    router folds (`RFUSE`, `RMT`, `RTOPK`) sit behind `!refs.normflag`.
  * `sw_capable` lists every `matvec_*` down kernel EXCEPT `matvec_q6_k_mv`,
    so the standalone swiglu dispatch drops out only where down is not Q6_K -
    and that is what decides how many layers the `DCOMB` fold saves a dispatch in.

So a reachability claim about these knobs is computable from the header. Run it
on a second model before quoting a per-layer dispatch count anywhere.

    python3 layer-types.py ../../../models/qwen3-30b-a3b-Q4_K_M.gguf

A split GGUF (`-00001-of-00002`) keeps only its OWN tensor infos in each part, so
every part is read and unioned; censusing one part against `block_count` was this
script's first bug and made GLM's reachable-layer count look like 27 instead of
22. Headers only - no tensor data is read, no GPU, no compile.
"""

import glob
import os
import re
import struct
import sys

# ggml.h GGML_TYPE_*, including the holes the enum keeps for the retired
# Q4_2/Q4_3 (4, 5) and Q8_1 (9): a map that fills those slots mislabels every
# K-quant above Q3_K, which is the whole point of this script.
TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "Q2_S", 19: "Q2_XS",
    20: "Q3_S", 21: "Q3_5_0", 22: "Q3_5_1", 23: "Q4_0_16_6", 24: "IQ4_XS",
    25: "I8", 26: "I16", 27: "I32", 28: "I64", 29: "F64", 30: "BF16",
}
# The two gates, transcribed so a change in metal.rs fails this script's output.
MV_Q4_K = "matvec_q4_k_mv"
DOWN_SW_CAPABLE = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q8_0"}


def read_gguf(path):
    with open(path, "rb") as handle:
        if handle.read(4) != b"GGUF":
            raise SystemExit("not a GGUF")
        handle.read(4)
        n_tensors, n_kv = struct.unpack("<QQ", handle.read(16))

        def string():
            return handle.read(struct.unpack("<Q", handle.read(8))[0]).decode()

        def value(t):
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                   6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}.get(t)
            if t == 8:
                return string()
            if t == 9:
                et = struct.unpack("<I", handle.read(4))[0]
                return [value(et) for _ in range(struct.unpack("<Q", handle.read(8))[0])]
            if fmt is None:
                raise SystemExit(f"unhandled GGUF value type {t}")
            return struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]

        kv = {}
        for _ in range(n_kv):
            key = string()
            kv[key] = value(struct.unpack("<I", handle.read(4))[0])
        tensors = []
        for _ in range(n_tensors):
            name = string()
            dims = [struct.unpack("<Q", handle.read(8))[0]
                    for _ in range(struct.unpack("<I", handle.read(4))[0])]
            dtype = struct.unpack("<I", handle.read(4))[0]
            handle.read(8)
            tensors.append((name, dims, TYPES.get(dtype, f"type{dtype}")))
    return kv, tensors


SPLIT_RE = re.compile(r"^(.*?)-(\d{5})-of-(\d{5})(\.gguf)$")


def shards(path):
    """Every part of a split GGUF. A shard's header lists only ITS OWN tensors, so
    reading part 1 alone censuses a prefix of the layers under the full
    block_count - which silently turns a per-layer claim into a per-shard one."""
    m = SPLIT_RE.match(os.path.basename(path))
    if not m:
        return [path]
    head, _part, total = m.group(1), m.group(2), m.group(3)
    directory = os.path.dirname(path)
    found = sorted(glob.glob(os.path.join(directory, f"{head}-*-of-{total}.gguf")))
    if len(found) != int(total):
        raise SystemExit(f"{path}: split model needs {total} parts, found {len(found)} "
                         f"in {directory}")
    return found


def read_model(path):
    """Union every part's tensor infos. Only the part that holds the model metadata
    has a `block_count`; the others carry just `split.*`, so the arch must be found
    by looking for that key rather than by taking part 1."""
    files = shards(path)
    kv, tensors = None, []
    for f in files:
        meta, part_tensors = read_gguf(f)
        if any(k.endswith(".block_count") for k in meta):
            if kv is not None:
                raise SystemExit(f"{f}: a second part also carries block_count - "
                                 "these parts are not one split model")
            kv = meta
        tensors += part_tensors
    if kv is None:
        raise SystemExit(f"{path}: no part of {files} carries the model metadata")
    total = kv.get("split.count")
    if total is not None and total != len(files):
        raise SystemExit(f"{path}: metadata says {total} parts, read {len(files)}")
    names = [t[0] for t in tensors]
    if len(names) != len(set(names)):
        raise SystemExit(f"{path}: duplicate tensor names across parts - the union "
                         "of these files is not one model")
    return kv, tensors, files


def layers(tensors):
    out = {}
    for name, _, dtype in tensors:
        if not name.startswith("blk."):
            continue
        _, index, suffix = name.split(".", 2)
        out.setdefault(int(index), {})[suffix.removesuffix(".weight")] = dtype
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../../../models/qwen3-30b-a3b-Q4_K_M.gguf"
    kv, tensors, files = read_model(path)
    arch = next(k for k in kv if k.endswith(".block_count"))[: -len(".block_count")]
    n_layers = kv[f"{arch}.block_count"]
    per = layers(tensors)
    missing = sorted(set(range(n_layers)) - set(per))
    if missing:
        raise SystemExit(f"{path}: {len(missing)} of {n_layers} layers have no tensors in "
                         f"{len(files)} part(s) (first missing: {missing[0]}) - the "
                         "census would be a prefix, not the model")
    dense = kv.get(f"{arch}.leading_dense_block_count", 0)
    print(f"{path}")
    print(f"  arch={arch} layers={n_layers} tensors={len(tensors)} "
          f"parts={len(files)} dense_first={dense} "
          f"experts={kv.get(f'{arch}.expert_count')} "
          f"experts_used={kv.get(f'{arch}.expert_used_count')} "
          f"hidden={kv.get(f'{arch}.embedding_length')} "
          f"head_kv={kv.get(f'{arch}.attention.head_count_kv')}")
    for suffix in ("attn_q", "attn_k", "attn_v", "ffn_down_exps"):
        census = {}
        for index in sorted(per):
            dtype = per[index].get(suffix)
            if dtype:
                census.setdefault(dtype, []).append(index)
        line = "  ".join(
            f"{dtype}:{len(idx)}" for dtype, idx in sorted(census.items(), key=lambda x: -len(x[1]))
        )
        print(f"  {suffix:16s} {line}")
    totals = {}
    for _, _, dtype in tensors:
        totals[dtype] = totals.get(dtype, 0) + 1
    print(f"  all tensors      {totals}"
          "   <- compare against the engine's own `tensor-types:` census; a mismatch"
          " here means the type table above is mis-numbered, not that the model changed")

    # refs.normflag also requires the layer to be MoE, and the folds sit behind
    # it, so the denominator is the expert layers, not block_count.
    moe = [i for i in sorted(per) if "ffn_gate_exps" in per[i]]
    gdn = [i for i in moe if "attn_q" not in per[i] and "attn_qkv" in per[i]]
    odd = [i for i in moe if "attn_q" not in per[i] and "attn_qkv" not in per[i]]
    if gdn:
        print(f"  {len(gdn)} MoE layers are GDN (linear-attention: attn_qkv + attn_gate, "
              "no q/k/v). refs.normflag cannot be TRUE there, so the router folds are "
              "reachable in them by a different route than the quant split.")
    if odd:
        seen = sorted({s for i in odd for s in per[i] if s.startswith("attn")})
        print(f"  WARNING {len(odd)} MoE layers have neither attn_q nor attn_qkv "
              f"(their attn suffixes are {seen}) - counted as reachable here only "
              "because the q,k,v conjunction cannot hold for them")
    normflag_on = [i for i in moe
                   if all(per[i].get(s) == "Q4_K" for s in ("attn_q", "attn_k", "attn_v"))]
    folds_on = [i for i in moe if per[i].get("ffn_down_exps") in DOWN_SW_CAPABLE]
    n_moe = len(moe)
    print(f"\n  MoE layers {n_moe}/{n_layers} (expert tensors present)")
    print(f"  refs.normflag can be TRUE in {len(normflag_on)}/{n_moe} MoE layers "
          f"(q,k,v all {MV_Q4_K}); the RFUSE/RMT/RTOPK router fold is reachable in the "
          f"other {n_moe - len(normflag_on)}")
    print(f"    -> RFUSE/RMT/RTOPK under the shipped policy: "
          f"{n_moe - len(normflag_on)} dispatches/token, "
          f"{(n_moe - len(normflag_on)) * 320} over TG=320")
    print(f"    -> same knobs on every MoE layer (NORMFLAG=0): {n_moe} dispatches/token, "
          f"{n_moe * 320} over TG=320")
    shared = [i for i in moe if "ffn_down_shexp" in per[i]]
    print(f"  down is sw_capable (no standalone swiglu) in {len(folds_on)}/{n_moe} "
          f"MoE layers; DCOMB saves a dispatch only where swiglu is standalone: "
          f"{n_moe - len(folds_on)} layers, "
          f"{(n_moe - len(folds_on)) * 320} over TG=320")
    if shared:
        print(f"  {len(shared)}/{n_moe} MoE layers have a shared expert, and the "
              "down+combine fold's gate requires shared.is_none() - so on this model "
              "DCOMB is not built at all and the line above is about SWFUSE only")
    print("  caveat: this maps a down *dtype* to sw_capable, while the gate reads the "
          "chosen kernel name; q4_k/q8_0 select the _mv variant by row shape. The "
          "dtype set above lists both spellings as capable, so a wrong count here can "
          "only come from a format whose kernel name this map does not know.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
