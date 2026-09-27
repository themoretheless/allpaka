#!/usr/bin/env python3
"""Largest tensor per model: the lower bound on CHUNK_OVERLAP.

A window starts every `step = cap - overlap` bytes and runs `cap` long, so a
tensor at any file offset fits inside one window only if its size is at most
`overlap` - allpaka indexes a tensor through a single (chunk, offset) pair, so
one straddling a boundary reads across into the wrong bytes. This prints the
measured maximum, per model, which is the number any smaller overlap has to be
checked against. Reads GGUF metadata only: no GPU, no data section, no timer.
"""
import importlib.util
import pathlib
import sys

spec = importlib.util.spec_from_file_location(
    "ggufbpt", pathlib.Path("scripts/gguf-bytes-per-token.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

GIB = 1 << 30
for arg in sys.argv[1:]:
    meta, sizes = m.load_model(arg)
    top = sorted(sizes.items(), key=lambda kv: -kv[1][0])[:6]
    total = sum(s for s, _, _ in sizes.values())
    print(f"{pathlib.Path(arg).name}: {len(sizes)} tensors, {total / GIB:.2f} GiB")
    for name, (nbytes, dims, tid) in top:
        label = m.TENSOR_TYPE.get(tid, (str(tid), 0))[0]
        print(f"    {nbytes / GIB:7.3f} GiB  {label:<8} {name}  {list(dims)}")
    print(f"    => overlap must be >= {top[0][1][0] / GIB:.3f} GiB; "
          f"charge at overlap O is size + O*(windows-1)")
