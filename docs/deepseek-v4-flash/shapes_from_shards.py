#!/usr/bin/env python3
"""Regenerate ``shapes_summary.json`` from the checkpoint's own shard headers.

``model.safetensors.index.json`` maps tensor name -> shard file and gives
``metadata.total_size``. It carries **no dtypes and no shapes**. Those live at
the front of each shard, in the safetensors header::

    [8 bytes: header length N, little-endian u64][N bytes: JSON header][tensor data]

One HTTP Range request for the first ``8 + N`` bytes (about 170 KiB of a
3.5 GB shard) is enough to read every dtype, shape and byte count in that
shard. This script fetches the index, the config and all 48 headers of
``deepseek-ai/DeepSeek-V4-Flash-0731`` at the pinned revision, checks that the
headers account for every byte the index declares, and writes the per-layer
shape table ``ledger.py`` consumes.

Usage (from the repo root)::

    python docs/deepseek-v4-flash/shapes_from_shards.py                 # full table -> shapes_summary.json
    python docs/deepseek-v4-flash/shapes_from_shards.py --shard 5 layers.3.attn   # inspect one shard

Fetched files are cached under ``~/.cache/gitm-shard-headers/<sha>/`` so a
second run does no network I/O. Nothing is downloaded but headers and the two
JSON files; no weights are read and no model code is executed.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import struct
import sys
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
SHA = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
N_SHARDS = 48
HERE = pathlib.Path(__file__).resolve().parent
CACHE = pathlib.Path.home() / ".cache" / "gitm-shard-headers" / SHA[:8]
OUT = HERE / "shapes_summary.json"


def _url(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{SHA}/{name}"


def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_json(name: str) -> dict:
    """The index or the config, fetched whole (they are small) and cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / name.replace("/", "_")
    if not cached.exists():
        cached.write_bytes(_get(_url(name)))
        print(f"   (fetched {name})", file=sys.stderr)
    return json.loads(cached.read_text())


def header(shard: int) -> dict:
    """JSON header of shard ``shard`` (1-based): {tensor: {dtype, shape, data_offsets}}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{shard:05d}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    name = f"model-{shard:05d}-of-{N_SHARDS:05d}.safetensors"
    n = struct.unpack("<Q", _get(_url(name), (0, 7)))[0]
    raw = _get(_url(name), (8, 8 + n - 1))
    if len(raw) != n:
        raise RuntimeError(f"short header read: got {len(raw)} of {n} bytes")
    h = json.loads(raw)
    h.pop("__metadata__", None)
    cached.write_text(json.dumps(h))
    print(f"   (fetched {n + 8:,} header bytes for {name})", file=sys.stderr)
    return h


def nbytes(t: dict) -> int:
    a, b = t["data_offsets"]
    return b - a


def canon(name: str) -> str:
    """Strip the layer/mtp index and the expert index so shapes can be compared."""
    name = re.sub(r"^(layers|mtp)\.\d+\.", "", name)
    return re.sub(r"experts\.\d+\.", "experts.N.", name)


def inspect_shard(shard: int, filt: str) -> None:
    h = {k: v for k, v in header(shard).items() if filt in k}
    print(f"shard {shard}: {len(h):,} tensors matching {filt!r}")
    print(f"  dtypes: {dict(collections.Counter(v['dtype'] for v in h.values()))}")
    print(f"  bytes : {sum(nbytes(v) for v in h.values()):,}")
    for k in sorted(h)[:20]:
        v = h[k]
        print(f"    {k:50s} {v['dtype']:8s} {str(v['shape']):18s} {nbytes(v):>12,} B")
    if len(h) > 20:
        print(f"    ... and {len(h) - 20:,} more")


def build_summary() -> dict:
    idx = fetch_json("model.safetensors.index.json")
    cfg = fetch_json("config.json")
    total_size = idx["metadata"]["total_size"]

    # ── 1. every tensor, from every shard header ────────────────────────────
    tensors: dict[str, dict] = {}
    for s in range(1, N_SHARDS + 1):
        for k, v in header(s).items():
            assert k not in tensors, k
            tensors[k] = {"dtype": v["dtype"], "shape": v["shape"], "bytes": nbytes(v), "shard": s}
    assert set(tensors) == set(idx["weight_map"]), "header tensor set != index weight_map"
    hdr_total = sum(t["bytes"] for t in tensors.values())
    print(f"tensors in headers: {len(tensors):,}   bytes: {hdr_total:,}")
    print(f"index total_size  : {total_size:,}   exact match: {hdr_total == total_size}")
    assert hdr_total == total_size, "headers do not account for the checkpoint"

    # ── 2. per-layer kind from config.compress_ratios (model.py:459, 472-477) ──
    ratios = cfg["compress_ratios"]
    n_layers = cfg["num_hidden_layers"]
    n_hash = cfg["num_hash_layers"]  # model.py:561 — layer_id < n_hash_layers routes by token id

    def kind(r: int) -> str:
        return {0: "swa", 4: "csa"}.get(r, "hca")

    layer_kind = {i: kind(ratios[i]) + ("+hash" if i < n_hash else "") for i in range(n_layers)}

    # per layer: {canon_name: {dtype, shape, bytes, count}}
    per_layer: dict[tuple[str, int], dict[str, dict]] = collections.defaultdict(dict)
    for k, t in tensors.items():
        m = re.match(r"^(layers|mtp)\.(\d+)\.", k)
        if not m:
            continue
        key = (m.group(1), int(m.group(2)))
        c = canon(k)
        e = per_layer[key].setdefault(
            c, {"dtype": t["dtype"], "shape": t["shape"], "bytes": 0, "count": 0}
        )
        assert e["dtype"] == t["dtype"] and e["shape"] == t["shape"], (k, e, t)
        e["bytes"] += t["bytes"]
        e["count"] += 1

    # ── 3. uniformity: every layer of a kind has the identical shape table ──
    by_kind: dict[str, list[int]] = collections.defaultdict(list)
    for i in range(n_layers):
        by_kind[layer_kind[i]].append(i)
    kind_table = {}
    for kd, layers in by_kind.items():
        ref = per_layer[("layers", layers[0])]
        for i in layers[1:]:
            assert per_layer[("layers", i)] == ref, f"layer {i} differs from layer {layers[0]}"
        kind_table[kd] = ref
        print(
            f"kind {kd:9s}: {len(layers):2d} layers, all identical; "
            f"{sum(e['bytes'] for e in ref.values()):,} B per layer"
        )

    # ── 4. globals, mtp, class totals over the executed stack ───────────────
    glob = {k: t for k, t in tensors.items() if not re.match(r"^(layers|mtp)\.", k)}

    def cls(c: str) -> str:
        if c.startswith("ffn.experts.N."):
            return "routed_expert_scale" if c.endswith(".scale") else "routed_expert_weight"
        if c.startswith("ffn.shared_experts."):
            return "shared_expert_scale" if c.endswith(".scale") else "shared_expert_weight"
        if c.startswith("ffn.gate."):
            return "router"
        if c.startswith("attn.indexer."):
            return "indexer"
        if c.startswith("attn.compressor."):
            return "compressor"
        if c.startswith("attn.") and c.endswith(".scale"):
            return "attn_linear_scale"
        if c.startswith("attn.w"):
            return "attn_linear_weight"
        if c.startswith("hc_"):
            return "hyper_connection"
        return "norm_sink_other"

    totals: collections.Counter = collections.Counter()
    for i in range(n_layers):
        for c, e in per_layer[("layers", i)].items():
            totals[cls(c)] += e["bytes"]
    stack = sum(totals.values())
    mtp_bytes = sum(t["bytes"] for k, t in tensors.items() if k.startswith("mtp."))
    print(f"executed stack (layers.0..{n_layers - 1}): {stack:,} B; "
          f"globals: {sum(t['bytes'] for t in glob.values()):,} B; mtp.*: {mtp_bytes:,} B")

    return {
        "total_size": total_size,
        "header_total": hdr_total,
        "layer_kind": layer_kind,
        "kind_table": kind_table,
        "globals": {
            k: {"dtype": t["dtype"], "shape": t["shape"], "bytes": t["bytes"]}
            for k, t in glob.items()
        },
        "mtp": {f"mtp.{m[1]}": per_layer[m] for m in per_layer if m[0] == "mtp"},
        "stack_class_totals": dict(totals),
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--shard":
        inspect_shard(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
        return
    summary = build_summary()
    OUT.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
