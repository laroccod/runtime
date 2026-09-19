#!/usr/bin/env python3
"""Part 1.3 reconciliation helpers for DeepSeek-V4-Flash-0731 against ``gitm.planner``.

Prints the two things RECONCILIATION.md quotes and that must not be typed by hand:

1. The key diff between the pinned HF ``config.json`` (copied verbatim next to this
   file; sha256 in RECONCILIATION.md) and the keys ``spec_from_hf_config`` /
   ``is_sparse_moe_config`` actually read, in both directions, with the default each
   read key would silently take if it were absent.
2. The planner's per-region floors for the catalogue entry at the baseline
   (8xH200, TP8, B=32, S=8192, speculation off), aggregated by op, so the before and
   after of each correction on this branch can be quoted from the same code path.

Inputs: ``config.json`` here, the planner itself. No remembered values.
Run from the repo root with the project virtualenv: ``python docs/deepseek-v4-flash/reconcile.py``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

from gitm.planner.context import hardware_spec_for, peak_for_sku
from gitm.planner.model_catalogue import load_spec
from gitm.planner.moe_graph import is_sparse_moe_config, predict_moe_graph, spec_from_hf_config
from gitm.planner.registry import detect_family
from gitm.planner.roofline import BatchConfig, ShardingConfig

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
ENTRY = "deepseek-v4-flash-0731"
GPU, TP, B, S = "H200", 8, 32, 8192


class _Recorder(dict):
    """A dict that remembers which keys were asked for, however they were asked."""

    def __init__(self, d: dict[str, Any]):
        super().__init__(d)
        self.seen: set[str] = set()

    def get(self, k, default=None):  # noqa: D102
        self.seen.add(k)
        return super().get(k, default)

    def __getitem__(self, k):
        self.seen.add(k)
        return super().__getitem__(k)

    def __contains__(self, k):
        self.seen.add(k)
        return super().__contains__(k)


def key_diff() -> None:
    det = _Recorder(CFG)
    fam = detect_family(det)
    assert fam == "sparse_moe" and is_sparse_moe_config(det), fam
    rec = _Recorder(CFG)
    spec = spec_from_hf_config(rec, name=ENTRY)
    read = set(rec.seen)
    present = set(CFG)
    defaults = spec_from_hf_config({}, name=ENTRY)

    print("## 1. Key diff: pinned config.json vs the keys the sparse_moe class reads")
    print(f"config keys: {len(present)}   keys read by spec_from_hf_config: {len(read)}")
    print(f"family detection ({fam}) additionally probes: {sorted(det.seen - read)}")
    print("  (the other families' discriminators; absent here, as they must be)")
    print("\nread AND present (value -> spec field; default the class would take if absent):")
    field_of = {  # config key -> spec field, as spec_from_hf_config maps them
        "hidden_size": "hidden", "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads", "num_key_value_heads": "num_kv_heads",
        "head_dim": "head_dim", "qk_rope_head_dim": "qk_rope_head_dim",
        "q_lora_rank": "q_lora_rank", "o_lora_rank": "o_lora_rank", "o_groups": "o_groups",
        "vocab_size": "vocab", "n_routed_experts": "n_routed_experts",
        "n_shared_experts": "n_shared_experts", "num_experts_per_tok": "num_experts_per_tok",
        "moe_intermediate_size": "moe_intermediate_size", "index_n_heads": "index_n_heads",
        "index_head_dim": "index_head_dim", "index_topk": "index_topk",
        "sliding_window": "sliding_window", "compress_ratios": "compress_ratios",
        "num_nextn_predict_layers": "num_nextn_predict_layers", "hc_mult": "hc_width",
        "hc_sinkhorn_iters": "hc_sinkhorn_iters", "num_hash_layers": "num_hash_layers",
        "dspark_target_layer_ids": "dspark_layer_ids", "dspark_markov_rank": "dspark_markov_rank",
        "expert_dtype": "expert_dtype", "torch_dtype": "act_dtype",
        "quantization_config": "weight_dtype", "model_type": "name",
    }
    for k in sorted(read & present):
        f = field_of.get(k)
        val = CFG[k]
        if k == "compress_ratios":
            val = f"[46 entries] -> first {spec.n_layers} kept"
        if k == "quantization_config":
            val = f"quant_method={val.get('quant_method')!r} (only that sub-key is read)"
        d = getattr(defaults, f) if f else "-"
        if f == "compress_ratios":
            d = "()"
        print(f"  {k:28s} {str(val)[:44]:44s} -> {f or '-':24s} default {d}")
    absent = sorted(read - present)
    print(f"\nread but ABSENT from this config (silent default applies): {absent or 'none'}")
    ignored = sorted(present - read)
    print(f"\npresent but NOT read by the class ({len(ignored)}):")
    for k in ignored:
        v = CFG[k]
        print(f"  {k:28s} {json.dumps(v)[:60]}")
    sub = sorted(set(CFG["quantization_config"]) - {"quant_method"})
    print(f"\nquantization_config sub-keys not read: {sub}")
    print("  (weight_block_size [128,128] describes the fp8 path only; routed experts are")
    print("   fp4 with one e8m0 scale per 32 elements, model.py fp4_block_size = 32.)")


def floors() -> None:
    spec = load_spec(ENTRY)
    hw = hardware_spec_for(peak_for_sku(GPU))
    g = predict_moe_graph(spec, hw, BatchConfig(batch=B, kv_cache_len=S), ShardingConfig(tp=TP))
    ms: dict[str, float] = defaultdict(float)
    mb: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    est: dict[str, bool] = defaultdict(bool)
    for node in g.nodes:
        ms[node.op] += node.prediction.t_pred_s * 1e3
        mb[node.op] += node.prediction.bytes / 1e6
        n[node.op] += 1
        est[node.op] |= node.prediction.estimated
    print(f"\n## 2. Planner floors per rank, {ENTRY} on {GPU}, TP{TP}, B={B}, S={S}, speculation off")
    print(f"{'op':24s} {'xN':>4s} {'ms':>8s} {'MB':>9s}  flag")
    for op in sorted(ms, key=lambda o: -ms[o]):
        print(f"{op:24s} {n[op]:4d} {ms[op]:8.4f} {mb[op]:9.1f}  {'estimated' if est[op] else ''}")
    print(f"{'TOTAL':24s} {len(g.nodes):4d} {g.total_pred_s * 1e3:8.4f} {sum(mb.values()):9.1f}")
    layers = {x.layer for x in g.nodes if x.layer is not None}
    print(f"layers emitted: {min(layers)}..{max(layers)} ({len(layers)}); spec fields: "
          + ", ".join(f"{f.name}={getattr(spec, f.name)}" for f in fields(spec)
                      if f.name in ("n_layers", "num_nextn_predict_layers", "o_groups",
                                    "head_dim", "qk_rope_head_dim", "expert_dtype",
                                    "weight_dtype", "kv_dtype")))
    print(f"q_head_dim={spec.q_head_dim} kv_latent_dim={spec.kv_latent_dim}")


if __name__ == "__main__":
    key_diff()
    floors()
