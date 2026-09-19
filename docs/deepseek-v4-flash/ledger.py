#!/usr/bin/env python3
"""Per-GPU ledger and decode-step floors for DeepSeek-V4-Flash-0731 @ 7872f01b.

Every number in ``DESIGN-NOTE.md`` is printed by this script. Inputs, and
nothing else:

* ``shapes_summary.json`` (next to this file): dtype, shape and byte count of
  every tensor in the checkpoint, read from the 48 safetensors shard headers
  (the index carries neither shapes nor dtypes). Its byte sum equals the
  index's ``metadata.total_size`` and the script asserts that first.
* the catalogue entry ``gitm/planner/models/deepseek-v4-flash-0731.yaml``
  (Part 1.1), for config-level values: layer schedule, window, top-k, batch
  shapes. Each of those carries its own ``config.json`` / ``model.py``
  citation in the YAML.
* the planner's H200 constants (``gitm/planner/context.py``) for peaks and
  bandwidths; HBM capacity is the NVIDIA datasheet's 141 GB, which the
  planner does not model.
* ``inference/model.py`` at the pinned revision, for which tensors shard
  under tensor parallelism and how the KV cache is laid out. Cited inline as
  ``model.py:N``. Never executed.

Baseline operating point (the case, verbatim): one 8xH200 SXM node over
NVLink, TP8, FP8 weights and KV where the checkpoint supports it; 32 active
sequences in decode, 8,192 cached tokens each, one new token per sequence per
step, text-only, speculation disabled.

Units: GB and MB are decimal (1e9, 1e6 bytes), as in the GLM-5.2 note and the
Hugging Face file sizes. Time floors are roofline floors at vendor peak,
``max(bytes / HBM_bw, flops / peak_for_dtype)``, per rank, summed without
overlap unless a line says otherwise. They are lower bounds, not targets.

Run from the repo root::

    .venv/bin/python docs/deepseek-v4-flash/ledger.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from gitm.planner.context import hardware_spec_for, peak_for_sku  # noqa: E402
from gitm.planner.model_catalogue import load_spec  # noqa: E402
from gitm.planner.roofline import distinct_experts  # noqa: E402

# ── baseline ────────────────────────────────────────────────────────────────
B = 32  # active sequences, one new token each
S = 8192  # cached tokens per sequence
TP = 8  # tensor-parallel ranks == world_size in model.py
GB, MB = 1e9, 1e6
HBM_CAPACITY = 141e9  # NVIDIA H200 SXM datasheet; not in the planner

# ── inputs ──────────────────────────────────────────────────────────────────
D = json.loads((HERE / "shapes_summary.json").read_text())
assert D["header_total"] == D["total_size"], "shard headers do not account for the checkpoint"
SPEC = load_spec("deepseek-v4-flash-0731")
HW = hardware_spec_for(peak_for_sku("H200"))
assert HW.peak_flops_fp8_per_s > 0 and HW.interconnect_bw_bytes_per_s > 0

N_LAYERS = SPEC.n_layers
KINDS = {int(k): v for k, v in D["layer_kind"].items()}  # "swa+hash" | "csa+hash" | "csa" | "hca"
assert len(KINDS) == N_LAYERS == 43
N_BY_KIND = {k: sum(1 for v in KINDS.values() if v == k) for k in set(KINDS.values())}
BASE_KIND = {k: k.split("+")[0] for k in N_BY_KIND}  # attention kind without the hash tag
assert sum(N_BY_KIND[k] for k in N_BY_KIND if BASE_KIND[k] == "csa") == 21
assert sum(N_BY_KIND[k] for k in N_BY_KIND if BASE_KIND[k] == "hca") == 20
assert sum(N_BY_KIND[k] for k in N_BY_KIND if BASE_KIND[k] == "swa") == 2

PEAK = {
    "fp8": HW.peak_flops_fp8_per_s,
    "fp4": HW.peak_flops_fp8_per_s,  # Hopper has no fp4 tensor core: fp8 fallback, flagged
    "bf16": HW.peak_flops_bf16_per_s,
    "fp32": HW.peak_flops_fp32_per_s,
}
BW = HW.peak_mem_bw_bytes_per_s
LINK = HW.interconnect_bw_bytes_per_s
AW = 2  # bf16 activation bytes (torch_dtype)


def fmt(b: float) -> str:
    return f"{b / GB:8.3f} GB" if b >= 0.5 * GB else f"{b / MB:8.1f} MB"


# ── 1. sharding rule per tensor, from model.py ──────────────────────────────
# "sharded": each rank holds 1/TP of the bytes. "replicated": every rank holds all.
def shard_rule(c: str) -> tuple[str, str]:
    """(rule, evidence) for a canonical (layer- and expert-index-stripped) tensor name."""
    if c.startswith("ffn.experts.N."):
        return "sharded", "model.py:623 n_local_experts = n_routed_experts // world_size"
    if c.startswith("ffn.shared_experts."):
        return "replicated", "model.py:632 Expert(...) -> plain Linear (model.py:601-603)"
    if c.startswith("ffn.gate."):
        return "replicated", "model.py:563-568 nn.Parameter, not sharded"
    if c.startswith("attn.indexer.compressor.") or c.startswith("attn.compressor."):
        return "replicated", "model.py:298-299 Compressor uses plain Linear"
    if c.startswith("attn.indexer.wq_b."):
        return "sharded", "model.py:399 ColumnParallelLinear"
    if c.startswith("attn.indexer.weights_proj."):
        return "sharded", "model.py:400 ColumnParallelLinear"
    if c.startswith("attn.wq_b.") or c.startswith("attn.wo_a."):
        return "sharded", "model.py:465, :468 ColumnParallelLinear"
    if c.startswith("attn.wo_b."):
        return "sharded", "model.py:469 RowParallelLinear"
    if c.startswith("attn.wq_a.") or c.startswith("attn.wkv."):
        return "replicated", "model.py:463, :466 plain Linear"
    if c == "attn.attn_sink":
        return "sharded", "model.py:462 n_local_heads"
    if "markov" in c:
        return "sharded", "model.py:776-777 ParallelEmbedding / ParallelHead"
    if c in ("embed.weight", "head.weight"):
        return "sharded", "model.py:97 ParallelEmbedding, :727 ParallelHead"
    # norms, hyper-connection generators, sinks-of-globals, main_proj, confidence head
    return "replicated", "model.py:672-677 / :832 / :790 nn.Parameter or plain Linear"


def ledger_class(c: str) -> str:
    if c.startswith("ffn.experts.N."):
        return "routed_expert_scale" if c.endswith(".scale") else "routed_expert_weight"
    if c.startswith("ffn.shared_experts."):
        return "shared_expert_scale" if c.endswith(".scale") else "shared_expert_weight"
    if c.startswith("ffn.gate."):
        return "router"
    if c.startswith("attn.indexer.compressor.") or c.startswith("attn.compressor."):
        return "compressor"
    if c.startswith("attn.indexer."):
        return "indexer_scale" if c.endswith(".scale") else "indexer_weight"
    if c.startswith("attn.w"):
        return "attn_linear_scale" if c.endswith(".scale") else "attn_linear_weight"
    if c.startswith("hc_"):
        return "hyper_connection"
    if c in ("embed.weight", "head.weight"):
        return "embed_head"
    return "norm_sink_other"


DTYPE_LABEL = {
    "I8": "fp4x2",
    "F8_E4M3": "fp8",
    "F8_E8M0": "e8m0",
    "BF16": "bf16",
    "F32": "fp32",
    "I64": "i64",
}


@dataclass
class Line:
    """One ledger class: bytes model-wide, per rank, and the part TP does not divide."""

    cls: str
    model_bytes: float = 0.0
    rank_bytes: float = 0.0
    replicated_bytes: float = 0.0
    by_dtype: dict[str, float] = field(default_factory=dict)

    @property
    def dtype_label(self) -> str:
        """Dominant dtype by bytes, '+' if the class mixes dtypes (compressor: bf16 + fp32 ape)."""
        top = max(self.by_dtype, key=self.by_dtype.get)
        return DTYPE_LABEL.get(top, top) + ("+" if len(self.by_dtype) > 1 else "")


def accumulate(table: dict[str, dict], count_layers: int, lines: dict[str, Line]) -> None:
    for c, e in table.items():
        cls = ledger_class(c)
        rule, _ = shard_rule(c)
        b = e["bytes"] * count_layers
        ln = lines.setdefault(cls, Line(cls))
        ln.by_dtype[e["dtype"]] = ln.by_dtype.get(e["dtype"], 0) + b
        ln.model_bytes += b
        if rule == "sharded":
            ln.rank_bytes += b / TP
        else:
            ln.rank_bytes += b
            ln.replicated_bytes += b


STACK: dict[str, Line] = {}
for kind, n in N_BY_KIND.items():
    accumulate(D["kind_table"][kind], n, STACK)
GLOBALS: dict[str, Line] = {}
accumulate(D["globals"], 1, GLOBALS)
MTP: dict[str, Line] = {}
for _m, table in D["mtp"].items():
    accumulate(table, 1, MTP)

stack_model = sum(x.model_bytes for x in STACK.values())
glob_model = sum(x.model_bytes for x in GLOBALS.values())
mtp_model = sum(x.model_bytes for x in MTP.values())
assert stack_model + glob_model + mtp_model == D["total_size"], (
    "ledger does not sum to the checkpoint"
)
assert stack_model == sum(D["stack_class_totals"].values())

# ── 2. quantization metadata, both granularities ────────────────────────────
rw = STACK["routed_expert_weight"]  # stored I8 = two fp4 values per byte (model.py:138-140)
rs = STACK["routed_expert_scale"]
fp4_logical_params = 2 * rw.model_bytes
assert rs.model_bytes == fp4_logical_params / 32, "expert scales are not one byte per 32 params"
scales_if_128x128 = fp4_logical_params / (128 * 128)
fp8_weights = (
    STACK["attn_linear_weight"].model_bytes
    + STACK["indexer_weight"].model_bytes
    - D["kind_table"]["csa"]["attn.indexer.weights_proj.weight"]["bytes"] * 21  # bf16, no scale
    + STACK["shared_expert_weight"].model_bytes
)
fp8_scales = (
    STACK["attn_linear_scale"].model_bytes
    + STACK["indexer_scale"].model_bytes
    + STACK["shared_expert_scale"].model_bytes
)

# ── 3. KV cache at B=32, S=8192 ─────────────────────────────────────────────
HD, RD, WIN = SPEC.head_dim, SPEC.qk_rope_head_dim, SPEC.sliding_window  # 512, 64, 128
IDX_HD, TOPK = SPEC.index_head_dim, SPEC.index_topk  # 128, 512
assert HD == 512 and RD == 64 and WIN == 128 and TOPK == 512
# One 512-wide entry is both key and value for all heads (model.py:466, :533).
# fp8 store: 448 non-RoPE dims fp8 with one e8m0 scale per 64 (model.py:511
# act_quant(kv[..., :-rd], 64, ...); scale_dtype e8m0 at :884-887), 64 RoPE dims bf16.
KV_ENTRY = {
    "fp8": (HD - RD) * 1 + (HD - RD) // 64 * 1 + RD * 2,  # 583 B
    "bf16": HD * 2,  # 1024 B — what the reference actually stores (model.py:480, :532)
}
# Indexer key: 128 dims. fp4-simulated with block 32 in the reference (model.py:356,
# :421-425), stored bf16. Priced at 1 B/elem for an fp8 store (planner convention).
IDX_ENTRY = {"fp8": IDX_HD, "bf16": IDX_HD * 2, "fp4": IDX_HD // 2 + IDX_HD // 32}
RATIO = {"csa": 4, "hca": 128}


def kv_entries_per_seq(kind: str, s: int) -> tuple[int, int]:
    """(attention entries held, index keys held) for one sequence at context s."""
    base = BASE_KIND[kind]
    held = WIN  # every layer keeps a window buffer (model.py:479), not only swa layers
    idx = 0
    if base != "swa":
        held += s // RATIO[base]
    if base == "csa":
        idx = s // 4  # indexer.kv_cache is [B, max_seq_len // 4, 128] (model.py:405)
    return held, idx


def kv_entries_read(kind: str, s: int) -> tuple[int, int]:
    """(attention entries read, index keys scanned) per sequence per step."""
    base = BASE_KIND[kind]
    if base == "swa":
        return WIN, 0  # model.py:513
    if base == "hca":
        return WIN + s // 128, 0  # every compressed entry (model.py:519-520)
    return WIN + min(TOPK, s // 4), s // 4  # model.py:433, :426


def kv_bytes_per_seq(s: int, store: str) -> tuple[float, float]:
    att = idx = 0.0
    for kind, n in N_BY_KIND.items():
        h, i = kv_entries_per_seq(kind, s)
        att += n * h * KV_ENTRY[store]
        idx += n * i * IDX_ENTRY[store]
    return att, idx


# Reference-implementation compressor state (model.py:306-307): kv_state and
# score_state, fp32, [coff*ratio, coff*head_dim] per sequence per compressor.
def compressor_state_per_seq() -> float:
    total = 0.0
    for kind, n in N_BY_KIND.items():
        base = BASE_KIND[kind]
        if base == "swa":
            continue
        r = RATIO[base]
        coff = 2 if r == 4 else 1
        total += n * 2 * (coff * r) * (coff * HD) * 4
        if base == "csa":
            total += n * 2 * (coff * r) * (coff * IDX_HD) * 4  # the indexer's own compressor
    return total


# ── 4. per-rank time floors, by region ──────────────────────────────────────
@dataclass
class Region:
    name: str
    flops: float = 0.0
    hbm_bytes: float = 0.0
    link_bytes: float = 0.0
    n_collectives: int = 0
    dtype: str = "fp8"
    launches: int = 0

    @property
    def t_mem(self) -> float:
        return self.hbm_bytes / BW

    @property
    def t_comp(self) -> float:
        return self.flops / PEAK[self.dtype]

    @property
    def t_link(self) -> float:
        return self.link_bytes / LINK

    @property
    def floor(self) -> float:
        return max(self.t_mem, self.t_comp, self.t_link)

    @property
    def bound(self) -> str:
        if self.n_collectives:
            return "communication"
        return "compute" if self.t_comp > self.t_mem else "memory"


def tensor(kind: str, name: str) -> dict:
    return D["kind_table"][kind][name]


def logical_elems(e: dict) -> int:
    n = math.prod(e["shape"])
    return 2 * n if e["dtype"] == "I8" else n  # fp4 packs two per byte (model.py:138)


def linear_cost(e: dict, scale: dict | None, sharded: bool, rows: int) -> tuple[float, float]:
    """(flops, hbm bytes) per rank for a GEMM of ``rows`` rows against one stored linear."""
    div = TP if sharded else 1
    w = (e["bytes"] + (scale["bytes"] if scale else 0)) / div
    n_out, n_in = e["shape"][0], e["shape"][1]
    if e["dtype"] == "I8":
        n_in *= 2
    act = AW * rows * (n_in + n_out / div)
    return 2.0 * rows * logical_elems(e) / div, w + act


REGIONS: dict[str, Region] = {}


def reg(name: str, dtype: str = "fp8") -> Region:
    """The region ``name``, created on first use; ``dtype`` picks its compute peak."""
    return REGIONS.setdefault(name, Region(name, dtype=dtype))


# embed: a gather of B rows of embed.weight (model.py:107-110); its all-reduce is
# counted under collectives
r = reg("embed", "bf16")
r.hbm_bytes += B * SPEC.hidden * AW * 2
r.launches += 1

for kind, n in N_BY_KIND.items():
    base = BASE_KIND[kind]
    t = D["kind_table"][kind]
    for _ in range(n):
        # attention projections, fp8 + 128x128 e8m0 scales: wq_a, wkv replicated;
        # wq_b, wo_a, wo_b sharded
        r = reg("attn_proj")
        for w, sh in (
            ("wq_a", False),
            ("wkv", False),
            ("wq_b", True),
            ("wo_a", True),
            ("wo_b", True),
        ):
            f, b = linear_cost(t[f"attn.{w}.weight"], t[f"attn.{w}.scale"], sh, B)
            r.flops += f
            r.hbm_bytes += b
            r.launches += 1
        r.launches += 3  # q_norm, kv_norm, rope/act_quant (model.py:501-512)

        # compressors: bf16 weights as stored, replicated, run every token
        # (model.py:329-330); csa layers also run the indexer's own compressor
        if base != "swa":
            r = reg("attn_compress", "bf16")
            for w in ("attn.compressor.wkv.weight", "attn.compressor.wgate.weight"):
                f, b = linear_cost(t[w], None, False, B)
                r.flops += f
                r.hbm_bytes += b
            r.launches += 3
            if base == "csa":
                for w in (
                    "attn.indexer.compressor.wkv.weight",
                    "attn.indexer.compressor.wgate.weight",
                ):
                    f, b = linear_cost(t[w], None, False, B)
                    r.flops += f
                    r.hbm_bytes += b
                r.launches += 3

        # indexer: query projection (sharded) + scan over all S/4 keys (replicated read)
        if base == "csa":
            r = reg("attn_index")
            f, b = linear_cost(t["attn.indexer.wq_b.weight"], t["attn.indexer.wq_b.scale"], True, B)
            r.flops += f
            r.hbm_bytes += b
            f, b = linear_cost(t["attn.indexer.weights_proj.weight"], None, True, B)
            r.flops += f
            r.hbm_bytes += b
            _, scanned = kv_entries_read(kind, S)
            local_heads = SPEC.index_n_heads // TP
            r.flops += 2.0 * B * local_heads * IDX_HD * scanned
            r.hbm_bytes += B * scanned * IDX_ENTRY["fp8"]
            r.launches += 5

        # attention core: every rank reads the whole selected set (one KV head, model.py:533)
        r = reg("attn_core")
        read, _ = kv_entries_read(kind, S)
        r.flops += 4.0 * B * (SPEC.n_heads // TP) * HD * read  # qk and pv
        r.hbm_bytes += B * read * KV_ENTRY["fp8"] + B * (SPEC.n_heads // TP) * HD * AW * 2
        r.launches += 1

        # hyper-connections: two fp32 generators per layer, replicated (model.py:672-678),
        # plus three passes over the 4x-wide fp32 residual state
        r = reg("hc_mix", "fp32")
        for w in ("hc_attn_fn", "hc_ffn_fn"):
            e = t[w]
            r.flops += 2.0 * B * math.prod(e["shape"]) + 4.0 * B * 16 * SPEC.hc_sinkhorn_iters
            r.hbm_bytes += e["bytes"] + 3 * B * SPEC.hc_width * SPEC.hidden * 4
            r.launches += 4  # norm, generator GEMM, sinkhorn, mix (model.py:680-693)

        # router: gate.weight bf16 [256, 4096] replicated, scored in fp32; runs on the
        # hash layers too (model.py:570 is unconditional)
        r = reg("moe_router", "fp32")
        f, b = linear_cost(t["ffn.gate.weight"], None, False, B)
        r.flops += f
        r.hbm_bytes += b
        r.launches += 2

        # shared expert: w1/w2/w3 fp8, REPLICATED because the reference builds it from
        # plain Linear (model.py:632); section H prices the sharded reading
        r = reg("moe_shared")
        for w in ("w1", "w2", "w3"):
            f, b = linear_cost(
                t[f"ffn.shared_experts.{w}.weight"], t[f"ffn.shared_experts.{w}.scale"], False, B
            )
            r.flops += f
            r.hbm_bytes += b
        r.launches += 3

        # routed experts: distinct-expert union under uniform routing, 1/TP per rank
        r = reg("moe_routed", "fp4")
        r.launches += 3
        # filled below once per layer count, from the same per-expert bytes

# routed experts, from the per-expert bytes measured on any layer (all identical)
csa_t = D["kind_table"]["csa"]
per_expert_w = (
    sum(csa_t[f"ffn.experts.N.{w}.weight"]["bytes"] for w in ("w1", "w2", "w3"))
    / SPEC.n_routed_experts
)
per_expert_s = (
    sum(csa_t[f"ffn.experts.N.{w}.scale"]["bytes"] for w in ("w1", "w2", "w3"))
    / SPEC.n_routed_experts
)
per_expert = per_expert_w + per_expert_s
per_expert_params = 3 * SPEC.hidden * SPEC.moe_intermediate_size
assert per_expert_w == per_expert_params / 2 and per_expert_s == per_expert_params / 32
K, E = SPEC.num_experts_per_tok, SPEC.n_routed_experts
distinct_uniform = E * (1 - (1 - K / E) ** B)
assert abs(distinct_uniform - distinct_experts(B, E, K)) < 1e-9, (
    "disagrees with roofline.distinct_experts"
)
distinct_min, distinct_max = K, min(B * K, E)
r = REGIONS["moe_routed"]
r.flops = N_LAYERS * 2.0 * B * K * per_expert_params / TP
r.hbm_bytes = N_LAYERS * (
    distinct_uniform * per_expert / TP
    + AW * B * K * (SPEC.hidden + 2 * SPEC.moe_intermediate_size) / TP
)

# collectives (model.py lines): per layer an fp32 all-reduce after wo_b (:182-183) and
# after the routed experts (:647); an fp32 index-score all-reduce on csa layers
# (:428-429); one bf16 embedding all-reduce (:110); one fp32 logits all-gather (:738).
r = reg("collectives", "bf16")
ring = 2.0 * (TP - 1) / TP
per_layer_ar = [B * SPEC.hidden * 4, B * SPEC.hidden * 4]
for kind, n in N_BY_KIND.items():
    for _ in range(n):
        for p in per_layer_ar:
            r.link_bytes += ring * p
            r.n_collectives += 1
        if BASE_KIND[kind] == "csa":
            r.link_bytes += ring * B * (S // 4) * 4
            r.n_collectives += 1
r.link_bytes += ring * B * SPEC.hidden * AW  # embedding all-reduce
r.n_collectives += 1
logits_per_rank = B * (SPEC.vocab // TP) * 4
r.link_bytes += (TP - 1) / TP * logits_per_rank * TP  # all-gather: receive the other 7 shards
r.n_collectives += 1
r.launches = r.n_collectives

# lm_head: head.weight bf16 [vocab/8, 4096] per rank, plus the final norm and hc_head
# (model.py:728 keeps an fp32 copy: see H)
r = reg("lm_head", "bf16")
f, b = linear_cost(D["globals"]["head.weight"], None, True, B)
r.flops += f
r.hbm_bytes += b + D["globals"]["hc_head_fn"]["bytes"] + B * SPEC.hc_width * SPEC.hidden * 4 * 2
r.launches += 3

ORDER = [
    "moe_routed",
    "attn_proj",
    "attn_compress",
    "moe_shared",
    "attn_core",
    "attn_index",
    "hc_mix",
    "collectives",
    "lm_head",
    "moe_router",
    "embed",
]
assert set(ORDER) == set(REGIONS)
step_floor = sum(x.floor for x in REGIONS.values())
n_launches = sum(x.launches for x in REGIONS.values())


# ── 5. print ────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n## {title}\n")


print("DeepSeek-V4-Flash-0731 @ 7872f01b — per-GPU ledger and decode floors")
print(f"baseline: 8xH200 SXM, TP{TP}, B={B}, S={S}, one token per sequence, speculation off")
print(
    f"hardware (gitm/planner/context.py, H200): HBM {BW / 1e12:.1f} TB/s, fp8 {PEAK['fp8'] / 1e12:.0f} TF/s, "
    f"bf16 {PEAK['bf16'] / 1e12:.0f} TF/s, fp32 {PEAK['fp32'] / 1e12:.0f} TF/s, NVLink {LINK / 1e9:.0f} GB/s; "
    f"HBM capacity {HBM_CAPACITY / GB:.0f} GB (datasheet)"
)
print(
    f"checkpoint: {D['total_size']:,} B = {D['total_size'] / GB:.2f} GB in 48 shards; "
    f"executed stack {stack_model / GB:.2f} GB, globals {glob_model / GB:.2f} GB, mtp.* {mtp_model / GB:.2f} GB"
)

section("A. Resident weights per rank at TP8 (executed stack + globals)")
print(
    f"{'class':24s} {'dtype':8s} {'model-wide':>12s} {'per rank':>12s} {'of which replicated':>20s}   ('+' = minor tensors of another dtype)"
)
rows = sorted(STACK.values(), key=lambda x: -x.rank_bytes) + sorted(
    GLOBALS.values(), key=lambda x: -x.rank_bytes
)
resident_rank = 0.0
for ln in rows:
    tag = "" if ln in STACK.values() else " (global)"
    print(
        f"{ln.cls + tag:24s} {ln.dtype_label:8s} {fmt(ln.model_bytes):>12s} {fmt(ln.rank_bytes):>12s} {fmt(ln.replicated_bytes):>20s}"
    )
    resident_rank += ln.rank_bytes
repl_rank = sum(x.replicated_bytes for x in rows)
print(
    f"{'TOTAL resident weights':24s} {'':8s} {fmt(stack_model + glob_model):>12s} {fmt(resident_rank):>12s} {fmt(repl_rank):>20s}"
)
check = resident_rank * TP - repl_rank * (TP - 1)
assert abs(check - (stack_model + glob_model)) < 1, "per-rank ledger does not re-sum to the model"
print(f"check: per-rank x {TP} - replicated x {TP - 1} = {check / GB:.3f} GB == model-wide (ok)")
mtp_rank = sum(x.rank_bytes for x in MTP.values())
print(
    f"mtp.* if loaded (engine-dependent, U9): {fmt(mtp_model)} model-wide -> {fmt(mtp_rank)} per rank under the same rules"
)

section("B. Quantization metadata")
print(
    f"routed experts: {fp4_logical_params / 1e9:.2f} G fp4 params -> payload {fmt(rw.model_bytes)} "
    f"+ scales {fmt(rs.model_bytes)} (one e8m0 byte per 32 params, model.py:18, :142) = "
    f"{100 * rs.model_bytes / rw.model_bytes:.3f} % of payload"
)
print(
    f"  if quantization_config's 128x128 block were applied to the experts: {fmt(scales_if_128x128)} "
    f"({rs.model_bytes / scales_if_128x128:.0f}x too small; {fmt((rs.model_bytes - scales_if_128x128) / TP)} per rank)"
)
print(
    f"fp8 linears (attention, indexer wq_b, shared expert): payload {fmt(fp8_weights)} + scales {fmt(fp8_scales)} "
    f"= {100 * fp8_scales / fp8_weights:.4f} % (128x128, quantization_config.weight_block_size)"
)
print(
    f"metadata per rank: fp4 scales {fmt(rs.rank_bytes)} + fp8 scales {fmt(STACK['attn_linear_scale'].rank_bytes + STACK['indexer_scale'].rank_bytes + STACK['shared_expert_scale'].rank_bytes)}"
)

section(f"C. KV cache at B={B}, S={S} (replicated on every rank: one KV head, model.py:466, :533)")
for kind in sorted(N_BY_KIND):
    h, i = kv_entries_per_seq(kind, S)
    rd, sc = kv_entries_read(kind, S)
    print(
        f"  {kind:9s} x{N_BY_KIND[kind]:<3d} holds {h:5d} entries + {i:5d} index keys per seq;  reads {rd:4d} entries + scans {sc:5d} keys per step"
    )
for store in ("fp8", "bf16"):
    att, idx = kv_bytes_per_seq(S, store)
    print(
        f"  {store:4s} store: entry {KV_ENTRY[store]} B, key {IDX_ENTRY[store]} B -> per seq {fmt(att + idx)} "
        f"(attention {fmt(att)}, index keys {fmt(idx)}); x{B} = {fmt(B * (att + idx))} per rank"
    )
kv_att_fp8, kv_idx_fp8 = kv_bytes_per_seq(S, "fp8")
kv_rank_fp8 = B * (kv_att_fp8 + kv_idx_fp8)
kv_rank_bf16 = B * sum(kv_bytes_per_seq(S, "bf16"))
comp_state = B * compressor_state_per_seq()
print(
    f"  compressor state buffers (reference impl., fp32, model.py:306-307): {fmt(compressor_state_per_seq())} per seq, {fmt(comp_state)} for {B}"
)
act_bytes = B * SPEC.hc_width * SPEC.hidden * AW * 4 + B * SPEC.vocab * 4
print(
    f"  activations + full logits (B x hc x hidden bf16, a few copies; B x vocab fp32): {fmt(act_bytes)}"
)

section("D. Per-GPU ledger against 141 GB")
lines = [
    ("weights, sharded + replicated (A)", resident_rank, "checkpoint"),
    (
        "KV, 32 x 8192, fp8 store (C)",
        kv_rank_fp8,
        "checkpoint layout; fp8 store is a serving choice (U1)",
    ),
    ("activations + logits", act_bytes, "checkpoint shapes"),
]
lower_bound = sum(v for _, v, _ in lines)
for name, v, src in lines:
    print(f"  {name:44s} {fmt(v):>12s}   {src}")
print(
    f"  {'CAPACITY LOWER BOUND':44s} {fmt(lower_bound):>12s}   {100 * lower_bound / HBM_CAPACITY:.1f} % of {HBM_CAPACITY / GB:.0f} GB"
)
allow = [
    ("mtp.* resident (U9)", mtp_rank, "assumed loaded; 0 if the engine skips a head it never runs"),
    (
        "compressor state (reference buffers)",
        comp_state,
        "reference impl.; an engine may recompute from the window",
    ),
    (
        "GEMM / kernel workspace",
        0.25 * GB,
        "ASSUMED allowance; read torch.cuda.memory_reserved on the served engine",
    ),
    (
        "communication buffers (NCCL)",
        0.5 * GB,
        "ASSUMED allowance; NCCL_BUFFSIZE x channels, engine-dependent",
    ),
    (
        "CUDA context + allocator reserve",
        1.0 * GB,
        "ASSUMED; nvidia-smi 'Used' on an idle loaded engine settles it",
    ),
]
deployable = lower_bound
for name, v, src in allow:
    print(f"  {name:44s} {fmt(v):>12s}   {src}")
    deployable += v
print(
    f"  {'DEPLOYABLE CONFIGURATION at B=32':44s} {fmt(deployable):>12s}   {100 * deployable / HBM_CAPACITY:.1f} % of {HBM_CAPACITY / GB:.0f} GB"
)
kv_budget = HBM_CAPACITY - (deployable - kv_rank_fp8)
per_seq_fp8 = kv_att_fp8 + kv_idx_fp8
print(
    f"  KV budget left after everything else: {fmt(kv_budget)} = {kv_budget / per_seq_fp8:.0f} sequences at 8,192 tokens (fp8 store), "
    f"or {kv_budget / (B * per_seq_fp8 / S):,.0f} tokens per sequence at B={B}"
)
print(f"  bf16 KV store instead (reference): +{fmt(kv_rank_bf16 - kv_rank_fp8)} per rank")
sharded_model = resident_rank * TP - repl_rank * TP  # bytes that divide by TP
for tp_alt in (4, 2):
    print(
        f"  for scale, weights per rank at TP{tp_alt}: {fmt(sharded_model / tp_alt + repl_rank)} "
        f"(+ KV {fmt(kv_rank_fp8)} replicated, unchanged)"
    )

section("E. Decode-step floors per rank (roofline at vendor peak, summed, no overlap)")
print(
    f"{'region':14s} {'GFLOP':>8s} {'HBM MB':>9s} {'link MB':>8s} {'t_comp':>8s} {'t_mem':>8s} {'floor ms':>9s} {'share':>6s}  bound          launches"
)
for name in sorted(ORDER, key=lambda n: -REGIONS[n].floor):
    x = REGIONS[name]
    print(
        f"{name:14s} {x.flops / 1e9:8.2f} {x.hbm_bytes / MB:9.1f} {x.link_bytes / MB:8.2f} "
        f"{x.t_comp * 1e3:8.4f} {x.t_mem * 1e3:8.4f} {x.floor * 1e3:9.4f} {100 * x.floor / step_floor:5.1f} %  "
        f"{x.bound:14s} {x.launches:5d}"
    )
print(
    f"{'STEP FLOOR':14s} {sum(x.flops for x in REGIONS.values()) / 1e9:8.2f} {sum(x.hbm_bytes for x in REGIONS.values()) / MB:9.1f} "
    f"{sum(x.link_bytes for x in REGIONS.values()) / MB:8.2f} {'':8s} {'':8s} {step_floor * 1e3:9.4f} {'':6s}  "
    f"{B / step_floor:,.0f} tok/s   {n_launches:5d}"
)
hbm_total = sum(x.hbm_bytes for x in REGIONS.values())
print(
    f"step AI against HBM: {sum(x.flops for x in REGIONS.values()) / hbm_total:.1f} FLOP/B vs fp8 ridge {PEAK['fp8'] / BW:.0f}"
)
print(
    "launches: ESTIMATED by counting kernel-sized ops in model.py's forward (unfused reference); a fused engine issues fewer "
    "(the planner at main emits 636 nodes for this entry). The count prices nothing above; see H."
)

section("F. Expert bank")
print(
    f"per expert: {per_expert_params / 1e6:.1f} M params -> {per_expert / MB:.3f} MB (payload {per_expert_w / MB:.3f} + scales {per_expert_s / MB:.3f})"
)
shared_bytes = sum(
    csa_t[f"ffn.shared_experts.{w}.weight"]["bytes"]
    + csa_t[f"ffn.shared_experts.{w}.scale"]["bytes"]
    for w in ("w1", "w2", "w3")
)
print(
    f"selected expert-weight bytes per token per layer: {K} x {per_expert / MB:.3f} MB routed = {K * per_expert / MB:.2f} MB, "
    f"+ shared {shared_bytes / MB:.2f} MB = {(K * per_expert + shared_bytes) / MB:.2f} MB; x{N_LAYERS} layers = {N_LAYERS * (K * per_expert + shared_bytes) / GB:.2f} GB per token if nothing were shared"
)
print(f"batch of {B}, {B * K} assignments over {E} experts, per layer, model-wide routed bytes:")
for label, d in (
    ("best case (all tokens agree)", distinct_min),
    ("uniform routing, expectation", distinct_uniform),
    ("worst case (no collisions)", distinct_max),
    ("whole bank", E),
):
    print(
        f"  {label:32s} {d:6.1f} distinct experts -> {fmt(d * per_expert):>12s} model-wide, {fmt(d * per_expert / TP):>12s} per rank; "
        f"x{N_LAYERS}: {fmt(N_LAYERS * d * per_expert / TP):>12s} per rank per step = {N_LAYERS * d * per_expert / TP / BW * 1e3:.3f} ms"
    )
print(
    f"  roofline.distinct_experts({B}, {E}, {K}) = {distinct_experts(B, E, K):.3f} (identical formula, asserted)"
)
print(
    f"  reuse assumed: one HBM read per touched expert per layer per step (grouped GEMM); no reuse across layers "
    f"(distinct weights) or across steps (per-rank bank {N_LAYERS * E * per_expert / TP / GB:.1f} GB >> L2)"
)

section("G. Overlap declarations for Part 2 (graph.overlaps)")
floor_ms = {n: REGIONS[n].floor * 1e3 for n in REGIONS}
ov1 = min(floor_ms["moe_routed"], floor_ms["moe_shared"])
csa_comp = 0.0
for w in ("attn.compressor.wkv.weight", "attn.compressor.wgate.weight"):
    csa_comp += 21 * linear_cost(csa_t[w], None, False, B)[1]
csa_comp_ms = csa_comp / BW * 1e3
ov2 = min(floor_ms["attn_index"], csa_comp_ms)
print("graph:")
print("  floors_ms:")
for n in ORDER:
    print(f"    {n}: {floor_ms[n]:.4f}")
print("  overlaps:")
print(
    f"    - {{earlier: moe_routed, later: moe_shared, overlap_ms: {ov1:.4f}}}   # independent inputs, model.py:641-648"
)
print(
    f"    - {{earlier: attn_index, later: attn_compress, overlap_ms: {ov2:.4f}}}   # csa main compressor does not consume the index, model.py:516-528"
)
print(
    "  # reference implementation is single-stream: realised overlap 0; these are the ceilings an engine can claim"
)

section("H. Sensitivities (what moves the table above)")
t_sh = REGIONS["moe_shared"].floor * 1e3
print(
    f"  shared expert TP-sharded instead of replicated (engine): moe_shared {t_sh:.3f} -> {t_sh / TP:.3f} ms, step -{(t_sh - t_sh / TP) / (step_floor * 1e3) * 100:.1f} %"
)
print(
    f"  compressor + router + lm_head kept fp32 in HBM as the reference does (model.py:298, :570, :728): "
    f"+{(REGIONS['attn_compress'].hbm_bytes + REGIONS['moe_router'].hbm_bytes + REGIONS['lm_head'].hbm_bytes) / BW * 1e3:.3f} ms"
)
bf16_att = (
    B
    * sum(kv_entries_read(k, S)[0] * N_BY_KIND[k] for k in N_BY_KIND)
    * (KV_ENTRY["bf16"] - KV_ENTRY["fp8"])
)
print(
    f"  bf16 KV store (reference): attn_core +{bf16_att / BW * 1e3:.3f} ms; index keys bf16: +{B * 21 * (S // 4) * IDX_HD / BW * 1e3:.3f} ms"
)
nc = REGIONS["collectives"].n_collectives
print(
    f"  collectives: {nc} per step; bandwidth floor {REGIONS['collectives'].floor * 1e3:.3f} ms; "
    f"at 2 us each (planner launch constant) {nc * 2e-3:.3f} ms; at 10 us (typical small-message NCCL on NVLink) {nc * 10e-3:.3f} ms"
)
print(
    f"  kernel launches: ~{n_launches} per step; at 2 us (CUDA-graph replay) {n_launches * 2e-3:.3f} ms, at 5 us (eager) {n_launches * 5e-3:.3f} ms "
    f"— comparable to the {step_floor * 1e3:.3f} ms memory floor; Part 2's additive constant c"
)
worst = N_LAYERS * (distinct_max - distinct_uniform) * per_expert / TP / BW * 1e3
print(
    f"  routing skew: worst-case union adds +{worst:.3f} ms to moe_routed; any skew below uniform subtracts"
)
print(
    "  hash layers 0-2 (tid2eid): distinct count follows the token-id distribution; not determinable without a token histogram"
)
