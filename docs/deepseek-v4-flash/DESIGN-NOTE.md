# DeepSeek-V4-Flash-0731 — Working Design Note

**Predicted decode step for `deepseek-ai/DeepSeek-V4-Flash-0731` @ `7872f01b` on 8×H200 SXM, TP8, FP8 weights and KV**

Built from the checkpoint's own files at one pinned revision: `config.json`,
`model.safetensors.index.json`, the 48 safetensors shard headers (the index
carries no shapes or dtypes; the headers do), and `inference/model.py`, the
modeling implementation the checkpoint ships. **No traces, no serving engine,
no third-party writeups.** Every time figure is a roofline floor at vendor peak:
a lower bound, not a target.

Baseline operating point for every claim here, verbatim from the case: one
8×H200 SXM node over NVLink, TP8, FP8 weights and KV where the checkpoint
supports it; 32 active sequences in decode, 8,192 cached tokens each, one new
token per sequence per step, text-only, speculation disabled.

Reproduce every number in this note:

```bash
.venv/bin/python docs/deepseek-v4-flash/ledger.py          # sections 2-7 below, verbatim
gitm plan deepseek-v4-flash-0731 --gpu H200 --batch 32 --kv-len 8192 --tp 8   # the planner's own view (§4.3)
```

The ledger script reads only `docs/deepseek-v4-flash/shapes_summary.json` (every
tensor's dtype, shape and bytes from the shard headers; byte sum asserted equal
to the index's `total_size`, 166,878,536,440 B; regenerate it with
`docs/deepseek-v4-flash/shapes_from_shards.py`, which fetches only the two JSON
files and the 48 headers from Hugging Face at the pinned revision), the Part 1.1 catalogue entry
`gitm/planner/models/deepseek-v4-flash-0731.yaml` for config-level values, and
the planner's H200 constants. Lines cited as `model.py:N` are the pinned
`inference/model.py`; it was read, never run.

**Which revision and why.** 0731 is the newest text-only V4 Flash and the most
downloaded. Its 43 executed layers are tensor-for-tensor identical to the base
`DeepSeek-V4-Flash` revision (all 3,150 differing tensors sit under `mtp.*`,
which the baseline never runs), so the choice affects the speculative head and
the reconciliation, not this graph. `DeepSeek-V4.1-Flash` is a different
architecture (`deepseek_v41`, vision tower); `Vision-Exp` is not text-only.

**Serving engine: none pinned.** The executable semantics are the reference
implementation's. Everything it does not fix (KV storage dtype, whether the
shared expert is sharded, collective payload dtype, whether `mtp.*` is resident,
kernel fusion) is labelled ⚑ engine-dependent, both readings are priced, and §8
names the observation that settles each. Citing an engine's source without
running it would add a second unverified narrator.

**Two configs, two vocabularies.** Every V4 Flash repo ships a top-level
`config.json` (HF names: `hidden_size`, `num_hidden_layers`, `sliding_window`,
`num_experts_per_tok`) and `inference/config.json` (reference names: `dim`,
`n_layers`, `window_size`, `n_activated_experts`). `model.py` reads the
reference names. A `model.py` line cited for an HF key crosses that boundary;
the YAML maps the pairs.

---

## 1. Hardware constants

From `gitm/planner/context.py` (H200 entry), which agrees with NVIDIA's H200 SXM
page once every "with sparsity" tensor figure is halved. A dense decode step
gets none of the 2:4 uplift, and rooflining against the sparse peak would make
every region look twice as memory-bound as it is.

| Constant | Value used | Source |
| --- | --- | --- |
| HBM3e bandwidth | **4.8 TB/s** | `context.py:42`; datasheet 4.8 TB/s |
| FP8 e4m3 tensor peak | **1,979 TFLOP/s** | `context.py:82`; datasheet 3,958 "with sparsity", halved |
| BF16 tensor peak | **989 TFLOP/s** | `context.py:42`; datasheet 1,979 "with sparsity", halved |
| FP32 vector peak | **67 TFLOP/s** | `context.py:82`; the router and the compressors run here |
| FP4 | none on Hopper | fp4 experts price against the fp8 peak, flagged `peak_is_fallback` |
| NVLink per GPU | **900 GB/s** bidirectional | `context.py:100` |
| HBM capacity | **141 GB** | NVIDIA datasheet only; the planner does not model capacity |

```
ridge = peak / HBM_bw:   fp8 412 FLOP/B   bf16 206   fp32 14
```

The step below runs at 15.4 FLOP/B against HBM: **27× below the fp8 ridge**.
Nothing in a B=32 decode step of this checkpoint is compute-bound except the
fp32 router, and that node is 1.4 % of the floor.

---

## 2. What the checkpoint is (summary; the YAML carries the evidence line by line)

| Property | Value | Evidence |
| --- | --- | --- |
| Layers | **43** transformer blocks, every one MoE; **3** `mtp.*` blocks that run only under speculation | `layers.0..42`, `mtp.0..2` in the index; `model.py:912-926` vs `:928-936` |
| Attention kinds | **2 sliding-window** (layers 0, 1), **21 compressed-by-4 with a lightning indexer** (2, 4, …, 42), **20 compressed-by-128 attended densely** (3, 5, …, 41) | `compress_ratios[layer_id]`, `model.py:459`, `:472-477`; indexer tensors exist on exactly the 21 ratio-4 layers |
| KV entry | **one 512-wide vector per token per layer, key and value at once, shared by all 64 query heads** | `wkv.weight [512, 4096]`, `model.py:466`, `:533`; there is no `kv_lora_rank` and no `wkv_b` |
| RoPE | 64 of the 512 dims, *inside* `head_dim`, not added to it | `model.py:455`, `:505`, `:510`; `wq_b.weight [32768, 1024]` = 64 × 512 |
| Query path | `wq_a [1024, 4096]` → norm → `wq_b [32768, 1024]` | `q_lora_rank: 1024` |
| Output projection | 8 groups, **each carrying the full rank 1024**: `wo_a [8192, 4096]`, `wo_b [4096, 8192]` | `model.py:468-469` |
| Experts | 256 routed, top-6, one shared, intermediate 2048; **all 43 layers** | `ffn.experts.0..255` on every layer; `model.py:631` asserts one shared |
| Routing | `sqrtsoftplus` scores, `noaux_tc` bias for selection only, renormalised, ×1.5; **layers 0-2 pick experts by token id** (`tid2eid [129280, 6]`) but still run the router GEMM | `model.py:570-588` |
| Hyper-connections | residual stream 4× wide; two fp32 mixers `[24, 16384]` per layer, 20 Sinkhorn iterations | `model.py:670-693`, `:916` |
| Vocabulary | 129,280, untied `embed.weight` and `head.weight`, both bf16 | `tie_word_embeddings: false` |

### Precision, read from the shard headers, not from `quantization_config`

| Tensor class | Stored as | Scales | Evidence |
| --- | --- | --- | --- |
| routed experts (w1, w2, w3) | **fp4**, two per byte (header dtype `I8`, `[2048, 2048]` for a logical `[2048, 4096]`) | **one e8m0 byte per 32 values along K**, `[2048, 128]` | `model.py:18` `fp4_block_size = 32`, `:138-142` |
| shared expert | **fp8** e4m3, not fp4 | 128×128 e8m0, `[16, 32]` | `model.py:632` builds it without `expert_dtype`; `:884` default fp8 |
| attention linears, indexer `wq_b` | fp8 e4m3 | 128×128 e8m0 (`wq_b` scale `[256, 8]` = 32768/128 × 1024/128) | `quantization_config.weight_block_size [128, 128]`, `:144-148` |
| compressors (`wkv`, `wgate`), indexer `weights_proj`, router `gate.weight` | **bf16**, replicated, computed in fp32 | — | `model.py:298-299` comment, `:327`, `:400`, `:570` |
| hyper-connection mixers, `ape`, `attn_sink` | **fp32** | — | `model.py:672-678` |
| `embed`, `head` | bf16 | — | `:97`, `:727-728` |
| KV cache | **bf16 in the reference**, fp8-simulated on 448 dims with a per-64 scale, 64 RoPE dims bf16 | e8m0, one per 64 | `model.py:480`, `:511-512`, `:532` ("could also use fp8 … current implementation uses bf16") |

**The `quantization_config` block describes the fp8 path only.** Its 128×128
block applied to the experts would put their scales at 16.9 MB model-wide
where the headers show 8.657 GB — 512× low, 1.080 GB per rank (§3.2). The
planner's fp4 constant (`roofline.py:41`, one byte per 32 values) already has
this right.

---

## 3. Per-GPU memory ledger at TP8

Sharding follows `model.py`: `ColumnParallelLinear` / `RowParallelLinear` /
`ParallelEmbedding` / `ParallelHead` and the per-rank expert slice are
**sharded** (1/8 of the bytes per rank); plain `Linear` and `nn.Parameter` are
**replicated** (every rank holds all of it). The script re-sums the per-rank
ledger back to the model-wide total as a check (per-rank × 8 − replicated × 7
= 156.016 GB, exact).

### 3.1 Resident weights (executed stack + globals)

| Class | dtype | Model-wide | **Per rank** | Of which replicated |
| --- | --- | ---: | ---: | ---: |
| routed expert weights | fp4 | 138.513 GB | **17.314 GB** | 0 |
| routed expert scales | e8m0 | 8.657 GB | **1.082 GB** | 0 |
| shared expert (+ scales) | fp8 | 1.082 GB | **1.082 GB** ⚑ | 1.082 GB |
| attention linears (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`) | fp8 | 4.599 GB | 0.812 GB | 270.5 MB (`wq_a`, `wkv`) |
| compressors (41 layers + 21 indexer compressors) | bf16 (+ fp32 `ape`) | 0.614 GB | 0.614 GB | 0.614 GB |
| `embed.weight` + `head.weight` | bf16 | 2.118 GB | 264.8 MB | 0 |
| hyper-connection mixers | fp32 | 135.3 MB | 135.3 MB | 135.3 MB |
| router (`gate.weight`, bias, `tid2eid`) | bf16 (+ i64) | 108.8 MB | 108.8 MB | 108.8 MB |
| indexer `wq_b` + `weights_proj` | fp8, bf16 | 187.2 MB | 23.4 MB | 0 |
| norms, sinks, scales of fp8 linears | — | < 2 MB | < 2 MB | ~1 MB |
| **Total resident weights** | | **156.016 GB** | **21.437 GB** | **2.212 GB** |

⚑ The reference builds the shared expert from plain `Linear` (`model.py:632`,
`:601-603`), so every rank holds and *runs* all of it. A TP engine that shards
it along the intermediate dimension holds 135 MB per rank instead. The
replicated reading is carried because it is what the pinned code does; §4 and
§9 price the other.

`mtp.*` (three full MoE blocks plus `main_proj`, the Markov and confidence
heads): 10.863 GB model-wide, **1.499 GB per rank** under the same sharding
rules, if the engine loads a draft head it will never run. Not determinable
from the checkpoint (§8).

### 3.2 Quantization metadata

```
routed experts:  277.03 G fp4 params  ->  payload 138.513 GB + scales 8.657 GB   (6.250 % of payload)
                 per rank: 1.082 GB of scales
                 if the 128x128 fp8 block were applied instead:   16.9 MB   (512x too small, 1.080 GB/rank)
fp8 linears:     payload 5.857 GB + scales 0.4 MB                              (0.006 %)
```

The expert scales are a **1.08 GB per-rank ledger line and 6.25 % of the
dominant traffic term** (§4). They are not a rounding term, and the block size
`quantization_config` advertises would hide them.

### 3.3 KV cache at B=32, S=8,192

**Replicated on every rank.** `wkv` is a plain `Linear` producing the full
512-vector on each rank (`model.py:466`); the cache buffer is per-rank and
whole (`:480`); the attention is sharded by query heads (8 per rank) and each
rank gathers from the whole cache (`:533`). **Tensor parallelism buys no KV
capacity and no KV bandwidth on this architecture.**

Entries held per sequence, from the buffer layout at `model.py:479`
(`window_size + max_seq_len // ratio`) counted at the used length:

| Layer kind | × | Attention entries held | Index keys held | Entries read per step | Keys scanned per step |
| --- | --- | ---: | ---: | ---: | ---: |
| sliding-window | 2 | 128 | 0 | 128 | 0 |
| compressed ×4 + indexer | 21 | 128 + 2,048 = 2,176 | 2,048 | 128 + min(512, 2048) = 640 | 2,048 |
| compressed ×128 | 20 | 128 + 64 = 192 | 0 | 192 | 0 |

Every layer keeps the 128-entry window (`:479`), not only the two window layers.
Bytes per entry: **fp8 store** 448 × 1 B + 7 e8m0 scales + 64 × 2 B = **583 B**
(`:511`, `act_quant(kv[..., :-rd], 64, …)`); **bf16 store**, which is what the
reference actually allocates, 1,024 B. Index keys 128 B (fp8, 1 B/elem) or
256 B (bf16).

```
fp8 store:   34.5 MB per sequence (attention 29.0 + index keys 5.5)   x32 =  1.105 GB per rank
bf16 store:  62.0 MB per sequence (51.0 + 11.0)                        x32 =  1.984 GB per rank
```

Two further per-sequence buffers the reference holds, both fp32
(`model.py:306-307`): the compressor's `kv_state` and `score_state`,
**12.2 MB per sequence, 390.6 MB for 32**. An engine may instead recompute the
pending window from the raw cache. ⚑ Engine-dependent; carried as its own line.

### 3.4 The ledger

| Line | Per rank | Source |
| --- | ---: | --- |
| weights, sharded + replicated (§3.1) | 21.437 GB | checkpoint |
| KV cache, 32 × 8,192, fp8 store (§3.3) | 1.105 GB | checkpoint layout; fp8 *storage* is a serving choice |
| activations + full logits (B × 4 × 4096 bf16, a few copies; B × 129,280 fp32) | 20.7 MB | checkpoint shapes |
| **Capacity lower bound** | **22.563 GB** | **16.0 % of 141 GB** |
| `mtp.*` resident | 1.499 GB | ⚑ assumed loaded; 0 if the engine skips it |
| compressor state buffers | 390.6 MB | ⚑ reference implementation |
| GEMM / kernel workspace | 250 MB | **assumed allowance** |
| communication buffers (NCCL) | 500 MB | **assumed allowance** |
| CUDA context + allocator reserve | 1.000 GB | **assumed allowance** |
| **Deployable configuration at B=32** | **26.202 GB** | **18.6 % of 141 GB** |

The three allowances are stated, not derived: the checkpoint says nothing about
them. `torch.cuda.memory_reserved()` and `nvidia-smi` on an idle loaded engine
replace all three with measurements.

**What is left is the real result.** After everything else, **115.9 GB per rank
remains for KV**, which is 3,356 sequences at 8,192 tokens, or 859 k tokens per
sequence at B=32 (fp8 store). At TP8 this checkpoint occupies under a fifth of
the node's HBM; the fit question is not whether B=32 × 8,192 fits but that
nothing in memory constrains the baseline at all. For scale, the weights alone
would be 40.7 GB per rank at TP4 and 79.1 GB at TP2, and the replicated KV
line does not change with TP. Memory does not argue for TP8 here; latency and
the collective count (§4) are where that argument has to be made.

---

## 4. The decode step: regions, floors, and the top three

One rank, B=32 positions, S=8,192, TP8. Each region is a set of kernels a
capture can pair against; floor = max(bytes / 4.8 TB/s, FLOPs / peak for the
op's dtype), per rank, summed with no overlap (§6 declares what could overlap).
Collectives are priced against NVLink bandwidth only, the planner's convention,
and the latency reading is in §9.

| Region | GFLOP | HBM MB | link MB | t_comp ms | t_mem ms | **Floor ms** | Share | Bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `moe_routed` | 51.94 | 9,800.6 | | 0.026 | 2.042 | **2.0418** | 67.1 % | **memory** |
| `moe_shared` ⚑ | 69.26 | 1,132.9 | | 0.035 | 0.236 | **0.2360** | 7.8 % | **memory** |
| `attn_proj` | 51.94 | 890.6 | | 0.026 | 0.186 | **0.1855** | 6.1 % | **memory** |
| `hc_mix` | 2.17 | 676.3 | | 0.032 | 0.141 | 0.1409 | 4.6 % | memory |
| `attn_compress` | 19.46 | 645.4 | | 0.020 | 0.135 | 0.1345 | 4.4 % | memory |
| `collectives` (109) | 0 | 0 | 103.5 | | | 0.1150 | 3.8 % | communication |
| `attn_core` | 9.19 | 349.7 | | 0.005 | 0.073 | 0.0729 | 2.4 % | memory |
| `attn_index` | 4.27 | 207.8 | | 0.002 | 0.043 | 0.0433 | 1.4 % | memory |
| `moe_router` | 2.89 | 102.2 | | 0.043 | 0.021 | 0.0431 | 1.4 % | compute (fp32) |
| `lm_head` | 4.24 | 138.1 | | 0.004 | 0.029 | 0.0288 | 0.9 % | memory |
| `embed` | 0 | 0.5 | | | | 0.0001 | 0.0 % | memory |
| **Step floor** | **215.4** | **13,944** | **103.5** | | | **3.0418** | | **10,520 tok/s** |

### 4.1 Ranked top-3, with the arithmetic

**1. `moe_routed` — memory-bound, 2.042 ms, 67 % of the step.**
Every layer's batch of 32 tokens issues 192 expert assignments over 256
experts. Under uniform routing the expected union is 256 × (1 − (1 − 6/256)^32)
= **136.1 distinct experts**, each 13.369 MB (12.583 MB of fp4 payload +
0.786 MB of scales). Per rank at TP8, 136.1 × 13.369 MB / 8 = 227.5 MB per
layer; × 43 layers = **9.784 GB per step**, plus 16 MB of expert-side
activations; at 4.8 TB/s, 2.042 ms. FLOPs are 32 × 6 × 6 × 4096 × 2048 / 8 per
layer = 51.9 GFLOP per step, 0.026 ms at the fp8 peak: arithmetic intensity
5.3 against a ridge of 412. Nothing about batch 32 moves this off the memory
roof; §5 gives the bounds.

**2. `moe_shared` — memory-bound, 0.236 ms, 7.8 %.** ⚑
The shared expert is three fp8 matrices, 25.17 MB with scales per layer, and
the reference runs it **whole on every rank** (`model.py:632`): 43 × 25.17 MB
= 1.083 GB per rank per step. Sharded along the intermediate dimension, as a
TP engine would, the same region is 0.030 ms and drops to eighth place; the
step floor falls 6.8 %. This is the single largest engine-dependent term in
the step and the first thing a per-rank trace settles (§8).

**3. `attn_proj` — memory-bound, 0.186 ms, 6.1 %.**
Five fp8 projections per layer: `wq_a` (4.19 MB) and `wkv` (2.10 MB) are
replicated, `wq_b`, `wo_a`, `wo_b` (33.55 MB each) are sharded, so a rank
streams 4.19 + 2.10 + 3 × 4.19 = 18.9 MB of weights plus scales and
activations per layer, 890.6 MB per step. The grouped output projection is
8× what an even split of `o_lora_rank` across groups would give (§2); at
8.4 MB per rank per layer it is the largest single tensor in this region.

**What the ranking rests on.** Ranks 1 and 3 are checkpoint facts; rank 2 is a
reference-implementation fact that an engine can change. Two terms not in the
table can outrank rank 2 or 3 without touching any byte count: the 109
collectives per step priced at a realistic small-message latency rather than
at bandwidth (§9: 1.09 ms at 10 µs each), and kernel launch overhead if the
step is not CUDA-graph captured (§9: ~1,479 launches in the unfused reference,
3.0 ms at 2 µs). Both are Part 2's additive constant `c`, not per-region time.

### 4.2 Where the rest of the byte budget goes

`hc_mix` (0.141 ms) is the 4×-wide residual stream: each of the two mixers per
layer reads its 1.57 MB fp32 generator and three passes over a 32 × 16,384 fp32
state. `attn_compress` (0.135 ms) is the bf16 compressor projections, replicated,
run on every token (`model.py:329-330`) even though the pooled entry is written
only every 4th or 128th token (`:350`); the 21 indexer-side compressors are
included. `attn_core` (0.073 ms) is the KV read: 32 × (2 × 128 + 21 × 640 + 20 ×
192) entries × 583 B = 327 MB, replicated across ranks. `attn_index` (0.043 ms)
scans all 2,048 compressed keys per sequence on the 21 indexer layers and is
**the only region whose bytes grow with context**: linear in S/4.

### 4.3 Why this floor is not `gitm plan`'s 2.837 ms

At `main` the planner's graph for the same catalogue entry prices 46 layers (the
three `mtp.*` blocks are emitted with speculation off), a sharded shared expert
priced at fp4, an output projection 8x too small, one fp8 projection in place of
the bf16 compressors, a 576-wide KV entry, one all-reduce per layer with a bf16
payload, and fp8 for the fp32 mixers. Each of those is a reconciliation item with
its config line and code line in `RECONCILIATION.md` on this branch. Four are
corrected there (MTP gating, output projection, 512-wide head, shared expert
fp8), which moves `gitm plan` to **2.691 ms**; the two largest were opposite in
sign, which is why a total-only comparison would look better than the
per-region one. The 0.35 ms that remains is the replicated shared expert (0.20,
engine-dependent), the compressors (0.09), the collective count (0.07) and the
per-op dtypes (0.02-0.05), less the mHC launch floor this ledger does not price.

---

## 5. The expert bank

**Per token.** Six routed experts × 13.369 MB + one shared expert × 25.17 MB =
**105.38 MB of expert weights per token per layer**; × 43 layers = 4.53 GB per
token per step if nothing were shared between tokens.

**Per batch of 32, per layer, routed experts, model-wide** (per rank = ÷ 8,
whether the engine holds whole experts per rank as the reference does,
`model.py:623`, or slices every expert eight ways; the bytes are the same, the
collective and the load balance are not):

| Routing case | Distinct experts | Bytes per layer | Per rank per step (× 43) | Floor |
| --- | ---: | ---: | ---: | ---: |
| best case, every token picks the same six | 6 | 80.2 MB | 431 MB | 0.090 ms |
| **uniform routing, expectation** | **136.1** | **1.820 GB** | **9.784 GB** | **2.038 ms** |
| worst case, 192 assignments with no collision | 192 | 2.567 GB | 13.797 GB | 2.874 ms |
| whole bank touched | 256 | 3.423 GB | 18.396 GB | 3.833 ms |

**Assumptions, stated.**

- *Routing.* Uniform and independent across tokens for the 40 learned-router
  layers. `noaux_tc` with a selection bias exists to spread load, which is the
  argument that uniform is the right central value; any skew makes tokens
  collide on hot experts and *lowers* the distinct count, so the expectation
  errs toward more traffic, never toward an optimistic floor. The worst-case
  row is a hard upper bound at this batch. Layers 0-2 route by token id
  (`tid2eid`, `model.py:582`): their distinct count follows the token-id
  frequency distribution of the workload and is **not determinable from the
  checkpoint** (the table is in shard 2; the token histogram is not).
- *Reuse.* An expert's weights are read from HBM **once per touched expert per
  layer per step** (a grouped GEMM over the tokens that chose it), regardless
  of how many tokens did. No reuse across layers (each layer has its own 256
  experts) and none across steps (the per-rank bank is 18.4 GB against a
  50 MB L2).
- *The planner's term.* `roofline.distinct_experts(32, 256, 6)` = 136.149,
  the same formula; the script asserts equality. The planner also prices the
  fp4 payload and its 1×32 scales exactly (`roofline.py:41`). The dominant
  term of the step is one the repo already models correctly.

---

## 6. Overlap declarations (consumed by Part 2)

Region pairs that *can* run concurrently at the baseline, and by how much. The
reference implementation is single-stream, so its realised overlap is zero;
what is declared is the ceiling a multi-stream engine can claim, and Part 2's
serialization fraction `s` is the knob between the two.

```yaml
graph:
  floors_ms:
    moe_routed: 2.0418
    attn_proj: 0.1855
    attn_compress: 0.1345
    moe_shared: 0.2360
    attn_core: 0.0729
    attn_index: 0.0433
    hc_mix: 0.1409
    collectives: 0.1150
    lm_head: 0.0288
    moe_router: 0.0431
    embed: 0.0001
  overlaps:
    - {earlier: moe_routed, later: moe_shared, overlap_ms: 0.2360}
    - {earlier: attn_index, later: attn_compress, overlap_ms: 0.0433}
```

- **`moe_routed` ∥ `moe_shared`.** Both consume the same normalised input and
  meet only at the final add (`model.py:641-648`: `y` is reduced, *then*
  `y += shared_experts(x)`). The shared expert can be fully hidden under the
  routed grouped GEMM; the overlap is the whole of `moe_shared`'s floor.
- **`attn_index` ∥ `attn_compress`.** On the 21 indexer layers the main
  compressor's projections do not consume the index (`model.py:516-528`: the
  indexer runs, then `self.compressor(x, start_pos)` on the same `x`). The
  indexer's score scan and its all-reduce can be hidden under those
  projections; the overlap is the smaller of the two, `attn_index`'s floor.

Not declared, and why: the two all-reduces per layer sit on the critical path
(the hyper-connection mix needs the reduced output); `attn_core` needs the
query and the selection; `lm_head` follows the last layer. Any overlap an
engine finds among those would be a scheduling discovery, not a property of
the graph, and Part 2 must not be told it exists.

---

## 7. What the checkpoint establishes, and what depends on the engine

| Established by the checkpoint | Evidence |
| --- | --- |
| Every shape, dtype and byte count in §3.1 and §3.2 | shard headers; byte sum = `total_size` |
| The layer schedule 2 / 21 / 20 and which layers carry an indexer | `compress_ratios`, `model.py:472-477`, tensor presence |
| One shared 512-wide KV entry per token per layer, 64 RoPE dims inside it, unsplittable under TP | `model.py:466`, `:480`, `:505-512`, `:533` |
| Entries read per step per kind (128 / 640 / 192) and keys scanned (2,048) | `model.py:426`, `:433`, `:513`, `:519-520` |
| Expert bytes: fp4 payload + 1×32 scales; shared expert fp8; top-6 of 256 on all 43 layers | `model.py:18`, `:138-142`, `:628-632` |
| Which tensors shard and which replicate *in the reference* | `Linear` vs `ColumnParallelLinear` / `RowParallelLinear`, §3 |
| Three `mtp.*` blocks exist and do not run without speculation | index; `model.py:900-902`, `:912-936` |

| ⚑ Depends on the engine | Both readings | Settled by |
| --- | --- | --- |
| KV storage dtype | fp8 1.105 GB / bf16 1.984 GB per rank; `attn_core` +0.052 ms and `attn_index` +0.037 ms at bf16 | `--kv-cache-dtype` or equivalent; `dram__bytes_read` on the attention kernel |
| Shared expert replicated or sharded | 0.236 / 0.030 ms; 1.082 GB / 135 MB per rank | per-rank kernel shape of the shared-expert GEMM |
| Compressors, router, `lm_head` kept fp32 in HBM as the reference does | +0.185 ms if fp32 | resident dtype of those tensors in the engine |
| Whole experts per rank + all-reduce (reference) vs sliced experts (TP) vs EP all-to-all | same bytes per rank; different collective, different balance | NCCL kernel names and count per step |
| Collective payload dtype and count | fp32 vs bf16 (2×); 109 per step in the reference | NCCL count per step |
| `mtp.*` resident | 0 / 1.499 GB per rank | `nvidia-smi` after load |
| Compressor state buffers | 0 / 390.6 MB per rank | engine source |
| Kernel fusion and launch count | 636 (planner at `main`, fused) to ~1,479 (reference, unfused) per step | `nsys` launch count with `--cuda-graph-trace=node` |
| Prefill chunking, CUDA-graph capture, scheduler gaps | not modelled here | launch arguments as text |

---

## 8. Not determinable from the checkpoint

Each with the single observation that would settle it.

1. **The realised distinct-expert count**, especially on the three hash-routed
   layers. Needs a token histogram of the workload, or a per-layer count of
   touched experts from a trace.
2. **Whether the shared expert is replicated** (0.236 ms) or sharded (0.030 ms).
   One per-rank kernel shape.
3. **KV storage dtype** and therefore the KV ledger line and the `attn_core`
   floor. The engine's launch arguments.
4. **The collective latency floor.** The bandwidth floor is 0.115 ms; at 10 µs
   per small-message ring reduction the same 109 collectives are 1.09 ms and
   the second-largest term in the step. One NCCL kernel duration from a trace.
5. **Launch overhead.** Whether the step is CUDA-graph captured decides whether
   ~1.5 k launches cost 3 ms or nothing visible. A launch count per step.
6. **Whether `mtp.*` occupies 1.5 GB per rank** while speculation is off.
   `nvidia-smi` after load.
7. **The workspace, communication-buffer and context reserve lines** (1.75 GB
   assumed). `torch.cuda.memory_reserved()` on the served engine.
8. **How many kernels a layer lowers to**, which decides both the launch term
   and how many regions a capture can pair. A kernel-name listing per step.

---

## 9. Sensitivities

| Change | Effect on the table in §4 |
| --- | --- |
| shared expert TP-sharded (engine) | `moe_shared` 0.236 → 0.030 ms; step −6.8 % |
| compressors + router + `lm_head` resident in fp32, as the reference keeps them | +0.185 ms |
| bf16 KV store (reference) | `attn_core` +0.052 ms, `attn_index` +0.037 ms |
| collectives at latency instead of bandwidth | 109 per step: 0.115 ms (bandwidth) / 0.218 ms (2 µs, the planner's launch constant) / **1.090 ms (10 µs)** |
| kernel launches without graph capture | ~1,479 per step: 2.96 ms at 2 µs, 7.40 ms at 5 µs — comparable to or larger than the 3.04 ms memory floor |
| routing skew | worst-case union +0.836 ms on `moe_routed`; any skew below uniform subtracts |
| context length S | only `attn_index` (keys scanned ∝ S/4) and the 20 dense-compressed layers' entries (∝ S/128) grow; every other region is flat |

The launch and collective-latency rows are the reason the step is described as
*memory-bound at the floor* rather than *memory-bound*. At B=32 the bytes say
3.0 ms; an unfused, uncaptured, latency-bound implementation of the same graph
can spend more time between kernels than in them, and none of that shows up as
per-region bytes. Part 2 is built to tell those apart.
