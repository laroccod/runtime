# Attribution over the DeepSeek V4 Flash decode graph (case Part 2)

Baseline operating point, restated: one 8xH200 SXM node over NVLink, TP8, FP8
weights and KV where the checkpoint supports it, 32 active sequences in decode,
8,192 cached tokens each, one new token per sequence per step, text-only,
speculation disabled. The graph is Part 1's (`DESIGN-NOTE.md` §4 and §6): eleven
regions with floors `f_r` and two declared overlap pairs, restated as data in
every `params.yaml` under `tests/fixtures/attribution/`.

Code: `gitm/attribution/{model,simulate,attribute,battery}.py`; tests:
`tests/test_attribution.py`. Contracts, verbatim (also `python -m gitm.attribution.simulate` / `.attribute`):

```
simulate  --params params.yaml --seed N --steps M --out data.jsonl
attribute --data data.jsonl --out report.json
```

## 1. Generative model (2a)

Regions `r` with floors `f_r > 0`; declared pairs `p = (e, l, o_p)`: the *later*
region `l` may run concurrently with the last `o_p` ms of the *earlier* region `e`,
`0 <= o_p <= min(f_e, f_l)`. Step `t` carries a regime covariate `q_t`.

Mechanisms 1 and 4 are per-region multipliers:

    m_{r,t} = alpha_r * gamma^{[r in G][q_t > tau]},        alpha_r >= 1, alpha_r = 1 off S.   (1)

**Overlap after slowdown is derived, not assumed.** In the two-node DAG where `l`
depends on the first `f_e - o_p` ms of `e`, `l` starts when that prefix ends, so the
concurrent window is the earlier region's tail, capped by the whole later region:

    omega_p = min(o_p * m_e,  f_l * m_l).                                             (2)

`omega_p` is homogeneous of degree one in the multipliers. (A1) Each pair is its own
sub-DAG; chained pairs sum. (A2) `o_p` is a piece of `e`'s tail and stretches with
`e`; reading it as a fixed number of ms would make a clock throttle *shrink* the
overlapped fraction, which a throttle does not do.

Mechanism 2 returns `s * omega_p` to the total and to the later region; mechanism 3
adds `c >= 0` to the total only:

    v_{r,t} = f_r m_{r,t} + s * sum_{p: later(p)=r} omega_{p,t}                        (3)
    T*_t    = sum_r f_r m_{r,t} - sum_p omega_{p,t} + s * sum_p omega_{p,t} + c        (4)
            = critical path of slowed/gated floors + returned overlap + c.

Mechanism 5 and noise:

    y_{r,t} = (1+beta) v_{r,t} eps_{r,t},        T_t = (1+beta) T*_t eta_t,             (5)

`eps`, `eta` independent log-normal, mean 1, constant CV (`noise: {cv_region: 0.03,
cv_total: 0.01}` by default, `cv_region` may be per region). *Why:* positive support
and constant CV are how kernel durations jitter; independence across regions,
steps, and between regions and total makes the joint law a product whose only
free quantities are the means (3)-(4), so identifiability is exactly the rank of
the mean map and the estimator cannot lean on a cross-covariance a real trace may
lack. Cost, stated: the recorded total does not inherit the regions' jitter.
(A3) `q_t` is exogenous, drawn from `noise.q` (default uniform on `[0, 2 tau]`
when a gate is declared, else `[0, 32]`). (A4) `beta, c >= 0`, `s in [0, 1]`.
(A5) Floors are the known Part 1 values; row 1 of `data.jsonl` carries them.
Arms: `conditions[].overrides` deep-merge onto the base (dicts merge, lists and
scalars replace). `instrumentation_level` is an ordinal label per arm (0 for
`beta = 0`, then rank of the distinct non-zero betas); the estimator uses it only
to group arms and does *not* assume level 0 is undistorted (§3).

## 2. Two non-identifiability results and their separating interventions (2b)

Write `a_r := (1+beta) alpha_r` and `c~ := (1+beta) c`.

### 2.1 Mechanism 5 ≡ a uniform mechanism 1

**Proposition 1.** The law of the data under `(beta, {alpha_r}, s, c, G, gamma, tau)`
equals the law under `(0, {(1+beta) alpha_r}, s, (1+beta) c, G, gamma, tau)`.

*Proof.* Put `m~ = (1+beta) m`. By (2), `omega_p(m~) = (1+beta) omega_p(m)`. By (3),
`(1+beta) v_r(m) = f_r m~_r + s sum omega_p(m~) = v_r(m~)`; by (4),
`(1+beta) T*(m, c) = T*(m~, (1+beta) c)`. The noise factors in (5) carry no
parameter, so the laws coincide step by step; `(1+beta) alpha_r >= 1` is feasible. QED.

*Conditions.* As "mechanism 5 alone ≡ uniform mechanism 1 alone": `c = 0` (else `c`
is rescaled too), `S` = all regions with one common `alpha`; the gate commutes. Under
the fixed-ms overlap convention rejected in (A2) the totals would differ by
`beta (sum omega - c)`, an artefact of that convention, not a handle. *Corner:*
`alpha_r >= 1`, `beta >= 0` make the boundary one-sided: a region at its floor,
`a_r = 1`, forces `beta = 0` and identifies every other `alpha`. A non-uniform
slowdown with `beta = 0` is therefore identified from passive data; a uniform
one, or any slowdown with `beta > 0`, is not (`pair1-passive.yaml` vs
`pair1-uniform.yaml` give the same report, §4).

**Intervention I: vary the instrumentation.** *Fixed:* workload, `q` generator,
every execution parameter. *Varied:* the instrumentation configuration, giving
arms at levels `k, k'`. *Observable:* `rho_r = mu_r^{(k')}/mu_r^{(k)}` for every
region and for the total. *Rule:* under the model all `rho_r` equal
`(1+beta_{k'})/(1+beta_k)`; a non-uniform `rho_r` means execution moved too and the
arm is rejected (warning). The ratio is identified from any two levels; absolute
values follow from a level asserted undistorted, or from the per-level corner
rule `beta_k <= min_{r not later} a_r^{(k)} - 1` when that bound is within
tolerance of 0. Encoded as `overrides: {observation: {beta: 0.0}}`
(`pair1-intervention.yaml`).

### 2.2 Serialization ≡ slowdown of the later regions

**Proposition 2.** Fix one instrumentation level and all parameters except `s` and
`{alpha_l : l in L}` (`L` = later regions). Assume each earlier region is not itself
later, and no pair is fully hidden (`o_p m_e < f_l m_l`, so `omega_p = o_p m_e`).
Then the mean map has a one-dimensional kernel spanned by

    delta alpha_l = - omega_{p(l)} / f_l  for every l in L,      delta s = +1.

*Proof (two lines).* From (3)-(4): for `l in L`, `d mu_l/d alpha_l = (1+beta) f_l` and
`d mu_l/d s = (1+beta) omega_p`; for `r not in L` both vanish; for the total
`d T/d alpha_l = (1+beta) f_l`, `d T/d s = (1+beta) sum_p omega_p`; and
`D := sum_r mu_r - T = (1+beta)(sum_p omega_p - c)` has zero derivative in both. Row
by row, `J[(mu_L, T, D, mu_{not L}); (alpha_L, s)] = [f_l delta_{ll'} | omega_p]`,
`[f_l ... | sum omega]`, `0`, `0`, and the stated vector is in its kernel:
rank `J = |L| < |L| + 1`. The law is determined by the means (§1), so a flat
direction of the means is a flat direction of the law. QED.

*Reading.* "The shared expert got slower by `s * 0.236` ms" and "a fraction `s` of
its overlap with the routed experts was serialized" are the same data. The
direction is always feasible towards `s = 0`, so passive data can never establish
serialization; it can only rule it out at the corner `mu_l = f_l (1+beta)`. In the
fully hidden regime (`o_p m_e >= f_l m_l`; the routed/shared pair at the baseline,
where `o_p = f_l`) the degeneracy persists and also involves `c` via `omega_p = f_l m_l`.

**Intervention II: perturb the earlier region only.** *Fixed:* instrumentation
level, `s`, `c`, the gate, `alpha` on every region but one. *Varied:* the earlier
region `e` of a pair whose later region is not fully hidden, by a perturbation
known to be local to `e` (a fixed delay injected into `e`'s stream, or more work
for `e` alone). *Observable:* the mean shifts `(Delta mu_e, Delta mu_l)` and the
invariance of every other region (the footprint check). *Rule:* the model gives
`Delta mu_l = s (o_p/f_e) Delta mu_e`; `s` is the least-squares solution over
`s in [0, 1]` of the later region's response with `alpha_l` from the reference arm
(exclusion restriction: the perturbation did not touch `l`); `absent` if the
interval contains 0 within tolerance. When the pair is fully hidden in both arms
the later region cannot be hidden more than its whole length, the response is
flat, and the estimator reports the arm uninformative. For this graph the
informative pair is `attn_index -> attn_compress` (`o = 0.0433 < f_l = 0.1345`),
the moe pair is not (`pair2-intervention.yaml`, `pair2-hidden-pair-uninformative.yaml`).
Encoded as `overrides: {slowdown: {alpha: {attn_index: 1.6}}}`. *Why not force
`s -> 1`?* Its footprint in `data.jsonl` is "only later regions moved", exactly the
flat direction; without being told the design the estimator cannot tell it from a
slowdown of `l`. Intervention II is self-identifying from the data.

## 3. Estimator decision rule (2c)

Nothing below uses `alpha_r` or `c`: every quantity is `a_r` or `c~` until `beta` is
resolved, and if it is not, the report names the products that were identified.
This is how mechanism 5 is carried through.

1. **Gate first.** For 60 candidate `tau` (quantiles of `q`, then an exact local
   scan) and every region, the within-arm log-ratio of means above/below `tau` and
   its delta-method z, pooled across arms (the ratio cancels `beta` and `alpha`).
   The scan statistic is the largest `|z|`; its null comes from permuting `q`
   within arms (200 draws). Detected only above the 99th percentile of that null
   (look-elsewhere controlled) and beyond `tol_gamma`. Members: `|z| > 3` at the
   selected `tau`; a member that is a later region whose earlier partner is also a
   member is flagged and excluded from `gamma`. `tau` is reported as the gap
   between the largest ungated and smallest gated `q`. Everything after uses
   ungated steps.
2. **Per-arm moments** on ungated steps: `mu_r`, their standard errors,
   `D = mean(sum_r y_r - T)`.
3. **`beta`.** Ratios between instrumentation levels (median over regions,
   dispersion checked); then the per-level corner rule of §2.1, each `a_r` taken at
   its upper limit before the minimum (a minimum of noisy estimates is biased low).
   Otherwise `beta in [0, beta_max]`.
4. **`s`.** From an arm carrying intervention II's footprint (checked: only an
   identified earlier region and its later partners moved); else the bound
   `s in [0, s_max]`, `s_max` the largest `s` keeping every `alpha_l >= 1`. No
   declared overlap: `absent`, nothing to act on.
5. **`a_l`, `c`.** `a_l` solved from (3) at the resolved `s`, or at both ends of its
   bound; `c~ = sum_p omega_p - D`, converted with the resolved `beta` or bounded.
6. **Intervals** by a step-level bootstrap within arms (200 replicates), gate and
   intervention design held at the point estimate.

**Statuses.** Tolerances separating `absent` from small (`TOL` in `attribute.py`):
`alpha - 1` 0.01, `beta` 0.01, `s` 0.02, `c` 0.01 ms, `gamma - 1` 0.01. With a point
estimate and interval: `estimated` if the whole interval lies beyond the band;
`absent` if the point lies inside it, or on the side the mechanism forbids
(`alpha >= 1`, `beta, s, c >= 0`), the interval being the upper limit; else
`not_identifiable_from_this_data`. For an identified *set* (a flat direction) the
whole set must sit inside the band to be `absent`; otherwise the report gives the
bounds, the identified combination, and the separating intervention. A gate with
`tau` outside the observed `q` range is `absent` with that caveat; constant `q` is
`not_identifiable`.

## 4. Battery (`python -m gitm.attribution.battery`; asserted in `tests/test_attribution.py`)

Recovery over 8 seeds on `full-battery.yaml` (all five on; arms passive 40%,
`beta = 0` 30%, `attn_index x1.6` 30%); pull = (estimate − truth)/(half-width/1.96):

| steps | mechanism | truth | bias | spread (sd) | mean 95% half-width | pull mean | pull sd |
|---|---|---|---|---|---|---|---|
| 600 | observation `beta` | 0.15 | -0.0002 | 0.0016 | 0.0038 | -0.08 | 0.85 |
| 600 | slowdown `alpha_moe_routed` | 1.4 | +0.0016 | 0.0044 | 0.0081 | +0.36 | 0.96 |
| 600 | serialization `s` | 0.5 | -0.0197 | 0.0315 | 0.0513 | -0.74 | 1.13 |
| 600 | additive `c` (ms) | 0.8 | -0.0004 | 0.0105 | 0.0191 | -0.05 | 1.15 |
| 600 | regime_gate `gamma` | 1.3 | -0.0019 | 0.0033 | 0.0058 | -0.70 | 1.14 |
| 3000 | observation `beta` | 0.15 | +0.0001 | 0.0009 | 0.0017 | +0.16 | 1.01 |
| 3000 | slowdown `alpha_moe_routed` | 1.4 | +0.0008 | 0.0015 | 0.0035 | +0.51 | 0.94 |
| 3000 | serialization `s` | 0.5 | +0.0056 | 0.0174 | 0.0237 | +0.51 | 1.49 |
| 3000 | additive `c` (ms) | 0.8 | -0.0026 | 0.0068 | 0.0081 | -0.77 | 1.88 |
| 3000 | regime_gate `gamma` | 1.3 | +0.0002 | 0.0010 | 0.0024 | +0.15 | 0.83 |

Biases sit inside one half-width and shrink with sample size; pull spreads are
near 1. The `additive` intervals are optimistic at 3000 steps because `c` inherits
the error of `s` through the fully hidden moe pair (§2.2), a known limit.

Statuses per fixture (seed 0, 1200 steps; `NI` = `not_identifiable_from_this_data`):

| fixture | slowdown | serialization | additive | regime_gate | observation |
|---|---|---|---|---|---|
| pair1-passive | NI | NI | absent | absent | NI |
| pair1-uniform (the exact copy) | NI | NI | absent | absent | NI |
| pair1-intervention | estimated | absent | absent | absent | estimated |
| pair2-passive | NI | NI (bound `[0, 0.51]`) | absent | absent | absent |
| pair2-intervention | absent | estimated | absent | absent | absent |
| pair2-hidden-pair-uninformative | NI | NI | absent | absent | absent |
| inert (every block present at its inert value, with noise) | absent | absent | absent | absent | absent |
| gate-out-of-range | absent | absent | absent | absent | absent |
| single-region | absent | absent | estimated | absent | absent |

In `pair1-passive` serialization is unresolved too, correctly: with `beta`
unresolved the later regions' 15% excess over their floors could be returned
overlap. In `pair2-passive` observation is `absent` by the corner rule (non-later
regions sit at their floors) and the `s` bound contains the truth.

## 5. Limits

Everything rests on the floors (A5): a wrong floor reads as a slowdown, or as a
floor violation the report flags. Chained pairs are not resolved. The gate is one
threshold with one `gamma` per region; a smooth dependence on `q` would be found
as a gate at the best split. `s` is one global fraction as the case defines it; a
per-pair `s` would appear as inconsistency across pairs, which is not tested. The
instrumentation label is never read as a value of `beta`; the "if level 0 is
undistorted" reading is reported only as a labelled conditional.
