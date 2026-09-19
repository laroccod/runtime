"""``attribute --data data.jsonl --out report.json``.

Works from ``data.jsonl`` alone. Per mechanism it reports a status in
``{estimated, not_identifiable_from_this_data, absent}``, a point estimate and
a bootstrap interval when ``estimated``, the tolerance that separates
``absent`` from small, and, whenever it abstains, the flat direction it hit
and the intervention that would break it.

Decision rule (note §3). Every recorded time carries the factor (1 + beta) of
mechanism 5, so the estimator never works with alpha_r directly; it works with
the *identified* products ``a_r = (1 + beta) alpha_r`` and ``c~ = (1 + beta) c``
and resolves beta only from the corner argument ``min_r a_r = 1  =>  beta = 0``
(because alpha_r >= 1 and beta >= 0) at some instrumentation level, carried to
the other levels by the between-level ratios of means. The label
``instrumentation_level 0`` is never read as ``beta = 0``.
Serialization ``s`` is resolved only from an arm in which an *earlier* region
of a declared pair was perturbed while everything else held (the footprint is
checked); otherwise ``s`` gets the bound ``[0, s_max]`` from ``alpha_l >= 1``.
Recorded time is never treated as ground truth anywhere in this file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gitm.attribution.model import Graph, forward

REPORT_SCHEMA = "gitm-attribution-report/1"
STATUS_EST, STATUS_NI, STATUS_ABS = "estimated", "not_identifiable_from_this_data", "absent"

#: Tolerances separating ``absent`` from small (note §3). Multiplicative
#: parameters are distances from 1, ``s`` and ``beta`` from 0, ``c`` in ms.
TOL: dict[str, float] = {"alpha": 0.01, "s": 0.02, "c_ms": 0.01, "gamma": 0.01, "beta": 0.01}
Z_MEMBER = 3.0  # per-region membership cut at the selected tau (about 11 regions, 99%)
MIN_SIDE = 8  # steps needed on each side of a candidate tau, per arm
S_GRID = np.linspace(0.0, 1.0, 401)
Z_BOUND = 2.7  # per-region upper limit used inside a min over ~11 regions (Bonferroni-style)


@dataclass
class Arm:
    name: str
    level: int
    q: np.ndarray  # (n,)
    Y: np.ndarray  # (n, R) recorded region times
    T: np.ndarray  # (n,) recorded totals


@dataclass
class Moments:
    mu: np.ndarray  # (R,) mean recorded region time
    se: np.ndarray  # (R,) standard error of mu
    Dbar: float  # mean of sum_r y_r - T


# --------------------------------------------------------------------------- I/O


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows or "graph" not in rows[0]:
        raise ValueError(f"{path}: first row must be the metadata row carrying 'graph'")
    return rows


def arms_from_rows(rows: list[dict[str, Any]], graph: Graph) -> list[Arm]:
    """Group step rows by condition.

    Order follows the metadata row's ``conditions`` list when present (the
    first listed condition is the reference arm, conventionally the passive
    one), else order of first appearance.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for name in rows[0].get("conditions") or []:
        buckets[str(name)] = []
    for row in rows[1:]:
        buckets.setdefault(str(row["condition"]), []).append(row)
    buckets = {k: v for k, v in buckets.items() if v}
    arms = []
    for name, steps in buckets.items():
        levels = {int(s["instrumentation_level"]) for s in steps}
        if len(levels) != 1:
            raise ValueError(f"condition {name!r} mixes instrumentation levels {sorted(levels)}")
        Y = np.array([[float(s["regions"][r]) for r in graph.regions] for s in steps])
        arms.append(
            Arm(
                name=name,
                level=levels.pop(),
                q=np.array([float(s["q"]) for s in steps]),
                Y=Y,
                T=np.array([float(s["total_ms"]) for s in steps]),
            )
        )
    return arms


# --------------------------------------------------------------------------- gate scan


def _split_stats(q: np.ndarray, Y: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-ratio of means (q > tau vs q <= tau) and its z-score, per (tau, region).

    Sorting once and using cumulative sums makes each threshold O(R); the
    variance of the log-ratio of two means is the delta-method expression.
    """
    order = np.argsort(q, kind="stable")
    qs, Ys = q[order], Y[order]
    n = len(qs)
    cs = np.vstack([np.zeros(Ys.shape[1]), np.cumsum(Ys, axis=0)])
    cs2 = np.vstack([np.zeros(Ys.shape[1]), np.cumsum(Ys * Ys, axis=0)])
    n_lo = np.searchsorted(qs, cands, side="right")
    logratio = np.zeros((len(cands), Ys.shape[1]))
    z = np.zeros_like(logratio)
    for i, nl in enumerate(n_lo):
        nh = n - nl
        if nl < MIN_SIDE or nh < MIN_SIDE:
            continue
        m_lo, m_hi = cs[nl] / nl, (cs[n] - cs[nl]) / nh
        v_lo = np.maximum(cs2[nl] / nl - m_lo**2, 0.0) * nl / max(nl - 1, 1)
        v_hi = np.maximum((cs2[n] - cs2[nl]) / nh - m_hi**2, 0.0) * nh / max(nh - 1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(m_hi / m_lo)
            var = v_hi / (nh * m_hi**2) + v_lo / (nl * m_lo**2)
            zz = np.where(var > 0, lr / np.sqrt(var), 0.0)
        logratio[i], z[i] = lr, zz
    return logratio, z


def _pooled_scan(arms: list[Arm], cands: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stouffer-pooled z over arms (within-arm ratios cancel beta and alpha)."""
    R = arms[0].Y.shape[1]
    zsum = np.zeros((len(cands), R))
    lrsum = np.zeros_like(zsum)
    wsum = np.zeros_like(zsum)  # 1/var weights for the pooled log-ratio
    for arm in arms:
        lr, z = _split_stats(arm.q, arm.Y, cands)
        valid = z != 0
        zsum += z
        w = np.where(valid & (lr != 0), (z / np.where(lr == 0, 1, lr)) ** 2, 0.0)  # 1/var
        lrsum += w * lr
        wsum += w
    with np.errstate(invalid="ignore", divide="ignore"):
        zpool = zsum / math.sqrt(len(arms))
        lrpool = np.where(wsum > 0, lrsum / np.maximum(wsum, 1e-300), 0.0)
    return zpool, lrpool


def detect_gate(
    arms: list[Arm], graph: Graph, rng: np.random.Generator, n_perm: int
) -> dict[str, Any]:
    q_all = np.concatenate([a.q for a in arms])
    out: dict[str, Any] = {"detected": False, "tau": None, "members": [], "q_range": [float(q_all.min()), float(q_all.max())]}
    if np.ptp(q_all) <= 1e-12:
        out["reason"] = "q is constant: a gate is either always on or always off and cannot be told from slowdown"
        out["degenerate"] = True
        return out
    cands = np.unique(np.quantile(q_all, np.linspace(0.02, 0.98, 60)))
    zpool, lrpool = _pooled_scan(arms, cands)
    Z = np.abs(zpool)
    i_best, r_best = np.unravel_index(int(np.argmax(Z)), Z.shape)
    Z_obs = float(Z[i_best, r_best])

    null = np.empty(n_perm)
    for b in range(n_perm):
        shuffled = [Arm(a.name, a.level, rng.permutation(a.q), a.Y, a.T) for a in arms]
        zp, _ = _pooled_scan(shuffled, cands)
        null[b] = float(np.abs(zp).max())
    thr = float(np.quantile(null, 0.99)) if n_perm else Z_MEMBER
    out.update({"scan_statistic": Z_obs, "threshold_99": thr, "candidates": int(len(cands))})

    # refine tau: exact scan over every distinct q between the neighbouring candidates
    lo_c = cands[max(i_best - 1, 0)]
    hi_c = cands[min(i_best + 1, len(cands) - 1)]
    fine = np.unique(q_all[(q_all >= lo_c) & (q_all <= hi_c)])
    if len(fine) > 1:
        zf, lrf = _pooled_scan(arms, fine)
        j_best = int(np.argmax(np.abs(zf[:, r_best])))
        tau, z_at, lr_at = float(fine[j_best]), zf[j_best], lrf[j_best]
        Z_obs = max(Z_obs, float(abs(z_at[r_best])))
    else:
        tau, z_at, lr_at = float(cands[i_best]), zpool[i_best], lrpool[i_best]
    members = [graph.regions[j] for j in range(len(graph.regions)) if abs(z_at[j]) > Z_MEMBER]
    # effect size at the selected tau, and the smallest effect the scan could have found
    se_lr = np.median(
        [abs(lr_at[j] / z_at[j]) for j in range(len(graph.regions)) if z_at[j] != 0] or [float("nan")]
    )
    out["min_detectable_gamma_minus_1"] = float(math.expm1(thr * se_lr)) if se_lr == se_lr else None
    detected = Z_obs > thr and abs(lr_at[r_best]) > TOL["gamma"]
    if not detected:
        return out
    below = q_all[q_all <= tau]
    above = q_all[q_all > tau]
    out.update(
        {
            "detected": True,
            "tau": tau,
            "tau_interval": [float(below.max()), float(above.min())],
            "members": members,
            "ratio": {graph.regions[j]: float(math.exp(lr_at[j])) for j in range(len(graph.regions))},
            "z": {graph.regions[j]: float(z_at[j]) for j in range(len(graph.regions))},
        }
    )
    return out


# --------------------------------------------------------------------------- moments


def moments(arm: Arm, mask: np.ndarray) -> Moments:
    Y, T = arm.Y[mask], arm.T[mask]
    n = int(mask.sum())
    mu = Y.mean(axis=0)
    se = Y.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.full(Y.shape[1], np.nan)
    return Moments(mu=mu, se=se, Dbar=float((Y.sum(axis=1) - T).mean()))


# --------------------------------------------------------------------------- core estimator


def _later_a(graph: Graph, il: int, s: float, a: np.ndarray, mu_l: float) -> float:
    """Solve mu_l = f_l a_l + s * sum_p min(o_p a_e, f_l a_l) for a_l (monotone; bisection)."""
    f = graph.floors
    idx = graph.index
    pairs = [(idx[o.earlier], o.overlap_ms) for o in graph.overlaps if idx[o.later] == il]

    def g(a_l: float) -> float:
        return f[il] * a_l + s * sum(min(o * a[ie], f[il] * a_l) for ie, o in pairs) - mu_l

    lo, hi = 0.0, mu_l / f[il] + 1e-9
    if g(hi) < 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _a_vector(graph: Graph, mom: Moments, s: float) -> np.ndarray:
    """Identified products a_r = mu_r / f_r for non-later regions; solved for later regions."""
    a = mom.mu / graph.floors
    later = graph.later_regions
    idx = graph.index
    for r in graph.regions:
        if r not in later:
            a[idx[r]] = mom.mu[idx[r]] / graph.floors[idx[r]]
    for r in later:  # earlier partners that are themselves later regions are unresolved (chain)
        a[idx[r]] = _later_a(graph, idx[r], s, a, mom.mu[idx[r]])
    return a


def _predict_mu(graph: Graph, a: np.ndarray, s: float) -> np.ndarray:
    v, _, _ = forward(graph, a, s, 0.0)
    return v


def _s_max(graph: Graph, mom: Moments, a_min: float, z: float = Z_BOUND) -> float:
    """Largest s consistent with alpha_l >= 1 (i.e. a_l >= a_min) for every later region.

    A minimum over noisy per-region values is biased low, so each later
    region's mean is first raised to its upper limit (``z`` standard errors).
    """
    later = [graph.index[r] for r in graph.later_regions]
    if not later:
        return 0.0
    mu = mom.mu.copy()
    mu[later] += z * np.nan_to_num(mom.se[later])
    mom = Moments(mu=mu, se=mom.se, Dbar=mom.Dbar)
    s_hi = 1.0
    for il in later:
        if _a_vector(graph, mom, 0.0)[il] < a_min:
            return 0.0  # later region below its floor: nothing to return to serialization
        lo, hi = 0.0, 1.0
        if _a_vector(graph, mom, 1.0)[il] >= a_min:
            continue
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _a_vector(graph, mom, mid)[il] >= a_min:
                lo = mid
            else:
                hi = mid
        s_hi = min(s_hi, lo)
    return s_hi


def _chained(graph: Graph) -> set[str]:
    return graph.later_regions & graph.earlier_regions


def _find_interventions(
    graph: Graph, moms: list[Moments], levels: list[int], ref: int, usable_pairs: list
) -> list[tuple[int, list[str], list[str]]]:
    """Arms at the reference level whose footprint is 'one or more earlier regions moved,
    and nothing else but their later partners'. Returns (arm index, E, affected later)."""
    R = len(graph.regions)
    later = graph.later_regions
    found = []
    for k, m in enumerate(moms):
        if k == ref or levels[k] != levels[ref]:
            continue
        diff = m.mu - moms[ref].mu
        se = np.sqrt(m.se**2 + moms[ref].se**2)
        moved = {
            graph.regions[i]
            for i in range(R)
            if abs(diff[i]) > 3.0 * se[i] and abs(diff[i]) / moms[ref].mu[i] > TOL["alpha"]
        }
        E = ({o.earlier for o in usable_pairs} & moved) - later
        allowed = E | {o.later for o in graph.overlaps if o.earlier in E}
        if not E or not moved <= allowed:
            continue
        found.append((k, sorted(E), sorted({o.later for o in graph.overlaps if o.earlier in E})))
    return found


def core(
    graph: Graph, moms: list[Moments], levels: list[int], ref: int, design: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Point estimates and identified bounds from per-arm moments (bootstrapped as a unit).

    ``design`` carries the intervention footprints selected at the point
    estimate so that bootstrap replicates condition on the same design rather
    than re-selecting it under resampling noise.
    """
    idx = graph.index
    f = graph.floors
    out: dict[str, Any] = {"warnings": []}
    later = graph.later_regions
    N = [i for i, r in enumerate(graph.regions) if r not in later]

    # ---- instrumentation ratios between levels (all regions scale by (1+beta), so use all)
    level_ref = levels[ref]
    rho: dict[int, float] = {level_ref: 1.0}
    disp: dict[int, float] = {}
    for lvl in sorted(set(levels) - {level_ref}):
        best = None
        for k, m in enumerate(moms):
            if levels[k] != lvl:
                continue
            lr = np.log(m.mu / moms[ref].mu)
            d = float(np.max(np.abs(lr - np.median(lr))))
            if best is None or d < best[0]:
                best = (d, float(np.exp(np.median(lr))))
        assert best is not None
        disp[lvl], rho[lvl] = best
        if best[0] > 0.05:
            out["warnings"].append(
                f"arms at instrumentation levels {level_ref} and {lvl} differ non-uniformly across "
                f"regions (max |log ratio deviation| = {best[0]:.3f}); they may differ in execution too"
            )
    out["rho_by_level"] = rho

    # ---- a-vectors per arm (s = 0 seed; later regions refined once s is known)
    a_ref0 = _a_vector(graph, moms[ref], 0.0)
    floor_violation = bool(np.any(a_ref0[N] < 1.0 - TOL["alpha"])) if N else False
    if floor_violation:
        out["warnings"].append("a region runs below its floor in the reference arm; the alpha >= 1 corner rule is not applied")

    # ---- beta. Per level, beta_lvl <= min_r a_r^(lvl) - 1 over non-later regions (alpha_r >= 1);
    # a minimum over noisy estimates is biased low, so each a_r is taken at its upper limit.
    # If some level's bound is within tolerance of 0 that level is undistorted and every
    # other level follows from the ratios. The label "level 0" is NOT assumed to mean
    # beta = 0; that reading is reported separately as a conditional estimate.
    bound_by_level: dict[int, float] = {}
    for k, m in enumerate(moms):
        if not N or floor_violation:
            continue
        a_up = _a_vector(graph, m, 0.0) + Z_BOUND * np.nan_to_num(m.se) / f
        b = float(a_up[N].min() - 1.0)
        bound_by_level[levels[k]] = min(bound_by_level.get(levels[k], math.inf), b)
    beta_by_level: dict[int, float] | None = None
    beta_max: float | None = None
    if bound_by_level:
        anchor = min(bound_by_level, key=bound_by_level.get)
        beta_max = (1.0 + bound_by_level[anchor]) / rho[anchor] - 1.0  # bound on the reference level
        if bound_by_level[anchor] <= TOL["beta"]:
            beta_ref = 1.0 / rho[anchor] - 1.0
            beta_by_level = {lvl: rho[lvl] * (1.0 + beta_ref) - 1.0 for lvl in rho}
            beta_by_level[anchor] = 0.0
            out["beta_source"] = (
                f"corner rule: at instrumentation level {anchor} some region sits at its floor "
                f"(bound {bound_by_level[anchor]:.4f} <= {TOL['beta']}), so beta = 0 there; other levels from ratios"
            )
        else:
            beta_ref = None
            out["beta_source"] = "unresolved"
    else:
        beta_ref = None
        out["beta_source"] = "unresolved (floor violated or no non-later region)"
    if 0 in rho:  # the conditional reading, never used for a status
        b0 = 1.0 / rho[0] - 1.0
        out["beta_if_level0_undistorted"] = {lvl: rho[lvl] * (1.0 + b0) - 1.0 for lvl in rho}
    out.update({"beta_ref": beta_ref, "beta_by_level": beta_by_level, "beta_max": beta_max,
                "bound_by_level": bound_by_level})
    a_min = 1.0 + (beta_ref if beta_ref is not None else 0.0)

    # ---- serialization: look for an earlier-region intervention footprint
    s_hat: float | None = None
    s_source = "no earlier-region intervention arm in the data"
    out["design"] = {"interventions": []}
    if not graph.overlaps:
        s_hat, s_source = 0.0, "no declared overlaps: serialization has nothing to act on"
    else:
        chained = _chained(graph)
        usable_pairs = [o for o in graph.overlaps if o.earlier not in chained]
        if not usable_pairs:
            out["warnings"].append("every overlap pair has a chained earlier region; s cannot be bounded")
        obj = np.zeros_like(S_GRID)
        info = 0.0
        used: list[str] = []
        if design is None:
            interventions = _find_interventions(graph, moms, levels, ref, usable_pairs)
        else:
            interventions = design.get("interventions", [])
        out["design"] = {"interventions": interventions}
        for k, E, affected in interventions:
            m = moms[k]
            se = np.sqrt(m.se**2 + moms[ref].se**2)
            a_k = _a_vector(graph, m, 0.0)  # arm-k earlier regions (non-later, so exact)
            for j, s in enumerate(S_GRID):
                a_pred = _a_vector(graph, moms[ref], s)  # a_l from the reference arm at this s
                a_pred[[idx[e] for e in E]] = a_k[[idx[e] for e in E]]
                mu_pred = _predict_mu(graph, a_pred, s)
                for l_ in affected:
                    obj[j] += ((mu_pred[idx[l_]] - m.mu[idx[l_]]) / max(se[idx[l_]], 1e-12)) ** 2
            # informativeness: does the prediction for the affected regions move with s at all?
            for l_ in affected:
                preds = []
                for s in (0.0, 1.0):
                    a_pred = _a_vector(graph, moms[ref], s)
                    a_pred[[idx[e] for e in E]] = a_k[[idx[e] for e in E]]
                    preds.append(_predict_mu(graph, a_pred, s)[idx[l_]])
                info = max(info, abs(preds[1] - preds[0]) / max(se[idx[l_]], 1e-12))
            used.append(f"arm {k}: perturbed {E}, response read on {affected}")
        if used and info > 3.0:
            s_hat = float(S_GRID[int(np.argmin(obj))])
            s_source = "earlier-region intervention: " + "; ".join(used)
        elif used:
            s_source = "earlier-region intervention found but uninformative (later region fully hidden in both arms)"
    s_max = _s_max(graph, moms[ref], a_min) if graph.overlaps else 0.0
    if s_hat is None and graph.overlaps and s_max <= TOL["s"]:
        s_hat, s_source = 0.0, "corner rule: every later region sits at its floor, so s = 0"
    out.update({"s_hat": s_hat, "s_max": s_max, "s_source": s_source})

    # ---- a and alpha in the reference arm, c~ and c
    s_lo, s_hi = (s_hat, s_hat) if s_hat is not None else (0.0, s_max)
    a_lo, a_hi = _a_vector(graph, moms[ref], s_hi), _a_vector(graph, moms[ref], s_lo)  # a_l decreases in s
    c_t = []
    for s_val, a_val in ((s_lo, a_hi), (s_hi, a_lo)):
        _, _, omega = forward(graph, a_val, s_val, 0.0)
        c_t.append(float(omega.sum() - moms[ref].Dbar))
    c_tilde = [min(c_t), max(c_t)]
    scale_lo, scale_hi = (1.0 + beta_ref, 1.0 + beta_ref) if beta_ref is not None else (1.0 + (beta_max or 0.0), 1.0)
    out.update(
        {
            "a_lo": a_lo,
            "a_hi": a_hi,
            "alpha_lo": a_lo / scale_lo,
            "alpha_hi": a_hi / scale_hi,
            "c_tilde": c_tilde,
            "c_lo": c_tilde[0] / scale_lo,
            "c_hi": c_tilde[1] / scale_hi,
        }
    )
    return out


# --------------------------------------------------------------------------- statuses


def _ci(samples: list[float]) -> list[float] | None:
    x = np.array([v for v in samples if v is not None and v == v], dtype=float)
    if len(x) < 10:
        return None
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def _status_from_interval(
    lo: float, hi: float, inert: float, tol: float, point: float | None = None, one_sided: bool = True
) -> str:
    """Decision rule against the tolerance band B = [inert - tol, inert + tol].

    * ``estimated`` when the whole interval lies beyond B (above it, for a
      one-sided parameter such as alpha >= 1, beta >= 0, s >= 0, c >= 0);
    * ``absent`` when the point estimate lies inside B, or below it for a
      one-sided parameter (a value the mechanism forbids is consistent with
      inert by definition); the interval is then the upper limit. For an
      identified *set* with no point estimate the whole set must lie inside B;
    * otherwise ``not_identifiable_from_this_data`` (compatible with both inert and small).
    """
    if lo > inert + tol or (not one_sided and hi < inert - tol):
        return STATUS_EST
    if point is None:
        inside = (lo >= inert - tol or one_sided) and hi <= inert + tol
    else:
        inside = point <= inert + tol and (one_sided or point >= inert - tol)
    return STATUS_ABS if inside else STATUS_NI


def attribute(rows: list[dict[str, Any]], n_boot: int = 200, n_perm: int = 200, seed: int = 0) -> dict[str, Any]:
    graph = Graph.from_dict(rows[0]["graph"])
    arms = arms_from_rows(rows, graph)
    if not arms:
        raise ValueError("no step rows")
    rng = np.random.default_rng(seed)
    levels = [a.level for a in arms]
    ref = 0
    names = [a.name for a in arms]

    gate = detect_gate(arms, graph, rng, n_perm)
    masks = [(a.q <= gate["tau"]) if gate["detected"] else np.ones(len(a.q), bool) for a in arms]
    for a, m in zip(arms, masks, strict=True):
        if m.sum() < 3:
            raise ValueError(f"condition {a.name!r} has fewer than 3 ungated steps; nothing to estimate")

    point = core(graph, [moments(a, m) for a, m in zip(arms, masks, strict=True)], levels, ref)

    boots: list[dict[str, Any]] = []
    gate_boot: list[dict[str, float]] = []
    for _ in range(n_boot):
        moms_b = []
        for a, m in zip(arms, masks, strict=True):
            ii = np.flatnonzero(m)
            pick = rng.choice(ii, size=len(ii), replace=True)
            arm_b = Arm(a.name, a.level, a.q[pick], a.Y[pick], a.T[pick])
            moms_b.append(moments(arm_b, np.ones(len(pick), bool)))
        boots.append(core(graph, moms_b, levels, ref, design=point["design"]))
        if gate["detected"]:
            picks = [rng.choice(len(a.q), size=len(a.q), replace=True) for a in arms]
            shuffled = [Arm(a.name, a.level, a.q[p], a.Y[p], a.T[p]) for a, p in zip(arms, picks, strict=True)]
            _, lr_b = _pooled_scan(shuffled, np.array([gate["tau"]]))
            gate_boot.append({graph.regions[j]: float(math.exp(lr_b[0, j])) for j in range(len(graph.regions))})

    report = _assemble(graph, arms, names, levels, ref, gate, point, boots, gate_boot)
    return report


def _assemble(graph, arms, names, levels, ref, gate, point, boots, gate_boot) -> dict[str, Any]:
    R = len(graph.regions)
    later = graph.later_regions
    mech: dict[str, Any] = {}
    warnings = list(point["warnings"])

    def boot(key: str, fn=lambda v: v) -> list[float]:
        return [fn(b[key]) for b in boots if b.get(key) is not None]

    # ---------------- observation (mechanism 5)
    obs: dict[str, Any] = {"tolerance": TOL["beta"], "inert_value": 0.0}
    if point["beta_ref"] is not None:
        by_level = {str(lvl): float(b) for lvl, b in point["beta_by_level"].items()}
        ci = {
            str(lvl): _ci([b["beta_by_level"][lvl] for b in boots if b.get("beta_by_level")])
            for lvl in point["beta_by_level"]
        }
        statuses = [
            _status_from_interval(*(ci[str(lvl)] or [by_level[str(lvl)]] * 2), 0.0, TOL["beta"], by_level[str(lvl)])
            for lvl in point["beta_by_level"]
        ]
        status = STATUS_EST if STATUS_EST in statuses else (STATUS_NI if STATUS_NI in statuses else STATUS_ABS)
        obs.update(
            status=status,
            estimate={"beta_by_level": by_level, "beta": by_level[str(levels[ref])]},
            interval={"beta_by_level": ci, "beta": ci[str(levels[ref])]},
            reason=f"beta resolved from: {point['beta_source']}",
        )
    else:
        bmax = _ci(boot("beta_max"))
        hi = bmax[1] if bmax else point["beta_max"]
        obs.update(
            status=STATUS_NI,
            estimate=None,
            interval=None,
            bounds={"beta": [0.0, hi]},
            identified={"a_r = (1+beta) alpha_r": {r: float(point["a_hi"][i]) for i, r in enumerate(graph.regions)}},
            reason=(
                "no instrumentation level has a region at its floor: the data determine only the "
                "products a_r = (1+beta) alpha_r (and the ratio (1+beta) between levels, if several), so a "
                "uniform slowdown and an observation distortion are the same law (note §2.1)"
            ),
            separating_intervention=(
                "add an arm at a different instrumentation level with execution held fixed. The ratio "
                "(1+beta_2)/(1+beta_1) is then identified; the absolute values follow once one level is "
                "known to be undistorted (an arm with instrumentation off) or once some region is seen at its floor"
            ),
        )
        if point.get("beta_if_level0_undistorted") is not None:
            obs["conditional_estimate"] = {
                "assumption": "instrumentation level 0 carries no distortion (beta = 0)",
                "beta_by_level": {str(lvl): float(b) for lvl, b in point["beta_if_level0_undistorted"].items()},
            }
    mech["observation"] = obs

    # ---------------- serialization (mechanism 2)
    ser: dict[str, Any] = {"tolerance": TOL["s"], "inert_value": 0.0}
    # Which pairs an earlier-region intervention can inform depends on where in the
    # identified set the truth lies: a pair fully hidden (omega_p = f_l a_l) cannot respond.
    # Evaluate at both ends, s = 0 (a_hi) and s = s_max (a_lo), and say which is which.
    def not_hidden(a: np.ndarray) -> list[str]:
        return [
            f"{o.earlier}->{o.later}" for o in graph.overlaps
            if o.overlap_ms * a[graph.index[o.earlier]] < graph.floors[graph.index[o.later]] * a[graph.index[o.later]] + 1e-12
        ]
    at_s_zero, at_s_max = not_hidden(point["a_hi"]), not_hidden(point["a_lo"])
    throughout = [p for p in at_s_zero if p in at_s_max]
    only_small_s = [p for p in at_s_zero if p not in at_s_max]
    intervention_text = (
        "add an arm at the same instrumentation level in which one *earlier* region of a declared pair "
        "is perturbed (e.g. a known extra delay injected into that region) and nothing else; s is read "
        "from the later region's response (note §2.2). Pairs whose later region is not fully hidden "
        f"anywhere in the identified set, hence informative: {throughout or 'none'}; pairs not fully "
        f"hidden only towards s = 0, uninformative if s is larger: {only_small_s or 'none'}"
    )
    if not graph.overlaps:
        ser.update(status=STATUS_ABS, estimate={"s": 0.0}, interval=None, reason=point["s_source"])
    elif point["s_hat"] is not None and point["s_source"].startswith("earlier-region"):
        ci = _ci(boot("s_hat"))
        st = _status_from_interval(*(ci or [point["s_hat"]] * 2), 0.0, TOL["s"], point["s_hat"])
        ser.update(status=st, estimate={"s": point["s_hat"]}, interval={"s": ci}, reason=point["s_source"])
    elif point["s_hat"] is not None:  # corner rule
        smax = _ci(boot("s_max"))
        ser.update(status=STATUS_ABS, estimate={"s": 0.0}, interval={"s_upper_limit": smax[1] if smax else point["s_max"]},
                   reason=point["s_source"], tolerance=max(TOL["s"], smax[1] if smax else point["s_max"]))
    else:
        smax = _ci(boot("s_max"))
        ser.update(
            status=STATUS_NI, estimate=None, interval=None,
            bounds={"s": [0.0, smax[1] if smax else point["s_max"]]},
            reason=(
                "passive data at one execution point: a serialized fraction s of an overlap and a slowdown "
                f"alpha_l = 1 + s*omega_p/f_l of the later region produce the same means (note §2.2); "
                f"{point['s_source']}"
            ),
            separating_intervention=intervention_text,
        )
    mech["serialization"] = ser

    # ---------------- slowdown (mechanism 1)
    slow: dict[str, Any] = {"tolerance": TOL["alpha"], "inert_value": 1.0}
    alpha_lo_ci = [_ci(boot("alpha_lo", lambda v, i=i: float(v[i]))) for i in range(R)]
    alpha_hi_ci = [_ci(boot("alpha_hi", lambda v, i=i: float(v[i]))) for i in range(R)]
    resolved_beta = point["beta_ref"] is not None
    resolved_s = point["s_hat"] is not None
    per_region: dict[str, Any] = {}
    present, unresolved_regions = [], []
    for i, r in enumerate(graph.regions):
        lo = (alpha_lo_ci[i] or [float(point["alpha_lo"][i])] * 2)[0]
        hi = (alpha_hi_ci[i] or [float(point["alpha_hi"][i])] * 2)[1]
        pt = float(point["alpha_hi"][i]) if resolved_beta and (resolved_s or r not in later) else None
        st = _status_from_interval(lo, hi, 1.0, TOL["alpha"], pt)
        per_region[r] = {"alpha": pt, "interval": [lo, hi], "status": st}
        if st == STATUS_EST:
            present.append(r)
        if pt is None and st != STATUS_ABS:
            unresolved_regions.append(r)
    if resolved_beta and (resolved_s or not any(per_region[r]["status"] != STATUS_ABS for r in later)):
        status = STATUS_EST if present else (STATUS_NI if any(v["status"] == STATUS_NI for v in per_region.values()) else STATUS_ABS)
        slow.update(
            status=status,
            estimate={"regions": present, "alpha": {r: per_region[r]["alpha"] for r in present}},
            interval={"alpha": {r: per_region[r]["interval"] for r in present}},
            reason="beta resolved, so alpha_r = a_r/(1+beta) region by region" if status != STATUS_ABS
            else "every alpha_r within tolerance of 1",
        )
    elif all(v["status"] == STATUS_ABS for v in per_region.values()):
        slow.update(status=STATUS_ABS, estimate={"regions": [], "alpha": {}}, interval=None,
                    reason="every identified product a_r = (1+beta) alpha_r is within tolerance of 1, which forces alpha_r = 1 (and beta = 0)")
    else:
        slow.update(
            status=STATUS_NI, estimate=None, interval=None,
            bounds={"alpha": {r: per_region[r]["interval"] for r in graph.regions}},
            reason=(
                ("beta unresolved: alpha_r is known only up to the common factor (1+beta); " if not resolved_beta else "")
                + (f"serialization unresolved: alpha for later regions {sorted(set(unresolved_regions) & later)} is confounded with s; " if not resolved_s and (set(unresolved_regions) & later) else "")
                + f"regions with slowdown established regardless: {present}"
            ),
            separating_intervention=(obs.get("separating_intervention") if not resolved_beta else intervention_text),
        )
    slow["per_region"] = per_region
    mech["slowdown"] = slow

    # ---------------- additive (mechanism 3)
    add: dict[str, Any] = {"tolerance": TOL["c_ms"], "inert_value": 0.0}
    c_lo_ci = _ci(boot("c_lo"))
    c_hi_ci = _ci(boot("c_hi"))
    lo = (c_lo_ci or [point["c_lo"]] * 2)[0]
    hi = (c_hi_ci or [point["c_hi"]] * 2)[1]
    tight = resolved_beta and abs(point["c_hi"] - point["c_lo"]) < 1e-9
    st = _status_from_interval(lo, hi, 0.0, TOL["c_ms"], float(point["c_hi"]) if tight else None)
    if tight:
        add.update(status=st, estimate={"c_ms": float(point["c_hi"])}, interval={"c_ms": [lo, hi]},
                   reason="c~ = sum_p omega_p - mean(sum_r y_r - T) with the realised overlaps at the resolved point, divided by (1+beta)")
    elif st == STATUS_ABS:
        add.update(status=STATUS_ABS, estimate={"c_ms": 0.0}, interval={"c_ms": [lo, hi]},
                   reason="the identified product (1+beta) c is within tolerance of 0, which forces c = 0")
    else:
        add.update(
            status=STATUS_NI, estimate=None, interval=None, bounds={"c_ms": [lo, hi]},
            identified={"(1+beta) c": point["c_tilde"]},
            reason=("c is identified only as (1+beta) c while beta is unresolved" if not resolved_beta
                    else "the realised overlap depends on the unresolved s (later region fully hidden), so c is bounded, not pinned"),
            separating_intervention=obs.get("separating_intervention") if not resolved_beta else intervention_text,
        )
    mech["additive"] = add

    # ---------------- regime gate (mechanism 4)
    rg: dict[str, Any] = {"tolerance": TOL["gamma"], "inert_value": 1.0, "scan": {k: v for k, v in gate.items() if k not in ("ratio", "z")}}
    if gate.get("degenerate"):
        rg.update(status=STATUS_NI, estimate=None, interval=None, reason=gate["reason"],
                  separating_intervention="run arms that straddle the suspected tau in q")
    elif not gate["detected"]:
        mdg = gate.get("min_detectable_gamma_minus_1")
        rg.update(status=STATUS_ABS, estimate={"gamma": 1.0, "regions": []}, interval=None,
                  tolerance=max(TOL["gamma"], mdg or 0.0),
                  reason=(f"no threshold in q within [{gate['q_range'][0]:.3g}, {gate['q_range'][1]:.3g}] splits any region's "
                          f"mean beyond the permutation null (99%); a gate with tau outside the observed q range is indistinguishable from absent"))
    else:
        members = gate["members"]
        clean = [r for r in members if r not in later or not any(o.earlier in members for o in graph.overlaps if o.later == r)]
        conf = [r for r in members if r not in clean]
        ratios = gate["ratio"]
        gam = {r: ratios[r] for r in members}
        gam_ci = {r: _ci([g[r] for g in gate_boot]) for r in members}
        if clean:
            logs = [math.log(ratios[r]) for r in clean]
            pooled = float(math.exp(np.mean(logs)))
            pooled_ci = _ci([float(math.exp(np.mean([math.log(g[r]) for r in clean]))) for g in gate_boot])
            rg.update(
                status=STATUS_EST,
                estimate={"gamma": pooled, "gamma_by_region": gam, "regions": members, "tau": gate["tau"]},
                interval={"gamma": pooled_ci, "gamma_by_region": gam_ci, "tau": gate["tau_interval"]},
                reason="within-arm ratio of means above/below tau cancels beta and alpha; tau located by the scan",
            )
            if conf:
                rg["caveat"] = f"regions {conf} respond to q but are later regions whose earlier partner is gated; their response may be inherited through serialization"
            if not resolved_s and any(r in later for r in clean):
                rg["caveat"] = (rg.get("caveat", "") + f" gamma for later regions {[r for r in clean if r in later]} is exact only if the later region is fully hidden; otherwise biased by the unresolved s").strip()
        else:
            rg.update(status=STATUS_NI, estimate=None, interval=None,
                      identified={"gamma_by_region": gam, "tau": gate["tau"]},
                      reason=f"only later regions {conf} respond to q while their earlier partners are gated: the response may be their own gate or serialization of the partner's gated overlap",
                      separating_intervention=intervention_text)
    mech["regime_gate"] = rg

    return {
        "schema": REPORT_SCHEMA,
        "n_steps": int(sum(len(a.q) for a in arms)),
        "reference_condition": names[ref],
        "conditions": {a.name: {"steps": int(len(a.q)), "instrumentation_level": a.level} for a in arms},
        "tolerances": TOL,
        "instrumentation_ratio_by_level": {str(k): v for k, v in point["rho_by_level"].items()},
        "warnings": warnings,
        "mechanisms": mech,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="attribute", description="Attribute mechanisms from data.jsonl alone.")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap replicates for intervals")
    ap.add_argument("--permutations", type=int, default=200, help="permutations for the gate-scan null")
    ap.add_argument("--seed", type=int, default=0, help="seed for bootstrap/permutations only")
    args = ap.parse_args(argv)
    rows = load_rows(args.data)
    report = attribute(rows, n_boot=args.bootstrap, n_perm=args.permutations, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
        fh.write("\n")
    for name, m in report["mechanisms"].items():
        print(f"{name:14s} {m['status']}", file=sys.stderr)
    return 0


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


if __name__ == "__main__":
    sys.exit(main())
