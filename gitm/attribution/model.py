"""Generative model for Part 2, shared by the simulator and the estimator.

Notation (the note's §1 uses the same symbols):

* regions ``r`` with floors ``f_r`` (ms) and declared overlap pairs
  ``p = (e, l, o_p)``: the *later* region ``l`` can run concurrently with the
  last ``o_p`` ms of the *earlier* region ``e``.
* multipliers ``m_{r,t} = alpha_r * gamma^{[r in G][q_t > tau]}`` (mechanisms 1 and 4).
* realised overlap of a pair, from the two-node DAG "``l`` depends on the
  first ``f_e - o_p`` ms of ``e``":  ``omega_p = min(o_p * m_e, f_l * m_l)``.
  It is homogeneous of degree one in the multipliers, which is what makes
  mechanism 5 an exact copy of a uniform mechanism 1 (note §2.1).
* serialization returns ``s * omega_p`` to the step total and to the later
  region (mechanism 2); ``c`` is added to the total only (mechanism 3).

Undistorted region time and step total (deterministic given ``q_t``)::

    v_r = f_r m_r + s * sum_{p: later(p) = r} omega_p
    T*  = sum_r f_r m_r - (1 - s) * sum_p omega_p + c        (critical path + returned overlap + c)

Recorded (mechanism 5 and noise)::

    y_r = (1 + beta) v_r  eps_r,   T = (1 + beta) T*  eta

with ``eps_r``, ``eta`` independent log-normal, mean one, constant CV.
Everything below is plain Python + numpy so it can be read as the equations.
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Parameter blocks the case names, with their inert values. A block that is
# absent from params.yaml is inert; a block that is present but partial is
# filled with inert values. Nothing is defaulted silently: `describe()` lists
# the resolved values and the simulator writes them to stderr on request.
MECHANISMS = ("slowdown", "serialization", "additive", "regime_gate", "observation")

#: Defaults for the documented noise block (note §1). ``q`` is the regime
#: covariate's generator; ``uniform(0, 2*tau)`` when a gate is declared with
#: tau > 0 so that the gate fires on about half the steps, else uniform(0, 32).
NOISE_DEFAULTS: dict[str, Any] = {"cv_region": 0.03, "cv_total": 0.01, "q": None}


def _warn(msg: str) -> None:
    print(f"[gitm.attribution] warning: {msg}", file=sys.stderr)


def deep_merge(base: Any, override: Any) -> Any:
    """Recursive dict merge; lists and scalars are replaced, not merged.

    This is the semantics of ``conditions[].overrides``: every key named in the
    override replaces the corresponding key of the base, nested dicts merge so
    that ``{slowdown: {alpha: {x: 1.5}}}`` touches only ``alpha.x``, and every
    key not named holds its base value.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(override)


@dataclass(frozen=True)
class Overlap:
    earlier: str
    later: str
    overlap_ms: float


@dataclass
class Graph:
    """Floors and overlap pairs, validated once."""

    regions: list[str]
    floors: np.ndarray  # (R,)
    overlaps: list[Overlap]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, g: dict[str, Any]) -> Graph:
        if not isinstance(g, dict) or "floors_ms" not in g:
            raise ValueError("graph must be a mapping with a 'floors_ms' block")
        floors_map = g["floors_ms"]
        regions = list(floors_map)
        floors = np.array([float(floors_map[r]) for r in regions], dtype=float)
        if (floors <= 0).any():
            bad = [r for r, f in zip(regions, floors, strict=True) if f <= 0]
            raise ValueError(f"floors must be positive; got non-positive floors for {bad}")
        overlaps: list[Overlap] = []
        for o in g.get("overlaps") or []:
            e, l_, ov = o["earlier"], o["later"], float(o["overlap_ms"])
            if e not in floors_map or l_ not in floors_map:
                raise ValueError(f"overlap names unknown region: {o}")
            if e == l_:
                raise ValueError(f"overlap pairs a region with itself: {o}")
            cap = min(floors_map[e], floors_map[l_])
            if ov < 0:
                raise ValueError(f"negative overlap: {o}")
            if ov > cap + 1e-12:
                _warn(
                    f"overlap {e}->{l_} of {ov} ms exceeds min(floors) = {cap} ms; "
                    f"clamped to {cap} ms for the computation (graph is copied verbatim)"
                )
                ov = cap
            overlaps.append(Overlap(e, l_, ov))
        return cls(regions=regions, floors=floors, overlaps=overlaps, raw=copy.deepcopy(g))

    @property
    def index(self) -> dict[str, int]:
        return {r: i for i, r in enumerate(self.regions)}

    @property
    def later_regions(self) -> set[str]:
        return {o.later for o in self.overlaps}

    @property
    def earlier_regions(self) -> set[str]:
        return {o.earlier for o in self.overlaps}


@dataclass
class Mechanisms:
    """Resolved mechanism parameters for one experimental arm."""

    alpha: dict[str, float]  # region -> alpha_r (1.0 when not slowed)
    s: float
    c_ms: float
    gate_regions: set[str]
    gamma: dict[str, float]  # region -> gamma (only for gate_regions)
    tau: float
    beta: float

    def alpha_vec(self, graph: Graph) -> np.ndarray:
        return np.array([self.alpha.get(r, 1.0) for r in graph.regions], dtype=float)

    def gate_vec(self, graph: Graph) -> np.ndarray:
        """Per-region multiplier applied on gated steps (1 off the gate set)."""
        return np.array(
            [self.gamma.get(r, 1.0) if r in self.gate_regions else 1.0 for r in graph.regions],
            dtype=float,
        )


def _as_region_map(value: Any, regions: list[str] | None, what: str) -> dict[str, float]:
    """``alpha``/``gamma`` may be a scalar (applied to ``regions``) or a map."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if regions is None:
        raise ValueError(f"{what}: a scalar needs a 'regions' list to apply to")
    return {str(r): float(value) for r in regions}


def resolve_mechanisms(params: dict[str, Any], graph: Graph) -> Mechanisms:
    """Fill every mechanism block with its inert value and validate names."""
    known = set(graph.regions)

    sd = params.get("slowdown") or {}
    regions = list(sd.get("regions") or [])
    alpha = _as_region_map(sd.get("alpha"), regions or None, "slowdown.alpha")
    for r in regions:
        if r not in alpha:
            _warn(f"slowdown.regions lists {r!r} with no alpha; treating alpha[{r}] = 1")
    for r, a in alpha.items():
        if r not in known:
            raise ValueError(f"slowdown.alpha names unknown region {r!r}")
        if a < 1.0:
            _warn(f"slowdown.alpha[{r}] = {a} < 1 violates the mechanism's alpha >= 1; kept as given")

    s = float((params.get("serialization") or {}).get("s", 0.0))
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"serialization.s must lie in [0, 1]; got {s}")

    c_ms = float((params.get("additive") or {}).get("c_ms", 0.0))
    if c_ms < 0:
        _warn(f"additive.c_ms = {c_ms} < 0; the model assumes c >= 0; kept as given")

    rg = params.get("regime_gate") or {}
    gate_regions = [str(r) for r in (rg.get("regions") or [])]
    for r in gate_regions:
        if r not in known:
            raise ValueError(f"regime_gate.regions names unknown region {r!r}")
    gamma = _as_region_map(rg.get("gamma", 1.0), gate_regions, "regime_gate.gamma")
    for r in gate_regions:
        gamma.setdefault(r, 1.0)
    tau = float(rg.get("tau", math.inf))  # inf: the gate never fires

    beta = float((params.get("observation") or {}).get("beta", 0.0))
    if beta < 0:
        _warn(f"observation.beta = {beta} < 0; the model assumes beta >= 0; kept as given")

    return Mechanisms(
        alpha=alpha, s=s, c_ms=c_ms, gate_regions=set(gate_regions), gamma=gamma, tau=tau, beta=beta
    )


def resolve_noise(params: dict[str, Any], graph: Graph, mech: Mechanisms) -> dict[str, Any]:
    """The documented noise block with defaults applied (note §1)."""
    nz = dict(NOISE_DEFAULTS)
    nz.update(params.get("noise") or {})
    cv_region = nz["cv_region"]
    if isinstance(cv_region, dict):
        vec = np.array([float(cv_region.get(r, NOISE_DEFAULTS["cv_region"])) for r in graph.regions])
    else:
        vec = np.full(len(graph.regions), float(cv_region))
    if (vec < 0).any() or float(nz["cv_total"]) < 0:
        raise ValueError("noise CVs must be non-negative")
    q = nz.get("q")
    if q is None:
        high = 2.0 * mech.tau if (mech.gate_regions and 0 < mech.tau < math.inf) else 32.0
        q = {"dist": "uniform", "low": 0.0, "high": high}
    return {"cv_region": vec, "cv_total": float(nz["cv_total"]), "q": dict(q)}


def draw_q(rng: np.random.Generator, qspec: dict[str, Any], n: int) -> np.ndarray:
    dist = str(qspec.get("dist", "uniform")).lower()
    if dist == "uniform":
        return rng.uniform(float(qspec.get("low", 0.0)), float(qspec.get("high", 32.0)), n)
    if dist == "normal":
        return rng.normal(float(qspec.get("mean", 16.0)), float(qspec.get("sd", 8.0)), n)
    if dist == "lognormal":
        return rng.lognormal(float(qspec.get("mu", 2.5)), float(qspec.get("sigma", 0.5)), n)
    if dist == "constant":
        return np.full(n, float(qspec.get("value", 0.0)))
    raise ValueError(f"unknown q distribution {dist!r} (uniform|normal|lognormal|constant)")


def lognormal_mean_one(rng: np.random.Generator, cv: np.ndarray | float, size: Any) -> np.ndarray:
    """Log-normal factor with E = 1 and coefficient of variation ``cv``."""
    cv = np.asarray(cv, dtype=float)
    sigma2 = np.log1p(cv * cv)
    return np.exp(rng.normal(-0.5 * sigma2, np.sqrt(sigma2), size))


def forward(
    graph: Graph, mult: np.ndarray, s: float, c_ms: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """Mean map: multipliers -> (region times v_r, undistorted total T*, omega_p).

    ``mult`` is the per-region multiplier vector m_r (slowdown times gate).
    This is the only place the critical path is computed; the estimator calls
    the same function with alpha := (1 + beta) alpha, beta := 0.
    """
    f = graph.floors
    idx = graph.index
    base = f * mult
    omega = np.zeros(len(graph.overlaps))
    v = base.copy()
    for k, o in enumerate(graph.overlaps):
        ie, il = idx[o.earlier], idx[o.later]
        omega[k] = min(o.overlap_ms * mult[ie], f[il] * mult[il])
        v[il] += s * omega[k]
    total = float(base.sum() - (1.0 - s) * omega.sum() + c_ms)
    return v, total, omega


def multipliers(mech: Mechanisms, graph: Graph, q: float) -> np.ndarray:
    m = mech.alpha_vec(graph)
    if mech.gate_regions and q > mech.tau:
        m = m * mech.gate_vec(graph)
    return m


def describe(mech: Mechanisms) -> dict[str, Any]:
    """Resolved parameters, for logging; never written into data.jsonl."""
    return {
        "slowdown": {"alpha": dict(mech.alpha)},
        "serialization": {"s": mech.s},
        "additive": {"c_ms": mech.c_ms},
        "regime_gate": {"regions": sorted(mech.gate_regions), "gamma": dict(mech.gamma), "tau": mech.tau},
        "observation": {"beta": mech.beta},
    }
