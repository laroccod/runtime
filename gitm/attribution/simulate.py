"""``simulate --params params.yaml --seed N --steps M --out data.jsonl``.

Row 1 of ``data.jsonl`` is a metadata row carrying ``graph`` verbatim; every
later row is one step ``{step, condition, q, instrumentation_level, total_ms,
regions: {name: ms}}``. The ground-truth parameters are never written to the
data file; ``--describe`` prints the resolved per-arm parameters to stderr.

``instrumentation_level`` is an ordinal label of the arm's instrumentation
configuration: 0 means instrumentation off, which by the definition of
mechanism 5 ("instrumentation inflates ...") carries no distortion; higher
levels are heavier configurations ranked by overhead. Its numeric value is
not beta, and the estimator is told nothing else about beta.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gitm.attribution.model import (
    MECHANISMS,
    Graph,
    Mechanisms,
    deep_merge,
    describe,
    draw_q,
    forward,
    lognormal_mean_one,
    multipliers,
    resolve_mechanisms,
    resolve_noise,
)

SCHEMA = "gitm-attribution/1"
LEVEL_CONVENTION = (
    "instrumentation_level is an ordinal label of the arm's instrumentation "
    "configuration: 0 = instrumentation off (no distortion by definition of "
    "mechanism 5); 1, 2, ... = heavier configurations ranked by overhead. "
    "The label is not the value of beta."
)


def _allocate_steps(conditions: list[dict[str, Any]], steps: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each of ``steps`` global step indices to an arm.

    Counts follow ``steps_fraction`` by largest remainder (fractions are
    normalised with a warning if they do not sum to one); the assignment is
    then shuffled so arms interleave in time, as randomised arms would.
    """
    fr = np.array([float(c.get("steps_fraction", 1.0)) for c in conditions])
    if (fr < 0).any():
        raise ValueError("steps_fraction must be non-negative")
    if fr.sum() <= 0:
        raise ValueError("steps_fraction values sum to zero")
    if not math.isclose(fr.sum(), 1.0, abs_tol=1e-6):
        print(
            f"[gitm.attribution] warning: steps_fraction sums to {fr.sum():.4f}; normalised to 1",
            file=sys.stderr,
        )
    fr = fr / fr.sum()
    raw = fr * steps
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: steps - counts.sum()]:
        counts[i] += 1
    assignment = np.repeat(np.arange(len(conditions)), counts)
    rng.shuffle(assignment)
    return assignment


def _arm_params(base: dict[str, Any], cond: dict[str, Any]) -> dict[str, Any]:
    overrides = cond.get("overrides") or {}
    unknown = set(overrides) - set(MECHANISMS) - {"noise", "graph"}
    if unknown:
        raise ValueError(f"condition {cond.get('name')!r} overrides unknown blocks {sorted(unknown)}")
    if "graph" in overrides:
        raise ValueError("conditions[].overrides may not change the graph")
    return deep_merge(base, overrides)


def instrumentation_levels(betas: list[float]) -> list[int]:
    """0 for beta == 0; else 1 + rank of beta among the distinct non-zero betas."""
    nonzero = sorted({b for b in betas if b != 0.0})
    return [0 if b == 0.0 else 1 + nonzero.index(b) for b in betas]


def simulate(params: dict[str, Any], seed: int, steps: int) -> list[dict[str, Any]]:
    """Generate ``data.jsonl`` rows (metadata row first) from a params mapping."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    graph = Graph.from_dict(params["graph"])
    conditions = list(params.get("conditions") or [{"name": "passive", "steps_fraction": 1.0}])
    names = [str(c.get("name", f"arm{i}")) for i, c in enumerate(conditions)]
    if len(set(names)) != len(names):
        raise ValueError(f"condition names must be unique: {names}")

    rng = np.random.default_rng(seed)
    arms: list[tuple[Mechanisms, dict[str, Any]]] = []
    for cond in conditions:
        p = _arm_params(params, cond)
        mech = resolve_mechanisms(p, graph)
        arms.append((mech, resolve_noise(p, graph, mech)))
    levels = instrumentation_levels([m.beta for m, _ in arms])

    assignment = _allocate_steps(conditions, steps, rng)
    rows: list[dict[str, Any]] = [
        {
            "schema": SCHEMA,
            "graph": params["graph"],
            "seed": int(seed),
            "steps": int(steps),
            "conditions": names,
            "instrumentation_level_convention": LEVEL_CONVENTION,
        }
    ]
    R = len(graph.regions)
    for t in range(steps):
        k = int(assignment[t])
        mech, noise = arms[k]
        q = float(draw_q(rng, noise["q"], 1)[0])
        v, total, _ = forward(graph, multipliers(mech, graph, q), mech.s, mech.c_ms)
        scale = 1.0 + mech.beta
        y = scale * v * lognormal_mean_one(rng, noise["cv_region"], R)
        T = scale * total * float(lognormal_mean_one(rng, noise["cv_total"], 1)[0])
        rows.append(
            {
                "step": t,
                "condition": names[k],
                "q": q,
                "instrumentation_level": levels[k],
                "total_ms": float(T),
                "regions": {r: float(y[i]) for i, r in enumerate(graph.regions)},
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def load_params(path: Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        params = yaml.safe_load(fh)
    if not isinstance(params, dict) or "graph" not in params:
        raise ValueError(f"{path}: params.yaml must be a mapping with a 'graph' block")
    return params


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="simulate", description="Simulate per-step region timings under the five mechanisms."
    )
    ap.add_argument("--params", required=True, type=Path)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--steps", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--describe", action="store_true", help="print the resolved per-arm parameters to stderr"
    )
    args = ap.parse_args(argv)
    params = load_params(args.params)
    rows = simulate(params, args.seed, args.steps)
    write_jsonl(rows, args.out)
    if args.describe:
        graph = Graph.from_dict(params["graph"])
        for cond in params.get("conditions") or [{"name": "passive"}]:
            mech = resolve_mechanisms(_arm_params(params, cond), graph)
            print(json.dumps({cond.get("name"): describe(mech)}, default=str), file=sys.stderr)
    print(f"wrote {len(rows) - 1} steps to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
