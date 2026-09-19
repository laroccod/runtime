"""The Part 2 test battery, as functions so the tests and the note share one set of numbers.

``python -m gitm.attribution.battery`` prints the tables the note quotes:

1. recovery bias and spread across seeds at two sample sizes (a pull distribution);
2. the two 2b pairs: abstain on passive data, recover once the intervention arm is present;
3. inert mechanisms reported ``absent``.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

from gitm.attribution.attribute import attribute
from gitm.attribution.simulate import simulate

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "attribution"

#: (mechanism, path into estimate/interval, truth) for full-battery.yaml
PULL_TARGETS: list[tuple[str, tuple[str, ...], float]] = [
    ("observation", ("beta",), 0.15),
    ("slowdown", ("alpha", "moe_routed"), 1.4),
    ("serialization", ("s",), 0.5),
    ("additive", ("c_ms",), 0.8),
    ("regime_gate", ("gamma",), 1.3),
]


def load(name: str) -> dict[str, Any]:
    with (FIXTURES / name).open() as fh:
        return yaml.safe_load(fh)


def run(params: dict[str, Any], seed: int, steps: int, n_boot: int = 80, n_perm: int = 80) -> dict[str, Any]:
    return attribute(simulate(params, seed, steps), n_boot=n_boot, n_perm=n_perm, seed=seed)


def _dig(d: Any, path: tuple[str, ...]) -> Any:
    for k in path:
        d = d[k]
    return d


def pulls(steps: int, seeds: range, n_boot: int = 80, n_perm: int = 80) -> dict[str, dict[str, float]]:
    """Per target: mean bias, spread (sd of estimates), mean interval half-width, pull mean/sd."""
    params = load("full-battery.yaml")
    est: dict[str, list[float]] = {m: [] for m, _, _ in PULL_TARGETS}
    hw: dict[str, list[float]] = {m: [] for m, _, _ in PULL_TARGETS}
    for seed in seeds:
        report = run(params, seed, steps, n_boot, n_perm)
        for mech, path, _ in PULL_TARGETS:
            m = report["mechanisms"][mech]
            assert m["status"] == "estimated", (mech, m["status"], m.get("reason"))
            est[mech].append(float(_dig(m["estimate"], path)))
            lo, hi = _dig(m["interval"], path)
            hw[mech].append(0.5 * (hi - lo))
    out = {}
    for mech, _, truth in PULL_TARGETS:
        e = est[mech]
        sigma = [h / 1.96 for h in hw[mech]]
        pull = [(x - truth) / s if s > 0 else float("nan") for x, s in zip(e, sigma, strict=True)]
        out[mech] = {
            "truth": truth,
            "mean": statistics.fmean(e),
            "bias": statistics.fmean(e) - truth,
            "spread": statistics.pstdev(e),
            "mean_halfwidth": statistics.fmean(hw[mech]),
            "pull_mean": statistics.fmean(pull),
            "pull_sd": statistics.pstdev(pull),
            "n": len(e),
        }
    return out


def statuses(name: str, seed: int = 0, steps: int = 1200) -> dict[str, str]:
    report = run(load(name), seed, steps)
    return {m: v["status"] for m, v in report["mechanisms"].items()}


def main() -> int:
    print("## Recovery across seeds (full-battery.yaml; 3 arms; bias = mean - truth)\n")
    print("| steps | mechanism | truth | mean | bias | spread (sd) | mean 95% half-width | pull mean | pull sd |")
    print("|---|---|---|---|---|---|---|---|---|")
    for steps in (600, 3000):
        for mech, row in pulls(steps, range(8)).items():
            print(
                f"| {steps} | {mech} | {row['truth']} | {row['mean']:.4f} | {row['bias']:+.4f} | "
                f"{row['spread']:.4f} | {row['mean_halfwidth']:.4f} | {row['pull_mean']:+.2f} | {row['pull_sd']:.2f} |"
            )
    print("\n## Statuses per fixture (seed 0, 1200 steps)\n")
    print("| fixture | slowdown | serialization | additive | regime_gate | observation |")
    print("|---|---|---|---|---|---|")
    for name in sorted(p.name for p in FIXTURES.glob("*.yaml")):
        st = statuses(name)
        print(
            f"| {name} | {st['slowdown']} | {st['serialization']} | {st['additive']} | "
            f"{st['regime_gate']} | {st['observation']} |"
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps({s: pulls(s, range(8)) for s in (600, 3000)}, indent=2))
    else:
        sys.exit(main())
