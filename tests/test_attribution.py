"""Part 2 test battery: simulator contract, estimator decision rule, and the numbers.

The case requires: recovery bias and spread across repeated seeds at two sample
sizes; both 2b confounded pairs shown empirically (abstain on passive data,
recover once the intervention arm is present); at least one truly inert
mechanism reported ``absent``. Each of those is a test here, and the numbers
the note quotes come from :mod:`gitm.attribution.battery`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from gitm.attribution import battery
from gitm.attribution.attribute import STATUS_ABS, STATUS_EST, STATUS_NI, attribute
from gitm.attribution.model import Graph, deep_merge, forward, resolve_mechanisms
from gitm.attribution.simulate import instrumentation_levels, simulate

FIX = battery.FIXTURES


def _report(name: str, seed: int = 0, steps: int = 1200) -> dict:
    return battery.run(battery.load(name), seed, steps)


def _mean(rows: list[dict], pick) -> float:
    return sum(pick(r) for r in rows) / len(rows)


# --------------------------------------------------------------------------- model


def test_forward_matches_the_equations():
    """v_r = f_r m_r + s sum omega; T* = sum f m - (1-s) sum omega + c; omega = min(o m_e, f_l m_l)."""
    g = Graph.from_dict(
        {"floors_ms": {"a": 2.0, "b": 0.5, "c": 1.0}, "overlaps": [{"earlier": "a", "later": "b", "overlap_ms": 0.4}]}
    )
    v, T, omega = forward(g, np.array([1.5, 1.0, 1.0]), s=0.5, c_ms=0.3)
    assert omega[0] == pytest.approx(min(0.4 * 1.5, 0.5 * 1.0))  # capped by the later region
    assert v.tolist() == pytest.approx([3.0, 0.5 + 0.5 * 0.5, 1.0])
    assert T == pytest.approx(3.0 + 0.5 + 1.0 - 0.5 * 0.5 + 0.3)


def test_overlap_is_homogeneous_so_observation_equals_uniform_slowdown():
    """Proposition 1 at the level of the mean map: scaling every multiplier by (1+beta)
    scales every region time and the total by (1+beta), overlaps included."""
    g = Graph.from_dict(battery.load("pair1-passive.yaml")["graph"])
    m = np.array([1.4] + [1.0] * (len(g.regions) - 1))
    v1, T1, _ = forward(g, m, 0.3, 0.0)
    v2, T2, _ = forward(g, 1.15 * m, 0.3, 0.0)
    assert v2.tolist() == pytest.approx((1.15 * v1).tolist())
    assert T2 == pytest.approx(1.15 * T1)


def test_deep_merge_overrides_only_named_keys():
    base = {"slowdown": {"regions": ["a"], "alpha": {"a": 1.4}}, "observation": {"beta": 0.15}}
    out = deep_merge(base, {"slowdown": {"alpha": {"b": 1.2}}})
    assert out["slowdown"]["alpha"] == {"a": 1.4, "b": 1.2}
    assert out["observation"] == {"beta": 0.15}
    assert base["slowdown"]["alpha"] == {"a": 1.4}  # base untouched


def test_instrumentation_levels_are_ordinal_with_zero_reserved():
    assert instrumentation_levels([0.15, 0.0, 0.3, 0.15]) == [1, 0, 2, 1]


# --------------------------------------------------------------------------- simulator contract


def test_simulator_rows_follow_the_contract(tmp_path: Path):
    params = battery.load("full-battery.yaml")
    rows = simulate(params, seed=3, steps=100)
    assert rows[0]["graph"] == params["graph"]  # verbatim
    assert len(rows) == 101
    for row in rows[1:]:
        assert set(row) == {"step", "condition", "q", "instrumentation_level", "total_ms", "regions"}
        assert set(row["regions"]) == set(params["graph"]["floors_ms"])
    counts = {}
    for row in rows[1:]:
        counts[row["condition"]] = counts.get(row["condition"], 0) + 1
    assert counts == {"passive": 40, "low_instr": 30, "slow_index": 30}
    levels = {row["condition"]: row["instrumentation_level"] for row in rows[1:]}
    assert levels == {"passive": 1, "low_instr": 0, "slow_index": 1}
    assert simulate(params, seed=3, steps=100) == rows  # deterministic in the seed


def test_simulator_honours_arbitrary_overrides():
    params = battery.load("pair2-passive.yaml")
    params["conditions"].append(
        {"name": "weird", "steps_fraction": 0.5, "overrides": {"additive": {"c_ms": 2.0}, "regime_gate": {"regions": ["hc_mix"], "gamma": 2.0, "tau": -1}}}
    )
    params["conditions"][0]["steps_fraction"] = 0.5
    rows = simulate(params, seed=0, steps=400)
    g = Graph.from_dict(params["graph"])
    passive = [r for r in rows[1:] if r["condition"] == "passive"]
    weird = [r for r in rows[1:] if r["condition"] == "weird"]
    hc = {k: _mean(v, lambda r: r["regions"]["hc_mix"]) for k, v in (("p", passive), ("w", weird))}
    tot = {k: _mean(v, lambda r: r["total_ms"]) for k, v in (("p", passive), ("w", weird))}
    assert hc["w"] == pytest.approx(2.0 * hc["p"], rel=0.05)  # gate fires on every step (tau = -1)
    assert tot["w"] - tot["p"] == pytest.approx(2.0 + (2.0 - 1.0) * g.floors[g.index["hc_mix"]], rel=0.1)
    with pytest.raises(ValueError, match="unknown blocks"):
        simulate(deep_merge(params, {"conditions": [{"name": "x", "overrides": {"nope": {}}}]}), 0, 10)


def test_resolve_mechanisms_fills_inert_values():
    g = Graph.from_dict(battery.load("inert.yaml")["graph"])
    m = resolve_mechanisms({}, g)
    assert m.alpha == {} and m.s == 0.0 and m.c_ms == 0.0 and m.beta == 0.0 and not m.gate_regions


# --------------------------------------------------------------------------- 2b pair 1


def test_pair1_passive_abstains_and_uniform_copy_is_indistinguishable():
    r = _report("pair1-passive.yaml")
    assert r["mechanisms"]["observation"]["status"] == STATUS_NI
    assert r["mechanisms"]["slowdown"]["status"] == STATUS_NI
    b = r["mechanisms"]["observation"]["bounds"]["beta"]
    assert b[0] == 0.0 and 0.15 <= b[1] <= 0.20  # truth inside the identified interval
    a = r["mechanisms"]["slowdown"]["bounds"]["alpha"]["moe_routed"]
    assert a[0] <= 1.4 <= a[1]
    # the exact copy (uniform alpha = 1.15, beta = 0) yields the same statuses
    u = _report("pair1-uniform.yaml")
    assert u["mechanisms"]["observation"]["status"] == STATUS_NI
    assert u["mechanisms"]["slowdown"]["status"] == STATUS_NI


def test_pair1_recovers_with_an_instrumentation_arm():
    r = _report("pair1-intervention.yaml")
    obs, slow = r["mechanisms"]["observation"], r["mechanisms"]["slowdown"]
    assert obs["status"] == STATUS_EST and slow["status"] == STATUS_EST
    lo, hi = obs["interval"]["beta_by_level"]["1"]
    assert lo <= 0.15 <= hi
    lo, hi = slow["interval"]["alpha"]["moe_routed"]
    assert lo <= 1.4 <= hi
    assert slow["estimate"]["regions"] == ["moe_routed"]
    for mech in ("serialization", "additive", "regime_gate"):
        assert r["mechanisms"][mech]["status"] == STATUS_ABS, mech


# --------------------------------------------------------------------------- 2b pair 2


def test_pair2_passive_abstains_with_bounds_containing_truth():
    r = _report("pair2-passive.yaml")
    ser = r["mechanisms"]["serialization"]
    assert ser["status"] == STATUS_NI
    assert ser["bounds"]["s"][0] == 0.0 and 0.5 <= ser["bounds"]["s"][1] <= 0.6
    assert "attn_index->attn_compress" in ser["separating_intervention"]
    assert r["mechanisms"]["slowdown"]["status"] == STATUS_NI  # alpha of the later regions is the confound


def test_pair2_recovers_with_an_earlier_region_intervention():
    r = _report("pair2-intervention.yaml")
    ser = r["mechanisms"]["serialization"]
    assert ser["status"] == STATUS_EST
    lo, hi = ser["interval"]["s"]
    assert lo <= 0.5 <= hi and hi - lo < 0.2
    assert r["mechanisms"]["slowdown"]["status"] == STATUS_ABS
    assert r["mechanisms"]["additive"]["status"] == STATUS_ABS


def test_pair2_fully_hidden_pair_is_uninformative():
    r = _report("pair2-hidden-pair-uninformative.yaml")
    ser = r["mechanisms"]["serialization"]
    assert ser["status"] == STATUS_NI
    assert "uninformative" in ser["reason"]


# --------------------------------------------------------------------------- inert -> absent


def test_inert_mechanisms_are_reported_absent():
    r = _report("inert.yaml")
    for mech, v in r["mechanisms"].items():
        assert v["status"] == STATUS_ABS, (mech, v.get("reason"))
        assert "tolerance" in v


def test_gate_with_tau_outside_the_observed_range_is_absent():
    r = _report("gate-out-of-range.yaml")
    assert r["mechanisms"]["regime_gate"]["status"] == STATUS_ABS
    assert "outside the observed q range" in r["mechanisms"]["regime_gate"]["reason"]


def test_single_region_graph():
    r = _report("single-region.yaml")
    assert r["mechanisms"]["serialization"]["status"] == STATUS_ABS
    add = r["mechanisms"]["additive"]
    assert add["status"] == STATUS_EST
    lo, hi = add["interval"]["c_ms"]
    assert lo <= 0.4 <= hi


# --------------------------------------------------------------------------- recovery across seeds


@pytest.mark.parametrize("steps", [600, 3000])
def test_recovery_bias_and_spread(steps: int):
    """Pull distribution at two sample sizes: mean bias small relative to the interval,
    spread shrinking with sample size, and the interval half-width tracking the spread."""
    rows = battery.pulls(steps, range(6), n_boot=60, n_perm=60)
    for mech, row in rows.items():
        assert abs(row["bias"]) < 2.0 * row["mean_halfwidth"], (mech, row)
        assert row["spread"] < 3.0 * row["mean_halfwidth"], (mech, row)  # intervals are not wildly optimistic


def test_spread_shrinks_with_sample_size():
    small = battery.pulls(400, range(5), n_boot=40, n_perm=40)
    large = battery.pulls(3200, range(5), n_boot=40, n_perm=40)
    for mech in small:
        assert large[mech]["spread"] < small[mech]["spread"] + 1e-9, mech
        assert large[mech]["mean_halfwidth"] < small[mech]["mean_halfwidth"], mech


# --------------------------------------------------------------------------- CLI contract


def test_cli_contract_end_to_end(tmp_path: Path):
    params = FIX / "pair1-intervention.yaml"
    data = tmp_path / "data.jsonl"
    report = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, "-m", "gitm.attribution.simulate", "--params", str(params), "--seed", "7", "--steps", "300", "--out", str(data)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "gitm.attribution.attribute", "--data", str(data), "--out", str(report), "--bootstrap", "40", "--permutations", "40"],
        check=True,
    )
    lines = data.read_text().splitlines()
    assert json.loads(lines[0])["graph"] == yaml.safe_load(params.read_text())["graph"]
    assert len(lines) == 301
    rep = json.loads(report.read_text())
    assert set(rep["mechanisms"]) == {"slowdown", "serialization", "additive", "regime_gate", "observation"}
    for m in rep["mechanisms"].values():
        assert m["status"] in {STATUS_EST, STATUS_NI, STATUS_ABS}


def test_attribute_reads_only_the_data_file(tmp_path: Path):
    """The estimator gets rows only: no params, no truth, no simulator state."""
    rows = simulate(battery.load("inert.yaml"), 1, 200)
    rep = attribute(rows, n_boot=20, n_perm=20)
    assert rep["n_steps"] == 200
