"""Attribution over a decode-step region graph (case Part 2).

Three modules, one forward model:

* :mod:`gitm.attribution.model` — the generative model shared by the simulator
  and the estimator: parameter parsing with inert defaults, deep-merge of
  ``conditions[].overrides``, and the deterministic mean map
  (floors, slowdown, gate, serialization, additive) -> (region times, total).
* :mod:`gitm.attribution.simulate` — ``simulate --params --seed --steps --out``.
* :mod:`gitm.attribution.attribute` — ``attribute --data --out``; works from
  ``data.jsonl`` alone and abstains where the data cannot separate mechanisms.

The model, the two non-identifiability proofs, the separating interventions,
and the estimator's decision rule are in
``docs/deepseek-v4-flash/ATTRIBUTION-NOTE.md``.
"""
