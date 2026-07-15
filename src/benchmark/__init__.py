"""PACT v1 (Metrics 2.0) — see docs/pact_v1_spec.md.

Pipeline: registry → generate (LLM-authored scenario packs, batched) →
items (frozen item set) → runner (two-turn, two-arm trials) → judges/awareness
(post-hoc labels) → aggregate (the six axes + rollup).
"""
