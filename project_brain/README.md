# PTIS Project Brain

The system's memory of **itself** — so we stop rebuilding what already exists.

**Rule of use (for any contributor, human or AI): before building or optimizing ANY capability,
read `01_canonical_owners.md` and `07_do_not_duplicate.md` first.** Find the owner. Use it / fix it /
extend it — do **not** build a parallel version. Update these files when ownership or status changes.

| File | What it answers |
|---|---|
| `01_canonical_owners.md` | Who OWNS each capability + who CONSUMES it. **(load-bearing — check first)** |
| `02_data_flow.md` | The pipeline: how data moves through the layers. |
| `03_module_registry.md` | Every core module → one-line role + layer. |
| `07_do_not_duplicate.md` | trigger → rule. The anti-duplication enforcement. **(load-bearing — check first)** |
| `08_feature_store_contract.md` | the shared perception layer: invariants, tiers, the two-lens warm-up resolution. |
| `09_frozen_core.md` | **Core v1 is FROZEN (bug-fixes only).** The frozen/active boundary + the dogfood roadmap. |

Built 2026-06-28 from a real audit of module docstrings (not assumptions). Keep entries terse.
