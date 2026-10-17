"""
avt.capital — the CAPITAL OPERATOR (design: docs/avt_adaptive_operator.md).

Phase 1 = the operator's BODY + IMAGINATION (machinery + Simulation World), NOT the reasoning agent (that is the
capstone, built last — capital errors multiply, so grounding comes first). This package is SYSTEM-level and
DECOUPLED from per-user AVT (reuses the cascade_forward pattern; never the user AVT/DB).

  machinery   — deterministic primitives: sizing kernels, lot/margin/leverage, bounded compounding, real F&O cost.
  metrics     — honest bounded CAGR / maxDD / MAR-Calmar / utilization (replaces the equity-scaling fantasy).
  sim_config  — the configurable settings surface (capital, charges, compounding, sizing, concurrency, caps).
  simworld    — v0 Simulation World: simulate(action) -> outcome distribution + confidence + sample size.
  calibration — the Simulation World's OWN validation (is its confidence trustworthy?).
"""
