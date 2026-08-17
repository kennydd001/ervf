
# Phase 4 follow-up — layer-selective weight conversion

This phase opens only after QFAST and MAMBA smoke fidelity are known.

Calibration and held-out prompts are disjoint. Calibration may rank:

- each of six attention-Q conversions;
- four frozen contiguous groups of Mamba layers;
- then individual Mamba layers only inside groups that dominate loss.

The ranking objective is Pareto, not one scalar invented after the data:

- mean CE delta;
- mean coarse KL;
- per-domain worst case;
- greedy first divergence;
- independently measured latency contribution.

A portfolio is frozen before held-out evaluation. Held-out gates remain the
phase-3 V18-fidelity gates. No matrix may be removed from the portfolio after
held-out results are visible.
