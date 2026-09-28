# Resolved historical configurations

These 35 YAML files record the configurations associated with the recovered
30 main PACE/MFM paper runs and five WOT ablations. They are independent of the
Hydra experiment presets, so later changes to inherited defaults do not alter
these records. Numeric training parameters are preserved; data paths and public
method names are normalized.

See [the recovery report](../docs/recovered-results.md) for checkpoint hashes,
replayed metrics, OT convergence diagnostics and commands. Stage 2 checkpoints
and historical Stage 1 weights are **not** distributed here. Seven PACE recipes
require `PACE_STAGE1_CHECKPOINT` for training; Stage 2 replay does not need it.
Use `configs_hydra/experiment/` for portable training from scratch.
