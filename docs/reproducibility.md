# Reproducibility and release audit

This release is aligned to [arXiv:2605.18587v2](https://arxiv.org/html/2605.18587v2)
for benchmark representations, train/test splits, method equations, and the
held-out evaluation protocol. **It is not yet a certified numerical reproduction
of all published tables.**

## What was imported

The development source was at commit
`d8fabaeefe417028ec45ea95055557d27f30f175`, with uncommitted edits to matching,
Stage 1 configs and timepoint sampling. The import includes those working-tree
changes. Public modules/classes/config names were normalized from `imfm` to
`pace`. Only PACE, MFM, their shared helpers, and paper-dataset loaders were
retained. Unrelated baselines, large raw expression matrices, private paths,
training logs and research scratch notebooks were omitted.

The source configuration templates were retained, except for portable data
paths, Hydra cleanup, and explicit single-holdout iPSC presets. We did **not**
replace them wholesale with current result-directory configs: some of those
files demonstrably describe subsequent experiments rather than the paper run.

## Concrete differences requiring a final paper archive

All 30 paper-suite runs have a current server result-directory config. The
[machine-readable audit](development-audit.json) records the comparison and
available Stage 2 metrics. These records are development evidence, not a new
benchmark leaderboard.

| Item | Current evidence |
| --- | --- |
| Ocean PACE | Current archived MMD at holdouts 1/3/5/7 is approximately 0.7212/0.5619/0.1971/0.4286; paper Table 1 reports 0.4504/0.2588/0.0735/0.2779. |
| EB PACE | The result-directory config contains an `ablation_name` of `pace_current_check`. |
| WOT PACE | The result-directory config identifies `pace_metric_spatial` and spatial-only weighting, despite living under the method's baseline directory. |
| iPSC 10D, holdout 4 | Template: 80 Stage 1 epochs, 16 neighbors; archive: 100 epochs, 12 neighbors. |
| iPSC 50D | Archived flow epoch counts vary by holdout (1000/300/550/650), whereas the template uses 350. Holdout 4 also changes the flow learning rate. |
| OP-Cite/OP-Multi 100D PACE | Archived configs use 1000 flow epochs and load a Stage 1 checkpoint; templates train fresh with 400/351 flow epochs. |
| OP-Cite 100D MFM | Archived config uses 340 flow epochs; template uses 350. |

The final paper checkpoints/configs need to be identified before claiming exact
published scores. Rerunning the current templates cannot establish which
historical run produced a published table. No reported paper score was copied
into a generated run's metrics.

## Evaluation semantics

- Per-experiment StandardScaler statistics use selected training snapshots only.
  The pre-existing PCA/PHATE/FLE embeddings are provided inputs; the release
  does not claim they were originally fitted without all-timepoint data.
- Stage 1 PACE anchors are trimmed to the smallest selected snapshot size.
- Paper-style `stage2 / ode_rollout` uses the nearest bracketing observed training
  snapshots, a normalized training-time scale, and 101 Euler grid points.
- First-snapshot-to-final-time trajectories are a separate diagnostic.
- Reference velocities are used only to evaluate direction/alignment.
- High-dimensional and full-population OT/GW metrics can require substantial
  memory and time. `samples_per_timepoint=null` means the full selected input;
  the inherited Stage 2 evaluator does not apply `eval_sample_cap` to these rows.
- Sinkhorn may warn about nonconvergence with inherited regularization settings.
  The paper-suite templates retain those settings. The separately named Ocean
  demonstration preset has a documented tuning history in
  [ocean-demo-tuning.md](ocean-demo-tuning.md).
- Existing Stage 1 checkpoints do not serialize the final matching list. Loading
  one in the inherited PACE runner rebuilds an initial coupling when needed;
  this is not equivalent to resuming the exact alternating state. The new
  `src.inference.load_run` loads Stage 2 for inference independently of this path.

## Local validation

See [validation.json](validation.json) for the tested environment and completed
checks. Short CPU runs validate software behavior, not convergence or GPU
numerical equivalence. The complete paper workload has not been rerun as part
of release preparation.

The README animation uses a fresh WOT reprogramming model trained on a remote
H800. Its three panels follow the original observed-anchor holdout protocol;
they do not claim a single continuous rollout across all snapshots. The separate
demo preset uses Stage 1 minibatches of 128, Hungarian matching, sigma 0.02, and
1,000 flow epochs. It was selected using user/holdout feedback after two fresh
refinement trials; it is a tuned demonstration, not an unbiased benchmark.
The paper-suite preset remains unchanged. The
[biological demo notes](development-demo.md) describe the training, rendering
protocol, diagnostics, and a secondary embryoid-body example. The earlier
holdout-tuned Ocean example remains in [its archived report](ocean-demo-tuning.md).
All three tutorial notebooks retain their executed CPU tutorial outputs, with
additional instructions for rendering the biological examples.

## Data provenance still to confirm before publication

The OP-Cite/OP-Multi processed source and CC BY 4.0 attribution are linked in
[data/README.md](../data/README.md). Other inherited derived files did not carry
complete preprocessing and redistribution records. Preserve their source
citations and confirm the final source terms/provenance for Ocean, EB,
Schiebinger2019 and iPSC before publishing this local candidate.
