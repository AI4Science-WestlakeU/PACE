# Reproducibility and release audit

This release is aligned to [arXiv:2605.18587v2](https://arxiv.org/html/2605.18587v2)
for benchmark representations, train/test splits, method equations, and the
held-out evaluation protocol. The [recovery audit](recovered-results.md) replays
all 30 main paper checkpoints
and five WOT ablations. It distinguishes historical metric agreement from
OT convergence and independent retraining.

## What was imported

The development source was at commit
`d8fabaeefe417028ec45ea95055557d27f30f175`, with uncommitted edits to matching,
Stage 1 configs and timepoint sampling. The import includes those working-tree
changes. Only PACE, MFM, their shared helpers, and paper-dataset loaders were
retained. Unrelated baselines, large raw expression matrices, private paths,
training logs and research scratch notebooks were omitted.

The historical checkpoint/config recovery is complete for the seven datasets
and both released methods. The portable presets now use the recovered epoch
counts and learning rates; 35 frozen resolved configs live in
[`configs_archive/`](../configs_archive/). See
[recovered-results.md](recovered-results.md) for the 27 exact rounded matches,
three small OP differences, seven warm-start dependencies, and OT convergence
limitations. No additional training was needed for this recovery.

The original [development snapshot](development-audit.json) is retained as an
initial audit record. It was superseded by the checkpoint-linked recovery:
backup Ocean and earlier iPSC2 models match the paper, and later EB/WOT configs
had overwritten the baseline result directories. The new report uses explicit
checkpoint files rather than the last config or newest file in each directory.

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
numerical equivalence. The complete training workload has not been rerun;
historical Stage 2 models
have instead been independently re-evaluated on the bundled data.

The README animation uses a fresh WOT reprogramming model trained on a remote
H800. Its three panels follow the original observed-anchor holdout protocol;
they do not claim a single continuous rollout across all snapshots. The separate
demo preset uses Stage 1 minibatches of 128, Hungarian matching, sigma 0.02, and
1,000 flow epochs. It was selected using user/holdout feedback after two fresh
refinement trials; it is a tuned demonstration, not an unbiased benchmark.
The paper-suite presets are documented separately in the recovery report. The
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
