# Recovered experiment results

The completed archives cover all seven datasets, both released methods, all
30 main paper runs, and five WOT ablations. We re-evaluated their Stage 2
checkpoints on the bundled data on 2026-09-28; **no retraining was needed**.
Twenty-seven main runs match the paper's MMD/W1/W2 entries at four decimal
places. The three remaining runs differ by at most 1.464% per metric:

| Run | Largest relative difference across its reported metrics |
| --- | ---: |
| OP-Cite 100D PACE | 1.143% |
| OP-Cite 100D MFM | 1.464% |
| OP-Multi 100D PACE | 0.907% |

This is a recovery of historical models, not an independent training-from-scratch
reproduction. The checkpoint files remain in the research archive and are not
bundled in Git. [The result index](recovered-results.json) contains checkpoint
SHA-256 hashes, data/config hashes, training-log identifiers, individual scores,
sample counts and evaluator warnings. [The CSV](recovered-metrics.csv) provides
paper, replay and OT-convergence values in separate columns.

## Configurations

- `configs_hydra/experiment/` contains portable presets that train from scratch.
  Recovered epoch counts and learning rates are applied to the relevant presets.
- [`configs_archive/`](../configs_archive/) contains 35 resolved historical
  configurations, including the five ablations. Their numeric parameters come
  from the checkpoint-linked Hydra logs; Ocean MFM uses its result-directory
  config, independently checked by checkpoint replay. Data paths and method
  names are normalized for this release.
- Seven historical PACE runs loaded a saved Stage 1 model: WOT, the four iPSC
  50D holdouts, and the two OP 100D settings. Those archived configs explicitly
  require `PACE_STAGE1_CHECKPOINT` for training. **A fresh Stage 1 is not a claim
  of identical historical weights or matchings.** The inherited loader rebuilds
  its initial Hungarian coupling because old checkpoints did not store the
  final alternating matching state.

The main corrections are iPSC 10D holdout 4 (100 bridge epochs, 12 neighbors),
iPSC 50D flow epochs (1000/300/550/650 for holdouts 4/8/12/16, with flow LR
0.0005 for holdout 4 and 0.001 for the others), OP PACE (1000 flow epochs),
and OP-Cite MFM (340 flow epochs). Other paper training parameters are retained.
The README demonstration remains a separate, previously feedback-tuned run.

## Replay a model

Obtain the trusted checkpoint identified by its hash in the result index, then:

```bash
python scripts/evaluate_checkpoint.py \
  --config configs_archive/oceans_positions_pace_ode.yaml \
  --checkpoint /path/to/flow-epoch=099-v3.ckpt \
  --output results/ocean-replay.json
```

The evaluator loads an explicit Stage 2 checkpoint; it does not choose the
newest file in a shared directory. It uses the same seeded data selection,
training-only standardization, observed brackets, and 101-point Euler grid as
the historical evaluation. It computes MMD, W1 and W2 on all configured frame
samples. Stage 1 weights are unnecessary for this inference command.

For a recovered configuration without a saved Stage 1 dependency, training is
also available directly:

```bash
python train.py --config-path configs_archive \
  --config-name oceans_positions_pace_ode
```

For a historical warm-start recipe, supply the corresponding trusted bridge
checkpoint rather than silently substituting a fresh model:

```bash
PACE_STAGE1_CHECKPOINT=/path/to/historical-stage1.ckpt \
python train.py --config-path configs_archive \
  --config-name ipsc_liu_pace_ode_50d_test4
```

The archive does not distribute those historical bridge weights or certify
that their original mutable `last.ckpt` paths still identify the same file.
Use the regular `experiment=...` presets for a self-contained fresh training run.

## OT convergence limitation

All 20 high-dimensional paper runs reproduce a `numItermax` warning with the
historical POT iteration limit (100,000). Matching a published score therefore
**does not certify an optimal transport solution**. One concrete example is
MFM, iPSC 10D, holdout 16: historical W1/W2 are approximately 2.7539/2.6699,
violating W1 ≤ W2 for the same distributions. Increasing the iteration limit
without changing data or weights gives approximately 2.7182/3.1363.

All 20 high-dimensional runs completed the convergence diagnostic with
1,000,000 OT iterations and no OT nonconvergence warning. The diagnostics use
the same predicted/reference clouds and are recorded separately. They do not overwrite
the paper values or the historical replay values, and no model was selected
using these diagnostic scores. To repeat one:

```bash
python scripts/evaluate_checkpoint.py \
  --config configs_archive/ipsc_liu_mfm_ode_10d_test16.yaml \
  --checkpoint /path/to/flow-epoch=349.ckpt \
  --ot-max-iter 1000000 \
  --output results/ipsc10-t16-convergence.json
```

The training runner retains the historical evaluator for compatibility. Use
the explicit larger-budget evaluation when interpreting high-dimensional OT
scores, and check the returned warnings. This audit does not re-evaluate GW,
RNA-velocity alignment, multi-seed statistics, or the unreleased baselines.

## Recovery decisions and coverage

- Ocean PACE's paper model was recovered from a backup. Later runs in the active
  directory had different scores.
- EB and WOT top-level configs had been overwritten by subsequent experiments.
  Their completed baseline checkpoints were linked to the original training
  logs; WOT's baseline was not the later spatial-only ablation.
- Earlier iPSC 2D PACE/MFM checkpoints reproduce the published table. The newest
  files in those directories do not.
- All five WOT ablations were found in separate output directories and replayed.
- The additional EB PCs, OP low-dimensional and WOT forecasting presets remain
  available as exploratory configurations. Missing archives for those variants
  were not filled by new training; they are outside this numerical verification.
