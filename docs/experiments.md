# Running the experiment settings

The [README matrix](../README.md#experiments-at-a-glance) maps paper names to
config names. Run from the repository root; use `trainer=cpu` for a CPU run.
The default trainer uses one GPU.

## Paper suite

```bash
python scripts/run_paper.py --method both       # print only
python scripts/run_paper.py --method pace --run # train 15 runs
python scripts/run_paper.py --method mfm --run  # train 15 runs
```

Five 2D experiments + eight single-holdout iPSC experiments + two 100D OP
experiments = 15 runs per method. The two iPSC dimensions each aggregate four
holdouts, accounting for nine reported benchmark settings.
`--seed` isolates suite runs in a seed-specific output directory. These commands train from scratch with corrected numerical presets. For the
35 resolved historical configurations, replayed scores, and seven Stage 1
warm-start dependencies, see [the recovery report](recovered-results.md).
A fresh run is not guaranteed to reproduce an archived checkpoint exactly.

## Dimensional and supplementary variants

All inherited PACE/MFM variants on the seven benchmark datasets are retained:
EB PCs; iPSC 5D/10D/50D/100D; OP-Cite 5D/100D; OP-Multi
5D/10D/50D/100D; and 2D WOT forecasting. These are additional configurations,
not additional claims about the main paper tables. Read the resolved labels:
some exploratory OP-Multi variants hold out 4 instead of 3.

```bash
python train.py experiment=ipsc_liu_pace_ode_50d_test12 --cfg job --resolve
python train.py experiment=op_multi_mfm_ode_100d --cfg job --resolve
```

The WOT input contains only two FLE coordinates. Development configs requesting
5D/10D from that file were excluded because they cannot supply those dimensions.
Forecasting labels lie outside the training interval, so the bracketed
`ode_rollout` evaluator cannot score them; consult `full_trajectory_rollout`
diagnostics instead. Forecasting is not part of the paper suite.

## Ablations and matching backends

Five Schiebinger2019 PACE ablations are under `experiment/ablations_pace/`:

| Config suffix | Change |
| --- | --- |
| `alpha0` | Set metric normal penalty to zero |
| `no_rematch` | Disable periodic coupling refinement |
| `knn_all` | Estimate geometry using all available neighbors |
| `metric_spatial` | Drop temporal kernel weighting |
| `metric_temporal` | Drop spatial kernel weighting |

```bash
python train.py experiment=ablations_pace/wot_pace_ode_no_rematch
python train.py experiment=oceans_positions_pace_ode model/stage1=pace_hungarian
```

`pace_hungarian`, `pace_sinkhorn`, `pace_emd`, and `pace_uot` are separate Stage 1
backends. The experiment defaults select the backend; printing `--cfg job
--resolve` is the safest way to check effective settings. A literal YAML key
`model/stage1: ...` does not select a Hydra config group and was removed from the
imported templates.

## Validation

The `wot_fle_reprogramming_demo` preset trains the current README model; see
[the biological demo notes](development-demo.md). It keeps the WOT split and
losses, uses a Stage 1 minibatch of 128, Hungarian matching, sigma 0.02, and
1,000 flow epochs. This feedback-tuned demonstration is separate from the
paper preset; [both refinement trials](reprogramming-refinement.json) are recorded.

The independent `oceans_positions_pace_demo` preset reproduces the training
settings used for the earlier Ocean animation. It is excluded from `run_paper.py` and
was selected using holdout feedback; see [the tuning report](ocean-demo-tuning.md)
for its full-trajectory and bracketed evaluation results.

```bash
python -m pip install -r requirements-dev.txt
python scripts/verify_data.py
python -m pytest -q
python scripts/smoke_test.py
```

The smoke test actually trains both stages for both methods, exercises a PACE
rematch, checks saved checkpoints and finite reconstruction metrics, and uses a
separate output directory. It does not assess scientific convergence.
