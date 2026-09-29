# From the paper to the implementation

Reference: [PACE, arXiv:2605.18587v2](https://arxiv.org/html/2605.18587v2).
`pace` implements PACE; `mfm` implements the Metric Flow Matching baseline.

| Paper component | Implementation | Configuration |
| --- | --- | --- |
| Local PCA and normal projectors; state/time weighting | `src/methods/pace/geometry.py` | `k_neighbors_local`, `geometry_window_segments`, `metric_alpha` |
| Endpoint-preserving bridge (Eq. 6) | `src/methods/pace/interpolant.py` | `hidden_dims_geopath`, `activation_geopath` |
| Bridge action and optional regularizers (Eqs. 7–8) | `src/methods/pace/psi_train.py` | `lambda_metric`, `lambda_ortho`, `lambda_global`, `T_steps` |
| Coupling refinement (Eqs. 9–10) | `src/methods/pace/matching.py` | `matching_method`, `rematch_every`, `t_probe_steps` |
| Bridge velocity → global velocity (Eq. 11) | `src/methods/pace/flow_matcher.py`, `src/methods/mfm/flow_train.py` | `use_stage1_matching`, `epochs`, `flow_lr` |
| Held-out marginal reconstruction | `src/methods/mfm/eval_metrics.py` | train/test labels in data configs |

The bridge is

$$\gamma_\theta(x,y,s)=(1-s)x+sy+s(1-s)\psi_\theta(x,y,s).$$

The factor $s(1-s)$ fixes both endpoints. The implementation differentiates
with respect to **local** time $s$, then divides by the segment duration when
creating physical-time velocity targets. `sigma` optionally adds training noise
around the bridge; it does not move the deterministic bridge endpoints.

PACE's metric has the form $G=I+\alpha C_N$. Its action charges more for motion
normal to locally supported geometry. The finite implementation uses
mini-batches, neural optimization, a sampled time grid, and periodic rematching;
it does not solve the ideal variational problem exactly.

The current Stage 1 anchor builder trims the selected training snapshots to their
smallest population size. Sinkhorn/EMD/UOT plans are converted to index matchings
using the configured `matching_postprocess`; this is not a full soft-coupling
sampler. Optional candidate pruning is an approximation for larger experiments.

MFM uses its LAND support metric and a learned geodesic interpolant. Both methods
share network, batching, ODE and evaluation utilities in `src/methods/mfm/`.
This shared location does not turn PACE's tangent metric into the MFM metric.

Only training snapshots fit the StandardScaler or provide Stage 1 anchors.
No RNA velocities, simulator velocities, latent branches, particle identities,
or cross-time labels supervise either model. The inferred trajectories are
population-level reconstructions; unpaired snapshots do not identify true
individual-cell lineages.
