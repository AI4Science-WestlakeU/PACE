#!/usr/bin/env bash
# Long-running PACE hyperparameter sweep on LARRY to maximize GPU utilization.
# Each experiment runs sequentially on the A100.

set -euo pipefail

PROJECT_ROOT="/wangchuanrui2/code/references/maincode/PACE_rebuttal"
source /opt/conda/etc/profile.d/conda.sh
conda activate sc
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/results/larry_sweep_logs"
mkdir -p "$LOG_DIR"

# Experiment 1: Baseline reproduction (already done, re-run for consistency with longer epochs)
python train.py experiment=larry_pca2_pace_ode \
    samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    metric_alpha=8.0 k_neighbors_local=32 \
    rematch_every=20 \
    ablation_name=exp1_baseline_long \
    > "$LOG_DIR/exp1_baseline_long.log" 2>&1

# Experiment 2: Larger network + longer training
python train.py experiment=larry_pca2_pace_ode \
    samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    hidden_dims_geopath=[128,256,256,128] hidden_dims_flow=[128,256,256,128] \
    metric_alpha=8.0 k_neighbors_local=32 \
    rematch_every=20 \
    ablation_name=exp2_large_net \
    > "$LOG_DIR/exp2_large_net.log" 2>&1

# Experiment 3: Softer geometry prior (lower alpha) + more neighbors
python train.py experiment=larry_pca2_pace_ode \
    samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    metric_alpha=2.0 k_neighbors_local=50 \
    rematch_every=20 \
    ablation_name=exp3_soft_geom \
    > "$LOG_DIR/exp3_soft_geom.log" 2>&1

# Experiment 4: Stronger geometry prior
python train.py experiment=larry_pca2_pace_ode \
    samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    metric_alpha=16.0 k_neighbors_local=25 \
    rematch_every=20 \
    ablation_name=exp4_strong_geom \
    > "$LOG_DIR/exp4_strong_geom.log" 2>&1

# Experiment 5: Sinkhorn matching instead of Hungarian
python train.py experiment=larry_pca2_pace_ode \
    samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    model/stage1=pace_sinkhorn sinkhorn_reg=0.05 \
    metric_alpha=8.0 k_neighbors_local=32 \
    rematch_every=20 \
    ablation_name=exp5_sinkhorn \
    > "$LOG_DIR/exp5_sinkhorn.log" 2>&1

# Experiment 6: Use Stage 1 matchings in Stage 2
python train.py experiment=larry_pca2_pace_ode \
    samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    use_stage1_matching=true optimal_transport_method=None \
    metric_alpha=8.0 k_neighbors_local=32 \
    rematch_every=20 \
    ablation_name=exp6_s2_s1match \
    > "$LOG_DIR/exp6_s2_s1match.log" 2>&1

# After all experiments, run barcode/fate evaluation for each
for exp_dir in results/larry_pca2_dim2_test1/exp*; do
    if [ -d "$exp_dir" ]; then
        python scripts/eval_larry_barcode.py \
            --results-dir "$exp_dir" \
            --data-path /wangchuanrui2/code/2flow/experiments/EXP-20260618-02/data/larry_pca2.npz \
            > "$exp_dir/eval.log" 2>&1
    fi
done

echo "LARRY sweep completed."
