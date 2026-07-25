#!/usr/bin/env bash
# Master PACE sweep: LARRY + Morris CellTag.
# Runs sequentially on the A100 to maximize sustained GPU utilization.

set -euo pipefail

PROJECT_ROOT="/wangchuanrui2/code/references/maincode/PACE_rebuttal"
source /opt/conda/etc/profile.d/conda.sh
conda activate sc
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/results/sweep_logs"
mkdir -p "$LOG_DIR"

MORRIS_DATA="/wangchuanrui2/code/references/maincode/PACE_rebuttal/data/morris_celltag.npz"

run_exp() {
    local name="$1"
    shift
    echo "============================================"
    echo "Starting experiment: $name"
    echo "============================================"
    set +e
    python train.py "$@" \
        ablation_name="$name" \
        enable_progress_bar=false \
        > "$LOG_DIR/${name}.log" 2>&1
    local status=$?
    set -e
    if [ $status -ne 0 ]; then
        echo "ERROR: experiment $name failed with exit code $status"
        return $status
    fi
    local results_dir="results/${data_dir_name}/${name}"
    if [ -d "$results_dir" ]; then
        python scripts/eval_larry_barcode.py \
            --results-dir "$results_dir" \
            --data-path "$data_path" \
            > "$results_dir/eval.log" 2>&1
    fi
    echo "Finished experiment: $name"
}

# ============================================================
# LARRY experiments
# ============================================================
data_dir_name="larry_pca2_dim2_test1"
data_path="/wangchuanrui2/code/2flow/experiments/EXP-20260618-02/data/larry_pca2.npz"

run_exp larry_baseline_long \
    experiment=larry_pca2_pace_ode samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    metric_alpha=8.0 k_neighbors_local=32 rematch_every=20

run_exp larry_large_net \
    experiment=larry_pca2_pace_ode samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    hidden_dims_geopath=[128,256,256,128] hidden_dims_flow=[128,256,256,128] \
    metric_alpha=8.0 k_neighbors_local=32 rematch_every=20

run_exp larry_soft_geom \
    experiment=larry_pca2_pace_ode samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    metric_alpha=2.0 k_neighbors_local=50 rematch_every=20

run_exp larry_strong_geom \
    experiment=larry_pca2_pace_ode samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    metric_alpha=16.0 k_neighbors_local=25 rematch_every=20

run_exp larry_sinkhorn \
    experiment=larry_pca2_pace_ode samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    model/stage1=pace_sinkhorn sinkhorn_reg=0.05 \
    metric_alpha=8.0 k_neighbors_local=32 rematch_every=20

run_exp larry_s2_s1match \
    experiment=larry_pca2_pace_ode samples_per_timepoint=none \
    total_epochs_stage1=200 epochs=500 \
    use_stage1_matching=true optimal_transport_method=None \
    metric_alpha=8.0 k_neighbors_local=32 rematch_every=20

# ============================================================
# Morris CellTag experiments
# ============================================================
data_dir_name="morris_celltag_dim2_test15"
data_path="$MORRIS_DATA"

run_exp morris_baseline \
    experiment=morris_celltag_pace_ode \
    total_epochs_stage1=120 epochs=350 \
    metric_alpha=8.0 k_neighbors_local=32 rematch_every=20

run_exp morris_large_net \
    experiment=morris_celltag_pace_ode \
    total_epochs_stage1=120 epochs=350 \
    hidden_dims_geopath=[128,256,256,128] hidden_dims_flow=[128,256,256,128] \
    metric_alpha=8.0 k_neighbors_local=32 rematch_every=20

run_exp morris_soft_geom \
    experiment=morris_celltag_pace_ode \
    total_epochs_stage1=120 epochs=350 \
    metric_alpha=2.0 k_neighbors_local=50 rematch_every=20

run_exp morris_strong_geom \
    experiment=morris_celltag_pace_ode \
    total_epochs_stage1=120 epochs=350 \
    metric_alpha=16.0 k_neighbors_local=25 rematch_every=20

run_exp morris_sinkhorn \
    experiment=morris_celltag_pace_ode \
    total_epochs_stage1=120 epochs=350 \
    model/stage1=pace_sinkhorn sinkhorn_reg=0.05 \
    metric_alpha=8.0 k_neighbors_local=32 rematch_every=20

run_exp morris_s2_s1match \
    experiment=morris_celltag_pace_ode \
    total_epochs_stage1=120 epochs=350 \
    use_stage1_matching=true optimal_transport_method=None \
    metric_alpha=8.0 k_neighbors_local=32 rematch_every=20

echo "All sweeps completed."
