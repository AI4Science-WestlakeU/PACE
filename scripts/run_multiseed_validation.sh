#!/usr/bin/env bash
# Multi-seed statistical validation for the best LARRY and Morris configs.

set -euo pipefail

PROJECT_ROOT="/wangchuanrui2/code/references/maincode/PACE_rebuttal"
source /opt/conda/etc/profile.d/conda.sh
conda activate sc
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/results/sweep_logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local name="$1"
    local data_path="$2"
    local seed="$3"
    shift 3
    echo "============================================"
    echo "Starting experiment: $name (seed=$seed)"
    echo "============================================"
    set +e
    python train.py "$@" \
        ablation_name="${name}_seed${seed}" \
        seed=$seed \
        enable_progress_bar=false \
        > "$LOG_DIR/${name}_seed${seed}.log" 2>&1
    local status=$?
    set -e
    if [ $status -ne 0 ]; then
        echo "ERROR: experiment ${name}_seed${seed} failed with exit code $status"
        return $status
    fi
    local results_dir="results/${data_dir_name}/${name}_seed${seed}"
    if [ -d "$results_dir" ]; then
        python scripts/eval_larry_barcode.py \
            --results-dir "$results_dir" \
            --data-path "$data_path" \
            > "$results_dir/eval.log" 2>&1
    fi
    echo "Finished experiment: ${name}_seed${seed}"
}

SEEDS=(42 123 456 789 1024)

# ============================================================
# LARRY Sinkhorn (best config from first sweep)
# ============================================================
data_dir_name="larry_pca2_dim2_test1"
data_path="/wangchuanrui2/code/2flow/experiments/EXP-20260618-02/data/larry_pca2.npz"
for seed in "${SEEDS[@]}"; do
    run_exp larry_sinkhorn "$data_path" "$seed" \
        experiment=larry_pca2_pace_ode samples_per_timepoint=none \
        total_epochs_stage1=200 epochs=500 \
        model/stage1=pace_sinkhorn sinkhorn_reg=0.05 \
        metric_alpha=8.0 k_neighbors_local=32 rematch_every=20
done

# ============================================================
# Morris soft geometry (best config from first sweep)
# ============================================================
data_dir_name="morris_celltag_dim2_test15"
data_path="/wangchuanrui2/code/references/maincode/PACE_rebuttal/data/morris_celltag.npz"
for seed in "${SEEDS[@]}"; do
    run_exp morris_soft_geom "$data_path" "$seed" \
        experiment=morris_celltag_pace_ode \
        total_epochs_stage1=120 epochs=350 \
        metric_alpha=2.0 k_neighbors_local=50 rematch_every=20
done

echo "Multi-seed validation sweep completed."
