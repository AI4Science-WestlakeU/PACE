#!/usr/bin/env bash
# Run SphereRot ablations and evaluate metric recovery / path error.

set -euo pipefail

PROJECT_ROOT="/wangchuanrui2/code/references/maincode/PACE_rebuttal"
source /opt/conda/etc/profile.d/conda.sh
conda activate sc
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/results/sphere_rot_logs"
mkdir -p "$LOG_DIR"

run_exp() {
    local name="$1"
    shift
    echo "============================================"
    echo "SphereRot experiment: $name"
    echo "============================================"
    python train.py experiment=sphere_rot_pace_ode "$@" \
        ablation_name="$name" \
        enable_progress_bar=false \
        > "$LOG_DIR/${name}.log" 2>&1
    python scripts/eval_sphere_rot.py \
        "results/sphere_rot/${name}" \
        --device cuda \
        > "$LOG_DIR/${name}_eval.log" 2>&1
    echo "Finished: $name"
}

# Approximate PACE metric (data-driven local PCA normals).
run_exp pace_approx \
    metric_alpha=8.0 lambda_ortho=1

# Euclidean baseline (alpha = 0, no orthogonality).
run_exp pace_euclidean \
    metric_alpha=0.0 lambda_ortho=0.0

# Transverse noise: observed points are perturbed along the normal direction.
run_exp pace_transverse_noise \
    metric_alpha=8.0 lambda_ortho=1 sphere_noise_sigma=0.05

echo "SphereRot ablations completed. Results are in results/sphere_rot/."
