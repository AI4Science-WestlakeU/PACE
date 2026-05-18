"""PACE training entry point (Hydra-managed).

Usage:
    # PACE on EB PHATE
    python train.py experiment=eb_phate_pace_ode

    # CPU debug run
    python train.py experiment=eb_phate_pace_ode trainer=cpu epochs=2 total_epochs_stage1=2

    # Print resolved config without running
    python train.py experiment=eb_phate_pace_ode --cfg job
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure PACE/ is on sys.path so `import src.*` works regardless of cwd.
_PACE_ROOT = Path(__file__).resolve().parent
if str(_PACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACE_ROOT))


@hydra.main(config_path="configs_hydra", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    method_name = cfg.get("method", {}).get("name", "")

    project_root = Path(cfg.get("working_dir", ".")).resolve()
    data_name_out = cfg.get("data_name", "unknown")
    dim = cfg.get("dim", None)
    if dim is not None:
        data_name_out = f"{data_name_out}_dim{dim}"
    test_labels = cfg.get("test_timepoint_labels", [])
    if test_labels:
        test_str = "-".join(str(t) for t in sorted(test_labels))
        data_name_out = f"{data_name_out}_test{test_str}"
    output_dir = project_root / "results" / data_name_out / method_name
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    if method_name == "pace":
        from src.methods.pace.runner import run_pace
        run_pace(cfg)
    else:
        raise ValueError(
            f"Unknown method '{method_name}'. Supported methods: pace"
        )


if __name__ == "__main__":
    main()
