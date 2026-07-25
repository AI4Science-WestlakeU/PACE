"""Preprocess Morris/Biddy CellTag h5ad into PACE-compatible NPZ.

Source:
    /wangchuanrui2/data/2flow/morris_celltag/cellrank/reprogramming_morris.h5ad

Output keys:
    positions      : (N, 2) standardized X_diff embedding
    timepoints     : (N,) integer reprogramming_day
    clone_ids      : (N,) CellTag clone ids (NaN for unbarcoded)
    cell_type      : (N,) cell type labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad-path",
        type=str,
        default="/wangchuanrui2/data/2flow/morris_celltag/cellrank/reprogramming_morris.h5ad",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/wangchuanrui2/code/references/maincode/PACE_rebuttal/data/morris_celltag.npz",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="X_diff",
        help="Key in adata.obsm to use as the 2-D embedding.",
    )
    parser.add_argument(
        "--clone-column",
        type=str,
        default="CellTagD3_85k",
        help="Column in adata.obs with clone barcode labels.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="85k",
        help="Use '85k' timecourse subset or 'all'.",
    )
    return parser.parse_args()


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    centered = x - x.mean(axis=0, keepdims=True)
    return centered / (centered.std(axis=0, keepdims=True) + 1e-6)


def main() -> None:
    args = parse_args()
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad_path, backed="r")
    obs = adata.obs.copy()

    if args.subset == "85k":
        mask = obs["timecourse"].notna().to_numpy()
    else:
        mask = np.ones(len(obs), dtype=bool)

    obs = obs.loc[mask].copy()
    embedding = np.asarray(adata.obsm[args.embedding_key])[mask].astype(np.float32)

    finite = np.isfinite(embedding).all(axis=1)
    obs = obs.loc[finite].copy()
    embedding = embedding[finite]

    day = pd.to_numeric(obs["reprogramming_day"], errors="coerce").astype(int).to_numpy()

    coords = standardize(embedding)

    clone_ids = obs[args.clone_column].to_numpy()
    # Convert numeric/NaN clone ids to a uniform string format.
    clone_ids = np.array(
        [f"clone_{int(float(x))}" if pd.notna(x) else "unassigned" for x in clone_ids],
        dtype=object,
    )

    cell_type = obs["cell_type"].to_numpy(dtype=object)

    np.savez(
        out,
        positions=coords,
        timepoints=day,
        clone_ids=clone_ids,
        cell_type=cell_type,
        fate_labels=cell_type,
    )
    print(f"Saved {len(coords)} cells to {out}")
    print(f"Timepoints: {sorted(np.unique(day))}")
    print(f"Day counts: {dict(zip(*np.unique(day, return_counts=True)))}")
    valid = np.array([str(c).startswith("clone_") for c in clone_ids])
    print(f"Barcoded cells: {valid.sum()} / {len(valid)}")
    print(f"Unique clones: {len(np.unique(clone_ids[valid]))}")


if __name__ == "__main__":
    main()
