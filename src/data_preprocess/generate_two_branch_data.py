from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SEED = 7
DEFAULT_N_TIMEPOINTS = 6
DEFAULT_CELLS_PER_TIMEPOINT = 64
DEFAULT_TIME_CENTERS = np.array([0.14, 0.30, 0.46, 0.60, 0.74, 0.88])
COLORS = {
    "time0": "#356AC3",
    "time1": "#18A6A6",
    "time2": "#E49A2C",
    "manifold": "#D7DDE7",
    "upper": "#C23A32",
    "lower": "#2458C7",
    "trunk": "#627084",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#D8DEE8",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def backbone(
    u: np.ndarray,
    branch: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-branch manifold from the notebook prototype."""
    u = np.asarray(u, dtype=float)
    branch_arr = np.broadcast_to(np.asarray(branch, dtype=float), u.shape)
    x = 0.15 + 4.45 * u
    y = 0.9 + 0.42 * np.sin(2.05 * np.pi * (u - 0.06))
    mask = u > 0.55
    y = y.copy()
    y[mask] += branch_arr[mask] * (
        1.82 * (u[mask] - 0.55) + 0.08 * np.sin(18.0 * u[mask])
    )
    return x, y


def sample_snapshot(
    *,
    center: float,
    spread: float,
    n: int,
    branch_bias: float,
    rng: np.random.Generator,
    noise: tuple[float, float] = (0.075, 0.075),
    balance_late_branches: bool = True,
) -> dict[str, np.ndarray]:
    if n <= 0:
        raise ValueError("snapshot size n must be positive")
    if spread <= 0.0:
        raise ValueError("snapshot spread must be positive")
    if not 0.0 <= branch_bias <= 1.0:
        raise ValueError("branch_bias must be in [0, 1]")

    u = np.clip(rng.normal(center, spread, n), 0.04, 0.96)
    branch = np.zeros(n, dtype=np.int32)
    late = u > 0.55
    late_count = int(late.sum())
    if balance_late_branches and late_count > 0:
        late_branch = np.empty(late_count, dtype=np.int32)
        late_branch[: late_count // 2] = 1
        late_branch[late_count // 2 :] = -1
        rng.shuffle(late_branch)
        branch[late] = late_branch
    else:
        branch[late] = np.where(rng.random(late_count) < branch_bias, 1, -1)

    clean_x, clean_y = backbone(u, branch)
    clean = np.column_stack([clean_x, clean_y])
    observed = clean + rng.normal(0.0, noise, size=clean.shape)

    return {
        "pseudotime": u.astype(np.float64),
        "branch": branch,
        "clean_x": clean[:, 0].astype(np.float64),
        "clean_y": clean[:, 1].astype(np.float64),
        "x": observed[:, 0].astype(np.float64),
        "y": observed[:, 1].astype(np.float64),
    }


def default_snapshot_specs(
    *,
    n_timepoints: int = DEFAULT_N_TIMEPOINTS,
    cells_per_timepoint: int = DEFAULT_CELLS_PER_TIMEPOINT,
) -> list[dict[str, float | int | str]]:
    if n_timepoints <= 0:
        raise ValueError("n_timepoints must be positive")
    if cells_per_timepoint <= 0:
        raise ValueError("cells_per_timepoint must be positive")

    if n_timepoints == DEFAULT_N_TIMEPOINTS:
        centers = DEFAULT_TIME_CENTERS.copy()
    else:
        centers = np.linspace(0.12, 0.88, n_timepoints)

    specs: list[dict[str, float | int | str]] = []
    for index, center in enumerate(centers):
        spread = 0.055 if center < 0.55 else 0.065
        specs.append(
            {
                "label": f"t{index}",
                "center": float(center),
                "spread": spread,
                "n": int(cells_per_timepoint),
                "branch_bias": 0.55,
            }
        )
    return specs


def parse_snapshot_specs(
    value: str | None,
    *,
    n_timepoints: int = DEFAULT_N_TIMEPOINTS,
    cells_per_timepoint: int = DEFAULT_CELLS_PER_TIMEPOINT,
) -> list[dict[str, float | int | str]]:
    if value is None:
        return default_snapshot_specs(
            n_timepoints=n_timepoints,
            cells_per_timepoint=cells_per_timepoint,
        )

    specs: list[dict[str, float | int | str]] = []
    for index, chunk in enumerate(value.split(";")):
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "Each snapshot spec must be center,spread,n,branch_bias; "
                f"got {chunk!r}."
            )
        center, spread, n, branch_bias = parts
        specs.append(
            {
                "label": f"t{index}",
                "center": float(center),
                "spread": float(spread),
                "n": int(n),
                "branch_bias": float(branch_bias),
            }
        )
    return specs


def generate_two_branch_dataset(
    *,
    snapshot_specs: list[dict[str, float | int | str]] | None = None,
    seed: int = DEFAULT_SEED,
    noise: tuple[float, float] = (0.075, 0.075),
    balance_late_branches: bool = True,
) -> np.ndarray:
    if snapshot_specs is None:
        snapshot_specs = default_snapshot_specs()

    rng = np.random.default_rng(seed)
    dtype = [
        ("time_idx", "i4"),
        ("time_label", "U16"),
        ("cell_id", "U32"),
        ("pseudotime", "f8"),
        ("branch", "i4"),
        ("clean_x", "f8"),
        ("clean_y", "f8"),
        ("x", "f8"),
        ("y", "f8"),
    ]

    chunks = []
    for time_idx, spec in enumerate(snapshot_specs):
        label = str(spec["label"])
        n = int(spec["n"])
        snap = sample_snapshot(
            center=float(spec["center"]),
            spread=float(spec["spread"]),
            n=n,
            branch_bias=float(spec["branch_bias"]),
            rng=rng,
            noise=noise,
            balance_late_branches=balance_late_branches,
        )

        arr = np.empty(n, dtype=dtype)
        arr["time_idx"] = time_idx
        arr["time_label"] = label
        arr["cell_id"] = [f"{label}_cell_{i:03d}" for i in range(n)]
        for key in ("pseudotime", "branch", "clean_x", "clean_y", "x", "y"):
            arr[key] = snap[key]
        chunks.append(arr)

    return np.concatenate(chunks)


def dataset_to_frames(dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timepoints = np.unique(dataset["time_idx"])
    positions = np.empty(len(timepoints), dtype=object)
    frame_sizes = np.zeros(len(timepoints), dtype=np.int64)
    for index, timepoint in enumerate(timepoints):
        mask = dataset["time_idx"] == timepoint
        frame = np.column_stack([dataset["x"][mask], dataset["y"][mask]]).astype(
            np.float32
        )
        positions[index] = frame
        frame_sizes[index] = frame.shape[0]
    return positions, timepoints.astype(np.int64), frame_sizes


def save_csv(dataset: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(dataset.dtype.names)
        for row in dataset:
            writer.writerow([row[name] for name in dataset.dtype.names])


def save_npz(
    dataset: np.ndarray,
    path: Path,
    *,
    seed: int,
    noise: tuple[float, float],
    snapshot_specs: list[dict[str, float | int | str]],
    balance_late_branches: bool,
) -> None:
    positions, timepoints, frame_sizes = dataset_to_frames(dataset)
    coordinates = np.column_stack([dataset["x"], dataset["y"]]).astype(np.float32)
    clean_coordinates = np.column_stack([dataset["clean_x"], dataset["clean_y"]]).astype(
        np.float32
    )

    np.savez_compressed(
        path,
        positions=positions,
        timepoints=timepoints,
        times=timepoints,
        frame_sizes=frame_sizes,
        coordinates=coordinates,
        clean_coordinates=clean_coordinates,
        time_idx=dataset["time_idx"],
        time_label=dataset["time_label"],
        cell_id=dataset["cell_id"],
        pseudotime=dataset["pseudotime"].astype(np.float32),
        branch=dataset["branch"],
        clean_x=dataset["clean_x"].astype(np.float32),
        clean_y=dataset["clean_y"].astype(np.float32),
        x=dataset["x"].astype(np.float32),
        y=dataset["y"].astype(np.float32),
        meta=dict(
            dataset="toy_bifurcation",
            generator="src/data_process/generate_two_branch_data.py",
            seed=int(seed),
            noise_x=float(noise[0]),
            noise_y=float(noise[1]),
            snapshot_specs=snapshot_specs,
            balance_late_branches=bool(balance_late_branches),
        ),
    )


def plot_manifold(ax: plt.Axes) -> None:
    u_trunk = np.linspace(0.03, 0.58, 220)
    x, y = backbone(u_trunk, 0.0)
    ax.plot(x, y, color=COLORS["manifold"], lw=3.0, zorder=0)

    for branch in (1.0, -1.0):
        u_branch = np.linspace(0.53, 0.98, 220)
        x, y = backbone(u_branch, branch)
        ax.plot(x, y, color=COLORS["manifold"], lw=3.0, zorder=0)


def format_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")


def save_preview_plot(dataset: np.ndarray, path: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    plot_manifold(axes[0])
    time_colors = {"t0": COLORS["time0"], "t1": COLORS["time1"], "t2": COLORS["time2"]}
    color_cycle = plt.cm.viridis(np.linspace(0.08, 0.92, len(np.unique(dataset["time_label"]))))
    labels = np.unique(dataset["time_label"])
    for index, label in enumerate(labels):
        mask = dataset["time_label"] == label
        color = time_colors.get(str(label), color_cycle[index])
        axes[0].scatter(
            dataset["x"][mask],
            dataset["y"][mask],
            s=28,
            color=color,
            alpha=0.85,
            label=str(label),
        )
    axes[0].legend(frameon=False, loc="best")
    format_axis(axes[0], "Colored by snapshot")

    plot_manifold(axes[1])
    branch_colors = {0: COLORS["trunk"], 1: COLORS["upper"], -1: COLORS["lower"]}
    branch_names = {0: "trunk", 1: "upper branch", -1: "lower branch"}
    for branch_value in (0, 1, -1):
        mask = dataset["branch"] == branch_value
        axes[1].scatter(
            dataset["x"][mask],
            dataset["y"][mask],
            s=28,
            color=branch_colors[branch_value],
            alpha=0.85,
            label=branch_names[branch_value],
        )
    axes[1].legend(frameon=False, loc="best")
    format_axis(axes[1], "Colored by branch")

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    path: Path,
    *,
    npz_name: str,
    csv_name: str,
    png_name: str,
    frame_sizes: np.ndarray,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Two-Branch Toy Dataset",
                "",
                "This directory contains a small two-branch temporal toy dataset generated by",
                "`src/data_process/generate_two_branch_data.py`.",
                "",
                "## Files",
                "",
                f"- `{npz_name}`: compressed NumPy dataset",
                f"- `{csv_name}`: flat table copy of the same cells",
                f"- `{png_name}`: preview colored by snapshot and branch",
                "",
                "## NPZ Keys",
                "",
                "- `positions`: object array of per-timepoint `(n_t, 2)` observed coordinates",
                "- `timepoints` / `times`: integer time labels",
                "- `frame_sizes`: number of points per timepoint",
                "- `coordinates`: flat observed `(x, y)` coordinates",
                "- `clean_coordinates`: noiseless manifold coordinates",
                "- `pseudotime`: latent scalar position along the manifold",
                "- `branch`: `0` trunk, `1` upper branch, `-1` lower branch",
                "",
                "## Current Data",
                "",
                f"- Timepoints: `{list(range(len(frame_sizes)))}`",
                f"- Frame sizes: `{frame_sizes.tolist()}`",
                "- Branching starts after pseudotime `0.55`.",
                "- Post-bifurcation snapshots are balanced across the two branches by default.",
                "",
                "## Regenerate",
                "",
                "```bash",
                "python src/data_process/generate_two_branch_data.py",
                "```",
                "",
                "The default generator writes six equal-size timepoints. Use",
                "`--n-timepoints` and `--cells-per-timepoint` to adjust that default,",
                "or `--snapshot-specs` for fully custom centers and spreads.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a small two-branch temporal toy dataset."
    )
    parser.add_argument("--out-dir", type=str, default="data/toy_bifurcation")
    parser.add_argument("--prefix", type=str, default="toy_bifurcation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-timepoints", type=int, default=DEFAULT_N_TIMEPOINTS)
    parser.add_argument(
        "--cells-per-timepoint",
        type=int,
        default=DEFAULT_CELLS_PER_TIMEPOINT,
    )
    parser.add_argument("--noise-x", type=float, default=0.075)
    parser.add_argument("--noise-y", type=float, default=0.075)
    parser.add_argument(
        "--snapshot-specs",
        type=str,
        default=None,
        help=(
            "Optional semicolon-separated specs: center,spread,n,branch_bias;..."
            " Labels are assigned as t0, t1, ..."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the flat CSV copy.",
    )
    parser.add_argument(
        "--no-balance-late-branches",
        action="store_true",
        help="Use branch_bias sampling instead of enforcing equal branch counts after the split.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_specs = parse_snapshot_specs(
        args.snapshot_specs,
        n_timepoints=args.n_timepoints,
        cells_per_timepoint=args.cells_per_timepoint,
    )
    noise = (float(args.noise_x), float(args.noise_y))

    dataset = generate_two_branch_dataset(
        snapshot_specs=snapshot_specs,
        seed=args.seed,
        noise=noise,
        balance_late_branches=not args.no_balance_late_branches,
    )

    npz_path = out_dir / f"{args.prefix}.npz"
    csv_path = out_dir / f"{args.prefix}.csv"
    preview_path = out_dir / f"{args.prefix}.png"
    save_npz(
        dataset,
        npz_path,
        seed=args.seed,
        noise=noise,
        snapshot_specs=snapshot_specs,
        balance_late_branches=not args.no_balance_late_branches,
    )
    if not args.no_csv:
        save_csv(dataset, csv_path)
    save_preview_plot(dataset, preview_path)

    _positions, _timepoints, frame_sizes = dataset_to_frames(dataset)
    write_readme(
        out_dir / "README.md",
        npz_name=npz_path.name,
        csv_name=csv_path.name,
        png_name=preview_path.name,
        frame_sizes=frame_sizes,
    )

    branch_values, branch_counts = np.unique(dataset["branch"], return_counts=True)
    print(f"saved: {npz_path}")
    if not args.no_csv:
        print(f"saved: {csv_path}")
    print(f"saved: {preview_path}")
    print(f"frame_sizes: {frame_sizes.tolist()}")
    print(f"branch_counts: {dict(zip(branch_values.tolist(), branch_counts.tolist()))}")


if __name__ == "__main__":
    main()
