from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGSHARE_URL = "https://ndownloader.figshare.com/files/45647244"
EXPECTED_ZIP_BYTES = 3_756_567_850
DEFAULT_ZIP_PATH = PROJECT_ROOT / "data/scnode/raw/20240414_new_data.zip"
DEFAULT_EXTRACT_DIR = PROJECT_ROOT / "data/scnode/extracted"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/scnode/wot_fle"
SPLITS = ("three_interpolation", "three_forecasting", "remove_recovery")
INTEGER_DAY_LABELS = list(range(19))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the WOT tutorial FLE coordinate h5ad file from the scNODE package.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the scNODE Figshare data package with curl resume support.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the downloaded scNODE package before conversion.",
    )
    parser.add_argument(
        "--split",
        choices=("all", *SPLITS),
        default="all",
        help=(
            "Deprecated compatibility option. The FLE data are shared across "
            "splits, so this script always writes one wot_fle.h5ad file."
        ),
    )
    parser.add_argument(
        "--cell-filter",
        choices=("serum", "2i", "all"),
        default="serum",
        help="Cell filter from the WOT tutorial raw files. Default: serum.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing downloads, extraction output, or converted h5ad files.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help=f"Download target zip path. Default: {DEFAULT_ZIP_PATH}",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=DEFAULT_EXTRACT_DIR,
        help=f"Directory containing the extracted scNODE package. Default: {DEFAULT_EXTRACT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for converted h5ad files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def download_package(zip_path: Path, force: bool) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and zip_path.stat().st_size == EXPECTED_ZIP_BYTES and not force:
        print(f"Download already complete: {zip_path}")
        return
    if zip_path.exists() and force:
        zip_path.unlink()

    cmd = [
        "curl",
        "-L",
        "-C",
        "-",
        "--fail",
        "-o",
        str(zip_path),
        FIGSHARE_URL,
    ]
    print("Downloading scNODE Figshare package with curl...")
    subprocess.run(cmd, check=True)


def extract_package(zip_path: Path, extract_dir: Path, force: bool) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing zip file: {zip_path}. Run with --download first.",
        )

    if not force:
        try:
            raw_dir = find_raw_dir(extract_dir)
            print(f"Extraction already present: {raw_dir}")
            return
        except FileNotFoundError:
            pass

    extract_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["unzip"]
    if force:
        cmd.append("-o")
    cmd.extend([str(zip_path), "-d", str(extract_dir)])
    print(f"Extracting {zip_path} to {extract_dir}...")
    subprocess.run(cmd, check=True)


def find_raw_dir(extract_dir: Path) -> Path:
    expected_root = (
        extract_dir
        / "new_data"
        / "data"
        / "single_cell"
        / "experimental"
        / "Schiebinger2019"
        / "raw"
    )
    if expected_root.exists():
        return expected_root

    for root in sorted(extract_dir.rglob("Schiebinger2019")):
        candidate = root / "raw"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find Schiebinger2019/raw under "
        f"{extract_dir}. Run with --download --extract first if needed.",
    )


def ensure_raw_dir(zip_path: Path, extract_dir: Path, force: bool) -> Path:
    try:
        return find_raw_dir(extract_dir)
    except FileNotFoundError:
        print(
            "WOT raw data not found. Downloading and extracting the scNODE "
            "Figshare package..."
        )

    download_package(zip_path, force=force)
    extract_package(zip_path, extract_dir, force=True)
    return find_raw_dir(extract_dir)


def read_cell_filter(raw_dir: Path, cell_filter: str) -> set[str] | None:
    if cell_filter == "all":
        return None
    filter_path = raw_dir / f"{cell_filter}_cell_ids.txt"
    if not filter_path.exists():
        raise FileNotFoundError(f"Missing WOT cell filter file: {filter_path}")
    cell_ids = pd.read_csv(filter_path, header=None, names=["id"])
    return set(cell_ids["id"].astype(str))


def convert_wot_fle(
    raw_dir: Path,
    output_dir: Path,
    cell_filter: str,
    force: bool,
) -> Path:
    fle_path = raw_dir / "fle_coords.txt"
    days_path = raw_dir / "cell_days.txt"
    output_path = output_dir / "wot_fle.h5ad"

    if output_path.exists() and not force:
        print(f"Converted file already exists: {output_path}")
        return output_path

    missing = [path for path in (fle_path, days_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing WOT tutorial raw files: " + ", ".join(str(path) for path in missing),
        )

    print(f"Reading FLE coordinates: {fle_path}")
    fle = pd.read_csv(fle_path, sep="\t")
    print(f"Reading day labels: {days_path}")
    days = pd.read_csv(days_path, sep="\t")
    allowed_ids = read_cell_filter(raw_dir, cell_filter)

    required_fle = {"id", "x", "y"}
    required_days = {"id", "day"}
    if not required_fle.issubset(fle.columns):
        raise ValueError(f"{fle_path} must contain columns {sorted(required_fle)}.")
    if not required_days.issubset(days.columns):
        raise ValueError(f"{days_path} must contain columns {sorted(required_days)}.")

    df = fle.merge(days, on="id", how="inner")
    if allowed_ids is not None:
        df = df[df["id"].astype(str).isin(allowed_ids)]
    labels = {float(label) for label in INTEGER_DAY_LABELS}
    df = df[df["day"].astype(float).isin(labels)]
    if df.empty:
        raise ValueError(f"No cells left after filtering {cell_filter=}.")

    obs = df[["id", "day"]].copy()
    obs["id"] = obs["id"].astype(str)
    obs["day"] = obs["day"].astype(float)
    obs["cell_filter"] = cell_filter
    obs = obs.set_index("id")
    var = pd.DataFrame(index=pd.Index(["fle_x", "fle_y"], name="coordinate"))
    x = df[["x", "y"]].to_numpy(dtype=np.float32, copy=True)

    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.uns["source"] = "scNODE Figshare data package"
    adata.uns["source_url"] = FIGSHARE_URL
    adata.uns["dataset"] = "Schiebinger2019/WOT"
    adata.uns["representation"] = "WOT tutorial FLE coordinates"
    adata.uns["split"] = "shared_all_integer_days"
    adata.uns["cell_filter"] = cell_filter
    adata.uns["raw_dir"] = str(raw_dir)
    adata.uns["fle_coords"] = str(fle_path)
    adata.uns["cell_days"] = str(days_path)
    adata.uns["timepoint_labels"] = sorted(labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_path} with shape {adata.shape}...")
    adata.write_h5ad(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    if args.download:
        download_package(args.zip_path, force=args.force)
    if args.extract:
        extract_package(args.zip_path, args.extract_dir, force=args.force)

    raw_dir = ensure_raw_dir(
        zip_path=args.zip_path,
        extract_dir=args.extract_dir,
        force=args.force,
    )
    print(f"Using WOT raw directory: {raw_dir}")
    if args.split != "all":
        print(
            f"Ignoring deprecated --split {args.split!r}; "
            "the shared FLE file is used by all WOT split configs."
        )
    output_path = convert_wot_fle(
        raw_dir=raw_dir,
        output_dir=args.output_dir,
        cell_filter=args.cell_filter,
        force=args.force,
    )
    print(f"Ready: {output_path}")


if __name__ == "__main__":
    main()
