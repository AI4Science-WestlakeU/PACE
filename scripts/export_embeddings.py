#!/usr/bin/env python3
"""Export the exact PCA inputs used by PACE without the raw expression matrix.

Usage: python scripts/export_embeddings.py INPUT.h5ad OUTPUT.h5ad --time-column day
The PCA values, their dtype, row order and cell IDs are preserved. No PCA is refit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import anndata as ad
import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--time-column', required=True)
    args = p.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.exists():
        p.error('Output must be a new file, distinct from the source.')
    source = ad.read_h5ad(args.source, backed='r')
    features = np.asarray(source.obsm['X_pca'])
    obs = source.obs[[args.time_column]].copy()
    compact = ad.AnnData(X=np.empty((len(obs), 0), dtype=np.float32), obs=obs,
                         obsm={'X_pca': features})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compact.write_h5ad(args.output, compression='gzip')
    check = ad.read_h5ad(args.output)
    np.testing.assert_array_equal(check.obsm['X_pca'], features)
    pd.testing.assert_frame_equal(check.obs, obs)
    report = {
        'source_file': args.source.name,
        'source_sha256': sha256(args.source),
        'export_file': args.output.name,
        'export_sha256': sha256(args.output),
        'cells': len(obs), 'pca_dimensions': features.shape[1],
        'pca_dtype': str(features.dtype), 'time_column': args.time_column,
        'timepoint_counts': {str(k): int(v) for k,v in obs[args.time_column].value_counts().sort_index().items()},
        'validation': 'Exact array and obs equality after H5AD round-trip; no rows removed; no PCA refit.',
    }
    source.file.close()
    args.output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
