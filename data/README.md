# Benchmark data

All seven benchmark inputs used by the released loaders are included locally.
No cell rows were removed during release preparation. PCA exports retain all
100 components exactly, allowing the dimensional variants to share one file.
The export removes unused expression matrices and unused metadata; it does not
rerun PCA, PHATE or FLE. The original preprocessing/embedding fit is inherited
from the development data, while per-experiment standardization is fit only
on the selected training snapshots.

| Benchmark | Included file | Training input | Provenance |
| --- | --- | --- | --- |
| Ocean | `oceans/oceans.npz` | 9 × 111 × 2 positions | HYCOM-derived benchmark cited by PACE Appendix A; velocity array is evaluation-only |
| Two-Branch | `toy_bifurcation/toy_bifurcation.npz` | Six 2D snapshots | Generated in-house; [generator](../src/data_preprocess/generate_two_branch_data.py) |
| EB PHATE / PCs | `eb_velocity_v5.npz` | 16,819 cells; 2D PHATE / 100 PCs | PHATE embryoid body data; `delta_embedding` is evaluation-only |
| Schiebinger2019 | `scnode/wot_fle/wot_fle.h5ad` | 87,114 rows, two FLE coordinates | Serum-filtered WOT/scNODE input; experiments select integer days |
| iPSC-Liu | `embeddings/ipsc_liu.h5ad` | 28,887 × 100 PCs | Liu et al. reprogramming dataset; compact export of development PCA file |
| OP-Cite | `embeddings/op_cite.h5ad` | 31,240 × 100 PCs | Open Problems CITE-seq, processed data by Alexander Tong |
| OP-Multi | `embeddings/op_multi.h5ad` | 33,701 × 100 PCs | Open Problems Multiome, processed data by Alexander Tong |

## Sources and attribution

The [paper's Appendix A](https://arxiv.org/html/2605.18587v2#A1) describes the
benchmark representations and splits, with the primary dataset references.

- **OP-Cite/OP-Multi:** [Processed Single-cell RNA Time Series Data](https://data.mendeley.com/datasets/hhny5ff7yj/1),
  Alexander Tong, version 1 (2023), DOI `10.17632/hhny5ff7yj.1`, CC BY 4.0.
  Changes for this release: retain existing PCA scores, row IDs and day labels;
  remove expression and other metadata. The original source is the
  [Open Problems multimodal challenge](https://www.kaggle.com/competitions/open-problems-multimodal).
- **EB:** [PHATE / TrajectoryNet data](https://github.com/KrishnaswamyLab/TrajectoryNet/tree/master/data).
  The supplied development NPZ also includes the velocity reference used in PACE.
- **Schiebinger2019:** WOT data as distributed with scNODE. The conversion script
  [scnode_wot.py](../src/data_preprocess/scnode_wot.py) records the source package
  URL and serum filter. Its optional download is large; it is unnecessary for
  running the bundled input.
- **iPSC-Liu:** Liu et al., *Reprogramming roadmap reveals route to human induced
  trophoblast stem cells*, Nature 586, 101–107 (2020), as cited in the PACE paper.
- **Ocean:** Shen et al. / CURLY benchmark as cited in PACE Appendix A.
- **Two-Branch:** synthetic data created for this project; code and generator
  are included under the repository license.

The repository's MIT license applies to code. Upstream data retain their
original terms. The development directory did not contain separate source
license/provenance records for every derived biological NPZ/H5AD; those
remaining attribution details are listed in the local release review.

## Verify or rebuild the compact exports

```bash
python scripts/verify_data.py
python scripts/export_embeddings.py /path/to/iPSC_liu_pca.h5ad \
  /path/to/new_ipsc_liu.h5ad --time-column timepoint
```

The exporter refuses to overwrite an existing file. It asserts exact equality
of the PCA array and selected observation metadata after an H5AD round trip,
and records source/output SHA-256 hashes in a `.provenance.json` sidecar.
[`manifest.json`](manifest.json) verifies all bundled inputs.
