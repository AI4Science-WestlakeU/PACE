# Attribution

The MFM implementation and shared training helpers in `src/methods/mfm/`
are adapted from [Metric Flow Matching](https://github.com/kkapusniak/metric-flow-matching)
(Kapusniak et al., NeurIPS 2024). Its MIT license and copyright notice are
preserved in [third_party/metric-flow-matching-LICENSE](third_party/metric-flow-matching-LICENSE).
The EMA helper also retains its existing Meta Platforms attribution.

Conditional-flow and OT utilities use the separately installed
[TorchCFM](https://github.com/atong01/conditional-flow-matching) package.
See [data/README.md](data/README.md) for dataset citations and provenance;
the code's MIT license does not replace the original dataset terms.
