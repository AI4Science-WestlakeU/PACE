# PACE — ICLR 2027 Submission (anonymous)

Converted on 2026-09-22 from the PACE NeurIPS 2026 paper source to the official
ICLR 2027 template. **This directory is a local-only working copy — do NOT
commit or push it** (same convention as the rebuttal docs in this repo).

## Provenance

| Item | Source |
|---|---|
| Paper content (`content/`, `table/`, `figures/`) | `~/research/pr/arxiv_extracted` (clean arXiv-state copy of Overleaf project `bczjgfxgdqgjfrmrbgjnnfccswztqpkb`, 2026-05-18). Compiled content is identical to the NeurIPS 2026 submission except where marked `PAGECUT`. |
| Bibliography (`references.bib`) | Overleaf zip `neurips_2026.bib` (43 entries; renamed) |
| Style files (`iclr2027_conference.sty/.bst`, `natbib.sty`, `fancyhdr.sty`) | official `iclr-2027-style-files.zip`, byte-identical to the copies on the `main` branch of `GEOMORPH-paper` (verified with `diff`) |
| Original files | **Nothing outside this directory was modified.** The sources above were copied, not edited. |

## Title

Current title kept: *"PACE: Geometry-Aware Bridge Transport for Single-Cell
Trajectory Inference"*. The NeurIPS 2026 anonymous submission was titled
*"PACE: Pseudotime Trajectory Inference via Spatiotemporal Tangent-Space
Projections"* (kept as a comment in `main.tex`). Change before submitting if
desired.

## ICLR 2027 compliance

- **Main text: 9 pages** (limit: ≤9, strict; conclusion ends on page 9).
  Statements (AI use / Ethics / Reproducibility) start on page 10 — excluded
  from the limit. References pp. 10–13 (unlimited). Appendix pp. 14–32
  (unlimited).
- Anonymous: `\author{Anonymous}`, `\iclrfinalcopy` commented out; acknowledgments
  and the `github.com/AI4Science-WestlakeU` link removed from the compiled PDF.
- **AI use statement: REQUIRED by ICLR 2027 — authors MUST verify/adjust the
  drafted wording in `main.tex` before submitting.**
- Deadline: full paper **2026-09-25 AoE** (abstract deadline 09-18 has passed —
  an abstract must already be registered on OpenReview).

## Edits relative to the NeurIPS/arXiv source (all marked `PAGECUT` in place)

Every edit is local to this copy; the original sentence is preserved in a
comment right next to each change.

1. `content/0_abstract.tex` — code link anonymized to
   `https://anonymous.4open.science/r/PACE-ICLR2027` (**placeholder: create the
   anonymous repo and update the link before submitting**).
2. `content/3_method.tex` — fig1 width `\textwidth` → `0.92\textwidth`; caption tail trimmed.
3. `content/4_experiment.tex` — datasets-overview figure `1.0`→`0.88\linewidth`;
   velocity-diagnostic panels `0.49`→`0.455\linewidth`; concentration/ablation
   panels `0.40/0.58`→`0.335/0.54\linewidth`; removed the manual `\newpage`
   before §4.4; Fig. 3 and Fig. 5 captions compressed; ablation paragraph and
   datasets paragraph tightened; redundant "Overall, ..." sentence in §4.3
   removed; **Table 3 (iPSC-Liu 10D/50D time-averaged) moved to Appendix H**
   as `table/ipsc_liu_dim10_50_mean_moved.tex` (same numbers, same label —
   cross-references auto-update to `Table A.x`).
4. `content/1_introduction.tex` — blank lines between contribution items
   removed; one iPSC sentence tightened.
5. `content/2_related_work.tex` — one trailing clause dropped.
6. `content/5_conclusion.tex` — first two sentences merged; last sentence
   compressed.
7. `content/6_appendix.tex` — inputs the moved table; one cross-reference
   phrase ("summarized in the main text in Table ...") updated accordingly.
8. `main.tex` — rewritten for ICLR 2027 (template, anonymous mode,
   statements); NeurIPS checklist omitted (not required by ICLR).

No equation, number, or result value was changed.

## Build

```bash
cd iclr2027_submission
tectonic main.tex    # single pass incl. bibliography; output: main.pdf
```

## Open items before submission (for the authors)

- [ ] Create the anonymous code repository and replace the placeholder link in
      `content/0_abstract.tex` (and mention it consistently in the
      Reproducibility statement).
- [ ] Verify the **AI use statement** wording reflects actual tool usage.
- [ ] Decide the final title (current vs. the NeurIPS submission title).
- [ ] Optional: restore acknowledgments only for the camera-ready
      (`\iclrfinalcopy`); keep them out of the submission.
