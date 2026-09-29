# Third-party method code

Trees under `comparators/methods/` are source-only snapshots required by the
benchmark adapters. Weights and data are not included. Comments and terminal
messages were translated or removed where necessary to keep the distribution
English-only. Machine-specific data defaults were replaced with relative paths,
environment prefixes were removed, and the HESpotEx environment export was
reduced to runtime dependencies. Model equations and scientific
hyperparameters were not changed. STAG is the exception: its benchmark adapter
is self-contained and the directory only documents the external encoder
checkpoint.

Each snapshot remains subject to its upstream license; no additional rights
are granted for files whose upstream source has no license. The manifest hashes
the portable files distributed here. Verify them from the repository root with:

```bash
(cd comparators/methods && sha256sum -c MANIFEST.sha256)
```

| Manuscript name | Directory |
|---|---|
| asGNN | `comparators/methods/asGNN` |
| BLEEP | `comparators/methods/BLEEP` |
| DeepPT | `comparators/methods/DeepPT` |
| DeepSpaCE | `comparators/methods/DeepSpaCE` |
| EGNv1 / EGNv2 | `comparators/methods/EGN` |
| GeneCodeR | `comparators/methods/GeneCodeR`, `comparators/methods/gcode_R` |
| GR2ST | `comparators/methods/GR2ST` |
| HE2Gene | `comparators/methods/HE2Gene` |
| HECLIP | `comparators/methods/HECLIP` |
| HESpotEx | `comparators/methods/HESpotEx` |
| HisToGene | `comparators/methods/mclSTExp/baselines/HisToGene` |
| Hist2ST | `comparators/methods/Hist2ST` |
| HoloTea | benchmark implementation in `experiments/benchmarks/train_additional_comparators.py` |
| IGI-DL | `comparators/methods/IGI-DL` |
| iStar | `comparators/methods/iStar` |
| mclSTExp | `comparators/methods/mclSTExp` |
| OmiCLIP | `comparators/methods/OmiCLIP_Loki` |
| SpaHGC | `comparators/methods/SpaHGC` |
| STAG | Self-contained benchmark adapter; checkpoint instructions in `comparators/methods/STAG/README.md` |
| STFlow | `comparators/methods/STFlow` |
| ST-Net | `comparators/methods/ST-Net` |
| TCGN | `comparators/methods/TCGN` |
| THItoGene | `comparators/methods/THItoGene` |
