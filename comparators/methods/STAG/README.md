# STAG resources used by Lin-MNP

The manuscript adapter is implemented in
[`experiments/benchmarks/train_additional_comparators.py`](../../../experiments/benchmarks/train_additional_comparators.py)
and does not import the upstream STAG training repository. The adapter retains
the released effective image-prediction and loss path while applying the common
benchmark folds and target panel; the exact adaptation is recorded in each run
metadata file.

STAG uses the Ciga *et al.* self-supervised ResNet18 checkpoint. Download it
before running the adapter:

```bash
mkdir -p comparators/methods/STAG/2D/weights
curl -L \
  https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt \
  -o comparators/methods/STAG/2D/weights/tenpercent_resnet18.ckpt
```

The checkpoint is not redistributed in this repository. The STAG publication
and its original resources remain subject to their authors' terms:

- *STAG: Biologically Guided Spatial Transcriptomics Prediction via Hypergraph
  Learning*, Medical Image Analysis (2026),
  <https://doi.org/10.1016/j.media.2026.104206>.
