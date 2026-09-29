# Lin-MNP implementation

[`model.py`](model.py) contains the reference patient-balanced Ridge readout
and one morphology-neighbour propagation step. Its fixed parameters are:

- Ridge regularization: `1e6`
- Neighbours: 64 per other section
- Inverse temperature: 10
- Propagation depth: one step
- Graph fusion: equal Borda ranks from contextual and local descriptors
- Training targets: `log1p(raw counts)` standardized within each section and gene

The predictions retain that standardized expression space. Within-section
gene-wise PCC is equivalent to scoring against unstandardized `log1p(raw
counts)`; MSE is reported in the standardized space.

```bash
lin-mnp train --fold 0 --device cuda:0
lin-mnp evaluate
```

The scripts in [`analysis/`](analysis/) import this implementation directly;
they do not reimplement the model.
