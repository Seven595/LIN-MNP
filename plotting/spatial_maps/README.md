# Spatial-expression maps

`build_cscc_spatial_expression_overviews.py` rebuilds the cSCC spatial maps
from Lin-MNP predictions and archived per-section metrics.

`build_hest_stimage_spatial_expression_overviews.py` applies the same visual
design to the HEST and STImage 50-gene benchmarks, using one deterministic
seed-1/fold-0 held-out sample per cancer group or organ.

Run from the repository root after configuring the paths described in the main
README:

```bash
python plotting/spatial_maps/build_cscc_spatial_expression_overviews.py \
  --output plotting/spatial_maps --dpi 220 --workers 4

python plotting/spatial_maps/build_hest_stimage_spatial_expression_overviews.py \
  --output plotting/spatial_maps --dpi 220
```

Both builders validate prediction identities and metric provenance before
rendering figures.
