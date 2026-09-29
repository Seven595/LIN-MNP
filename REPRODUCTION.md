# Reproducing the manuscript

Run all commands from the repository root after installing the required profile
and setting the path variables described in the main
[`README.md`](README.md#data-and-path-configuration).

## Reproducibility levels

The release separates three workflow classes:

1. **Reference method.** `lin-mnp train` creates cSCC outer-test predictions
   from frozen descriptor caches, and `lin-mnp evaluate` verifies and scores
   the four folds.
2. **Published comparators.** Scripts in `comparators/adapters/` place a
   published architecture on a declared fold, spot, gene, and target contract.
   Each run records source hashes, the seed, selected checkpoint, and output
   hashes.
3. **Manuscript aggregation.** Scripts in `experiments/` and `plotting/`
   consume sealed predictions and the companion artifact bundle. They do not
   redefine Lin-MNP.

Large source cohorts, pretrained weights, descriptor matrices, prediction
artifacts, and the companion manuscript bundle are not stored in Git. Workflows
fail early when a required file or provenance field is absent and refuse to
overwrite completed outputs.

## 1. Reference cSCC result

Required inputs are the cSCC label shards and the raw-patch, contextual
CTransPath, and local CTransPath caches documented in
`configs/paths.example.yaml`.

```bash
for fold in 0 1 2 3; do
  lin-mnp train --fold "$fold" --device cuda:0
done
lin-mnp evaluate
```

Use `--device cpu` when CUDA is unavailable. The training phase fits only the
two training patients. It does not open validation or outer-test expression.
The evaluation phase verifies all four prediction artifacts before opening the
test targets and applies equal patient weighting.

Default outputs:

```text
$LIN_MNP_RESULTS_ROOT/cscc/folds/
├── fold0/
├── fold1/
├── fold2/
├── fold3/
└── evaluation.json
```

## 2. Table workflows

### HEST and STImage benchmark table

First create target-blind patch caches and fold contracts:

```bash
python experiments/benchmarks/prepare_hest_stimage_comparator_contracts.py \
  --stage all
```

Cache the representation required by a comparator when needed. For example:

```bash
python experiments/benchmarks/run_hest_stimage_benchmarks.py \
  cache --kind egn50 --gpu 0
```

Run and independently evaluate one method at a time:

```bash
python experiments/benchmarks/run_hest_stimage_benchmarks.py \
  run --method METHOD_ID --gpus 0 --epochs 30 --threads 4

python experiments/benchmarks/run_hest_stimage_benchmarks.py \
  evaluate --method METHOD_ID
```

Use `--help` to list the supported method IDs and cache kinds. Comparator
environments and weights are method-specific; see
[`comparators/README.md`](comparators/README.md) and
[`THIRD_PARTY.md`](THIRD_PARTY.md).

### MNP transfer table

HEST and STImage use both CTransPath and UNI-v1 representations:

```bash
for encoder in ctranspath uni_v1; do
  python experiments/transfer/run_hest_within_section.py --encoder "$encoder"
  python experiments/transfer/run_stimage_within_section.py --encoder "$encoder"
done
```

HESpotEx cSCC and HER2+ use sealed base predictions:

```bash
for cohort in cscc her2; do
  python experiments/transfer/run_hespotex_within_section.py build --cohort "$cohort"
  python experiments/transfer/run_hespotex_within_section.py evaluate --cohort "$cohort"
  python experiments/transfer/run_hespotex_within_section.py metrics --cohort "$cohort"
done
```

Comparator-transfer results are generated with:

```bash
python -m lin_mnp.analysis.evaluate_comparator_transfer
```

All transfer analyses retain the base predictor and scoring target unchanged
before and after MNP.

### cSCC comparator metrics

After the additional comparator runs and HESpotEx/STFlow artifacts are
available:

```bash
python experiments/benchmarks/evaluate_cscc_added_benchmark_metrics.py
```

The script writes its tables below
`$LIN_MNP_RESULTS_ROOT/additional_comparators/cscc/evaluation/` unless
`--output` is supplied.

## 3. Figure workflows

Install the figure extras first:

```bash
python -m pip install -e ".[analysis,figures]"
```

### Figure 1: framework

Requires cSCC H&E images, geometry, labels, and both CTransPath descriptor
views:

```bash
python plotting/manuscript/make_framework_figure.py \
  --output "$LIN_MNP_RESULTS_ROOT/figures/figure1.png"
```

### Figure 2: matched-descriptor attribution

Requires
`$LIN_MNP_ARCHIVE_ROOT/reports/comparator_results/benchmark_common_contract_attribution.csv`:

```bash
python plotting/manuscript/make_attribution_figure.py \
  --output "$LIN_MNP_RESULTS_ROOT/figures/figure2.png"
```

The upstream attribution runs are in `experiments/component_analysis/`. The
portable entry points retained in this release are:

```bash
python experiments/component_analysis/prepare_benchmark_384_patch_cache.py
python experiments/component_analysis/run_benchmark_common_ctranspath_ridge.py \
  --scale 384
python experiments/component_analysis/run_benchmark_common_original_downstream.py \
  --method METHOD_ID --fold FOLD_ID --device cuda:0
```

Some encoder-specific attribution runners call preserved helpers in the
companion artifact bundle; this dependency is explicit through
`LIN_MNP_ARCHIVE_ROOT`.

### Benchmark end-point composite

Requires the three sealed source panels under
`$LIN_MNP_ARCHIVE_ROOT/figures/benchmark/panels/`:

```bash
python plotting/manuscript/make_benchmark_figure.py \
  --output "$LIN_MNP_RESULTS_ROOT/figures/supplementary_benchmark_endpoints.png"
```

### Spatial-expression maps

```bash
python plotting/spatial_maps/build_cscc_spatial_expression_overviews.py \
  --output "$LIN_MNP_RESULTS_ROOT/figures/spatial_maps" \
  --dpi 220 --workers 4

python plotting/spatial_maps/build_hest_stimage_spatial_expression_overviews.py \
  --output "$LIN_MNP_RESULTS_ROOT/figures/spatial_maps" \
  --dpi 220
```

Both builders validate prediction identities and metric provenance before
rendering.

## 4. Supplementary analyses

| Analysis | Command |
|---|---|
| Encoder transfer | `python -m lin_mnp.analysis.evaluate_encoder_transfer --device cuda:0` |
| Neighbour accuracy sensitivity | `python -m lin_mnp.analysis.analyze_accuracy_sensitivity` |
| Propagation depth | `python -m lin_mnp.analysis.analyze_diffusion_depth` |
| Descriptor graph views | `python -m lin_mnp.analysis.analyze_graph_views` |
| Within-section controls | `python experiments/neighbor_controls/run_within_section_control.py train-fold --fold FOLD_ID` |
| Control aggregation | `python experiments/neighbor_controls/run_within_section_control.py evaluate` |
| Single-fold efficiency | `python experiments/benchmarks/run_cscc_single_fold_efficiency.py --help` |

## Manuscript-to-code map

| Manuscript item | Primary entry point |
|---|---|
| cSCC linear readout and Lin-MNP | `lin-mnp train`, then `lin-mnp evaluate` |
| HEST and STImage benchmark table | `experiments/benchmarks/run_hest_stimage_benchmarks.py` |
| Benchmark and encoder transfer | `experiments/transfer/run_{hest,stimage,hespotex}_within_section.py` |
| Comparator transfer | `python -m lin_mnp.analysis.evaluate_comparator_transfer` |
| Framework figure | `plotting/manuscript/make_framework_figure.py` |
| Attribution figure | `plotting/manuscript/make_attribution_figure.py` |
| Benchmark composite | `plotting/manuscript/make_benchmark_figure.py` |
| Spatial maps | `plotting/spatial_maps/` |
| Component attribution | `experiments/component_analysis/` |
| Neighbour controls | `experiments/neighbor_controls/` |
| Runtime analysis | `experiments/benchmarks/run_cscc_single_fold_efficiency.py` |

## Expression targets and scoring

Training targets and scoring references serve different roles:

| Analysis | Training targets | PCC scoring reference |
|---|---|---|
| Native modules on matched cSCC descriptors | Ordered 997-gene `log1p(raw counts)` | Common `log1p(raw counts)` within section |
| Readouts paired with method-specific representations | Same unstandardised `log1p(raw counts)` | Same common expression reference |
| Primary CTransPath-384 readout | `log1p(raw counts)` standardised within each training section and gene | Same common expression reference; section-standardised labels give identical PCC |
| Complete cSCC pipelines | Method-specific target transformations | Each method's own target space |
| Complete HEST and STImage pipelines | Deposited 50-gene targets supplied by the external contract | Deposited `log1p(raw counts)`; the archived PACE-ST branch also standardises targets and predictions within section |
| MNP on 18 cSCC predictors | Base predictors are not refitted | Each predictor's own target, unchanged before and after MNP |

For each comparator MNP pair, predictions are standardised within section and
gene before propagation, the same measured-expression target is retained for
both scores, and the base predictor is unchanged. Linear-readout transfer
analyses likewise retain each benchmark's target before and after MNP.

A positive affine transformation of the scoring target leaves PCC unchanged
for fixed predictions within a section. Applying section-specific transforms
before fitting can change the fitted predictor, and applying them before
pooling sections can change pooled PCC. Per-spot library-size normalisation
(CP10K or CPM) changes the scoring reference rather than merely changing its
units.

## Verification checklist

Before reporting reproduced values:

1. Run `python -m unittest discover -s tests -v`.
2. Confirm every expected fold has `complete.json` and
   `run_metadata.json`.
3. Let the evaluation entry point verify hashes; do not score unsealed
   prediction files directly.
4. Record the five configured roots, Python environment, GPU model, and command
   line.
5. Verify vendored-source integrity with
   `(cd comparators/methods && sha256sum -c MANIFEST.sha256)`.
