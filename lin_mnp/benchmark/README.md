# Benchmark evaluation contract

These four modules are **not a prediction method**. They only define the
patient folds, gene panel, patch cache and metrics used to put published
comparators on the same contract as Lin-MNP.

They contain no trainable prediction model and do not import project history or
development-only implementations.

Shared gene identities and folds do not imply a single training or scoring
target for every analysis. Attribution uses a common log1p raw-count PCC
reference; complete cSCC pipeline scoring retains method-specific targets,
and each MNP pair retains its original target. The primary cSCC readout fits
section-standardized targets. See the expression-target table in
[`REPRODUCTION.md`](../../REPRODUCTION.md#expression-targets-and-scoring).

| File | Role |
|---|---|
| `benchmark_external.py` | Folds, slides, 997-gene panel, patch cache, target transforms |
| `hespotex_metrics.py` | HESPotEx gene-wise metrics |
| `metrics.py` | Spot-level PCC / MI / SSIM / AUC |
| `evaluator.py` | Patient-equal aggregation |

Adapters import these helpers from `lin_mnp.benchmark`. Install the repository
in editable mode before launching an adapter.
