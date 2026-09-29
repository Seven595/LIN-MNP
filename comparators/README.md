# Published comparators

`methods/` holds the upstream source required by the benchmark adapters.
`adapters/` puts each published method on an explicit manuscript fold and gene
panel contract. Vendored sources retain their upstream licensing. Their
machine-specific path defaults were made portable without changing model
equations; runtime adaptations made by an adapter are recorded in its output
metadata.

Install Lin-MNP in editable mode before launching an adapter so
`lin_mnp.benchmark` and the shared path policy are importable. Comparator
environments remain method-specific and are not installed by the base package.

HoloTea has no usable upstream package; the implementation used here is in
`experiments/benchmarks/train_additional_comparators.py`.

The STAG adapter is also self-contained; only its released ResNet18 checkpoint
is required. Download instructions are in `methods/STAG/README.md`.

HisToGene lives at `methods/mclSTExp/baselines/HisToGene`.
