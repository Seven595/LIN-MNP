#!/usr/bin/env python3
"""Run Lin-MNP's numerical core on deterministic synthetic descriptors."""

from __future__ import annotations

import numpy as np

from lin_mnp import model


def main() -> None:
    rng = np.random.default_rng(7)
    spot_count = 65 * 3
    context = rng.normal(size=(spot_count, model.CONTEXT_DIM)).astype(np.float32)
    local = rng.normal(size=(spot_count, 32)).astype(np.float32)
    target = rng.normal(size=(spot_count, 5)).astype(np.float32)
    patients = np.asarray(["P1"] * 98 + ["P2"] * 97)
    sections = np.repeat(("S1", "S2", "S3"), 65)

    artifact = model.fit_patient_balanced_ridge(
        context, target, patients, ("P1", "P2"), device="cpu"
    )
    pointwise = model.predict_ridge(context, artifact)
    propagated = model.forward(context, local, sections, artifact)

    print(f"pointwise shape: {pointwise.shape}")
    print(f"Lin-MNP shape:   {propagated.shape}")
    print(f"finite output:   {bool(np.isfinite(propagated).all())}")


if __name__ == "__main__":
    main()
