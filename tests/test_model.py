"""Unit tests for the label-free Lin-MNP numerical core."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from lin_mnp import model
from lin_mnp import within_section


class PatientBalancedRidgeTests(unittest.TestCase):
    def test_each_patient_has_equal_total_weight(self) -> None:
        patients = np.asarray(["A", "A", "A", "B"])
        weights = model.patient_equal_weights(patients, ("A", "B"))

        self.assertAlmostEqual(float(weights[patients == "A"].sum()), 2.0)
        self.assertAlmostEqual(float(weights[patients == "B"].sum()), 2.0)
        self.assertAlmostEqual(float(weights.sum()), 4.0)

    def test_cpu_ridge_returns_finite_predictions(self) -> None:
        rng = np.random.default_rng(4)
        features = rng.normal(size=(8, model.CONTEXT_DIM)).astype(np.float32)
        targets = rng.normal(size=(8, 3)).astype(np.float32)
        patients = np.asarray(["A"] * 4 + ["B"] * 4)

        artifact = model.fit_patient_balanced_ridge(
            features, targets, patients, ("A", "B"), torch.device("cpu")
        )
        predictions = model.predict_ridge(features, artifact)

        self.assertEqual(predictions.shape, targets.shape)
        self.assertTrue(np.isfinite(predictions).all())

    def test_ridge_rejects_misaligned_targets(self) -> None:
        features = np.zeros((4, model.CONTEXT_DIM), dtype=np.float32)
        targets = np.zeros((3, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "align by spot"):
            model.fit_patient_balanced_ridge(
                features, targets, ("A", "A", "B", "B"), ("A", "B"), "cpu"
            )


class PropagationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(9)
        self.sections = np.asarray(["S1"] * 65 + ["S2"] * 65 + ["S3"] * 65)
        self.context = rng.normal(size=(195, 12)).astype(np.float32)
        self.local = rng.normal(size=(195, 12)).astype(np.float32)

    def test_cross_section_graph_is_balanced_and_row_stochastic(self) -> None:
        graph = model.build_section_balanced_graph(
            self.sections, self.context, self.local
        )

        self.assertEqual(graph["index"].shape, (195, 128))
        np.testing.assert_allclose(graph["weight"].sum(axis=1), 1.0, atol=2e-6)
        for row, neighbours in enumerate(graph["index"]):
            self.assertTrue(np.all(self.sections[neighbours] != self.sections[row]))
            masses = [
                graph["weight"][row, self.sections[neighbours] == section].sum()
                for section in ("S1", "S2", "S3")
                if section != self.sections[row]
            ]
            np.testing.assert_allclose(masses, (0.5, 0.5), atol=2e-6)

    def test_constant_predictions_are_preserved(self) -> None:
        graph = model.build_section_balanced_graph(
            self.sections, self.context, self.local
        )
        values = np.ones((len(self.sections), 5), dtype=np.float32)

        np.testing.assert_allclose(model.diffuse(values, graph), values, atol=2e-6)

    def test_similarity_variants_have_expected_shape(self) -> None:
        context = model.normalize_rows(self.context)
        local = model.normalize_rows(self.local)
        query = np.arange(7)
        gallery = np.arange(11, 31)

        for fusion in ("borda", "score", "context"):
            similarity = model.variant_similarity(
                context, local, query, gallery, fusion
            )
            self.assertEqual(similarity.shape, (7, 20))
            self.assertTrue(np.isfinite(similarity).all())

    def test_within_section_propagation_excludes_query_spot(self) -> None:
        prediction = np.arange(3, dtype=np.float32)[:, None]
        descriptors = np.eye(3, dtype=np.float32)

        propagated, count = within_section.apply(
            prediction,
            descriptors,
            descriptors,
            neighbour_count=2,
            fusion="context",
            weighting="uniform",
        )

        self.assertEqual(count, 2)
        np.testing.assert_allclose(propagated[:, 0], (1.5, 1.0, 0.5))


if __name__ == "__main__":
    unittest.main()
