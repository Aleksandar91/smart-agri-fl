import json
import unittest

import numpy as np

from app.dp_updates import (
    clip_trainable_update,
    parse_dp_scope_indices,
    parse_trainable_indices_json,
    trainable_indices,
)


class TrainableUpdateTests(unittest.TestCase):
    def test_skips_integer_and_frozen_float_tensors(self):
        current = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3, 4], dtype=np.int64),
            np.array([[0.1, 0.2]], dtype=np.float32),
        ]
        client = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([9, 9], dtype=np.int64),
            np.array([[0.4, 0.6]], dtype=np.float32),
        ]
        self.assertEqual(trainable_indices(current, [client]), [2])

    def test_clips_concatenated_trainable_l2(self):
        current = [
            np.zeros(2, dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        ]
        updated = [
            np.array([3.0, 4.0], dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        ]
        clipped = clip_trainable_update(current, updated, [0], clipping_norm=2.5)
        self.assertAlmostEqual(float(np.linalg.norm(clipped[0])), 2.5, places=5)
        np.testing.assert_array_equal(clipped[1], current[1])

    def test_preserves_non_trainable_client_tensors(self):
        current = [
            np.zeros(2, dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        ]
        updated = [
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([9.0, 8.0], dtype=np.float32),
        ]
        clipped = clip_trainable_update(current, updated, [0], clipping_norm=100.0)
        np.testing.assert_array_almost_equal(clipped[0], updated[0])
        np.testing.assert_array_equal(clipped[1], updated[1])

    def test_parse_trainable_indices_requires_agreement(self):
        self.assertEqual(
            parse_trainable_indices_json(
                [
                    {"trainable_indices_json": "[70,71]"},
                    {"trainable_indices_json": json.dumps([70, 71])},
                ]
            ),
            [70, 71],
        )

    def test_parse_trainable_indices_rejects_missing_and_mismatch(self):
        with self.assertRaises(ValueError):
            parse_trainable_indices_json([{"train_loss": 1.0}])
        with self.assertRaises(ValueError):
            parse_trainable_indices_json(
                [
                    {"trainable_indices_json": "[1]"},
                    {"trainable_indices_json": "[2]"},
                ]
            )

    def test_last_layer_scope_must_be_subset(self):
        metrics = [
            {
                "trainable_indices_json": "[240,241,242,243]",
                "last_layer_indices_json": "[242,243]",
            },
            {
                "trainable_indices_json": "[240,241,242,243]",
                "last_layer_indices_json": "[242,243]",
            },
        ]
        self.assertEqual(parse_dp_scope_indices(metrics, "head"), [240, 241, 242, 243])
        self.assertEqual(parse_dp_scope_indices(metrics, "last"), [242, 243])
        outside = [
            {
                "trainable_indices_json": "[240,241,242,243]",
                "last_layer_indices_json": "[242,999]",
            }
        ] * 2
        with self.assertRaises(ValueError):
            parse_dp_scope_indices(outside, "last")


if __name__ == "__main__":
    unittest.main()
