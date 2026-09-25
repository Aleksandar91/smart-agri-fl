import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.attacks import (
    load_attack_config,
    manipulate_model_update,
    select_label_flip_positions,
)


class AttackConfigTests(unittest.TestCase):
    def _write_config(self, payload) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "attack.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_round_gated_label_flip(self):
        path = self._write_config(
            {
                "seed": 7,
                "active_rounds": {"start": 2, "end": 4},
                "label_flip": {
                    "source_class": "A",
                    "target_class": "B",
                    "fraction": 0.5,
                },
            }
        )
        config = load_attack_config(path)
        config.validate_classes(["A", "B"])

        self.assertFalse(config.is_active(1))
        self.assertTrue(config.is_active(2))
        self.assertTrue(config.is_active(4))
        self.assertFalse(config.is_active(5))
        self.assertEqual(config.attack_types(3), ["label_flip"])

    def test_rejects_unknown_class_during_resolution(self):
        path = self._write_config(
            {
                "label_flip": {
                    "source_class": "missing",
                    "target_class": "B",
                    "fraction": 1.0,
                }
            }
        )
        config = load_attack_config(path)
        with self.assertRaisesRegex(ValueError, "global class list"):
            config.validate_classes(["A", "B"])

    def test_rejects_empty_attack(self):
        path = self._write_config({"seed": 1337})
        with self.assertRaisesRegex(ValueError, "must define"):
            load_attack_config(path)


class LabelSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_source_only(self):
        labels = [0, 1, 0, 0, 2, 0, 1, 0]
        first = select_label_flip_positions(labels, 0, 0.6, seed=42)
        second = select_label_flip_positions(labels, 0, 0.6, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(labels[index] == 0 for index in first))

    def test_full_fraction_selects_every_source_position(self):
        labels = [0, 1, 0, 2, 0]
        selected = select_label_flip_positions(labels, 0, 1.0, seed=1)
        self.assertEqual(selected, {0, 2, 4})


class ModelUpdateAttackTests(unittest.TestCase):
    def _config(self):
        path_payload = {
            "active_rounds": {"start": 1, "end": 2},
            "model_update": {"mode": "scale_delta", "scale": -5.0},
        }
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "attack.json"
        path.write_text(json.dumps(path_payload), encoding="utf-8")
        return load_attack_config(path)

    def test_scale_delta_reverses_floating_update_and_preserves_integer_state(self):
        global_weights = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array(4, dtype=np.int64),
        ]
        local_weights = [
            np.array([2.0, 0.0], dtype=np.float32),
            np.array(5, dtype=np.int64),
        ]

        outgoing, metrics = manipulate_model_update(
            global_weights, local_weights, self._config(), server_round=1
        )

        np.testing.assert_allclose(outgoing[0], np.array([-4.0, 12.0], dtype=np.float32))
        np.testing.assert_array_equal(outgoing[1], local_weights[1])
        self.assertTrue(metrics["model_update_attack"])
        self.assertAlmostEqual(metrics["update_scale"], -5.0)
        self.assertAlmostEqual(
            metrics["sent_delta_l2"], 5.0 * metrics["clean_delta_l2"]
        )

    def test_inactive_round_sends_honest_local_weights(self):
        global_weights = [np.array([1.0], dtype=np.float32)]
        local_weights = [np.array([3.0], dtype=np.float32)]

        outgoing, metrics = manipulate_model_update(
            global_weights, local_weights, self._config(), server_round=3
        )

        np.testing.assert_allclose(outgoing[0], local_weights[0])
        self.assertFalse(metrics["model_update_attack"])
        self.assertEqual(metrics["update_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
