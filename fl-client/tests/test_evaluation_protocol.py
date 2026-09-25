import unittest

import inspect

from app.evaluate_global import metrics_from_confusion
from app.fl_task import _validate_external_eval_manifest, train_one_round


class EvaluationProtocolTests(unittest.TestCase):
    def test_manifest_compatibility_accepts_disjoint_locked_validation(self):
        partition = {
            "classes": ["A", "B"],
            "source_dataset_id": "dataset-1",
            "samples": [["A/train.jpg", 0]],
        }
        validation = {
            "classes": ["A", "B"],
            "dataset_id": "dataset-1",
            "split": "validation",
            "samples": [["A/validation.jpg", 0], ["B/validation.jpg", 1]],
        }

        _validate_external_eval_manifest(partition, validation)

    def test_manifest_compatibility_rejects_test_during_fl_rounds(self):
        partition = {
            "classes": ["A", "B"],
            "source_dataset_id": "dataset-1",
            "samples": [["A/train.jpg", 0]],
        }
        test = {
            "classes": ["A", "B"],
            "dataset_id": "dataset-1",
            "split": "test",
            "samples": [["A/test.jpg", 0]],
        }

        with self.assertRaisesRegex(ValueError, "development validation"):
            _validate_external_eval_manifest(partition, test)

    def test_manifest_compatibility_rejects_path_overlap(self):
        partition = {
            "classes": ["A", "B"],
            "source_dataset_id": "dataset-1",
            "samples": [["A/same.jpg", 0]],
        }
        validation = {
            "classes": ["A", "B"],
            "dataset_id": "dataset-1",
            "split": "validation",
            "samples": [["A/same.jpg", 0]],
        }

        with self.assertRaisesRegex(ValueError, "overlap"):
            _validate_external_eval_manifest(partition, validation)

    def test_metrics_from_confusion(self):
        result = metrics_from_confusion([[8, 2], [1, 9]])

        self.assertAlmostEqual(result["accuracy"], 0.85)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.85)
        self.assertAlmostEqual(result["worst_class_recall"], 0.8)
        self.assertEqual(result["per_class_support"], [10, 10])

    def test_local_training_accepts_fedprox_mu(self):
        parameters = inspect.signature(train_one_round).parameters
        self.assertIn("proximal_mu", parameters)
        self.assertEqual(parameters["proximal_mu"].default, 0.0)


if __name__ == "__main__":
    unittest.main()
