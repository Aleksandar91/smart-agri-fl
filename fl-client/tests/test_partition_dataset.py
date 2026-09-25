import json
import random
import tempfile
import unittest
from pathlib import Path

from app.partition_dataset import (
    _heterogeneity_summary,
    _load_grouped_source_manifest,
    _partition_grouped_dirichlet,
    _partition_grouped_iid,
)


class GroupSafePartitionTests(unittest.TestCase):
    def _source_manifest(self) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "train.json"
        classes = ["A", "B"]
        records = []
        samples = []
        for label, class_name in enumerate(classes):
            for group_number in range(6):
                for image_number in range(2):
                    relative_path = (
                        f"{class_name}/{group_number}-{image_number}.jpg"
                    )
                    records.append(
                        {
                            "path": relative_path,
                            "class_name": class_name,
                            "group_id": f"{class_name}-group-{group_number}",
                            "split": "train",
                        }
                    )
                    samples.append([relative_path, label])
        path.write_text(
            json.dumps(
                {
                    "dataset_id": "synthetic",
                    "dataset_manifest_sha256": "abc",
                    "split": "train",
                    "source": {"raw_root": "/raw"},
                    "classes": classes,
                    "samples": samples,
                    "records": records,
                }
            ),
            encoding="utf-8",
        )
        return path

    def _assert_groups_do_not_cross_clients(self, parts):
        group_client = {}
        for client_index, samples in enumerate(parts):
            for relative_path, _label in samples:
                group_id = relative_path.rsplit("-", 1)[0]
                previous = group_client.setdefault(group_id, client_index)
                self.assertEqual(previous, client_index)

    def test_loads_only_locked_train_manifest(self):
        manifest, groups, records = _load_grouped_source_manifest(
            self._source_manifest()
        )

        self.assertEqual(manifest["split"], "train")
        self.assertEqual(len(groups), 2)
        self.assertEqual(sum(len(group) for group in groups[0]), 12)
        self.assertEqual(len(records), 24)

    def test_grouped_iid_never_splits_a_group(self):
        _manifest, groups, _records = _load_grouped_source_manifest(
            self._source_manifest()
        )

        parts = _partition_grouped_iid(groups, 3, random.Random(11))

        self.assertEqual(sum(len(part) for part in parts), 24)
        self._assert_groups_do_not_cross_clients(parts)

    def test_grouped_dirichlet_is_deterministic_and_group_safe(self):
        _manifest, groups, _records = _load_grouped_source_manifest(
            self._source_manifest()
        )

        first = _partition_grouped_dirichlet(
            groups, 3, alpha=0.5, rng=random.Random(17), min_per_client=2
        )
        second = _partition_grouped_dirichlet(
            groups, 3, alpha=0.5, rng=random.Random(17), min_per_client=2
        )

        self.assertEqual(first, second)
        self._assert_groups_do_not_cross_clients(first)

    def test_rejects_non_train_source_manifest(self):
        path = self._source_manifest()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["split"] = "test"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "locked train split"):
            _load_grouped_source_manifest(path)

    def test_reports_class_monopoly_and_entropy(self):
        parts = [
            [("a1.jpg", 0), ("a2.jpg", 0), ("b1.jpg", 1)],
            [("b2.jpg", 1)],
        ]

        result = _heterogeneity_summary(parts, ["A", "B"])

        self.assertEqual(result["class_metrics"]["A"]["monopoly"], 1.0)
        self.assertEqual(
            result["class_metrics"]["A"]["clients_with_samples"],
            1,
        )
        self.assertAlmostEqual(
            result["class_metrics"]["B"]["normalized_entropy"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
