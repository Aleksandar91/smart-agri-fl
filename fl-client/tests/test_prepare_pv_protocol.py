import unittest
from pathlib import Path

from app.prepare_pv_protocol import (
    _class_aware_leaf_id,
    assign_splits,
    attach_groups,
    select_capped_groups,
    validate,
)


def _record(path, class_name, digest, leaf_id=None):
    return {
        "path": path,
        "raw_class": class_name,
        "class_name": class_name,
        "size_bytes": 10,
        "sha256": digest,
        "leaf_id": leaf_id,
    }


class PreparePlantVillageProtocolTests(unittest.TestCase):
    def test_leaf_lookup_is_class_aware_when_stems_collide(self):
        leaf_map = {
            "rs_hl 100": [
                "Apple___healthy:::1.0",
                "Blueberry___healthy:::7.0",
            ]
        }
        path = Path("uuid___RS_HL 100.JPG")

        self.assertEqual(
            _class_aware_leaf_id("Apple___healthy", path, leaf_map),
            "Apple___healthy:::1.0",
        )
        self.assertEqual(
            _class_aware_leaf_id("Blueberry___healthy", path, leaf_map),
            "Blueberry___healthy:::7.0",
        )
        self.assertEqual(
            _class_aware_leaf_id(
                "Apple___Black_rot",
                Path("uuid___JR_FrgE.S 1.JPG"),
                {"jr_frge.s 1": ["Apple_Frogeye Spot:::3.0"]},
            ),
            "Apple_Frogeye Spot:::3.0",
        )

    def test_leaf_siblings_and_exact_duplicates_form_indivisible_groups(self):
        records = [
            _record("A/a1.jpg", "A", "h1", "A:::1"),
            _record("A/a2.jpg", "A", "h2", "A:::1"),
            _record("A/a2-copy.jpg", "A", "h2"),
            _record("A/a3.jpg", "A", "h3"),
        ]

        audit = attach_groups(records)

        self.assertEqual(records[0]["group_id"], records[1]["group_id"])
        self.assertEqual(records[1]["group_id"], records[2]["group_id"])
        self.assertNotEqual(records[2]["group_id"], records[3]["group_id"])
        self.assertEqual(audit["exact_duplicate_sets"], 1)

    def test_split_is_deterministic_and_has_no_group_leakage(self):
        original = []
        for class_name in ("A", "B"):
            for group_number in range(12):
                leaf_id = f"{class_name}:::{group_number}"
                for image_number in range(2):
                    original.append(
                        _record(
                            f"{class_name}/{group_number}-{image_number}.jpg",
                            class_name,
                            f"{class_name}-{group_number}-{image_number}",
                            leaf_id,
                        )
                    )

        first = [dict(record) for record in original]
        second = [dict(record) for record in original]
        attach_groups(first)
        attach_groups(second)
        assign_splits(first, (0.7, 0.15, 0.15), split_seed=17)
        assign_splits(second, (0.7, 0.15, 0.15), split_seed=17)

        self.assertEqual(
            [(record["path"], record["split"]) for record in first],
            [(record["path"], record["split"]) for record in second],
        )
        result = validate(first, ["A", "B"])
        self.assertTrue(result["passed"], result["errors"])

    def test_capped_selection_never_splits_a_group_or_exceeds_cap(self):
        records = []
        for group_number, size in enumerate((3, 3, 2, 2, 1)):
            for image_number in range(size):
                records.append(
                    _record(
                        f"A/{group_number}-{image_number}.jpg",
                        "A",
                        f"h-{group_number}-{image_number}",
                        f"A:::{group_number}",
                    )
                )
        attach_groups(records)

        selected = select_capped_groups(records, per_class=7, seed=23)

        self.assertLessEqual(len(selected), 7)
        selected_groups = {record["group_id"] for record in selected}
        for group_id in selected_groups:
            full_size = sum(record["group_id"] == group_id for record in records)
            selected_size = sum(record["group_id"] == group_id for record in selected)
            self.assertEqual(selected_size, full_size)

    def test_validation_rejects_group_crossing_split_boundary(self):
        records = [
            {
                **_record("A/a.jpg", "A", "h1"),
                "group_id": "same-group",
                "split": "train",
            },
            {
                **_record("A/b.jpg", "A", "h2"),
                "group_id": "same-group",
                "split": "test",
            },
            {
                **_record("A/c.jpg", "A", "h3"),
                "group_id": "other-group",
                "split": "validation",
            },
        ]

        result = validate(records, ["A"])

        self.assertFalse(result["passed"])
        self.assertIn("same-group", result["leaking_group_ids"])


if __name__ == "__main__":
    unittest.main()
