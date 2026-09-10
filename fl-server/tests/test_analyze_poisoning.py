import unittest

from app.analyze_poisoning import compare_histories


def _history(experiment_id, accuracies, class_values, attacked=False):
    records = []
    for round_number, accuracy in enumerate(accuracies, start=1):
        records.append(
            {
                "round": round_number,
                "phase": "fit",
                "client_metrics": [
                    {
                        "client_id": "client-0",
                        "attack_active": attacked,
                        "attack_types": "label_flip" if attacked else "",
                    }
                ],
            }
        )
        records.append(
            {
                "round": round_number,
                "phase": "evaluate",
                "global_val_acc": accuracy,
                "per_class_acc": class_values[round_number - 1],
                "per_class_total": {"A": 5, "B": 5},
            }
        )
    return {"experiment_id": experiment_id, "history": records}


class PoisoningAnalysisTests(unittest.TestCase):
    def test_compares_final_peak_per_class_and_audit(self):
        clean = _history(
            "clean",
            [0.5, 0.8],
            [{"A": 0.4, "B": 0.6}, {"A": 0.8, "B": 0.8}],
        )
        attacked = _history(
            "attack",
            [0.4, 0.6],
            [{"A": 0.2, "B": 0.6}, {"A": 0.4, "B": 0.8}],
            attacked=True,
        )

        report = compare_histories(clean, attacked)

        self.assertAlmostEqual(report["final_accuracy_delta"], -0.2)
        self.assertEqual(report["clean_peak"], {"round": 2, "accuracy": 0.8})
        self.assertAlmostEqual(report["per_class_final"]["A"]["delta"], -0.4)
        self.assertEqual(
            report["attack_audit"][0]["active_clients"][0]["client_id"],
            "client-0",
        )

    def test_rejects_different_round_sets(self):
        clean = _history("clean", [0.5, 0.8], [{"A": 0.5}, {"A": 0.8}])
        attacked = _history("attack", [0.4], [{"A": 0.4}], attacked=True)

        with self.assertRaisesRegex(ValueError, "rounds differ"):
            compare_histories(clean, attacked)


if __name__ == "__main__":
    unittest.main()
