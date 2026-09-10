import unittest

from app.analyze_dp import analyze_dp


def _payload(experiment_id, accuracy, class_a, class_b, noise=None):
    payload = {
        "experiment_id": experiment_id,
        "history": [
            {
                "round": 1,
                "phase": "evaluate",
                "global_val_acc": accuracy,
                "per_class_acc": {"A": class_a, "B": class_b},
            }
        ],
    }
    if noise is not None:
        payload["dp_config"] = {"noise_multiplier": noise}
    return payload


class DpAnalysisTests(unittest.TestCase):
    def test_reports_utility_cost_and_worst_class(self):
        report = analyze_dp(
            _payload("base", 0.80, 0.90, 0.70),
            [_payload("dp1", 0.70, 0.50, 0.55, noise=1.0)],
        )
        run = report["runs"][0]
        self.assertAlmostEqual(run["utility_cost"], -0.10)
        self.assertAlmostEqual(run["peak_accuracy"], 0.70)
        self.assertEqual(run["peak_round"], 1)
        self.assertEqual(run["worst_class_drops"][0]["class"], "A")
        self.assertAlmostEqual(run["worst_class_drops"][0]["delta"], -0.40)
        self.assertEqual(run["classes_drop_ge_10pp"], 2)

    def test_rejects_round_mismatch(self):
        short = {
            "experiment_id": "dp",
            "history": [
                {
                    "round": 1,
                    "phase": "evaluate",
                    "global_val_acc": 0.5,
                    "per_class_acc": {"A": 0.5, "B": 0.5},
                }
            ],
        }
        long = _payload("base", 0.8, 0.9, 0.7)
        long["history"].append(
            {
                "round": 2,
                "phase": "evaluate",
                "global_val_acc": 0.81,
                "per_class_acc": {"A": 0.9, "B": 0.7},
            }
        )
        with self.assertRaisesRegex(ValueError, "rounds differ"):
            analyze_dp(long, [short])


if __name__ == "__main__":
    unittest.main()
