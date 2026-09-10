import unittest

from app.analyze_defense import analyze_defense


def _payload(experiment_id, aggregation, accuracy, class_a):
    return {
        "experiment_id": experiment_id,
        "aggregation": aggregation,
        "history": [
            {
                "round": 1,
                "phase": "evaluate",
                "global_val_acc": accuracy,
                "per_class_acc": {"A": class_a},
            }
        ],
    }


class DefenseAnalysisTests(unittest.TestCase):
    def test_computes_clean_penalty_attack_damage_and_recovery(self):
        report = analyze_defense(
            _payload("cf", "fedavg", 0.80, 0.90),
            _payload("af", "fedavg", 0.60, 0.40),
            _payload("cd", "median", 0.75, 0.80),
            _payload("ad", "median", 0.70, 0.70),
        )

        self.assertAlmostEqual(report["clean_defense_penalty"], -0.05)
        self.assertAlmostEqual(report["fedavg_attack_damage"], -0.20)
        self.assertAlmostEqual(report["defense_attack_damage"], -0.05)
        self.assertAlmostEqual(report["recovery_vs_attacked_fedavg"], 0.10)
        self.assertAlmostEqual(
            report["per_class_final"]["A"]["recovery_vs_attacked_fedavg"], 0.30
        )


if __name__ == "__main__":
    unittest.main()
