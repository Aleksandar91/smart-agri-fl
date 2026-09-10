import math
import unittest

from app.dp_accounting import account_epsilon, gaussian_rdp, noise_stdv, rdp_to_epsilon


class DpAccountingTests(unittest.TestCase):
    def test_higher_noise_gives_smaller_epsilon(self):
        weak = account_epsilon(0.5, steps=10, delta=1e-5)
        strong = account_epsilon(2.0, steps=10, delta=1e-5)
        self.assertLess(strong["epsilon"], weak["epsilon"])
        self.assertEqual(weak["sampling_rate"], 1.0)

    def test_more_rounds_increase_epsilon(self):
        short = account_epsilon(1.0, steps=1, delta=1e-5)
        long = account_epsilon(1.0, steps=10, delta=1e-5)
        self.assertGreater(long["epsilon"], short["epsilon"])

    def test_gaussian_rdp_formula(self):
        self.assertAlmostEqual(gaussian_rdp(1.0, 2.0, 1), 1.0)
        self.assertAlmostEqual(gaussian_rdp(1.0, 2.0, 10), 10.0)

    def test_rdp_conversion_matches_hand_calculation(self):
        rdp = gaussian_rdp(1.0, 2.0, 10)
        eps = rdp_to_epsilon(rdp, 2.0, 1e-5)
        self.assertAlmostEqual(eps, 10.0 + math.log(1e5), places=6)

    def test_zero_noise_is_unbounded(self):
        spent = account_epsilon(0.0, steps=10, delta=1e-5)
        self.assertTrue(math.isinf(spent["epsilon"]))

    def test_noise_stdv_matches_flower_convention(self):
        self.assertAlmostEqual(noise_stdv(1.0, 10.0, 4), 2.5)

    def test_isotropic_snr_independent_of_clip(self):
        from app.dp_accounting import expected_noise_l2, isotropic_snr_upper_bound

        head_params = 618523
        snr = isotropic_snr_upper_bound(0.5, 4, head_params)
        self.assertAlmostEqual(snr, 4.0 / (0.5 * math.sqrt(head_params)), places=8)
        # Signal ≤ C, noise L2 = z C √d / n, so C cancels.
        noise_c10 = expected_noise_l2(0.5, 10.0, 4, head_params)
        noise_c1 = expected_noise_l2(0.5, 1.0, 4, head_params)
        self.assertAlmostEqual(10.0 / noise_c10, 1.0 / noise_c1, places=8)
        self.assertAlmostEqual(10.0 / noise_c10, snr, places=8)


if __name__ == "__main__":
    unittest.main()
