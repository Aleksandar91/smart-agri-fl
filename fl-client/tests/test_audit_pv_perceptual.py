import unittest

from PIL import Image

from app.audit_pv_perceptual import BKTree, difference_hash, perceptual_hash


class PerceptualAuditTests(unittest.TestCase):
    def test_bk_tree_returns_values_inside_hamming_radius(self):
        tree = BKTree()
        values = [0b0000, 0b0011, 0b1111]
        for value in values:
            tree.add(value)

        matches = dict(tree.query(0b0001, radius=1))

        self.assertEqual(matches[0b0000], 1)
        self.assertEqual(matches[0b0011], 1)
        self.assertNotIn(0b1111, matches)

    def test_hashes_are_deterministic(self):
        image = Image.new("RGB", (32, 32), color=(255, 255, 255))
        for coordinate in range(8, 24):
            image.putpixel((coordinate, coordinate), (0, 128, 0))

        self.assertEqual(perceptual_hash(image), perceptual_hash(image.copy()))
        self.assertEqual(difference_hash(image), difference_hash(image.copy()))

    def test_hash_distance_detects_identical_images(self):
        image = Image.new("RGB", (24, 24), color=(20, 80, 120))
        copy = image.copy()

        self.assertEqual(
            BKTree.distance(perceptual_hash(image), perceptual_hash(copy)),
            0,
        )
        self.assertEqual(
            BKTree.distance(difference_hash(image), difference_hash(copy)),
            0,
        )


if __name__ == "__main__":
    unittest.main()
