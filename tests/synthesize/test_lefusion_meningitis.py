from __future__ import annotations

import unittest


class TestNamingAndSplit(unittest.TestCase):
    def test_nnunet_case_id_parsing(self):
        from src.synthesize.lefusion_meningitis.io import case_id_from_image

        self.assertEqual(case_id_from_image("case_012_0000.nii.gz"), "case_012")
        with self.assertRaisesRegex(ValueError, "CCCC"):
            case_id_from_image("case_012.nii.gz")

    def test_stratified_split_is_reproducible_and_disjoint(self):
        from src.synthesize.lefusion_meningitis.data import stratified_split

        statistics = [
            {"case_id": f"case_{index:03d}", "lesion_voxels": index * 11, "components": index % 7}
            for index in range(40)
        ]
        first = stratified_split(statistics, {"train": 28, "val": 6, "test": 6}, 42)
        second = stratified_split(statistics, {"train": 28, "val": 6, "test": 6}, 42)
        self.assertEqual(first, second)
        self.assertEqual({name: len(items) for name, items in first.items()}, {
            "train": 28,
            "val": 6,
            "test": 6,
        })
        all_cases = first["train"] + first["val"] + first["test"]
        self.assertEqual(len(all_cases), len(set(all_cases)))


class TestArrayOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
        except ImportError as exc:
            raise unittest.SkipTest("NumPy is unavailable") from exc

    def test_normalization_round_trip_inside_clip_range(self):
        import numpy as np

        from src.synthesize.lefusion_meningitis.data import (
            denormalize_image,
            robust_normalize,
        )

        image = np.linspace(-2.0, 2.0, 125, dtype=np.float32).reshape(5, 5, 5)
        normalized, metadata = robust_normalize(image, clip_z=20.0)
        restored = denormalize_image(normalized, metadata)
        self.assertTrue(np.allclose(restored, image, atol=1e-5))

    def test_hard_composite_preserves_every_exterior_voxel(self):
        import numpy as np

        from src.synthesize.lefusion_meningitis.synthesis import hard_composite

        background = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        generated = np.full_like(background, -9)
        mask = np.zeros_like(background, dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        result = hard_composite(background, generated, mask)
        self.assertTrue(np.array_equal(result[~mask], background[~mask]))
        self.assertTrue(np.array_equal(result[mask], generated[mask]))


class TestForegroundLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("PyTorch is unavailable") from exc

    def test_sparse_mask_is_normalized_by_foreground(self):
        import torch

        from src.synthesize.lefusion_meningitis.losses import masked_foreground_loss

        prediction = torch.zeros((1, 1, 4, 4, 4))
        target = torch.zeros_like(prediction)
        target[0, 0, 2, 2, 2] = 3.0
        mask = torch.zeros_like(prediction, dtype=torch.bool)
        mask[0, 0, 2, 2, 2] = True
        self.assertAlmostEqual(
            float(masked_foreground_loss(prediction, target, mask)), 3.0
        )

    def test_empty_mask_is_rejected(self):
        import torch

        from src.synthesize.lefusion_meningitis.losses import masked_foreground_loss

        value = torch.zeros((1, 1, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "empty"):
            masked_foreground_loss(value, value, torch.zeros_like(value, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()

