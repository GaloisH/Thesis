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

    def test_lesion_brightening_is_stronger_toward_center(self):
        import numpy as np

        from src.synthesize.lefusion_meningitis.synthesis import (
            brighten_lesion_interior,
        )

        background = np.zeros((9, 9, 9), dtype=np.float32)
        generated = np.full_like(background, -0.5)
        mask = np.zeros_like(background, dtype=bool)
        mask[2:7, 2:7, 2:7] = True

        adjusted = brighten_lesion_interior(
            background,
            generated,
            mask,
            margin=0.1,
            transition_voxels=3.0,
        )

        self.assertTrue(np.array_equal(adjusted[~mask], generated[~mask]))
        self.assertGreater(adjusted[4, 4, 4], adjusted[2, 2, 2])
        self.assertGreater(adjusted[4, 4, 4], background[4, 4, 1])

    def test_fixed_mask_roi_is_centered_and_rejects_invalid_masks(self):
        import numpy as np

        from src.synthesize.lefusion_meningitis.synthesis import roi_from_mask

        mask = np.zeros((48, 48, 48), dtype=bool)
        mask[21:26, 22:27, 23:28] = True
        roi, metadata = roi_from_mask(mask, (32, 32, 32), margin=4)
        self.assertEqual(mask[roi].shape, (32, 32, 32))
        self.assertTrue(mask[roi].any())
        self.assertEqual(metadata["bbox_shape"], [5, 5, 5])

        with self.assertRaisesRegex(ValueError, "empty"):
            roi_from_mask(np.zeros_like(mask), (32, 32, 32), margin=4)

        edge = np.zeros_like(mask)
        edge[:3, 20:23, 20:23] = True
        with self.assertRaisesRegex(ValueError, "image edge"):
            roi_from_mask(edge, (32, 32, 32), margin=4)

        oversized = np.zeros_like(mask)
        oversized[8:36, 15:20, 15:20] = True
        with self.assertRaisesRegex(ValueError, "does not fit"):
            roi_from_mask(oversized, (32, 32, 32), margin=4)

    def test_visualization_difference_is_zero_outside_mask(self):
        import numpy as np

        from src.synthesize.lefusion_meningitis.visualization import (
            masked_absolute_difference,
        )

        before = np.zeros((8, 8, 8), dtype=np.float32)
        after = np.ones_like(before)
        mask = np.zeros_like(before, dtype=bool)
        mask[3:5, 3:5, 3:5] = True
        difference = masked_absolute_difference(before, after, mask)
        self.assertTrue(np.array_equal(difference[~mask], np.zeros_like(difference[~mask])))
        self.assertTrue(np.array_equal(difference[mask], np.ones_like(difference[mask])))

    def test_prepared_component_mask_is_restored_to_full_volume(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        from src.synthesize.lefusion_meningitis.visualization import (
            _mask_from_prepared_entry,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch = np.zeros((1, 8, 8, 8), dtype=np.uint8)
            patch[0, 3:5, 3:5, 3:5] = 1
            np.savez(root / "component.npz", mask=patch)
            entry = {
                "patch_id": "case_000_lesion001",
                "patch": "component.npz",
                "crop": {
                    "start": [6, 6, 6],
                    "end": [14, 14, 14],
                    "padding": [[0, 0], [0, 0], [0, 0]],
                },
            }
            restored = _mask_from_prepared_entry((20, 20, 20), root, entry)
            self.assertEqual(int(restored.sum()), 8)
            self.assertTrue(restored[9:11, 9:11, 9:11].all())


class TestVisualizationPipeline(unittest.TestCase):
    def test_fake_model_writes_nifti_metadata_and_all_figures(self):
        try:
            import matplotlib  # noqa: F401
            import nibabel as nib
            import numpy as np
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"visualization dependency unavailable: {exc}") from exc
        import tempfile
        from pathlib import Path

        from src.synthesize.lefusion_meningitis.visualization import _process_case

        class FakeModel:
            def sample_patch(self, background, mask, histogram, *, generator=None):
                return torch.where(mask.bool(), background + 0.1, background)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "case_000_0000.nii.gz"
            mask_path = root / "case_000.nii.gz"
            output = root / "visualization"
            image = np.linspace(-1, 1, 48**3, dtype=np.float32).reshape((48, 48, 48))
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[22:26, 22:26, 22:26] = 1
            affine = np.diag([0.8, 0.9, 1.2, 1.0])
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(mask, affine), mask_path)
            config = {
                "data": {"patch_size": [32, 32, 32], "patch_margin": 4},
                "normalization": {"clip_z": 5.0, "foreground_epsilon": 1e-6},
                "synthesis": {
                    "checkpoint": "fake.pt",
                    "histogram_jitter": 0.0,
                    "intensity_z_limit": 5.0,
                    "max_boundary_jump_z": 4.0,
                },
                "visualization": {
                    "dpi": 50,
                    "format": "png",
                    "mask_color": "lime",
                    "mask_alpha": 0.45,
                    "roi_padding": 4,
                    "max_contact_slices": 4,
                    "intensity_percentiles": [1.0, 99.0],
                    "keep_intermediate_nifti": True,
                    "save_failed_qc": True,
                },
            }
            record = _process_case(
                config,
                FakeModel(),
                {"global_step": 10},
                torch.device("cpu"),
                image_path=image_path,
                mask_path=mask_path,
                output_dir=output,
                case_id="case_000",
                seed=42,
                histogram_library=np.ones((1, 16), dtype=np.float32) / 16,
            )
            self.assertEqual(record["case_id"], "case_000")
            for name in (
                "original.nii.gz",
                "generated_patch.nii.gz",
                "synthetic.nii.gz",
                "inserted_mask.nii.gz",
                "metadata.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            for name in (
                "01_mask_orthogonal.png",
                "02_generated_lesion.png",
                "03_before_after_full.png",
                "04_before_after_zoom.png",
                "05_multislice_axial.png",
                "06_intensity_qc.png",
                "comparison.png",
            ):
                self.assertTrue((output / "figures" / name).is_file(), name)

            original = nib.load(image_path)
            synthetic = nib.load(output / "synthetic.nii.gz")
            inserted = nib.load(output / "inserted_mask.nii.gz")
            self.assertEqual(original.shape, synthetic.shape)
            self.assertTrue(np.allclose(original.affine, synthetic.affine))
            self.assertTrue(np.allclose(original.affine, inserted.affine))
            synthetic_data = synthetic.get_fdata(dtype=np.float32)
            self.assertTrue(np.array_equal(synthetic_data[mask == 0], image[mask == 0]))


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
