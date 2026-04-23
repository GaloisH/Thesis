import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def predict_single(
    image_paths: list[str],       # [t1, t1ce, t2, flair]
    output_dir: str,
    model_dir: str,
    fold: tuple = (0,),
    use_gaussian: bool = True,
    device: str = "cuda",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=use_gaussian,
        use_mirroring=True,
        device=torch.device(device),
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=fold)

    img = nib.load(image_paths[0])
    data = np.stack([nib.load(p).get_fdata(dtype=np.float32) for p in image_paths])  # (4, H, W, D)

    props = {"spacing": img.header.get_zooms()[:3]}

    seg, prob = predictor.predict_single_npy_array(
        input_image=data,
        image_properties=props,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=True,
    )

    sigmoid_prob = torch.sigmoid(torch.tensor(prob)).numpy()

    stem = Path(image_paths[0]).name.replace(".nii.gz", "").replace(".nii", "")
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), img.affine),
             f"{output_dir}/{stem}_seg.nii.gz")
    np.save(f"{output_dir}/{stem}_sigmoid_prob.npy", sigmoid_prob)

    print(f"Seg   -> {output_dir}/{stem}_seg.nii.gz")
    print(f"Prob  -> {output_dir}/{stem}_sigmoid_prob.npy  shape={sigmoid_prob.shape}")
    return seg, sigmoid_prob


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-i",  "--images",    required=True, nargs=4,
                   metavar=("T1","T1CE","T2","FLAIR"), help="4个模态路径")
    p.add_argument("-o",  "--output",    required=True)
    p.add_argument("-m",  "--model_dir", required=True)
    p.add_argument("-f",  "--fold",      default="0")
    p.add_argument("--device",           default="cuda")
    args = p.parse_args()

    folds = tuple(int(f) for f in args.fold.split(","))
    predict_single(args.images, args.output, args.model_dir, folds, device=args.device)