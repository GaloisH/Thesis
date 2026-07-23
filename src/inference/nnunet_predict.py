import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ==========================================
# 用户默认配置 (User Default Configuration)
# ==========================================
USER_CONFIG = {
    "image_dir": None,          
    "output": "./output",    # 默认输出路径
    "model_dir": "./model",  # 默认模型路径
    "fold": "0",             # 默认推理折数
    "device": "cuda"         # 默认运行设备
}
# ==========================================

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

    softmax_prob = torch.softmax(torch.tensor(prob), dim=0).numpy()

    stem = Path(image_paths[0]).name.replace(".nii.gz", "").replace(".nii", "")
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), img.affine),
             f"{output_dir}/{stem}_seg.nii.gz")
    np.save(f"{output_dir}/{stem}_softmax_prob.npy", softmax_prob)

    print(f"Seg   -> {output_dir}/{stem}_seg.nii.gz")
    print(f"Prob  -> {output_dir}/{stem}_softmax_prob.npy  shape={softmax_prob.shape}")
    return seg, softmax_prob

def predict_from_dir(
    image_dir: str,
    output_dir: str,
    model_dir: str,
    fold: tuple = (0,),
    use_gaussian: bool = True,
    device: str = "cuda",
):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=use_gaussian,
        use_mirroring=True,
        device=torch.device(device),
        verbose=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=fold)
    predictor.predict_from_files(image_dir,output_dir)

if __name__ == "__main__":
    predict_from_dir(USER_CONFIG["image_dir"], USER_CONFIG["output"], USER_CONFIG["model_dir"], fold=USER_CONFIG["fold"], device=USER_CONFIG["device"])