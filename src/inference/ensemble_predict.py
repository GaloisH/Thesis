import os
import glob
import argparse
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

def resample_prob(prob_tensor, target_shape):
    if prob_tensor.shape[2:] != tuple(target_shape):
        prob_tensor = F.interpolate(
            prob_tensor.float(), 
            size=target_shape, 
            mode='trilinear', 
            align_corners=False
        )
    return prob_tensor

def main():
    parser = argparse.ArgumentParser(description="Ensemble nnUNet and SwinUNETR softmax probabilities.")
    parser.add_argument("--nnunet_dir", type=str, default="/root/autodl-tmp/Thesis/prediction_results/nnUNet/1", help="nnUNet概率图所在目录")
    parser.add_argument("--swin_dir", type=str, default="/root/autodl-tmp/Thesis/prediction_results/swinunetr/prob", help="SwinUNETR概率图所在目录")
    parser.add_argument("--raw_dir", type=str, default="/root/autodl-tmp/Thesis/datasets/nnUNet_raw/Dataset101_Meningioma/imagesTs", help="原始测试集数据目录")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/Thesis/prediction_results/ensemble", help="集成结果输出目录")
    parser.add_argument("--w_nnunet", type=float, default=0.7, help="nnUNet权重")
    parser.add_argument("--w_swin", type=float, default=0.3, help="SwinUNETR权重")
    parser.add_argument("--max_cases", type=int, default=None, help="最大处理的case数量，None表示全部处理")
    args = parser.parse_args()
    
    ensemble_seg_dir = os.path.join(args.output_dir, "seg")
    ensemble_prob_dir = os.path.join(args.output_dir, "prob")
    os.makedirs(ensemble_seg_dir, exist_ok=True)
    os.makedirs(ensemble_prob_dir, exist_ok=True)
    
    t1_images = sorted(glob.glob(os.path.join(args.raw_dir, "*_0000.nii.gz")))
    if len(t1_images) == 0:
        return

    processed_count = 0
    for t1_img_path in t1_images:
        if args.max_cases is not None and processed_count >= args.max_cases:
            break
            
        case_id = os.path.basename(t1_img_path).replace("_0000.nii.gz", "")
        # The nnUNet predict stem is exactly the basename replacing .nii.gz
        stem = os.path.basename(t1_img_path).replace(".nii.gz", "")
        
        print(f"Processing Ensemble: {case_id} (stem: {stem})")
        
        ref_nib = nib.load(t1_img_path)
        ref_affine = ref_nib.affine
        ref_shape = ref_nib.shape
        
        nn_prob_path = os.path.join(args.nnunet_dir, f"{stem}_softmax_prob.npy")
        if not os.path.exists(nn_prob_path):
            print(f"  [Skip] nnUNet prob missing: {nn_prob_path}")
            continue
            
        prob_nnunet = np.load(nn_prob_path).astype(np.float32)
        prob_nnunet_t = torch.from_numpy(prob_nnunet).unsqueeze(0)
        
        swin_candidates = glob.glob(os.path.join(args.swin_dir, f"{case_id}*prob.nii.gz"))
        if not swin_candidates:
            swin_candidates = glob.glob(os.path.join(args.swin_dir, case_id, f"{case_id}*prob.nii.gz"))
            
        if not swin_candidates:
            print(f"  [Skip] SwinUNETR prob missing for: {case_id}")
            continue
            
        swin_prob_path = swin_candidates[0]
        swin_nib = nib.load(swin_prob_path)
        prob_swin = swin_nib.get_fdata().astype(np.float32)
        
        if prob_swin.ndim == 4:
            if prob_swin.shape[-1] == 4:
                prob_swin = np.transpose(prob_swin, (3, 0, 1, 2))
        prob_swin_t = torch.from_numpy(prob_swin).unsqueeze(0)
        
        prob_nnunet_t = resample_prob(prob_nnunet_t, ref_shape)
        prob_swin_t = resample_prob(prob_swin_t, ref_shape)
        
        ensemble_prob_t = args.w_nnunet * prob_nnunet_t + args.w_swin * prob_swin_t
        ensemble_seg_np = torch.argmax(ensemble_prob_t, dim=1).squeeze(0).numpy().astype(np.uint8)
        
        seg_out_path = os.path.join(ensemble_seg_dir, f"{case_id}_seg.nii.gz")
        prob_out_path = os.path.join(ensemble_prob_dir, f"{case_id}_prob.nii.gz")
        
        nib.save(nib.Nifti1Image(ensemble_seg_np, ref_affine), seg_out_path)
        
        prob_save_np = ensemble_prob_t.squeeze(0).permute(1, 2, 3, 0).numpy()
        nib.save(nib.Nifti1Image(prob_save_np, ref_affine), prob_out_path)
        
        print(f"  -> Saved Seg: {seg_out_path}")
        print(f"  -> Saved Prob: {prob_out_path}")
        processed_count += 1

if __name__ == "__main__":
    main()
