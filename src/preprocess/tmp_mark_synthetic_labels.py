import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main(mask_dir: str, label_dir: str):
    mask_dir = Path(mask_dir)
    label_dir = Path(label_dir)

    mask_files = sorted(list(mask_dir.glob("*.nii.gz")) + list(mask_dir.glob("*.nii")))
    if not mask_files:
        raise FileNotFoundError(f"未找到 mask 文件: {mask_dir}")

    for m in mask_files:
        l = label_dir / m.name
        if not l.exists():
            print(f"[跳过] 找不到对应 label: {l.name}")
            continue

        m_img = nib.load(str(m))
        l_img = nib.load(str(l))

        m_data = m_img.get_fdata() > 0
        l_data = l_img.get_fdata().astype(np.int16)

        if m_data.shape != l_data.shape:
            print(f"[跳过] 尺寸不一致: {m.name}")
            continue

        l_data[m_data] = 2
        out = nib.Nifti1Image(l_data, l_img.affine, l_img.header)
        nib.save(out, str(l))
        print(f"[完成] {l.name}")

    print("全部处理完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 label 中 mask 区域标为 2")
    parser.add_argument("--mask_dir", required=True, help="mask 文件夹路径")
    parser.add_argument("--label_dir", required=True, help="label 文件夹路径")
    args = parser.parse_args()
    main(args.mask_dir, args.label_dir)