"""
Extract small lesion patches (≤24³) from nnUNet brain MRI data.
Supports both single-file and batch folder processing.
"""
import SimpleITK as sitk
from skimage.measure import regionprops, label
import numpy as np
import argparse
import json
from pathlib import Path

# ── constants ─────────────────────────────────────────────────
PATCH_SIZE = 32
HALF_PATCH = PATCH_SIZE // 2
SIZE_THRESHOLD = 24
MIN_AREA = 20
OUTPUT_DIR = Path("data")


# ── functions ────────────────────────────────────────────

def resample2ras(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "RAS")


def get_connected_components(mask: sitk.Image):
    """
    获取mask中所有连通分量
    """
    arr = sitk.GetArrayFromImage(mask)
    labeled = label(arr)
    return regionprops(labeled)


def filter_components(props: list, size_threshold: int = SIZE_THRESHOLD) -> list:
    """
    过滤连通分量，保留尺寸小于等于size_threshold的分量
    """
    kept = []
    for prop in props:
        z0, y0, x0, z1, y1, x1 = prop.bbox
        dz, dy, dx = z1 - z0, y1 - y0, x1 - x0
        if (0 < dz <= size_threshold and
            0 < dy <= size_threshold and
            0 < dx <= size_threshold and
            prop.area >= MIN_AREA):
            kept.append(prop)
    print(f"  lesions kept: {len(kept)} / {len(props)}  "
          f"(max dim ≤ {size_threshold}, area ≥ {MIN_AREA})")
    return kept


def extract_patch(img: sitk.Image, mask: sitk.Image,
                  centroid_zyx: tuple) -> tuple[sitk.Image, sitk.Image]:
    """提取以centroid为中心的PATCH_SIZE³的图像和mask块"""
    # numpy (z,y,x) → SimpleITK (x,y,z)
    cx, cy, cz = (int(centroid_zyx[2]),
                   int(centroid_zyx[1]),
                   int(centroid_zyx[0]))

    def clamped_start(c, dim):
        """
        计算ROI起始坐标，确保ROI在图像范围内
        """
        lo = c - HALF_PATCH
        if lo < 0:
            return 0
        if lo + PATCH_SIZE > dim:
            return dim - PATCH_SIZE
        return lo

    idx = (clamped_start(cx, img.GetSize()[0]),
           clamped_start(cy, img.GetSize()[1]),
           clamped_start(cz, img.GetSize()[2]))

    img_patch = sitk.RegionOfInterest(img, [PATCH_SIZE] * 3, idx)
    mask_patch = sitk.RegionOfInterest(mask, [PATCH_SIZE] * 3, idx)
    return img_patch, mask_patch


def process_one_case(img_path: Path, mask_path: Path,
                     out_dir: Path) -> list[dict]:
    """处理单个病例，提取所有小病灶块并保存"""
    print(f"\n  file: {img_path.name}")
    img = sitk.ReadImage(str(img_path))
    mask = sitk.ReadImage(str(mask_path))

    # 重采样至RAS方向
    img_ras = resample2ras(img)
    mask_ras = resample2ras(mask)
    # 获取并筛选连通分量
    props = get_connected_components(mask_ras)
    kept = filter_components(props)

    records = []
    # 提取每个小病灶块并保存
    stem = img_path.name.replace(".nii.gz", "").rsplit("_", 1)[0]

    for i, prop in enumerate(kept):
        img_patch, mask_patch = extract_patch(img_ras, mask_ras, prop.centroid)

        f_img = out_dir / f"{stem}_lesion{i:03d}_img.nii.gz"
        f_mask = out_dir / f"{stem}_lesion{i:03d}_mask.nii.gz"

        sitk.WriteImage(img_patch, str(f_img))
        sitk.WriteImage(mask_patch, str(f_mask))

        records.append({
            "lesion_id": i,
            "image_file": f_img.name,
            "mask_file": f_mask.name,
            "bbox": list(prop.bbox),
            "centroid": list(prop.centroid),
            "area": int(prop.area),
        })

    print(f"  → saved {len(records)} patch(es)")
    return records


# ── batch processing (folder) ─────────────────────────────────


def batch_process(images_dir: Path, labels_dir: Path,
                  out_dir: Path = OUTPUT_DIR) -> None:
    """
    批量处理文件夹中的图像和mask，提取所有小病灶块并保存
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    images = sorted(images_dir.glob("*.nii.gz"))
    print(f"Found {len(images)} image(s) in {images_dir}")

    for img_path in images:
        case_id = img_path.name.replace(".nii.gz", "").rsplit("_", 1)[0]
        mask_path = labels_dir / f"{case_id}.nii.gz"

        if not mask_path.exists():
            print(f"  [skip] {case_id}: no matching mask")
            continue

        try:
            records = process_one_case(img_path, mask_path, out_dir)
            all_records.extend(records)
        except Exception as e:
            print(f"  [error] {case_id}: {e}")

    summary = {
        "total_images": len(images),
        "total_lesions": len(all_records),
        "records": all_records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDone. {summary['total_lesions']} lesions extracted. "
          f"Summary → {out_dir / 'summary.json'}")


# ── CLI ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract small lesion patches from nnUNet brain MRI data.")

    # single-file mode
    parser.add_argument("--image", "-i", type=Path, default=None)
    parser.add_argument("--mask", "-m", type=Path, default=None)

    # batch mode
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)

    parser.add_argument("--output", "-o", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--size-threshold", type=int, default=SIZE_THRESHOLD)
    parser.add_argument("--min-area", type=int, default=MIN_AREA)

    args = parser.parse_args()

    # ── batch ──
    if args.images_dir and args.labels_dir:
        batch_process(args.images_dir.resolve(),
                       args.labels_dir.resolve(),
                       args.output.resolve())
        return

    # ── single ──
    if args.image and args.mask:
        args.output.mkdir(parents=True, exist_ok=True)
        process_one_case(args.image.resolve(),
                          args.mask.resolve(),
                          args.output.resolve())
        return

    parser.print_help()


if __name__ == "__main__":
    main()
