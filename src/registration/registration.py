"""Main entry for registration pipeline."""

from __future__ import annotations

import argparse
import os
import time

import registration_core as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registration and lesion probability mapping")
    parser.add_argument("--max-cases", type=int, default=int(os.getenv("MAX_CASES", "0")))
    parser.add_argument("--n-jobs", type=int, default=core.DEFAULT_CONFIG.n_jobs)
    parser.add_argument("--output-dir", type=str, default=core.DEFAULT_CONFIG.output_dir)
    parser.add_argument("--atlas-path", type=str, default="")
    parser.add_argument("--atlas-labels", type=str, default="")
    parser.add_argument("--region-prob-out", type=str, default="")
    parser.add_argument("--min-region-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70 + "\nRegistration Pipeline\n" + "=" * 70)

    cfg = core.Config(
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )

    dataset_json = os.path.join(cfg.dataset_dir, "dataset.json")
    if not os.path.exists(dataset_json):
        raise FileNotFoundError(f"dataset.json not found: {dataset_json}")

    case_ids = core.load_case_ids(cfg.labels_dir, max_cases=args.max_cases)
    if not case_ids:
        raise RuntimeError(f"No labels found in: {cfg.labels_dir}")

    if args.max_cases > 0:
        print(f"Debug mode: {len(case_ids)} cases")

    channel_idx = core.resolve_channel_index(dataset_json, cfg.modality)
    mni = core.resolve_mni_template()
    print(f"Modality: {cfg.modality}, channel: {channel_idx:04d}")

    start = time.time()
    core.build_maps(case_ids, cfg, channel_idx, mni)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed/60:.1f} min, Throughput: {len(case_ids)/(elapsed/60+1e-6):.2f} cases/min")

    prob_path = os.path.join(cfg.output_dir, "lesion_probability_map.nii.gz")
    region_csv = args.region_prob_out or os.path.join(cfg.output_dir, "region_probability_by_atlas.csv")

    atlas_path = args.atlas_path
    labels_path = args.atlas_labels
    if not atlas_path:
        atlas_path, labels_path = core.resolve_aal_atlas(cfg.output_dir)
        print(f"Using AAL atlas: {atlas_path}")

    rows = core.compute_region_probs(prob_path, atlas_path, region_csv, labels_path, args.min_region_size)
    print(f"Atlas done. Regions: {len(rows)}")
    for row in rows[:10]:
        print(f"Top: {row['region_name']} mean={row['mean_probability']:.4f}")

    print("Pipeline completed")


if __name__ == "__main__":
    main()


# python .\registration.py --max-cases 20 --atlas-path D:\python_code\projects\thesis\src\registration_simple\atlas_ready\AAL3v2.nii.gz --atlas-labels D:\python_code\projects\thesis\src\registration_simple\atlas_ready\AAL3v2.json