from __future__ import annotations

import csv
import html
from pathlib import Path

from ..io import require_nibabel, require_numpy


def save_patch_nifti(data, ras_reference, start, path):
    np = require_numpy()
    nib = require_nibabel()
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = np.asarray(start, dtype=np.float64)
    affine = ras_reference.affine @ translation
    header = ras_reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine, header), str(path))


def write_case_index(output_dir: Path, records):
    csv_path = output_dir / "index.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("case_id", "qc_passed", "failures", "output_dir", "comparison"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record["case_id"],
                    "qc_passed": record["qc"]["passed"],
                    "failures": ";".join(record["qc"]["failures"]),
                    "output_dir": str(Path(record["outputs"]["synthetic"]).parent),
                    "comparison": record["outputs"]["figures"]["comparison"],
                }
            )
    cards = []
    for record in records:
        case = html.escape(record["case_id"])
        status = "PASS" if record["qc"]["passed"] else "FAIL"
        comparison = Path(record["outputs"]["figures"]["comparison"])
        relative = comparison.relative_to(output_dir).as_posix()
        failures = html.escape(", ".join(record["qc"]["failures"]) or "none")
        cards.append(
            f"<article><h2>{case} — {status}</h2>"
            f'<a href="{html.escape(relative)}"><img src="{html.escape(relative)}" '
            f'alt="Synthesis comparison for {case}"></a>'
            f"<p>QC failures: {failures}</p></article>"
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>LeFusion-H visualization index</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;max-width:1400px}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));"
        "gap:1.5rem}article{border:1px solid #bbb;padding:1rem}img{width:100%;height:auto}"
        "h1,h2{font-weight:500}</style><h1>LeFusion-H visualization index</h1><main>"
        + "".join(cards)
        + "</main></html>"
    )
    (output_dir / "index.html").write_text(document, encoding="utf-8")
