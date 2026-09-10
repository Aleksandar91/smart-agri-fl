"""Build a reproducible, capped PlantVillage training subset from a raw
PlantVillage extraction (e.g. the Kaggle "New Plant Diseases Dataset" mirror
mounted at ``--raw-root``).

Two things this script does that plain ``cp -r`` doesn't:

1. **Canonical renaming.** The common Kaggle mirror mixes naming styles
   across crops (``Apple___Apple_scab`` with a triple underscore, but
   ``Tomato_Bacterial_spot`` / ``Tomato__Target_Spot`` without one, and
   ``Pepper,_bell___...`` with a comma). This script renames every class to
   a single consistent ``Crop___Disease`` scheme (see ``CANONICAL_CLASSES``)
   so downstream tooling (partitioning, field_validate's ``FARM_TO_PV``,
   reports) doesn't have to special-case each crop.
2. **Fixed per-class sampling.** Copies (does not symlink, for portability
   across Docker volumes) exactly ``--per-class`` images per class, chosen
   deterministically via ``--seed``, so the resulting subset size is
   predictable and comparable across dataset versions.

Usage:

    python -m app.build_pv_subset \\
        --raw-root /raw/PlantVillage \\
        --output-dir /data/fl-data-pv-v2 \\
        --per-class 300 \\
        --seed 1337
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

# raw folder name (as found under --raw-root) -> canonical "Crop___Disease" name.
# Only classes listed here are included in the subset - add a crop by adding
# its raw folder name(s) below.
CANONICAL_CLASSES: Dict[str, str] = {
    "Apple___Apple_scab": "Apple___Apple_scab",
    "Apple___Black_rot": "Apple___Black_rot",
    "Apple___Cedar_apple_rust": "Apple___Cedar_apple_rust",
    "Apple___healthy": "Apple___healthy",
    "Blueberry___healthy": "Blueberry___healthy",
    "Cherry_(including_sour)___healthy": "Cherry___healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry___Powdery_mildew",
    "Grape___Black_rot": "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)": "Grape___Esca",
    "Grape___healthy": "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape___Leaf_blight",
    "Pepper,_bell___Bacterial_spot": "Pepper___Bacterial_spot",
    "Pepper,_bell___healthy": "Pepper___healthy",
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___healthy": "Potato___healthy",
    "Potato___Late_blight": "Potato___Late_blight",
    "Squash___Powdery_mildew": "Squash___Powdery_mildew",
    "Tomato__Target_Spot": "Tomato___Target_Spot",
    "Tomato__Tomato_mosaic_virus": "Tomato___mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato_Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_Early_blight": "Tomato___Early_blight",
    "Tomato_healthy": "Tomato___healthy",
    "Tomato_Late_blight": "Tomato___Late_blight",
    "Tomato_Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato___Spider_mites",
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _list_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True, help="Raw PlantVillage extraction with one folder per class")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--classes",
        default=None,
        help="Comma-separated canonical class names to include (default: all of CANONICAL_CLASSES)",
    )
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    wanted_canonical = (
        set(s.strip() for s in args.classes.split(",")) if args.classes else None
    )

    report = {}
    missing_raw = []
    for raw_name, canonical in sorted(CANONICAL_CLASSES.items()):
        if wanted_canonical is not None and canonical not in wanted_canonical:
            continue
        src_dir = raw_root / raw_name
        if not src_dir.is_dir():
            missing_raw.append(raw_name)
            continue
        imgs = _list_images(src_dir)
        if not imgs:
            missing_raw.append(raw_name)
            continue
        chosen = imgs[:] if len(imgs) <= args.per_class else rng.sample(imgs, args.per_class)
        dst_dir = out_root / canonical
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_path in chosen:
            shutil.copy2(src_path, dst_dir / src_path.name)
        report[canonical] = {"available": len(imgs), "copied": len(chosen)}
        print(f"[build_pv_subset] {canonical}: {len(chosen)}/{len(imgs)} images")

    if missing_raw:
        print(f"[build_pv_subset] WARNING - raw folders not found or empty, skipped: {missing_raw}")

    total = sum(v["copied"] for v in report.values())
    print(f"[build_pv_subset] done: {len(report)} classes, {total} images -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
