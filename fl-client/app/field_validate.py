"""Field validation: run the FL-trained global model on real farm photos.

This is a *domain-gap* check, not a normal test-set evaluation. The model was
trained only on PlantVillage (PV) - lab photos of a single isolated leaf on a
plain background. Our farm photos are a different distribution, so we don't
expect PV-level accuracy; the point is to get a first honest number and see
where the model breaks.

Why "mapping" is needed
------------------------
Farm photos are organised as ``<species>/healthy`` or ``<species>/suspect``
(no specific disease name - we don't have lab-confirmed diagnoses). The PV
model instead outputs one of the *training* classes (27 classes / 8 crops
as of the v2 dataset expansion, see ``app.build_pv_subset.CANONICAL_CLASSES``).

To turn "the model said Apple___healthy" into "was that correct for this
farm photo", we need a lookup table that says, per farm species, which PV
classes count as a "healthy" match and which count as a "disease" match.
For an exact species match (Apple, Tomato, Cherry, Grape, Pepper) this is
straightforward. For species PV does not train a disease class for
(Blueberry) or does not train a healthy class for (Cucumber/Zucchini, via
the closest available proxy Squash___Powdery_mildew - same family, no PV
healthy-squash images exist at all), the mapping is necessarily incomplete -
this script reports those gaps explicitly instead of hiding them behind a
fake number.

Usage (run inside the same image used for training, e.g. infra_fl-client-0):

    python -m app.field_validate \\
        --checkpoint /models/global_latest.npz \\
        --partition-manifest /partitions/client_0.json \\
        --data-root /farm \\
        --img-size 128 \\
        --output-json /models/field_validation.json
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from app.fl_task import IMAGENET_MEAN, IMAGENET_STD, build_model, set_weights

# species (farm folder name) -> PV classes that count as "healthy" / "disease"
# for that species, given our current PV training subset (see
# app.build_pv_subset.CANONICAL_CLASSES - 27 classes / 8 crops as of the v2
# dataset expansion). Empty list == no matching PV class exists for that
# bucket (documented gap, not a bug).
FARM_TO_PV: Dict[str, Dict[str, List[str]]] = {
    "Apple": {
        "healthy": ["Apple___healthy"],
        "disease": ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust"],
    },
    "Tomato": {
        "healthy": ["Tomato___healthy"],
        "disease": [
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites",
            "Tomato___Target_Spot",
            "Tomato___Yellow_Leaf_Curl_Virus",
            "Tomato___mosaic_virus",
        ],
    },
    "Cherry": {
        "healthy": ["Cherry___healthy"],
        "disease": ["Cherry___Powdery_mildew"],
    },
    "Grape": {
        # No farm photos yet (planned) - kept here so field_validate/personalize
        # pick it up automatically once iot-edge/sensors/data/original/Grape/ exists.
        "healthy": ["Grape___healthy"],
        "disease": ["Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight"],
    },
    "Pepper": {
        # No farm photos yet (planned, e.g. paprika).
        "healthy": ["Pepper___healthy"],
        "disease": ["Pepper___Bacterial_spot"],
    },
    "Blueberry": {
        # PlantVillage has zero diseased-blueberry images (only healthy exists
        # in the whole dataset), so there is no PV class this could map to.
        "healthy": ["Blueberry___healthy"],
        "disease": [],
    },
    "Cucumber": {
        # PlantVillage has zero healthy-squash images either (Squash only has
        # the Powdery_mildew class). Cucumber != Squash botanically, but it's
        # the closest cucurbit proxy available in our subset.
        "healthy": [],
        "disease": ["Squash___Powdery_mildew"],
    },
    "Zucchini": {
        "healthy": [],
        "disease": ["Squash___Powdery_mildew"],
    },
    # "Potato": in the PV training subset (control crop, no farm counterpart -
    # no photos planned) so intentionally left out of FARM_TO_PV; it still
    # gets an anchor slot in app.personalize's catastrophic-forgetting check
    # automatically since that logic keys off "no farm training images", not
    # this dict.
}


@dataclass
class ImageResult:
    species: str
    label: str  # "healthy" or "suspect" (ground truth farm bucket)
    path: str
    top1_class: str
    top1_prob: float
    top3: List[Tuple[str, float]]
    bucket_match: str  # "healthy_match" | "disease_match" | "other_class" | "unmappable"


def load_checkpoint_ordered(npz_path: Path) -> List:
    """Load a Flower-saved .npz in the same arr_0, arr_1, ... order as get_weights()."""
    import numpy as np  # type: ignore

    data = np.load(str(npz_path))
    names = sorted(data.files, key=lambda n: int(n.split("_")[1]))
    return [data[n] for n in names]


def iter_farm_images(data_root: Path):
    """Yields (species, label, path) for every farm photo under <species>/<healthy|suspect>/*.jpg."""
    for species_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        species = species_dir.name
        for label in ("healthy", "suspect"):
            label_dir = species_dir / label
            if not label_dir.is_dir():
                continue
            for img_path in sorted(label_dir.glob("*.jpg")) + sorted(label_dir.glob("*.jpeg")):
                yield species, label, img_path


def build_eval_transform(img_size: int):
    """Deterministic (no augmentation) eval-time transform - same recipe as fl_task's val_tf."""
    from torchvision import transforms  # type: ignore

    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image_tensor(img_path: Path, img_size: int):
    """Opens + transforms one image, returns a (1, C, H, W) batch tensor."""
    from PIL import Image  # type: ignore

    tf = build_eval_transform(img_size)
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0)


def predict_topk(model, img_path: Path, classes: List[str], img_size: int, k: int = 3):
    import torch  # type: ignore

    x = load_image_tensor(img_path, img_size)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top_vals, top_idxs = torch.topk(probs, k=min(k, len(classes)))
    return [(classes[i], float(v)) for v, i in zip(top_vals.tolist(), top_idxs.tolist())]


def classify_bucket(species: str, top1_class: str) -> str:
    mapping = FARM_TO_PV.get(species)
    if mapping is None:
        return "unmappable"
    if top1_class in mapping["healthy"]:
        return "healthy_match"
    if top1_class in mapping["disease"]:
        return "disease_match"
    return "other_class"


def run(args: argparse.Namespace) -> Dict:
    manifest = json.loads(Path(args.partition_manifest).read_text(encoding="utf-8"))
    classes = list(manifest["classes"])

    model = build_model(num_classes=len(classes), freeze_backbone=False)
    weights = load_checkpoint_ordered(Path(args.checkpoint))
    set_weights(model, weights)
    model.eval()

    results: List[ImageResult] = []
    data_root = Path(args.data_root)
    n_seen = 0
    for species, label, img_path in iter_farm_images(data_root):
        if species not in FARM_TO_PV:
            continue  # e.g. Cherry: no PV classes at all in this subset
        top3 = predict_topk(model, img_path, classes, args.img_size)
        top1_class, top1_prob = top3[0]
        bucket = classify_bucket(species, top1_class)
        results.append(
            ImageResult(species, label, str(img_path), top1_class, top1_prob, top3, bucket)
        )
        n_seen += 1
        if args.limit_per_folder and n_seen % 50 == 0:
            print(f"  ... {n_seen} images processed")

    summary = _summarize(results)
    report = {
        "checkpoint": str(args.checkpoint),
        "img_size": args.img_size,
        "classes": classes,
        "n_images": len(results),
        "mapping_gaps": {
            sp: {k: v for k, v in m.items() if not v} for sp, m in FARM_TO_PV.items()
        },
        "skipped_species": sorted(
            {p.parent.parent.name for p in data_root.glob("*/*/*.jpg")}
            - set(FARM_TO_PV.keys())
        ),
        "summary": summary,
    }
    return report, results


def _summarize(results: List[ImageResult]) -> Dict:
    by_species_label = defaultdict(lambda: defaultdict(int))
    for r in results:
        by_species_label[(r.species, r.label)][r.bucket_match] += 1

    out = {}
    for (species, label), counts in sorted(by_species_label.items()):
        total = sum(counts.values())
        out[f"{species}/{label}"] = {
            "n": total,
            **{k: v for k, v in counts.items()},
        }
        expected = "healthy_match" if label == "healthy" else "disease_match"
        if expected in ("healthy_match",) and not FARM_TO_PV[species]["healthy"]:
            out[f"{species}/{label}"]["note"] = "no PV healthy class for this species (see mapping_gaps)"
        elif expected == "disease_match" and not FARM_TO_PV[species]["disease"]:
            out[f"{species}/{label}"]["note"] = "no PV disease class for this species (see mapping_gaps)"
        else:
            out[f"{species}/{label}"]["expected_match_rate"] = round(counts.get(expected, 0) / max(1, total), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to global_latest.npz (or any global_round_XXX.npz)")
    p.add_argument("--partition-manifest", required=True, help="Any client_*.json from the matching training run (for the class list)")
    p.add_argument("--data-root", required=True, help="Root of farm photos: <species>/{healthy,suspect}/*.jpg")
    p.add_argument("--img-size", type=int, default=128, help="MUST match the img size used to train the checkpoint (see docs/faza1-flower.md)")
    p.add_argument("--limit-per-folder", type=int, default=0, help="0 = no limit; else cap images per species/label folder")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    report, results = run(args)

    print(json.dumps(report["summary"], indent=2))
    print("\nMapping gaps (species/bucket with NO matching PV class):")
    for sp, gaps in report["mapping_gaps"].items():
        print(f"  {sp}: {gaps}")
    if report["skipped_species"]:
        print(f"\nSkipped entirely (no PV classes in current subset): {report['skipped_species']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
