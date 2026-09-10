import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _iter_images(source: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if source.is_file():
        return [source]
    out: List[Path] = []
    for p in source.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


def _build_model(arch: str, num_classes: int):
    if arch != "mobilenet_v3_small":
        raise SystemExit(f"Unsupported arch {arch!r}. Supported: mobilenet_v3_small")

    import torch.nn as nn  # type: ignore
    from torchvision import models  # type: ignore

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model, weights


def _load_labels(labels_path: Path) -> Tuple[List[str], Dict[str, int]]:
    obj = json.loads(labels_path.read_text(encoding="utf-8"))
    return list(obj["classes"]), dict(obj["class_to_idx"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference for the plant disease classifier on images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--labels", type=str, default="", help="Path to labels.json (optional; can be read from ckpt).")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory to run inference on.")
    parser.add_argument("--topk", type=int, default=3, help="Top-K predictions to print.")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    import torch  # type: ignore
    from PIL import Image  # type: ignore
    from torchvision import transforms  # type: ignore

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    arch = ckpt.get("arch", "mobilenet_v3_small")
    classes = ckpt.get("classes")
    class_to_idx = ckpt.get("class_to_idx")
    img_size = int(ckpt.get("img_size", 224))

    if args.labels:
        lbl_path = Path(args.labels).expanduser().resolve()
        classes2, class_to_idx2 = _load_labels(lbl_path)
        classes = classes2
        class_to_idx = class_to_idx2

    if not classes or not class_to_idx:
        # Try default neighbor file: labels.json next to checkpoint
        neighbor = ckpt_path.parent / "labels.json"
        if neighbor.is_file():
            classes, class_to_idx = _load_labels(neighbor)
        else:
            raise SystemExit("Could not load labels/classes from checkpoint and labels.json not provided.")

    model, _ = _build_model(arch=arch, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Same ImageNet normalization as in training
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    tf = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    imgs = _iter_images(src)
    if not imgs:
        raise SystemExit(f"No images found under {src}")

    topk = max(1, int(args.topk))
    print(
        "[predict-plant-disease] starting",
        {"checkpoint": str(ckpt_path), "source": str(src), "images": len(imgs), "classes": len(classes), "topk": topk},
    )

    with torch.no_grad():
        for p in imgs:
            img = Image.open(p).convert("RGB")
            x = tf(img).unsqueeze(0)  # (1,C,H,W)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            k = min(topk, probs.shape[0])
            vals, idxs = torch.topk(probs, k=k)
            preds = [(classes[int(i)], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]
            print("[predict-plant-disease]", str(p), preds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

