"""Generate a tiny synthetic ImageFolder dataset for smoke-testing the FL pipeline.

Creates N classes of images where each class has a distinct dominant color +
noise, so even one quick epoch reaches high accuracy. This lets us verify the
whole Flower loop (partitioning, rounds, aggregation, saving) in minutes,
without downloading PlantVillage.

Usage:
    python -m app.make_synth_dataset --output-dir /data-synth --classes 3 --per-class 60
"""

import argparse
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Make a synthetic ImageFolder dataset.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--per-class", type=int, default=60)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # Distinct base colors per class
    palette = [
        (200, 40, 40),
        (40, 180, 60),
        (50, 80, 200),
        (210, 190, 40),
        (160, 60, 190),
        (40, 190, 190),
    ]

    s = args.img_size
    for c in range(args.classes):
        cls_dir = out / f"synth_class_{c}"
        cls_dir.mkdir(exist_ok=True)
        base = np.array(palette[c % len(palette)], dtype=np.float32)
        for i in range(args.per_class):
            noise = rng.normal(0, 35, size=(s, s, 3)).astype(np.float32)
            img = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(cls_dir / f"img_{i:04d}.png")
        print(f"[synth] {cls_dir.name}: {args.per_class} images")

    print(f"[synth] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
