"""Generate a tiny SYNTHETIC chest-X-ray-like dataset for smoke tests and CI.

These images are drawings (noise + two 'lungs'; the 'PNEUMONIA' class gets cloudy
opacities). They have NOTHING to do with real medicine. Their only purpose is to prove
the code runs end-to-end without downloading the real dataset.
NEVER report results from this data as medical results.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_image(rng: np.random.Generator, pneumonia: bool, size: int = 128) -> np.ndarray:
    img = rng.normal(40, 12, (size, size)).astype(np.float32)
    mask = np.zeros_like(img)
    for cx in (size * 0.32, size * 0.68):
        cv2.ellipse(mask, (int(cx), int(size * 0.5)), (int(size * 0.17), int(size * 0.33)),
                    0, 0, 360, 1.0, -1)
    img += mask * 70  # brighter "air" regions
    if pneumonia:
        opacity = np.zeros_like(img)
        for _ in range(rng.integers(3, 6)):
            x, y = rng.integers(int(size * 0.2), int(size * 0.8), 2)
            cv2.circle(opacity, (int(x), int(y)), int(rng.integers(8, 16)), 1.0, -1)
        opacity = cv2.GaussianBlur(opacity, (0, 0), 5) * mask
        img += opacity * 90
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return np.clip(img, 0, 255).astype(np.uint8)


def build(root: Path, per_class=None, seed: int = 0) -> Path:
    per_class = per_class or {"train": 40, "val": 4, "test": 16}
    rng = np.random.default_rng(seed)
    base = Path(root) / "chest_xray"
    for split, n in per_class.items():
        for cls in ("NORMAL", "PNEUMONIA"):
            folder = base / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                size = int(rng.integers(100, 160))  # varied sizes, like the real data
                cv2.imwrite(str(folder / f"{cls.lower()}_{i:03d}.jpeg"),
                            make_image(rng, cls == "PNEUMONIA", size))
    return base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("Synthetic dataset written to", build(Path(a.out), seed=a.seed))
