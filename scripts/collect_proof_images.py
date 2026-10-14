"""Copy the figures the README expects from outputs/ into images/ (the committed folder).

Run AFTER you trained on the real dataset:  python main.py train && python main.py evaluate
                                            python main.py predict --image <folder> --gradcam
                                            python main.py explore && python main.py simulate
Usage:  python scripts/collect_proof_images.py [--outputs outputs] [--images images]
"""

import argparse
import shutil
import sys
from pathlib import Path

# (source path relative to outputs/, required for the README?)
FIGURES = [
    ("training_curves.png", True),
    ("confusion_matrix.png", True),
    ("roc_curve.png", True),
    ("predictions_sample.png", True),
    ("predictions/gradcam_predictions.png", True),
    ("simulation/simulation_dashboard.png", True),
    ("class_distribution.png", False),
    ("sample_images.png", False),
    ("preprocessing_demo.png", False),
    ("augmentation_demo.png", False),
    ("misclassified.png", False),
    ("simulation/simulation_gradcam_top_cases.png", False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--images", default="images")
    a = ap.parse_args()
    out, dst = Path(a.outputs), Path(a.images)
    dst.mkdir(parents=True, exist_ok=True)

    missing_required = []
    for rel, required in FIGURES:
        src = out / rel
        if src.is_file():
            shutil.copy2(src, dst / src.name)
            print(f"copied  {rel} -> {dst / src.name}")
        else:
            print(f"missing {rel}" + ("  (needed by README)" if required else ""))
            if required:
                missing_required.append(rel)

    if missing_required:
        print("\nRun the matching commands first (train, evaluate, explore, predict --gradcam, simulate).")
        return 1
    print("\nAll README images are in place. Only commit figures from YOUR real-data run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
