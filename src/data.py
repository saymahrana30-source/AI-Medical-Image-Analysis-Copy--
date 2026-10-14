"""Dataset discovery, validation and tf.data loading.

Expected layout (the Kaggle "Chest X-Ray Images (Pneumonia)" layout):

    chest_xray/
        train/NORMAL  train/PNEUMONIA
        val/NORMAL    val/PNEUMONIA     (only 16 images - we do not rely on it)
        test/NORMAL   test/PNEUMONIA
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------------------ discovery
def _looks_like_dataset(path: Path) -> bool:
    return all(
        (path / split / cls).is_dir()
        for split in ("train", "test")
        for cls in config.CLASS_NAMES
    )


def resolve_data_dir(data_dir=None) -> Path:
    """Return the folder that directly contains train/ and test/.

    The Kaggle zip often extracts to chest_xray/chest_xray/..., so we also look
    one and two levels down instead of failing with a confusing error.
    """
    base = Path(data_dir) if data_dir else config.DATA_DIR
    candidates = [base, base / "chest_xray", base / "chest_xray" / "chest_xray"]
    for candidate in candidates:
        if candidate.is_dir() and _looks_like_dataset(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find the dataset under '{base}'.\n"
        "Expected train/NORMAL, train/PNEUMONIA, test/NORMAL and test/PNEUMONIA.\n"
        "Download it from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia "
        "and unzip it into the data/ folder (see README, step 3)."
    )


def list_images(folder: Path):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def count_images(data_dir) -> dict:
    """{'train': {'NORMAL': n, 'PNEUMONIA': n}, 'val': {...}, 'test': {...}}"""
    data_dir = Path(data_dir)
    counts = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        counts[split] = {
            cls: len(list_images(split_dir / cls)) if (split_dir / cls).is_dir() else 0
            for cls in config.CLASS_NAMES
        }
    return counts


# ------------------------------------------------------------------ tf.data
@dataclass
class Datasets:
    train: object
    val: object
    test: object
    test_files: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    class_weight: dict = field(default_factory=dict)


def compute_class_weights(train_counts: dict) -> dict:
    """Balanced weights: the rarer class counts for more during training.

    The Kaggle training set is ~74% PNEUMONIA, so without this a model can score
    well by leaning towards "pneumonia".
    """
    n = [train_counts[c] for c in config.CLASS_NAMES]
    total = sum(n)
    if min(n) == 0:
        raise ValueError(f"Both classes need images in train/, got {train_counts}")
    return {i: total / (len(n) * n[i]) for i in range(len(n))}


def load_datasets(
    data_dir=None,
    img_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    val_split=config.VAL_SPLIT,
    seed=config.SEED,
) -> Datasets:
    """Build train / validation / test tf.data pipelines.

    * Images are decoded as 3-channel (grayscale X-rays are replicated) because
      MobileNetV2 expects RGB. Pixels stay in 0-255; scaling happens INSIDE the model.
    * train/val come from ONE call with the same seed, so they never overlap.
    * test is never shuffled, so predictions line up with Datasets.test_files.
    """
    import tensorflow as tf

    root = resolve_data_dir(data_dir)
    counts = count_images(root)

    common = dict(
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        image_size=tuple(img_size),
        batch_size=batch_size,
    )

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        root / "train",
        validation_split=val_split,
        subset="both",
        seed=seed,
        shuffle=True,
        **common,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        root / "test", shuffle=False, **common
    )
    test_files = list(test_ds.file_paths)  # must be read before .prefetch()

    autotune = tf.data.AUTOTUNE
    return Datasets(
        train=train_ds.prefetch(autotune),
        val=val_ds.prefetch(autotune),
        test=test_ds.prefetch(autotune),
        test_files=test_files,
        counts=counts,
        class_weight=compute_class_weights(counts["train"]),
    )


def labels_from_dataset(ds) -> np.ndarray:
    """Collect labels (0/1) in dataset order as a flat int array."""
    return np.concatenate([y.numpy().reshape(-1) for _, y in ds]).astype(int)


def load_test_dataset(data_dir=None, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE):
    """Test split only (unshuffled). Returns (dataset, file_paths_in_order)."""
    import tensorflow as tf

    root = resolve_data_dir(data_dir)
    ds = tf.keras.utils.image_dataset_from_directory(
        root / "test",
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        image_size=tuple(img_size),
        batch_size=batch_size,
        shuffle=False,
    )
    files = list(ds.file_paths)
    return ds.prefetch(tf.data.AUTOTUNE), files
