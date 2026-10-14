"""All figures. Each function saves a PNG (and returns the path) so the CLI can run headless."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .data import list_images
from .preprocessing import apply_clahe, build_augmentation, load_gray_uint8
from .utils import ensure_dir


def _save(fig, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if fig._suptitle is not None else None)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ data exploration
def plot_class_distribution(counts: dict, path) -> Path:
    splits = [s for s in ("train", "val", "test") if s in counts]
    x = np.arange(len(splits))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cls in enumerate(config.CLASS_NAMES):
        values = [counts[s][cls] for s in splits]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=cls)
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset class distribution (note the imbalance)")
    ax.legend()
    return _save(fig, path)


def plot_sample_grid(data_dir, path, n_per_class: int = 4, seed: int = config.SEED) -> Path:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6.4))
    for row, cls in enumerate(config.CLASS_NAMES):
        files = list_images(Path(data_dir) / "train" / cls)
        picks = rng.choice(len(files), size=min(n_per_class, len(files)), replace=False)
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(picks):
                ax.imshow(load_gray_uint8(files[picks[col]]), cmap="gray")
                ax.set_title(cls, fontsize=10)
    fig.suptitle("Sample chest X-rays from the training set", fontsize=13)
    return _save(fig, path)


def plot_preprocessing_demo(image_path, path, img_size=config.IMG_SIZE) -> Path:
    original = load_gray_uint8(image_path)
    resized = load_gray_uint8(image_path, img_size)
    enhanced = apply_clahe(resized)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title(f"1. Original {original.shape[1]}x{original.shape[0]}")
    axes[1].imshow(resized, cmap="gray")
    axes[1].set_title(f"2. Resized {img_size[1]}x{img_size[0]} (model input)")
    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("3. CLAHE contrast (optional,\nNOT used by the model)")
    axes[3].hist(resized.ravel(), bins=64, color="steelblue", alpha=0.75, label="resized")
    axes[3].hist(enhanced.ravel(), bins=64, color="darkorange", alpha=0.55, label="CLAHE")
    axes[3].set_title("4. Pixel intensity histogram")
    axes[3].legend()
    for ax in axes[:3]:
        ax.axis("off")
    return _save(fig, path)


def plot_augmentation_demo(image_path, path, img_size=config.IMG_SIZE, n: int = 7) -> Path:
    import tensorflow as tf

    from .preprocessing import preprocess_path

    batch = tf.convert_to_tensor(preprocess_path(image_path, img_size))
    aug = build_augmentation()
    fig, axes = plt.subplots(1, n + 1, figsize=(2.6 * (n + 1), 3.2))
    axes[0].imshow(batch[0].numpy().astype("uint8")[..., 0], cmap="gray")
    axes[0].set_title("original")
    for i in range(1, n + 1):
        out = aug(batch, training=True)[0].numpy()
        axes[i].imshow(np.clip(out, 0, 255).astype("uint8")[..., 0], cmap="gray")
        axes[i].set_title(f"augmented {i}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Data augmentation (rotation, shift, zoom, contrast - no flips)")
    return _save(fig, path)


# ------------------------------------------------------------------ training / evaluation
def plot_training_history(history: dict, path) -> Path:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (key, title) in zip(
        axes, [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "ROC-AUC")]
    ):
        if key in history:
            ax.plot(epochs, history[key], marker="o", label="train")
        if f"val_{key}" in history:
            ax.plot(epochs, history[f"val_{key}"], marker="o", label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch (phase 1 + fine-tuning combined)")
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(cm, path, title="Confusion matrix (test set)") -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(config.CLASS_NAMES)
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=16)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_roc(fpr, tpr, auc, path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    label = f"AUC = {auc:.3f}" if auc is not None else "AUC undefined"
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_prediction_grid(files, y_true, probs, path, threshold=config.DECISION_THRESHOLD,
                         max_images: int = 16, only_wrong: bool = False,
                         title="Predicted vs actual") -> Path:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).reshape(-1)
    y_pred = (probs >= threshold).astype(int)
    idx = np.arange(len(files))
    if only_wrong:
        idx = idx[y_pred != y_true]
    if len(idx) > max_images:
        idx = np.random.default_rng(config.SEED).choice(idx, size=max_images, replace=False)
    cols = 4
    rows = max(1, int(np.ceil(max(len(idx), 1) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(load_gray_uint8(files[i]), cmap="gray")
        ok = y_pred[i] == y_true[i]
        ax.set_title(
            f"Actual: {config.CLASS_NAMES[y_true[i]]}\nPred: {config.CLASS_NAMES[y_pred[i]]} "
            f"(P={probs[i]:.2f})",
            fontsize=9, color="green" if ok else "red",
        )
    if len(idx) == 0:
        axes[0, 0].text(0.5, 0.5, "No images to show", ha="center")
    fig.suptitle(title, fontsize=13)
    return _save(fig, path)


def plot_gradcam_grid(items, path, title="Grad-CAM: where the model looked") -> Path:
    """items: list of (overlay_rgb_uint8, caption, is_correct)."""
    cols = min(4, max(1, len(items)))
    rows = int(np.ceil(max(len(items), 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (overlay, caption, ok) in zip(axes.ravel(), items):
        ax.imshow(overlay)
        ax.set_title(caption, fontsize=9, color="green" if ok else "red")
    fig.suptitle(title, fontsize=13)
    return _save(fig, path)
