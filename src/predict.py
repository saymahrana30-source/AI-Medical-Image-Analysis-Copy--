"""Load a trained model and predict on single images or folders."""

from pathlib import Path

import keras
import numpy as np

from . import config
from .data import IMAGE_SUFFIXES
from .gradcam import make_gradcam_heatmap, overlay_heatmap
from .preprocessing import load_gray_uint8, preprocess_path
from .utils import ensure_dir
from .visualize import plot_gradcam_grid


def load_trained_model(model_path=config.MODEL_PATH) -> keras.Model:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Train one first: python main.py train"
        )
    return keras.models.load_model(model_path)


def model_img_size(model: keras.Model):
    return tuple(model.input_shape[1:3])


def interpret(prob: float, threshold: float = config.DECISION_THRESHOLD) -> dict:
    is_pos = prob >= threshold
    return {
        "label": config.POSITIVE_CLASS if is_pos else "NORMAL",
        "probability_pneumonia": round(float(prob), 4),
        "confidence": round(float(prob if is_pos else 1.0 - prob), 4),
        "threshold": threshold,
    }


def predict_batch(model: keras.Model, batch: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) raw pixels -> (N,) probabilities of PNEUMONIA."""
    return np.asarray(model(batch, training=False)).reshape(-1)


def predict_image(model: keras.Model, path, threshold: float = config.DECISION_THRESHOLD) -> dict:
    batch = preprocess_path(path, model_img_size(model))
    result = interpret(predict_batch(model, batch)[0], threshold)
    result["image"] = str(path)
    return result


def predict_paths(model, paths, threshold=config.DECISION_THRESHOLD, gradcam=False, out_dir=None):
    """Predict many images. Optionally save a Grad-CAM grid to out_dir."""
    results, cam_items = [], []
    size = model_img_size(model)
    for p in paths:
        batch = preprocess_path(p, size)
        if gradcam:
            heat, prob = make_gradcam_heatmap(model, batch, threshold=threshold)
            overlay = overlay_heatmap(load_gray_uint8(p, size), heat)
        else:
            prob = predict_batch(model, batch)[0]
        res = interpret(prob, threshold)
        res["image"] = str(p)
        results.append(res)
        if gradcam:
            cam_items.append((overlay, f"{Path(p).name[:22]}\n{res['label']} ({res['confidence']:.0%})", True))
    if gradcam and out_dir and cam_items:
        ensure_dir(out_dir)
        plot_gradcam_grid(cam_items[:12], Path(out_dir) / "gradcam_predictions.png")
    return results


def collect_images(target) -> list:
    target = Path(target)
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"No images found in {target}")
        return files
    raise FileNotFoundError(f"Path not found: {target}")
