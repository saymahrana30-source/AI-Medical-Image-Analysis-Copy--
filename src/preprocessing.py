"""Image preprocessing used by training, evaluation, the CLI and the API.

ONE shared code path for decoding + resizing avoids train/serve skew: the
bytes -> pixels logic here mirrors what tf.keras.utils.image_dataset_from_directory
does, and tests/test_pipeline.py checks the two agree numerically.
"""

from pathlib import Path

import cv2
import numpy as np

from . import config


def decode_image_bytes(data: bytes, img_size=config.IMG_SIZE) -> np.ndarray:
    """Decode JPEG/PNG/BMP bytes -> float32 array (H, W, 3) in the 0-255 range.

    Grayscale input is replicated to 3 channels, exactly like training did.
    Raises ValueError for anything that is not a decodable image.
    """
    import tensorflow as tf

    if not data:
        raise ValueError("Empty file.")
    try:
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError as exc:
        raise ValueError("File is not a valid JPEG/PNG/BMP image.") from exc
    img = tf.image.resize(img, tuple(img_size), method="bilinear")
    return img.numpy().astype("float32")


def preprocess_bytes(data: bytes, img_size=config.IMG_SIZE) -> np.ndarray:
    """Bytes -> model-ready batch of shape (1, H, W, 3)."""
    return decode_image_bytes(data, img_size)[np.newaxis, ...]


def preprocess_path(path, img_size=config.IMG_SIZE) -> np.ndarray:
    """Image file -> model-ready batch of shape (1, H, W, 3)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return preprocess_bytes(path.read_bytes(), img_size)


def build_augmentation():
    """Light, anatomically sensible augmentation (active only while training).

    Deliberately NO horizontal flip: in a chest X-ray the heart sits on the
    patient's left, so a mirrored image is not a realistic X-ray.
    """
    import keras
    from keras import layers

    return keras.Sequential(
        [
            layers.RandomRotation(0.03, name="aug_rotate"),  # +-0.03 * 360 = ~+-11 degrees
            layers.RandomTranslation(0.05, 0.05, name="aug_shift"),
            layers.RandomZoom(0.1, name="aug_zoom"),
            layers.RandomContrast(0.1, name="aug_contrast"),
        ],
        name="augmentation",
    )


# ------------------------------------------------------------ display helpers
def load_gray_uint8(path, img_size=None) -> np.ndarray:
    """Read an image as 8-bit grayscale (for plots and Grad-CAM overlays)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"OpenCV could not read image: {path}")
    if img_size is not None:
        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    return img


def apply_clahe(gray_uint8: np.ndarray, clip_limit=2.0, tile=8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation (optional enhancement).

    Used only in the preprocessing *demo figure*; it is NOT part of the model input.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray_uint8)
