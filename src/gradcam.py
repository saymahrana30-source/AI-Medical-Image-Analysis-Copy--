"""Grad-CAM: highlights the image regions that pushed the model towards its decision.

It is a sanity-check tool, NOT proof the model is looking at the right anatomy.
If the heat sits on text markers or image corners, the model learned a shortcut.
"""

import cv2
import keras
import numpy as np
import tensorflow as tf

from .model import get_backbone


def make_gradcam_heatmap(model: keras.Model, batch: np.ndarray, explain: str = "predicted",
                         threshold: float = 0.5):
    """Return (heatmap in [0,1] shaped like the input, P(PNEUMONIA)).

    explain="predicted"  -> evidence for whichever class the model chose
    explain="pneumonia"  -> evidence for PNEUMONIA only
    The gradient is taken on the pre-sigmoid logit, because a saturated sigmoid
    (probability ~1.0) has near-zero gradient and would give a blank heatmap.
    """
    backbone = get_backbone(model)
    preprocess = model.get_layer("preprocess")
    gap = model.get_layer("gap")
    dense = model.get_layer("pred")

    x = tf.convert_to_tensor(batch, dtype=tf.float32)
    feats = backbone(preprocess(x), training=False)

    with tf.GradientTape() as tape:
        tape.watch(feats)  # needed: the frozen backbone's variables are not auto-watched
        pooled = gap(feats)
        logit = keras.ops.matmul(pooled, dense.kernel) + dense.bias  # (1, 1)
        prob = float(keras.ops.sigmoid(logit)[0, 0])
        sign = 1.0 if (explain == "pneumonia" or prob >= threshold) else -1.0
        score = sign * logit[:, 0]

    grads = tape.gradient(score, feats)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(feats * weights, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (batch.shape[2], batch.shape[1]), interpolation=cv2.INTER_CUBIC)
    return np.clip(cam, 0.0, 1.0), prob


def overlay_heatmap(gray_or_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """Blend a JET heatmap over an image. Returns an RGB uint8 array."""
    img = gray_or_rgb_uint8
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
