"""Model definitions.

Both architectures share the same skeleton so training, evaluation and Grad-CAM
work identically for either:

    raw pixels (0-255)
      -> augmentation        (training only)
      -> preprocess          (rescaling, lives INSIDE the model)
      -> backbone            (MobileNetV2 transfer learning  OR  small custom CNN)
      -> gap                 (global average pooling)
      -> dropout
      -> pred                (Dense(1, sigmoid) = P(PNEUMONIA))

Because scaling lives inside the model, the saved model accepts raw 0-255 pixels
and the API cannot accidentally apply different scaling than training.
"""

import keras
from keras import layers

from . import config
from .preprocessing import build_augmentation

ARCHITECTURES = ("mobilenetv2", "cnn")


def _custom_cnn_backbone(input_shape) -> keras.Model:
    """A small 4-block CNN (feature extractor only)."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i, filters in enumerate((32, 64, 128, 128), start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    return keras.Model(inputs, x, name="backbone")


def _mobilenet_backbone(input_shape, pretrained: bool) -> keras.Model:
    return keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet" if pretrained else None,
        name="backbone",
    )


def build_model(
    arch: str = config.DEFAULT_ARCH,
    img_size=config.IMG_SIZE,
    pretrained: bool = True,
    dropout: float = 0.4,
) -> keras.Model:
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {ARCHITECTURES}, got '{arch}'")
    input_shape = (img_size[0], img_size[1], 3)

    inputs = keras.Input(shape=input_shape, name="image")
    x = build_augmentation()(inputs)

    if arch == "mobilenetv2":
        # MobileNetV2 expects inputs in [-1, 1]
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)
        backbone = _mobilenet_backbone(input_shape, pretrained)
        backbone.trainable = False  # phase 1: train only the new head
    else:
        x = layers.Rescaling(1.0 / 255.0, name="preprocess")(x)
        backbone = _custom_cnn_backbone(input_shape)

    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

    model = keras.Model(inputs, outputs, name=f"medical_{arch}")
    compile_model(model, learning_rate=1e-3)
    return model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def get_backbone(model: keras.Model) -> keras.Model:
    return model.get_layer("backbone")


def unfreeze_for_fine_tuning(
    model: keras.Model,
    n_layers: int = config.FINE_TUNE_LAYERS,
    learning_rate: float = 1e-5,
) -> int:
    """Phase 2 of transfer learning: unfreeze the last `n_layers` of MobileNetV2.

    BatchNorm layers stay frozen - fine-tuning them with small batches is a
    classic source of accuracy collapse. Returns the number of trainable weights
    tensors in the backbone.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, learning_rate=learning_rate)  # must recompile after changing trainable
    return len(backbone.trainable_weights)
