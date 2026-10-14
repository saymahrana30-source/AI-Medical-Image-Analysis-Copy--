"""Training loop: class weights, callbacks, optional fine-tuning, best-model saving."""

import time
from pathlib import Path

import keras

from . import config
from .data import load_datasets
from .model import build_model, unfreeze_for_fine_tuning
from .utils import ensure_dir, save_json, set_global_seed


def _callbacks(model_path: Path, csv_path: Path, initial_best=None):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            initial_value_threshold=initial_best,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=4, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(str(csv_path), append=initial_best is not None),
    ]


def _merge(h1: dict, h2: dict) -> dict:
    return {k: list(h1.get(k, [])) + list(h2.get(k, [])) for k in set(h1) | set(h2)}


def train(
    data_dir=None,
    arch: str = config.DEFAULT_ARCH,
    epochs: int = 8,
    fine_tune_epochs: int = 5,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    img_size=config.IMG_SIZE,
    pretrained: bool = True,
    model_path=config.MODEL_PATH,
    outputs_dir=config.OUTPUTS_DIR,
    seed: int = config.SEED,
) -> dict:
    """Train a model and write: the model, training_history.json, training_log.csv, metadata.json."""
    set_global_seed(seed)
    model_path = Path(model_path)
    outputs_dir = ensure_dir(outputs_dir)
    ensure_dir(model_path.parent)

    data = load_datasets(data_dir, img_size, batch_size, val_split, seed)
    print("Image counts:", data.counts)
    print("Class weights:", {config.CLASS_NAMES[k]: round(v, 3) for k, v in data.class_weight.items()})

    model = build_model(arch, img_size, pretrained)
    model.summary()

    started = time.time()
    csv_path = outputs_dir / "training_log.csv"

    print(f"\n=== Phase 1: training ({arch}) ===")
    h1 = model.fit(
        data.train,
        validation_data=data.val,
        epochs=epochs,
        class_weight=data.class_weight,
        callbacks=_callbacks(model_path, csv_path),
    ).history
    history = h1

    if arch == "mobilenetv2" and fine_tune_epochs > 0:
        best_so_far = min(h1.get("val_loss", [float("inf")]))
        n = unfreeze_for_fine_tuning(model)
        print(f"\n=== Phase 2: fine-tuning last layers ({n} trainable backbone tensors) ===")
        h2 = model.fit(
            data.train,
            validation_data=data.val,
            epochs=fine_tune_epochs,
            class_weight=data.class_weight,
            callbacks=_callbacks(model_path, csv_path, initial_best=best_so_far),
        ).history
        history = _merge(h1, h2)

    minutes = (time.time() - started) / 60
    save_json(history, outputs_dir / "training_history.json")
    save_json(
        {
            "arch": arch,
            "img_size": list(img_size),
            "class_names": config.CLASS_NAMES,
            "pretrained": pretrained,
            "epochs_phase1": epochs,
            "epochs_phase2": fine_tune_epochs if arch == "mobilenetv2" else 0,
            "batch_size": batch_size,
            "val_split": val_split,
            "seed": seed,
            "image_counts": data.counts,
            "class_weight": {config.CLASS_NAMES[k]: v for k, v in data.class_weight.items()},
            "training_minutes": round(minutes, 2),
            "tensorflow": __import__("tensorflow").__version__,
            "keras": keras.__version__,
        },
        outputs_dir / "metadata.json",
    )
    print(f"\nDone in {minutes:.1f} min. Best model saved to {model_path}")
    return history
