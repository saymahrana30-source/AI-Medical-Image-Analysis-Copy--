import io
import json

import cv2
import numpy as np
import pytest

from src import config
from src.data import compute_class_weights, count_images, load_datasets, load_test_dataset, resolve_data_dir
from src.evaluate import compute_metrics
from src.model import build_model, unfreeze_for_fine_tuning
from src.preprocessing import decode_image_bytes, preprocess_bytes, preprocess_path

SIZE = (96, 96)


# ------------------------------------------------------------------ data
def test_resolve_nested_dir(tmp_path):
    # Mimics the Kaggle zip layout data/chest_xray/chest_xray/train/... (no symlinks: they need
    # admin rights on Windows). build() creates <root>/chest_xray/{train,val,test}/...
    from make_synthetic_dataset import build

    nested_parent = tmp_path / "chest_xray"
    build(nested_parent, per_class={"train": 2, "val": 1, "test": 2})
    assert resolve_data_dir(tmp_path) == nested_parent / "chest_xray"


def test_resolve_missing_dir_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        resolve_data_dir(tmp_path / "nope")


def test_counts_and_class_weights(synthetic_root):
    counts = count_images(synthetic_root)
    assert counts["train"] == {"NORMAL": 40, "PNEUMONIA": 40}
    assert compute_class_weights({"NORMAL": 100, "PNEUMONIA": 300}) == {0: 2.0, 1: pytest.approx(2 / 3)}


def test_datasets_shapes_and_split(synthetic_root):
    d = load_datasets(synthetic_root, SIZE, batch_size=8)
    x, y = next(iter(d.train))
    assert x.shape[1:] == (*SIZE, 3) and y.shape[1:] == (1,)
    assert float(x.numpy().max()) > 1.0  # raw 0-255 pixels, scaling is inside the model
    assert len(d.test_files) == 32


# ------------------------------------------------------------------ preprocessing
def test_preprocess_formats_and_errors(synthetic_root):
    gray = np.random.default_rng(0).integers(0, 255, (50, 70), dtype=np.uint8)
    for ext in (".png", ".jpg"):
        ok, buf = cv2.imencode(ext, gray)
        out = preprocess_bytes(buf.tobytes(), SIZE)
        assert out.shape == (1, *SIZE, 3) and out.dtype == np.float32
        assert np.allclose(out[..., 0], out[..., 1])  # grayscale replicated to 3 channels
    with pytest.raises(ValueError):
        decode_image_bytes(b"this is not an image", SIZE)
    with pytest.raises(ValueError):
        decode_image_bytes(b"", SIZE)


def test_train_serve_parity(synthetic_root):
    """Pixels from the training pipeline == pixels from the inference pipeline."""
    ds, files = load_test_dataset(synthetic_root, SIZE, batch_size=4)
    x, _ = next(iter(ds))
    for i in range(4):
        np.testing.assert_allclose(x[i].numpy(), preprocess_path(files[i], SIZE)[0], atol=1e-3)


# ------------------------------------------------------------------ model
@pytest.mark.parametrize("arch", ["cnn", "mobilenetv2"])
def test_model_builds_and_is_deterministic_at_inference(arch):
    m = build_model(arch, SIZE, pretrained=False)
    x = np.random.default_rng(1).uniform(0, 255, (2, *SIZE, 3)).astype("float32")
    p1, p2 = m(x, training=False).numpy(), m(x, training=False).numpy()
    assert p1.shape == (2, 1) and ((p1 >= 0) & (p1 <= 1)).all()
    np.testing.assert_allclose(p1, p2)  # augmentation/dropout off at inference


def test_fine_tuning_keeps_batchnorm_frozen():
    m = build_model("mobilenetv2", SIZE, pretrained=False)
    assert len(m.get_layer("backbone").trainable_weights) == 0  # phase 1: frozen
    n = unfreeze_for_fine_tuning(m, n_layers=20)
    assert n > 0
    bn = [l for l in m.get_layer("backbone").layers if l.__class__.__name__ == "BatchNormalization"]
    assert bn and not any(l.trainable for l in bn)


# ------------------------------------------------------------------ metrics
def test_compute_metrics_known_values():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.2, 0.6, 0.4, 0.9, 0.8, 0.3, 0.7]
    m = compute_metrics(y, p, 0.5)
    assert m["tn_fp_fn_tp"] == [3, 1, 1, 3]
    assert m["accuracy"] == 0.75 and m["recall_sensitivity"] == 0.75 and m["specificity"] == 0.75
    assert compute_metrics([1, 1], [0.9, 0.8])["roc_auc"] is None  # one class only: no crash


# ------------------------------------------------------------------ end to end
def test_training_artifacts(trained):
    out = trained["outputs"]
    assert trained["model_path"].is_file()
    hist = json.loads((out / "training_history.json").read_text())
    assert len(hist["loss"]) == 2 and "val_auc" in hist
    assert (out / "metadata.json").is_file()


def test_evaluate_and_predict_and_gradcam(trained):
    from src.evaluate import evaluate
    from src.gradcam import make_gradcam_heatmap, overlay_heatmap
    from src.predict import load_trained_model, predict_image

    out = trained["outputs"]
    m = evaluate(trained["data"], trained["model_path"], out, batch_size=16)
    assert m["n_samples"] == 32 and sum(m["tn_fp_fn_tp"]) == 32
    for name in ("metrics.json", "confusion_matrix.png", "roc_curve.png", "test_predictions.csv",
                 "results_table.md", "predictions_sample.png", "misclassified.png"):
        assert (out / name).is_file(), name

    model = load_trained_model(trained["model_path"])
    files = sorted((trained["data"] / "test" / "PNEUMONIA").glob("*.jpeg"))
    res = predict_image(model, files[0])
    assert res["label"] in config.CLASS_NAMES and 0 <= res["probability_pneumonia"] <= 1

    batch = preprocess_path(files[0], SIZE)
    heat, prob = make_gradcam_heatmap(model, batch)
    assert heat.shape == SIZE and 0 <= heat.min() and heat.max() <= 1
    assert abs(prob - res["probability_pneumonia"]) < 1e-3
    assert overlay_heatmap(np.zeros(SIZE, np.uint8), heat).shape == (*SIZE, 3)


def test_simulation(trained):
    from src.simulate import run_simulation

    s = run_simulation(trained["data"], trained["model_path"], trained["outputs"], n_patients=12)
    assert s["patients"] == 12
    sim = trained["outputs"] / "simulation"
    for name in ("worklist.csv", "simulation_summary.json", "simulation_dashboard.png",
                 "simulation_gradcam_top_cases.png"):
        assert (sim / name).is_file(), name


# ------------------------------------------------------------------ API
def test_api(trained):
    from src.api import create_app

    client = create_app(trained["model_path"]).test_client()
    assert client.get("/health").get_json()["status"] == "ok"
    assert client.get("/").status_code == 200

    img = next((trained["data"] / "test" / "NORMAL").glob("*.jpeg"))
    r = client.post("/predict", data={"image": (io.BytesIO(img.read_bytes()), "x.jpeg")})
    body = r.get_json()
    assert r.status_code == 200 and body["label"] in config.CLASS_NAMES and "disclaimer" in body

    r = client.post("/predict?gradcam=true", data={"image": (io.BytesIO(img.read_bytes()), "x.jpeg")})
    assert r.status_code == 200 and r.get_json()["gradcam_png_base64"]

    assert client.post("/predict").status_code == 400  # no file
    assert client.post("/predict", data={"image": (io.BytesIO(b"junk"), "x.jpeg")}).status_code == 400
    assert client.post("/predict", data={"image": (io.BytesIO(b"junk"), "x.txt")}).status_code == 400
