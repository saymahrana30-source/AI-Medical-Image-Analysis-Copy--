"""Virtual hospital simulation: an AI-assisted radiology triage queue.

Test images play the role of patients arriving at a radiology department. Their
folder label stands in for the radiologist-confirmed diagnosis. The model reads each
image, and the worklist is re-ordered so likely-pneumonia studies are reviewed first.
Everything reported here is MEASURED on your run - nothing is assumed.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import load_test_dataset
from .gradcam import make_gradcam_heatmap, overlay_heatmap
from .predict import load_trained_model, model_img_size
from .preprocessing import load_gray_uint8, preprocess_path
from .utils import ensure_dir, save_json
from .visualize import _save, plot_gradcam_grid


def triage_tier(prob: float, threshold: float) -> str:
    if prob >= max(0.85, threshold):
        return "URGENT"
    if prob >= threshold:
        return "REVIEW"
    return "ROUTINE"


def run_simulation(
    data_dir=None,
    model_path=config.MODEL_PATH,
    outputs_dir=config.OUTPUTS_DIR,
    n_patients: int = 60,
    threshold: float = config.DECISION_THRESHOLD,
    seed: int = config.SEED,
) -> dict:
    out = ensure_dir(Path(outputs_dir) / "simulation")
    model = load_trained_model(model_path)
    size = model_img_size(model)

    _, files = load_test_dataset(data_dir, size, batch_size=1)
    rng = np.random.default_rng(seed)
    n = min(n_patients, len(files))
    picks = rng.choice(len(files), size=n, replace=False)  # random arrival order

    rows = []
    for k, idx in enumerate(picks, start=1):
        path = Path(files[idx])
        actual = path.parent.name  # NORMAL / PNEUMONIA
        batch = preprocess_path(path, size)
        t0 = time.perf_counter()
        prob = float(np.asarray(model(batch, training=False)).reshape(-1)[0])
        ms = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "arrival_no": k,
                "patient_id": f"SIM-{k:04d}",
                "file": str(path),
                "radiologist_label": actual,
                "ai_probability": round(prob, 4),
                "ai_label": config.POSITIVE_CLASS if prob >= threshold else "NORMAL",
                "triage_tier": triage_tier(prob, threshold),
                "inference_ms": round(ms, 1),
            }
        )
    df = pd.DataFrame(rows)
    df["agrees"] = df["ai_label"] == df["radiologist_label"]
    queue = df.sort_values("ai_probability", ascending=False).reset_index(drop=True)
    queue.insert(0, "ai_queue_position", np.arange(1, len(queue) + 1))
    queue.to_csv(out / "worklist.csv", index=False)

    is_pos = (queue["radiologist_label"] == config.POSITIVE_CLASS).to_numpy()
    n_pos = int(is_pos.sum())
    fifo_positions = df.index[df["radiologist_label"] == config.POSITIVE_CLASS].to_numpy() + 1
    ai_positions = np.where(is_pos)[0] + 1
    summary = {
        "patients": int(n),
        "true_pneumonia_cases": n_pos,
        "ai_flagged": int((queue["ai_label"] == config.POSITIVE_CLASS).sum()),
        "agreement_with_radiologist": float(df["agrees"].mean()),
        "missed_cases_false_negatives": queue[
            is_pos & (queue["ai_label"] == "NORMAL")]["patient_id"].tolist(),
        "mean_queue_position_of_true_cases_fifo": float(fifo_positions.mean()) if n_pos else None,
        "mean_queue_position_of_true_cases_ai": float(ai_positions.mean()) if n_pos else None,
        "mean_inference_ms": float(df["inference_ms"].mean()),
        "median_inference_ms": float(df["inference_ms"].median()),
        "note": "Simulation on public test images; not a clinical evaluation.",
    }
    save_json(summary, out / "simulation_summary.json")

    # ---- dashboard
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    colors = ["#d62728" if p else "#1f77b4" for p in is_pos]
    axes[0].bar(range(1, len(queue) + 1), queue["ai_probability"], color=colors)
    axes[0].axhline(threshold, color="k", ls="--", lw=1)
    axes[0].set_title("AI-prioritised worklist\n(red = radiologist-confirmed pneumonia)")
    axes[0].set_xlabel("Queue position")
    axes[0].set_ylabel("P(pneumonia)")
    tiers = queue["triage_tier"].value_counts().reindex(["URGENT", "REVIEW", "ROUTINE"]).fillna(0)
    axes[1].bar(tiers.index, tiers.values, color=["#d62728", "#ff7f0e", "#2ca02c"])
    axes[1].set_title("Triage tiers")
    axes[1].set_ylabel("Patients")
    axes[2].hist(df["inference_ms"], bins=15, color="steelblue")
    axes[2].set_title(f"Inference time per image\n(mean {summary['mean_inference_ms']:.0f} ms, this machine)")
    axes[2].set_xlabel("milliseconds")
    _save(fig, out / "simulation_dashboard.png")

    # ---- Grad-CAM for the top flagged cases
    items = []
    for _, r in queue.head(8).iterrows():
        batch = preprocess_path(r["file"], size)
        heat, _ = make_gradcam_heatmap(model, batch, threshold=threshold)
        overlay = overlay_heatmap(load_gray_uint8(r["file"], size), heat)
        items.append((overlay, f"{r['patient_id']}  AI:{r['ai_label']} ({r['ai_probability']:.2f})\n"
                               f"Radiologist: {r['radiologist_label']}", bool(r["agrees"])))
    plot_gradcam_grid(items, out / "simulation_gradcam_top_cases.png",
                      title="Top of the AI worklist with Grad-CAM")

    print("\n=== Simulation summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nFiles written to {out}")
    return summary
