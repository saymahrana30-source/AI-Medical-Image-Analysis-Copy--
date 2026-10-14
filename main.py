"""Command-line entry point.

    python main.py check-data        verify the dataset is in place, print counts
    python main.py explore           sample grid, class distribution, preprocessing demo
    python main.py train             train the model (default: MobileNetV2 transfer learning)
    python main.py evaluate          metrics + confusion matrix + ROC on the test set
    python main.py predict --image X predict one image or a folder (add --gradcam)
    python main.py simulate          virtual hospital triage simulation
    python main.py serve             start the Flask API on http://127.0.0.1:5000
"""

import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")  # figures are saved to files, no display needed
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from src import config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI-Powered Medical Image Analysis")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, data=True, model=False):
        if data:
            sp.add_argument("--data-dir", default=str(config.DATA_DIR))
        if model:
            sp.add_argument("--model-path", default=str(config.MODEL_PATH))
        sp.add_argument("--outputs-dir", default=str(config.OUTPUTS_DIR))

    common(sub.add_parser("check-data", help="verify dataset + print counts"), data=True)
    common(sub.add_parser("explore", help="data exploration figures"), data=True)

    t = sub.add_parser("train", help="train a model")
    common(t, model=True)
    t.add_argument("--arch", choices=["mobilenetv2", "cnn"], default=config.DEFAULT_ARCH)
    t.add_argument("--epochs", type=int, default=8, help="phase 1 epochs (use ~25 for --arch cnn)")
    t.add_argument("--fine-tune-epochs", type=int, default=5, help="phase 2 (mobilenetv2 only)")
    t.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    t.add_argument("--val-split", type=float, default=config.VAL_SPLIT)
    t.add_argument("--no-pretrained", action="store_true", help="random weights (offline / tests)")
    t.add_argument("--seed", type=int, default=config.SEED)

    e = sub.add_parser("evaluate", help="evaluate on the test split")
    common(e, model=True)
    e.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    e.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)

    pr = sub.add_parser("predict", help="predict image(s)")
    common(pr, data=False, model=True)
    pr.add_argument("--image", required=True, help="image file or folder")
    pr.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    pr.add_argument("--gradcam", action="store_true", help="save Grad-CAM overlays")

    s = sub.add_parser("simulate", help="virtual hospital triage simulation")
    common(s, model=True)
    s.add_argument("--patients", type=int, default=60)
    s.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)

    sv = sub.add_parser("serve", help="run the Flask API")
    sv.add_argument("--model-path", default=str(config.MODEL_PATH))
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5000)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.command

    if cmd == "check-data":
        from src.data import count_images, resolve_data_dir

        root = resolve_data_dir(args.data_dir)
        counts = count_images(root)
        print(f"Dataset found at: {root}")
        print(json.dumps(counts, indent=2))
        if counts.get("val") and sum(counts["val"].values()) < 100:
            print("Note: the official val/ folder is tiny; training carves validation out of train/.")

    elif cmd == "explore":
        from pathlib import Path

        from src import visualize as viz
        from src.data import count_images, list_images, resolve_data_dir

        root = resolve_data_dir(args.data_dir)
        out = Path(args.outputs_dir)
        sample = list_images(root / "train" / "PNEUMONIA")[0]
        for path in (
            viz.plot_class_distribution(count_images(root), out / "class_distribution.png"),
            viz.plot_sample_grid(root, out / "sample_images.png"),
            viz.plot_preprocessing_demo(sample, out / "preprocessing_demo.png"),
            viz.plot_augmentation_demo(sample, out / "augmentation_demo.png"),
        ):
            print("saved", path)

    elif cmd == "train":
        from pathlib import Path

        from src.train import train
        from src.visualize import plot_training_history

        history = train(
            data_dir=args.data_dir, arch=args.arch, epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs, batch_size=args.batch_size,
            val_split=args.val_split, pretrained=not args.no_pretrained,
            model_path=args.model_path, outputs_dir=args.outputs_dir, seed=args.seed,
        )
        print("saved", plot_training_history(history, Path(args.outputs_dir) / "training_curves.png"))

    elif cmd == "evaluate":
        from src.evaluate import evaluate

        evaluate(args.data_dir, args.model_path, args.outputs_dir, args.batch_size, args.threshold)

    elif cmd == "predict":
        from pathlib import Path

        from src.predict import collect_images, load_trained_model, predict_paths

        model = load_trained_model(args.model_path)
        results = predict_paths(
            model, collect_images(args.image), args.threshold, args.gradcam,
            Path(args.outputs_dir) / "predictions",
        )
        for r in results[:50]:
            print(f"{r['image']}: {r['label']}  (P(pneumonia)={r['probability_pneumonia']:.3f}, "
                  f"confidence={r['confidence']:.1%})")
        if len(results) > 50:
            print(f"... and {len(results) - 50} more")
        print("\n" + config.DISCLAIMER)

    elif cmd == "simulate":
        from src.simulate import run_simulation

        run_simulation(args.data_dir, args.model_path, args.outputs_dir, args.patients, args.threshold)

    elif cmd == "serve":
        from src.api import create_app

        app = create_app(args.model_path)
        print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)\n{config.DISCLAIMER}")
        app.run(host=args.host, port=args.port, debug=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
