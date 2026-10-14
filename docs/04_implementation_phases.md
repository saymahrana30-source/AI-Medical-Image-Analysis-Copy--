# 4. Implementation phases

Each phase: **what / why / expected output / common mistakes / how to verify**.

## Phase 1 - Setup
- **What**: create the repo folder, virtual environment, `pip install -r requirements.txt`.
- **Why**: isolated, reproducible environment.
- **Output**: working `python -c "import tensorflow as tf; print(tf.__version__)"`.
- **Mistakes**: unsupported Python version (use 3.10-3.12); installing into the global Python; forgetting to activate the venv.
- **Verify**: `python -m pytest -q tests` passes.

## Phase 2 - Dataset collection
- **What**: download the Kaggle dataset into `data/`.
- **Why**: models learn from labelled examples.
- **Output**: `python main.py check-data` prints 5,216 / 624 / 16 images.
- **Mistakes**: nested `chest_xray/chest_xray` folder (auto-handled), `__MACOSX` junk, committing the dataset to Git.
- **Verify**: counts match; `python main.py explore` produces `sample_images.png` and `class_distribution.png`.

## Phase 3 - Image preprocessing
- **What**: decode, resize to 224x224, 3 channels, augmentation (rotation, shift, zoom, contrast).
- **Why**: the network needs a fixed input size; augmentation reduces over-fitting.
- **Output**: `preprocessing_demo.png`, `augmentation_demo.png`.
- **Mistakes**: horizontal flips on X-rays; scaling twice (once in the loader, once in the model); different preprocessing at inference.
- **Verify**: `test_train_serve_parity` passes; augmented images still look like plausible X-rays.

## Phase 4 - Model building
- **What**: `build_model("mobilenetv2")` (frozen backbone + new head) and `--arch cnn` baseline.
- **Why**: transfer learning works well with ~5k images.
- **Output**: model summary in the console.
- **Mistakes**: forgetting `include_top=False`; unfreezing BatchNorm layers; not recompiling after changing `trainable`.
- **Verify**: `test_model_builds_and_is_deterministic_at_inference`, `test_fine_tuning_keeps_batchnorm_frozen`.

## Phase 5 - Training
- **What**: `python main.py train` (phase 1 head, phase 2 fine-tune), class weights, early stopping on validation loss.
- **Why**: the training set is ~74% pneumonia, so class weights stop the model "cheating".
- **Output**: `models/medical_ai_model.keras`, `outputs/training_curves.png`, `training_history.json`, `training_log.csv`.
- **Mistakes**: judging by training accuracy; tuning on the test set; stopping too early.
- **Verify**: validation loss falls then flattens; no huge gap between train and validation curves.

## Phase 6 - Evaluation
- **What**: `python main.py evaluate` on the untouched test set.
- **Why**: honest estimate of performance on unseen images.
- **Output**: `metrics.json`, `results_table.md`, `confusion_matrix.png`, `roc_curve.png`, `misclassified.png`.
- **Mistakes**: quoting accuracy only; ignoring false negatives; evaluating with a shuffled test set.
- **Verify**: confusion-matrix counts sum to 624; sensitivity and specificity both reported.

## Phase 7 - Prediction
- **What**: `python main.py predict --image <file or folder> [--gradcam]`.
- **Why**: shows the model used like a tool.
- **Output**: label, P(pneumonia), confidence per image.
- **Mistakes**: using an image type the model never saw (e.g. a photo of a screen, CT, MRI) and trusting the answer.
- **Verify**: known test images give plausible outputs.

## Phase 8 - Visualisation
- **What**: prediction grids, error analysis, Grad-CAM, simulation dashboard.
- **Why**: communicates results and exposes shortcuts.
- **Output**: PNGs in `outputs/`.
- **Mistakes**: cherry-picking only correct predictions; reading Grad-CAM as proof of correctness.
- **Verify**: heat falls on lung fields; look at wrong predictions too.

## Phase 9 - GitHub publishing
- **What**: commit in small steps, push, add README + images. See [06_github_guide.md](06_github_guide.md).
- **Mistakes**: committing data/weights; vague commit messages; placeholder results left in README.
- **Verify**: clone the repo into a new folder and follow the README from scratch.

## Phase 10 - Final output
- **What**: tagged release, README with your real results, short demo video/GIF, model card.
- **Verify**: the proof checklist in [07_proof_plan.md](07_proof_plan.md) is fully ticked.
