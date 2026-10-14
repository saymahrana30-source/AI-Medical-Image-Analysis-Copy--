# 7. Step-by-step GitHub proof plan (7 days) + checklist

| Day | Goal | Commits (examples) | Proof to capture |
|---|---|---|---|
| 1 Setup | env + repo skeleton, tests pass | `chore: project skeleton`, `ci: add test workflow` | terminal showing tests passing; green CI badge |
| 2 Dataset | download, check, explore | `feat: dataset loader and validation`, `docs: dataset notes` | `class_distribution.png`, `sample_images.png` |
| 3 Preprocessing | pipeline + augmentation | `feat: preprocessing and augmentation` | `preprocessing_demo.png`, `augmentation_demo.png` |
| 4 Model | build MobileNetV2 + CNN baseline | `feat: model builders`, `test: model tests` | model summary screenshot |
| 5 Training | train (Colab GPU) | `feat: two-phase training`, `docs: training notes` | `training_curves.png`, training log |
| 6 Evaluation | metrics, Grad-CAM, simulation | `feat: evaluation`, `feat: grad-cam`, `feat: triage simulation` | confusion matrix, ROC, Grad-CAM, simulation dashboard |
| 7 Upload | README with real results, release | `docs: add results and limitations`, `docs: model card` | repo page screenshot, release `v1.0.0`, API demo GIF |

## Proof checklist
- [ ] Sample X-ray grid and class-distribution plot
- [ ] Preprocessing and augmentation figures
- [ ] Training logs (`training_log.csv`) and `training_curves.png`
- [ ] `confusion_matrix.png`, `roc_curve.png`, `results_table.md` pasted in README
- [ ] `predictions_sample.png` and `misclassified.png` (error analysis)
- [ ] Grad-CAM figure
- [ ] Simulation dashboard + worklist excerpt
- [ ] API screenshot or GIF (`python main.py serve`)
- [ ] Green CI badge, 10+ meaningful commits, tagged release
- [ ] README states dataset, licence, limitations and the not-a-medical-device disclaimer

## File naming and storage
Use the names the code generates (lowercase, underscores). Keep raw output in `outputs/` (ignored) and copy
selected figures to `images/` (committed) by running `python scripts/collect_proof_images.py`. Reference them in the README as `images/<name>.png`.
