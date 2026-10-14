# 9. What was fixed from the original draft

You asked for a project with no mistakes, so I checked your draft document line by line. Here is what
was wrong or risky and how this repo handles it.

## Code problems in the draft
| # | Problem in draft | Why it matters | Fix in this repo |
|---|---|---|---|
| 1 | `test_data` is used in evaluation but never created | `NameError` | `load_test_dataset()` in `src/data.py` |
| 2 | `flow_from_directory` shuffles by default; `y_true = test_data.classes` is compared with predictions | labels and predictions misaligned, so accuracy/confusion matrix are meaningless | test set loaded with `shuffle=False`; test checks counts match |
| 3 | `y_pred` has shape (N,1) while `y_true` has shape (N,) | sklearn errors or mis-scores | predictions flattened in `compute_metrics` |
| 4 | Model is never saved, but the Flask code loads `medical_ai_model.h5` | API crashes on start | `train` saves the best model to `models/medical_ai_model.keras` |
| 5 | Stray text `CopyEdit` inside the Flask code | `SyntaxError` | removed |
| 6 | Flask: no validation, `debug=True` | a non-image upload crashes the server (`cv2.resize` on `None`); debug mode is a security risk | validated uploads, size limit, JSON errors (400/413), `debug=False`, localhost by default |
| 7 | `horizontal_flip=True` | mirrored chest X-rays are anatomically unrealistic (heart side) | no flips |
| 8 | Same generator with augmentation easily reused for test data | test set gets augmented, inflating/deforming results | augmentation lives in the model and is off at inference |
| 9 | No validation set (official `val/` has only 16 images) | no reliable way to pick epochs | 15% of train used for validation, seeded |
| 10 | No class-imbalance handling (about 74% pneumonia) | "always pneumonia" scores well | class weights; sensitivity/specificity reported |
| 11 | Accuracy + raw confusion matrix only | hides false negatives | precision, recall, specificity, F1, ROC-AUC, error analysis |
| 12 | `ImageDataGenerator` | deprecated in current Keras | `image_dataset_from_directory` + Keras preprocessing layers |
| 13 | NIH Chest X-ray dataset suggested with the same folder code | NIH ChestX-ray14 is multi-label with a CSV, not NORMAL/PNEUMONIA folders | project uses the Kaggle dataset only; NIH listed as an advanced extension |
| 14 | Empty dataset links `()` | nothing to click | links in README |

## Factual claims in the draft to fix or remove before publishing
I did not independently verify every statistic. These are the ones I know are wrong, or that I could not
confirm and would not put on a GitHub README or say in an interview without a real source.
- **AlphaFold** predicts protein structures; it is not a medical-image diagnostic tool.
- **Freenome** develops blood-based tests, not imaging AI. IBM **divested Watson Health** in 2022, and Caption Health is now part of GE HealthCare. A long name-drop list invites follow-up questions you may not be able to answer.
- The breast-cancer study (McKinney et al., 2020) was published in **Nature**, not Nature Medicine. It reported absolute reductions of 5.7% (US) / 1.2% (UK) in false positives and 9.4% (US) / 2.7% (UK) in false negatives. Quote both settings, and note it does not mean AI beats radiologists everywhere.
- Unsourced or unverified: "$208B by 2030", "analyse in under 10 seconds (IBM)", "over 400 hospitals (WEF)", "lung cancer 5 years before symptoms (MIT Technology Review)", "COVID CT in 20 seconds vs 15-20 minutes (The Lancet)", "less than 1% of AI engineers (McKinsey)", salary figures. Treat them as **unverified**; delete them or replace with a source you opened and read.
- Interview answers said "datasets from Kaggle and NIH" and "outperformed radiologists / faster than traditional methods". Your project uses Kaggle only and has not been compared with radiologists. See [08_interview_qa.md](08_interview_qa.md) for accurate wording.

## Verified claims
- Dataset: Kaggle *Chest X-Ray Images (Pneumonia)*: 5,863 images (5,216 train, 624 test, 16 val), CC BY 4.0, from Kermany et al., Cell 2018.
