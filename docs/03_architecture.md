# 3. Architecture

```
                 +------------------+
  Kaggle X-rays  |  data.py         |  folder labels, counts, class weights,
  (train/test) ->|  tf.data loader  |  train/val split (seeded), test (unshuffled)
                 +--------+---------+
                          | raw pixels 0-255, (224,224,3), label 0/1
                          v
                 +------------------+      +------------------------------+
                 |  model.py        |      |  preprocessing.py            |
                 |  augmentation -> |<-----|  same decode+resize used by  |
                 |  rescale ->      |      |  CLI, API, simulation        |
                 |  MobileNetV2 ->  |      +------------------------------+
                 |  GAP->Dropout->  |
                 |  Dense(1,sigmoid)|
                 +--------+---------+
                          | P(PNEUMONIA)
        +-----------------+---------------------+---------------------+
        v                 v                     v                     v
   train.py          evaluate.py           gradcam.py            api.py / simulate.py
  (callbacks,      (metrics, CM, ROC,     (heatmap on the       (REST /predict,
   fine-tuning)     error analysis)        pre-sigmoid logit)    triage worklist)
```

## Modules
| File | Responsibility |
|---|---|
| `src/config.py` | paths, class names, image size, thresholds |
| `src/data.py` | find/validate dataset, counts, class weights, `tf.data` pipelines |
| `src/preprocessing.py` | bytes/file -> model input; augmentation layers; display helpers |
| `src/model.py` | `build_model`, compile, fine-tuning unfreeze |
| `src/train.py` | two-phase training with checkpoint/early stopping/LR schedule |
| `src/evaluate.py` | metrics, confusion matrix, ROC, error analysis, `results_table.md` |
| `src/gradcam.py` | Grad-CAM heatmap + overlay |
| `src/predict.py` | load model, single/folder prediction |
| `src/simulate.py` | virtual hospital triage run |
| `src/api.py` | Flask API + upload page |
| `src/visualize.py` | every figure |

## Data flow decisions that prevent bugs
- **Scaling inside the model** -> training, CLI and API can never disagree on normalisation.
- **Shared decode/resize** -> a test (`test_train_serve_parity`) proves training and inference see identical pixels.
- **Test set never shuffled** -> predictions line up with labels and filenames.
- **Validation carved from train/** with a fixed seed (the official `val/` has only 16 images).
- **No horizontal flip** augmentation (anatomy is not left/right symmetric).
