# 1. Project explanation

## In simple language
Doctors look at X-rays to find signs of disease. An AI model can be shown thousands of X-rays that
doctors already labelled ("normal" or "pneumonia"). It slowly learns which visual patterns go with
which label. Later, when it sees a new X-ray, it gives a probability, e.g. "87% likely pneumonia".
It does not replace the doctor. It is a **second pair of eyes** that can also put suspicious scans at
the top of the reading queue.

## In technical language
Binary image classification with a convolutional neural network (CNN). Convolution layers learn
spatial filters (edges -> textures -> lung opacities). We use **transfer learning**: MobileNetV2,
pre-trained on ImageNet, supplies general visual features; we train a new head (global average pooling,
dropout, one sigmoid unit), then fine-tune the last backbone layers at a low learning rate. The output
is P(PNEUMONIA); a threshold turns it into a label. Loss: binary cross-entropy with class weights.
Evaluation: confusion matrix, sensitivity, specificity, precision, F1, ROC-AUC. Explainability: Grad-CAM.

## Problems it addresses
- **Workload and delay**: urgent studies can wait behind routine ones -> AI-based prioritisation.
- **Fatigue and variability**: readers can miss subtle findings or disagree -> consistent second reader.
- **Access**: places with few radiologists -> pre-screening support (always with human sign-off).

## Why it matters by setting
| Setting | Value |
|---|---|
| Hospitals | Triage of ER/ICU chest films; second-reader safety net |
| Diagnostic labs | Higher throughput, consistent quality checks |
| Radiology centres | Worklist prioritisation, fewer missed findings |
| Health-tech companies | Product category with real regulatory and engineering challenges |

## How AI helps (and its limits)
- **Disease detection**: pneumonia, nodules/tumours, fractures, haemorrhage - each needs its own labelled data and validation. This repo covers pneumonia only.
- **Faster diagnosis**: inference takes milliseconds to seconds; the rest of the clinical process is the bottleneck. (Measure your own latency with `python main.py simulate`.)
- **Reducing human error**: can catch misses, but also makes its own errors; the best setting is human + AI.
- **Assisting doctors**: heatmaps and probabilities support, never replace, the clinical decision.

## The workflow (what the code does)
1. **Data collection** - public Kaggle chest X-rays (`python main.py check-data`).
2. **Preprocessing** - decode, resize to 224x224, 3 channels; scaling lives inside the model.
3. **Feature extraction** - convolutional backbone learns the features automatically.
4. **Model training** - class-weighted loss, augmentation, 2-phase fine-tuning (`train`).
5. **Prediction** - probability + label + confidence (`predict`, API).
6. **Evaluation** - held-out test set, confusion matrix, ROC (`evaluate`).
7. **Visualisation** - training curves, prediction grids, error analysis, Grad-CAM.
