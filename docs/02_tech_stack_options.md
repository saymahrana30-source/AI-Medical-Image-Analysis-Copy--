# 2. Tech stack options

| | **A - Easiest** | **B - Intermediate (SELECTED)** | **C - Advanced** |
|---|---|---|---|
| Tools | Python, Keras, Matplotlib | + OpenCV, scikit-learn, Flask, Grad-CAM | + PyTorch/MONAI, DICOM (pydicom), MLflow, Docker |
| Model | Small custom CNN | MobileNetV2 transfer learning + fine-tuning | EfficientNet/DenseNet, multi-label (e.g. NIH ChestX-ray14), U-Net segmentation |
| Dataset | Kaggle Chest X-Ray (Pneumonia) | Same + error analysis | NIH ChestX-ray14 / CheXpert / MIMIC-CXR (access terms apply), BraTS MRI |
| Difficulty | Beginner | Beginner+ / intermediate | Advanced |
| GPU | Not needed (slow but OK) | Helpful (Colab free GPU is enough) | Recommended / required |
| Output | Accuracy + confusion matrix | Metrics, ROC, Grad-CAM, API, simulation, tests | Research-grade pipeline, experiment tracking, deployment |

## Selection: Option B
It keeps the stack you asked for (Python, OpenCV, NumPy, Matplotlib, TensorFlow/Keras, lightweight
pre-trained model, public chest X-rays) while adding what makes a repo look professional: honest
metrics, explainability, serving and tests. It runs on Colab's free GPU in minutes and is still easy to
explain line by line in an interview. The custom-CNN baseline (`--arch cnn`) from Option A is included, so
you can show a baseline-vs-transfer-learning comparison.
