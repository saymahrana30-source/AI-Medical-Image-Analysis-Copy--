# Model card - chest X-ray pneumonia classifier

- **Intended use**: education and portfolio demonstration of an AI triage workflow.
- **Not intended for**: diagnosis, treatment decisions, screening of real patients, or any clinical use.
- **Model**: MobileNetV2 (ImageNet pre-trained) + global average pooling + dropout + sigmoid unit; input 224x224x3 raw pixels; output P(PNEUMONIA).
- **Training data**: Kaggle *Chest X-Ray Images (Pneumonia)* training split (pediatric patients, Guangzhou), 85% train / 15% validation, class-weighted loss.
- **Evaluation data**: official test split (624 images). Fill in your measured results from `outputs/results_table.md`.
- **Known limitations**: single source, pediatric only, possible patient overlap between random splits, no external validation, no calibration study, binary labels only, test set known to be harder than validation, can learn non-clinical shortcuts.
- **Ethical considerations**: false negatives (missed pneumonia) are the costliest error; a deployed system would need prospective validation, regulatory clearance, bias/subgroup analysis and human oversight.
- **Reproducibility**: seed 42; versions recorded in `outputs/metadata.json`.
