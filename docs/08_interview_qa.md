# 8. Interview Q&A (accurate wording)

**Q: Explain the project.**
I built a pipeline that classifies chest X-rays as normal or pneumonia using MobileNetV2 transfer learning on the public Kaggle dataset. It handles class imbalance, reports sensitivity/specificity/AUC on a held-out test set, explains predictions with Grad-CAM, serves predictions through a Flask API, and includes a simulation where AI re-orders a radiology worklist. It is an educational project, not a clinical tool.

**Q: Why transfer learning?**
About 5k images is small for training a deep CNN from scratch. ImageNet features (edges, textures) transfer well. I train the new head first, then fine-tune the last layers at a low learning rate and keep BatchNorm frozen.

**Q: Why not just report accuracy?**
The data is imbalanced (~74% pneumonia) and errors have different costs. A missed pneumonia (false negative) is worse than a false alarm, so I report sensitivity, specificity, precision, F1, ROC-AUC and the confusion matrix.

**Q: How did you handle class imbalance?**
Class weights in the loss, plus metrics that expose it. (Alternatives: oversampling, focal loss.)

**Q: How did you avoid data leakage?**
Test images are never used for training or model selection. Validation is a seeded split of the training data. Caveat I acknowledge: the dataset has multiple images per patient, so a random split can still leak; a patient-level split would be better.

**Q: Why no horizontal flip augmentation?**
Chest anatomy is not left-right symmetric (heart position), so flipped images are unrealistic.

**Q: How do you know the model looks at the right region?**
Grad-CAM. It helps spot shortcuts (e.g. markers, borders), but it is a sanity check, not proof.

**Q: How do you prevent train/serve skew?**
Rescaling is inside the model, and one shared decode+resize function is used by the CLI, API and simulation; a unit test checks it matches the training pipeline.

**Q: How did you deploy it?**
A Flask API with input validation, size limits and JSON errors, run locally. For production I would add authentication, logging, a proper WSGI server and Docker.

**Q: What are the limitations?**
Single pediatric dataset, no external validation, possible patient overlap in splits, binary task, not calibrated, not clinically validated.

**Q: Did it beat radiologists?**
No - I did not compare against radiologists. I only measured performance on a public test set.

**Q: What would you do next?**
Patient-level splits, external datasets (e.g. other hospitals), probability calibration, threshold selection for a target sensitivity, DICOM input, multi-label chest findings, experiment tracking and Docker.

**Q: Which companies work in this area?**
Examples of radiology-AI vendors include Aidoc, Qure.ai, Annalise.ai, Lunit and Oxipit, alongside large imaging companies (Siemens Healthineers, GE HealthCare, Philips). Say only what you actually know about them.
