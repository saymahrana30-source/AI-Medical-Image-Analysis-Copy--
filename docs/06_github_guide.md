# 6. GitHub upload strategy

**Repo name**: `AI-Medical-Image-Analysis` (or `chest-xray-pneumonia-ai`)
**Description**: `Chest X-ray pneumonia detection with MobileNetV2 transfer learning, Grad-CAM, Flask API and a virtual hospital triage simulation (TensorFlow/Keras)`
**Topics**: `deep-learning`, `medical-imaging`, `computer-vision`, `tensorflow`, `keras`, `transfer-learning`, `grad-cam`, `healthcare-ai`, `flask`, `python`

## Steps
1. Create an empty repo on github.com (no README, no .gitignore - we already have them). Add the MIT license name in `LICENSE`.
2. In the project folder:
```bash
git init
git branch -M main
git add .gitignore README.md LICENSE requirements.txt requirements-dev.txt
git commit -m "chore: project skeleton, requirements and README draft"
git remote add origin https://github.com/<your-username>/AI-Medical-Image-Analysis.git
git push -u origin main
```
3. Continue with small commits (see [07_proof_plan.md](07_proof_plan.md)).
4. Before every push check `git status` - **no** `data/`, `*.keras`, or `kaggle.json` should be listed.

## Commit style
Small, one idea each, imperative message: `feat: add Grad-CAM`, `fix: keep test set unshuffled`, `docs: add results`, `test: add API tests`, `chore: ...`.

## Images and video
- Put 5-8 chosen figures in `images/` (PNG, under ~1 MB each) and reference them in the README.
- For a demo video: record the API page or terminal (30-60 s), upload to YouTube (unlisted) and link it in the README, or add a small GIF to `images/`. Do not commit large videos.
- Model weights: if you want to share them, attach to a **GitHub Release** instead of committing.

## Make it professional
README with real results and limitations; passing CI badge (`.github/workflows/ci.yml` runs the tests); a `v1.0.0` release;
pinned repo on your profile; issues/TODO list of future work (external validation, patient-level split, DICOM support).
