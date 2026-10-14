#!/usr/bin/env bash
# Download the Kaggle chest X-ray dataset with the Kaggle CLI (needs ~/.kaggle/kaggle.json).
# Manual alternative: download the zip in your browser and unzip it into ./data/
set -euo pipefail
pip install --quiet kaggle
mkdir -p data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
rm -rf data/__MACOSX
python main.py check-data
