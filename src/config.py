"""Central configuration. Every other module imports its defaults from here."""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "chest_xray"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODEL_PATH = MODELS_DIR / "medical_ai_model.keras"

# ---------------------------------------------------------------- data
# Folder names double as labels. Alphabetical order => NORMAL=0, PNEUMONIA=1.
# The model's single sigmoid output is therefore P(PNEUMONIA).
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
POSITIVE_CLASS = "PNEUMONIA"
IMG_SIZE = (224, 224)  # MobileNetV2's native size
BATCH_SIZE = 32
VAL_SPLIT = 0.15  # carved out of train/ (the official val/ folder has only 16 images)
SEED = 42

# ---------------------------------------------------------------- model
DEFAULT_ARCH = "mobilenetv2"  # or "cnn"
DECISION_THRESHOLD = 0.5
FINE_TUNE_LAYERS = 30  # how many of MobileNetV2's last layers to unfreeze in phase 2

# ---------------------------------------------------------------- api
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DISCLAIMER = (
    "Educational demo only. Not a medical device. Not validated for clinical use. "
    "Never use it to diagnose, treat or make decisions about real patients."
)
