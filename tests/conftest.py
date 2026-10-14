import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

from make_synthetic_dataset import build  # noqa: E402

TEST_SIZE = (96, 96)  # small so the tests are fast


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    return build(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="session")
def trained(synthetic_root, tmp_path_factory):
    """Train a tiny model once for all tests that need one."""
    from src.train import train

    work = tmp_path_factory.mktemp("run")
    model_path, outputs = work / "model.keras", work / "outputs"
    train(data_dir=synthetic_root, arch="cnn", epochs=2, fine_tune_epochs=0, batch_size=16,
          img_size=TEST_SIZE, pretrained=False, model_path=model_path, outputs_dir=outputs)
    return {"model_path": model_path, "outputs": outputs, "data": synthetic_root}
