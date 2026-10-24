"""Repository explicit setup script.

Run this script once after cloning to setup the local environment and
download the required MediaPipe model files.
"""

import hashlib
import os
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
MODELS_DIR = ROOT_DIR / "src" / "models"
STAMP_FILE = VENV_DIR / ".requirements.sha256"


def _python_bin() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _requirements_hash() -> str:
    content = REQUIREMENTS_FILE.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _deps_are_current() -> bool:
    if not STAMP_FILE.exists() or not REQUIREMENTS_FILE.exists():
        return False
    return STAMP_FILE.read_text(encoding="utf-8").strip() == _requirements_hash()


def setup_venv():
    python_bin = _python_bin()
    if not python_bin.exists():
        print("Setting up local Python environment in .venv/ ...")
        builder = venv.EnvBuilder(with_pip=True, clear=False)
        builder.create(VENV_DIR)

    if not _deps_are_current():
        print("Installing Python dependencies from requirements.txt ...")
        subprocess.check_call([str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
        STAMP_FILE.write_text(_requirements_hash(), encoding="utf-8")


def download_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    }
    
    for filename, url in models.items():
        filepath = MODELS_DIR / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"Found {filename}")

if __name__ == "__main__":
    setup_venv()
    download_models()
    print("\nSetup complete! You can now activate the environment:")
    if os.name == "nt":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")
