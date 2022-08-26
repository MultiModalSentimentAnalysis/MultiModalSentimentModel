import os, torch
from pathlib import Path

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = DATA_DIR / "saved_features"
TEXTS_DIR = DATA_DIR / "texts"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

FACE_EMBEDDING_SIZE = 1280
TEXT_EMBEDDING_SIZE = 768
POSE_EMBEDDING_SIZE = 34
SCENE_EMBEDDING_SIZE = 2048

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")