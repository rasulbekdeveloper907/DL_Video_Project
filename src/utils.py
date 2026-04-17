import json
import random
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay



PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
EXTRACTED_FRAMES_DIR = DATA_DIR / "extracted_frames"
SEQUENCES_DIR = DATA_DIR / "sequences"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"



CLASS_NAMES = ["sitting", "standing"]

IMAGE_SIZE = 128
SEQUENCE_LENGTH = 8

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)



def ensure_directories():
    for folder in [
        RAW_VIDEOS_DIR,
        EXTRACTED_FRAMES_DIR,
        SEQUENCES_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        PREDICTIONS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)



def reset_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cpu")



def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    info = {
        "video_name": video_path.name,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    cap.release()
    return info



def sample_frame_indices(total_frames: int, sequence_length: int):
    if total_frames <= 0:
        return [0] * sequence_length

    if total_frames < sequence_length:
        indices = list(range(total_frames))
        while len(indices) < sequence_length:
            indices.append(total_frames - 1)
        return indices

    return np.linspace(0, total_frames - 1, sequence_length, dtype=int).tolist()



def count_items_per_class(root_dir: Path):
    counts = {}

    if not root_dir.exists():
        return counts

    for class_dir in sorted(root_dir.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = len(list(class_dir.iterdir()))

    return counts



def plot_training_history(history: dict, save_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()



def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    plt.figure(figsize=(6, 6))

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
    )

    display.ax_.set_title("Confusion Matrix")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()