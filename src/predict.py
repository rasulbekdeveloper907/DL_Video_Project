import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from model import SimpleVideoCNN
from utils import (
    IMAGE_SIZE,
    MEAN,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RAW_VIDEOS_DIR,
    SEQUENCE_LENGTH,
    STD,
    get_device,
    get_video_info,
    load_json,
    sample_frame_indices,
)


def build_frame_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_video_as_sequence(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(total_frames, SEQUENCE_LENGTH))
    transform = build_frame_transform()

    collected_frames = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index in indices:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            collected_frames.append(transform(pil_image))
        frame_index += 1

    capture.release()

    while len(collected_frames) < SEQUENCE_LENGTH:
        collected_frames.append(collected_frames[-1].clone())

    sequence_tensor = torch.stack(collected_frames[:SEQUENCE_LENGTH], dim=0)
    return sequence_tensor


def get_default_video():
    video_paths = sorted(RAW_VIDEOS_DIR.glob("*/*.*"))
    if not video_paths:
        raise FileNotFoundError(
            f"No videos found in {RAW_VIDEOS_DIR}. Run download_videos.py first."
        )
    return video_paths[0]


def save_prediction_preview(video_path: Path, predicted_class: str, confidence: float):
    capture = cv2.VideoCapture(str(video_path))
    success, frame = capture.read()
    capture.release()
    if not success:
        return

    label_text = f"{predicted_class} ({confidence:.2f})"
    cv2.putText(
        frame,
        label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    save_path = PREDICTIONS_DIR / f"prediction_{video_path.stem}.jpg"
    cv2.imwrite(str(save_path), frame)
    print("Saved prediction preview to:", save_path)


def main():
    parser = argparse.ArgumentParser(description="Predict one video using the trained model.")
    parser.add_argument("--video_path", type=str, default=None, help="Optional path to a video file.")
    args = parser.parse_args()

    video_path = Path(args.video_path) if args.video_path else get_default_video()
    print("Using video:", video_path)
    print("Video info:", get_video_info(video_path))

    device = get_device()
    class_names = load_json(MODELS_DIR / "class_names.json")

    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_video_model.pth", map_location=device))
    model.eval()

    sequence_tensor = load_video_as_sequence(video_path).unsqueeze(0).to(device)
    # Shape becomes [1, T, C, H, W], for example [1, 8, 3, 128, 128].

    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_index.item()]
    confidence_value = confidence.item()

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence_value:.4f}")
    save_prediction_preview(video_path, predicted_class, confidence_value)


if __name__ == "__main__":
    main()
