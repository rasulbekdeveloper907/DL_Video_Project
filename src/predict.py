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
    STD,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RAW_VIDEOS_DIR,
    SEQUENCE_LENGTH,
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

    if total_frames <= 0:
        raise ValueError("Video has no frames")

    indices = set(sample_frame_indices(total_frames, SEQUENCE_LENGTH))
    transform = build_frame_transform()

    frames = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            frames.append(transform(pil))

        frame_index += 1

    capture.release()


    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1].clone())

    frames = frames[:SEQUENCE_LENGTH]

    return torch.stack(frames, dim=0)



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

    label = f"{predicted_class} ({confidence:.2f})"

    cv2.putText(
        frame,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    save_path = PREDICTIONS_DIR / f"prediction_{video_path.stem}.jpg"
    cv2.imwrite(str(save_path), frame)

    print("Saved prediction preview:", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None)
    args = parser.parse_args()

    video_path = Path(args.video_path) if args.video_path else get_default_video()

    print("\n Using video:", video_path)
    print("Info:", get_video_info(video_path))

    device = get_device()


    class_names = load_json(MODELS_DIR / "class_names.json")
    print(" Classes:", class_names)

  
    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)

    model.load_state_dict(
        torch.load(MODELS_DIR / "best_video_model.pth", map_location=device)
    )

    model.eval()


    sequence = load_video_as_sequence(video_path)
    sequence = sequence.unsqueeze(0).to(device)  

    with torch.no_grad():
        outputs = model(sequence)
        probs = torch.softmax(outputs, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class = class_names[pred_idx.item()]
    confidence_value = confidence.item()

  
    print("\nPREDICTION RESULT")
    print("Class:", predicted_class)
    print(f"Confidence: {confidence_value:.4f}")

    save_prediction_preview(video_path, predicted_class, confidence_value)


if __name__ == "__main__":
    main()