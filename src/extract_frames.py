from pathlib import Path

import cv2

from utils import EXTRACTED_FRAMES_DIR, RAW_VIDEOS_DIR, reset_directory, save_json


def extract_all_frames(video_path: Path, output_dir: Path):
    """Read every frame from one video and save it as an image."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        output_path = output_dir / f"frame_{frame_index:04d}.jpg"
        cv2.imwrite(str(output_path), frame)
        frame_index += 1
        saved_count += 1

    capture.release()
    return saved_count


def main():
    video_paths = sorted(RAW_VIDEOS_DIR.glob("*/*.*"))
    if not video_paths:
        raise FileNotFoundError(
            f"No videos found in {RAW_VIDEOS_DIR}. Please run download_videos.py first."
        )

    reset_directory(EXTRACTED_FRAMES_DIR)
    summary = []

    for video_path in video_paths:
        class_name = video_path.parent.name
        video_name = video_path.stem
        output_dir = EXTRACTED_FRAMES_DIR / class_name / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_count = extract_all_frames(video_path, output_dir)
        summary.append(
            {
                "class_name": class_name,
                "video_name": video_name,
                "frames_saved": saved_count,
            }
        )
        print(f"Extracted {saved_count} frames from {video_path.name}")

    save_json(summary, EXTRACTED_FRAMES_DIR / "frame_extraction_summary.json")
    print("\nSaved extraction summary to:", EXTRACTED_FRAMES_DIR / "frame_extraction_summary.json")


if __name__ == "__main__":
    main()
