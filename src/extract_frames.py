from pathlib import Path
import cv2

from utils import EXTRACTED_FRAMES_DIR, RAW_VIDEOS_DIR, reset_directory, save_json


def extract_all_frames(video_path: Path, output_dir: Path):

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return 0

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        # 🧼 safe check
        if frame is None:
            frame_index += 1
            continue

        output_path = output_dir / f"frame_{frame_index:04d}.jpg"

        cv2.imwrite(str(output_path), frame)

        frame_index += 1
        saved_count += 1

    capture.release()

    return saved_count


def main():

    video_paths = sorted(RAW_VIDEOS_DIR.glob("*/*.*"))

    if not video_paths:
        raise FileNotFoundError("Run download_videos.py first")

    reset_directory(EXTRACTED_FRAMES_DIR)

    summary = []

    for video_path in video_paths:

        class_name = video_path.parent.name
        video_name = video_path.stem

        output_dir = EXTRACTED_FRAMES_DIR / class_name / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = extract_all_frames(video_path, output_dir)

        print(f"✅ {class_name}/{video_name}: {saved} frames")

        summary.append({
            "class_name": class_name,
            "video_name": video_name,
            "frames_saved": saved,
        })

    save_json(summary, EXTRACTED_FRAMES_DIR / "frame_extraction_summary.json")

    print("\n✅ DONE")
    print("📁 Frames saved to:", EXTRACTED_FRAMES_DIR)


if __name__ == "__main__":
    main()