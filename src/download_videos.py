import urllib.request
from pathlib import Path

import cv2
import numpy as np

from utils import CLASS_NAMES, RAW_VIDEOS_DIR, ensure_directories, save_json, set_seed


PUBLIC_VIDEO_URLS = {
    "walking": [
        "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi",
    ],
    "running": [
        "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/tracking/faceocc2.webm",
    ],
}


def try_download_file(url: str, save_path: Path):
    """Try to download a file from the internet."""
    urllib.request.urlretrieve(url, save_path)
    return save_path.exists() and save_path.stat().st_size > 0


def create_synthetic_motion_video(save_path: Path, motion_type: str, frame_size=(128, 128), num_frames=24, fps=8):
    """
    Create a tiny synthetic motion video with OpenCV.
    walking: slower horizontal motion
    running: faster horizontal motion
    """
    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    background_color = (245, 245, 245)
    object_color = (30, 90, 220) if motion_type == "walking" else (40, 180, 40)
    speed = 3 if motion_type == "walking" else 7

    for frame_index in range(num_frames):
        frame = np.full((height, width, 3), background_color, dtype=np.uint8)

        x = 10 + frame_index * speed
        x = min(x, width - 30)
        y = height // 2

        cv2.circle(frame, (x, y - 18), 10, object_color, -1)
        cv2.line(frame, (x, y - 8), (x, y + 20), object_color, 3)
        cv2.line(frame, (x, y), (x - 10, y + 12), object_color, 3)
        cv2.line(frame, (x, y), (x + 10, y + 12), object_color, 3)

        if motion_type == "walking":
            cv2.line(frame, (x, y + 20), (x - 8, y + 34 + (frame_index % 2) * 2), object_color, 3)
            cv2.line(frame, (x, y + 20), (x + 8, y + 34 - (frame_index % 2) * 2), object_color, 3)
        else:
            cv2.line(frame, (x, y + 20), (x - 12, y + 38 + (frame_index % 2) * 5), object_color, 3)
            cv2.line(frame, (x, y + 20), (x + 12, y + 30 - (frame_index % 2) * 5), object_color, 3)

        cv2.putText(
            frame,
            motion_type,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()


def create_fallback_dataset(videos_per_class: int = 6):
    """Create a tiny synthetic dataset when public downloads are unavailable."""
    metadata = {"source": "synthetic_fallback", "classes": CLASS_NAMES, "videos": []}

    for class_name in CLASS_NAMES:
        class_dir = RAW_VIDEOS_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for video_index in range(videos_per_class):
            save_path = class_dir / f"{class_name}_{video_index:02d}.mp4"
            create_synthetic_motion_video(save_path, motion_type=class_name)
            metadata["videos"].append({"class_name": class_name, "path": str(save_path)})

    return metadata


def main():
    set_seed(42)
    ensure_directories()

    metadata = {"source": "public_download", "classes": CLASS_NAMES, "videos": []}
    downloaded_any = False

    for class_name in CLASS_NAMES:
        class_dir = RAW_VIDEOS_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        urls = PUBLIC_VIDEO_URLS.get(class_name, [])
        for index, url in enumerate(urls):
            suffix = Path(url).suffix or ".mp4"
            save_path = class_dir / f"{class_name}_{index:02d}{suffix}"
            try:
                print(f"Trying to download {url}")
                if try_download_file(url, save_path):
                    downloaded_any = True
                    metadata["videos"].append({"class_name": class_name, "path": str(save_path), "url": url})
            except Exception as error:
                print(f"Download failed for {url}: {error}")

    if not downloaded_any:
        print("Public download failed. Creating a synthetic fallback video dataset...")
        metadata = create_fallback_dataset()

    save_json(metadata, RAW_VIDEOS_DIR / "dataset_info.json")
    print("\nSaved videos to:", RAW_VIDEOS_DIR)
    print("Saved dataset info to:", RAW_VIDEOS_DIR / "dataset_info.json")


if __name__ == "__main__":
    main()
