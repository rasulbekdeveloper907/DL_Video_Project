import urllib.request
from pathlib import Path

import cv2
import numpy as np

from utils import CLASS_NAMES, RAW_VIDEOS_DIR, ensure_directories, save_json, set_seed


# ❗ For testing only (internet fallback)
PUBLIC_VIDEO_URLS = {
    "sitting": [
        "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi",
    ],
    "standing": [
        "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/tracking/faceocc2.webm",
    ],
}


def try_download_file(url: str, save_path: Path):
    """Try to download a file from the internet."""
    urllib.request.urlretrieve(url, save_path)
    return save_path.exists() and save_path.stat().st_size > 0


# 🧠 SYNTHETIC VIDEO GENERATOR (MAIN PART)
def create_synthetic_motion_video(
    save_path: Path,
    motion_type: str,
    frame_size=(128, 128),
    num_frames=24,
    fps=8
):
    """
    sitting: almost no movement
    standing: completely static
    """

    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    background_color = (245, 245, 245)

    # 🎯 colors
    object_color = (80, 80, 200)

    # 🪑 sitting = tiny jitter
    if motion_type == "sitting":
        jitter = 1
    else:
        jitter = 0  # standing

    base_x = width // 2
    base_y = height // 2

    for frame_index in range(num_frames):
        frame = np.full((height, width, 3), background_color, dtype=np.uint8)

        # 📍 motion logic
        x = base_x + np.random.randint(-jitter, jitter + 1)
        y = base_y + np.random.randint(-jitter, jitter + 1)

        # 👤 simple stick figure
        cv2.circle(frame, (x, y - 18), 10, object_color, -1)
        cv2.line(frame, (x, y - 8), (x, y + 20), object_color, 3)
        cv2.line(frame, (x, y), (x - 10, y + 12), object_color, 3)
        cv2.line(frame, (x, y), (x + 10, y + 12), object_color, 3)
        cv2.line(frame, (x, y + 20), (x - 8, y + 34), object_color, 3)
        cv2.line(frame, (x, y + 20), (x + 8, y + 34), object_color, 3)

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
    """Create synthetic dataset for sitting / standing"""

    metadata = {
        "source": "synthetic_fallback",
        "classes": CLASS_NAMES,
        "videos": []
    }

    for class_name in CLASS_NAMES:
        class_dir = RAW_VIDEOS_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for video_index in range(videos_per_class):
            save_path = class_dir / f"{class_name}_{video_index:02d}.mp4"

            create_synthetic_motion_video(
                save_path,
                motion_type=class_name
            )

            metadata["videos"].append({
                "class_name": class_name,
                "path": str(save_path)
            })

    return metadata


def main():
    set_seed(42)
    ensure_directories()

    metadata = {
        "source": "public_download",
        "classes": CLASS_NAMES,
        "videos": []
    }

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
                    metadata["videos"].append({
                        "class_name": class_name,
                        "path": str(save_path),
                        "url": url
                    })

            except Exception as error:
                print(f"Download failed for {url}: {error}")

    # 🔁 fallback if no internet
    if not downloaded_any:
        print("No public videos found → creating synthetic dataset...")
        metadata = create_fallback_dataset()

    save_json(metadata, RAW_VIDEOS_DIR / "dataset_info.json")

    print("\n✅ Saved videos to:", RAW_VIDEOS_DIR)
    print("📄 Dataset info:", RAW_VIDEOS_DIR / "dataset_info.json")


if __name__ == "__main__":
    main()