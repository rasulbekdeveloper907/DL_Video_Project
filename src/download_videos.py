import cv2
import numpy as np
from pathlib import Path

from utils import CLASS_NAMES, RAW_VIDEOS_DIR, ensure_directories, save_json, set_seed





def create_synthetic_motion_video(
    save_path: Path,
    motion_type: str,
    frame_size=(128, 128),
    num_frames=24,
    fps=8
):
    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

    background = (245, 245, 245)
    color = (60, 80, 200)

    base_x = width // 2
    base_y = height // 2

    for i in range(num_frames):
        frame = np.full((height, width, 3), background, dtype=np.uint8)

       
        if motion_type == "sitting":
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-1, 2)

        
        else:
            dx = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)

        x = base_x + dx
        y = base_y + dy

        
        cv2.circle(frame, (x, y - 18), 10, color, -1)
        cv2.line(frame, (x, y - 8), (x, y + 20), color, 3)
        cv2.line(frame, (x, y), (x - 10, y + 12), color, 3)
        cv2.line(frame, (x, y), (x + 10, y + 12), color, 3)
        cv2.line(frame, (x, y + 20), (x - 8, y + 34), color, 3)
        cv2.line(frame, (x, y + 20), (x + 8, y + 34), color, 3)

        cv2.putText(
            frame,
            motion_type,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()


def create_dataset(videos_per_class: int = 6):
    metadata = {
        "source": "synthetic_only",
        "classes": CLASS_NAMES,
        "videos": []
    }

    for class_name in CLASS_NAMES:
        class_dir = RAW_VIDEOS_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(videos_per_class):
            save_path = class_dir / f"{class_name}_{i:02d}.mp4"

            create_synthetic_motion_video(
                save_path=save_path,
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

    print("Creating synthetic dataset (sitting vs standing)...")

    metadata = create_dataset(videos_per_class=8)

    save_json(metadata, RAW_VIDEOS_DIR / "dataset_info.json")

    print("\n DONE!")
    print(" Dataset saved to:", RAW_VIDEOS_DIR)
    print(" Metadata:", RAW_VIDEOS_DIR / "dataset_info.json")


if __name__ == "__main__":
    main()