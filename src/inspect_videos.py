from utils import RAW_VIDEOS_DIR, get_video_info, save_json


def main():
    video_paths = sorted(RAW_VIDEOS_DIR.glob("*/*.*"))
    if not video_paths:
        raise FileNotFoundError(
            f"No videos found in {RAW_VIDEOS_DIR}. Please run download_videos.py first."
        )

    inspection_results = []

    print("Video inspection results:\n")
    for video_path in video_paths:
        info = get_video_info(video_path)
        inspection_results.append(info)
        print(
            f"{info['video_name']} | "
            f"frames={info['total_frames']} | fps={info['fps']:.2f} | "
            f"width={info['width']} | height={info['height']}"
        )

    save_json(inspection_results, RAW_VIDEOS_DIR / "video_inspection.json")
    print("\nSaved inspection file to:", RAW_VIDEOS_DIR / "video_inspection.json")


if __name__ == "__main__":
    main()
