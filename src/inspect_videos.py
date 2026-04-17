from utils import RAW_VIDEOS_DIR, get_video_info, save_json


def main():
    video_paths = sorted(RAW_VIDEOS_DIR.glob("*/*.*"))

    if not video_paths:
        raise FileNotFoundError(
            f"No videos found in {RAW_VIDEOS_DIR}. Run download_videos.py first."
        )

    inspection_results = []

    print("\n📊 VIDEO INSPECTION RESULTS\n")

    for video_path in video_paths:

        try:
            info = get_video_info(video_path)

            inspection_results.append(info)

            print(
                f"🎥 {info['video_name']} | "
                f"class={video_path.parent.name} | "
                f"frames={info['total_frames']} | "
                f"fps={info['fps']:.2f} | "
                f"size={info['width']}x{info['height']}"
            )

        except Exception as e:
            print(f"⚠️ Failed to read {video_path.name}: {e}")

    # 💾 save results
    output_path = RAW_VIDEOS_DIR / "video_inspection.json"
    save_json(inspection_results, output_path)

    print("\n📄 Saved inspection file to:", output_path)


if __name__ == "__main__":
    main()