import random
import shutil
from pathlib import Path

from utils import (
    CLASS_NAMES,
    EXTRACTED_FRAMES_DIR,
    SEQUENCE_LENGTH,
    SEQUENCES_DIR,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
    count_items_per_class,
    reset_directory,
    sample_frame_indices,
    save_json,
    set_seed,
)


def safe_copytree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_sequence_frames(frame_paths, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(frame_paths):
        shutil.copy2(fp, dest_dir / f"frame_{i:02d}.jpg")


def split_list(items, train_ratio=0.7, val_ratio=0.15):
    if len(items) < 3:
        return items, [], []

    random.shuffle(items)

    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)

    train = items[:train_end]
    val = items[train_end:val_end]
    test = items[val_end:]

    if len(val) == 0 and len(train) > 1:
        val.append(train.pop())

    if len(test) == 0 and len(val) > 1:
        test.append(val.pop())

    return train, val, test


def main():
    set_seed(42)

    if not EXTRACTED_FRAMES_DIR.exists():
        raise FileNotFoundError("Run extract_frames.py first")

    for d in [SEQUENCES_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
        reset_directory(d)

    summary = {}

    for class_name in CLASS_NAMES:

        video_dirs = list((EXTRACTED_FRAMES_DIR / class_name).glob("*"))
        random.shuffle(video_dirs)

        sequences = []

        for vdir in video_dirs:
            frames = sorted(vdir.glob("*.jpg"))

            # ✔ skip too short videos safely
            if len(frames) < SEQUENCE_LENGTH:
                continue

            indices = sample_frame_indices(len(frames), SEQUENCE_LENGTH)
            selected = [frames[i] for i in indices]

            seq_dir = SEQUENCES_DIR / class_name / vdir.name
            copy_sequence_frames(selected, seq_dir)

            sequences.append(seq_dir)

        train, val, test = split_list(sequences)

        for s in train:
            safe_copytree(s, TRAIN_DIR / class_name / s.name)

        for s in val:
            safe_copytree(s, VAL_DIR / class_name / s.name)

        for s in test:
            safe_copytree(s, TEST_DIR / class_name / s.name)

        summary[class_name] = {
            "total": len(sequences),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

    save_json(summary, SEQUENCES_DIR / "sequence_summary.json")

    print("\n✅ DONE")
    print("Train:", count_items_per_class(TRAIN_DIR))
    print("Val:", count_items_per_class(VAL_DIR))
    print("Test:", count_items_per_class(TEST_DIR))


if __name__ == "__main__":
    main()