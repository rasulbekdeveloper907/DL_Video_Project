import random
import shutil

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


# 📦 copy frames into sequence folder
def copy_sequence_frames(source_frame_paths, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)

    for i, frame_path in enumerate(source_frame_paths):
        shutil.copy2(frame_path, destination_dir / f"frame_{i:02d}.jpg")


# ✂️ dataset split
def split_list(items, train_ratio=0.7, val_ratio=0.15):
    total = len(items)

    if total < 3:
        return items, [], []

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    # safety fix
    train_count = max(1, train_count)
    val_count = max(1, val_count)

    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)

    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]

    # ensure no empty split
    if len(test_items) == 0 and len(val_items) > 0:
        test_items = [val_items.pop()]

    if len(val_items) == 0 and len(train_items) > 1:
        val_items = [train_items.pop()]

    return train_items, val_items, test_items


def main():
    set_seed(42)

    # 📁 check extracted frames
    if not EXTRACTED_FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"No extracted frames found in {EXTRACTED_FRAMES_DIR}. "
            "Run extract_frames.py first."
        )

    # 🧹 reset output dirs
    for folder in [SEQUENCES_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
        reset_directory(folder)

    summary = {}

    # 🔁 loop over classes
    for class_name in CLASS_NAMES:

        class_frame_dirs = sorted((EXTRACTED_FRAMES_DIR / class_name).glob("*"))
        random.shuffle(class_frame_dirs)

        created_sequences = []

        for video_dir in class_frame_dirs:

            frame_paths = sorted(video_dir.glob("*.jpg"))

            if len(frame_paths) < SEQUENCE_LENGTH:
                continue  # skip too short videos

            indices = sample_frame_indices(len(frame_paths), SEQUENCE_LENGTH)
            selected_frames = [frame_paths[i] for i in indices]

            sequence_dir = SEQUENCES_DIR / class_name / video_dir.name

            copy_sequence_frames(selected_frames, sequence_dir)

            created_sequences.append(sequence_dir)

        # ✂️ split dataset
        train_sequences, val_sequences, test_sequences = split_list(created_sequences)

        # 📦 copy into train/val/test
        for seq in train_sequences:
            shutil.copytree(seq, TRAIN_DIR / class_name / seq.name)

        for seq in val_sequences:
            shutil.copytree(seq, VAL_DIR / class_name / seq.name)

        for seq in test_sequences:
            shutil.copytree(seq, TEST_DIR / class_name / seq.name)

        summary[class_name] = {
            "total_sequences": len(created_sequences),
            "train": len(train_sequences),
            "val": len(val_sequences),
            "test": len(test_sequences),
        }

    # 💾 save summary
    save_json(summary, SEQUENCES_DIR / "sequence_summary.json")

    print("\n✅ Sequence preparation complete!\n")

    print("📊 Train:", count_items_per_class(TRAIN_DIR))
    print("📊 Val:", count_items_per_class(VAL_DIR))
    print("📊 Test:", count_items_per_class(TEST_DIR))