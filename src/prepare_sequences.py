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


def copy_sequence_frames(source_frame_paths, destination_dir):
    """Copy selected frames into one sequence folder in the correct time order."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for new_index, frame_path in enumerate(source_frame_paths):
        shutil.copy2(frame_path, destination_dir / f"frame_{new_index:02d}.jpg")


def split_list(items, train_ratio=0.7, val_ratio=0.15):
    total = len(items)
    train_count = max(1, int(total * train_ratio)) if total >= 3 else max(1, total - 2)
    val_count = max(1, int(total * val_ratio)) if total >= 3 else 1

    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)

    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]

    if not test_items and val_items:
        test_items = [val_items.pop()]
    if not val_items and train_items:
        val_items = [train_items.pop()]

    return train_items, val_items, test_items


def main():
    set_seed(42)

    if not EXTRACTED_FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"No extracted frames found in {EXTRACTED_FRAMES_DIR}. Please run extract_frames.py first."
        )

    for folder in [SEQUENCES_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
        reset_directory(folder)

    summary = {}

    for class_name in CLASS_NAMES:
        class_video_dirs = sorted((EXTRACTED_FRAMES_DIR / class_name).glob("*"))
        random.shuffle(class_video_dirs)

        created_sequences = []

        for video_dir in class_video_dirs:
            frame_paths = sorted(video_dir.glob("*.jpg"))
            indices = sample_frame_indices(len(frame_paths), sequence_length=SEQUENCE_LENGTH)
            selected_frames = [frame_paths[index] for index in indices]

            sequence_dir = SEQUENCES_DIR / class_name / video_dir.name
            copy_sequence_frames(selected_frames, sequence_dir)
            created_sequences.append(sequence_dir)

        train_sequences, val_sequences, test_sequences = split_list(created_sequences)

        for sequence_dir in train_sequences:
            shutil.copytree(sequence_dir, TRAIN_DIR / class_name / sequence_dir.name)
        for sequence_dir in val_sequences:
            shutil.copytree(sequence_dir, VAL_DIR / class_name / sequence_dir.name)
        for sequence_dir in test_sequences:
            shutil.copytree(sequence_dir, TEST_DIR / class_name / sequence_dir.name)

        summary[class_name] = {
            "total_sequences": len(created_sequences),
            "train": len(train_sequences),
            "val": len(val_sequences),
            "test": len(test_sequences),
        }

    save_json(summary, SEQUENCES_DIR / "sequence_summary.json")

    print("Sequence preparation is complete.\n")
    print("Train sequence counts:", count_items_per_class(TRAIN_DIR))
    print("Val sequence counts:", count_items_per_class(VAL_DIR))
    print("Test sequence counts:", count_items_per_class(TEST_DIR))


if __name__ == "__main__":
    main()
