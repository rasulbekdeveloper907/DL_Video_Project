import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, MEAN, STD, TEST_DIR, TRAIN_DIR, VAL_DIR


CLASSES = ["sitting", "standing"]


def get_train_frame_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_eval_frame_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class VideoSequenceDataset(Dataset):
    """
    One sample = one sequence folder
    Shape: [T, C, H, W]
    """

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        
        self.class_names = CLASSES
        self.class_to_index = {c: i for i, c in enumerate(self.class_names)}

        # load dataset
        for class_name in self.class_names:
            class_path = self.root_dir / class_name

            if not class_path.exists():
                print(f"Warning: Missing folder {class_path}")
                continue

            for sequence_dir in sorted(class_path.iterdir()):
                if sequence_dir.is_dir():
                    self.samples.append(
                        (sequence_dir, self.class_to_index[class_name])
                    )

        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence_dir, label = self.samples[index]
        frame_paths = sorted(sequence_dir.glob("*.jpg"))

        frames = []

        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")

            if self.transform:
                frame_tensor = self.transform(image)
            else:
                frame_tensor = transforms.ToTensor()(image)

            frames.append(frame_tensor)

        
        if len(frames) == 0:
            raise ValueError(f"No frames found in {sequence_dir}")

        sequence_tensor = torch.stack(frames, dim=0)

        return sequence_tensor, label


def create_dataloaders(batch_size: int = 2):
    train_dataset = VideoSequenceDataset(TRAIN_DIR, transform=get_train_frame_transform())
    val_dataset = VideoSequenceDataset(VAL_DIR, transform=get_eval_frame_transform())
    test_dataset = VideoSequenceDataset(TEST_DIR, transform=get_eval_frame_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    
    sample_sequences, sample_labels = next(iter(train_loader))

    print("\nDATASET INFO")
    print(f"Sample sequence shape: {sample_sequences.shape}")
    print(f"Sample label shape: {sample_labels.shape}")
    print("Class names:", train_dataset.class_names)
    print("One sample shape [T, C, H, W] =", sample_sequences[0].shape)
    print("One batch shape [B, T, C, H, W] =", sample_sequences.shape)

    return train_loader, val_loader, test_loader, train_dataset.class_names