import torch
import torch.nn as nn


# 🧠 Frame feature extractor (CNN)
class FrameEncoder(nn.Module):
    """Extract spatial features from a single frame."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [B, 3, 128, 128]
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 16, 64, 64]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 32, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # [B, 64]
        return x


# 🎥 Video classifier
class SimpleVideoCNN(nn.Module):
    """
    Input: [B, T, C, H, W]
    Output: logits for 2 classes (sitting / standing)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.frame_encoder = FrameEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),   # 🔥 reduce overfitting

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # reshape: treat each frame independently
        x = x.view(B * T, C, H, W)

        frame_features = self.frame_encoder(x)  # [B*T, 64]

        frame_features = frame_features.view(B, T, -1)

        # temporal pooling (average over time)
        video_features = frame_features.mean(dim=1)  # [B, 64]

        logits = self.classifier(video_features)

        return logits


# 📊 model debugging helper
def print_model_summary(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n🧠 Model Summary")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")