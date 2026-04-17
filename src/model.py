import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """A tiny CNN that extracts features from one frame."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input frame shape: [batch_size, 3, 128, 128]
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Shape: [batch_size, 16, 64, 64]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Shape: [batch_size, 32, 32, 32]
            nn.AdaptiveAvgPool2d((1, 1)),
            # Shape: [batch_size, 32, 1, 1]
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        # Shape becomes [batch_size, 32]
        return x


class SimpleVideoCNN(nn.Module):
    """
    Process each frame with the same CNN, then average features across time.
    Input shape: [B, T, C, H, W]
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.frame_encoder = FrameEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape

        x = x.view(batch_size * time_steps, channels, height, width)
        frame_features = self.frame_encoder(x)
        frame_features = frame_features.view(batch_size, time_steps, -1)
        # Shape is now [B, T, feature_dim].

        video_features = frame_features.mean(dim=1)
        # Temporal average pooling makes shape [B, feature_dim].

        logits = self.classifier(video_features)
        return logits


def print_model_summary(model: nn.Module):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print("\nModel structure:")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
