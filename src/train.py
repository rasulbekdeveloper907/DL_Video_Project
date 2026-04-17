import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from model import SimpleVideoCNN, print_model_summary
from utils import MODELS_DIR, PLOTS_DIR, get_device, plot_training_history, save_json, set_seed


def run_one_epoch(model, dataloader, criterion, optimizer, device, training: bool):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train a simple video classifier.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate.")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print("Using device:", device)

    train_loader, val_loader, _, class_names = create_dataloaders(batch_size=args.batch_size)

    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)
    print_model_summary(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    model_save_path = MODELS_DIR / "best_video_model.pth"

    for epoch in range(args.epochs):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}")

    save_json(history, PLOTS_DIR / "training_history.json")
    save_json(class_names, MODELS_DIR / "class_names.json")
    plot_training_history(history, PLOTS_DIR / "training_curves.png")

    print("\nTraining complete.")
    print("Best validation accuracy:", round(best_val_acc, 4))
    print("Saved model to:", model_save_path)
    print("Saved class names to:", MODELS_DIR / "class_names.json")
    print("Saved plots to:", PLOTS_DIR / "training_curves.png")


if __name__ == "__main__":
    main()
