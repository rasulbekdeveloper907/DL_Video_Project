import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from model import SimpleVideoCNN, print_model_summary
from utils import (
    MODELS_DIR,
    PLOTS_DIR,
    get_device,
    plot_training_history,
    save_json,
    set_seed,
)


# 🧠 one epoch runner
def run_one_epoch(model, dataloader, criterion, optimizer, device, training: bool):

    model.train() if training else model.eval()

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

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()

    # 🔒 reproducibility
    set_seed(42)

    device = get_device()
    print("Using device:", device)

    # 📦 dataset
    train_loader, val_loader, _, class_names = create_dataloaders(
        batch_size=args.batch_size
    )

    print("\n📦 Classes:", class_names)

    # 🧠 model
    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)
    print_model_summary(model)

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer (slightly improved)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 📊 tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "best_video_model.pth"

    # 🚀 training loop
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
            f"\n📊 Epoch [{epoch+1}/{args.epochs}]"
            f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            f"\nVal Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}\n"
        )

        # 💾 save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved: {best_model_path}")

    # 💾 save history + class names
    save_json(history, PLOTS_DIR / "training_history.json")
    save_json(class_names, MODELS_DIR / "class_names.json")

    # 📈 plot curves
    plot_training_history(history, PLOTS_DIR / "training_curves.png")

    print("\n✅ Training complete")
    print("🏆 Best Val Accuracy:", round(best_val_acc, 4))
    print("📁 Model saved:", best_model_path)


if __name__ == "__main__":
    main()