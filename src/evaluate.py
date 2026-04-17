import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from dataset import create_dataloaders
from model import SimpleVideoCNN
from utils import MODELS_DIR, PLOTS_DIR, get_device, load_json, plot_confusion_matrix


def main():
    device = get_device()
    print("Using device:", device)

    # 📦 dataset
    _, _, test_loader, _ = create_dataloaders(batch_size=2)

    # 📂 load class names
    class_names = load_json(MODELS_DIR / "class_names.json")

    print("Classes:", class_names)

    # 🧠 model
    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)

    model_path = MODELS_DIR / "best_video_model.pth"

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for sequences, labels in test_loader:

            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())

    # 📊 metrics
    test_loss = total_loss / total_samples
    test_accuracy = accuracy_score(all_labels, all_predictions)

    print("\n📊 TEST RESULTS")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")

    # 📉 confusion matrix
    confusion_matrix_path = PLOTS_DIR / "confusion_matrix.png"

    plot_confusion_matrix(
        all_labels,
        all_predictions,
        class_names,
        confusion_matrix_path
    )

    print("Saved confusion matrix to:", confusion_matrix_path)


if __name__ == "__main__":
    main()