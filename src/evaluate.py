import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from dataset import create_dataloaders
from model import SimpleVideoCNN
from utils import MODELS_DIR, PLOTS_DIR, get_device, load_json, plot_confusion_matrix


def main():
    device = get_device()
    print("Using device:", device)

    _, _, test_loader, _ = create_dataloaders(batch_size=2)

    class_names = load_json(MODELS_DIR / "class_names.json")
    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_video_model.pth", map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_labels = []
    all_predictions = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    test_loss = running_loss / total
    test_accuracy = accuracy_score(all_labels, all_predictions)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    confusion_matrix_path = PLOTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(all_labels, all_predictions, class_names, confusion_matrix_path)
    print("Saved confusion matrix to:", confusion_matrix_path)


if __name__ == "__main__":
    main()
