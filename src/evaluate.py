import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from dataset import create_dataloaders
from model import SimpleVideoCNN
from utils import (
    MODELS_DIR,
    PLOTS_DIR,
    get_device,
    load_json,
    plot_confusion_matrix
)


def main():
    device = get_device()
    print("Using device:", device)

   
    _, _, test_loader, _ = create_dataloaders(batch_size=2)

    if len(test_loader.dataset) == 0:
        raise ValueError(" Test dataset is empty! Check prepare_sequences.py")

    
    class_names_path = MODELS_DIR / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(f" Missing class names: {class_names_path}")

    class_names = load_json(class_names_path)
    print("Classes:", class_names)

    
    model_path = MODELS_DIR / "best_video_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f" Model not found: {model_path}")

    model = SimpleVideoCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(preds.cpu().numpy().tolist())

    
    test_loss = total_loss / total_samples if total_samples > 0 else 0
    test_accuracy = accuracy_score(all_labels, all_predictions)

    print("\n TEST RESULTS")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")

    
    confusion_matrix_path = PLOTS_DIR / "confusion_matrix.png"

    plot_confusion_matrix(
        all_labels,
        all_predictions,
        class_names,
        confusion_matrix_path
    )

    print(" Saved confusion matrix to:", confusion_matrix_path)


if __name__ == "__main__":
    main()