import torch
import torch.nn as nn
from metrics import print_classification_report

from config import *
from dataset import test_loader
from model import DeepFER
from engine import evaluate_model
from utils import load_model
from metrics import (
    plot_confusion_matrix,
    print_classification_report
)

# ------------------------
# Device
# ------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

# ------------------------
# Model
# ------------------------

model = DeepFER().to(device)

model = load_model(
    model,
    MODEL_PATH,
    device
)

# ------------------------
# Loss Function
# ------------------------

criterion = nn.CrossEntropyLoss()

# ------------------------
# Evaluate
# ------------------------

(
    test_loss,
    test_accuracy,
    predictions,
    labels
) = evaluate_model(
    model,
    test_loader,
    criterion,
    device
)

print("\nTest Results")
print("-" * 30)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.2f}%")

# ------------------------
# Confusion Matrix
# ------------------------

class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

plot_confusion_matrix(
    labels,
    predictions,
    class_names
)

print_classification_report(
    labels,
    predictions,
    class_names
)