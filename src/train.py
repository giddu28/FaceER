import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from dataset import train_loader, val_loader
from model import DeepFER
from engine import train_one_epoch, evaluate_model
from utils import save_model, set_seed
from plots import plot_training_history

# -----------------------------------
# Set Random Seed
# -----------------------------------

set_seed(42)

# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

# -----------------------------------
# Model
# -----------------------------------

model = DeepFER().to(device)

# -----------------------------------
# Loss Function
# -----------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------------
# Variables
# -----------------------------------

best_accuracy = 0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

# -----------------------------------
# Training Loop
# -----------------------------------

for epoch in range(NUM_EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # -------------------------
    # Validation
    # -------------------------

    val_loss, val_accuracy, _, _ = evaluate_model(
        model,
        val_loader,
        criterion,
        device
    )

    # -------------------------
    # Store History
    # -------------------------

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    # -------------------------
    # Print Results
    # -------------------------

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

    print(f"Train Loss         : {train_loss:.4f}")
    print(f"Train Accuracy     : {train_accuracy:.2f}%")

    print(f"Validation Loss    : {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # -------------------------
    # Save Best Model
    # -------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        save_model(
            model,
            MODEL_PATH
        )

print("\nTraining Completed!")

# -----------------------------------
# Plot Training Curves
# -----------------------------------

plot_training_history(history)