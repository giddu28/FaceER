import os
import matplotlib.pyplot as plt


def plot_training_history(history):
    """
    Plot and save training history.
    """

    os.makedirs("outputs", exist_ok=True)

    # -----------------------------
    # Loss Curve
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/loss_curve.png", dpi=300)

    plt.close()

    # -----------------------------
    # Accuracy Curve
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.legend()
    plt.grid(True)

    plt.savefig("outputs/accuracy_curve.png", dpi=300)

    plt.close()

    print("✅ Training curves saved in outputs/")