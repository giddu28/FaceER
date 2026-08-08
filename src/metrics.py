import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names
):
    """
    Plot and save confusion matrix.
    """

    os.makedirs("outputs", exist_ok=True)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    plt.figure(figsize=(8, 8))

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix")

    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("✅ Confusion matrix saved.")
    
def print_classification_report(
    y_true,
    y_pred,
    class_names
):
    """
    Print and save the classification report.
    """

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print("\nClassification Report")
    print("-" * 60)
    print(report)

    with open("outputs/classification_report.txt", "w") as file:
        file.write(report)

    print("✅ Classification report saved.")  