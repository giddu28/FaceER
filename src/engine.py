import torch

from utils import calculate_accuracy

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = calculate_accuracy(correct, total)

    return epoch_loss, epoch_accuracy


def evaluate_model(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy()
            )

            all_labels.extend(labels.cpu().numpy()
            )

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = calculate_accuracy(correct, total)

    return (
    epoch_loss,
    epoch_accuracy,
    all_predictions,
    all_labels
)