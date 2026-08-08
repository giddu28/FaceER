import random
import numpy as np
import torch


def calculate_accuracy(correct, total):
    return 100 * correct / total


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")


def load_model(model, path, device):
    model.load_state_dict(
        torch.load(path, map_location=device)
    )
    return model


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)