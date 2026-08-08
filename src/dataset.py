from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from config import *

# --------------------------------
# Image Transformations
# --------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# --------------------------------
# Full Training Dataset
# --------------------------------

full_train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)

# --------------------------------
# Train / Validation Split
# --------------------------------

train_size = int(TRAIN_SPLIT * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size]
)

# --------------------------------
# Test Dataset
# --------------------------------

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)

# --------------------------------
# Data Loaders
# --------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)