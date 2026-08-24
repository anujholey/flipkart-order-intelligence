
import os
import json
import copy
import random
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"

PART2_DIR = os.path.join(
    PROJECT_DIR,
    "part2_image_classifier"
)

DATA_DIR = os.path.join(
    PART2_DIR,
    "data"
)

REPORT_DIR = os.path.join(
    PART2_DIR,
    "reports"
)

CACHE_DIR = os.path.join(
    PART2_DIR,
    "feature_cache"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

SAMPLE_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "sample_images"
)

for directory in [
    DATA_DIR,
    REPORT_DIR,
    CACHE_DIR,
    MODEL_DIR,
    SAMPLE_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE")
print("=" * 70)

print("Using:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. FASHION-MNIST CLASSES
# ============================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

# ResNet-18 expects ImageNet-style RGB input.
# Fashion-MNIST is grayscale 28x28, so:
# 1. replicate grayscale channel to 3 channels
# 2. resize to 224x224
# 3. normalize using ImageNet mean/std

imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]


transform = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    )
])


# ============================================================
# 6. DOWNLOAD FASHION-MNIST
# ============================================================

print("\n" + "=" * 70)
print("LOADING FASHION-MNIST")
print("=" * 70)

full_train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)


print(
    "Original training split:",
    len(full_train_dataset)
)

print(
    "Test split:",
    len(test_dataset)
)


# ============================================================
# 7. STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

targets = np.array(
    full_train_dataset.targets
)

all_indices = np.arange(
    len(full_train_dataset)
)

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=5000,
    stratify=targets,
    random_state=SEED
)


train_dataset = Subset(
    full_train_dataset,
    train_indices
)

val_dataset = Subset(
    full_train_dataset,
    val_indices
)


print("\nFinal split sizes:")

print(
    "Training:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Test:",
    len(test_dataset)
)


# ============================================================
# 8. DATA LOADERS
# ============================================================

BATCH_SIZE = 256

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 9. LOAD PRETRAINED RESNET-18
# ============================================================

print("\n" + "=" * 70)
print("LOADING PRETRAINED RESNET-18")
print("=" * 70)

weights = ResNet18_Weights.DEFAULT

backbone_model = models.resnet18(
    weights=weights
)


# Remove original ImageNet classifier.
# The output after avgpool is a 512-dimensional feature vector.

feature_extractor = nn.Sequential(
    *list(backbone_model.children())[:-1]
)

feature_extractor = feature_extractor.to(
    device
)

feature_extractor.eval()


for parameter in feature_extractor.parameters():
    parameter.requires_grad = False


print(
    "Frozen backbone loaded successfully."
)


# ============================================================
# 10. FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(
    loader,
    name
):

    features_list = []
    labels_list = []

    total_batches = len(loader)

    print(
        f"\nExtracting {name} features..."
    )

    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True
            )

            feature_output = feature_extractor(
                images
            )

            feature_output = torch.flatten(
                feature_output,
                1
            )

            features_list.append(
                feature_output.cpu()
            )

            labels_list.append(
                labels.cpu()
            )

            if (
                batch_index + 1
            ) % 25 == 0:

                print(
                    f"{batch_index + 1}"
                    f"/{total_batches} batches"
                )

    features_tensor = torch.cat(
        features_list
    )

    labels_tensor = torch.cat(
        labels_list
    )

    print(
        name,
        "feature shape:",
        features_tensor.shape
    )

    return (
        features_tensor,
        labels_tensor
    )


# ============================================================
# 11. CACHE FEATURES
# ============================================================

train_cache = os.path.join(
    CACHE_DIR,
    "train_features.pt"
)

val_cache = os.path.join(
    CACHE_DIR,
    "val_features.pt"
)

test_cache = os.path.join(
    CACHE_DIR,
    "test_features.pt"
)


if (
    os.path.exists(train_cache)
    and os.path.exists(val_cache)
    and os.path.exists(test_cache)
):

    print(
        "\nLoading previously cached features..."
    )

    train_features, train_labels = torch.load(
        train_cache,
        weights_only=False
    )

    val_features, val_labels = torch.load(
        val_cache,
        weights_only=False
    )

    test_features, test_labels = torch.load(
        test_cache,
        weights_only=False
    )

else:

    train_features, train_labels = extract_features(
        train_loader,
        "training"
    )

    val_features, val_labels = extract_features(
        val_loader,
        "validation"
    )

    test_features, test_labels = extract_features(
        test_loader,
        "test"
    )

    torch.save(
        (
            train_features,
            train_labels
        ),
        train_cache
    )

    torch.save(
        (
            val_features,
            val_labels
        ),
        val_cache
    )

    torch.save(
        (
            test_features,
            test_labels
        ),
        test_cache
    )

    print(
        "\nCached frozen-backbone features."
    )


# ============================================================
# 12. HEAD-ONLY DATA LOADERS
# ============================================================

head_train_dataset = TensorDataset(
    train_features,
    train_labels
)

head_val_dataset = TensorDataset(
    val_features,
    val_labels
)

head_test_dataset = TensorDataset(
    test_features,
    test_labels
)


HEAD_BATCH_SIZE = 512

head_train_loader = DataLoader(
    head_train_dataset,
    batch_size=HEAD_BATCH_SIZE,
    shuffle=True
)

head_val_loader = DataLoader(
    head_val_dataset,
    batch_size=HEAD_BATCH_SIZE,
    shuffle=False
)

head_test_loader = DataLoader(
    head_test_dataset,
    batch_size=HEAD_BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 13. CLASSIFIER HEAD
# ============================================================

classifier_head = nn.Linear(
    512,
    10
).to(device)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    classifier_head.parameters(),
    lr=0.001
)


EPOCHS = 12


# ============================================================
# 14. EVALUATION FUNCTION
# ============================================================

def evaluate_head(
    loader
):

    classifier_head.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for (
            features_batch,
            labels_batch
        ) in loader:

            features_batch = (
                features_batch.to(device)
            )

            labels_batch = (
                labels_batch.to(device)
            )

            logits = classifier_head(
                features_batch
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels_batch.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return (
        accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# 15. TRAIN CLASSIFIER HEAD
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASSIFIER HEAD")
print("=" * 70)

best_val_accuracy = 0.0

best_head_state = None

training_history = []


for epoch in range(EPOCHS):

    classifier_head.train()

    running_loss = 0.0

    for (
        features_batch,
        labels_batch
    ) in head_train_loader:

        features_batch = (
            features_batch.to(device)
        )

        labels_batch = (
            labels_batch.to(device)
        )

        optimizer.zero_grad()

        logits = classifier_head(
            features_batch
        )

        loss = criterion(
            logits,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    train_accuracy, _, _ = evaluate_head(
        head_train_loader
    )

    val_accuracy, _, _ = evaluate_head(
        head_val_loader
    )

    average_loss = (
        running_loss
        / len(head_train_loader)
    )


    training_history.append({
        "epoch": epoch + 1,
        "loss": average_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy
    })


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_head_state = copy.deepcopy(
            classifier_head.state_dict()
        )


classifier_head.load_state_dict(
    best_head_state
)


print(
    "\nBest feature-extraction validation accuracy:",
    round(best_val_accuracy, 4)
)


# ============================================================
# 16. FINAL TEST EVALUATION
# ============================================================

test_accuracy, true_labels, predicted_labels = (
    evaluate_head(
        head_test_loader
    )
)


print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    "Test accuracy:",
    round(test_accuracy, 4)
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
)


print(
    "\nClassification Report:"
)

print(
    report_text
)


report_dict = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report_dict
).transpose()


# ============================================================
# 19. FIND MOST COMMON CONFUSION PAIRS
# ============================================================

confusion_pairs = []


for true_class in range(10):

    for predicted_class in range(10):

        if true_class != predicted_class:

            confusion_pairs.append({
                "true_class": class_names[
                    true_class
                ],
                "predicted_class": class_names[
                    predicted_class
                ],
                "count": int(
                    cm[
                        true_class,
                        predicted_class
                    ]
                )
            })


confusion_pairs_df = pd.DataFrame(
    confusion_pairs
).sort_values(
    "count",
    ascending=False
)


print(
    "\nTop confusion pairs:"
)

print(
    confusion_pairs_df
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 20. SAVE TRAINING HISTORY + REPORTS
# ============================================================

history_df = pd.DataFrame(
    training_history
)


history_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "training_history.csv"
    ),
    index=False
)


report_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "classification_report.csv"
    )
)


np.savetxt(
    os.path.join(
        REPORT_DIR,
        "confusion_matrix.csv"
    ),
    cm,
    delimiter=",",
    fmt="%d"
)


confusion_pairs_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "confusion_pairs.csv"
    ),
    index=False
)


# ============================================================
# 21. BUILD FINAL MODEL STRUCTURE
# ============================================================

final_model = models.resnet18(
    weights=None
)

final_model.fc = nn.Linear(
    final_model.fc.in_features,
    10
)


# Copy pretrained backbone weights.

pretrained_state = backbone_model.state_dict()

final_state = final_model.state_dict()


for key in pretrained_state:

    if not key.startswith("fc."):

        final_state[key] = (
            pretrained_state[key]
        )


# Copy trained classifier head.

final_state["fc.weight"] = (
    classifier_head.weight
    .detach()
    .cpu()
)

final_state["fc.bias"] = (
    classifier_head.bias
    .detach()
    .cpu()
)


final_model.load_state_dict(
    final_state
)


# ============================================================
# 22. SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "product_classifier.pt"
)


torch.save(
    final_model.state_dict(),
    MODEL_PATH
)


print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)


# ============================================================
# 23. SAVE MODEL METADATA
# ============================================================

metadata = {
    "architecture": "resnet18",
    "input_size": 224,
    "num_classes": 10,
    "class_names": class_names,
    "batch_size": BATCH_SIZE,
    "head_batch_size": HEAD_BATCH_SIZE,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "epochs": EPOCHS,
    "train_size": len(train_dataset),
    "validation_size": len(val_dataset),
    "test_size": len(test_dataset),
    "feature_extraction_validation_accuracy":
        float(best_val_accuracy),
    "test_accuracy":
        float(test_accuracy),
    "fine_tuning_required":
        bool(best_val_accuracy < 0.80)
}


with open(
    os.path.join(
        MODEL_DIR,
        "product_classifier_metadata.json"
    ),
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 24. EXPORT REAL TEST PNG IMAGES
# ============================================================

print("\n" + "=" * 70)
print("EXPORTING SAMPLE TEST IMAGES")
print("=" * 70)


# Load raw Fashion-MNIST without transform,
# so original grayscale images can be exported as real PNG files.

raw_test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=False
)


# Pick one example from several different classes.

wanted_classes = [
    0,  # T-shirt/top
    1,  # Trouser
    5,  # Sandal
    7,  # Sneaker
    8,  # Bag
    9   # Ankle boot
]


saved_classes = set()


for index in range(
    len(raw_test_dataset)
):

    image, label = raw_test_dataset[
        index
    ]

    if (
        label in wanted_classes
        and label not in saved_classes
    ):

        safe_label = (
            class_names[label]
            .lower()
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )

        filename = (
            f"{index:05d}_{safe_label}.png"
        )

        image.save(
            os.path.join(
                SAMPLE_DIR,
                filename
            )
        )

        print(
            "Saved:",
            filename
        )

        saved_classes.add(
            label
        )

        if len(saved_classes) == len(
            wanted_classes
        ):
            break


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PART 2 TRAINING COMPLETE")
print("=" * 70)

print(
    "Train size:",
    len(train_dataset)
)

print(
    "Validation size:",
    len(val_dataset)
)

print(
    "Test size:",
    len(test_dataset)
)

print(
    "Feature-extraction validation accuracy:",
    round(best_val_accuracy, 4)
)

print(
    "Test accuracy:",
    round(test_accuracy, 4)
)

print(
    "Fine-tuning required:",
    best_val_accuracy < 0.80
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Sample image directory:",
    SAMPLE_DIR
)
