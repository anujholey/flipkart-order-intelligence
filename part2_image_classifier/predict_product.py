
import os
import json
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "product_classifier.pt"
)

METADATA_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "product_classifier_metadata.json"
)


# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


class_names = metadata["class_names"]


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

def load_product_classifier():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ------------------------------------------------------------
# Predict one image
# ------------------------------------------------------------

def classify_product_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_product_classifier()

    image = Image.open(
        image_path
    ).convert("L")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = int(
        predicted_index.item()
    )

    confidence = float(
        confidence.item()
    )

    predicted_label = class_names[
        predicted_index
    ]

    return {
        "predicted_category": predicted_label,
        "confidence": round(confidence, 4)
    }


# ------------------------------------------------------------
# Simple command-line test
# ------------------------------------------------------------

if __name__ == "__main__":

    sample_dir = os.path.join(
        PROJECT_DIR,
        "data",
        "sample_images"
    )

    files = sorted(
        [
            f
            for f in os.listdir(sample_dir)
            if f.lower().endswith(".png")
        ]
    )

    if not files:

        print(
            "No sample PNG files found."
        )

    else:

        test_image = os.path.join(
            sample_dir,
            files[0]
        )

        print(
            "Testing image:",
            test_image
        )

        result = classify_product_image(
            test_image
        )

        print(
            "Prediction result:"
        )

        print(result)
