
import os
import json
import joblib
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"


# ============================================================
# PART 1 PATHS
# ============================================================

RETURN_MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "return_risk_model.pkl"
)

RETURN_THRESHOLD_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "return_risk_threshold.txt"
)


# ============================================================
# PART 2 PATHS
# ============================================================

PRODUCT_MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "product_classifier.pt"
)

PRODUCT_METADATA_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "product_classifier_metadata.json"
)


# ============================================================
# RETURN-RISK TOOL
# ============================================================

def check_return_risk(
    order_features: dict
) -> dict:

    import pandas as pd

    model = joblib.load(
        RETURN_MODEL_PATH
    )

    with open(
        RETURN_THRESHOLD_PATH,
        "r"
    ) as f:
        t_rf = float(
            f.read().strip()
        )

    required_features = [
        "product_category",
        "price_inr",
        "discount_pct",
        "payment_method",
        "customer_tenure_days",
        "num_previous_orders",
        "num_previous_returns",
        "delivery_distance_km",
        "delivery_days",
        "is_weekend_order",
        "rating_given"
    ]

    missing = [
        feature
        for feature in required_features
        if feature not in order_features
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    input_df = pd.DataFrame(
        [order_features]
    )

    probability = float(
        model.predict_proba(
            input_df
        )[0, 1]
    )

    medium_cutoff = t_rf
    high_cutoff = min(
        t_rf + 0.15,
        1.0
    )

    if probability < medium_cutoff:
        risk_bucket = "Low"

    elif probability >= high_cutoff:
        risk_bucket = "High"

    else:
        risk_bucket = "Medium"

    return {
        "return_probability":
            round(probability, 4),

        "risk_bucket":
            risk_bucket,

        "t_rf":
            round(t_rf, 4),

        "low_medium_cutoff":
            round(medium_cutoff, 4),

        "medium_high_cutoff":
            round(high_cutoff, 4)
    }


# ============================================================
# IMAGE CLASSIFIER SETUP
# ============================================================

with open(
    PRODUCT_METADATA_PATH,
    "r"
) as f:
    product_metadata = json.load(f)


CLASS_NAMES = product_metadata[
    "class_names"
]


product_transform = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


def load_product_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    state_dict = torch.load(
        PRODUCT_MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# IMAGE-CLASSIFICATION TOOL
# ============================================================

def classify_product_image(
    image_path: str
) -> dict:

    if not os.path.exists(
        image_path
    ):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_product_model()

    image = Image.open(
        image_path
    ).convert("L")

    tensor = product_transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )

    predicted_index = int(
        predicted_index.item()
    )

    confidence = float(
        confidence.item()
    )

    return {
        "predicted_category":
            CLASS_NAMES[
                predicted_index
            ],

        "confidence":
            round(
                confidence,
                4
            )
    }
