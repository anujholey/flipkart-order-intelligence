
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"

DATA_PATH = (
    PROJECT_DIR
    + "/part1_return_risk/orders_dataset.csv"
)

MODEL_PATH = (
    PROJECT_DIR
    + "/models/return_risk_model.pkl"
)

THRESHOLD_PATH = (
    PROJECT_DIR
    + "/models/return_risk_threshold.txt"
)


# Load dataset
df = pd.read_csv(DATA_PATH)


# Features
features = [
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

target = "returned"


X = df[features]
y = df[target]


# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# Load model
model = joblib.load(MODEL_PATH)


# Load threshold
with open(
    THRESHOLD_PATH,
    "r"
) as f:
    threshold = float(f.read())


# Predict
probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


# Metrics
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)


print("=" * 60)
print("FINAL RETURN RISK MODEL EVALUATION")
print("=" * 60)

print(
    f"Threshold: {threshold:.2f}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1: {f1:.4f}"
)

print(
    f"ROC-AUC: {auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nModel loaded from:")
print(MODEL_PATH)
