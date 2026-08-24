
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.inspection import permutation_importance


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"

DATA_PATH = (
    PROJECT_DIR
    + "/part1_return_risk/orders_dataset.csv"
)

MODEL_DIR = PROJECT_DIR + "/models"

REPORT_DIR = PROJECT_DIR + "/reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. BASIC DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA CHECK")
print("=" * 70)

print("\nMissing values:")
print(df.isnull().sum())

print("\nOverall return rate:")
print(round(df["returned"].mean() * 100, 2), "%")


# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

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


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "Training return rate:",
    round(y_train.mean() * 100, 2),
    "%"
)

print(
    "Testing return rate:",
    round(y_test.mean() * 100, 2),
    "%"
)


# ============================================================
# 6. FEATURE TYPES
# ============================================================

numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given"
]

categorical_features = [
    "product_category",
    "payment_method"
]


# ============================================================
# 7. NUMERIC PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 8. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# ============================================================
# 9. COMPLETE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 10. DUMMY CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("DUMMY CLASSIFIER")
print("=" * 70)

dummy_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ]
)

dummy_model.fit(
    X_train,
    y_train
)

dummy_pred = dummy_model.predict(X_test)

dummy_accuracy = accuracy_score(
    y_test,
    dummy_pred
)

dummy_f1 = f1_score(
    y_test,
    dummy_pred,
    zero_division=0
)

print("Dummy Accuracy:", round(dummy_accuracy, 4))
print("Dummy F1:", round(dummy_f1, 4))


# ============================================================
# 11. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_pred = logistic_model.predict(X_test)

logistic_prob = (
    logistic_model
    .predict_proba(X_test)[:, 1]
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_prob
)

print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1       :", round(logistic_f1, 4))
print("ROC-AUC  :", round(logistic_auc, 4))


# ============================================================
# 12. LOGISTIC REGRESSION THRESHOLD SWEEP
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION THRESHOLD SWEEP")
print("=" * 70)

threshold_results = []

for threshold in np.arange(
    0.10,
    0.901,
    0.01
):

    predictions = (
        logistic_prob >= threshold
    ).astype(int)

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

    threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


logistic_threshold_df = pd.DataFrame(
    threshold_results
)

best_logistic_row = (
    logistic_threshold_df
    .loc[
        logistic_threshold_df["f1"].idxmax()
    ]
)

print(
    "Best Logistic threshold:",
    best_logistic_row["threshold"]
)

print(
    "Best Logistic F1:",
    round(best_logistic_row["f1"], 4)
)


# ============================================================
# 13. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST GRID SEARCH")
print("=" * 70)

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],
    "classifier__max_depth": [
        6,
        10,
        None
    ]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "\nBest CV ROC-AUC:",
    round(grid_search.best_score_, 4)
)


# ============================================================
# 14. BEST RANDOM FOREST
# ============================================================

best_rf_model = grid_search.best_estimator_

rf_prob = (
    best_rf_model
    .predict_proba(X_test)[:, 1]
)

rf_pred_default = (
    rf_prob >= 0.50
).astype(int)


rf_test_auc = roc_auc_score(
    y_test,
    rf_prob
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred_default
)

rf_precision = precision_score(
    y_test,
    rf_pred_default,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred_default,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred_default,
    zero_division=0
)


print("\n" + "=" * 70)
print("RANDOM FOREST TEST RESULTS")
print("=" * 70)

print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1       :", round(rf_f1, 4))
print("ROC-AUC  :", round(rf_test_auc, 4))

print(
    "Best CV ROC-AUC:",
    round(grid_search.best_score_, 4)
)


# ============================================================
# 15. RANDOM FOREST THRESHOLD SWEEP
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST THRESHOLD SWEEP")
print("=" * 70)

rf_threshold_results = []

for threshold in np.arange(
    0.10,
    0.901,
    0.01
):

    predictions = (
        rf_prob >= threshold
    ).astype(int)

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

    rf_threshold_results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


rf_threshold_df = pd.DataFrame(
    rf_threshold_results
)


best_rf_row = (
    rf_threshold_df
    .loc[
        rf_threshold_df["f1"].idxmax()
    ]
)

t_rf = float(
    best_rf_row["threshold"]
)

print(
    "t*_rf:",
    round(t_rf, 2)
)

print(
    "F1:",
    round(best_rf_row["f1"], 4)
)

print(
    "Recall:",
    round(best_rf_row["recall"], 4)
)

print(
    "Precision:",
    round(best_rf_row["precision"], 4)
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

rf_classifier = (
    best_rf_model
    .named_steps["classifier"]
)

feature_names = (
    best_rf_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_values = (
    rf_classifier.feature_importances_
)

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

feature_importance_df = (
    feature_importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)

print("\nTop 10 features:")

print(
    feature_importance_df.head(10)
    .to_string(index=False)
)


# ============================================================
# 17. PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE")
print("=" * 70)

permutation_result = permutation_importance(
    best_rf_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": (
        permutation_result.importances_mean
    )
})

permutation_df = (
    permutation_df
    .sort_values(
        "importance",
        ascending=False
    )
)

print(
    permutation_df.to_string(index=False)
)


# ============================================================
# 18. SUBGROUP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SUBGROUP ANALYSIS")
print("=" * 70)

test_results = X_test.copy()

test_results["actual"] = y_test.values

test_results["predicted"] = (
    rf_prob >= t_rf
).astype(int)

test_results["probability"] = rf_prob


# ---------------- PRODUCT CATEGORY ----------------

category_metrics = []

for category in sorted(
    test_results["product_category"].unique()
):

    subset = test_results[
        test_results["product_category"] == category
    ]

    category_metrics.append({
        "product_category": category,
        "samples": len(subset),
        "recall": recall_score(
            subset["actual"],
            subset["predicted"],
            zero_division=0
        ),
        "precision": precision_score(
            subset["actual"],
            subset["predicted"],
            zero_division=0
        )
    })


category_metrics_df = pd.DataFrame(
    category_metrics
)

print("\nProduct category metrics:")

print(
    category_metrics_df.to_string(index=False)
)


# ---------------- PAYMENT METHOD ----------------

payment_metrics = []

for payment in sorted(
    test_results["payment_method"].unique()
):

    subset = test_results[
        test_results["payment_method"] == payment
    ]

    payment_metrics.append({
        "payment_method": payment,
        "samples": len(subset),
        "recall": recall_score(
            subset["actual"],
            subset["predicted"],
            zero_division=0
        ),
        "precision": precision_score(
            subset["actual"],
            subset["predicted"],
            zero_division=0
        )
    })


payment_metrics_df = pd.DataFrame(
    payment_metrics
)

print("\nPayment method metrics:")

print(
    payment_metrics_df.to_string(index=False)
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    test_results["predicted"]
)

print(cm)


# ============================================================
# 20. SAVE REPORTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING REPORTS")
print("=" * 70)

feature_importance_df.to_csv(
    REPORT_DIR
    + "/feature_importance.csv",
    index=False
)

permutation_df.to_csv(
    REPORT_DIR
    + "/permutation_importance.csv",
    index=False
)

category_metrics_df.to_csv(
    REPORT_DIR
    + "/category_metrics.csv",
    index=False
)

payment_metrics_df.to_csv(
    REPORT_DIR
    + "/payment_metrics.csv",
    index=False
)

rf_threshold_df.to_csv(
    REPORT_DIR
    + "/rf_threshold_results.csv",
    index=False
)

logistic_threshold_df.to_csv(
    REPORT_DIR
    + "/logistic_threshold_results.csv",
    index=False
)


# ============================================================
# 21. SAVE FINAL MODEL
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    + "/return_risk_model.pkl"
)

joblib.dump(
    best_rf_model,
    MODEL_PATH
)


# Save threshold separately
THRESHOLD_PATH = (
    MODEL_DIR
    + "/return_risk_threshold.txt"
)

with open(
    THRESHOLD_PATH,
    "w"
) as f:
    f.write(str(t_rf))


# ============================================================
# 22. VERIFY SAVED MODEL
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED MODEL")
print("=" * 70)

loaded_model = joblib.load(
    MODEL_PATH
)

loaded_probability = (
    loaded_model
    .predict_proba(X_test)[:, 1]
)

print(
    "Model loaded successfully."
)

print(
    "Example probabilities:"
)

print(
    loaded_probability[:5]
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PART 1 COMPLETE")
print("=" * 70)

print(
    "Dataset rows:",
    len(df)
)

print(
    "Dataset columns:",
    len(df.columns)
)

print(
    "Logistic ROC-AUC:",
    round(logistic_auc, 4)
)

print(
    "Random Forest CV ROC-AUC:",
    round(grid_search.best_score_, 4)
)

print(
    "Random Forest Test ROC-AUC:",
    round(rf_test_auc, 4)
)

print(
    "Random Forest t*_rf:",
    round(t_rf, 2)
)

print(
    "\nFinal model:"
)

print(MODEL_PATH)

print(
    "\nFinal threshold:"
)

print(THRESHOLD_PATH)

print("\nAll Part 1 files saved successfully.")
