
import os
import json
import joblib
import pandas as pd

PROJECT_DIR = "/content/drive/MyDrive/flipkart-order-intelligence"

results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(
        f"[{status}] {name}"
        + (f" -> {detail}" if detail else "")
    )

print("=" * 75)
print("FLIPKART ORDER INTELLIGENCE - FINAL ACCEPTANCE CHECK")
print("=" * 75)

# ============================================================
# PART 1
# ============================================================

print("\n--- PART 1 ---")

orders_path = os.path.join(
    PROJECT_DIR,
    "part1_return_risk",
    "orders_dataset.csv"
)

check(
    "orders_dataset.csv exists",
    os.path.exists(orders_path)
)

if os.path.exists(orders_path):

    df = pd.read_csv(orders_path)

    check(
        "Dataset has 6000 rows",
        len(df) == 6000,
        f"rows={len(df)}"
    )

    check(
        "Dataset has 13 columns",
        len(df.columns) == 13,
        f"columns={len(df.columns)}"
    )

    return_rate = df["returned"].mean()

    check(
        "Overall return rate is 18%-27%",
        0.18 <= return_rate <= 0.27,
        f"{return_rate:.4f}"
    )

    missing_rating = (
        df["rating_given"]
        .isna()
        .mean()
    )

    check(
        "rating_given missingness is 8%-18%",
        0.08 <= missing_rating <= 0.18,
        f"{missing_rating:.4f}"
    )

    cod_missing = (
        df.loc[
            df["payment_method"] == "COD",
            "rating_given"
        ]
        .isna()
        .mean()
    )

    non_cod_missing = (
        df.loc[
            df["payment_method"] != "COD",
            "rating_given"
        ]
        .isna()
        .mean()
    )

    check(
        "COD missing rate exceeds non-COD",
        cod_missing > non_cod_missing,
        (
            f"COD={cod_missing:.4f}, "
            f"non-COD={non_cod_missing:.4f}, "
            f"gap={cod_missing - non_cod_missing:.4f}"
        )
    )

return_model_path = os.path.join(
    PROJECT_DIR,
    "models",
    "return_risk_model.pkl"
)

check(
    "Return-risk model exists",
    os.path.exists(return_model_path)
)

if os.path.exists(return_model_path):

    try:

        model = joblib.load(
            return_model_path
        )

        check(
            "Return-risk model loads",
            True,
            type(model).__name__
        )

        classifier = model.named_steps[
            "classifier"
        ]

        check(
            "Final return model is Random Forest",
            classifier.__class__.__name__
            == "RandomForestClassifier",
            classifier.__class__.__name__
        )

    except Exception as e:

        check(
            "Return-risk model loads",
            False,
            str(e)
        )

threshold_path = os.path.join(
    PROJECT_DIR,
    "models",
    "return_risk_threshold.txt"
)

check(
    "Random Forest threshold exists",
    os.path.exists(threshold_path)
)

if os.path.exists(threshold_path):

    with open(
        threshold_path,
        "r"
    ) as f:

        t_rf = float(
            f.read().strip()
        )

    check(
        "t*_rf is valid probability threshold",
        0.0 < t_rf < 1.0,
        f"t*_rf={t_rf:.4f}"
    )

# ============================================================
# PART 2
# ============================================================

print("\n--- PART 2 ---")

product_model_path = os.path.join(
    PROJECT_DIR,
    "models",
    "product_classifier.pt"
)

check(
    "Product classifier exists",
    os.path.exists(product_model_path)
)

if os.path.exists(product_model_path):

    size_mb = (
        os.path.getsize(product_model_path)
        / (1024 * 1024)
    )

    check(
        "Product classifier below GitHub 100MB limit",
        size_mb < 100,
        f"{size_mb:.2f} MB"
    )

metadata_path = os.path.join(
    PROJECT_DIR,
    "models",
    "product_classifier_metadata.json"
)

check(
    "Product classifier metadata exists",
    os.path.exists(metadata_path)
)

if os.path.exists(metadata_path):

    with open(
        metadata_path,
        "r"
    ) as f:

        metadata = json.load(f)

    check(
        "Fashion-MNIST train subset = 55000",
        metadata["train_size"] == 55000,
        str(metadata["train_size"])
    )

    check(
        "Validation subset = 5000",
        metadata["validation_size"] == 5000,
        str(metadata["validation_size"])
    )

    check(
        "Test split = 10000",
        metadata["test_size"] == 10000,
        str(metadata["test_size"])
    )

    test_accuracy = float(
        metadata["test_accuracy"]
    )

    check(
        "Part 2 test accuracy >= 80%",
        test_accuracy >= 0.80,
        f"{test_accuracy:.4f}"
    )

sample_dir = os.path.join(
    PROJECT_DIR,
    "data",
    "sample_images"
)

png_files = []

if os.path.exists(sample_dir):

    png_files = [
        f
        for f in os.listdir(sample_dir)
        if f.lower().endswith(".png")
    ]

check(
    "At least 5 real PNG sample images",
    len(png_files) >= 5,
    f"count={len(png_files)}"
)

confusion_path = os.path.join(
    PROJECT_DIR,
    "part2_image_classifier",
    "reports",
    "confusion_matrix.csv"
)

check(
    "Confusion matrix exists",
    os.path.exists(confusion_path)
)

if os.path.exists(confusion_path):

    cm = pd.read_csv(
        confusion_path,
        header=None
    )

    check(
        "Confusion matrix is 10x10",
        cm.shape == (10, 10),
        str(cm.shape)
    )

# ============================================================
# PART 3
# ============================================================

print("\n--- PART 3 ---")

policy_path = os.path.join(
    PROJECT_DIR,
    "part3_support_agent",
    "knowledge_base",
    "policies.json"
)

check(
    "Policy KB exists",
    os.path.exists(policy_path)
)

if os.path.exists(policy_path):

    with open(
        policy_path,
        "r"
    ) as f:

        policies = json.load(f)

    check(
        "At least 12 policy documents",
        len(policies) >= 12,
        f"count={len(policies)}"
    )

answer_key_path = os.path.join(
    PROJECT_DIR,
    "part3_support_agent",
    "knowledge_base",
    "retrieval_answer_key.json"
)

check(
    "Retrieval answer key exists",
    os.path.exists(answer_key_path)
)

if os.path.exists(answer_key_path):

    with open(
        answer_key_path,
        "r"
    ) as f:

        answer_key = json.load(f)

    check(
        "At least 5 retrieval evaluation queries",
        len(answer_key) >= 5,
        f"count={len(answer_key)}"
    )

faiss_path = os.path.join(
    PROJECT_DIR,
    "part3_support_agent",
    "vector_index",
    "policy_index.faiss"
)

check(
    "FAISS index exists",
    os.path.exists(faiss_path)
)

retrieval_report = os.path.join(
    PROJECT_DIR,
    "part3_support_agent",
    "reports",
    "retrieval_evaluation.csv"
)

check(
    "Retrieval evaluation exists",
    os.path.exists(retrieval_report)
)

transcript_dir = os.path.join(
    PROJECT_DIR,
    "transcripts"
)

transcript_files = []

if os.path.exists(transcript_dir):

    transcript_files = [
        f
        for f in os.listdir(transcript_dir)
        if f.endswith(".txt")
    ]

check(
    "At least 8 transcripts",
    len(transcript_files) >= 8,
    f"count={len(transcript_files)}"
)

required_transcript_terms = [
    "multiturn",
    "fresh_conversation",
    "prompt_injection",
    "ungrounded",
    "return_risk",
    "image_classifier"
]

for term in required_transcript_terms:

    check(
        f"Transcript coverage: {term}",
        any(
            term in name
            for name in transcript_files
        )
    )

# ============================================================
# REQUIRED SOURCE FILES
# ============================================================

print("\n--- REQUIRED FILES ---")

required_files = [
    "part1_return_risk/generate_orders.py",
    "part1_return_risk/train_return_risk.py",
    "part1_return_risk/evaluate_return_risk.py",
    "part2_image_classifier/train_product_classifier.py",
    "part2_image_classifier/predict_product.py",
    "part3_support_agent/build_index.py",
    "part3_support_agent/rag.py",
    "part3_support_agent/tools.py",
    "part3_support_agent/support_agent.py",
    "part3_support_agent/generate_transcripts.py",
    "part3_support_agent/evaluate_retrieval.py",
    "models/return_risk_model.pkl",
    "models/product_classifier.pt"
]

for relative_path in required_files:

    check(
        relative_path,
        os.path.exists(
            os.path.join(
                PROJECT_DIR,
                relative_path
            )
        )
    )

# ============================================================
# SUMMARY
# ============================================================

passes = sum(
    1
    for _, status, _
    in results
    if status == "PASS"
)

fails = sum(
    1
    for _, status, _
    in results
    if status == "FAIL"
)

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

print("PASS:", passes)
print("FAIL:", fails)

if fails == 0:
    print(
        "\nALL AUTOMATED ACCEPTANCE CHECKS PASSED."
    )
else:
    print(
        "\nSome checks failed. Fix them before submission."
    )
