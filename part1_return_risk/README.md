
# Part 1 - Return Risk Prediction

This module generates the deterministic order dataset and trains a return-risk classification model.

## Files

- `generate_orders.py` - generates the 6000-row order dataset.
- `orders_dataset.csv` - generated dataset.
- `train_return_risk.py` - trains DummyClassifier, Logistic Regression and tuned Random Forest models.
- `evaluate_return_risk.py` - evaluates the saved Random Forest model.

## Final Model

The final model is a tuned Random Forest classifier.

The model is saved as:

`models/return_risk_model.pkl`

The selected Random Forest F1 threshold is saved as:

`models/return_risk_threshold.txt`
