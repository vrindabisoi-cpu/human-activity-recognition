"""Choose the app's "uncertain" threshold using TRAINING data only.

Idea: with GroupKFold, every training sample gets a prediction from a model
that never saw that person ("out-of-fold" predictions). We then check how often
the model is right at each confidence level, and pick a threshold below which
the app should warn the user instead of giving a confident answer.

Run from the project root (after train.py):
    python -m src.confidence
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.data_loader import PROJECT_ROOT, load_split

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Chosen from the table this script prints: below 0.80 the model is right only
# ~60% of the time, and flagging that band (~8% of samples) catches ~half of all
# errors while leaving the rest of the predictions at ~96% accuracy.
THRESHOLD = 0.80


def main() -> None:
    X, y, subjects = load_split("train")
    # clone() = same model settings, but untrained, so CV re-fits it per fold.
    model = clone(joblib.load(MODELS_DIR / "best_model.joblib"))
    proba = cross_val_predict(model, X, y, groups=subjects, cv=GroupKFold(5),
                              method="predict_proba", n_jobs=-1)

    classes = np.sort(y.unique())          # predict_proba columns are in sorted order
    confidence = proba.max(axis=1)         # probability of the predicted class
    correct = classes[proba.argmax(axis=1)] == y.to_numpy()

    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    table = (pd.DataFrame({"confidence": pd.cut(confidence, bins, right=False),
                           "correct": correct})
             .groupby("confidence", observed=False)["correct"]
             .agg(n_samples="size", accuracy="mean").round(3))
    print("Out-of-fold accuracy by confidence band (training data):")
    print(table.to_string())
    table.to_csv(REPORTS_DIR / "confidence_calibration.csv")

    low = confidence < THRESHOLD
    summary = {
        "threshold": THRESHOLD,
        "share_flagged": round(float(low.mean()), 3),
        "accuracy_flagged": round(float(correct[low].mean()), 3),
        "accuracy_not_flagged": round(float(correct[~low].mean()), 3),
        "share_of_errors_flagged": round(float((low & ~correct).sum() / (~correct).sum()), 3),
    }
    print("\nWith threshold", THRESHOLD, "->", summary)

    # Store the threshold next to the model so the app reads it from one place.
    meta_path = MODELS_DIR / "best_model_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["uncertainty"] = summary
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved threshold to {meta_path.name}")


if __name__ == "__main__":
    main()
