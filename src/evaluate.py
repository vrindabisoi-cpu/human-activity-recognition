"""Evaluate all three tuned models on the held-out TEST set.

This is the first and only place the test set is used. Nothing here feeds back
into model choices; those were all made with cross-validation in train.py.

Outputs (in reports/):
    test_accuracy.csv                  accuracy per model
    classification_report_<model>.csv  per-activity precision / recall / F1
    confusion_matrix_<model>.png       confusion matrix image
    top_confusions.csv                 most frequent mistakes per model
    top10_features.csv                 most important features of the best model

Run from the project root:
    python -m src.evaluate
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, not a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data_loader import PROJECT_ROOT, load_activity_labels, load_split

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest"]

# Fixed, meaningful label order (moving activities first, then still postures)
# so every confusion matrix has the same rows/columns.
LABELS = list(load_activity_labels().values())


def save_confusion_matrix(cm: np.ndarray, name: str, accuracy: float) -> None:
    """Rows = true activity, columns = predicted activity."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    ax.set_title(f"{name.replace('_', ' ').title()}  (test accuracy {accuracy:.1%})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)


def top_confusions(cm: np.ndarray, name: str, k: int = 5) -> pd.DataFrame:
    """Largest off-diagonal cells = the most common mistakes."""
    rows = []
    for i, true_label in enumerate(LABELS):
        for j, pred_label in enumerate(LABELS):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "model": name,
                    "true": true_label,
                    "predicted": pred_label,
                    "count": int(cm[i, j]),
                    # share of that true class that got this wrong label
                    "pct_of_true_class": round(cm[i, j] / cm[i].sum() * 100, 1),
                })
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(k)


def feature_importance(model, feature_names) -> pd.DataFrame:
    """Return all features ranked by importance for the given model.

    - Tree models: feature_importances_ (how much each feature reduced impurity
      across all splits).
    - Logistic regression: coef_ has one weight per (class, feature). Because the
      inputs were standardized, weights are comparable across features. We rank
      by the mean absolute weight over the 6 classes and also record which
      activity each feature pushes toward most strongly.
    """
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(clf, "feature_importances_"):
        return (pd.DataFrame({"feature": feature_names,
                              "importance": clf.feature_importances_})
                .sort_values("importance", ascending=False))

    coef = clf.coef_  # shape (n_classes, n_features)
    strongest_class = np.array(clf.classes_)[np.abs(coef).argmax(axis=0)]
    sign = np.sign(coef[np.abs(coef).argmax(axis=0), np.arange(coef.shape[1])])
    return (pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(coef).mean(axis=0),
        "strongest_for_activity": strongest_class,
        "direction": np.where(sign > 0, "higher -> more likely", "higher -> less likely"),
    }).sort_values("importance", ascending=False))


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    X_test, y_test, _ = load_split("test")
    best_name = json.loads((MODELS_DIR / "best_model_meta.json").read_text())["best_model"]

    accuracy_rows, confusion_tables = [], []

    for name in MODEL_NAMES:
        model = joblib.load(MODELS_DIR / f"{name}.joblib")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracy_rows.append({"model": name, "test_accuracy": round(acc, 4)})

        # Precision: of everything predicted as X, how much really was X?
        # Recall:    of everything that really was X, how much did we catch?
        report = classification_report(y_test, y_pred, labels=LABELS,
                                       output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.round(3)
        report_df.to_csv(REPORTS_DIR / f"classification_report_{name}.csv")

        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        save_confusion_matrix(cm, name, acc)
        confusion_tables.append(top_confusions(cm, name))

        print(f"\n=== {name}  |  test accuracy {acc:.4f} ===")
        print(report_df.loc[LABELS, ["precision", "recall", "f1-score", "support"]]
              .to_string())

    acc_df = pd.DataFrame(accuracy_rows)
    acc_df.to_csv(REPORTS_DIR / "test_accuracy.csv", index=False)
    conf_df = pd.concat(confusion_tables)
    conf_df.to_csv(REPORTS_DIR / "top_confusions.csv", index=False)

    print("\n=== Test accuracy ===")
    print(acc_df.to_string(index=False))
    print("\n=== Most common mistakes (true -> predicted) ===")
    print(conf_df.to_string(index=False))

    best_model = joblib.load(MODELS_DIR / "best_model.joblib")
    top10 = feature_importance(best_model, X_test.columns).head(10)
    top10.to_csv(REPORTS_DIR / "top10_features.csv", index=False)
    print(f"\n=== Top 10 features ({best_name}) ===")
    print(top10.to_string(index=False, float_format="%.4f"))

    print(f"\nConfusion matrix images saved in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
