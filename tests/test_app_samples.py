"""Test the Streamlit app on 10 test samples, including 4 the model gets wrong.

This drives the real app.py with Streamlit's AppTest (a headless browser-like
runner): for each sample it sets the widgets, re-runs the app, and reads back
what a user would see. It checks that:
  * the app shows the same prediction/confidence/true label as the model,
  * wrong predictions show the red "Incorrect" banner,
  * predictions under the threshold show the "Uncertain" warning.

Run from the project root:
    python -m tests.test_app_samples
Results: reports/app_test_10_samples.csv
"""
import json

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data_loader import PROJECT_ROOT

THRESHOLD = json.loads((PROJECT_ROOT / "models" / "best_model_meta.json").read_text())[
    "uncertainty"]["threshold"]
SEED = 42


def pick_samples() -> list:
    """Choose 10 test rows: 4 wrong (2 confident, 2 uncertain), 6 correct."""
    df = pd.read_parquet(PROJECT_ROOT / "app_data" / "test_set.parquet")
    model = joblib.load(PROJECT_ROOT / "models" / "best_model.joblib")
    proba = model.predict_proba(df.drop(columns=["activity", "subject"]))
    pred = model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    wrong = pred != df["activity"].to_numpy()
    unsure = conf < THRESHOLD
    rng = np.random.default_rng(SEED)

    def take(mask, k, spread_classes=True):
        idx = np.flatnonzero(mask)
        rng.shuffle(idx)
        if not spread_classes:
            return idx[:k].tolist()
        chosen, seen = [], set()  # prefer different true activities
        for i in idx:
            if df["activity"].iat[i] not in seen:
                chosen.append(int(i))
                seen.add(df["activity"].iat[i])
            if len(chosen) == k:
                break
        return chosen

    rows = (take(wrong & ~unsure, 2) + take(wrong & unsure, 2)
            + take(~wrong & unsure, 1) + take(~wrong & ~unsure, 5))
    # Return the unrounded confidence too (the app displays it rounded).
    return [(r, float(conf[r])) for r in rows]


def run_app_on(row_id: int) -> dict:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    at.run()
    at.selectbox(key="activity_filter").select("All")
    at.selectbox(key="show").select("All samples")
    at.number_input(key="pos").set_value(row_id)  # with no filter, position == row id
    at.run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    return {
        "test_row": row_id,
        "true_label": metrics["True label"],
        "predicted": metrics["Predicted activity"],
        "confidence": metrics["Model confidence"],
        "app_shows_correct": len(at.success) == 1,
        "app_shows_incorrect": len(at.error) == 1,
        "app_shows_uncertain_warning": len(at.warning) == 1,
    }


def main() -> None:
    picks = pick_samples()
    out = pd.DataFrame([run_app_on(r) for r, _ in picks])
    raw_conf = np.array([c for _, c in picks])

    # Checks: banners must agree with the model's numbers.
    is_wrong = out["predicted"] != out["true_label"]
    assert (out["app_shows_incorrect"] == is_wrong).all(), "Incorrect banner mismatch"
    assert (out["app_shows_correct"] == ~is_wrong).all(), "Correct banner mismatch"
    assert (out["app_shows_uncertain_warning"] == (raw_conf < THRESHOLD)).all(), \
        "Uncertain warning mismatch"
    assert is_wrong.sum() >= 3, "Need at least 3 wrong predictions"

    out.to_csv(PROJECT_ROOT / "reports" / "app_test_10_samples.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nPASS: {len(out)} samples, {int(is_wrong.sum())} wrong, "
          f"{int(out['app_shows_uncertain_warning'].sum())} flagged uncertain")


if __name__ == "__main__":
    main()
