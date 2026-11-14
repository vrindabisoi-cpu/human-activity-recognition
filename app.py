"""One-screen Streamlit app: pick a test sample, see the model's prediction.

Run locally from the project root:
    streamlit run app.py

How Streamlit works: every time the user touches a widget, this whole script
re-runs from top to bottom. Slow steps (loading the model, predicting) are
cached so they only happen once.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# 1. Paths: relative to this file, so it works locally and on the cloud server.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "best_model_meta.json"
DATA_PATH = ROOT / "app_data" / "test_set.parquet"

ACTIVITY_ORDER = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                  "SITTING", "STANDING", "LAYING"]
SHOW_OPTIONS = ["All samples", "Only model mistakes", "Only uncertain predictions"]


def pretty(label: str) -> str:
    """WALKING_UPSTAIRS -> Walking upstairs"""
    return label.replace("_", " ").capitalize()


# Page settings must be the first Streamlit command in the script.
st.set_page_config(page_title="Activity Recognition Demo", page_icon="🏃", layout="wide")

# ---------------------------------------------------------------------------
# 2. Load once, then reuse on every rerun.
# ---------------------------------------------------------------------------
@st.cache_resource  # for objects like models: loaded once, shared, not copied
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


@st.cache_data  # for data: result is stored and reused
def load_test_set_with_predictions(_model):
    """Return the test features plus the model's prediction for every row.

    The leading underscore in _model tells Streamlit not to try to hash the
    model object when deciding whether the cache is still valid.
    """
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=["activity", "subject"])
    proba = _model.predict_proba(X)           # shape (n_rows, 6)
    classes = _model.classes_                 # column order of proba
    results = pd.DataFrame(proba, columns=classes)
    results["predicted"] = classes[proba.argmax(axis=1)]
    results["confidence"] = proba.max(axis=1)
    results["true"] = df["activity"].to_numpy()
    results["subject"] = df["subject"].to_numpy()
    return results


model, meta = load_model()
THRESHOLD = meta["uncertainty"]["threshold"]
results = load_test_set_with_predictions(model)

# ---------------------------------------------------------------------------
# 3. Page header
# ---------------------------------------------------------------------------
st.title("🏃 Human Activity Recognition")
st.caption(
    "Predicts what a person was doing from 2.56 seconds of smartphone sensor data "
    "(UCI HAR test set: 2,947 samples from 9 people the model never saw in training)."
)

# ---------------------------------------------------------------------------
# 4. Controls. Session state = Streamlit's memory between reruns.
# ---------------------------------------------------------------------------
if "pos" not in st.session_state:
    st.session_state.pos = 0


def reset_position():
    """When a filter changes, jump back to the first matching sample."""
    st.session_state.pos = 0


def pick_random(n: int):
    st.session_state.pos = int(np.random.randint(n))


c1, c2, c3, c4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
activity_filter = c1.selectbox("True activity", ["All"] + ACTIVITY_ORDER,
                               format_func=lambda a: a if a == "All" else pretty(a),
                               key="activity_filter", on_change=reset_position)
show = c2.selectbox("Show", SHOW_OPTIONS, key="show", on_change=reset_position)

mask = pd.Series(True, index=results.index)
if activity_filter != "All":
    mask &= results["true"] == activity_filter
if show == "Only model mistakes":
    mask &= results["predicted"] != results["true"]
elif show == "Only uncertain predictions":
    mask &= results["confidence"] < THRESHOLD
candidates = results.index[mask].to_numpy()

if len(candidates) == 0:
    st.info("No test samples match these filters.")
    st.stop()

n = len(candidates)
c3.number_input(f"Sample # (0–{n - 1} of {n} matching)", min_value=0,
                max_value=n - 1, step=1, key="pos")
c4.button("🎲 Random", on_click=pick_random, args=(n,), width="stretch")

row_id = int(candidates[st.session_state.pos])
row = results.loc[row_id]

# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------
st.caption(f"Test row {row_id} · volunteer #{row['subject']}")

m1, m2, m3 = st.columns(3)
m1.metric("Predicted activity", pretty(row["predicted"]))
m2.metric("Model confidence", f"{row['confidence']:.0%}")
m3.metric("True label", pretty(row["true"]))

is_uncertain = row["confidence"] < THRESHOLD
probs = (row[ACTIVITY_ORDER].astype(float)
         .rename_axis("activity").reset_index(name="probability"))

if is_uncertain:
    top2 = probs.nlargest(2, "probability")
    a, b = top2.itertuples(index=False)
    st.warning(
        f"⚠️ **Uncertain prediction** (confidence below {THRESHOLD:.0%}). "
        f"Top guesses: **{pretty(a.activity)}** ({a.probability:.0%}) or "
        f"**{pretty(b.activity)}** ({b.probability:.0%}). In validation, predictions "
        f"this uncertain were right only {meta['uncertainty']['accuracy_flagged']:.0%} "
        "of the time; treat this as \"needs more data\", not a final answer."
    )

if row["predicted"] == row["true"]:
    st.success("✅ Correct: the prediction matches the true label.")
else:
    st.error(f"❌ Incorrect: predicted **{pretty(row['predicted'])}**, "
             f"but the person was actually **{pretty(row['true'])}**.")

# Probability chart: one bar per activity, predicted one highlighted,
# dashed line at the uncertainty threshold.
probs["label"] = probs["activity"].map(pretty)
probs["is_predicted"] = probs["activity"] == row["predicted"]
bars = alt.Chart(probs).mark_bar().encode(
    x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"), title="Model probability"),
    y=alt.Y("label:N", sort=[pretty(a) for a in ACTIVITY_ORDER], title=None),
    color=alt.condition("datum.is_predicted", alt.value("#1f77b4"), alt.value("#c7c7c7")),
    tooltip=["label", alt.Tooltip("probability:Q", format=".1%")],
)
rule = alt.Chart(pd.DataFrame({"x": [THRESHOLD]})).mark_rule(
    strokeDash=[4, 4], color="#d62728").encode(x="x:Q")
st.altair_chart((bars + rule).properties(height=230), width="stretch")

overall_acc = (results["predicted"] == results["true"]).mean()
st.caption(
    f"Model: logistic regression (C={meta['best_params']['clf__C']}) · "
    f"CV accuracy {meta['mean_cv_accuracy']:.1%} · test accuracy {overall_acc:.1%} · "
    f"dashed line = {THRESHOLD:.0%} uncertainty threshold"
)
